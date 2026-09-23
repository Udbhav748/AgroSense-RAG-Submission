"""Concrete BaseTool implementations wrapping the existing service
functions.

Each tool's execute() calls the *core* logic (the _<name>_core functions
in retrieval_service / web_search_service / summarization_service) rather
than the @track_tool-wrapped public functions — the registry is the
single instrumentation point on the agent path, so calling a wrapped
function here would double-count every tool call in Tool Success Rate
metrics.

Tools run CPU/IO-bound sync work via asyncio.to_thread so execute() stays
awaitable and the registry can be driven from async agent loops without
blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import BaseModel, Field

from app.models.document import RetrievedChunk, VisionPrediction, WebSearchResult
from app.services.retrieval_service import _retrieve_core
from app.services.summarization_service import _summarize_document_core
from app.services.tools.base import ToolContext, ToolResult
from app.services.vision_client import diagnose_image
from app.services.vision_qa_service import try_vision_qa
from app.services.web_search_service import _search_web_core, web_search_ready

logger = logging.getLogger(__name__)

# ---------- Retrieval -------------------------------------------------------


class RetrievalInput(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, gt=0)
    min_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    document_ids: list[str] | None = None


class RetrievalTool:
    name = "retrieval"
    description = (
        "Vector search over the uploaded PDFs' embedded chunks; returns the "
        "top-k passages most relevant to the query."
    )
    args_schema = RetrievalInput
    output_schema: Any = list[RetrievedChunk]

    async def execute(self, args: RetrievalInput, context: ToolContext) -> ToolResult:
        if context.vector_store is None:
            return ToolResult(success=False, error="VectorStore not available in context")
        chunks = await asyncio.to_thread(
            _retrieve_core,
            args.query,
            context.vector_store,
            top_k=args.top_k,
            min_score=args.min_score,
            tenant_id=context.tenant_id,
            document_ids=args.document_ids,
        )
        return ToolResult(success=True, data=chunks)


# ---------- Web search ------------------------------------------------------


class WebSearchInput(BaseModel):
    query: str = Field(min_length=1)
    max_results: int | None = Field(default=None, gt=0)


class WebSearchTool:
    name = "web_search"
    description = (
        "Web search for information the uploaded corpus can't answer "
        "(news, prices, current events); opt-in and human-approval gated."
    )
    args_schema = WebSearchInput
    output_schema: Any = list[WebSearchResult]

    async def execute(self, args: WebSearchInput, context: ToolContext) -> ToolResult:
        # Master switch (Settings.web_search_enabled): checked here, not
        # just left to the planner's prompt ("only if web_search_enabled
        # in settings" — planning_agent.py's _PLAN_PROMPT_TEMPLATE), which
        # is an instruction the LLM can ignore, not an enforcement point.
        # web_search_ready() below deliberately does NOT check this flag
        # (see its own docstring — it's "distinct from the master switch",
        # only provider/key availability), so without this check a
        # deployment that sets AGENT_EXECUTOR_ENABLED=true and leaves
        # WEB_SEARCH_ENABLED at its documented-safe default of False would
        # still let the planner trigger real outbound search calls (the
        # default provider, DuckDuckGo, needs no key, so web_search_ready()
        # alone would happily let it through). Same degrade-to-empty
        # semantics as every other "can't search right now" branch below —
        # a disabled tool is a normal outcome, never an error.
        if not context.settings.web_search_enabled:
            return ToolResult(success=True, data=[])
        # Human-approval gate (Settings.web_search_requires_approval):
        # the tool only fires when the client explicitly confirmed the
        # search on the request (confirm_web_search=true), mirroring the
        # inline path's check in rag_service._search_web. A skipped-but-
        # requested search registers a pending approval for an operator
        # rather than vanishing into a log line.
        if context.settings.web_search_requires_approval and not context.confirm_web_search:
            from app.core.request_context import get_client_name
            from app.services.approval_service import get_approval_store

            approval = get_approval_store().register(
                action="web_search",
                requested_by=get_client_name(),
                payload={"query": args.query},
            )
            logger.info(
                "approval_created_for_skipped_web_search",
                extra={
                    "extra_fields": {
                        "approval_id": approval.approval_id,
                        "query_length": len(args.query),
                    }
                },
            )
            return ToolResult(success=True, data=[])
        if not web_search_ready():
            # Disabled (no key / unknown provider) is "no web context
            # available" — same semantics as the inline search_web() path,
            # which returns [] rather than raising.
            return ToolResult(success=True, data=[])
        try:
            results = await asyncio.to_thread(_search_web_core, args.query, args.max_results)
        except Exception as exc:
            return ToolResult(success=False, error=f"{type(exc).__name__}: {exc}")
        return ToolResult(success=True, data=results)


# ---------- Summarization ---------------------------------------------------


class SummarizationInput(BaseModel):
    document_id: str = Field(min_length=1)


class SummarizationTool:
    name = "summarization"
    description = (
        "Whole-document summary built from every chunk belonging to one uploaded document."
    )
    args_schema = SummarizationInput
    output_schema: Any = tuple[str, list[RetrievedChunk]]

    async def execute(self, args: SummarizationInput, context: ToolContext) -> ToolResult:
        if context.vector_store is None or context.llm_client is None:
            return ToolResult(
                success=False, error="VectorStore or LLMClient not available in context"
            )
        try:
            summary, chunks = await asyncio.to_thread(
                _summarize_document_core,
                args.document_id,
                context.vector_store,
                context.llm_client,
                context.tenant_id,
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"{type(exc).__name__}: {exc}")
        return ToolResult(success=True, data=(summary, chunks))


# ---------- Diagnose (vision classification) --------------------------------


class DiagnoseInput(BaseModel):
    contents: bytes
    filename: str = Field(min_length=1)
    content_type: str = Field(min_length=1)
    query: str | None = None


class DiagnoseTool:
    name = "diagnose"
    description = (
        "LeafSense HTTP vision integration: classifies a crop-leaf image "
        "into a disease class, mapped to a plain-language crop/disease/confidence."
    )
    args_schema = DiagnoseInput
    output_schema: Any = VisionPrediction

    async def execute(self, args: DiagnoseInput, context: ToolContext) -> ToolResult:
        try:
            prediction = await asyncio.to_thread(
                diagnose_image, args.contents, args.filename, args.content_type
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"{type(exc).__name__}: {exc}")
        return ToolResult(success=True, data=prediction)


# ---------- Vision QA (image-grounded answer from page rasters) -------------


class VisionQAInput(BaseModel):
    query: str = Field(min_length=1)
    document_id: str = Field(min_length=1)


class VisionQATool:
    name = "vision_qa"
    description = (
        "Answers a query by sending full-page rasters of a document to a "
        "vision-capable LLM, for pages whose text layer is too thin to "
        "retrieve well."
    )
    args_schema = VisionQAInput
    output_schema: Any = str

    async def execute(self, args: VisionQAInput, context: ToolContext) -> ToolResult:
        if context.llm_client is None:
            return ToolResult(success=False, error="LLMClient not available in context")
        chunks = await asyncio.to_thread(
            context.vector_store.get_chunks_by_document, args.document_id, context.tenant_id
        )
        if not chunks:
            return ToolResult(
                success=False, error=f"No chunks found for document {args.document_id}"
            )
        answer = await asyncio.to_thread(try_vision_qa, args.query, chunks, context.llm_client)
        if not answer:
            return ToolResult(success=False, error="No vision-grounded answer available")
        return ToolResult(success=True, data=answer)
