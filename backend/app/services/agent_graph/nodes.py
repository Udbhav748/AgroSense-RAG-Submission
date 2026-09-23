"""Graph node functions wrapping AgroSense-RAG's core multi-agent capabilities.

Nodes:
- planner_node: routes query intent (document_analysis, summarization, research, conversational).
- document_analyst_node: dense + BM25 retrieval and chunk extraction.
- fact_checker_node: verifies citations and claims against ground-truth chunks.
- web_researcher_node: external search fallback when context is weak or query is outside corpus.
- summarizer_node: full-document summarization.
- synthesizer_node: LLM generation with prompt injection delimiters and reflection context.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import app.services.rag_service as _rag_service_module
from app.core.config import settings
from app.core.exceptions import AppError
from app.core.metrics import get_metrics
from app.models.schemas import ChatResponse, SourceReference
from app.services.agent_graph.events import (
    FINALIZED,
    GENERATION_COMPLETED,
    GENERATION_STARTED,
    REQUEST_VALIDATED,
    RETRIEVAL_COMPLETED,
    RETRIEVAL_GRADED,
    RETRIEVAL_STARTED,
    VALIDATION_FAILED,
    VALIDATION_STARTED,
    VISION_COMPLETED,
    VISION_STARTED,
    WEB_SEARCH_COMPLETED,
    WEB_SEARCH_STARTED,
    NodeTimer,
    emit_node_trace,
)
from app.services.prompt_builder import (
    FALLBACK_REPLY,
    GENERATION_ERROR_REPLY,
    REFLECTION_INSTRUCTION,
    build_prompt,
    strip_sources_section,
)
from app.services.prompt_injection_service import detect_possible_injection
from app.services.rag_service import (
    _CONVERSATIONAL_INTENTS,
    _DOCUMENT_ID_RE,
    _MAX_HISTORY_TURNS,
    _SUMMARIZE_RE,
    _normalize,
    _source_references,
    _web_source_references,
)
from app.services.retrieval_service import retrieve
from app.services.summarization_service import summarize_document
from app.services.web_search_service import search_web

if TYPE_CHECKING:
    from app.services.agent_graph.state import AgentState
    from app.services.llm_client import LLMClient
    from app.services.router_agent import RouterAgent
    from app.services.tools.registry import ToolRegistry
    from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class GraphContext:
    """Dependencies injected into node functions during graph execution."""

    llm_client: LLMClient | None = None
    vector_store: VectorStore | None = None
    tool_registry: ToolRegistry | None = None
    image_vector_store: VectorStore | None = None
    router_agent: RouterAgent | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Phase 1 addition: the production nodes below (validate_request_node
    # through finalizer_node) are thin wrappers that delegate to the
    # already-proven ChatService methods (_plan, _grade_retrieval,
    # _generate, _correct, _search_web, ...) rather than reimplementing
    # them, so they need a ChatService instance to call into. Optional so
    # existing GraphContext construction sites (the old planner_node etc.
    # above) are unaffected.
    chat_service: Any = None


async def planner_node(state: AgentState, context: GraphContext | None = None) -> AgentState:
    """Classifies user intent and sets state.plan."""
    query = state.query.strip()
    norm = _normalize(query)

    # 1. Fast path: conversational small talk
    for phrases, canned_resp in _CONVERSATIONAL_INTENTS:
        if norm in phrases:
            return state.copy_with(
                plan={"action": "conversational", "canned": canned_resp, "reason": "fast_path"},
                draft_answer=canned_resp,
                steps_taken=state.steps_taken + 1,
            )

    # 2. Fast path: document summarization with explicit UUID
    if _SUMMARIZE_RE.search(query):
        match = _DOCUMENT_ID_RE.search(query)
        if match:
            doc_id = match.group(0)
            return state.copy_with(
                plan={"action": "summarize", "document_id": doc_id, "reason": "fast_path"},
                document_id=doc_id,
                steps_taken=state.steps_taken + 1,
            )

    # 3. Router agent / LLM classification if available
    if context and context.router_agent:
        decision = context.router_agent.decide(query, state.history)
        action = decision.action
        routed_doc_id = decision.document_id or state.document_id
        return state.copy_with(
            plan={"action": action, "document_id": routed_doc_id, "reason": "router_agent"},
            document_id=routed_doc_id,
            steps_taken=state.steps_taken + 1,
        )

    # 4. Keyword heuristic fallback
    lower = query.lower()
    if any(term in lower for term in ["latest", "recent", "today", "news", "price of", "weather"]):
        action = "research" if settings.web_search_enabled else "retrieve"
    else:
        action = "retrieve"

    return state.copy_with(
        plan={"action": action, "document_id": state.document_id, "reason": "heuristic_fallback"},
        steps_taken=state.steps_taken + 1,
    )


async def document_analyst_node(
    state: AgentState,
    context: GraphContext | None = None,
) -> AgentState:
    """Executes dense + BM25 retrieval and chunk extraction."""
    if not context or not context.vector_store:
        logger.warning("document_analyst_node: no vector_store in context")
        return state.copy_with(steps_taken=state.steps_taken + 1)

    doc_filter = [state.document_id] if state.document_id else None
    chunks = retrieve(
        query=state.query,
        vector_store=context.vector_store,
        tenant_id=state.tenant_id,
        document_ids=doc_filter,
        image_vector_store=context.image_vector_store,
    )

    # Prompt injection heuristic logging for retrieved chunks
    for chunk in chunks:
        injections = detect_possible_injection(chunk.text)
        if injections:
            logger.warning(
                "possible_injection_detected",
                extra={"extra_fields": {"source": "chunk", "categories": injections}},
            )

    return state.copy_with(
        retrieved_chunks=chunks,
        steps_taken=state.steps_taken + 1,
    )


async def web_researcher_node(
    state: AgentState,
    context: GraphContext | None = None,
) -> AgentState:
    """Conducts external web search fallback when context is weak or query is external."""
    if not settings.web_search_enabled:
        return state.copy_with(
            metadata={"web_search": "disabled"},
            steps_taken=state.steps_taken + 1,
        )

    if settings.web_search_requires_approval and not state.confirm_web_search:
        return state.copy_with(
            metadata={"web_search": "pending_approval"},
            steps_taken=state.steps_taken + 1,
        )

    try:
        results = search_web(state.query, max_results=3)
    except Exception as exc:
        logger.warning("web_researcher_failed", extra={"extra_fields": {"error": str(exc)}})
        results = []

    return state.copy_with(
        web_results=results,
        steps_taken=state.steps_taken + 1,
    )


async def summarizer_node(
    state: AgentState,
    context: GraphContext | None = None,
) -> AgentState:
    """Full-document summarization for target document_id."""
    if not state.document_id:
        return state.copy_with(
            draft_answer="No document_id provided for summarization.",
            steps_taken=state.steps_taken + 1,
        )

    if not context or not context.vector_store or not context.llm_client:
        return state.copy_with(
            draft_answer="Summarization dependencies unavailable.",
            steps_taken=state.steps_taken + 1,
        )

    try:
        summary, chunks = summarize_document(
            document_id=state.document_id,
            vector_store=context.vector_store,
            llm_client=context.llm_client,
            tenant_id=state.tenant_id,
        )
        return state.copy_with(
            draft_answer=summary or "Could not generate summary.",
            retrieved_chunks=chunks,
            steps_taken=state.steps_taken + 1,
        )
    except Exception as exc:
        logger.warning("summarizer_node_failed", extra={"extra_fields": {"error": str(exc)}})
        return state.copy_with(
            draft_answer="I couldn't summarize that document.",
            error=str(exc),
            steps_taken=state.steps_taken + 1,
        )


async def synthesizer_node(
    state: AgentState,
    context: GraphContext | None = None,
) -> AgentState:
    """Synthesizes an answer using LLM with prompt injection delimiters & reflection."""
    # If conversational fast path already set draft_answer, construct final response
    if isinstance(state.plan, dict) and state.plan.get("action") == "conversational":
        answer = state.draft_answer or "Hello! How can I assist you with your documents?"
        chat_resp = ChatResponse(
            answer=answer,
            retrieved_chunks=[],
            sources=[],
            processing_time=0.0,
            tool_used="conversational",
            steps_taken=state.steps_taken + 1,
            session_id=state.session_id or "",
        )
        return state.copy_with(
            draft_answer=answer,
            final_response=chat_resp,
            steps_taken=state.steps_taken + 1,
        )

    if not context or not context.llm_client:
        answer = state.draft_answer or "Synthesizer LLM unavailable."
        return state.copy_with(
            draft_answer=answer,
            steps_taken=state.steps_taken + 1,
        )

    # Empty context check
    if not state.retrieved_chunks and not state.web_results:
        answer = FALLBACK_REPLY
        chat_resp = ChatResponse(
            answer=answer,
            retrieved_chunks=[],
            sources=[],
            processing_time=0.0,
            tool_used="retrieval",
            steps_taken=state.steps_taken + 1,
            session_id=state.session_id or "",
            retrieval_confidence="insufficient",
        )
        return state.copy_with(
            draft_answer=answer,
            final_response=chat_resp,
            steps_taken=state.steps_taken + 1,
        )

    extra_instruction = None
    if state.reflection_count > 0:
        extra_instruction = (
            f"{REFLECTION_INSTRUCTION} "
            "Ensure all cited claims strictly match the provided excerpts."
        )

    prompt = build_prompt(
        query=state.query,
        chunks=state.retrieved_chunks,
        history=state.history,
        extra_instruction=extra_instruction,
        web_results=state.web_results,
        persona=state.persona,
    )

    try:
        raw_answer = context.llm_client.generate(prompt)
        answer = strip_sources_section(raw_answer)
    except Exception as exc:
        logger.error("synthesizer_llm_failed", extra={"extra_fields": {"error": str(exc)}})
        # GENERATION_ERROR_REPLY, not FALLBACK_REPLY: this is a real LLM
        # provider failure (timeout/rate-limit/API error), not a grounded
        # "not in the documents" answer -- see the faithfulness-regression
        # fix in generator_node / docs/PHASE3_PRODUCTION_HARDENING_REPORT.md.
        answer = GENERATION_ERROR_REPLY

    # Build structured sources
    sources: list[SourceReference] = []
    if state.retrieved_chunks:
        sources.extend(_source_references(state.retrieved_chunks, start=1))
    if state.web_results:
        sources.extend(_web_source_references(state.web_results, start=len(sources) + 1))

    answer_source = "documents"
    if state.web_results and not state.retrieved_chunks:
        answer_source = "web"
    elif state.web_results and state.retrieved_chunks:
        answer_source = "mixed"

    chat_resp = ChatResponse(
        answer=answer,
        retrieved_chunks=state.retrieved_chunks,
        sources=sources,
        processing_time=0.0,
        tool_used="agent_graph",
        steps_taken=state.steps_taken + 1,
        answer_source=answer_source,
        session_id=state.session_id or "",
    )

    return state.copy_with(
        draft_answer=answer,
        final_response=chat_resp,
        steps_taken=state.steps_taken + 1,
    )


async def fact_checker_node(
    state: AgentState,
    context: GraphContext | None = None,
) -> AgentState:
    """Verifies answer citations and claims against ground-truth source chunks."""
    answer = state.draft_answer
    chunks = state.retrieved_chunks
    web_results = state.web_results

    if not answer or (not chunks and not web_results):
        return state.copy_with(
            fact_check_result={"verified": True, "skipped": True, "reason": "no_citations_needed"},
            steps_taken=state.steps_taken + 1,
        )

    citation_numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    if not citation_numbers:
        return state.copy_with(
            fact_check_result={"verified": True, "skipped": True, "reason": "no_inline_citations"},
            steps_taken=state.steps_taken + 1,
        )

    all_texts = {i + 1: c.text for i, c in enumerate(chunks)}
    start_web = len(chunks) + 1
    for i, w in enumerate(web_results):
        all_texts[start_web + i] = w.snippet

    # Check for non-existent citation references
    invalid_citations = [n for n in citation_numbers if n not in all_texts]
    if invalid_citations:
        return state.copy_with(
            fact_check_result={
                "verified": False,
                "score": 0.0,
                "invalid_citations": invalid_citations,
                "reason": "hallucinated_citation_index",
            },
            reflection_count=state.reflection_count + 1,
            steps_taken=state.steps_taken + 1,
        )

    if not settings.citation_verification_enabled or not context or not context.llm_client:
        return state.copy_with(
            fact_check_result={
                "verified": True,
                "skipped": True,
                "reason": "verification_disabled",
            },
            steps_taken=state.steps_taken + 1,
        )

    cited_pairs = [(n, all_texts[n]) for n in citation_numbers if n in all_texts]
    prompt = (
        "Answer:\n"
        + answer
        + "\n\n"
        + "\n\n".join(f"Excerpt [{n}]:\n{text}" for n, text in cited_pairs)
        + "\n\nFor each excerpt number above, does the answer's claim attributed to it "
        "match what that excerpt says? "
        'Respond with ONLY a JSON object like {"1": true, "2": false}.'
    )

    try:
        import json

        raw = context.llm_client.generate(prompt)
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        verifications = json.loads(raw)
        verified = all(bool(verifications.get(str(n), True)) for n, _ in cited_pairs)
        score = sum(1 for n, _ in cited_pairs if verifications.get(str(n), True)) / len(cited_pairs)
        result = {"verified": verified, "score": score, "details": verifications}
    except Exception as exc:
        logger.warning(
            "fact_check_verification_failed", extra={"extra_fields": {"error": str(exc)}}
        )
        result = {"verified": True, "skipped": True, "reason": "parse_error"}

    return state.copy_with(
        fact_check_result=result,
        reflection_count=state.reflection_count + (0 if result["verified"] else 1),
        steps_taken=state.steps_taken + 1,
    )


# ======================================================================
# Phase 1 production nodes — thin delegating wrappers around ChatService's
# existing, proven methods. These are what backend/app/services/agent_graph
# /graph.py's build_chat_graph() wires up; they are additive (the nodes
# above stay in place, unused by the new graph, until the deferred cleanup
# commit removes them together with the old /chat/agent-graph/stream path).
#
# Every node here follows the same shape: time the delegate call, run it in
# a try/except mapped onto the existing exceptions.py taxonomy, and emit one
# NodeTrace via events.emit_node_trace before returning the updated state.
# None of them contain business logic of their own.
# ======================================================================


def _node_error(exc: Exception) -> tuple[str, str]:
    """Map an exception to (error_type, root_cause) using the existing
    taxonomy where possible, 'unknown' root_cause when it can't be
    determined — never inventing a cause the system doesn't actually know."""
    if isinstance(exc, AppError):
        return exc.taxonomy_category, type(exc).__name__
    return "reasoning", "unknown"


def validate_request_node(state: AgentState, context: GraphContext | None = None) -> AgentState:
    """Initializes request_id/trace_id/workflow_start_time and rejects
    malformed input (empty query) safely rather than letting it fall
    through to the planner."""
    timer = NodeTimer()
    with timer:
        request_id = state.request_id or str(uuid.uuid4())
        trace_id = state.trace_id or request_id
        if not state.query or not state.query.strip():
            emit_node_trace(
                trace_id=trace_id,
                request_id=request_id,
                node="validate_request",
                status="failure",
                latency_ms=0.0,
                error_type="input",
            )
            return state.copy_with(
                request_id=request_id,
                trace_id=trace_id,
                workflow_status="failed",
                error_type="input",
                error_message="Empty query.",
                root_cause="empty_query",
                termination_reason="validation_failure",
                workflow_start_time=time.time(),
            )
        new_state = state.copy_with(
            request_id=request_id,
            trace_id=trace_id,
            workflow_status="running",
            workflow_start_time=state.workflow_start_time or time.time(),
            # Preserve a caller-supplied perf_start (handle_query passes its
            # own time.perf_counter() taken before routing/memory-injection,
            # so ChatResponse.processing_time covers the same window it
            # always has) rather than resetting the clock here.
            perf_start=state.perf_start if state.perf_start is not None else time.perf_counter(),
            steps_taken=state.steps_taken + 1,
        )
    emit_node_trace(
        trace_id=trace_id,
        request_id=request_id,
        node="validate_request",
        status="success",
        latency_ms=timer.latency_ms,
    )
    new_state.node_timings["validate_request"] = timer.latency_ms
    logger.info(REQUEST_VALIDATED, extra={"extra_fields": {"request_id": request_id}})
    return new_state


def planner_node_v2(state: AgentState, context: GraphContext | None = None) -> AgentState:
    """Delegates to ChatService._route — the deterministic keyword/regex
    planner (_plan), optionally upgraded by the LLM RouterAgent when
    Settings.agent_routing_enabled. Never replaces the deterministic
    planner itself with an LLM call; _route's own upgrade is an existing,
    already-gated behavior this just inherits.

    If the caller already computed a plan (state.planned_action is already
    set — handle_query does this before entering the graph, since it needs
    the decision to choose between the graph and the AgentExecutor branch
    before either runs), this is a no-op pass-through rather than a second,
    redundant _route call (which could re-invoke the RouterAgent LLM for no
    reason and, in principle, return a different decision on a second
    call).
    """
    if state.planned_action:
        return state.copy_with(steps_taken=state.steps_taken + 1)

    timer = NodeTimer()
    chat_service = context.chat_service if context else None
    try:
        with timer:
            if chat_service is None:
                raise RuntimeError("planner_node_v2 requires context.chat_service")
            decision = chat_service._route(state.query, state.history)  # noqa: SLF001
        new_state = state.copy_with(
            plan={
                "action": decision.action,
                "document_id": decision.document_id,
                "crop": decision.crop,
                "collection": decision.collection,
            },
            planned_action=decision.action,
            intent=decision.action,
            document_id=decision.document_id or state.document_id,
            planned_steps=[decision.action],
            steps_taken=state.steps_taken + 1,
        )
        emit_node_trace(
            trace_id=state.trace_id,
            request_id=state.request_id,
            node="planner",
            status="success",
            latency_ms=timer.latency_ms,
            extra={"action": decision.action},
        )
    except Exception as exc:
        error_type, root_cause = _node_error(exc)
        emit_node_trace(
            trace_id=state.trace_id,
            request_id=state.request_id,
            node="planner",
            status="failure",
            latency_ms=timer.latency_ms,
            error_type=error_type,
        )
        # Recovered, not fatal: the workflow continues on the safe
        # deterministic default exactly as ChatService's own planner
        # degrades — so this must NOT leave error_type/root_cause set on
        # state, or finalizer_node would misread a later-successful
        # workflow as having failed. The failure is still fully traced via
        # emit_node_trace above.
        new_state = state.copy_with(
            planned_action="retrieve",
            intent="retrieve",
            steps_taken=state.steps_taken + 1,
        )
    new_state.node_timings["planner"] = timer.latency_ms
    return new_state


def vision_node(state: AgentState, context: GraphContext | None = None) -> AgentState:
    """Delegates to vision_client.diagnose_image (LeafSense/Gemini
    prediction) and rag_service's _build_diagnosis_query/_build_diagnosis_info
    — the same functions handle_diagnose already used inline. Image bytes
    are passed via context.metadata (not AgentState) deliberately: the
    engine deep-copies a state snapshot on every node transition for its
    step-history, and an image can be multiple MB — repeating that copy at
    every step for the rest of the workflow would be wasteful and is
    exactly what the "don't unnecessarily duplicate large document bodies
    inside state" state-design rule is for. Only the resulting prediction
    (DiagnosisInfo, already small) and the derived text query go into
    state.

    Never raises credentials/secrets into logs or state — diagnose_image's
    own errors (VisionServiceError) already carry no image bytes.
    """
    timer = NodeTimer()
    logger.info(VISION_STARTED, extra={"extra_fields": {"request_id": state.request_id}})
    image_bytes = context.metadata.get("image_bytes") if context else None
    if image_bytes is None:
        emit_node_trace(
            trace_id=state.trace_id,
            request_id=state.request_id,
            node="vision",
            status="failure",
            latency_ms=0.0,
            error_type="tool",
        )
        raise RuntimeError("vision_node requires context.metadata['image_bytes'].")
    try:
        with timer:
            # Delegate through the rag_service module attribute — same
            # monkeypatch-compatibility reasoning as retrieval_node's
            # `_rag_service_module.retrieve(...)` call (see its comment).
            filename = context.metadata.get("filename", "upload")
            content_type = context.metadata.get("content_type", "application/octet-stream")
            engine = context.metadata.get("engine", "hybrid")
            prediction = _rag_service_module.diagnose_image(
                image_bytes, filename, content_type, engine=engine
            )
            _build_diagnosis_query = _rag_service_module._build_diagnosis_query
            _build_diagnosis_info = _rag_service_module._build_diagnosis_info
            crop_context = (
                prediction.crop if prediction.crop and prediction.crop != "unknown" else None
            )
            disease_context = (
                prediction.disease if prediction.disease and prediction.disease != "unknown" else None
            )
            diagnosis_query = _build_diagnosis_query(prediction, state.query, collection=crop_context)
            diagnosis_info = _build_diagnosis_info(prediction)
        emit_node_trace(
            trace_id=state.trace_id,
            request_id=state.request_id,
            node="vision",
            status="success",
            latency_ms=timer.latency_ms,
            extra={"crop": prediction.crop, "disease": prediction.disease},
        )
        logger.info(
            VISION_COMPLETED,
            extra={"extra_fields": {"crop": prediction.crop, "disease": prediction.disease}},
        )
        new_state = state.copy_with(
            diagnosis=diagnosis_info,
            retrieval_query=diagnosis_query,
            metadata={**state.metadata, "crop_context": crop_context, "disease_context": disease_context},
            steps_taken=state.steps_taken + 1,
        )
    except Exception as exc:
        # Not swallowed: a vision failure (VisionServiceError, etc.) has no
        # fallback path in handle_diagnose — the whole request depends on
        # the prediction — so this propagates exactly like handle_diagnose's
        # own `except AppError: raise` / `except Exception: raise
        # ChatServiceError(...)` always has, rather than silently
        # degrading into a request with no diagnosis. Traced before
        # re-raising, same pattern as retrieval_node's real-failure path.
        error_type, _root_cause = _node_error(exc)
        emit_node_trace(
            trace_id=state.trace_id,
            request_id=state.request_id,
            node="vision",
            status="failure",
            latency_ms=timer.latency_ms,
            error_type=error_type,
        )
        raise
    new_state.node_timings["vision"] = timer.latency_ms
    return new_state


def retrieval_node(state: AgentState, context: GraphContext | None = None) -> AgentState:
    """Delegates to retrieval_service.retrieve(), which already performs
    hybrid BM25+FAISS retrieval AND cross-encoder reranking internally when
    enabled. This node records the combined outcome (result_count,
    whether reranking ran, latency) as ONE traced step — it does not run a
    second, independent reranking pass, and does not claim to."""
    timer = NodeTimer()
    logger.info(RETRIEVAL_STARTED, extra={"extra_fields": {"request_id": state.request_id}})
    if not context or not context.vector_store:
        emit_node_trace(
            trace_id=state.trace_id,
            request_id=state.request_id,
            node="retrieval",
            status="failure",
            latency_ms=0.0,
            error_type="retriever",
        )
        return state.copy_with(
            retrieved_chunks=[],
            reranked_chunks=[],
            error_type="retriever",
            error_message="No vector store available.",
            root_cause="vector_store_unavailable",
            steps_taken=state.steps_taken + 1,
        )
    chat_service = context.chat_service if context else None
    plan = state.plan if isinstance(state.plan, dict) else {}
    try:
        with timer:
            # Query contextualization: rewrite a follow-up into a
            # standalone question before retrieval, exactly like
            # handle_query's `retrieval_query` split — every other use of
            # the request (generation, citations, caching, logging) stays
            # on state.query, only the value sent to retrieve() changes.
            retrieval_query = state.query
            recent_history = state.history[-_MAX_HISTORY_TURNS:] if state.history else None
            if settings.query_contextualization_enabled and recent_history and chat_service is not None:
                retrieval_query = chat_service._contextualize_query(  # noqa: SLF001
                    state.query, recent_history
                )

            retrieve_kwargs: dict[str, Any] = {
                "top_k": state.retrieval_top_k,
                "min_score": state.retrieval_min_score,
                "tenant_id": state.tenant_id,
                "image_vector_store": context.image_vector_store,
            }
            doc_ids = state.document_ids or ([state.document_id] if state.document_id else None)
            if doc_ids is not None:
                retrieve_kwargs["document_ids"] = doc_ids
            elif plan.get("collection"):
                retrieve_kwargs["collection"] = plan.get("collection")
            if plan.get("crop") is not None or plan.get("collection") is not None:
                retrieve_kwargs["rerank"] = True

            # Delegate through the rag_service module attribute (not the
            # `retrieve` name imported above) so this node observes the
            # same monkeypatch surface existing tests already rely on
            # (`monkeypatch.setattr(rag_service_module, "retrieve", ...)`),
            # matching handle_query's own call site exactly.
            chunks = _rag_service_module.retrieve(retrieval_query, context.vector_store, **retrieve_kwargs)
        was_reranked = any(
            isinstance(getattr(c, "metadata", None), dict) and "rerank_score" in c.metadata
            for c in chunks
        )
        emit_node_trace(
            trace_id=state.trace_id,
            request_id=state.request_id,
            node="retrieval",
            status="success",
            latency_ms=timer.latency_ms,
            extra={"result_count": len(chunks), "reranked": was_reranked},
        )
        logger.info(
            RETRIEVAL_COMPLETED,
            extra={"extra_fields": {"result_count": len(chunks), "reranked": was_reranked}},
        )
        new_state = state.copy_with(
            retrieved_chunks=chunks,
            reranked_chunks=chunks,
            retrieval_query=retrieval_query,
            steps_taken=state.steps_taken + 1,
        )
    except Exception as exc:
        # Deliberately NOT swallowed into a degraded state: retrieve()
        # raising is a real failure (a corrupted/misconfigured vector
        # store, an embedding error, ...) that handle_query/stream_query
        # have always let propagate into a 500 (ChatServiceError) / SSE
        # "error" event respectively — not a silent empty-context fallback.
        # Only the "no vector_store in context" case above is a genuine,
        # intentional degrade — this is not. Traced before re-raising so
        # the failure is still observable, then handled by whichever
        # caller invoked this node: the graph engine's own try/except (for
        # handle_query, which stops the run and surfaces
        # result_state.error_message via _run_chat_graph) or stream_query's
        # surrounding try/except (called directly, not through the
        # engine, so the exception reaches it unchanged).
        error_type, root_cause = _node_error(exc)
        emit_node_trace(
            trace_id=state.trace_id,
            request_id=state.request_id,
            node="retrieval",
            status="failure",
            latency_ms=timer.latency_ms,
            error_type=error_type,
        )
        raise
    new_state.node_timings["retrieval"] = timer.latency_ms
    return new_state


def retrieval_grader_node(state: AgentState, context: GraphContext | None = None) -> AgentState:
    """Delegates to ChatService._grade_retrieval (heuristic good/weak/
    insufficient grade already used by the production /chat pipeline)."""
    timer = NodeTimer()
    chat_service = context.chat_service if context else None
    with timer:
        if chat_service is not None:
            grade = chat_service._grade_retrieval(state.query, state.retrieved_chunks)  # noqa: SLF001
        else:
            grade = "insufficient" if not state.retrieved_chunks else "weak"
    reason = (
        "no_chunks_survived_min_score"
        if grade == "insufficient"
        else "below_grade_threshold"
        if grade == "weak"
        else "above_grade_threshold"
    )
    emit_node_trace(
        trace_id=state.trace_id,
        request_id=state.request_id,
        node="retrieval_grader",
        status="success",
        latency_ms=timer.latency_ms,
        extra={"grade": grade},
    )
    logger.info(RETRIEVAL_GRADED, extra={"extra_fields": {"grade": grade, "reason": reason}})

    # PHASE 5 FIX: web search is the guarded action a weak/insufficient
    # grade escalates to (see route_after_grader/context_augmentation_node).
    # When Settings.web_search_requires_approval is on and the caller
    # hasn't already satisfied the gate (confirm_web_search=true, the
    # existing fast path, or a genuinely-approved approval already
    # attached via approval_payload_reference), flag the state so
    # route_after_grader sends this request through human_approval_node
    # instead of straight to context_augmentation -- making the
    # already-registered Approval record (see ChatService._search_web)
    # actually gate the action, not just log it for visibility.
    approval_required = (
        grade != "good"
        and settings.web_search_requires_approval
        and not state.confirm_web_search
        and state.approval_status != "approved"
    )
    new_state = state.copy_with(
        retrieval_grade=grade,
        retrieval_grade_reason=reason,
        steps_taken=state.steps_taken + 1,
        **(
            {
                "approval_required": True,
                "approval_type": "web_search",
                "approval_reason": "weak_retrieval_web_search_fallback",
            }
            if approval_required
            else {}
        ),
    )
    new_state.node_timings["retrieval_grader"] = timer.latency_ms
    return new_state


def web_research_node_v2(state: AgentState, context: GraphContext | None = None) -> AgentState:
    """Delegates to ChatService._search_web, which already enforces the
    web_search_requires_approval gate (skips + registers a pending Approval
    when confirm_web_search wasn't set) — this node does not duplicate that
    check, it inherits it."""
    timer = NodeTimer()
    chat_service = context.chat_service if context else None
    logger.info(WEB_SEARCH_STARTED, extra={"extra_fields": {"request_id": state.request_id}})
    tool_call_entry: dict[str, Any] = {
        "tool_name": "web_search",
        "input_summary": {"query_length": len(state.query)},
        "timestamp": time.time(),
    }
    try:
        with timer:
            if chat_service is not None:
                results = chat_service._search_web(  # noqa: SLF001
                    state.query, confirm_web_search=state.confirm_web_search
                )
            else:
                results = []
        tool_call_entry.update(
            success=True, output_summary={"result_count": len(results)}, latency_ms=timer.latency_ms
        )
        emit_node_trace(
            trace_id=state.trace_id,
            request_id=state.request_id,
            node="web_research",
            status="success",
            latency_ms=timer.latency_ms,
            extra={"result_count": len(results)},
        )
        logger.info(WEB_SEARCH_COMPLETED, extra={"extra_fields": {"result_count": len(results)}})
        new_state = state.copy_with(
            web_results=results,
            tool_calls=[*state.tool_calls, tool_call_entry],
            steps_taken=state.steps_taken + 1,
        )
    except Exception as exc:
        error_type, root_cause = _node_error(exc)
        tool_call_entry.update(success=False, error_type=error_type, latency_ms=timer.latency_ms)
        emit_node_trace(
            trace_id=state.trace_id,
            request_id=state.request_id,
            node="web_research",
            status="failure",
            latency_ms=timer.latency_ms,
            error_type=error_type,
        )
        new_state = state.copy_with(
            web_results=[],
            tool_calls=[*state.tool_calls, tool_call_entry],
            error_type=error_type,
            error_message=str(exc),
            root_cause=root_cause,
            steps_taken=state.steps_taken + 1,
        )
    new_state.node_timings["web_research"] = timer.latency_ms
    return new_state


def generator_node(state: AgentState, context: GraphContext | None = None) -> AgentState:
    """Delegates to ChatService._generate (or _generate_structured when the
    caller opted into structured mode via state.metadata["structured"])."""
    timer = NodeTimer()
    chat_service = context.chat_service if context else None
    logger.info(GENERATION_STARTED, extra={"extra_fields": {"request_id": state.request_id}})
    extra_instruction = None
    if state.reflection_count_v2 > 0:
        extra_instruction = REFLECTION_INSTRUCTION
    recent_history = state.history[-_MAX_HISTORY_TURNS:] if state.history else None
    language = state.metadata.get("language", "en")
    structured_payload: dict[str, Any] | None = None
    try:
        with timer:
            if chat_service is None:
                raise RuntimeError("generator_node requires context.chat_service")
            if settings.structured_output_enabled and state.metadata.get("structured_response"):
                answer, structured_payload = chat_service._generate_structured(  # noqa: SLF001
                    state.query,
                    state.retrieved_chunks,
                    recent_history,
                    web_results=state.web_results,
                    persona=state.persona,
                    language=language,
                )
            else:
                answer = chat_service._generate(  # noqa: SLF001
                    state.query,
                    state.retrieved_chunks,
                    recent_history,
                    extra_instruction=extra_instruction,
                    web_results=state.web_results,
                    persona=state.persona,
                    language=language,
                )
        emit_node_trace(
            trace_id=state.trace_id,
            request_id=state.request_id,
            node="generator",
            status="success",
            latency_ms=timer.latency_ms,
        )
        logger.info(GENERATION_COMPLETED, extra={"extra_fields": {"answer_length": len(answer)}})
        new_state = state.copy_with(
            draft_answer=answer, structured_output=structured_payload, steps_taken=state.steps_taken + 1
        )
    except Exception as exc:
        error_type, root_cause = _node_error(exc)
        emit_node_trace(
            trace_id=state.trace_id,
            request_id=state.request_id,
            node="generator",
            status="failure",
            latency_ms=timer.latency_ms,
            error_type=error_type,
        )
        # PHASE 3 FIX (faithfulness regression root cause): this except
        # block used to set draft_answer=FALLBACK_REPLY here -- the exact
        # same text used for a genuine "the documents don't contain this"
        # answer. That made an LLM provider failure (timeout, rate limit,
        # API error surviving all 3 tenacity retries -- see
        # groq_client.py/gemini_client.py) INDISTINGUISHABLE from a
        # legitimate grounded non-answer, both to end users and to the
        # Faithfulness/grounding evaluators (which correctly scored 0.0 for
        # a response containing zero real claims -- the metric was right,
        # the underlying answer was mislabeled). error_type/root_cause are
        # still recorded on state for tracing; GENERATION_ERROR_REPLY makes
        # the user-visible text honest about what actually happened.
        new_state = state.copy_with(
            draft_answer=GENERATION_ERROR_REPLY,
            error_type=error_type,
            error_message=str(exc),
            root_cause=root_cause,
            steps_taken=state.steps_taken + 1,
        )
    new_state.node_timings["generator"] = timer.latency_ms
    return new_state


def reflection_node(state: AgentState, context: GraphContext | None = None) -> AgentState:
    """Delegates wholesale to ChatService._correct — the corrective RAG
    loop that already regenerates once if the initial answer looks
    ungrounded, then escalates to a web-search fallback and regenerates
    again if still ungrounded (capped at _MAX_LLM_CALLS total generate()
    calls). This is a single node call, not a generic "reflect, then loop
    back to generator" graph edge: _correct's second stage specifically
    escalates to web search rather than just repeating the same
    instruction, which doesn't decompose cleanly into a generic instruction-
    repeat loop without either reimplementing that escalation logic here
    (duplicating proven, already-tested code) or losing it. Delegating the
    whole method preserves it exactly, at the cost of the loop being
    internal to this one node rather than expressed as a StateGraph edge —
    still fully bounded (by _MAX_LLM_CALLS), still traced, still a distinct
    named node per the workflow spec.

    Runs unconditionally after generator_node, exactly like handle_query's
    unconditional `self._correct(...)` call after its initial `_generate`.
    """
    timer = NodeTimer()
    chat_service = context.chat_service if context else None
    recent_history = state.history[-_MAX_HISTORY_TURNS:] if state.history else None
    language = state.metadata.get("language", "en")
    if chat_service is None:
        new_state = state.copy_with(steps_taken=state.steps_taken + 1)
        new_state.node_timings["reflection"] = 0.0
        return new_state
    try:
        with timer:
            answer, llm_calls, steps_taken, web_results, web_search_attempted = chat_service._correct(  # noqa: SLF001
                state.query,
                state.retrieved_chunks,
                state.draft_answer,
                recent_history,
                state.web_results,
                bool(state.metadata.get("web_search_attempted", False)),
                1,  # llm_calls so far: generator_node's initial call
                state.steps_taken,
                confirm_web_search=state.confirm_web_search,
                persona=state.persona,
                language=language,
            )
        corrected = answer != state.draft_answer
        emit_node_trace(
            trace_id=state.trace_id,
            request_id=state.request_id,
            node="reflection",
            status="success",
            latency_ms=timer.latency_ms,
            extra={"corrected": corrected, "llm_calls": llm_calls},
        )
        if corrected:
            get_metrics().record_agent_reflection()
            if llm_calls > 2:  # initial call + reflection regenerate + web-fallback regenerate
                get_metrics().record_agent_retry(node="reflection")
        new_state = state.copy_with(
            draft_answer=answer,
            web_results=web_results,
            metadata={**state.metadata, "web_search_attempted": web_search_attempted},
            reflection_count_v2=state.reflection_count_v2 + (1 if corrected else 0),
            steps_taken=steps_taken,
        )
    except Exception as exc:
        error_type, root_cause = _node_error(exc)
        emit_node_trace(
            trace_id=state.trace_id,
            request_id=state.request_id,
            node="reflection",
            status="failure",
            latency_ms=timer.latency_ms,
            error_type=error_type,
        )
        new_state = state.copy_with(
            error_type=error_type,
            error_message=str(exc),
            root_cause=root_cause,
            steps_taken=state.steps_taken + 1,
        )
    new_state.node_timings["reflection"] = timer.latency_ms
    return new_state


def output_validation_node(state: AgentState, context: GraphContext | None = None) -> AgentState:
    """Post-hoc bookkeeping after reflection_node's correction pass:
    records whether the (already-corrected) answer is still ungrounded, for
    tracing/termination_reason purposes. Never loops back — _correct
    already exhausted its own correction budget, so there is nothing left
    to retry; this node's only job is to make that outcome explicit and
    traceable rather than silent."""
    timer = NodeTimer()
    chat_service = context.chat_service if context else None
    logger.info(VALIDATION_STARTED, extra={"extra_fields": {"request_id": state.request_id}})
    with timer:
        if chat_service is not None:
            ungrounded = chat_service._is_ungrounded(  # noqa: SLF001
                state.draft_answer, state.retrieved_chunks, state.web_results
            )
        else:
            ungrounded = not state.draft_answer
    errors = ["ungrounded_answer"] if ungrounded else []
    if errors:
        logger.info(VALIDATION_FAILED, extra={"extra_fields": {"errors": errors}})
    emit_node_trace(
        trace_id=state.trace_id,
        request_id=state.request_id,
        node="output_validation",
        status="success" if not errors else "failure",
        latency_ms=timer.latency_ms,
        extra={"validation_errors": errors},
    )
    new_state = state.copy_with(
        validation_errors=errors,
        final_answer=state.draft_answer,
        steps_taken=state.steps_taken + 1,
    )
    new_state.node_timings["output_validation"] = timer.latency_ms
    return new_state


def finalizer_node(state: AgentState, context: GraphContext | None = None) -> AgentState:
    """Builds the final ChatResponse.

    When a ChatService is available (the real production path), this
    delegates to `ChatService._respond` — which already does usage/cost
    rollup, hallucination detection, structured "chat_query_handled"
    logging, and agent-memory recording — so none of that is reimplemented
    here. It also applies the same post-processing handle_query did inline:
    `_maybe_ask_clarifying_question` and `_suggest_follow_ups` (for
    document-query answers only), and writes the response cache for plain
    "retrieve" actions via `_cache_response` (never for "research", whose
    answer depends on live web state — same rule handle_query followed).

    Without a ChatService (isolated node/graph unit tests), falls back to a
    minimal hand-built ChatResponse using the existing citation helpers —
    enough to exercise routing/state without needing a full ChatService
    double wired up for every test.
    """
    timer = NodeTimer()
    chat_service = context.chat_service if context else None
    action = state.planned_action or "retrieve"
    query_type = {"conversational": "conversational", "summarize": "summarize"}.get(
        action, "document_query"
    )
    answer = state.final_answer or state.draft_answer or FALLBACK_REPLY
    tool_used = state.metadata.get("tool_used") or state.metadata.get("direct_answer_hit")
    if tool_used is None:
        if action == "conversational":
            tool_used = "none"
        elif action == "summarize":
            tool_used = "summarization"
        elif state.web_results:
            tool_used = "web_search"
        else:
            tool_used = "retrieval"

    is_clarifying_question = False
    follow_up_questions: list[str] = []

    with timer:
        if chat_service is not None:
            if query_type == "document_query":
                answer, is_clarifying_question = chat_service._maybe_ask_clarifying_question(  # noqa: SLF001
                    state.query, answer, state.retrieval_grade or "good"
                )
                if settings.follow_up_questions_enabled:
                    follow_up_questions = chat_service._suggest_follow_ups(state.query, answer)  # noqa: SLF001

            chat_resp = chat_service._respond(  # noqa: SLF001
                answer=answer,
                retrieved_chunks=state.retrieved_chunks,
                query=state.query,
                query_type=query_type,
                tool_used=tool_used,
                steps_taken=state.steps_taken + 1,
                start=state.perf_start if state.perf_start is not None else time.perf_counter(),
                web_results=state.web_results,
                session_id=state.session_id,
                retrieval_confidence=state.retrieval_grade or "good",
                is_clarifying_question=is_clarifying_question,
                follow_up_questions=follow_up_questions,
            )
            # PHASE 5 FIX: never cache a response that only exists because a
            # guarded action (web search) was blocked pending/rejected/
            # expired approval -- it's an incomplete answer for this
            # specific request, not a reusable good answer for the query in
            # general. A later, approved retry of the same query must not
            # be served this placeholder from cache.
            if action == "retrieve" and state.approval_status not in ("pending", "rejected", "expired"):
                plan = state.plan if isinstance(state.plan, dict) else {}
                chat_service._cache_response(  # noqa: SLF001
                    query=state.query,
                    response=chat_resp,
                    crop=plan.get("crop"),
                    disease=plan.get("disease"),
                    tenant_id=state.tenant_id,
                    document_ids=state.document_ids,
                )
            source_type = chat_resp.answer_source
            final_sources = [s.model_dump() for s in chat_resp.sources]
            if state.metadata.get("structured_response"):
                # Only set when the caller actually opted into structured
                # mode (ChatRequest.structured_response=true) -- absent
                # otherwise, so a normal free-text response's metadata is
                # unchanged. True only when the provider's output genuinely
                # parsed/validated (see ChatService._generate_structured);
                # a fallback-to-free-text degrade is reported as False, not
                # silently presented as a successful structured response.
                chat_resp.metadata["structured_output_used"] = state.structured_output is not None
                if state.structured_output is not None:
                    chat_resp.metadata["structured_output"] = state.structured_output
            if state.approval_status in ("pending", "rejected", "expired"):
                chat_resp.metadata["approval_status"] = state.approval_status
                if state.approval_payload_reference:
                    chat_resp.metadata["approval_id"] = state.approval_payload_reference
        else:
            sources: list[SourceReference] = []
            if state.retrieved_chunks:
                sources.extend(_source_references(state.retrieved_chunks, start=1))
            if state.web_results:
                sources.extend(_web_source_references(state.web_results, start=len(sources) + 1))
            source_type = "documents"
            if state.web_results and not state.retrieved_chunks:
                source_type = "web"
            elif state.web_results and state.retrieved_chunks:
                source_type = "mixed"
            chat_resp = ChatResponse(
                answer=answer,
                retrieved_chunks=state.retrieved_chunks,
                sources=sources,
                processing_time=(time.perf_counter() - state.perf_start) if state.perf_start else 0.0,
                tool_used=tool_used,
                steps_taken=state.steps_taken + 1,
                answer_source=source_type,
                session_id=state.session_id or "",
            )
            final_sources = [s.model_dump() for s in sources]

        termination_reason = state.termination_reason
        if termination_reason is None:
            if state.error_type:
                termination_reason = "model_failure" if state.error_type == "reasoning" else "tool_failure"
            elif state.validation_errors and state.reflection_count_v2 >= 2:
                termination_reason = "loop_limit_reached"
            elif state.approval_status == "rejected":
                termination_reason = "approval_rejected"
            else:
                termination_reason = "success"
        workflow_end_time = time.time()

    emit_node_trace(
        trace_id=state.trace_id,
        request_id=state.request_id,
        node="finalizer",
        status="success",
        latency_ms=timer.latency_ms,
        extra={"termination_reason": termination_reason},
    )
    logger.info(FINALIZED, extra={"extra_fields": {"termination_reason": termination_reason}})
    new_state = state.copy_with(
        final_answer=chat_resp.answer,
        final_response=chat_resp,
        source_type=source_type,
        final_sources=final_sources,
        workflow_end_time=workflow_end_time,
        workflow_status="completed" if termination_reason == "success" else "failed",
        termination_reason=termination_reason,
        steps_taken=state.steps_taken + 1,
    )
    new_state.node_timings["finalizer"] = timer.latency_ms
    return new_state
