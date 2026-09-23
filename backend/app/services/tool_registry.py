"""Formal tool specs (name, description, I/O schema) and a tool-invocation
tracking decorator.

The agent's tools (retrieval, summarization, web search, diagnose) each get
a name, a human-readable description, and a declared Pydantic input and
output schema here, so tool arguments are validated at the boundary instead
of being untyped function parameters, and so the tool registry is the single
source of truth for what each tool is, takes, and returns — read directly by
prompt_builder.AGENT_TOOLS and the eval harness.

@track_tool wraps a tool function and emits a structured "tool_invocation"
log event on every call — name, input summary, output summary, success/
failure flag, error type, and latency — which is what Tool Success Rate is
computed from at runtime (see eval/ and monitoring/).

Inputs are validated against the schema *before* the wrapped call runs; a
validation failure raises ToolInputError (an AppError, 422) rather than
proceeding, so a malformed tool call surfaces loudly instead of failing
halfway through the pipeline. Outputs are validated *after* the wrapped
call returns, against the tool's actual return type (see TOOL_SCHEMAS);
a violation raises ToolOutputError (500) — the tool ran, but broke its
own contract, which is a bug worth surfacing loudly rather than passing
malformed data downstream.
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from app.core.exceptions import AppError, ChatServiceError
from app.core.metrics import get_metrics
from app.models.document import RetrievedChunk, VisionPrediction, WebSearchResult

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class ToolInputError(ChatServiceError):
    """Raised when a tool is invoked with arguments that fail its schema."""

    status_code = 422
    error_code = "TOOL_INPUT_ERROR"

    def __init__(self, tool: str, detail: str):
        super().__init__(f"Invalid arguments for tool '{tool}': {detail}")


class ToolOutputError(ChatServiceError):
    """Raised when a tool's own return value fails its declared output
    schema — the tool ran successfully but produced something outside its
    contract. Distinct from ToolInputError (422, caller's fault): this is
    a 500, since it means the tool itself is broken, not the request."""

    status_code = 500
    error_code = "TOOL_OUTPUT_ERROR"

    def __init__(self, tool: str, detail: str):
        super().__init__(f"Tool '{tool}' returned output that fails its schema: {detail}")


# --- Per-tool input schemas -------------------------------------------------


class RetrievalInput(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, gt=0)
    min_score: float | None = Field(default=None, ge=-1.0, le=1.0)


class SummarizationInput(BaseModel):
    document_id: str = Field(min_length=1)


class WebSearchInput(BaseModel):
    query: str = Field(min_length=1)
    max_results: int | None = Field(default=None, gt=0)


class DiagnoseInput(BaseModel):
    contents: bytes
    filename: str = Field(min_length=1)
    content_type: str = Field(min_length=1)
    query: str | None = None


# --- Registry ----------------------------------------------------------------
#
# "output" is the tool function's actual return type (not a separate wrapper
# schema nothing produces) — retrieve() really does return list[RetrievedChunk],
# summarize_document() really does return tuple[str, list[RetrievedChunk]], and
# so on. track_tool validates the real return value against this type via
# TypeAdapter, so a tool whose implementation drifts from its declared
# contract is caught at the boundary instead of passing malformed data on.

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "retrieval": {
        "description": "Vector search over the uploaded PDFs' embedded chunks; returns the top-k passages most relevant to the query.",
        "input": RetrievalInput,
        "output": list[RetrievedChunk],
    },
    "summarization": {
        "description": "Whole-document summary built from every chunk belonging to one uploaded document.",
        "input": SummarizationInput,
        "output": tuple[str, list[RetrievedChunk]],
    },
    "web_search": {
        "description": "DuckDuckGo web search fallback for queries the uploaded corpus can't answer (opt-in, human-approval gated).",
        "input": WebSearchInput,
        "output": list[WebSearchResult],
    },
    "diagnose": {
        "description": "LeafSense HTTP vision integration: classifies a crop-leaf image into a disease class, mapped to a plain-language crop/disease/confidence.",
        "input": DiagnoseInput,
        "output": VisionPrediction,
    },
}
# Descriptions are read directly (TOOL_SCHEMAS[name]["description"]) by
# prompt_builder.AGENT_TOOLS, so the model-facing tool list can't drift
# from what's actually registered.

# Built once rather than per-call — TypeAdapter's schema build isn't free,
# and the output type per tool is static.
_OUTPUT_ADAPTERS: dict[str, TypeAdapter[Any]] = {
    tool_name: TypeAdapter(spec["output"]) for tool_name, spec in TOOL_SCHEMAS.items()
}


def track_tool(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: validates args against the tool's input schema, runs the
    wrapped function, and logs one structured tool_invocation event.

    Success, failure, error type, and latency are all captured so runtime
    Tool Success Rate is computable from the log (each event is one
    attempt, exactly, on success or failure).
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Bind call-site args to parameter names so schemas can be applied
        # regardless of whether callers pass args positionally or by keyword.
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            input_schema = TOOL_SCHEMAS[name]["input"]
            # Only schema-declared fields are validated — tool functions take
            # injected collaborators (vector_store, llm_client) that aren't
            # part of the tool's user-facing I/O contract.
            schema_fields = {
                k: v for k, v in bound.arguments.items() if k in input_schema.model_fields
            }
            try:
                validated = input_schema.model_validate(schema_fields)
            except ValidationError as exc:
                raise ToolInputError(name, str(exc)) from exc

            input_summary = _summarize_input(validated.model_dump())
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except AppError as exc:
                latency_ms = (time.perf_counter() - start) * 1000
                get_metrics().record_tool_invocation(
                    tool=name, success=False, latency_ms=latency_ms
                )
                logger.warning(
                    "tool_invocation",
                    extra={
                        "extra_fields": {
                            "tool": name,
                            "success": False,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc)[:200],
                            "timed_out": _looks_like_timeout(exc),
                            "input": input_summary,
                            "latency_ms": round(latency_ms, 2),
                        }
                    },
                )
                raise
            except Exception as exc:
                latency_ms = (time.perf_counter() - start) * 1000
                get_metrics().record_tool_invocation(
                    tool=name, success=False, latency_ms=latency_ms
                )
                logger.warning(
                    "tool_invocation",
                    extra={
                        "extra_fields": {
                            "tool": name,
                            "success": False,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc)[:200],
                            "timed_out": _looks_like_timeout(exc),
                            "input": input_summary,
                            "latency_ms": round(latency_ms, 2),
                        }
                    },
                )
                raise

            try:
                _OUTPUT_ADAPTERS[name].validate_python(result)
            except ValidationError as exc:
                latency_ms = (time.perf_counter() - start) * 1000
                get_metrics().record_tool_invocation(
                    tool=name, success=False, latency_ms=latency_ms
                )
                logger.warning(
                    "tool_invocation",
                    extra={
                        "extra_fields": {
                            "tool": name,
                            "success": False,
                            "error_type": "ToolOutputError",
                            "input": input_summary,
                            "latency_ms": round(latency_ms, 2),
                        }
                    },
                )
                raise ToolOutputError(name, str(exc)) from exc

            latency_ms = (time.perf_counter() - start) * 1000
            get_metrics().record_tool_invocation(tool=name, success=True, latency_ms=latency_ms)
            logger.info(
                "tool_invocation",
                extra={
                    "extra_fields": {
                        "tool": name,
                        "success": True,
                        "input": input_summary,
                        "output": _summarize_output(result),
                        "latency_ms": round(latency_ms, 2),
                    }
                },
            )
            return result

        return wrapper

    return decorator


def _looks_like_timeout(exc: Exception) -> bool:
    """Best-effort timeout detection for the Timeout Rate metric (see
    monitoring/log_aggregate.py, eval/metrics_report.py).

    Each tool wraps heterogeneous underlying errors (httpx, DDGS, thread
    futures) into one broad AppError subclass — e.g. VisionServiceError
    covers a LeafSense timeout, a bad status, and a malformed response
    alike — so there's no single exception *type* that means "this was a
    timeout" the way there would be if each failure mode got its own
    exception class. The type name and message are what's actually
    distinctive instead.
    """
    return "timeout" in type(exc).__name__.lower() or "timed out" in str(exc).lower()


def _summarize_input(data: dict[str, Any]) -> dict[str, Any]:
    """Trim values that would bloat the log (long query strings, image
    bytes) to a short length — the event records shape, not content."""
    summary = {}
    for key, value in data.items():
        if isinstance(value, bytes):
            summary[key] = f"<{len(value)} bytes>"
        elif isinstance(value, str) and len(value) > 120:
            summary[key] = value[:120] + "..."
        else:
            summary[key] = value
    return summary


def _summarize_output(result: Any) -> Any:
    """Trim a tool's return value to a short, loggable shape.

    Mirrors _summarize_input's philosophy (shape over content, bounded
    size) for the success case — the checklist's "tool outputs visible"
    gap. Handles the shapes the registry's tools actually return:
    - list[RetrievedChunk] / list[WebSearchResult] → count + chunk ids
      (what the tool produced, not its full text)
    - tuple (summarization's real (summary, chunks) return) → each
      element summarized the same way, recursively
    - str → truncated text
    - dict / a model with model_dump() (diagnose's VisionPrediction) →
      truncated fields
    """
    if result is None:
        return None
    if isinstance(result, list):
        items = [getattr(item, "chunk_id", None) for item in result[:10]]
        items = [item for item in items if item is not None]
        return {"count": len(result), "ids": items[:10]}
    if isinstance(result, tuple):
        return [_summarize_output(item) for item in result]
    if isinstance(result, str):
        return result[:120] + "..." if len(result) > 120 else result
    if isinstance(result, dict):
        return _summarize_input(result)
    if hasattr(result, "model_dump"):
        return _summarize_input(result.model_dump())
    return str(result)[:120]
