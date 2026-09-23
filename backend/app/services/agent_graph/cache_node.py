"""Cache-lookup node: a thin wrapper around ChatService._get_cached_response,
mirroring handle_query's exact cache-check semantics (see rag_service.py's
handle_query, the block right after `plan = self._route(...)` that checks
`if not history and not recent_history:` before calling
`_get_cached_response`). A hit short-circuits the workflow straight to END
with a fully-formed ChatResponse; a miss is a no-op pass-through into
retrieval — caching is a best-effort optimization here exactly as it is in
the legacy path, never a hard dependency the workflow can fail on.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.models.schemas import ChatResponse
from app.services.agent_graph.engine import END
from app.services.agent_graph.events import NodeTimer, emit_node_trace

if TYPE_CHECKING:
    from app.services.agent_graph.nodes import GraphContext
    from app.services.agent_graph.state import AgentState

logger = logging.getLogger(__name__)


def cache_lookup_node(state: AgentState, context: GraphContext | None = None) -> AgentState:
    """Look up a cached response for this query when there's no
    conversation history (handle_query's `if not history and not
    recent_history:` gate — this graph has no separate `recent_history`
    field, so `state.history` alone stands in for both checks).

    Never raises: any exception from the underlying cache lookup is treated
    as a miss, matching how `_get_cached_response`/`cache_service` already
    degrade elsewhere in this codebase.
    """
    timer = NodeTimer()
    chat_service = context.chat_service if context else None
    cached: ChatResponse | None = None

    with timer:
        if chat_service is not None and not state.history:
            plan_crop = state.plan.get("crop") if isinstance(state.plan, dict) else None
            plan_disease = None  # PlanDecision has no `disease` field today;
            # getattr(plan, "disease", None) in handle_query is always None.
            try:
                cached = chat_service._get_cached_response(  # noqa: SLF001
                    query=state.query,
                    crop=plan_crop,
                    disease=plan_disease,
                    tenant_id=state.tenant_id,
                    document_ids=state.document_ids,
                )
            except Exception as exc:
                logger.warning(
                    "cache_lookup_failed", extra={"extra_fields": {"error": str(exc)}}
                )
                cached = None

    cache_hit = cached is not None
    emit_node_trace(
        trace_id=state.trace_id,
        request_id=state.request_id,
        node="cache_lookup",
        status="success",
        latency_ms=timer.latency_ms,
        extra={"cache_hit": cache_hit},
    )

    if not cache_hit:
        new_state = state.copy_with(
            steps_taken=state.steps_taken + 1,
            metadata={**state.metadata, "cache_hit": False},
        )
        new_state.node_timings["cache_lookup"] = timer.latency_ms
        return new_state

    # Mirrors handle_query's cache-hit mutation exactly: stamp the current
    # session_id onto the cached response and flag it as served-from-cache.
    cached_dict = cached.model_dump()
    cached_dict["session_id"] = state.session_id or cached_dict.get("session_id", "")
    if "metadata" not in cached_dict or not isinstance(cached_dict["metadata"], dict):
        cached_dict["metadata"] = {}
    cached_dict["metadata"]["cached"] = True
    final_response = ChatResponse.model_validate(cached_dict)

    new_state = state.copy_with(
        final_response=final_response,
        final_answer=final_response.answer,
        workflow_status="completed",
        termination_reason="success",
        metadata={**state.metadata, "cache_hit": True},
        steps_taken=state.steps_taken + 1,
    )
    new_state.node_timings["cache_lookup"] = timer.latency_ms
    return new_state


def route_after_cache_lookup(state: AgentState) -> str:
    """END on a cache hit (the workflow is already finished); otherwise
    continue into retrieval exactly as handle_query falls through to its
    retrieval step on a miss."""
    if state.metadata.get("cache_hit"):
        return END
    return "retrieval"


__all__ = ["cache_lookup_node", "route_after_cache_lookup"]
