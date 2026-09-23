"""Weak-retrieval escalation node: vision QA -> local research agent ->
research agent -> plain web search, in that precedence order.

Delegates entirely to `ChatService._augment_weak_retrieval` (rag_service.py)
— the exact same method `handle_query`'s legacy path calls — so this graph
node and the legacy procedural path can never drift apart. This node
contains no escalation logic of its own; it only translates the
`RetrievalAugmentation` result into `AgentState` fields and traces the step.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from app.services.agent_graph.events import NodeTimer, emit_node_trace
from app.services.agent_graph.nodes import _node_error

if TYPE_CHECKING:
    from app.services.agent_graph.nodes import GraphContext
    from app.services.agent_graph.state import AgentState

logger = logging.getLogger(__name__)


def context_augmentation_node(state: AgentState, context: GraphContext | None = None) -> AgentState:
    """Delegates to ChatService._augment_weak_retrieval. A direct hit
    (vision QA / local research / research agent) sets final_answer and
    routes straight to the finalizer (see route_after_augmentation); no hit
    just folds in web_results and continues to the generator, exactly
    matching handle_query's original inline control flow."""
    timer = NodeTimer()
    chat_service = context.chat_service if context else None

    if chat_service is None:
        # Safe degradation: no ChatService to delegate to (e.g. a graph run
        # driven directly in a test/tool context) — pass state through
        # unchanged rather than failing the workflow, matching the same
        # pattern retrieval_node uses when context.vector_store is missing.
        new_state = state.copy_with(steps_taken=state.steps_taken + 1)
        new_state.node_timings["context_augmentation"] = 0.0
        return new_state

    # PHASE 5 FIX: a request that just resumed from human_approval_node
    # with a genuinely resolved approval_status=="approved" must be
    # treated as confirmed for this call, even though the caller's own
    # confirm_web_search flag is still false -- that flag is the
    # separate, pre-existing self-service fast path, not the only way to
    # satisfy the gate now that the graph-level approval queue is wired
    # in. Never the reverse: a non-approved status never sets this true.
    effective_confirm_web_search = state.confirm_web_search or state.approval_status == "approved"

    try:
        with timer:
            augmentation = chat_service._augment_weak_retrieval(  # noqa: SLF001
                state.query,
                state.retrieval_query or state.query,
                state.retrieved_chunks,
                state.retrieval_grade,
                state.planned_action or "retrieve",
                state.tenant_id,
                effective_confirm_web_search,
            )
    except Exception as exc:
        error_type, root_cause = _node_error(exc)
        emit_node_trace(
            trace_id=state.trace_id,
            request_id=state.request_id,
            node="context_augmentation",
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
        new_state.node_timings["context_augmentation"] = timer.latency_ms
        return new_state

    if augmentation.final_answer is not None:
        tool_call_entry: dict[str, Any] = {
            "tool_name": augmentation.tool_used,
            "success": True,
            "output_summary": {"answer_length": len(augmentation.final_answer)},
            "latency_ms": timer.latency_ms,
            "timestamp": time.time(),
        }
        emit_node_trace(
            trace_id=state.trace_id,
            request_id=state.request_id,
            node="context_augmentation",
            status="success",
            latency_ms=timer.latency_ms,
            extra={
                "direct_answer_hit": augmentation.tool_used,
                "web_result_count": len(augmentation.web_results),
            },
        )
        new_state = state.copy_with(
            final_answer=augmentation.final_answer,
            draft_answer=augmentation.final_answer,
            retrieved_chunks=augmentation.final_chunks or state.retrieved_chunks,
            web_results=augmentation.web_results,
            tool_calls=[*state.tool_calls, tool_call_entry],
            metadata={**state.metadata, "direct_answer_hit": augmentation.tool_used},
            steps_taken=state.steps_taken + 1 + augmentation.extra_steps,
        )
        new_state.node_timings["context_augmentation"] = timer.latency_ms
        return new_state

    tool_calls = state.tool_calls
    if augmentation.web_search_attempted:
        tool_calls = [
            *state.tool_calls,
            {
                "tool_name": "web_search",
                "success": True,
                "output_summary": {"result_count": len(augmentation.web_results)},
                "latency_ms": timer.latency_ms,
                "timestamp": time.time(),
            },
        ]

    emit_node_trace(
        trace_id=state.trace_id,
        request_id=state.request_id,
        node="context_augmentation",
        status="success",
        latency_ms=timer.latency_ms,
        extra={"direct_answer_hit": None, "web_result_count": len(augmentation.web_results)},
    )
    new_state = state.copy_with(
        web_results=augmentation.web_results,
        tool_calls=tool_calls,
        metadata={**state.metadata, "web_search_attempted": augmentation.web_search_attempted},
        steps_taken=state.steps_taken + 1,
    )
    new_state.node_timings["context_augmentation"] = timer.latency_ms
    return new_state


def route_after_augmentation(state: AgentState) -> str:
    """finalizer when a direct answer was found (vision QA / local
    research / research agent); generator otherwise."""
    if state.metadata.get("direct_answer_hit"):
        return "finalizer"
    return "generator"
