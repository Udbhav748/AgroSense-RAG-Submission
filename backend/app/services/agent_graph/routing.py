"""Explicit conditional-routing functions for the production chat graph.

Each function is a pure `(AgentState) -> str` returning one of the keys used
in the corresponding `add_conditional_edges(...)` mapping in `graph.py`. Kept
separate from `nodes.py` so routing decisions are reviewable/testable in
isolation from node side effects, per the PDF's "explicit edges, not one
giant method" requirement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.agent_graph.state import AgentState

MAX_REFLECTIONS = 2  # mirrors the existing agent_graph reflection cap
MAX_WEB_SEARCH_ATTEMPTS = 1  # mirrors ChatService._correct's single web-fallback escalation


def route_after_planner(state: AgentState) -> str:
    """conversational | summarize | retrieve | diagnose — falls back to
    'retrieve' for any action the planner didn't recognize, matching
    ChatService's existing heuristic-fallback behavior rather than failing
    the request outright."""
    action = state.planned_action or (
        state.plan.get("action") if isinstance(state.plan, dict) else None
    )
    if action in ("conversational", "summarize", "retrieve", "research", "diagnose"):
        return "diagnose" if action == "diagnose" else ("retrieve" if action == "research" else action)
    return "retrieve"


def route_after_grader(state: AgentState) -> str:
    """good -> generator ; weak/insufficient -> web_research, unless
    retrieval_grader_node flagged approval_required=True (Phase 5), in
    which case the request goes through human_approval_node first so the
    web-search approval gate is actually enforced against a resolved
    Approval record, not just a client-supplied boolean.

    A caller with web search disabled/unavailable (and no approval gate
    active) routes to web_research anyway — that node itself degrades to
    an empty result (see ChatService._search_web), so the state machine's
    shape doesn't change based on config, only the outcome does.
    """
    if state.retrieval_grade == "good":
        return "generator"
    if state.approval_required and state.approval_type == "web_search":
        return "approval_required"
    return "web_research"


def route_after_validation(state: AgentState) -> str:
    """valid -> finalizer ; invalid -> reflection, bounded by MAX_REFLECTIONS."""
    if not state.validation_errors:
        return "finalizer"
    if state.reflection_count_v2 >= MAX_REFLECTIONS:
        return "finalizer"  # safe_finalizer path — same node, termination_reason distinguishes it
    return "reflection"


def route_after_approval(state: AgentState) -> str:
    """approved -> resume the guarded action ; rejected/expired -> safe_finalizer.

    Never returns a route that would let a rejected/expired approval reach
    the guarded node — see human_approval_node, which never auto-approves.
    """
    if state.approval_status == "approved":
        return "resume"
    return "safe_finalizer"
