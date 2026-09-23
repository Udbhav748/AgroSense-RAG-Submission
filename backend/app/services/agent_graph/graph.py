"""Production chat workflow topology.

`build_chat_graph()` wires the explicit nodes in `nodes.py` (validate_request
through finalizer) into the topology approved for Phase 1:

    validate_request
      -> planner
      -> [conversational|summarize|retrieve|diagnose]  (route_after_planner)

    retrieve -> retrieval -> retrieval_grader
      -> [generator]                 (grade == good)
      -> [web_research -> generator] (grade == weak/insufficient)

    generator -> reflection -> output_validation -> finalizer

    reflection delegates wholesale to ChatService._correct (see
    reflection_node's docstring for why this is one node call rather than a
    generic "invalid -> loop back to generator" edge: _correct's own
    internal escalation — regenerate, then regenerate again with a web-
    search fallback if still ungrounded — doesn't decompose into a generic
    repeat-same-instruction loop without either reimplementing that
    escalation here or losing it). output_validation runs after reflection
    purely as post-hoc bookkeeping/tracing of the final groundedness — it
    never loops back, since _correct already exhausted its own bounded
    correction budget (_MAX_LLM_CALLS). The generic reflection loop
    (route_after_validation's "reflection" branch, MAX_REFLECTIONS in
    routing.py) is kept defined and unit-tested for future use but is not
    wired into this production topology.

    conversational / summarize -> finalizer directly (their nodes already
    populate draft_answer; diagnose is not yet wired to a vision node in
    this commit — see Remaining gaps in the implementation report; it
    currently falls through to retrieval like `research` did before this
    phase, preserving today's behavior rather than half-wiring a new path).

This graph now backs ChatService.handle_query's non-executor path (see
handle_query's call into `run_chat_graph`/`build_chat_graph` in
rag_service.py). AgentExecutor's ReAct path (Settings.agent_executor_enabled)
remains a separate, untouched branch, exactly as before.
"""

from __future__ import annotations

from app.services.agent_graph.augmentation_node import (
    context_augmentation_node,
    route_after_augmentation,
)
from app.services.agent_graph.cache_node import cache_lookup_node, route_after_cache_lookup
from app.services.agent_graph.engine import END, START, CompiledGraph, StateGraph
from app.services.agent_graph.human_approval import human_approval_node
from app.services.agent_graph.nodes import (
    finalizer_node,
    generator_node,
    output_validation_node,
    planner_node_v2,
    reflection_node,
    retrieval_grader_node,
    retrieval_node,
    summarizer_node,
    validate_request_node,
)
from app.services.agent_graph.routing import (
    route_after_approval,
    route_after_grader,
    route_after_planner,
)
from app.services.prompt_builder import FALLBACK_REPLY
from app.services.rag_service import _match_conversational_reply


def _conversational_node(state, context=None):  # noqa: ANN001, ANN201
    """Delegates to ChatService's existing `_match_conversational_reply`
    (the same canned-reply lookup `_plan` used to decide this was a
    conversational turn in the first place) — not a second copy of the
    canned-reply table."""
    reply = _match_conversational_reply(state.query) or FALLBACK_REPLY
    return state.copy_with(draft_answer=reply, final_answer=reply, steps_taken=state.steps_taken + 1)


def build_chat_graph(max_steps: int = 16) -> CompiledGraph:
    """Compile the production chat graph. The longest real path is
    validate(1) -> planner(2) -> cache_lookup(3) -> retrieval(4) ->
    retrieval_grader(5) -> context_augmentation(6) -> generator(7) ->
    reflection(8) -> output_validation(9) -> finalizer(10) — max_steps=16
    leaves comfortable headroom while still capping runaway execution,
    mirroring the existing agent_graph engine's default guardrail pattern.
    """
    graph = StateGraph()

    graph.add_node("validate_request", validate_request_node)
    graph.add_node("planner", planner_node_v2)
    graph.add_node("conversational", _conversational_node)
    graph.add_node("summarize", summarizer_node)
    graph.add_node("cache_lookup", cache_lookup_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("retrieval_grader", retrieval_grader_node)
    graph.add_node("context_augmentation", context_augmentation_node)
    graph.add_node("generator", generator_node)
    graph.add_node("output_validation", output_validation_node)
    graph.add_node("reflection", reflection_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("finalizer", finalizer_node)

    graph.set_entry_point("validate_request")
    graph.add_edge("validate_request", "planner")

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "conversational": "conversational",
            "summarize": "summarize",
            "retrieve": "cache_lookup",
            "diagnose": "cache_lookup",  # vision_node lands in a follow-up commit; see docstring
        },
    )

    graph.add_edge("conversational", "finalizer")
    graph.add_edge("summarize", "finalizer")

    # cache_lookup mirrors handle_query's "if not history: check cache
    # first" — a hit ends the workflow immediately with the cached
    # ChatResponse (no finalizer re-processing, matching handle_query
    # returning the cached response directly); a miss falls through to
    # retrieval exactly as handle_query does.
    graph.add_conditional_edges(
        "cache_lookup",
        route_after_cache_lookup,
        {"retrieval": "retrieval"},  # END is returned as the literal END sentinel, not a mapping key
    )

    graph.add_edge("retrieval", "retrieval_grader")
    graph.add_conditional_edges(
        "retrieval_grader",
        route_after_grader,
        {
            "generator": "generator",
            "web_research": "context_augmentation",
            "approval_required": "human_approval",
        },
    )
    # context_augmentation replicates handle_query's full weak-retrieval
    # escalation (vision QA -> local research -> research agent -> web
    # search) via ChatService._augment_weak_retrieval — a direct hit
    # (vision/local-research/research-agent) skips straight to the
    # finalizer with its already-complete answer, exactly like
    # handle_query's early `return self._respond(...)`; anything else
    # (including a plain web-search fallback) continues to generation.
    graph.add_conditional_edges(
        "context_augmentation",
        route_after_augmentation,
        {"finalizer": "finalizer", "generator": "generator"},
    )

    # reflection (= ChatService._correct, a single call) always runs right
    # after the initial generation, exactly like handle_query's
    # unconditional `self._correct(...)` call — see reflection_node's
    # docstring for why this isn't a generic loop-back edge.
    graph.add_edge("generator", "reflection")
    graph.add_edge("reflection", "output_validation")
    graph.add_edge("output_validation", "finalizer")

    # PHASE 5 FIX: human_approval is now genuinely reachable — retrieval_grader_node
    # flags approval_required=True for the web-search escalation when
    # Settings.web_search_requires_approval is on and the caller hasn't
    # already satisfied the gate (confirm_web_search=true or an
    # already-approved reference), and route_after_grader sends it here.
    # "resume" routes to context_augmentation (the actual guarded action),
    # not straight to generator — an approved request must still perform
    # the web search it was approved for, not skip past it. A
    # pending/rejected/expired approval routes to "generator" (NOT straight
    # to finalizer): web search specifically is the guarded action, not
    # generation itself — the request should still get the best answer the
    # LLM can produce from whatever chunks retrieval already, legitimately
    # found, exactly as it would have before this approval gate existed.
    # human_approval_node's own never-auto-approve guarantee is preserved:
    # only "resume" (approval_status == "approved") ever reaches
    # context_augmentation, so web search itself is never performed
    # without a genuine approval.
    graph.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {"resume": "context_augmentation", "safe_finalizer": "generator"},
    )

    return graph.compile(max_steps=max_steps)


__all__ = ["END", "START", "build_chat_graph"]
