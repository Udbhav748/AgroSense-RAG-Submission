"""Module 10 gap-closure (P6), TASK 1 finding: a cache-hit response
(cache_lookup_node) short-circuits straight to END and never reaches
finalizer_node -- which is the ONLY place ChatService._respond() (the
sole source of the "chat_query_handled" log line monitoring/
log_aggregate.py's aggregate() counts toward `requests`) is called on
the plain /chat path.

This means: log-based aggregate() request/error/latency counts
undercount total traffic whenever the response cache serves an answer.
This was discovered empirically while building this pass's controlled-
traffic observability report (see run_observability_final_eval.py's
_distinct_fake_embedding docstring) and is documented here as a
regression-pinning test, not fixed -- log_aggregate.py is existing,
working instrumentation this pass is told not to rewrite unnecessarily,
and the live GET /metrics Prometheus registry (app/core/metrics.py, via
its own request-latency middleware) is unaffected by this gap since it
instruments at the HTTP layer, not via this specific log line -- see
docs/MODULE10_RESULTS.md's Observability section for the full
disclosure and docs/CHECKLIST.md's Alerts/Monitoring rows.
"""

from __future__ import annotations

from app.models.schemas import ChatResponse
from app.services.agent_graph.cache_node import route_after_cache_lookup
from app.services.agent_graph.engine import END
from app.services.agent_graph.graph import build_chat_graph
from app.services.agent_graph.state import AgentState


def test_cache_hit_routes_straight_to_end_bypassing_finalizer():
    """Structural proof: on a cache hit, route_after_cache_lookup sends
    the workflow to END, never to 'retrieval' -> ... -> 'finalizer' --
    finalizer_node is the only caller of ChatService._respond() (the
    "chat_query_handled" log source) on this path."""
    state = AgentState(request_id="r1", trace_id="t1", query="q", metadata={"cache_hit": True})
    assert route_after_cache_lookup(state) == END


def test_cache_miss_routes_to_retrieval_which_eventually_reaches_finalizer():
    state = AgentState(request_id="r1", trace_id="t1", query="q", metadata={"cache_hit": False})
    assert route_after_cache_lookup(state) == "retrieval"


def test_cache_lookup_conditional_edge_never_maps_directly_to_finalizer():
    """Confirms the graph topology itself: 'cache_lookup' is a real node
    in the compiled graph, and its conditional-edge mapping only ever
    resolves to 'retrieval' (a miss) or the literal END sentinel
    (a hit, returned directly by route_after_cache_lookup, not through
    this mapping) -- 'finalizer' is never a direct destination from
    'cache_lookup', so a hit's only route to a final response is via
    END, never via finalizer_node's logging."""
    graph = build_chat_graph()
    assert "cache_lookup" in graph.nodes
    assert "finalizer" in graph.nodes

    route_fn, mapping = graph.conditional_edges["cache_lookup"]
    assert route_fn is route_after_cache_lookup
    assert "finalizer" not in (mapping or {}).values()
