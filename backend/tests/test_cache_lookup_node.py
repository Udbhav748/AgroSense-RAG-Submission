"""Phase 1: cache_lookup_node — hit short-circuits to END, miss passes
through to retrieval, and history present skips the lookup entirely."""

from __future__ import annotations

from app.models.schemas import ChatResponse
from app.services.agent_graph.cache_node import cache_lookup_node, route_after_cache_lookup
from app.services.agent_graph.nodes import GraphContext
from app.services.agent_graph.state import AgentState


class FakeChatService:
    def __init__(self, cached_response: ChatResponse | None):
        self._cached_response = cached_response
        self.calls: list[dict] = []

    def _get_cached_response(self, query, crop=None, disease=None, tenant_id=None, document_ids=None):
        self.calls.append(
            {
                "query": query,
                "crop": crop,
                "disease": disease,
                "tenant_id": tenant_id,
                "document_ids": document_ids,
            }
        )
        return self._cached_response


def make_response(answer="cached answer") -> ChatResponse:
    return ChatResponse(
        answer=answer,
        retrieved_chunks=[],
        sources=[],
        processing_time=0.01,
        tool_used="retrieval",
        steps_taken=3,
        session_id="old-session",
        follow_up_questions=[],
        metadata={},
    )


def test_cache_miss_passes_through_to_retrieval():
    fake = FakeChatService(cached_response=None)
    ctx = GraphContext(chat_service=fake)
    state = AgentState(query="what is the scope?")

    result = cache_lookup_node(state, ctx)

    assert result.metadata["cache_hit"] is False
    assert result.final_response is None
    assert len(fake.calls) == 1
    assert route_after_cache_lookup(result) == "retrieval"


def test_cache_hit_sets_final_response_and_routes_to_end():
    cached = make_response()
    fake = FakeChatService(cached_response=cached)
    ctx = GraphContext(chat_service=fake)
    state = AgentState(query="what is the scope?", session_id="new-session")

    result = cache_lookup_node(state, ctx)

    assert result.metadata["cache_hit"] is True
    assert result.workflow_status == "completed"
    assert result.termination_reason == "success"
    assert result.final_response is not None
    assert result.final_response.answer == "cached answer"
    # session_id is overridden onto the cached response
    assert result.final_response.session_id == "new-session"
    assert result.final_response.metadata["cached"] is True
    assert result.final_answer == "cached answer"
    assert route_after_cache_lookup(result) == "__end__"


def test_history_present_skips_cache_lookup_entirely():
    fake = FakeChatService(cached_response=make_response())
    ctx = GraphContext(chat_service=fake)
    state = AgentState(query="follow up question", history=[{"role": "user", "content": "hi"}])

    result = cache_lookup_node(state, ctx)

    assert len(fake.calls) == 0
    assert result.metadata["cache_hit"] is False
    assert route_after_cache_lookup(result) == "retrieval"


def test_no_chat_service_is_a_safe_miss():
    state = AgentState(query="q")
    result = cache_lookup_node(state, None)
    assert result.metadata["cache_hit"] is False
    assert route_after_cache_lookup(result) == "retrieval"


def test_node_timing_recorded():
    fake = FakeChatService(cached_response=None)
    ctx = GraphContext(chat_service=fake)
    state = AgentState(query="q")
    result = cache_lookup_node(state, ctx)
    assert "cache_lookup" in result.node_timings
