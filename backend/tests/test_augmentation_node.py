"""Phase 1: context_augmentation_node — delegates to
ChatService._augment_weak_retrieval, translates the result into AgentState.
"""

from __future__ import annotations

from app.models.document import RetrievedChunk
from app.services.agent_graph.augmentation_node import (
    context_augmentation_node,
    route_after_augmentation,
)
from app.services.agent_graph.nodes import GraphContext
from app.services.agent_graph.state import AgentState
from app.services.rag_service import RetrievalAugmentation


def make_chunk(chunk_id: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, document_id="d1", text="content", score=0.5)


class FakeChatService:
    def __init__(self, result: RetrievalAugmentation | None = None, raise_exc: Exception | None = None):
        self._result = result or RetrievalAugmentation()
        self._raise_exc = raise_exc
        self.calls = 0

    def _augment_weak_retrieval(self, *args, **kwargs):
        self.calls += 1
        if self._raise_exc:
            raise self._raise_exc
        return self._result


def base_state() -> AgentState:
    return AgentState(query="q", retrieval_grade="weak", planned_action="retrieve")


def test_vision_qa_hit_routes_to_finalizer():
    fake = FakeChatService(RetrievalAugmentation(final_answer="vision answer", tool_used="vision_qa"))
    ctx = GraphContext(chat_service=fake)
    result = context_augmentation_node(base_state(), ctx)
    assert result.final_answer == "vision answer"
    assert route_after_augmentation(result) == "finalizer"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["tool_name"] == "vision_qa"


def test_local_research_hit_replaces_chunks_and_routes_to_finalizer():
    new_chunks = [make_chunk("local1")]
    fake = FakeChatService(
        RetrievalAugmentation(
            final_answer="local research answer", tool_used="local_research", final_chunks=new_chunks
        )
    )
    ctx = GraphContext(chat_service=fake)
    result = context_augmentation_node(base_state(), ctx)
    assert result.final_answer == "local research answer"
    assert result.retrieved_chunks == new_chunks
    assert route_after_augmentation(result) == "finalizer"


def test_web_search_only_routes_to_generator():
    fake = FakeChatService(
        RetrievalAugmentation(web_search_attempted=True, web_results=["r1", "r2"])
    )
    ctx = GraphContext(chat_service=fake)
    result = context_augmentation_node(base_state(), ctx)
    assert result.web_results == ["r1", "r2"]
    assert route_after_augmentation(result) == "generator"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["tool_name"] == "web_search"


def test_no_hit_no_web_search_routes_to_generator_with_empty_results():
    fake = FakeChatService(RetrievalAugmentation())
    ctx = GraphContext(chat_service=fake)
    result = context_augmentation_node(base_state(), ctx)
    assert result.web_results == []
    assert result.tool_calls == []
    assert route_after_augmentation(result) == "generator"


def test_no_chat_service_degrades_safely():
    ctx = GraphContext(chat_service=None)
    state = base_state()
    result = context_augmentation_node(state, ctx)
    assert result.steps_taken == state.steps_taken + 1
    assert route_after_augmentation(result) == "generator"

    # Also covers context=None entirely.
    result2 = context_augmentation_node(state, None)
    assert result2.steps_taken == state.steps_taken + 1


def test_exception_is_caught_and_recorded_without_raising():
    fake = FakeChatService(raise_exc=RuntimeError("boom"))
    ctx = GraphContext(chat_service=fake)
    result = context_augmentation_node(base_state(), ctx)
    assert result.error_type is not None
    assert result.error_message is not None
    # Node returns a valid state, does not propagate the exception, and
    # does not claim a direct answer hit.
    assert route_after_augmentation(result) == "generator"
