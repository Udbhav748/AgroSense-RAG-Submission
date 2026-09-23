"""Phase 1: node/workflow metrics recorded by agent_graph/events.py and
agent_graph/engine.py — the PDF's agent_workflow_*/agent_node_*/agent_
reflections_total/agent_loop_limit_hits_total/agent_approval_* counters and
the node latency histogram."""

from __future__ import annotations

import pytest

from app.core.metrics import get_metrics, reset_metrics
from app.services.agent_graph.graph import build_chat_graph
from app.services.agent_graph.human_approval import human_approval_node
from app.services.agent_graph.nodes import GraphContext
from app.services.agent_graph.state import AgentState
from app.services.approval_service import get_approval_store
from app.services.rag_service import PlanDecision, RetrievalAugmentation


@pytest.fixture(autouse=True)
def _isolate_metrics():
    reset_metrics()
    yield
    reset_metrics()


def _counter(name: str, labels: dict | None = None) -> float:
    metrics = get_metrics()
    with metrics._lock:  # test-only introspection, mirrors test_metrics.py's own approach
        from app.core.metrics import _label_key

        return metrics._counters.get((name, _label_key(labels)), 0.0)


class FakeChatService:
    def __init__(self, ungrounded=False, corrected_answer=None):
        self._ungrounded = ungrounded
        self._corrected_answer = corrected_answer

    def _plan(self, query, history=None):
        return PlanDecision(action="retrieve")

    def _route(self, query, history=None):
        return PlanDecision(action="retrieve")

    def _grade_retrieval(self, query, chunks):
        return "good"

    def _generate(self, *a, **k):
        return "answer [1]"

    def _generate_structured(self, *a, **k):
        answer = self._generate(*a, **k)
        return answer, {"answer": answer, "sources": []}

    def _is_ungrounded(self, answer, chunks, web_results):
        return self._ungrounded

    def _augment_weak_retrieval(self, *a, **k):
        return RetrievalAugmentation()

    def _correct(self, query, chunks, answer, *a, **k):
        final = self._corrected_answer if self._corrected_answer is not None else answer
        return final, 2, 1, [], False

    def _get_cached_response(self, *a, **k):
        return None

    def _cache_response(self, *a, **k):
        pass

    def _maybe_ask_clarifying_question(self, query, answer, grade):
        return answer, False

    def _suggest_follow_ups(self, query, answer):
        return []

    def _respond(self, *, answer, retrieved_chunks, query, query_type, tool_used, steps_taken, start, **kw):
        from app.models.schemas import ChatResponse

        return ChatResponse(
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            sources=[],
            processing_time=0.0,
            tool_used=tool_used,
            steps_taken=steps_taken,
            answer_source="documents",
            session_id="",
        )


class FakeVectorStore:
    def search(self, *a, **k):
        return []

    def search_bm25(self, *a, **k):
        return []


@pytest.mark.asyncio
async def test_node_execution_metrics_recorded_per_run():
    graph = build_chat_graph()
    ctx = GraphContext(chat_service=FakeChatService(), vector_store=FakeVectorStore())
    await graph.run(AgentState(query="q"), ctx)

    assert _counter("agent_node_executions_total", {"node": "validate_request", "status": "success"}) == 1
    assert _counter("agent_node_executions_total", {"node": "planner", "status": "success"}) == 1
    assert _counter("agent_node_executions_total", {"node": "retrieval", "status": "success"}) == 1
    assert _counter("agent_node_executions_total", {"node": "finalizer", "status": "success"}) == 1


@pytest.mark.asyncio
async def test_workflow_completion_metrics_recorded():
    graph = build_chat_graph()
    ctx = GraphContext(chat_service=FakeChatService(), vector_store=FakeVectorStore())
    await graph.run(AgentState(query="q"), ctx)

    assert _counter("agent_workflow_started_total") == 1
    assert _counter("agent_workflow_completed_total") == 1
    assert _counter("agent_workflow_failed_total") == 0

    metrics = get_metrics()
    hist = metrics._histograms.get(("agent_workflow_duration_seconds", ()))
    assert hist is not None and hist.count == 1

    steps_hist = metrics._counters.get(("agent_steps_total", ()))
    assert steps_hist is not None and steps_hist > 0


@pytest.mark.asyncio
async def test_reflection_metric_recorded_only_when_answer_actually_changes():
    graph = build_chat_graph()

    # Case 1: _correct doesn't change the answer -> no reflection counted.
    ctx_unchanged = GraphContext(chat_service=FakeChatService(ungrounded=True), vector_store=FakeVectorStore())
    await graph.run(AgentState(query="q"), ctx_unchanged)
    assert _counter("agent_reflections_total") == 0

    # Case 2: _correct produces a different answer -> one reflection counted.
    ctx_changed = GraphContext(
        chat_service=FakeChatService(ungrounded=True, corrected_answer="a different answer"),
        vector_store=FakeVectorStore(),
    )
    await graph.run(AgentState(query="q"), ctx_changed)
    assert _counter("agent_reflections_total") == 1


def test_approval_metrics_recorded_for_required_approved_rejected():
    store = get_approval_store()

    # required
    state = AgentState(query="q", approval_required=True, approval_type="web_search")
    human_approval_node(state)
    assert _counter("agent_approval_required_total", {"action": "web_search"}) == 1

    # approved
    approval = store.register(action="web_search", payload={"query": "x"})
    store.resolve(approval.approval_id, approved=True, resolved_by="op")
    state2 = AgentState(
        query="q",
        approval_required=True,
        approval_type="web_search",
        approval_payload_reference=approval.approval_id,
    )
    human_approval_node(state2)
    assert _counter("agent_approval_approved_total", {"action": "web_search"}) == 1

    # rejected
    approval2 = store.register(action="document_delete", payload={"document_id": "d1"})
    store.resolve(approval2.approval_id, approved=False, resolved_by="op")
    state3 = AgentState(
        query="q",
        approval_required=True,
        approval_type="document_delete",
        approval_payload_reference=approval2.approval_id,
    )
    human_approval_node(state3)
    assert _counter("agent_approval_rejected_total", {"action": "document_delete"}) == 1


def test_approval_not_required_records_nothing():
    human_approval_node(AgentState(query="q", approval_required=False))
    assert _counter("agent_approval_required_total", {"action": "unknown"}) == 0
