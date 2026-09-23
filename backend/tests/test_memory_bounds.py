"""Tests for the "no unnecessary data retained" half of Memory: the
conversation-history cap ChatService.handle_query applies before
generation (rag_service._MAX_HISTORY_TURNS), and InMemorySessionStore's
own bounds (max_turns_per_session, max_sessions LRU eviction).

The "relevant context is remembered" half already has eval coverage
(run_eval.py's memory_recall_rate); this file covers the other half,
which had none.
"""

import app.services.rag_service as rag_service_module
from app.services.rag_service import _MAX_HISTORY_TURNS, ChatService
from app.services.session_store import InMemorySessionStore
from tests.test_rag_service import FakeLLMClient, FakeVectorStore, make_chunk


class _CountedFakeVectorStore(FakeVectorStore):
    """FakeVectorStore plus total_vectors() — handle_query's response-cache
    key needs it (_make_cache_key), unlike _plan/_grade_retrieval/_correct,
    which is all the shared FakeVectorStore was built for."""

    def total_vectors(self):
        return 0


def _make_service(llm_client):
    return ChatService(_CountedFakeVectorStore(), llm_client)


class TestHandleQueryTruncatesHistory:
    def test_only_the_most_recent_max_history_turns_reach_the_prompt(self, monkeypatch):
        llm_client = FakeLLMClient(response="grounded answer using context")
        service = _make_service(llm_client)
        monkeypatch.setattr(
            rag_service_module,
            "retrieve",
            lambda query, vector_store, top_k=None, min_score=None, tenant_id=None, document_ids=None, image_vector_store=None: [make_chunk()],
        )
        # More turns than _MAX_HISTORY_TURNS — each with a unique marker so
        # presence/absence in the prompt is unambiguous.
        turn_count = _MAX_HISTORY_TURNS + 4
        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn-{i}-marker"}
            for i in range(turn_count)
        ]

        service.handle_query("what does the document say about scope?", history=history)

        assert len(llm_client.calls) == 1  # grounded on the first try, no reflection retry
        prompt = llm_client.calls[0]
        kept = range(turn_count - _MAX_HISTORY_TURNS, turn_count)
        dropped = range(0, turn_count - _MAX_HISTORY_TURNS)
        for i in kept:
            assert f"turn-{i}-marker" in prompt
        for i in dropped:
            assert f"turn-{i}-marker" not in prompt

    def test_history_within_the_cap_is_kept_in_full(self, monkeypatch):
        llm_client = FakeLLMClient(response="grounded answer using context")
        service = _make_service(llm_client)
        monkeypatch.setattr(
            rag_service_module,
            "retrieve",
            lambda query, vector_store, top_k=None, min_score=None, tenant_id=None, document_ids=None, image_vector_store=None: [make_chunk()],
        )
        history = [{"role": "user", "content": "turn-0-marker"}, {"role": "assistant", "content": "turn-1-marker"}]

        service.handle_query("what does the document say about scope?", history=history)

        prompt = llm_client.calls[0]
        assert "turn-0-marker" in prompt
        assert "turn-1-marker" in prompt


class TestInMemorySessionStoreBounds:
    def test_history_trimmed_to_max_turns_per_session(self):
        store = InMemorySessionStore(max_turns_per_session=3)
        session_id = store.create_session()

        for i in range(5):
            store.append_turn(session_id, "user", f"turn-{i}")

        history = store.get_history(session_id)
        # Only the most recent 3 turns survive, oldest-first.
        assert [turn["content"] for turn in history] == ["turn-2", "turn-3", "turn-4"]

    def test_turns_within_the_cap_are_kept_in_full(self):
        store = InMemorySessionStore(max_turns_per_session=5)
        session_id = store.create_session()

        store.append_turn(session_id, "user", "turn-0")
        store.append_turn(session_id, "assistant", "turn-1")

        history = store.get_history(session_id)
        assert [turn["content"] for turn in history] == ["turn-0", "turn-1"]

    def test_lru_eviction_at_max_sessions(self):
        store = InMemorySessionStore(max_sessions=2)
        session_1 = store.create_session()
        session_2 = store.create_session()

        # Force session_1 to look more recently accessed than session_2,
        # deterministically — real time.time() calls made microseconds
        # apart aren't a reliable ordering signal to build a test on (and
        # Session's last_accessed uses field(default_factory=time.time),
        # which captures the function at class-definition time, so
        # monkeypatching time.time afterwards wouldn't reach it anyway).
        store._sessions[session_1].last_accessed = store._sessions[session_2].last_accessed + 1000

        session_3 = store.create_session()  # capacity reached — evicts the LRU session (session_2)

        assert store.session_exists(session_1)
        assert not store.session_exists(session_2)
        assert store.session_exists(session_3)
        assert store.get_session_count() == 2

    def test_under_capacity_no_eviction(self):
        store = InMemorySessionStore(max_sessions=5)
        session_1 = store.create_session()
        session_2 = store.create_session()

        assert store.session_exists(session_1)
        assert store.session_exists(session_2)
        assert store.get_session_count() == 2
