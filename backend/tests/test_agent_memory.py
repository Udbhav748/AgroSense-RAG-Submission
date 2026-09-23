"""Tests for agent memory (services/agent_memory.py) and its integration
into ChatService (rag_service.py).

Store tests cover the two halves — turns (add/trim/retrieve) and facts
(upsert/retrieve/eviction) — plus the LRU bounds that keep memory from
growing without limit. Integration tests cover the two wiring points:
_inject_memory (remembered facts reaching the router/generation prompts as
a synthetic system turn, no-op when disabled/empty) and _remember
(recording turns and, with fact extraction enabled, storing durable facts
extracted from the answer — best-effort, never a new failure mode).
"""

import app.services.rag_service as rag_service_module
from app.core.config import settings
from app.services.agent_memory import AgentMemory
from app.services.rag_service import ChatService
from tests.test_rag_service import FakeLLMClient, FakeVectorStore, make_chunk


class _CountedFakeVectorStore(FakeVectorStore):
    """FakeVectorStore plus total_vectors() — handle_query's response-cache
    key needs it (_make_cache_key)."""

    def total_vectors(self):
        return 0


def _make_service(llm_client=None, agent_memory=None):
    return ChatService(
        _CountedFakeVectorStore(), llm_client or FakeLLMClient(), agent_memory=agent_memory
    )


# --------------------------------------------------------------------------
# AgentMemory store
# --------------------------------------------------------------------------


class TestMemoryStoreTurns:
    def test_add_turn_and_get_recent_turns(self):
        memory = AgentMemory()
        memory.add_turn("s1", "user", "q1")
        memory.add_turn("s1", "assistant", "a1")
        turns = memory.get_recent_turns("s1")
        assert turns == [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]

    def test_turns_trimmed_to_max_turns_per_session(self):
        memory = AgentMemory(max_turns_per_session=3)
        for i in range(5):
            memory.add_turn("s1", "user", f"turn-{i}")
        turns = memory.get_recent_turns("s1")
        assert [t["content"] for t in turns] == ["turn-2", "turn-3", "turn-4"]

    def test_unknown_session_reads_are_empty_not_errors(self):
        memory = AgentMemory()
        assert memory.get_recent_turns("nope") == []
        assert memory.get_all_facts("nope") == []
        assert memory.get_fact("nope", "key") is None


class TestMemoryStoreFacts:
    def test_upsert_fact_stores_and_normalizes_key(self):
        memory = AgentMemory()
        memory.upsert_fact("s1", "Deadline", "March 1", confidence="high")
        assert memory.get_fact("s1", "deadline") == {
            "key": "deadline",
            "value": "March 1",
            "confidence": "high",
            "source_chunk_ids": [],
        }

    def test_upsert_same_key_updates_in_place(self):
        memory = AgentMemory()
        memory.upsert_fact("s1", "team", "3 people", confidence="low")
        memory.upsert_fact("s1", "team", "5 people", confidence="high", source_chunk_ids=["c1"])
        facts = memory.get_all_facts("s1")
        assert len(facts) == 1  # updated, not duplicated
        assert facts[0]["value"] == "5 people"
        assert facts[0]["confidence"] == "high"
        assert facts[0]["source_chunk_ids"] == ["c1"]

    def test_empty_key_or_value_is_ignored(self):
        memory = AgentMemory()
        memory.upsert_fact("s1", "", "value")
        memory.upsert_fact("s1", "key", "   ")
        assert memory.get_all_facts("s1") == []

    def test_fact_lru_eviction_beyond_max_facts(self):
        memory = AgentMemory(max_facts_per_session=2)
        memory.upsert_fact("s1", "a", "1")
        memory.upsert_fact("s1", "b", "2")
        # New fact forces eviction of the least-recently-updated fact ("a").
        memory.upsert_fact("s1", "c", "3")
        assert memory.get_fact("s1", "a") is None
        assert memory.get_fact("s1", "b") is not None
        assert memory.get_fact("s1", "c") is not None


class TestMemoryStoreBounds:
    def test_session_lru_eviction_beyond_max_sessions(self):
        memory = AgentMemory(max_sessions=2)
        memory.add_turn("s1", "user", "1")
        memory.add_turn("s2", "user", "2")
        # Deterministic ordering, same trick as test_memory_bounds.py.
        memory._sessions["s1"].last_accessed = memory._sessions["s2"].last_accessed + 1000
        memory.add_turn("s3", "user", "3")
        assert memory.session_exists("s1")
        assert not memory.session_exists("s2")
        assert memory.session_exists("s3")
        assert memory.get_session_count() == 2

    def test_delete_session_forgets_everything(self):
        memory = AgentMemory()
        memory.add_turn("s1", "user", "q")
        memory.upsert_fact("s1", "deadline", "March 1")
        assert memory.delete_session("s1") is True
        assert not memory.session_exists("s1")
        assert memory.delete_session("s1") is False  # already gone

    def test_construct_rejects_nonpositive_bounds(self):
        try:
            AgentMemory(max_sessions=0)
            assert False, "expected ValueError"
        except ValueError:
            pass


class TestMemoryStoreBuildContext:
    def test_build_context_empty_when_no_facts(self):
        assert AgentMemory().build_context("s1") == ""

    def test_build_context_formats_facts(self):
        memory = AgentMemory()
        memory.upsert_fact("s1", "deadline", "March 1", confidence="high")
        memory.upsert_fact("s1", "team", "5 people", confidence="medium")
        context = memory.build_context("s1")
        assert "Remembered from earlier in this conversation" in context
        assert "- deadline: March 1" in context
        assert "- team: 5 people" in context


# --------------------------------------------------------------------------
# ChatService integration
# --------------------------------------------------------------------------


class TestInjectMemory:
    def test_disabled_leaves_history_unchanged(self, monkeypatch):
        monkeypatch.setattr(settings, "agent_memory_enabled", False)
        service = _make_service(agent_memory=AgentMemory())
        service._agent_memory.upsert_fact("s1", "deadline", "March 1")
        history = [{"role": "user", "content": "q"}]
        assert service._inject_memory(history, "s1") is history

    def test_no_agent_memory_leaves_history_unchanged(self, monkeypatch):
        monkeypatch.setattr(settings, "agent_memory_enabled", True)
        service = _make_service(agent_memory=None)
        history = [{"role": "user", "content": "q"}]
        assert service._inject_memory(history, "s1") is history

    def test_no_facts_leaves_history_unchanged(self, monkeypatch):
        monkeypatch.setattr(settings, "agent_memory_enabled", True)
        service = _make_service(agent_memory=AgentMemory())
        history = [{"role": "user", "content": "q"}]
        assert service._inject_memory(history, "s1") is history

    def test_appends_system_turn_when_facts_remembered(self, monkeypatch):
        monkeypatch.setattr(settings, "agent_memory_enabled", True)
        service = _make_service(agent_memory=AgentMemory())
        service._agent_memory.upsert_fact("s1", "deadline", "March 1", confidence="high")
        history = service._inject_memory([{"role": "user", "content": "q"}], "s1")
        assert history[-1]["role"] == "system"
        assert "deadline: March 1" in history[-1]["content"]

    def test_does_not_stack_stale_system_turns(self, monkeypatch):
        monkeypatch.setattr(settings, "agent_memory_enabled", True)
        service = _make_service(agent_memory=AgentMemory())
        service._agent_memory.upsert_fact("s1", "deadline", "March 1", confidence="high")
        first = service._inject_memory([], "s1")
        second = service._inject_memory(first, "s1")
        assert [t["role"] for t in second].count("system") == 1

    def test_remembered_facts_reach_the_generation_prompt(self, monkeypatch):
        monkeypatch.setattr(settings, "agent_memory_enabled", True)
        llm_client = FakeLLMClient(response="grounded answer using context")
        service = _make_service(llm_client, agent_memory=AgentMemory())
        service._agent_memory.upsert_fact("s1", "deadline", "March 1", confidence="high")

        history = service._inject_memory([], "s1")
        service._generate("when is the deadline?", [make_chunk()], history)

        assert "deadline: March 1" in llm_client.calls[-1]


class TestRemember:
    def test_records_turns(self, monkeypatch):
        monkeypatch.setattr(settings, "agent_memory_enabled", True)
        memory = AgentMemory()
        service = _make_service(agent_memory=memory)
        service._remember("s1", "question", "answer", [make_chunk()], "document_query")
        turns = memory.get_recent_turns("s1")
        assert [t["role"] for t in turns] == ["user", "assistant"]
        assert turns[0]["content"] == "question"
        assert turns[1]["content"] == "answer"

    def test_extracts_and_stores_facts_when_enabled(self, monkeypatch):
        monkeypatch.setattr(settings, "agent_memory_enabled", True)
        monkeypatch.setattr(settings, "agent_memory_fact_extraction_enabled", True)
        memory = AgentMemory()
        llm_client = FakeLLMClient(
            responses=[
                '{"facts": [{"key": "deadline", "value": "March 1", "confidence": "high"}]}'
            ]
        )
        service = _make_service(llm_client, agent_memory=memory)
        service._remember("s1", "when is the deadline?", "March 1", [make_chunk()], "document_query")
        assert memory.get_fact("s1", "deadline")["value"] == "March 1"
        assert memory.get_fact("s1", "deadline")["confidence"] == "high"

    def test_skips_fact_extraction_for_small_talk(self, monkeypatch):
        monkeypatch.setattr(settings, "agent_memory_enabled", True)
        monkeypatch.setattr(settings, "agent_memory_fact_extraction_enabled", True)
        memory = AgentMemory()
        llm_client = FakeLLMClient()
        service = _make_service(llm_client, agent_memory=memory)
        service._remember("s1", "hi", "Hello!", [], "conversational")
        assert llm_client.calls == []  # no extraction LLM call
        assert memory.get_all_facts("s1") == []

    def test_failed_extraction_degrades_silently(self, monkeypatch):
        monkeypatch.setattr(settings, "agent_memory_enabled", True)
        monkeypatch.setattr(settings, "agent_memory_fact_extraction_enabled", True)
        memory = AgentMemory()
        llm_client = FakeLLMClient(responses=["not json at all"])
        service = _make_service(llm_client, agent_memory=memory)
        service._remember("s1", "q", "a", [make_chunk()], "document_query")
        assert memory.get_all_facts("s1") == []  # no crash, no facts


class TestHandleQueryMemory:
    def test_remembered_facts_flow_into_handle_query_prompt(self, monkeypatch):
        monkeypatch.setattr(settings, "agent_memory_enabled", True)
        llm_client = FakeLLMClient(response="grounded answer using context")
        memory = AgentMemory()
        memory.upsert_fact("s1", "team", "5 people", confidence="high")
        service = _make_service(llm_client, agent_memory=memory)
        monkeypatch.setattr(
            rag_service_module,
            "retrieve",
            lambda query, vector_store, top_k=None, min_score=None, tenant_id=None, document_ids=None, image_vector_store=None: [make_chunk()],
        )

        service.handle_query("how big is the team?", history=None, session_id="s1")

        assert "team: 5 people" in llm_client.calls[0]

    def test_handle_query_records_exchange_and_facts(self, monkeypatch):
        monkeypatch.setattr(settings, "agent_memory_enabled", True)
        monkeypatch.setattr(settings, "agent_memory_fact_extraction_enabled", True)
        memory = AgentMemory()
        llm_client = FakeLLMClient(
            responses=[
                "grounded answer using context",
                '{"facts": [{"key": "team", "value": "5 people", "confidence": "high"}]}',
            ]
        )
        service = _make_service(llm_client, agent_memory=memory)
        monkeypatch.setattr(
            rag_service_module,
            "retrieve",
            lambda query, vector_store, top_k=None, min_score=None, tenant_id=None, document_ids=None, image_vector_store=None: [make_chunk()],
        )

        service.handle_query("how big is the team?", history=None, session_id="s1")

        assert memory.get_fact("s1", "team")["value"] == "5 people"
        turns = memory.get_recent_turns("s1")
        assert [t["role"] for t in turns] == ["user", "assistant"]

    def test_memory_disabled_no_side_effects(self, monkeypatch):
        monkeypatch.setattr(settings, "agent_memory_enabled", False)
        monkeypatch.setattr(
            rag_service_module,
            "retrieve",
            lambda query, vector_store, top_k=None, min_score=None, tenant_id=None, document_ids=None, image_vector_store=None: [make_chunk()],
        )
        memory = AgentMemory()
        service = _make_service(agent_memory=memory)
        service.handle_query("how big is the team?", history=None, session_id="s1")
        assert memory.get_recent_turns("s1") == []
        assert memory.get_all_facts("s1") == []
