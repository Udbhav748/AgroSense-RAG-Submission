"""Tests for Agent 1-2 features: retrieval confidence (1.1), query
contextualization (1.2), citation verification (1.3), clarifying
questions (1.4), persona presets (2.1), follow-up questions (2.2), and
the local research agent (2.3). Uses the same lightweight fakes as
test_rag_service.py."""

import app.services.rag_service as rag_service_module
from app.core.config import settings
from app.models.document import RetrievedChunk
from app.services.local_research_agent import LocalResearchAgent
from app.services.prompt_builder import FALLBACK_REPLY, PERSONAS, build_prompt
from app.services.query_planning import plan_subqueries
from app.services.rag_service import ChatService


class FakeLLMClient:
    def __init__(self, response="grounded answer", responses=None):
        self.response = response
        self._responses = list(responses) if responses is not None else None
        self.calls = []

    def generate(self, prompt):
        self.calls.append(prompt)
        if self._responses:
            return self._responses.pop(0)
        return self.response

    def generate_stream(self, prompt):
        self.calls.append(prompt)
        yield self.response


class FakeVectorStore:
    def total_vectors(self):
        return 1


def make_service(llm_client=None):
    return ChatService(FakeVectorStore(), llm_client or FakeLLMClient())


def make_chunk(document_id="doc-1", text="chunk text", score=0.9):
    return RetrievedChunk(
        chunk_id="chunk-1", document_id=document_id, text=text, score=score, metadata={}
    )


class TestRetrievalConfidence:
    def test_respond_surfaces_grade(self):
        service = make_service()
        response = service._respond(
            answer="x",
            retrieved_chunks=[make_chunk()],
            query="q",
            query_type="document_query",
            tool_used="retrieval",
            steps_taken=2,
            start=__import__("time").perf_counter(),
            retrieval_confidence="weak",
        )
        assert response.retrieval_confidence == "weak"

    def test_handle_query_threads_grade_into_response(self, monkeypatch):
        monkeypatch.setattr(settings, "retrieval_grade_threshold", 0.5)
        monkeypatch.setattr(
            rag_service_module,
            "retrieve",
            lambda *a, **k: [make_chunk(score=0.4)],
        )
        monkeypatch.setattr(settings, "agent_routing_enabled", False)
        monkeypatch.setattr(settings, "web_search_enabled", False)

        service = make_service()
        response = service.handle_query("what does the document say?")

        assert response.retrieval_confidence == "weak"


class TestContextualization:
    def test_contextualize_query_rewrites_using_history(self):
        llm = FakeLLMClient(response="What is the deadline for the project?")
        service = make_service(llm)
        history = [{"role": "user", "content": "Tell me about the project."}]
        assert service._contextualize_query("what is its deadline?", history) == (
            "What is the deadline for the project?"
        )

    def test_contextualize_query_degrades_to_raw_on_failure(self):
        llm = FakeLLMClient()
        llm.generate = lambda prompt: (_ for _ in ()).throw(RuntimeError("llm down"))
        service = make_service(llm)
        assert service._contextualize_query("what is its deadline?", [{"role": "user", "content": "x"}]) == (
            "what is its deadline?"
        )

    def test_no_history_returns_query_unchanged(self):
        service = make_service()
        assert service._contextualize_query("what is its deadline?", None) == "what is its deadline?"


class TestCitationVerification:
    def test_no_citations_passes(self):
        service = make_service()
        assert service._verify_citations("no citations here", [make_chunk()]) is True

    def test_supported_citations_pass(self):
        llm = FakeLLMClient(response='{"1": true}')
        service = make_service(llm)
        assert service._verify_citations("It costs $3 [1].", [make_chunk(text="it costs $3")]) is True

    def test_unsupported_citation_fails(self):
        llm = FakeLLMClient(response='{"1": false}')
        service = make_service(llm)
        assert service._verify_citations("It costs $3 [1].", [make_chunk(text="different text")]) is False

    def test_parse_failure_degrades_to_pass(self):
        llm = FakeLLMClient(response="not json")
        service = make_service(llm)
        assert service._verify_citations("It costs $3 [1].", [make_chunk()]) is True

    def test_citation_verification_in_correct_loop(self, monkeypatch):
        monkeypatch.setattr(settings, "citation_verification_enabled", True)
        llm = FakeLLMClient(responses=['{"1": false}', "better grounded answer"])
        service = make_service(llm)
        chunks = [make_chunk(text="the real content")]

        answer, _, _, _, _ = service._correct(
            "query", chunks, "An unsupported claim [1].", None, [], False, 1, 0
        )

        assert answer == "better grounded answer"


class TestClarifyingQuestion:
    def test_insufficient_grade_with_fallback_asks_clarification(self, monkeypatch):
        monkeypatch.setattr(settings, "clarifying_question_enabled", True)
        monkeypatch.setattr(settings, "agent_routing_enabled", False)
        monkeypatch.setattr(settings, "web_search_enabled", False)
        monkeypatch.setattr(settings, "retrieval_grade_threshold", 0.5)
        monkeypatch.setattr(
            rag_service_module, "retrieve", lambda *a, **k: []
        )
        llm = FakeLLMClient(responses=[FALLBACK_REPLY, "Which crop are you asking about?"])
        service = make_service(llm)

        response = service.handle_query("what about the crop?")

        assert response.is_clarifying_question is True
        assert response.answer == "Which crop are you asking about?"

    def test_flag_off_keeps_fallback(self, monkeypatch):
        monkeypatch.setattr(settings, "clarifying_question_enabled", False)
        monkeypatch.setattr(settings, "agent_routing_enabled", False)
        monkeypatch.setattr(settings, "web_search_enabled", False)
        monkeypatch.setattr(settings, "retrieval_grade_threshold", 0.5)
        monkeypatch.setattr(rag_service_module, "retrieve", lambda *a, **k: [])
        service = make_service(FakeLLMClient(response=FALLBACK_REPLY))

        response = service.handle_query("what about the crop?")

        assert response.is_clarifying_question is False
        assert response.answer == FALLBACK_REPLY


class TestPersona:
    def test_build_prompt_includes_persona_instruction(self):
        prompt = build_prompt("q", [make_chunk()], persona="concise")
        assert PERSONAS["concise"] in prompt

    def test_grounding_instructions_always_present(self):
        for persona in PERSONAS:
            prompt = build_prompt("q", [make_chunk()], persona=persona)
            assert "Answer the user's question" in prompt  # from _INSTRUCTIONS
            assert "Never hallucinate" in prompt

    def test_unknown_persona_ignored(self):
        prompt = build_prompt("q", [make_chunk()], persona="hacker")
        assert "hacker" not in prompt

    def test_persona_never_replaces_instructions(self):
        # The persona fragment comes AFTER the grounding instructions.
        prompt = build_prompt("q", [make_chunk()], persona="eli5")
        assert prompt.index("Never hallucinate") < prompt.index(PERSONAS["eli5"])

    def test_generate_threads_persona_to_prompt(self):
        llm = FakeLLMClient()
        service = make_service(llm)
        service._generate("q", [make_chunk()], None, persona="concise")
        assert PERSONAS["concise"] in llm.calls[0]


class TestFollowUpQuestions:
    def test_parses_valid_json_array(self):
        llm = FakeLLMClient(response='["q1?", "q2?", "q3?", "q4?"]')
        service = make_service(llm)
        follow_ups = service._suggest_follow_ups("q", "a")
        assert follow_ups == ["q1?", "q2?", "q3?"]  # capped at 3

    def test_malformed_response_degrades_to_empty(self):
        llm = FakeLLMClient(response="not json at all")
        service = make_service(llm)
        assert service._suggest_follow_ups("q", "a") == []

    def test_llm_failure_degrades_to_empty(self):
        llm = FakeLLMClient()
        llm.generate = lambda prompt: (_ for _ in ()).throw(RuntimeError("down"))
        service = make_service(llm)
        assert service._suggest_follow_ups("q", "a") == []

    def test_handle_query_lands_follow_ups_when_enabled(self, monkeypatch):
        monkeypatch.setattr(settings, "follow_up_questions_enabled", True)
        monkeypatch.setattr(settings, "agent_routing_enabled", False)
        monkeypatch.setattr(settings, "web_search_enabled", False)
        monkeypatch.setattr(settings, "retrieval_grade_threshold", 0.5)
        monkeypatch.setattr(
            rag_service_module, "retrieve", lambda *a, **k: [make_chunk(score=0.9)]
        )
        llm = FakeLLMClient(responses=["grounded answer.", '["more?", "later?"]'])
        service = make_service(llm)

        response = service.handle_query("what is scope?")

        assert response.follow_up_questions == ["more?", "later?"]


class TestLocalResearchAgent:
    def test_plan_subqueries_parses_json(self):
        llm = FakeLLMClient(response='{"queries": ["a", "b"]}')
        assert plan_subqueries(llm, "question", 3) == ["a", "b"]

    def test_plan_subqueries_degrades_to_original(self):
        llm = FakeLLMClient(response="not json")
        assert plan_subqueries(llm, "question", 3) == ["question"]

    def test_local_research_agent_returns_empty_when_no_chunks(self):
        class EmptyStore:
            def search(self, *a, **k):
                return []

        llm = FakeLLMClient(response='{"queries": ["q"]}')
        agent = LocalResearchAgent(llm, EmptyStore())
        findings = agent.run("question")
        assert findings.answer == ""
        assert findings.queries == ["q"]

    def test_local_research_agent_synthesizes_from_merged_chunks(self, monkeypatch):
        def fake_retrieve(query, vector_store, top_k=None, min_score=None, tenant_id=None, document_ids=None, image_vector_store=None):
            return [
                RetrievedChunk(
                    chunk_id=f"chunk-{query}",
                    document_id="doc-1",
                    text=f"chunk for {query}",
                    score=0.9,
                    metadata={},
                )
            ]

        monkeypatch.setattr("app.services.local_research_agent.retrieve", fake_retrieve)

        llm = FakeLLMClient(responses=['{"queries": ["a", "b"]}', "synthesized answer"])
        agent = LocalResearchAgent(llm, FakeVectorStore())
        findings = agent.run("question")

        assert findings.answer == "synthesized answer"
        assert len(findings.chunks) == 2


class TestLocalResearchBranch:
    def test_handle_query_uses_local_research_when_enabled(self, monkeypatch):
        monkeypatch.setattr(settings, "local_research_agent_enabled", True)
        monkeypatch.setattr(settings, "agent_routing_enabled", False)
        monkeypatch.setattr(settings, "web_search_enabled", False)
        monkeypatch.setattr(settings, "retrieval_grade_threshold", 0.5)
        monkeypatch.setattr(rag_service_module, "retrieve", lambda *a, **k: [])
        monkeypatch.setattr(
            rag_service_module,
            "LocalResearchAgent",
            lambda *a, **k: type(
                "FakeLocal",
                (),
                {"run": lambda self, q, tenant_id=None: type(
                    "F", (), {"answer": "local research answer", "chunks": [make_chunk()]}
                )()},
            )(),
        )
        service = make_service()

        response = service.handle_query("complex question?")

        assert response.answer == "local research answer"
        assert response.tool_used == "local_research"
