"""Module 10 gap-closure (P4): structured output as a real production
path, not just an isolated parser.

Companion to:
- test_structured_output_eval.py (the 17-case parser dataset regression)
- test_human_approval_structured_output.py (ChatService._generate_structured
  unit behavior)

This file adds the piece those two don't cover: the actual application/API
path (POST /chat -> ChatService.handle_query -> the agent graph's
generator_node -> ChatService._generate_structured -> Pydantic validation
-> ChatResponse), proving structured mode is genuinely wired end to end
and that a malformed provider response degrades safely through that same
real path rather than only being testable via a direct parser call.
"""

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.models.document import EmbeddedChunk
from app.models.schemas import StructuredAnswer
from app.services.faiss_vector_store import FAISSVectorStore
from app.services.structured_output import parse_structured_answer

FAKE_EMBEDDING_DIM = 8
FAKE_EMBEDDING = [1.0] + [0.0] * (FAKE_EMBEDDING_DIM - 1)
SEEDED_DOCUMENT_ID = "22222222-2222-2222-2222-222222222222"
SEEDED_CHUNK_TEXT = "Apple scab is treated with sulfur or captan fungicide."


class FakeStructuredLLMClient:
    """Stands in for GeminiClient/GroqClient: implements both generate()
    (plain free-text) and generate_structured() (JSON mode), so both the
    happy path and the malformed-provider-output path can be exercised
    through the real endpoint without a live provider call."""

    def __init__(self, structured_response: str, plain_response: str = "plain fallback answer [1]."):
        self.structured_response = structured_response
        self.plain_response = plain_response
        self.structured_calls: list[str] = []
        self.generate_calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.generate_calls.append(prompt)
        return self.plain_response

    def generate_stream(self, prompt: str):
        self.generate_calls.append(prompt)
        yield self.plain_response

    def generate_structured(self, prompt: str) -> str:
        self.structured_calls.append(prompt)
        return self.structured_response


@pytest.fixture
def seeded_vector_store(tmp_path):
    store = FAISSVectorStore(index_path=tmp_path / "index.faiss", metadata_path=tmp_path / "metadata.json")
    store.create_index(dimension=FAKE_EMBEDDING_DIM)
    store.add_embeddings(
        [
            EmbeddedChunk(
                chunk_id="chunk-1",
                document_id=SEEDED_DOCUMENT_ID,
                embedding=FAKE_EMBEDDING,
                metadata={
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "source": "pdf",
                    "tenant_id": 1,
                    "text": SEEDED_CHUNK_TEXT,
                },
            )
        ]
    )
    return store


def _make_client(monkeypatch, seeded_vector_store, tmp_path, fake_llm):
    monkeypatch.setattr(settings, "structured_output_enabled", True)
    monkeypatch.setattr("app.api.v1.routes.query.get_vector_store", lambda: seeded_vector_store)
    monkeypatch.setattr("app.api.v1.routes.documents.get_vector_store", lambda: seeded_vector_store)
    monkeypatch.setattr("app.api.v1.routes.query.get_llm_client", lambda: fake_llm)
    monkeypatch.setattr("app.services.retrieval_service.embed_query", lambda query: FAKE_EMBEDDING)
    monkeypatch.setattr("app.services.hybrid_search.embed_query", lambda query: FAKE_EMBEDDING)

    from app.services.session_store import InMemorySessionStore

    test_session_store = InMemorySessionStore()
    monkeypatch.setattr("app.api.v1.routes.query.get_session_store", lambda: test_session_store)
    monkeypatch.setattr("app.services.session_store.get_session_store", lambda: test_session_store)

    feedback_path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr("app.services.feedback_service.FEEDBACK_DIR", tmp_path)
    monkeypatch.setattr("app.services.feedback_service.FEEDBACK_PATH", feedback_path)

    return TestClient(app)


VALID_HEADERS = {"X-API-Key": settings.api_key}


class TestEndpointStructuredOutputSuccess:
    """TASK 7: request -> application -> structured generation -> Pydantic
    validation -> response, exercised through the real POST /chat route."""

    def test_structured_response_true_returns_provider_structured_answer(
        self, monkeypatch, seeded_vector_store, tmp_path
    ):
        fake_llm = FakeStructuredLLMClient(
            structured_response='{"answer": "Sulfur or captan fungicide treats apple scab [1].", "sources": ["doc-1"]}'
        )
        client = _make_client(monkeypatch, seeded_vector_store, tmp_path, fake_llm)
        with client:
            response = client.post(
                "/chat",
                json={"query": "How do I treat apple scab?", "structured_response": True},
                headers=VALID_HEADERS,
            )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Sulfur or captan fungicide treats apple scab [1]."
        assert body["metadata"]["structured_output_used"] is True
        assert body["metadata"]["structured_output"]["answer"] == body["answer"]
        assert body["metadata"]["structured_output"]["sources"] == ["doc-1"]
        assert fake_llm.structured_calls  # the provider was actually asked for JSON mode
        assert not fake_llm.generate_calls  # no fallback needed


class TestEndpointStructuredOutputSafeRecovery:
    """Same real path, but the provider returns malformed JSON -- proves
    the degrade-to-free-text fallback is safe and never silently reported
    as a successful structured response."""

    def test_malformed_provider_json_degrades_to_free_text_safely(
        self, monkeypatch, seeded_vector_store, tmp_path
    ):
        fake_llm = FakeStructuredLLMClient(
            structured_response="not valid json at all",
            plain_response="Apple scab is treated with sulfur-based fungicide [1].",
        )
        client = _make_client(monkeypatch, seeded_vector_store, tmp_path, fake_llm)
        with client:
            response = client.post(
                "/chat",
                json={"query": "How do I treat apple scab?", "structured_response": True},
                headers=VALID_HEADERS,
            )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Apple scab is treated with sulfur-based fungicide [1]."
        assert body["metadata"]["structured_output_used"] is False
        assert "structured_output" not in body["metadata"]  # never fabricated on a fallback
        assert fake_llm.structured_calls  # JSON mode was genuinely attempted
        assert fake_llm.generate_calls  # and it fell back

    def test_schema_invalid_provider_json_also_degrades_safely(
        self, monkeypatch, seeded_vector_store, tmp_path
    ):
        """Valid JSON syntax but missing the required `answer` field --
        a schema-validation failure, not a parse failure -- must also
        degrade, not raise or half-succeed."""
        fake_llm = FakeStructuredLLMClient(
            structured_response='{"sources": ["doc-1"]}',
            plain_response="Fallback text answer [1].",
        )
        client = _make_client(monkeypatch, seeded_vector_store, tmp_path, fake_llm)
        with client:
            response = client.post(
                "/chat",
                json={"query": "How do I treat apple scab?", "structured_response": True},
                headers=VALID_HEADERS,
            )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Fallback text answer [1]."
        assert body["metadata"]["structured_output_used"] is False


class TestEndpointStructuredResponseFlagOff:
    def test_structured_response_false_never_calls_json_mode(self, monkeypatch, seeded_vector_store, tmp_path):
        fake_llm = FakeStructuredLLMClient(
            structured_response='{"answer": "should never be used", "sources": []}',
            plain_response="Normal free-text answer [1].",
        )
        client = _make_client(monkeypatch, seeded_vector_store, tmp_path, fake_llm)
        with client:
            response = client.post(
                "/chat",
                json={"query": "How do I treat apple scab?"},  # structured_response defaults False
                headers=VALID_HEADERS,
            )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Normal free-text answer [1]."
        assert "structured_output_used" not in body["metadata"]
        assert not fake_llm.structured_calls


class TestStreamingCompatibility:
    """TASK 3/8: /chat/stream is an intentionally free-form contract --
    structured_response is never forwarded to it, and it must keep
    streaming plain text unaffected by structured mode being enabled."""

    def test_chat_stream_ignores_structured_response_and_streams_plain_text(
        self, monkeypatch, seeded_vector_store, tmp_path
    ):
        fake_llm = FakeStructuredLLMClient(
            structured_response='{"answer": "should never be requested here", "sources": []}',
            plain_response="Streamed plain-text answer [1].",
        )
        client = _make_client(monkeypatch, seeded_vector_store, tmp_path, fake_llm)
        with client:
            response = client.post(
                "/chat/stream",
                json={"query": "How do I treat apple scab?", "structured_response": True},
                headers=VALID_HEADERS,
            )

        assert response.status_code == 200
        body = response.text
        assert "Streamed plain-text answer" in body
        assert not fake_llm.structured_calls  # JSON mode is never requested on the streaming path


class TestSchemaValidationEdgeCases:
    """TASK 4: StructuredAnswer / parse_structured_answer edge cases not
    already covered by the 17-case dataset, exercised directly against the
    real Pydantic model (not a hand-rolled duplicate)."""

    def test_null_answer_is_rejected(self):
        assert parse_structured_answer('{"answer": null, "sources": []}') is None

    def test_null_sources_falls_back_to_default(self):
        # sources has a default_factory, but an explicit `null` is still a
        # type mismatch against `list[str]` and must be rejected, not
        # silently coerced to [].
        assert parse_structured_answer('{"answer": "ok", "sources": null}') is None

    def test_valid_complete_response_round_trips_through_pydantic_directly(self):
        answer = StructuredAnswer.model_validate({"answer": "Complete answer.", "sources": ["doc-1", "doc-2"]})
        assert answer.answer == "Complete answer."
        assert answer.sources == ["doc-1", "doc-2"]

    def test_missing_answer_field_raises_on_direct_pydantic_validation(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            StructuredAnswer.model_validate({"sources": ["doc-1"]})
