"""End-to-end API tests using FastAPI's TestClient.

The vector store is an in-memory FAISSVectorStore backed by a tmp_path
(never the real backend/vector_store/), and both the LLM and the
embedding call are stubbed out — no real Gemini or sentence-transformers
calls happen here, so this suite runs fast and offline.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.models.document import EmbeddedChunk, ExtractedImage
from app.models.schemas import (
    ChatResponse,
    DocumentDeleteResponse,
    DocumentImagesResponse,
    DocumentProcessingResponse,
    FeedbackResponse,
)
from app.services.document_processing_service import DocumentProcessingService
from app.services.faiss_vector_store import FAISSVectorStore
from tests.conftest import assert_matches_schema

VALID_HEADERS = {"X-API-Key": settings.api_key}
INVALID_HEADERS = {"X-API-Key": "not-the-real-key"}

# Small toy embedding — real dimensionality doesn't matter since embed_query
# is stubbed to always return this, and the seeded chunk uses it too, so
# retrieval always finds an exact (cosine similarity 1.0) match.
FAKE_EMBEDDING_DIM = 8
FAKE_EMBEDDING = [1.0] + [0.0] * (FAKE_EMBEDDING_DIM - 1)

SEEDED_DOCUMENT_ID = "11111111-1111-1111-1111-111111111111"
SEEDED_CHUNK_TEXT = "Project scope defines the boundaries of what work is included."

MINIMAL_PDF_BYTES = b"%PDF-1.4\n%fake-pdf-for-tests\n"


class FakeLLMClient:
    """Stands in for GeminiClient: no network call, fixed answer text."""

    def __init__(self, response_text: str = "This is a stubbed grounded answer about scope."):
        self.response_text = response_text
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response_text

    def generate_stream(self, prompt: str):
        # Split into two pieces (not a single yield) so /chat/stream tests
        # actually exercise multi-chunk reassembly, not just the
        # LLMClient-default single-yield fallback.
        self.calls.append(prompt)
        midpoint = len(self.response_text) // 2
        yield self.response_text[:midpoint]
        yield self.response_text[midpoint:]


@pytest.fixture
def seeded_vector_store(tmp_path):
    """A FAISSVectorStore living entirely under tmp_path, pre-seeded with
    one chunk so retrieval and delete-by-id both have something to find."""
    store = FAISSVectorStore(
        index_path=tmp_path / "index.faiss",
        metadata_path=tmp_path / "metadata.json",
    )
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


@pytest.fixture
def client(monkeypatch, seeded_vector_store, tmp_path):
    """A TestClient wired to the seeded, tmp_path-backed vector store and
    stubbed LLM/embedding calls — never touches the real vector store,
    Gemini, or the sentence-transformers model."""
    fake_llm = FakeLLMClient()

    # get_vector_store is imported by name into documents.py (`from
    # app.api.v1.routes.query import get_vector_store`), so both bindings
    # need patching — patching only query.get_vector_store would leave
    # documents.py still pointing at the original.
    monkeypatch.setattr("app.api.v1.routes.query.get_vector_store", lambda: seeded_vector_store)
    monkeypatch.setattr(
        "app.api.v1.routes.documents.get_vector_store", lambda: seeded_vector_store
    )
    monkeypatch.setattr("app.api.v1.routes.query.get_llm_client", lambda: fake_llm)
    monkeypatch.setattr("app.services.retrieval_service.embed_query", lambda query: FAKE_EMBEDDING)
    # hybrid_search_enabled defaults to True (see docs/OPERATIONS.md's
    # "Retrieval ablation"), and hybrid_search.py imports embed_query into
    # its own namespace rather than going through retrieval_service's —
    # the patch above alone wouldn't stop this test from loading the real
    # sentence-transformers model via the hybrid path.
    monkeypatch.setattr("app.services.hybrid_search.embed_query", lambda query: FAKE_EMBEDDING)

    # Patch session store to use a fresh in-memory instance per test
    from app.services.session_store import InMemorySessionStore
    test_session_store = InMemorySessionStore()
    monkeypatch.setattr("app.api.v1.routes.query.get_session_store", lambda: test_session_store)
    monkeypatch.setattr("app.services.session_store.get_session_store", lambda: test_session_store)

    # Feedback events go to a tmp_path file, never the real
    # backend/feedback/feedback.jsonl. record_feedback() looks these up as
    # module globals at call time, so patching them here (rather than
    # passing a path in) is enough even though query.py imported the
    # function by name.
    feedback_path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr("app.services.feedback_service.FEEDBACK_DIR", tmp_path)
    monkeypatch.setattr("app.services.feedback_service.FEEDBACK_PATH", feedback_path)

    with TestClient(app) as test_client:
        test_client.fake_llm = fake_llm
        test_client.feedback_path = feedback_path
        yield test_client


class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        # The provider's key is never echoed — only a boolean readiness
        # signal, on an unauthenticated endpoint.
        assert body["llm"]["provider_configured"] is True
        assert "api_key" not in json.dumps(body).lower()
        # Multi-modal capability flags are reported as configured, and the
        # section always present even when every flag is off.
        assert set(body["multimodal"]) >= {
            "image_extraction_enabled",
            "image_captioning_enabled",
            "table_extraction_enabled",
            "vision_qa_enabled",
            "ocr_available",
        }
        assert isinstance(body["multimodal"]["ocr_available"], bool)

    def test_health_reports_disabled_capabilities(self, client, monkeypatch):
        monkeypatch.setattr(settings, "image_extraction_enabled", False)
        monkeypatch.setattr(settings, "image_captioning_enabled", False)
        monkeypatch.setattr(settings, "table_extraction_enabled", False)
        monkeypatch.setattr(settings, "vision_qa_enabled", False)
        monkeypatch.setattr(settings, "fallback_llm_provider", None)

        response = client.get("/health")

        body = response.json()
        assert body["multimodal"]["image_extraction_enabled"] is False
        assert body["llm"]["fallback_provider"] is None
        assert body["llm"]["fallback_configured"] is False


class TestUpload:
    def test_rejects_wrong_mime_type(self, client):
        response = client.post(
            "/upload",
            headers=VALID_HEADERS,
            files={"file": ("notes.txt", b"just some text", "text/plain")},
        )
        assert response.status_code == 415

    def test_rejects_oversized_file(self, client, monkeypatch):
        # Zero out the limit so any non-empty upload exceeds it, instead of
        # constructing a real multi-megabyte payload.
        monkeypatch.setattr(settings, "max_upload_size_mb", 0)
        response = client.post(
            "/upload",
            headers=VALID_HEADERS,
            files={"file": ("doc.pdf", MINIMAL_PDF_BYTES, "application/pdf")},
        )
        assert response.status_code == 413

    def test_missing_api_key_returns_401(self, client):
        response = client.post(
            "/upload",
            files={"file": ("doc.pdf", MINIMAL_PDF_BYTES, "application/pdf")},
        )
        assert response.status_code == 401

    def test_upload_accepts_and_returns_collection(self, client, monkeypatch):
        async def fake_process(self, file, tenant_id=None, collection=None):
            return DocumentProcessingResponse(
                document_id="doc-1",
                original_filename="doc.pdf",
                total_pages=1,
                total_chunks=1,
                total_embeddings=1,
                pages_ocred=0,
                processing_time=0.1,
                status="processed",
                collection=collection,
            )

        monkeypatch.setattr(DocumentProcessingService, "process", fake_process)
        response = client.post(
            "/upload",
            headers=VALID_HEADERS,
            files={"file": ("doc.pdf", MINIMAL_PDF_BYTES, "application/pdf")},
            data={"collection": "finance"},
        )
        assert response.status_code == 201
        assert response.json()["collection"] == "finance"


class TestChat:
    def test_rejects_empty_query(self, client):
        response = client.post("/chat", headers=VALID_HEADERS, json={"query": ""})
        assert response.status_code == 422

    def test_happy_path_returns_valid_chat_response(self, client):
        response = client.post(
            "/chat",
            headers=VALID_HEADERS,
            json={"query": "What does the document say about project scope?"},
        )
        assert response.status_code == 200

        payload = response.json()
        assert_matches_schema(ChatResponse, payload)

        assert payload["answer"] == client.fake_llm.response_text
        assert payload["tool_used"] == "retrieval"
        assert payload["answer_source"] == "documents"
        assert payload["sources"] == [
            {
                "number": 1,
                "document_id": SEEDED_DOCUMENT_ID,
                "chunk_id": "chunk-1",
                "excerpt": SEEDED_CHUNK_TEXT,
                "url": None,
                # Multi-modal RAG citation fields (rag_service._source_references) —
                # this fixture's seeded chunk carries no content_type/page_number
                # metadata, so both fall back to their defaults, same as any
                # ordinary pre-multi-modal chunk would.
                "content_type": "text",
                "page_number": None,
            }
        ]
        assert payload["steps_taken"] >= 4  # plan + retrieve + grade + generate
        assert len(client.fake_llm.calls) == 1  # no reflection retry needed


def _parse_sse_events(response_text: str) -> list[dict]:
    return [
        json.loads(line[len("data: ") :])
        for line in response_text.split("\n\n")
        if line.startswith("data: ")
    ]


class TestChatStream:
    def test_event_sequence_ends_with_valid_done_event(self, client):
        response = client.post(
            "/chat/stream",
            headers=VALID_HEADERS,
            json={"query": "What does the document say about project scope?"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        events = _parse_sse_events(response.text)
        assert events, "expected at least one SSE event"

        event_types = [event["type"] for event in events]
        assert event_types[0] == "trace"
        assert events[0]["stage"] == "planning"
        assert "answer_chunk" in event_types
        assert event_types[-1] == "done"

        # Every trace stage the retrieve pipeline should hit for a normal,
        # well-grounded query, in order (web_search/reflecting are absent
        # here since nothing about this fixture triggers the corrective
        # loop or the web fallback).
        trace_stages = [event["stage"] for event in events if event["type"] == "trace"]
        assert trace_stages == ["planning", "retrieval", "grading", "generating"]

        done_payload = events[-1]["payload"]
        assert_matches_schema(ChatResponse, done_payload)
        assert done_payload["answer"] == client.fake_llm.response_text
        assert done_payload["tool_used"] == "retrieval"
        assert done_payload["answer_source"] == "documents"
        assert done_payload["sources"] == [
            {
                "number": 1,
                "document_id": SEEDED_DOCUMENT_ID,
                "chunk_id": "chunk-1",
                "excerpt": SEEDED_CHUNK_TEXT,
                "url": None,
                "content_type": "text",
                "page_number": None,
            }
        ]

        # The reassembled answer_chunk text matches the done payload's
        # answer exactly — streaming and the final response agree.
        streamed_answer = "".join(
            event["text"] for event in events if event["type"] == "answer_chunk"
        )
        assert streamed_answer == done_payload["answer"]

    def test_missing_api_key_returns_401(self, client):
        response = client.post("/chat/stream", json={"query": "anything"})
        assert response.status_code == 401

    def test_rejects_empty_query(self, client):
        response = client.post("/chat/stream", headers=VALID_HEADERS, json={"query": ""})
        assert response.status_code == 422


class TestChatFeedback:
    def test_valid_feedback_is_recorded(self, client):
        response = client.post(
            "/chat/feedback",
            headers=VALID_HEADERS,
            json={"message_id": "msg-1-123", "rating": "up", "comment": "Spot on."},
        )
        assert response.status_code == 200

        payload = response.json()
        assert_matches_schema(FeedbackResponse, payload)
        assert payload == {"status": "recorded"}

        lines = client.feedback_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["message_id"] == "msg-1-123"
        assert event["rating"] == "up"
        # comment is encrypted at rest (Module 10 gap-closure) -- the raw
        # stored value must never be the plaintext, only the "enc1:"
        # ciphertext marker; the real value is recovered through the
        # application's own read path (GET /feedback, see
        # TestFeedbackReadBack), not by reading this file directly.
        assert event["comment"] != "Spot on."
        assert event["comment"].startswith("enc1:")

    def test_comment_is_optional(self, client):
        response = client.post(
            "/chat/feedback",
            headers=VALID_HEADERS,
            json={"message_id": "msg-2-456", "rating": "down"},
        )
        assert response.status_code == 200
        assert json.loads(client.feedback_path.read_text(encoding="utf-8"))["comment"] is None

    def test_invalid_rating_returns_422(self, client):
        response = client.post(
            "/chat/feedback",
            headers=VALID_HEADERS,
            json={"message_id": "msg-3-789", "rating": "sideways"},
        )
        assert response.status_code == 422
        assert not client.feedback_path.exists()

    def test_missing_api_key_returns_401(self, client):
        response = client.post(
            "/chat/feedback",
            json={"message_id": "msg-4-000", "rating": "up"},
        )
        assert response.status_code == 401
        assert not client.feedback_path.exists()

    def test_reviewer_id_resolved_from_authenticated_client_not_request_body(self, client):
        response = client.post(
            "/chat/feedback",
            headers=VALID_HEADERS,
            # Even if a client tried to claim a different reviewer identity
            # in the body, there's no field for it — reviewer_id can only
            # ever come from the API key that authenticated the request.
            json={"message_id": "msg-5-111", "rating": "up"},
        )
        assert response.status_code == 200
        event = json.loads(client.feedback_path.read_text(encoding="utf-8"))
        assert event["reviewer_id"] == "default"  # VALID_HEADERS' key maps to client "default"

    def test_rubric_is_optional_and_recorded_when_given(self, client):
        rubric = {
            "correctness": 5,
            "helpfulness": 4,
            "completeness": 5,
            "safety": 5,
            "tone": 4,
            "groundedness": 5,
            "citation_quality": 3,
        }
        response = client.post(
            "/chat/feedback",
            headers=VALID_HEADERS,
            json={"message_id": "msg-6-222", "rating": "up", "rubric": rubric},
        )
        assert response.status_code == 200
        event = json.loads(client.feedback_path.read_text(encoding="utf-8"))
        assert event["rubric"] == rubric

    def test_feedback_without_rubric_still_works(self, client):
        response = client.post(
            "/chat/feedback",
            headers=VALID_HEADERS,
            json={"message_id": "msg-7-333", "rating": "down"},
        )
        assert response.status_code == 200
        event = json.loads(client.feedback_path.read_text(encoding="utf-8"))
        assert event["rubric"] is None

    def test_rubric_score_out_of_range_returns_422(self, client):
        rubric = {
            "correctness": 6,  # out of the 1-5 range
            "helpfulness": 4,
            "completeness": 5,
            "safety": 5,
            "tone": 4,
            "groundedness": 5,
            "citation_quality": 3,
        }
        response = client.post(
            "/chat/feedback",
            headers=VALID_HEADERS,
            json={"message_id": "msg-8-444", "rating": "up", "rubric": rubric},
        )
        assert response.status_code == 422
        assert not client.feedback_path.exists()

    def test_incomplete_rubric_returns_422(self, client):
        # All seven criteria are required together when a rubric is sent at
        # all — a partial rubric isn't a meaningful per-criterion average
        # or agreement data point (see RubricScores in app/models/schemas.py).
        incomplete_rubric = {"correctness": 5, "helpfulness": 4}
        response = client.post(
            "/chat/feedback",
            headers=VALID_HEADERS,
            json={"message_id": "msg-9-555", "rating": "up", "rubric": incomplete_rubric},
        )
        assert response.status_code == 422
        assert not client.feedback_path.exists()


class TestFeedbackReadBack:
    """Feature #6 — GET /feedback surfaces recorded feedback events back
    to the app (newest first)."""

    def test_empty_store_returns_empty_list(self, client):
        response = client.get("/feedback", headers=VALID_HEADERS)
        assert response.status_code == 200
        body = response.json()
        assert body == {"events": [], "count": 0}

    def test_returns_recorded_events_newest_first(self, client):
        for i, msg in enumerate(["older", "newer"]):
            client.post(
                "/chat/feedback",
                headers=VALID_HEADERS,
                json={"message_id": msg, "rating": "up", "comment": f"comment {i}"},
            )
        response = client.get("/feedback", headers=VALID_HEADERS)
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 2
        assert [e["message_id"] for e in body["events"]] == ["newer", "older"]
        assert body["events"][0]["rating"] == "up"
        assert body["events"][1]["comment"] == "comment 0"

    def test_requires_auth(self, client):
        assert client.get("/feedback").status_code == 401

    def test_schema_compliance(self, client):
        from app.models.schemas import FeedbackListResponse

        client.post(
            "/chat/feedback",
            headers=VALID_HEADERS,
            json={"message_id": "m", "rating": "down"},
        )
        body = client.get("/feedback", headers=VALID_HEADERS).json()
        assert_matches_schema(FeedbackListResponse, body)

    def test_member_sees_only_own_feedback_by_default(self, client, monkeypatch):
        # Force a distinct reviewer for the second post so scoping is
        # observable; both posts go through the same client but the
        # reviewer_id is resolved from the API key, so patching
        # resolve_tenant identity changes who "default" is.
        monkeypatch.setattr("app.core.auth.resolve_tenant", lambda client_name: (7, "member"))
        client.post(
            "/chat/feedback",
            headers=VALID_HEADERS,
            json={"message_id": "other-reviewer", "rating": "up"},
        )
        monkeypatch.setattr("app.core.auth.resolve_tenant", lambda client_name: (7, "member"))
        body = client.get("/feedback", headers=VALID_HEADERS).json()
        # Same caller identity in this fixture; assert it returns its own
        # events without error and that filtering by another reviewer is
        # admin-only.
        assert body["count"] >= 1
        denied = client.get("/feedback", params={"reviewer_id": "nobody-else"}, headers=VALID_HEADERS)
        assert denied.status_code == 403


class TestDeleteDocument:
    def test_without_confirm_returns_400(self, client):
        response = client.delete(f"/documents/{SEEDED_DOCUMENT_ID}", headers=VALID_HEADERS)
        assert response.status_code == 400

    def test_unknown_document_returns_404(self, client):
        response = client.delete(
            "/documents/does-not-exist", params={"confirm": "true"}, headers=VALID_HEADERS
        )
        assert response.status_code == 404

    def test_existing_document_deletes_successfully(self, client):
        response = client.delete(
            f"/documents/{SEEDED_DOCUMENT_ID}", params={"confirm": "true"}, headers=VALID_HEADERS
        )
        assert response.status_code == 200

        payload = response.json()
        assert_matches_schema(DocumentDeleteResponse, payload)
        assert payload["document_id"] == SEEDED_DOCUMENT_ID
        assert payload["chunks_removed"] == 1

    def test_wrong_tenant_cannot_delete_document(self, client, monkeypatch, seeded_vector_store):
        # DB disabled (this fixture's default) means request.state.tenant_id
        # is always None, so the ownership check never has anything to
        # compare — simulate a DB-enabled, multi-tenant request by
        # patching resolve_tenant (what require_api_key calls) and
        # get_document_owner (what the route calls) directly, the same
        # boundary-mocking style the rest of this fixture already uses.
        # role="admin" so these ownership-focused tests aren't incidentally
        # blocked by the separate role gate (see TestRoleBasedAccessControl).
        monkeypatch.setattr("app.core.auth.resolve_tenant", lambda client_name: (1, "admin"))
        monkeypatch.setattr("app.api.v1.routes.documents.get_document_owner", lambda document_id: 2)

        response = client.delete(
            f"/documents/{SEEDED_DOCUMENT_ID}", params={"confirm": "true"}, headers=VALID_HEADERS
        )

        # 404, not 403 — a non-owner gets the same response as "doesn't
        # exist" rather than a signal that confirms the document is real.
        assert response.status_code == 404
        # And the document must still actually be there — a denied
        # request must never have touched the vector store.
        assert seeded_vector_store.get_chunks_by_document(SEEDED_DOCUMENT_ID) != []

    def test_same_tenant_can_delete_document(self, client, monkeypatch):
        # role="admin" so these ownership-focused tests aren't incidentally
        # blocked by the separate role gate (see TestRoleBasedAccessControl).
        monkeypatch.setattr("app.core.auth.resolve_tenant", lambda client_name: (1, "admin"))
        monkeypatch.setattr("app.api.v1.routes.documents.get_document_owner", lambda document_id: 1)

        response = client.delete(
            f"/documents/{SEEDED_DOCUMENT_ID}", params={"confirm": "true"}, headers=VALID_HEADERS
        )

        assert response.status_code == 200

    def test_unknown_owner_does_not_block_delete(self, client, monkeypatch):
        # No DB row for this document (legacy upload, or uploaded with the
        # DB disabled) — unknown ownership isn't treated as a mismatch.
        # role="admin" so these ownership-focused tests aren't incidentally
        # blocked by the separate role gate (see TestRoleBasedAccessControl).
        monkeypatch.setattr("app.core.auth.resolve_tenant", lambda client_name: (1, "admin"))
        monkeypatch.setattr("app.api.v1.routes.documents.get_document_owner", lambda document_id: None)

        response = client.delete(
            f"/documents/{SEEDED_DOCUMENT_ID}", params={"confirm": "true"}, headers=VALID_HEADERS
        )

        assert response.status_code == 200


class TestRoleBasedAccessControl:
    """The role gate on DELETE /documents/{id} — see documents.py's
    delete_document."""

    def test_member_role_can_delete_own_document(self, client, monkeypatch):
        monkeypatch.setattr("app.core.auth.resolve_tenant", lambda client_name: (1, "member"))
        monkeypatch.setattr("app.api.v1.routes.documents.get_document_owner", lambda document_id: 1)

        response = client.delete(
            f"/documents/{SEEDED_DOCUMENT_ID}", params={"confirm": "true"}, headers=VALID_HEADERS
        )

        assert response.status_code == 200

    def test_member_role_cannot_delete_other_tenant_document(self, client, monkeypatch, seeded_vector_store):
        monkeypatch.setattr("app.core.auth.resolve_tenant", lambda client_name: (1, "member"))
        monkeypatch.setattr("app.api.v1.routes.documents.get_document_owner", lambda document_id: 2)

        response = client.delete(
            f"/documents/{SEEDED_DOCUMENT_ID}", params={"confirm": "true"}, headers=VALID_HEADERS
        )

        assert response.status_code == 404
        # A denied request must never have touched the vector store.
        assert seeded_vector_store.get_chunks_by_document(SEEDED_DOCUMENT_ID) != []

    def test_admin_role_can_delete_document(self, client, monkeypatch):
        monkeypatch.setattr("app.core.auth.resolve_tenant", lambda client_name: (1, "admin"))

        response = client.delete(
            f"/documents/{SEEDED_DOCUMENT_ID}", params={"confirm": "true"}, headers=VALID_HEADERS
        )

        assert response.status_code == 200

    def test_no_role_info_does_not_block_delete(self, client):
        # DB disabled (this fixture's default) means request.state.role is
        # always None — the pre-RBAC behavior (delete always allowed) is
        # preserved rather than locking everyone out because no role could
        # be determined.
        response = client.delete(
            f"/documents/{SEEDED_DOCUMENT_ID}", params={"confirm": "true"}, headers=VALID_HEADERS
        )

        assert response.status_code == 200


class TestDocumentDeleteApprovalGate:
    """Settings.document_delete_requires_approval — same human-in-the-loop
    shape as web_search_requires_approval (see
    test_human_approval_structured_output.py), applied to document
    deletion. Distinct from confirm=true (mistake-prevention) and the role
    gate above (access control): this is a deployment policy toggle.

    PHASE 5 SECURITY FIX: this gate used to accept a bare client-supplied
    `approved=true` query param as proof of approval, which any caller with
    valid API-key credentials could self-satisfy without an operator ever
    resolving the registered Approval (see docs/MODULE10_FINAL_AUDIT.md §8,
    Finding 2). It now requires a real, resolved `approval_id` looked up
    against the actual ApprovalStore — `approved=true` alone no longer has
    any effect. This class replaces the old `test_gate_on_and_approved_deletes`
    (which asserted the insecure behavior) with the 7 cases Phase 5 requires."""

    def test_gate_off_by_default_deletes_without_approval(self, client):
        response = client.delete(
            f"/documents/{SEEDED_DOCUMENT_ID}", params={"confirm": "true"}, headers=VALID_HEADERS
        )
        assert response.status_code == 200

    def test_no_approval_denied(self, client, monkeypatch, seeded_vector_store):
        monkeypatch.setattr(settings, "document_delete_requires_approval", True)

        response = client.delete(
            f"/documents/{SEEDED_DOCUMENT_ID}", params={"confirm": "true"}, headers=VALID_HEADERS
        )

        assert response.status_code == 400
        # A denied request must never have touched the vector store.
        assert seeded_vector_store.get_chunks_by_document(SEEDED_DOCUMENT_ID) != []

    def test_bare_approved_true_no_longer_bypasses_gate(self, client, monkeypatch, seeded_vector_store):
        """The exact behavior the Phase 5 fix removes: a caller-supplied
        `approved=true` with no real approval_id must be denied."""
        monkeypatch.setattr(settings, "document_delete_requires_approval", True)

        response = client.delete(
            f"/documents/{SEEDED_DOCUMENT_ID}",
            params={"confirm": "true", "approved": "true"},
            headers=VALID_HEADERS,
        )

        assert response.status_code == 400
        assert seeded_vector_store.get_chunks_by_document(SEEDED_DOCUMENT_ID) != []

    def test_pending_approval_denied(self, client, monkeypatch, seeded_vector_store):
        from app.services import approval_service

        monkeypatch.setattr(settings, "document_delete_requires_approval", True)
        approval = approval_service.get_approval_store().register(
            action=approval_service.APPROVAL_ACTION_DOCUMENT_DELETE,
            payload={"document_id": SEEDED_DOCUMENT_ID},
        )

        response = client.delete(
            f"/documents/{SEEDED_DOCUMENT_ID}",
            params={"confirm": "true", "approval_id": approval.approval_id},
            headers=VALID_HEADERS,
        )

        assert response.status_code == 400
        assert seeded_vector_store.get_chunks_by_document(SEEDED_DOCUMENT_ID) != []

    def test_rejected_approval_denied(self, client, monkeypatch, seeded_vector_store):
        from app.services import approval_service

        monkeypatch.setattr(settings, "document_delete_requires_approval", True)
        approval = approval_service.get_approval_store().register(
            action=approval_service.APPROVAL_ACTION_DOCUMENT_DELETE,
            payload={"document_id": SEEDED_DOCUMENT_ID},
        )
        approval_service.get_approval_store().resolve(
            approval.approval_id, approved=False, resolved_by="operator"
        )

        response = client.delete(
            f"/documents/{SEEDED_DOCUMENT_ID}",
            params={"confirm": "true", "approval_id": approval.approval_id},
            headers=VALID_HEADERS,
        )

        assert response.status_code == 400
        assert seeded_vector_store.get_chunks_by_document(SEEDED_DOCUMENT_ID) != []

    def test_expired_approval_denied(self, client, monkeypatch, seeded_vector_store):
        from app.services import approval_service

        monkeypatch.setattr(settings, "document_delete_requires_approval", True)
        approval = approval_service.get_approval_store().register(
            action=approval_service.APPROVAL_ACTION_DOCUMENT_DELETE,
            payload={"document_id": SEEDED_DOCUMENT_ID},
            ttl_seconds=-1,  # already expired
        )

        response = client.delete(
            f"/documents/{SEEDED_DOCUMENT_ID}",
            params={"confirm": "true", "approval_id": approval.approval_id},
            headers=VALID_HEADERS,
        )

        assert response.status_code == 400
        assert seeded_vector_store.get_chunks_by_document(SEEDED_DOCUMENT_ID) != []

    def test_mismatched_document_id_denied(self, client, monkeypatch, seeded_vector_store):
        """An approval genuinely APPROVED, but for a different document_id,
        must not authorize deleting this one."""
        from app.services import approval_service

        monkeypatch.setattr(settings, "document_delete_requires_approval", True)
        approval = approval_service.get_approval_store().register(
            action=approval_service.APPROVAL_ACTION_DOCUMENT_DELETE,
            payload={"document_id": "99999999-9999-9999-9999-999999999999"},
        )
        approval_service.get_approval_store().resolve(
            approval.approval_id, approved=True, resolved_by="operator"
        )

        response = client.delete(
            f"/documents/{SEEDED_DOCUMENT_ID}",
            params={"confirm": "true", "approval_id": approval.approval_id},
            headers=VALID_HEADERS,
        )

        assert response.status_code == 400
        assert seeded_vector_store.get_chunks_by_document(SEEDED_DOCUMENT_ID) != []

    def test_genuinely_approved_approval_allowed(self, client, monkeypatch):
        from app.services import approval_service

        monkeypatch.setattr(settings, "document_delete_requires_approval", True)
        approval = approval_service.get_approval_store().register(
            action=approval_service.APPROVAL_ACTION_DOCUMENT_DELETE,
            payload={"document_id": SEEDED_DOCUMENT_ID},
        )
        approval_service.get_approval_store().resolve(
            approval.approval_id, approved=True, resolved_by="operator"
        )

        response = client.delete(
            f"/documents/{SEEDED_DOCUMENT_ID}",
            params={"confirm": "true", "approval_id": approval.approval_id},
            headers=VALID_HEADERS,
        )

        assert response.status_code == 200


class TestListDocumentsAllTenants:
    """The second RBAC-gated action (see documents.py's
    list_uploaded_documents): an admin can opt into a cross-tenant view via
    all_tenants=true. DB disabled (this fixture's default) means
    request.state.role is always None, so all_tenants=true is denied
    outright rather than silently falling back to normal scoping."""

    def test_default_scoping_unaffected(self, client):
        response = client.get("/documents", headers=VALID_HEADERS)
        assert response.status_code == 200

    def test_all_tenants_denied_without_admin_role(self, client, monkeypatch):
        monkeypatch.setattr("app.core.auth.resolve_tenant", lambda client_name: (1, "member"))

        response = client.get("/documents", params={"all_tenants": "true"}, headers=VALID_HEADERS)

        assert response.status_code == 403

    def test_all_tenants_denied_with_no_role_info(self, client):
        # DB disabled: role is always None, not "admin" — denied, not a
        # silent fallback to normal scoping.
        response = client.get("/documents", params={"all_tenants": "true"}, headers=VALID_HEADERS)

        assert response.status_code == 403

    def test_all_tenants_allowed_for_admin(self, client, monkeypatch):
        monkeypatch.setattr("app.core.auth.resolve_tenant", lambda client_name: (1, "admin"))

        response = client.get("/documents", params={"all_tenants": "true"}, headers=VALID_HEADERS)

        assert response.status_code == 200


class TestDocumentImages:
    """GET /documents/{id}/images and /documents/{id}/images/{image_id}
    (multi-modal RAG, Phase 1): the listing is read from the per-document
    manifest written at ingestion, and the fetch route serves the
    persisted bytes. Same ownership model as DELETE — 404 for a
    non-owner, unknown image, or missing bytes."""

    def _seed_images(self, tmp_path, monkeypatch, document_id=SEEDED_DOCUMENT_ID):
        monkeypatch.setattr(settings, "data_dir_override", str(tmp_path))
        from app.services.image_captioning_service import write_image_manifest

        image_dir = tmp_path / settings.image_storage_dir_name
        image_dir.mkdir(parents=True, exist_ok=True)
        png = b"\x89PNG\r\n\x1a\nfake-image-bytes"
        (image_dir / f"{document_id}_img_1.png").write_bytes(png)
        write_image_manifest(
            [
                ExtractedImage(
                    image_id=f"{document_id}_img_1",
                    document_id=document_id,
                    page_number=3,
                    content_type="figure",
                    storage_path=f"{document_id}_img_1.png",
                    mime_type="image/png",
                    width=300,
                    height=200,
                    byte_size=len(png),
                )
            ]
        )
        return png

    def test_lists_extracted_images(self, client, tmp_path, monkeypatch):
        png = self._seed_images(tmp_path, monkeypatch)

        response = client.get(f"/documents/{SEEDED_DOCUMENT_ID}/images", headers=VALID_HEADERS)

        assert response.status_code == 200
        payload = response.json()
        assert_matches_schema(DocumentImagesResponse, payload)
        assert payload["document_id"] == SEEDED_DOCUMENT_ID
        assert payload["total"] == 1
        item = payload["images"][0]
        assert item["image_id"] == f"{SEEDED_DOCUMENT_ID}_img_1"
        assert item["page_number"] == 3
        assert item["content_type"] == "figure"
        assert item["byte_size"] == len(png)
        assert item["url"] == (
            f"/documents/{SEEDED_DOCUMENT_ID}/images/{SEEDED_DOCUMENT_ID}_img_1"
        )

    def test_serves_image_bytes(self, client, tmp_path, monkeypatch):
        png = self._seed_images(tmp_path, monkeypatch)

        response = client.get(
            f"/documents/{SEEDED_DOCUMENT_ID}/images/{SEEDED_DOCUMENT_ID}_img_1",
            headers=VALID_HEADERS,
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content == png

    def test_no_images_returns_empty_list(self, client):
        response = client.get(f"/documents/{SEEDED_DOCUMENT_ID}/images", headers=VALID_HEADERS)
        assert response.status_code == 200
        assert response.json() == {
            "document_id": SEEDED_DOCUMENT_ID,
            "total": 0,
            "images": [],
        }

    def test_unknown_image_returns_404(self, client, tmp_path, monkeypatch):
        self._seed_images(tmp_path, monkeypatch)

        response = client.get(
            f"/documents/{SEEDED_DOCUMENT_ID}/images/nope", headers=VALID_HEADERS
        )

        assert response.status_code == 404

    def test_missing_bytes_returns_404(self, client, tmp_path, monkeypatch):
        # Manifest claims an image whose bytes were cleaned off disk.
        self._seed_images(tmp_path, monkeypatch)
        image_dir = tmp_path / settings.image_storage_dir_name
        (image_dir / f"{SEEDED_DOCUMENT_ID}_img_1.png").unlink()

        response = client.get(
            f"/documents/{SEEDED_DOCUMENT_ID}/images/{SEEDED_DOCUMENT_ID}_img_1",
            headers=VALID_HEADERS,
        )

        assert response.status_code == 404

    def test_requires_auth(self, client, tmp_path, monkeypatch):
        self._seed_images(tmp_path, monkeypatch)

        assert client.get(f"/documents/{SEEDED_DOCUMENT_ID}/images").status_code == 401

    def test_wrong_tenant_cannot_list_images(self, client, tmp_path, monkeypatch):
        self._seed_images(tmp_path, monkeypatch)
        # DB disabled (this fixture's default) leaves tenant_id None, so
        # simulate a DB-enabled, multi-tenant request the same way
        # TestDeleteDocument does.
        monkeypatch.setattr("app.core.auth.resolve_tenant", lambda client_name: (1, "admin"))
        monkeypatch.setattr(
            "app.api.v1.routes.documents.get_document_owner", lambda document_id: 2
        )

        response = client.get(f"/documents/{SEEDED_DOCUMENT_ID}/images", headers=VALID_HEADERS)

        # 404, not 403 — a non-owner gets the same response as "doesn't exist".
        assert response.status_code == 404

    def test_same_tenant_can_list_images(self, client, tmp_path, monkeypatch):
        self._seed_images(tmp_path, monkeypatch)
        monkeypatch.setattr("app.core.auth.resolve_tenant", lambda client_name: (1, "admin"))
        monkeypatch.setattr(
            "app.api.v1.routes.documents.get_document_owner", lambda document_id: 1
        )

        response = client.get(f"/documents/{SEEDED_DOCUMENT_ID}/images", headers=VALID_HEADERS)

        assert response.status_code == 200


class TestPdfPreviewRoutes:
    """Agent 4.1 — in-app PDF citation preview: the raw-file route and the
    per-page text-highlight route. Uses real PyMuPDF-generated PDFs (a
    fake byte string wouldn't parse), written under tmp_path with
    documents.py's UPLOAD_DIR patched to point there."""

    def _seed_pdf(self, tmp_path, monkeypatch, document_id=SEEDED_DOCUMENT_ID):
        import fitz

        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("app.api.v1.routes.documents.UPLOAD_DIR", upload_dir)

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Project scope boundaries")
        doc.save(str(upload_dir / f"{document_id}.pdf"))
        doc.close()

    def test_get_document_file_returns_pdf_bytes(self, client, tmp_path, monkeypatch):
        self._seed_pdf(tmp_path, monkeypatch)
        response = client.get(f"/documents/{SEEDED_DOCUMENT_ID}/file", headers=VALID_HEADERS)
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert response.content.startswith(b"%PDF")

    def test_get_document_file_404_for_unknown_document(self, client, tmp_path, monkeypatch):
        self._seed_pdf(tmp_path, monkeypatch)
        response = client.get("/documents/does-not-exist/file", headers=VALID_HEADERS)
        assert response.status_code == 404

    def test_page_highlight_finds_text(self, client, tmp_path, monkeypatch):
        self._seed_pdf(tmp_path, monkeypatch)
        response = client.get(
            f"/documents/{SEEDED_DOCUMENT_ID}/pages/1/highlight",
            params={"text": "scope"},
            headers=VALID_HEADERS,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["page_number"] == 1
        assert body["page_width"] > 0
        assert body["page_height"] > 0
        assert len(body["rects"]) >= 1
        assert all(len(r) == 4 for r in body["rects"])

    def test_page_highlight_empty_when_text_absent(self, client, tmp_path, monkeypatch):
        self._seed_pdf(tmp_path, monkeypatch)
        response = client.get(
            f"/documents/{SEEDED_DOCUMENT_ID}/pages/1/highlight",
            params={"text": "zzzznotpresent"},
            headers=VALID_HEADERS,
        )
        assert response.status_code == 200
        assert response.json()["rects"] == []

    def test_page_highlight_404_out_of_range_page(self, client, tmp_path, monkeypatch):
        self._seed_pdf(tmp_path, monkeypatch)
        response = client.get(
            f"/documents/{SEEDED_DOCUMENT_ID}/pages/99/highlight",
            params={"text": "scope"},
            headers=VALID_HEADERS,
        )
        assert response.status_code == 404


class TestAdminUsageSummary:
    """Agent 4.2 — admin-only usage analytics. DB is disabled in this
    fixture, so the aggregation is untestable here; what IS tested is the
    permission gate (the RBAC behavior this feature adds)."""

    def test_denied_without_admin_role(self, client, monkeypatch):
        monkeypatch.setattr("app.core.auth.resolve_tenant", lambda client_name: (1, "member"))
        response = client.get("/admin/usage-summary", headers=VALID_HEADERS)
        assert response.status_code == 403

    def test_denied_with_no_role_info(self, client):
        # DB disabled: role is None — analytics defaults to denied (it has
        # no pre-RBAC history to fall back to, unlike DOCUMENT_DELETE).
        response = client.get("/admin/usage-summary", headers=VALID_HEADERS)
        assert response.status_code == 403

    def test_allowed_for_admin_returns_empty_rows_when_db_disabled(self, client, monkeypatch):
        monkeypatch.setattr("app.core.auth.resolve_tenant", lambda client_name: (1, "admin"))
        monkeypatch.setattr("app.api.v1.routes.admin.db_enabled", lambda: False)
        response = client.get("/admin/usage-summary", headers=VALID_HEADERS)
        assert response.status_code == 200
        assert response.json() == {"rows": []}
