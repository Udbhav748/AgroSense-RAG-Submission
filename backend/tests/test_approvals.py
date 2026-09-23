"""Tests for Feature #5 — the human approval queue.

Two layers:
1. Unit tests of ApprovalStore (register / list / resolve semantics,
   LRU bounding, one-shot resolution).
2. API tests exercising GET /approvals + POST /approvals/{id}/resolve
   (auth + admin role gating), plus the wiring-in tests proving the two
   existing gates now *record* a pending approval: the delete gate under
   document_delete_requires_approval, and _search_web under
   web_search_requires_approval.
"""

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.request_context import get_client_name, set_client_name
from app.main import app
from app.services.approval_service import (
    APPROVAL_ACTION_DOCUMENT_DELETE,
    APPROVAL_ACTION_WEB_SEARCH,
    STATUS_APPROVED,
    STATUS_PENDING,
    STATUS_REJECTED,
    ApprovalStore,
)
from app.services.rag_service import ChatService

VALID_HEADERS = {"X-API-Key": settings.api_key}


class FakeVectorStore:
    pass


class FakeLLMClient:
    def generate(self, prompt):
        return "regenerated answer"


def make_service():
    return ChatService(FakeVectorStore(), FakeLLMClient())


def make_chunk(text="chunk text", score=0.9):
    from app.models.document import RetrievedChunk

    return RetrievedChunk(
        chunk_id="chunk-1", document_id="doc-1", text=text, score=score, metadata={}
    )


# ---------------------------------------------------------------------------
# Unit tests: ApprovalStore
# ---------------------------------------------------------------------------


class TestApprovalStore:
    def test_register_creates_pending_approval(self):
        store = ApprovalStore()
        approval = store.register(action=APPROVAL_ACTION_WEB_SEARCH, requested_by="alice", payload={"query": "q"})
        assert approval.approval_id
        assert approval.status == STATUS_PENDING
        assert approval.requested_by == "alice"

    def test_list_newest_first_and_filters(self):
        store = ApprovalStore()
        first = store.register(action=APPROVAL_ACTION_WEB_SEARCH, payload={"query": "q1"})
        second = store.register(action=APPROVAL_ACTION_DOCUMENT_DELETE, payload={"document_id": "d1"})
        entries = store.list_approvals()
        assert [e["approval_id"] for e in entries] == [second.approval_id, first.approval_id]
        assert len(store.list_approvals(action=APPROVAL_ACTION_WEB_SEARCH)) == 1
        assert len(store.list_approvals(status=STATUS_PENDING)) == 2

    def test_resolve_approved_and_rejected(self):
        store = ApprovalStore()
        a = store.register(action=APPROVAL_ACTION_DOCUMENT_DELETE, payload={"document_id": "d1"})
        store.resolve(a.approval_id, approved=True, resolved_by="bob")
        stored = store.get(a.approval_id)
        assert stored.status == STATUS_APPROVED
        assert stored.resolved_by == "bob"
        assert stored.resolved_at is not None

        b = store.register(action=APPROVAL_ACTION_WEB_SEARCH, payload={"query": "q"})
        store.resolve(b.approval_id, approved=False, resolved_by="bob", note="not needed")
        assert store.get(b.approval_id).status == STATUS_REJECTED
        assert store.get(b.approval_id).note == "not needed"

    def test_resolve_unknown_returns_none(self):
        store = ApprovalStore()
        assert store.resolve("nope", approved=True, resolved_by="bob") is None

    def test_resolution_is_one_shot(self):
        store = ApprovalStore()
        a = store.register(action=APPROVAL_ACTION_WEB_SEARCH)
        store.resolve(a.approval_id, approved=True, resolved_by="bob")
        # Second resolve on the now-approved entry is a no-op (still approved).
        store.resolve(a.approval_id, approved=False, resolved_by="eve")
        assert store.get(a.approval_id).status == STATUS_APPROVED

    def test_lru_eviction_bounds_memory(self):
        store = ApprovalStore(max_approvals=3)
        for i in range(5):
            store.register(action=APPROVAL_ACTION_WEB_SEARCH, payload={"i": i})
        assert len(store.list_approvals()) <= 3

    def test_negative_max_raises(self):
        with pytest.raises(ValueError):
            ApprovalStore(max_approvals=0)


# ---------------------------------------------------------------------------
# API tests: routes + gating + wiring-in
# ---------------------------------------------------------------------------


@pytest.fixture
def admin_client(monkeypatch, tmp_path):
    """TestClient with role forced to admin (DB-disabled normally gives
    role None, which would 403 the admin-gated approval routes)."""
    monkeypatch.setattr("app.core.auth.resolve_tenant", lambda client_name: (1, "admin"))
    # Fresh approval store per test so pending approvals never leak across
    # cases — mirror how the session store fixture does the same.
    fresh_store = ApprovalStore()
    monkeypatch.setattr("app.services.approval_service.get_approval_store", lambda: fresh_store)
    monkeypatch.setattr("app.api.v1.routes.approvals.get_approval_store", lambda: fresh_store)
    with TestClient(app) as test_client:
        test_client.approval_store = fresh_store
        yield test_client


@pytest.fixture
def member_client(monkeypatch, tmp_path):
    monkeypatch.setattr("app.core.auth.resolve_tenant", lambda client_name: (1, "member"))
    fresh_store = ApprovalStore()
    monkeypatch.setattr("app.services.approval_service.get_approval_store", lambda: fresh_store)
    monkeypatch.setattr("app.api.v1.routes.approvals.get_approval_store", lambda: fresh_store)
    with TestClient(app) as test_client:
        test_client.approval_store = fresh_store
        yield test_client


class TestApprovalRoutes:
    def test_list_requires_auth(self, admin_client, monkeypatch):
        monkeypatch.setattr("app.core.auth.resolve_tenant", lambda c: (1, "admin"))
        response = admin_client.get("/approvals", headers=VALID_HEADERS)
        assert response.status_code == 200

    def test_member_cannot_list(self, member_client):
        assert member_client.get("/approvals", headers=VALID_HEADERS).status_code == 403

    def test_member_cannot_resolve(self, member_client):
        entry = member_client.approval_store.register(action=APPROVAL_ACTION_WEB_SEARCH)
        response = member_client.post(
            f"/approvals/{entry.approval_id}/resolve",
            json={"approved": True},
            headers=VALID_HEADERS,
        )
        assert response.status_code == 403

    def test_list_and_resolve_roundtrip(self, admin_client):
        store = admin_client.approval_store
        entry = store.register(
            action=APPROVAL_ACTION_WEB_SEARCH, requested_by="alice", payload={"query": "what is x"}
        )

        list_resp = admin_client.get("/approvals", headers=VALID_HEADERS)
        assert list_resp.status_code == 200
        body = list_resp.json()
        assert body["count"] >= 1
        assert any(e["approval_id"] == entry.approval_id for e in body["approvals"])

        resolve_resp = admin_client.post(
            f"/approvals/{entry.approval_id}/resolve",
            json={"approved": True, "note": "granted"},
            headers=VALID_HEADERS,
        )
        assert resolve_resp.status_code == 200
        resolved = resolve_resp.json()
        assert resolved["status"] == STATUS_APPROVED
        assert resolved["note"] == "granted"

        # Listing with status filter now shows it as approved, not pending.
        filter_resp = admin_client.get(
            "/approvals", params={"status": "pending"}, headers=VALID_HEADERS
        )
        assert all(e["status"] == STATUS_PENDING for e in filter_resp.json()["approvals"])

    def test_resolve_unknown_id_404(self, admin_client):
        response = admin_client.post(
            "/approvals/nope/resolve", json={"approved": True}, headers=VALID_HEADERS
        )
        assert response.status_code == 404

    def test_filter_unknown_action_404(self, admin_client):
        assert (
            admin_client.get("/approvals", params={"action": "launch_missiles"}, headers=VALID_HEADERS).status_code
            == 404
        )

    def test_schema_compliance(self, admin_client):
        from app.models.schemas import ApprovalListResponse, ApprovalResolveResponse
        from tests.conftest import assert_matches_schema

        entry = admin_client.approval_store.register(action=APPROVAL_ACTION_WEB_SEARCH)
        list_body = admin_client.get("/approvals", headers=VALID_HEADERS).json()
        assert_matches_schema(ApprovalListResponse, list_body)
        resolve_body = admin_client.post(
            f"/approvals/{entry.approval_id}/resolve",
            json={"approved": True},
            headers=VALID_HEADERS,
        ).json()
        assert_matches_schema(ApprovalResolveResponse, resolve_body)


class TestDeleteGateRecordsApproval:
    def test_gate_on_and_not_approved_returns_400_and_records_approval(self, monkeypatch, tmp_path):
        """The existing 400 MUST be preserved (a denied request still never
        touches the vector store), but the attempt now also appears in the
        approval queue so an operator can grant it later."""
        from app.models.document import EmbeddedChunk
        from app.services.faiss_vector_store import FAISSVectorStore

        monkeypatch.setattr(settings, "document_delete_requires_approval", True)
        monkeypatch.setattr("app.core.auth.resolve_tenant", lambda c: (1, "admin"))

        fresh_store = ApprovalStore()
        monkeypatch.setattr("app.services.approval_service.get_approval_store", lambda: fresh_store)
        monkeypatch.setattr(
            "app.api.v1.routes.documents.approval_service.get_approval_store", lambda: fresh_store
        )

        store = FAISSVectorStore(
            index_path=tmp_path / "index.faiss", metadata_path=tmp_path / "metadata.json"
        )
        store.create_index(dimension=4)
        store.add_embeddings(
            [
                EmbeddedChunk(
                    chunk_id="chunk-1",
                    document_id="doc-delete-me",
                    embedding=[1.0, 0.0, 0.0, 0.0],
                    metadata={"chunk_index": 0, "total_chunks": 1, "source": "pdf", "text": "text"},
                )
            ]
        )
        monkeypatch.setattr("app.api.v1.routes.query.get_vector_store", lambda: store)
        monkeypatch.setattr("app.api.v1.routes.documents.get_vector_store", lambda: store)

        with TestClient(app) as client:
            response = client.delete(
                "/documents/doc-delete-me", params={"confirm": "true"}, headers=VALID_HEADERS
            )
        assert response.status_code == 400
        # Document untouched, as before.
        assert store.get_chunks_by_document("doc-delete-me") != []
        # ...but a pending approval now exists for the deletion.
        deletes = fresh_store.list_approvals(action=APPROVAL_ACTION_DOCUMENT_DELETE)
        assert len(deletes) == 1
        assert deletes[0]["payload"]["document_id"] == "doc-delete-me"


class TestWebSearchGateRecordsApproval:
    def test_gate_on_and_not_confirmed_records_approval(self, monkeypatch):
        import app.services.rag_service as rag_service_module

        monkeypatch.setattr(settings, "web_search_requires_approval", True)
        service = make_service()

        # The store is read lazily inside _search_web via
        # get_approval_store(); give it a fresh instance.
        fresh_store = ApprovalStore()
        monkeypatch.setattr("app.services.approval_service.get_approval_store", lambda: fresh_store)

        monkeypatch.setattr(
            rag_service_module,
            "search_web",
            lambda q, **k: (_ for _ in ()).throw(AssertionError("search should not run")),
        )
        results = service._search_web("a query")
        assert results == []
        searched = fresh_store.list_approvals(action=APPROVAL_ACTION_WEB_SEARCH)
        assert len(searched) == 1
        assert searched[0]["payload"]["query"] == "a query"

    def test_client_name_attributed_to_web_search_approval(self, monkeypatch):

        monkeypatch.setattr(settings, "web_search_requires_approval", True)
        service = make_service()
        fresh_store = ApprovalStore()
        monkeypatch.setattr("app.services.approval_service.get_approval_store", lambda: fresh_store)

        set_client_name("carol")
        try:
            results = service._search_web("a query")
        finally:
            set_client_name(None)
        assert results == []
        assert fresh_store.list_approvals(action=APPROVAL_ACTION_WEB_SEARCH)[0]["requested_by"] == "carol"

    def test_client_name_defaults_to_none_outside_request(self, monkeypatch):
        assert get_client_name() is None
