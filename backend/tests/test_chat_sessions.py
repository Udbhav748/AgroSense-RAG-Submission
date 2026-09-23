"""End-to-end tests for GET /chat/sessions, GET /chat/sessions/{id}, and
DELETE /chat/sessions/{id} (app/api/v1/routes/query.py) — the
history-list/browse/resume feature.

Ownership checks are exercised the same way test_main.py's cross-tenant
document-delete tests already do: patch app.core.auth.resolve_tenant
(what require_api_key/require_auth calls) to give the requester a known
tenant_id, and patch get_session_owner directly, rather than requiring a
real Postgres instance.
"""

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.models.document import EmbeddedChunk
from app.services.faiss_vector_store import FAISSVectorStore
from app.services.session_store import InMemorySessionStore

VALID_HEADERS = {"X-API-Key": settings.api_key}
FAKE_EMBEDDING_DIM = 8
FAKE_EMBEDDING = [1.0] + [0.0] * (FAKE_EMBEDDING_DIM - 1)


@pytest.fixture
def client(monkeypatch, tmp_path):
    store = FAISSVectorStore(index_path=tmp_path / "index.faiss", metadata_path=tmp_path / "metadata.json")
    store.create_index(dimension=FAKE_EMBEDDING_DIM)
    store.add_embeddings(
        [EmbeddedChunk(chunk_id="c1", document_id="d1", embedding=FAKE_EMBEDDING, metadata={"text": "x"})]
    )
    monkeypatch.setattr("app.api.v1.routes.query.get_vector_store", lambda: store)
    monkeypatch.setattr("app.api.v1.routes.documents.get_vector_store", lambda: store)

    session_store = InMemorySessionStore()
    monkeypatch.setattr("app.api.v1.routes.query.get_session_store", lambda: session_store)

    with TestClient(app) as test_client:
        test_client.session_store = session_store
        yield test_client


class TestListChatSessions:
    def test_empty_when_db_disabled(self, client, monkeypatch):
        monkeypatch.setattr("app.api.v1.routes.query.list_sessions", lambda tenant_id: [])
        response = client.get("/chat/sessions", headers=VALID_HEADERS)
        assert response.status_code == 200
        assert response.json() == {"sessions": [], "total": 0}

    def test_returns_mocked_sessions(self, client, monkeypatch):
        canned = [{"session_id": "s1", "title": "hello", "created_at": None, "last_accessed_at": None}]
        monkeypatch.setattr("app.api.v1.routes.query.list_sessions", lambda tenant_id: canned)

        response = client.get("/chat/sessions", headers=VALID_HEADERS)

        assert response.status_code == 200
        assert response.json() == {"sessions": canned, "total": 1}


class TestGetChatSession:
    def test_unknown_session_returns_404(self, client):
        response = client.get("/chat/sessions/does-not-exist", headers=VALID_HEADERS)
        assert response.status_code == 404

    def test_existing_session_returns_turns(self, client):
        session_id = client.session_store.create_session()
        client.session_store.append_turn(session_id, "user", "hi")
        client.session_store.append_turn(session_id, "assistant", "hello")

        response = client.get(f"/chat/sessions/{session_id}", headers=VALID_HEADERS)

        assert response.status_code == 200
        payload = response.json()
        assert payload["session_id"] == session_id
        assert payload["turns"] == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_cross_tenant_access_returns_404(self, client, monkeypatch):
        session_id = client.session_store.create_session()
        client.session_store.append_turn(session_id, "user", "hi")

        # Requester resolves to tenant 1; the session's real owner is
        # tenant 2 — a genuine mismatch, not just "unknown."
        monkeypatch.setattr("app.core.auth.resolve_tenant", lambda client_name: (1, "admin"))
        monkeypatch.setattr("app.api.v1.routes.query.get_session_owner", lambda session_id: 2)

        response = client.get(f"/chat/sessions/{session_id}", headers=VALID_HEADERS)

        # 404, not 403 — same "non-owner sees the same response as
        # doesn't exist" convention documents.py's delete route uses.
        assert response.status_code == 404


class TestDeleteChatSession:
    def test_deleting_existing_session_returns_deleted(self, client):
        session_id = client.session_store.create_session()

        response = client.delete(f"/chat/sessions/{session_id}", headers=VALID_HEADERS)

        assert response.status_code == 200
        assert response.json() == {"status": "deleted", "session_id": session_id}
        assert client.session_store.session_exists(session_id) is False

    def test_deleting_unknown_session_returns_not_found(self, client):
        response = client.delete("/chat/sessions/does-not-exist", headers=VALID_HEADERS)

        assert response.status_code == 200
        assert response.json() == {"status": "not_found", "session_id": "does-not-exist"}

    def test_cross_tenant_delete_denied_and_session_survives(self, client, monkeypatch):
        session_id = client.session_store.create_session()

        monkeypatch.setattr("app.core.auth.resolve_tenant", lambda client_name: (1, "admin"))
        monkeypatch.setattr("app.api.v1.routes.query.get_session_owner", lambda session_id: 2)

        response = client.delete(f"/chat/sessions/{session_id}", headers=VALID_HEADERS)

        assert response.status_code == 404
        # A denied request must never have touched the store.
        assert client.session_store.session_exists(session_id) is True
