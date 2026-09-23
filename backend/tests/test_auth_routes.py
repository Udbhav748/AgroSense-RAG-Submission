"""End-to-end tests for POST /auth/signup, POST /auth/login, GET /auth/me
(app/api/v1/routes/auth.py) and the require_auth dependency's JWT path
(app/core/auth.py).

Mocks at the same service-function boundary the rest of this suite
already uses for DB-touching logic (see test_main.py's tenant-ownership
tests) — user_service.create_user/authenticate_user/get_user_by_id are
patched directly rather than requiring a real Postgres instance.
"""

import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import EmailAlreadyRegisteredError, UnauthorizedError
from app.main import app
from app.models.document import EmbeddedChunk
from app.services.faiss_vector_store import FAISSVectorStore

FAKE_EMBEDDING_DIM = 8
FAKE_EMBEDDING = [1.0] + [0.0] * (FAKE_EMBEDDING_DIM - 1)


@pytest.fixture
def client(monkeypatch, tmp_path):
    """A TestClient with an empty-but-nonzero vector store (so the
    lifespan's seed_if_empty is a fast no-op) — auth routes don't touch
    retrieval at all, so nothing else needs stubbing."""
    store = FAISSVectorStore(index_path=tmp_path / "index.faiss", metadata_path=tmp_path / "metadata.json")
    store.create_index(dimension=FAKE_EMBEDDING_DIM)
    store.add_embeddings(
        [EmbeddedChunk(chunk_id="c1", document_id="d1", embedding=FAKE_EMBEDDING, metadata={"text": "x"})]
    )
    monkeypatch.setattr("app.api.v1.routes.query.get_vector_store", lambda: store)
    monkeypatch.setattr("app.api.v1.routes.documents.get_vector_store", lambda: store)

    with TestClient(app) as test_client:
        yield test_client


class TestSignup:
    def test_missing_consent_returns_422(self, client):
        response = client.post(
            "/auth/signup",
            json={"email": "a@example.com", "password": "correct-horse-battery"},
        )
        assert response.status_code == 422

    def test_consent_false_returns_422(self, client):
        response = client.post(
            "/auth/signup",
            json={"email": "a@example.com", "password": "correct-horse-battery", "consent": False},
        )
        assert response.status_code == 422

    def test_successful_signup_returns_token(self, client, monkeypatch):
        monkeypatch.setattr("app.api.v1.routes.auth.create_user", lambda email, password: (1, 10, "member"))
        monkeypatch.setattr(
            "app.api.v1.routes.auth.create_access_token", lambda user_id, email: "fake-jwt-token"
        )

        response = client.post(
            "/auth/signup",
            json={"email": "a@example.com", "password": "correct-horse-battery", "consent": True},
        )

        assert response.status_code == 201
        assert response.json() == {"access_token": "fake-jwt-token", "token_type": "bearer"}

    def test_duplicate_email_returns_409(self, client, monkeypatch):
        def _raise(email, password):
            raise EmailAlreadyRegisteredError(f"An account with email {email} already exists.")

        monkeypatch.setattr("app.api.v1.routes.auth.create_user", _raise)

        response = client.post(
            "/auth/signup",
            json={"email": "a@example.com", "password": "correct-horse-battery", "consent": True},
        )

        assert response.status_code == 409


class TestLogin:
    def test_successful_login_returns_token(self, client, monkeypatch):
        monkeypatch.setattr(
            "app.api.v1.routes.auth.authenticate_user", lambda email, password: (1, 10, "member")
        )
        monkeypatch.setattr(
            "app.api.v1.routes.auth.create_access_token", lambda user_id, email: "fake-jwt-token"
        )

        response = client.post("/auth/login", json={"email": "a@example.com", "password": "correct-horse"})

        assert response.status_code == 200
        assert response.json()["access_token"] == "fake-jwt-token"

    def test_wrong_password_returns_401(self, client, monkeypatch):
        def _raise(email, password):
            raise UnauthorizedError("Invalid email or password.")

        monkeypatch.setattr("app.api.v1.routes.auth.authenticate_user", _raise)

        response = client.post("/auth/login", json={"email": "a@example.com", "password": "wrong"})

        assert response.status_code == 401


class TestLogout:
    def test_jwt_caller_revokes_and_returns_204(self, client, monkeypatch):
        monkeypatch.setattr("app.core.auth.decode_access_token", lambda token: 1)
        monkeypatch.setattr(
            "app.core.auth.get_user_by_id", lambda user_id: ("a@example.com", 10, "member")
        )
        calls = []
        monkeypatch.setattr(
            "app.api.v1.routes.auth.revoke_all_tokens", lambda user_id: calls.append(user_id)
        )

        response = client.post("/auth/logout", headers={"Authorization": "Bearer fake-jwt-token"})

        assert response.status_code == 204
        assert calls == [1]

    def test_api_key_caller_is_a_noop(self, client, monkeypatch):
        # No token to revoke on the API-key path (request.state.user_id
        # is None there — see require_auth) — must not error or call
        # revoke_all_tokens with a nonsensical id.
        from app.core.config import settings

        calls = []
        monkeypatch.setattr(
            "app.api.v1.routes.auth.revoke_all_tokens", lambda user_id: calls.append(user_id)
        )

        response = client.post("/auth/logout", headers={"X-API-Key": settings.api_key})

        assert response.status_code == 204
        assert calls == []

    def test_no_credentials_returns_401(self, client):
        response = client.post("/auth/logout")
        assert response.status_code == 401


class TestMe:
    def test_valid_bearer_token_returns_identity(self, client, monkeypatch):
        monkeypatch.setattr("app.core.auth.decode_access_token", lambda token: 1)
        monkeypatch.setattr(
            "app.core.auth.get_user_by_id", lambda user_id: ("a@example.com", 10, "member")
        )

        response = client.get("/auth/me", headers={"Authorization": "Bearer fake-jwt-token"})

        assert response.status_code == 200
        assert response.json() == {"email": "a@example.com", "tenant_id": 10, "role": "member"}

    def test_invalid_bearer_token_returns_401(self, client, monkeypatch):
        monkeypatch.setattr("app.core.auth.decode_access_token", lambda token: None)

        response = client.get("/auth/me", headers={"Authorization": "Bearer garbage"})

        assert response.status_code == 401

    def test_no_credentials_returns_401(self, client):
        response = client.get("/auth/me")
        assert response.status_code == 401

    def test_bearer_takes_precedence_over_api_key(self, client, monkeypatch):
        # A request carrying both a valid Bearer token and a valid
        # X-API-Key resolves via the JWT path — Bearer is checked first,
        # per require_auth's own precedence (see app/core/auth.py).
        from app.core.config import settings

        monkeypatch.setattr("app.core.auth.decode_access_token", lambda token: 1)
        monkeypatch.setattr(
            "app.core.auth.get_user_by_id", lambda user_id: ("a@example.com", 10, "member")
        )

        response = client.get(
            "/auth/me",
            headers={"Authorization": "Bearer fake-jwt-token", "X-API-Key": settings.api_key},
        )

        assert response.status_code == 200
        assert response.json()["email"] == "a@example.com"


class TestRolePermissionParityAcrossAuthMethods:
    """permissions.py's check_permission() only ever reads
    request.state.role — it has no idea whether that role came from an
    API key (tenant_service.resolve_tenant) or a JWT
    (user_service.get_user_by_id). This class proves that structural
    guarantee holds end-to-end via the JWT path specifically, mirroring
    test_main.py's TestRoleBasedAccessControl assertions for the
    API-key path exactly — same route, same expected status codes, only
    the auth method differs."""

    def test_jwt_admin_passes_the_permission_gate(self, client, monkeypatch):
        monkeypatch.setattr("app.core.auth.decode_access_token", lambda token: 1)
        monkeypatch.setattr("app.core.auth.get_user_by_id", lambda user_id: ("admin@example.com", 10, "admin"))

        response = client.get(
            "/admin/usage-summary",
            headers={"Authorization": "Bearer fake-jwt-token"},
        )

        assert response.status_code == 200

    def test_jwt_member_is_denied_the_same_way_api_key_member_is(self, client, monkeypatch):
        monkeypatch.setattr("app.core.auth.decode_access_token", lambda token: 1)
        monkeypatch.setattr("app.core.auth.get_user_by_id", lambda user_id: ("member@example.com", 10, "member"))

        response = client.get(
            "/admin/usage-summary",
            headers={"Authorization": "Bearer fake-jwt-token"},
        )

        assert response.status_code == 403
