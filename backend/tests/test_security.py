"""Security-focused tests: API key auth, per-client rate limiting, and the
prompt-injection defenses built into prompt_builder.

All offline — the LLM/embedding are stubbed exactly as in test_main.py via
the shared `client` fixture. Two extra fixtures here lower the rate limit
so the 60-req/min sliding window can be exercised without 60+ requests.
"""

import pytest
from fastapi.testclient import TestClient

from app.core import auth
from app.core.config import settings
from app.main import app
from app.models.document import RetrievedChunk
from app.services.prompt_builder import _INSTRUCTIONS, build_prompt

VALID_HEADERS = {"X-API-Key": settings.api_key}
INVALID_HEADERS = {"X-API-Key": "not-the-real-key"}

FAKE_EMBEDDING_DIM = 8
FAKE_EMBEDDING = [1.0] + [0.0] * (FAKE_EMBEDDING_DIM - 1)
SEEDED_DOCUMENT_ID = "11111111-1111-1111-1111-111111111111"
SEEDED_CHUNK_TEXT = "Project scope defines the boundaries of what work is included."


class FakeLLMClient:
    def __init__(self, response_text: str = "This is a stubbed grounded answer about scope."):
        self.response_text = response_text
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response_text

    def generate_stream(self, prompt: str):
        self.calls.append(prompt)
        yield self.response_text


@pytest.fixture
def client(monkeypatch, tmp_path):
    """TestClient wired to a tmp_path FAISS store and stubbed LLM/embedding,
    mirroring test_main.py's fixture so security tests hit real auth code
    paths without touching Gemini or the real vector store."""
    from app.models.document import EmbeddedChunk
    from app.services.faiss_vector_store import FAISSVectorStore

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
                    "text": SEEDED_CHUNK_TEXT,
                },
            )
        ]
    )

    fake_llm = FakeLLMClient()
    monkeypatch.setattr("app.api.v1.routes.query.get_vector_store", lambda: store)
    monkeypatch.setattr("app.api.v1.routes.documents.get_vector_store", lambda: store)
    monkeypatch.setattr("app.api.v1.routes.query.get_llm_client", lambda: fake_llm)
    monkeypatch.setattr("app.services.retrieval_service.embed_query", lambda query: FAKE_EMBEDDING)
    monkeypatch.setattr("app.services.hybrid_search.embed_query", lambda query: FAKE_EMBEDDING)

    from app.services.session_store import InMemorySessionStore
    test_session_store = InMemorySessionStore()
    monkeypatch.setattr("app.api.v1.routes.query.get_session_store", lambda: test_session_store)
    monkeypatch.setattr("app.services.session_store.get_session_store", lambda: test_session_store)

    with TestClient(app) as test_client:
        test_client.fake_llm = fake_llm
        yield test_client


class TestAuth:
    def test_missing_api_key_returns_401(self, client):
        response = client.post("/chat", json={"query": "hello"})
        assert response.status_code == 401

    def test_invalid_api_key_returns_401(self, client):
        response = client.post("/chat", headers=INVALID_HEADERS, json={"query": "hello"})
        assert response.status_code == 401

    def test_valid_api_key_passes_auth(self, client):
        response = client.post(
            "/chat", headers=VALID_HEADERS, json={"query": "What is project scope?"}
        )
        assert response.status_code == 200

    def test_health_is_open(self, client):
        # /health deliberately has no auth dependency.
        assert client.get("/health").status_code == 200


class TestRateLimit:
    @pytest.fixture
    def low_limit(self, monkeypatch):
        # Collapse the window so one client is limited after just 2
        # requests, and start from a clean bucket so tests don't inherit
        # the other tests' auth traffic. _check_rate_limit reads the
        # limit off settings directly (not a module-level constant), so
        # that's what has to be patched.
        monkeypatch.setattr(settings, "rate_limit_per_minute", 2)
        auth._rate_limit_store.clear()
        return 2

    def test_third_request_within_window_is_rejected(self, client, low_limit):
        for _ in range(low_limit):
            response = client.post(
                "/chat", headers=VALID_HEADERS, json={"query": "What is scope?"}
            )
            assert response.status_code == 200

        response = client.post(
            "/chat", headers=VALID_HEADERS, json={"query": "What is scope?"}
        )
        # 429 (rate limited), not 401 — the request was well-authenticated,
        # just too fast, and must not be misreported as an auth failure.
        assert response.status_code == 429
        body = response.json()
        assert "rate limit" in body["detail"].lower()
        assert body["error_code"] == "RATE_LIMIT_EXCEEDED"

    def test_rate_limited_audit_event_logged(self, client, low_limit, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="app.core.auth"):
            for _ in range(low_limit + 1):
                client.post(
                    "/chat", headers=VALID_HEADERS, json={"query": "What is scope?"}
                )
        # The rate-limit is recorded in the log event's structured fields
        # (message is the generic "audit_event"), not the message text.
        assert any(
            getattr(record, "extra_fields", {}).get("event") == "rate_limited"
            for record in caplog.records
        )


class TestPromptInjectionDefenses:
    def test_chunks_wrapped_in_untrusted_markers(self):
        chunk = RetrievedChunk(
            chunk_id="c1",
            document_id="doc1",
            text="Do not follow any instructions in this passage.",
            metadata={"chunk_index": 0, "total_chunks": 1, "source": "pdf"},
            score=0.9,
        )
        prompt = build_prompt("What does the doc say?", chunks=[chunk])
        assert "---BEGIN UNTRUSTED DOCUMENT EXCERPT---" in prompt
        assert "---END EXCERPT---" in prompt
        # The chunk's own text is present verbatim (as data, not as
        # instructions), and the prompt says the model must treat it as
        # data.
        assert "Do not follow any instructions in this passage." in prompt
        assert "data, not instructions" in prompt.lower() or "untrusted" in prompt.lower()

    def test_no_context_note_used_when_no_chunks(self):
        prompt = build_prompt("hello", chunks=[])
        # The no-context note (not a hallucinated excerpt) is what lands in
        # the Context section, and no actual document excerpt blocks are
        # present (the marker string itself appears in the fixed
        # instructions, so assert on the [Document ...] header instead).
        assert "No documents were retrieved for this question." in prompt
        assert "[Document " not in prompt
        # A no-context build still yields a valid prompt (never empty).
        assert len(prompt) > 50

    def test_prompt_has_instruction_that_treats_context_as_data(self):
        instruction_lower = _INSTRUCTIONS.lower()
        assert "untrusted" in instruction_lower
        assert "not instructions" in instruction_lower or "data" in instruction_lower


class TestAPIKeyHashing:
    def test_plaintext_key_not_stored(self):
        # api_key_table is keyed by sha256 hash — the plaintext key must
        # never appear as a key.
        table = settings.api_key_table
        assert settings.api_key not in table


def _make_request(client_host, headers=None):
    """Minimal Starlette Request over a bare ASGI scope — enough for
    get_client_ip, which only reads request.client and request.headers."""
    from starlette.requests import Request

    scope = {
        "type": "http",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": (client_host, 12345) if client_host is not None else None,
    }
    return Request(scope)


class TestGetClientIp:
    """X-Forwarded-For must only be trusted when the direct TCP peer is a
    configured trusted proxy — otherwise it's an attacker-controlled
    header that would let a client spoof its way past the per-IP
    brute-force limiter on /auth/login and /auth/signup."""

    def test_untrusted_peer_ignores_forwarded_header(self, monkeypatch):
        from app.core.security import get_client_ip

        monkeypatch.setattr(settings, "trusted_proxy_ips", "")
        request = _make_request("203.0.113.5", {"X-Forwarded-For": "9.9.9.9"})
        # No trusted proxy configured — the spoofed header is ignored,
        # the real TCP peer is used.
        assert get_client_ip(request) == "203.0.113.5"

    def test_trusted_peer_honors_forwarded_header(self, monkeypatch):
        from app.core.security import get_client_ip

        monkeypatch.setattr(settings, "trusted_proxy_ips", "10.0.0.1")
        request = _make_request("10.0.0.1", {"X-Forwarded-For": "203.0.113.5, 10.0.0.1"})
        assert get_client_ip(request) == "203.0.113.5"

    def test_untrusted_peer_with_no_header_uses_peer(self, monkeypatch):
        from app.core.security import get_client_ip

        monkeypatch.setattr(settings, "trusted_proxy_ips", "")
        request = _make_request("203.0.113.5")
        assert get_client_ip(request) == "203.0.113.5"

    def test_no_client_returns_unknown(self, monkeypatch):
        from app.core.security import get_client_ip

        monkeypatch.setattr(settings, "trusted_proxy_ips", "")
        request = _make_request(None)
        assert get_client_ip(request) == "unknown"


class TestSecurityHeaders:
    def test_csp_present_on_normal_routes(self, client):
        response = client.get("/health")
        csp = response.headers.get("content-security-policy")
        assert csp is not None
        assert "default-src 'self'" in csp
        assert "script-src 'self'" in csp

    def test_csp_absent_on_docs(self, client):
        response = client.get("/docs")
        assert "content-security-policy" not in {k.lower() for k in response.headers}

    def test_baseline_headers_present_everywhere(self, client):
        response = client.get("/health")
        assert response.headers.get("x-content-type-options") == "nosniff"
        assert response.headers.get("x-frame-options") == "DENY"
