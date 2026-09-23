"""Unit tests for the human-approval gate on web search and the
JSON-mode structured-output path (rag_service.py + structured_output.py).

- Approval gate: with Settings.web_search_requires_approval on, _search_web
  returns [] and logs web_search_skipped_pending_approval unless the client
  sent confirm_web_search=true; with it off (default), behavior is unchanged.
- Structured output: _generate_structured uses generate_structured() and
  returns the validated `answer` field; on an unparseable response it falls
  back to the plain-text _generate path instead of raising.

Uses the same lightweight fakes as test_rag_service.py.
"""

import app.services.rag_service as rag_service_module
from app.core.config import settings
from app.models.document import RetrievedChunk, WebSearchResult
from app.services.rag_service import ChatService
from app.services.structured_output import parse_structured_answer


class FakeLLMClient:
    """Returns `response` for generate(), and `structured` for
    generate_structured() when provided (else the same response)."""

    def __init__(self, response="regenerated answer", structured=None):
        self.response = response
        self.structured = structured if structured is not None else response
        self.generate_calls = []
        self.structured_calls = []

    def generate(self, prompt):
        self.generate_calls.append(prompt)
        return self.response

    def generate_structured(self, prompt):
        self.structured_calls.append(prompt)
        return self.structured


class FakeVectorStore:
    pass


def make_service(llm_client=None):
    return ChatService(FakeVectorStore(), llm_client or FakeLLMClient())


def make_chunk():
    return RetrievedChunk(
        chunk_id="chunk-1", document_id="doc-1", text="chunk text", score=0.9, metadata={}
    )


class TestWebSearchApprovalGate:
    def test_gate_off_by_default_runs_search(self, monkeypatch):
        monkeypatch.setattr(settings, "web_search_requires_approval", False)
        service = make_service()
        calls = []
        monkeypatch.setattr(
            rag_service_module, "search_web", lambda q, **k: calls.append(q) or [make_web_result()]
        )
        results = service._search_web("a query")
        assert results  # search ran
        assert calls == ["a query"]

    def test_gate_on_and_not_confirmed_skips_search(self, monkeypatch, caplog):
        monkeypatch.setattr(settings, "web_search_requires_approval", True)
        service = make_service()
        monkeypatch.setattr(
            rag_service_module,
            "search_web",
            lambda q, **k: (_ for _ in ()).throw(AssertionError("search should not run")),
        )
        import logging

        with caplog.at_level(logging.INFO, logger="app.services.rag_service"):
            results = service._search_web("a query")
        assert results == []
        assert any(r.message == "web_search_skipped_pending_approval" for r in caplog.records)

    def test_gate_on_and_confirmed_runs_search(self, monkeypatch):
        monkeypatch.setattr(settings, "web_search_requires_approval", True)
        service = make_service()
        calls = []
        monkeypatch.setattr(
            rag_service_module, "search_web", lambda q, **k: calls.append(q) or [make_web_result()]
        )
        results = service._search_web("a query", confirm_web_search=True)
        assert results
        assert calls == ["a query"]


def make_web_result():
    return WebSearchResult(title="Example", url="https://example.com/a", snippet="a web snippet")


class TestStructuredOutput:
    def test_valid_json_uses_generate_structured_and_returns_answer(self, monkeypatch):
        monkeypatch.setattr(settings, "structured_output_enabled", True)
        llm_client = FakeLLMClient(structured='{"answer": "the structured answer", "sources": ["doc-1"]}')
        service = make_service(llm_client)
        answer, structured_payload = service._generate_structured("q", [make_chunk()], None)
        assert answer == "the structured answer"
        assert structured_payload == {"answer": "the structured answer", "sources": ["doc-1"]}
        assert llm_client.structured_calls  # JSON-mode was used
        assert not llm_client.generate_calls  # no fallback needed

    def test_unparseable_json_falls_back_to_plain_text(self, monkeypatch, caplog):
        monkeypatch.setattr(settings, "structured_output_enabled", True)
        llm_client = FakeLLMClient(response="plain fallback", structured="not json at all")
        service = make_service(llm_client)
        import logging

        with caplog.at_level(logging.INFO, logger="app.services.rag_service"):
            answer, structured_payload = service._generate_structured("q", [make_chunk()], None)
        assert answer == "plain fallback"
        assert structured_payload is None  # fallback must never be reported as a successful structured response
        assert llm_client.structured_calls
        assert llm_client.generate_calls  # fell back
        assert any(r.message == "structured_output_fallback" for r in caplog.records)


class TestParseStructuredAnswer:
    def test_plain_json(self):
        parsed = parse_structured_answer('{"answer": "hello", "sources": ["d1"]}')
        assert parsed is not None
        assert parsed.answer == "hello"
        assert parsed.sources == ["d1"]

    def test_fenced_json(self):
        parsed = parse_structured_answer('```json\n{"answer": "fenced"}\n```')
        assert parsed is not None
        assert parsed.answer == "fenced"

    def test_json_with_trailing_prose(self):
        parsed = parse_structured_answer('{"answer": "block"} hope this helps')
        assert parsed is not None
        assert parsed.answer == "block"

    def test_invalid_returns_none(self):
        assert parse_structured_answer("not json") is None
        assert parse_structured_answer("   ") is None

    def test_schema_mismatch_returns_none(self):
        assert parse_structured_answer('{"foo": "bar"}') is None

    def test_empty_answer_returns_none(self):
        assert parse_structured_answer('{"answer": "   "}') is None
