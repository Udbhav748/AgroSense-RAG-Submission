"""Tests for Feature #7 — web search reliability and provider config.

Covers web_search_ready()'s force-disable-without-key behavior, the
provider dispatch (duckduckgo vs keyed providers over httpx), and the
ChatService._search_web integration (a non-ready provider degrades to []
rather than erroring, preserving the existing approval gate behavior).
"""

import app.services.rag_service as rag_service_module
import app.services.web_search_service as web_search_service_module
from app.core.config import settings
from app.models.document import WebSearchResult


def make_web_result():
    return WebSearchResult(title="Example", url="https://example.com/a", snippet="a web snippet")


class TestWebSearchReady:
    def test_duckduckgo_ready_without_key(self, monkeypatch):
        monkeypatch.setattr(settings, "web_search_provider", "duckduckgo")
        monkeypatch.setattr(settings, "web_search_api_key", "")
        assert web_search_service_module.web_search_ready() is True

    def test_keyed_provider_without_key_not_ready(self, monkeypatch, caplog):
        import logging

        monkeypatch.setattr(settings, "web_search_provider", "brave")
        monkeypatch.setattr(settings, "web_search_api_key", "")
        with caplog.at_level(logging.WARNING, logger="app.services.web_search_service"):
            assert web_search_service_module.web_search_ready() is False
        assert any(r.message == "web_search_unavailable" for r in caplog.records)

    def test_keyed_provider_with_key_ready(self, monkeypatch):
        monkeypatch.setattr(settings, "web_search_provider", "bing")
        monkeypatch.setattr(settings, "web_search_api_key", "sekret")
        assert web_search_service_module.web_search_ready() is True

    def test_unknown_provider_not_ready(self, monkeypatch):
        monkeypatch.setattr(settings, "web_search_provider", "the-internet-archive")
        assert web_search_service_module.web_search_ready() is False


class TestSearchWebDispatch:
    def test_duckduckgo_provider_returns_results(self, monkeypatch):
        monkeypatch.setattr(settings, "web_search_provider", "duckduckgo")
        monkeypatch.setattr(settings, "web_search_api_key", "")
        monkeypatch.setattr(
            web_search_service_module,
            "_search_duckduckgo",
            lambda q, max_results: [make_web_result()],
        )
        results = web_search_service_module.search_web("q")
        assert len(results) == 1
        assert results[0].url == "https://example.com/a"

    def test_force_disable_returns_empty_without_calling_provider(self, monkeypatch):
        monkeypatch.setattr(settings, "web_search_provider", "brave")
        monkeypatch.setattr(settings, "web_search_api_key", "")
        monkeypatch.setattr(
            web_search_service_module,
            "_search_keys",
            lambda q, max_results: (_ for _ in ()).throw(
                AssertionError("provider should not be called when force-disabled")
            ),
        )
        assert web_search_service_module.search_web("q") == []

    def test_keyed_provider_dispatch_parses_brave(self, monkeypatch):
        import httpx

        monkeypatch.setattr(settings, "web_search_provider", "brave")
        monkeypatch.setattr(settings, "web_search_api_key", "sekret")
        captured = {}

        def fake_get(url, headers, timeout):
            captured["headers"] = headers
            captured["url"] = url

            class FakeResp:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {"web": {"results": [{"title": "T", "url": "https://b/x", "description": "s"}]}}

            return FakeResp()

        monkeypatch.setattr(httpx, "get", fake_get)
        results = web_search_service_module.search_web("hello world")
        assert len(results) == 1
        assert results[0].title == "T"
        assert results[0].url == "https://b/x"
        assert "X-Subscription-Token" in captured["headers"]
        assert captured["headers"]["X-Subscription-Token"] == "sekret"
        assert "count=3" in captured["url"]  # uses web_search_result_count

    def test_keyed_provider_dispatch_parses_bing(self, monkeypatch):
        import httpx

        monkeypatch.setattr(settings, "web_search_provider", "bing")
        monkeypatch.setattr(settings, "web_search_api_key", "sekret")
        captured = {}

        def fake_get(url, headers, timeout):
            captured["headers"] = headers
            return type("FakeResp", (), {
                "raise_for_status": lambda self: None,
                "json": lambda self: {"webPages": {"value": [{"name": "N", "url": "https://b/y", "snippet": "sn"}]}},
            })()

        monkeypatch.setattr(httpx, "get", fake_get)
        results = web_search_service_module.search_web("q")
        assert len(results) == 1
        assert results[0].title == "N"
        assert "Ocp-Apim-Subscription-Key" in captured["headers"]

    def test_keyed_provider_http_error_becomes_web_search_error(self, monkeypatch):
        import httpx

        monkeypatch.setattr(settings, "web_search_provider", "brave")
        monkeypatch.setattr(settings, "web_search_api_key", "sekret")

        def raiser(*a, **k):
            raise httpx.ConnectError("boom")

        monkeypatch.setattr(httpx, "get", raiser)
        # Retries 3 times then raises — but the tenacity retry wraps the
        # whole decorated function; call the inner helper directly to keep
        # this test fast and focused on the error translation.
        try:
            web_search_service_module._search_keys("q", 3)
            raise AssertionError("expected WebSearchError")
        except Exception as exc:
            from app.core.exceptions import WebSearchError

            assert isinstance(exc, WebSearchError)


class TestRagServiceWebSearchIntegration:
    def make_service(self):
        from app.services.rag_service import ChatService

        class FakeStore:
            pass

        class FakeLLM:
            def generate(self, prompt):
                return "answer"

        return ChatService(FakeStore(), FakeLLM())

    def test_not_ready_provider_degrades_to_empty(self, monkeypatch, caplog):
        import logging

        monkeypatch.setattr(settings, "web_search_enabled", True)
        monkeypatch.setattr(settings, "web_search_requires_approval", False)
        monkeypatch.setattr(settings, "web_search_provider", "brave")
        monkeypatch.setattr(settings, "web_search_api_key", "")
        service = self.make_service()

        # If the provider were called it must raise — the tool treats a
        # missing key as disabled, not as a search of empty results.
        monkeypatch.setattr(
            rag_service_module,
            "search_web",
            lambda q, **k: (_ for _ in ()).throw(AssertionError("search should not run")),
        )
        with caplog.at_level(logging.WARNING, logger="app.services.web_search_service"):
            results = service._search_web("q")
        assert results == []

    def test_ready_provider_searches_normally(self, monkeypatch):
        monkeypatch.setattr(settings, "web_search_enabled", True)
        monkeypatch.setattr(settings, "web_search_requires_approval", False)
        monkeypatch.setattr(settings, "web_search_provider", "duckduckgo")
        monkeypatch.setattr(settings, "web_search_api_key", "")
        service = self.make_service()
        calls = []
        monkeypatch.setattr(
            rag_service_module, "search_web", lambda q, **k: calls.append(q) or [make_web_result()]
        )
        results = service._search_web("q")
        assert results
        assert calls == ["q"]
