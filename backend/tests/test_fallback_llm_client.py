"""Unit tests for FallbackLLMClient's failover behavior."""

import pytest

from app.core.exceptions import LLMAPIError, LLMEmptyResponseError, LLMTimeoutError
from app.services.fallback_llm_client import FallbackLLMClient


class _StubClient:
    """A minimal LLMClient stub: either returns a fixed string or raises."""

    def __init__(self, result=None, raises=None):
        self._result = result
        self._raises = raises
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._result


class TestFallbackLLMClient:
    def test_primary_success_never_calls_fallback(self):
        primary = _StubClient(result="primary answer")
        fallback = _StubClient(result="fallback answer")
        client = FallbackLLMClient(primary, "gemini", fallback, "groq")

        assert client.generate("q") == "primary answer"
        assert fallback.calls == 0

    @pytest.mark.parametrize("exc_cls", [LLMTimeoutError, LLMAPIError])
    def test_primary_transient_failure_falls_over(self, exc_cls):
        primary = _StubClient(raises=exc_cls("primary down"))
        fallback = _StubClient(result="fallback answer")
        client = FallbackLLMClient(primary, "gemini", fallback, "groq")

        assert client.generate("q") == "fallback answer"
        assert primary.calls == 1
        assert fallback.calls == 1

    def test_non_transient_failure_is_not_caught(self):
        primary = _StubClient(raises=LLMEmptyResponseError("no content"))
        fallback = _StubClient(result="fallback answer")
        client = FallbackLLMClient(primary, "gemini", fallback, "groq")

        with pytest.raises(LLMEmptyResponseError):
            client.generate("q")
        assert fallback.calls == 0

    def test_both_providers_failing_propagates_fallback_error(self):
        primary = _StubClient(raises=LLMAPIError("primary down"))
        fallback = _StubClient(raises=LLMTimeoutError("fallback also down"))
        client = FallbackLLMClient(primary, "gemini", fallback, "groq")

        with pytest.raises(LLMTimeoutError):
            client.generate("q")


class TestFallbackLLMClientGenerateStream:
    """FallbackLLMClient doesn't override generate_stream — it inherits
    LLMClient's default, which yields self.generate(prompt) once. Since
    self.generate() IS FallbackLLMClient's own override, fallback behavior
    is preserved automatically; what's lost is real token-by-token
    streaming when this client is active (see LLMClient.generate_stream's
    docstring for why that trade-off is deliberate)."""

    def test_yields_primary_result_once_when_primary_succeeds(self):
        primary = _StubClient(result="primary answer")
        fallback = _StubClient(result="fallback answer")
        client = FallbackLLMClient(primary, "gemini", fallback, "groq")

        pieces = list(client.generate_stream("q"))

        assert pieces == ["primary answer"]
        assert fallback.calls == 0

    def test_falls_over_to_secondary_provider_on_transient_failure(self):
        primary = _StubClient(raises=LLMTimeoutError("primary down"))
        fallback = _StubClient(result="fallback answer")
        client = FallbackLLMClient(primary, "gemini", fallback, "groq")

        pieces = list(client.generate_stream("q"))

        assert pieces == ["fallback answer"]
        assert primary.calls == 1
        assert fallback.calls == 1
