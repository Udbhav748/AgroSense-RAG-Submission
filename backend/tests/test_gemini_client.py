"""Unit tests for GeminiClient.generate_stream (real token streaming).

generate()'s own behavior has no existing dedicated test file — out of
scope here. genai.Client() is constructed for real (it just builds an SDK
client object, no network call, same assumption test_groq_client.py makes
about groq.Groq()), then self._client.models.generate_content_stream is
monkeypatched per test.
"""

import httpx
import pytest
from google.genai import errors as genai_errors

from app.core.config import settings
from app.core.exceptions import LLMAPIError, LLMEmptyResponseError, LLMTimeoutError
from app.services.gemini_client import GeminiClient


class _FakeChunk:
    def __init__(self, text, usage_metadata=None):
        self.text = text
        self.usage_metadata = usage_metadata


class _FakeUsage:
    def __init__(self, prompt_token_count=10, candidates_token_count=5):
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count


def _client(monkeypatch, gemini_api_key="test-gemini-key"):
    monkeypatch.setattr(settings, "gemini_api_key", gemini_api_key)
    return GeminiClient()


class TestGeminiClientGenerateStream:
    def test_happy_path_yields_pieces_in_order(self, monkeypatch):
        client = _client(monkeypatch)
        chunks = [_FakeChunk("Hello "), _FakeChunk("grounded "), _FakeChunk("world.", usage_metadata=_FakeUsage())]
        monkeypatch.setattr(
            client._client.models, "generate_content_stream", lambda **kwargs: iter(chunks)
        )

        pieces = list(client.generate_stream("some prompt"))

        assert pieces == ["Hello ", "grounded ", "world."]

    def test_empty_stream_raises_llm_empty_response_error(self, monkeypatch):
        client = _client(monkeypatch)
        monkeypatch.setattr(
            client._client.models, "generate_content_stream", lambda **kwargs: iter([_FakeChunk("")])
        )

        with pytest.raises(LLMEmptyResponseError):
            list(client.generate_stream("some prompt"))

    def test_timeout_during_iteration_raises_llm_timeout_error(self, monkeypatch):
        client = _client(monkeypatch)

        def _raising_stream(**kwargs):
            def _gen():
                yield _FakeChunk("partial ")
                raise httpx.TimeoutException("timed out")

            return _gen()

        monkeypatch.setattr(client._client.models, "generate_content_stream", _raising_stream)

        with pytest.raises(LLMTimeoutError):
            list(client.generate_stream("some prompt"))

    def test_api_error_during_iteration_raises_llm_api_error(self, monkeypatch):
        client = _client(monkeypatch)

        def _raising_stream(**kwargs):
            def _gen():
                yield _FakeChunk("partial ")
                raise genai_errors.APIError("500", {"error": {"message": "boom"}})

            return _gen()

        monkeypatch.setattr(client._client.models, "generate_content_stream", _raising_stream)

        with pytest.raises(LLMAPIError):
            list(client.generate_stream("some prompt"))

    def test_failure_before_any_piece_still_raises_mapped_error(self, monkeypatch):
        client = _client(monkeypatch)

        def _raise_immediately(**kwargs):
            raise httpx.TimeoutException("timed out before first chunk")

        monkeypatch.setattr(client._client.models, "generate_content_stream", _raise_immediately)

        with pytest.raises(LLMTimeoutError):
            list(client.generate_stream("some prompt"))
