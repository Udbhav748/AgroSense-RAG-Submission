"""Unit tests for RoutingLLMClient's per-request provider selection."""

import pytest

from app.core.config import settings
from app.services.prompt_builder import REFLECTION_INSTRUCTION
from app.services.routing_llm_client import RoutingLLMClient


class _StubClient:
    """A minimal LLMClient stub: returns a fixed string, tracking calls
    and (for the streaming test) yielding pieces instead."""

    def __init__(self, result="answer", stream_pieces=None):
        self._result = result
        self._stream_pieces = stream_pieces or [result]
        self.calls = 0
        self.stream_calls = 0
        self.structured_calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return self._result

    def generate_stream(self, prompt: str):
        self.stream_calls += 1
        yield from self._stream_pieces

    def generate_structured(self, prompt: str) -> str:
        self.structured_calls += 1
        return self._result


@pytest.fixture(autouse=True)
def _default_threshold(monkeypatch):
    # Pin the length threshold so these tests aren't sensitive to the
    # actual configured default changing later.
    monkeypatch.setattr(settings, "model_routing_complex_prompt_chars", 100)


class TestRoutingLLMClientGenerate:
    def test_short_prompt_without_reflection_uses_simple_client(self):
        simple = _StubClient(result="simple answer")
        complex_ = _StubClient(result="complex answer")
        client = RoutingLLMClient(simple, "groq", complex_, "gemini")

        assert client.generate("a short prompt") == "simple answer"
        assert simple.calls == 1
        assert complex_.calls == 0

    def test_long_prompt_uses_complex_client(self):
        simple = _StubClient(result="simple answer")
        complex_ = _StubClient(result="complex answer")
        client = RoutingLLMClient(simple, "groq", complex_, "gemini")

        long_prompt = "x" * 200  # over the pinned 100-char threshold
        assert client.generate(long_prompt) == "complex answer"
        assert simple.calls == 0
        assert complex_.calls == 1

    def test_reflection_instruction_present_uses_complex_client_even_if_short(self):
        simple = _StubClient(result="simple answer")
        complex_ = _StubClient(result="complex answer")
        client = RoutingLLMClient(simple, "groq", complex_, "gemini")

        prompt = f"short but retried\n{REFLECTION_INSTRUCTION}"
        assert client.generate(prompt) == "complex answer"
        assert complex_.calls == 1
        assert simple.calls == 0


class TestRoutingLLMClientGenerateStream:
    def test_routes_and_streams_from_the_picked_client(self):
        simple = _StubClient(stream_pieces=["a", "b"])
        complex_ = _StubClient(stream_pieces=["c", "d"])
        client = RoutingLLMClient(simple, "groq", complex_, "gemini")

        assert list(client.generate_stream("short")) == ["a", "b"]
        assert simple.stream_calls == 1
        assert complex_.stream_calls == 0

        long_prompt = "x" * 200
        assert list(client.generate_stream(long_prompt)) == ["c", "d"]
        assert complex_.stream_calls == 1


class TestRoutingLLMClientGenerateStructured:
    def test_routes_structured_calls_too(self):
        simple = _StubClient(result="{}")
        complex_ = _StubClient(result='{"answer": "x"}')
        client = RoutingLLMClient(simple, "groq", complex_, "gemini")

        long_prompt = "x" * 200
        assert client.generate_structured(long_prompt) == '{"answer": "x"}'
        assert complex_.structured_calls == 1
        assert simple.structured_calls == 0
