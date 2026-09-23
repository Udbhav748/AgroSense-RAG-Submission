"""Tests for app/services/mock_llm_client.py -- the Module 10 P7
addition that lets POST /chat be benchmarked over a real HTTP/uvicorn
process with a deterministic, zero-cost, zero-network LLM stage.
"""

from app.core.config import settings
from app.services.llm_client import LLMClient
from app.services.llm_provider import _PROVIDERS, get_llm_client_for_provider
from app.services.mock_llm_client import MockLLMClient


class TestMockLLMClient:
    def test_implements_llm_client_interface(self):
        assert isinstance(MockLLMClient(), LLMClient)

    def test_generate_is_deterministic(self):
        client = MockLLMClient()
        assert client.generate("any prompt") == client.generate("a completely different prompt")

    def test_generate_never_echoes_the_prompt(self):
        """A benchmark workload might embed retrieved document text in the
        prompt -- the mock must never leak it back, keeping the response
        shape independent of prompt content (and, incidentally, unable to
        leak anything sensitive from the prompt)."""
        client = MockLLMClient()
        secret_looking_prompt = "CONFIDENTIAL: api_key=sk-should-never-appear"
        assert "sk-should-never-appear" not in client.generate(secret_looking_prompt)

    def test_generate_stream_yields_the_same_text_as_generate(self):
        client = MockLLMClient()
        streamed = "".join(client.generate_stream("prompt"))
        assert streamed == client.generate("prompt")

    def test_generate_structured_returns_valid_json_with_required_fields(self):
        import json

        client = MockLLMClient()
        payload = json.loads(client.generate_structured("prompt"))
        assert "answer" in payload
        assert "sources" in payload
        assert isinstance(payload["sources"], list)

    def test_no_network_calls_made(self, monkeypatch):
        """Structural guarantee: patching socket creation to raise proves
        this client makes no network I/O at all."""
        import socket

        def _forbidden(*args, **kwargs):
            raise AssertionError("MockLLMClient must never open a network socket")

        monkeypatch.setattr(socket, "socket", _forbidden)
        client = MockLLMClient()
        client.generate("prompt")
        list(client.generate_stream("prompt"))
        client.generate_structured("prompt")


class TestMockProviderRegistration:
    def test_mock_is_registered_but_never_the_default(self):
        assert "mock" in _PROVIDERS
        assert settings.llm_provider != "mock"  # never activated by default

    def test_get_llm_client_for_provider_mock_returns_mock_client(self):
        client = get_llm_client_for_provider("mock")
        assert isinstance(client, MockLLMClient)

    def test_explicit_opt_in_required(self, monkeypatch):
        """Only an explicit Settings.llm_provider == "mock" reaches this
        client -- any other configured provider does not."""
        monkeypatch.setattr(settings, "llm_provider", "groq")
        from app.services.groq_client import GroqClient

        client = get_llm_client_for_provider(settings.llm_provider)
        assert not isinstance(client, MockLLMClient)
        assert isinstance(client, GroqClient)
