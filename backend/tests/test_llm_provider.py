"""Unit tests for the provider factory and fallback-wiring logic in
app/services/llm_provider.py.

Constructing GeminiClient/GroqClient here is safe offline — both SDK
clients validate their config and build an HTTP client object without
making a network call.
"""

import pytest

from app.core.config import settings
from app.core.exceptions import LLMConfigurationError
from app.services.fallback_llm_client import FallbackLLMClient
from app.services.gemini_client import GeminiClient
from app.services.groq_client import GroqClient
from app.services.llm_provider import build_llm_client, get_llm_client_for_provider
from app.services.routing_llm_client import RoutingLLMClient


class TestGetLLMClientForProvider:
    def test_gemini_returns_gemini_client(self):
        assert isinstance(get_llm_client_for_provider("gemini"), GeminiClient)

    def test_groq_returns_groq_client(self, monkeypatch):
        monkeypatch.setattr(settings, "groq_api_key", "test-groq-key")
        assert isinstance(get_llm_client_for_provider("groq"), GroqClient)

    def test_unknown_provider_raises_configuration_error(self):
        with pytest.raises(LLMConfigurationError):
            get_llm_client_for_provider("not-a-real-provider")


class TestBuildLLMClient:
    def test_no_fallback_configured_returns_bare_primary(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "gemini")
        monkeypatch.setattr(settings, "fallback_llm_provider", None)
        client = build_llm_client()
        assert isinstance(client, GeminiClient)
        assert not isinstance(client, FallbackLLMClient)

    def test_fallback_same_as_primary_returns_bare_primary(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "gemini")
        monkeypatch.setattr(settings, "fallback_llm_provider", "gemini")
        client = build_llm_client()
        assert isinstance(client, GeminiClient)
        assert not isinstance(client, FallbackLLMClient)

    def test_different_fallback_wraps_in_fallback_client(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "gemini")
        monkeypatch.setattr(settings, "fallback_llm_provider", "groq")
        monkeypatch.setattr(settings, "groq_api_key", "test-groq-key")
        client = build_llm_client()
        assert isinstance(client, FallbackLLMClient)
        assert isinstance(client._primary, GeminiClient)
        assert isinstance(client._fallback, GroqClient)


class TestBuildLLMClientRouting:
    def test_routing_disabled_by_default_returns_bare_primary(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "gemini")
        monkeypatch.setattr(settings, "fallback_llm_provider", None)
        monkeypatch.setattr(settings, "model_routing_enabled", False)
        client = build_llm_client()
        assert isinstance(client, GeminiClient)
        assert not isinstance(client, RoutingLLMClient)

    def test_routing_enabled_with_different_complex_provider_wraps_router(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "groq")
        monkeypatch.setattr(settings, "groq_api_key", "test-groq-key")
        monkeypatch.setattr(settings, "fallback_llm_provider", None)
        monkeypatch.setattr(settings, "model_routing_enabled", True)
        monkeypatch.setattr(settings, "model_routing_complex_provider", "gemini")
        client = build_llm_client()
        assert isinstance(client, RoutingLLMClient)
        assert isinstance(client._simple_client, GroqClient)
        assert isinstance(client._complex_client, GeminiClient)

    def test_routing_enabled_with_same_complex_provider_is_a_no_op(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "gemini")
        monkeypatch.setattr(settings, "fallback_llm_provider", None)
        monkeypatch.setattr(settings, "model_routing_enabled", True)
        monkeypatch.setattr(settings, "model_routing_complex_provider", "gemini")
        client = build_llm_client()
        assert isinstance(client, GeminiClient)
        assert not isinstance(client, RoutingLLMClient)

    def test_routing_composes_with_fallback(self, monkeypatch):
        # fallback wraps the primary first, then routing wraps that —
        # both are just LLMClient implementations, so this should compose
        # without either needing to know about the other.
        monkeypatch.setattr(settings, "llm_provider", "groq")
        monkeypatch.setattr(settings, "groq_api_key", "test-groq-key")
        monkeypatch.setattr(settings, "fallback_llm_provider", "gemini")
        monkeypatch.setattr(settings, "model_routing_enabled", True)
        monkeypatch.setattr(settings, "model_routing_complex_provider", "gemini")
        client = build_llm_client()
        assert isinstance(client, RoutingLLMClient)
        assert isinstance(client._simple_client, FallbackLLMClient)
        assert isinstance(client._complex_client, GeminiClient)
