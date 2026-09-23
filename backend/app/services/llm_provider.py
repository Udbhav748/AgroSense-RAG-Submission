"""Picks and wires the LLMClient implementation(s) named in Settings.

query.py's get_llm_client() calls build_llm_client() rather than
constructing GeminiClient/GroqClient itself, so the provider choice and
optional fallback/routing wiring live in one place instead of being
duplicated wherever an LLMClient is needed.
"""

from app.core.config import settings
from app.core.exceptions import LLMConfigurationError
from app.services.fallback_llm_client import FallbackLLMClient
from app.services.gemini_client import GeminiClient
from app.services.groq_client import GroqClient
from app.services.llm_client import LLMClient
from app.services.mock_llm_client import MockLLMClient
from app.services.routing_llm_client import RoutingLLMClient

_PROVIDERS = {
    "gemini": GeminiClient,
    "groq": GroqClient,
    # Deterministic, zero-cost, zero-network stand-in -- reachable only
    # via an explicit LLM_PROVIDER=mock, never a default. Added for
    # Module 10's load/concurrency evaluation (P7) so POST /chat can be
    # benchmarked over a real HTTP/uvicorn process without live provider
    # cost/rate limits. See app/services/mock_llm_client.py.
    "mock": MockLLMClient,
}


def get_llm_client_for_provider(provider: str) -> LLMClient:
    """Construct the LLMClient for a given provider name.

    Raises LLMConfigurationError for an unrecognized provider name — the
    same exception each concrete client already raises for a missing API
    key, so callers only need to handle one error type either way.
    """
    provider_cls = _PROVIDERS.get(provider)
    if provider_cls is None:
        raise LLMConfigurationError(
            f"Unknown llm_provider '{provider}'. Expected one of: {', '.join(sorted(_PROVIDERS))}."
        )
    return provider_cls()


def build_llm_client() -> LLMClient:
    """Build the primary LLMClient from Settings, wrapped in a fallback
    provider if Settings.fallback_llm_provider is configured and differs
    from the primary, then wrapped again in a RoutingLLMClient if
    Settings.model_routing_enabled is set. Fallback and routing compose
    cleanly because both are just LLMClient implementations wrapping
    another LLMClient — callers of this function never see which (if
    any) apply.
    """
    primary = get_llm_client_for_provider(settings.llm_provider)

    if settings.fallback_llm_provider and settings.fallback_llm_provider != settings.llm_provider:
        fallback = get_llm_client_for_provider(settings.fallback_llm_provider)
        primary = FallbackLLMClient(
            primary=primary,
            primary_name=settings.llm_provider,
            fallback=fallback,
            fallback_name=settings.fallback_llm_provider,
        )

    if not settings.model_routing_enabled:
        return primary
    if settings.model_routing_complex_provider == settings.llm_provider:
        # Nothing to route to — same early-return shape as the fallback
        # check above when fallback_llm_provider == llm_provider.
        return primary

    complex_client = get_llm_client_for_provider(settings.model_routing_complex_provider)
    return RoutingLLMClient(
        simple_client=primary,
        simple_provider=settings.llm_provider,
        complex_client=complex_client,
        complex_provider=settings.model_routing_complex_provider,
    )
