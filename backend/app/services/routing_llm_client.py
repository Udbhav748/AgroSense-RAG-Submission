"""Routes each call to one of two LLMClient providers based on
complexity/risk signals already present in the built prompt — not a new
classification step, just inspecting what the RAG pipeline already
assembled before this client ever sees it.

Constructed by build_llm_client() (llm_provider.py) only when
Settings.model_routing_enabled is True; ChatService/ResearchAgent/
summarize_document never know they're talking to a router instead of a
single provider — the same "depend on the interface, not the
implementation" shape FallbackLLMClient already uses.

"Complex" here means either: the corrective loop already retried once
(REFLECTION_INSTRUCTION present — the first attempt didn't use its
context, a real signal this query is harder than average) or the prompt
is unusually large (Settings.model_routing_complex_prompt_chars — more
context to reason over, and more tokens billed either way, so cost and
complexity move together). Both are cheap string/length checks on a
prompt that's already fully built; no extra LLM call, no added latency.
"""

import logging
from collections.abc import Iterator

from app.core.config import settings
from app.services.llm_client import LLMClient
from app.services.prompt_builder import REFLECTION_INSTRUCTION

logger = logging.getLogger(__name__)


def _is_complex(prompt: str) -> bool:
    return (
        REFLECTION_INSTRUCTION in prompt
        or len(prompt) > settings.model_routing_complex_prompt_chars
    )


class RoutingLLMClient(LLMClient):
    def __init__(
        self,
        simple_client: LLMClient,
        simple_provider: str,
        complex_client: LLMClient,
        complex_provider: str,
    ):
        self._simple_client = simple_client
        self._simple_provider = simple_provider
        self._complex_client = complex_client
        self._complex_provider = complex_provider

    def _pick(self, prompt: str) -> tuple[LLMClient, str]:
        if _is_complex(prompt):
            return self._complex_client, self._complex_provider
        return self._simple_client, self._simple_provider

    def _log_decision(self, prompt: str, provider: str, mode: str) -> None:
        logger.info(
            "model_routing_decision",
            extra={
                "extra_fields": {
                    "provider": provider,
                    "prompt_length": len(prompt),
                    "mode": mode,
                }
            },
        )

    def generate(self, prompt: str) -> str:
        client, provider = self._pick(prompt)
        self._log_decision(prompt, provider, "generate")
        return client.generate(prompt)

    def generate_stream(self, prompt: str) -> Iterator[str]:
        client, provider = self._pick(prompt)
        self._log_decision(prompt, provider, "generate_stream")
        yield from client.generate_stream(prompt)

    def generate_structured(self, prompt: str) -> str:
        client, provider = self._pick(prompt)
        self._log_decision(prompt, provider, "generate_structured")
        return client.generate_structured(prompt)
