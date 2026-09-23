"""Wraps two LLMClient implementations so a second provider is tried if
the primary fails after its own internal retries are exhausted.

This is deliberately a thin LLMClient itself — ChatService only ever sees
"an LLMClient", never knows whether it's talking to one provider or two,
consistent with the rest of the codebase depending on interfaces rather
than concrete implementations.
"""

import logging

from app.core.exceptions import LLMAPIError, LLMTimeoutError
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Both GeminiClient and GroqClient already retry internally (tenacity,
# stop_after_attempt(3)) on these two exception types. If generate() still
# raises one of them, the primary provider is genuinely down/degraded —
# that's when falling over to the secondary provider is worth it. Any
# other AppError (e.g. LLMEmptyResponseError, LLMConfigurationError) is
# not retried here — it isn't a "provider is unavailable" failure.
_FALLBACK_TRIGGERS = (LLMTimeoutError, LLMAPIError)


class FallbackLLMClient(LLMClient):
    def __init__(
        self,
        primary: LLMClient,
        primary_name: str,
        fallback: LLMClient,
        fallback_name: str,
    ):
        self._primary = primary
        self._primary_name = primary_name
        self._fallback = fallback
        self._fallback_name = fallback_name

    def generate(self, prompt: str) -> str:
        try:
            return self._primary.generate(prompt)
        except _FALLBACK_TRIGGERS as exc:
            logger.warning(
                "llm_fallback_triggered",
                extra={
                    "extra_fields": {
                        "primary_provider": self._primary_name,
                        "fallback_provider": self._fallback_name,
                        "primary_exception": str(exc),
                    }
                },
            )
            return self._fallback.generate(prompt)
