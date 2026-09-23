"""Groq implementation of the LLMClient interface.

Mirrors GeminiClient's shape deliberately: same retry policy, same log
schema (with an added "provider" field), same error mapping — so the two
providers are directly comparable in eval/run_eval.py and in production
logs, and either can be the primary or the fallback (see
fallback_llm_client.py) without special-casing.

Isolated from the rest of the application: nothing outside this module
should import the groq SDK directly.
"""

import logging
import time
from collections.abc import Iterator

try:
    import groq
except ImportError:
    groq = None  # type: ignore[assignment]
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.exceptions import (
    LLMAPIError,
    LLMConfigurationError,
    LLMEmptyResponseError,
    LLMTimeoutError,
)
from app.core.metrics import get_metrics
from app.core.usage_tracking import record_usage
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


def _log_retry(retry_state: RetryCallState) -> None:
    logger.warning(
        "llm_generation_retrying",
        extra={
            "extra_fields": {
                "provider": "groq",
                "attempt": retry_state.attempt_number,
                "exception": str(retry_state.outcome.exception() if retry_state.outcome else None),
            }
        },
    )


class GroqClient(LLMClient):
    def __init__(self) -> None:
        if not settings.groq_api_key:
            raise LLMConfigurationError("GROQ_API_KEY is not configured.")

        try:
            self._client = groq.Groq(
                api_key=settings.groq_api_key,
                timeout=settings.groq_timeout_seconds,
            )
        except Exception as exc:
            raise LLMConfigurationError(f"Failed to configure Groq client: {exc}") from exc

    # Same transient-vs-permanent split as GeminiClient: only retry on
    # timeout/API errors, not on an empty response (a retry won't fix that).
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((LLMTimeoutError, LLMAPIError)),
        reraise=True,
        before_sleep=_log_retry,
    )
    def generate(self, prompt: str) -> str:
        start = time.perf_counter()

        try:
            response = self._client.chat.completions.create(
                model=settings.groq_model_name,
                messages=[{"role": "user", "content": prompt}],
            )
        except groq.APITimeoutError as exc:
            raise LLMTimeoutError(
                f"Groq request timed out after {settings.groq_timeout_seconds}s: {exc}"
            ) from exc
        except groq.APIError as exc:
            # Covers RateLimitError, APIStatusError, APIConnectionError, etc.
            # — all groq SDK exceptions subclass APIError.
            raise LLMAPIError(f"Groq API request failed: {exc}") from exc
        except Exception as exc:
            raise LLMAPIError(f"Unexpected error calling Groq: {exc}") from exc

        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise LLMEmptyResponseError("Groq returned an empty response.")

        processing_duration = time.perf_counter() - start

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) or 0
        completion_tokens = getattr(usage, "completion_tokens", None) or 0
        total_tokens = prompt_tokens + completion_tokens
        estimated_cost_usd = round((total_tokens / 1000) * settings.groq_cost_per_1k_tokens, 6)

        record_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )

        get_metrics().record_llm_generation(
            provider="groq",
            model=settings.groq_model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )

        logger.info(
            "llm_generation_completed",
            extra={
                "extra_fields": {
                    "provider": "groq",
                    "model_name": settings.groq_model_name,
                    "prompt_length": len(prompt),
                    "response_length": len(text),
                    "processing_duration": round(processing_duration, 4),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "estimated_cost_usd": estimated_cost_usd,
                }
            },
        )

        return text

    def generate_structured(self, prompt: str) -> str:
        """JSON-mode generation: response_format json_object so the model is
        constrained to emit valid JSON (the schema is enforced by the prompt
        + parse step rather than a provider schema, to stay
        provider-agnostic). Mirrors generate()'s error mapping, retry
        policy, usage logging, and cost tracking."""
        start = time.perf_counter()

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((LLMTimeoutError, LLMAPIError)),
            reraise=True,
            before_sleep=_log_retry,
        )
        def _generate_json(prompt: str) -> tuple[str, object]:
            try:
                response = self._client.chat.completions.create(
                    model=settings.groq_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
            except groq.APITimeoutError as exc:
                raise LLMTimeoutError(
                    f"Groq request timed out after {settings.groq_timeout_seconds}s: {exc}"
                ) from exc
            except groq.APIError as exc:
                raise LLMAPIError(f"Groq API request failed: {exc}") from exc
            except Exception as exc:
                raise LLMAPIError(f"Unexpected error calling Groq: {exc}") from exc

            text = (response.choices[0].message.content or "").strip()
            if not text:
                raise LLMEmptyResponseError("Groq returned an empty response.")
            return text, response

        text, response = _generate_json(prompt)

        processing_duration = time.perf_counter() - start

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) or 0
        completion_tokens = getattr(usage, "completion_tokens", None) or 0
        total_tokens = prompt_tokens + completion_tokens
        estimated_cost_usd = round((total_tokens / 1000) * settings.groq_cost_per_1k_tokens, 6)

        record_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )

        get_metrics().record_llm_generation(
            provider="groq",
            model=settings.groq_model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )

        logger.info(
            "llm_generation_completed",
            extra={
                "extra_fields": {
                    "provider": "groq",
                    "model_name": settings.groq_model_name,
                    "prompt_length": len(prompt),
                    "response_length": len(text),
                    "processing_duration": round(processing_duration, 4),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "estimated_cost_usd": estimated_cost_usd,
                    "structured": True,
                }
            },
        )

        return text

    # Not wrapped in @retry — same reasoning as GeminiClient.generate_stream:
    # a generator's body doesn't run until iterated, so @retry re-calling
    # this function wouldn't see a failure that happens mid-iteration, and
    # silently restarting a generation that's already partially reached a
    # live client isn't something to do transparently at this layer.
    def generate_stream(self, prompt: str) -> Iterator[str]:
        start = time.perf_counter()
        text_parts: list[str] = []

        try:
            stream = self._client.chat.completions.create(
                model=settings.groq_model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    text_parts.append(delta)
                    yield delta
        except groq.APITimeoutError as exc:
            raise LLMTimeoutError(
                f"Groq request timed out after {settings.groq_timeout_seconds}s: {exc}"
            ) from exc
        except groq.APIError as exc:
            raise LLMAPIError(f"Groq API request failed: {exc}") from exc
        except Exception as exc:
            raise LLMAPIError(f"Unexpected error calling Groq: {exc}") from exc

        text = "".join(text_parts).strip()
        if not text:
            raise LLMEmptyResponseError("Groq returned an empty streamed response.")

        processing_duration = time.perf_counter() - start

        # Groq's streamed chunks don't carry usage/token counts the way the
        # non-streaming response does — token/cost logging is skipped here
        # rather than guessed at (see the non-streaming generate() above for
        # where that accounting does happen).
        logger.info(
            "llm_generation_completed",
            extra={
                "extra_fields": {
                    "provider": "groq",
                    "model_name": settings.groq_model_name,
                    "prompt_length": len(prompt),
                    "response_length": len(text),
                    "processing_duration": round(processing_duration, 4),
                    "streamed": True,
                }
            },
        )
