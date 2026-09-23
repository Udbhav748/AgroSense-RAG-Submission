"""Gemini implementation of the LLMClient interface.

Isolated from the rest of the application: nothing outside this module
should import the google-genai SDK directly.
"""

import logging
import time
from collections.abc import Iterator
from typing import cast

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
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
                "provider": "gemini",
                "attempt": retry_state.attempt_number,
                "exception": str(retry_state.outcome.exception() if retry_state.outcome else None),
            }
        },
    )


class GeminiClient(LLMClient):
    def __init__(self) -> None:
        if not settings.gemini_api_key:
            raise LLMConfigurationError("GEMINI_API_KEY is not configured.")

        try:
            self._client = genai.Client(
                api_key=settings.gemini_api_key,
                http_options=types.HttpOptions(timeout=settings.gemini_timeout_seconds * 1000),
            )
        except Exception as exc:
            raise LLMConfigurationError(f"Failed to configure Gemini client: {exc}") from exc

    # Only retries on LLMTimeoutError/LLMAPIError (transient failure modes)
    # — not on LLMEmptyResponseError, which usually means the prompt or
    # safety filters produced no content and a retry won't help.
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
            response = self._client.models.generate_content(
                model=settings.gemini_model_name,
                contents=prompt,
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"Gemini request timed out after {settings.gemini_timeout_seconds}s: {exc}"
            ) from exc
        except genai_errors.APIError as exc:
            raise LLMAPIError(f"Gemini API request failed: {exc}") from exc
        except Exception as exc:
            raise LLMAPIError(f"Unexpected error calling Gemini: {exc}") from exc

        text = (response.text or "").strip()
        if not text:
            raise LLMEmptyResponseError("Gemini returned an empty response.")

        processing_duration = time.perf_counter() - start

        # usage_metadata can be absent for some response shapes (e.g. certain
        # safety-filtered paths); default to 0 rather than letting a missing
        # field break logging for an otherwise-successful generation.
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", None) or 0
        completion_tokens = getattr(usage, "candidates_token_count", None) or 0
        total_tokens = prompt_tokens + completion_tokens
        estimated_cost_usd = round((total_tokens / 1000) * settings.cost_per_1k_tokens, 6)

        record_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )

        get_metrics().record_llm_generation(
            provider="gemini",
            model=settings.gemini_model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )

        logger.info(
            "llm_generation_completed",
            extra={
                "extra_fields": {
                    "provider": "gemini",
                    "model_name": settings.gemini_model_name,
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
        """JSON-mode generation: response_mime_type=application/json so the
        model is constrained to emit a JSON object (the schema is enforced
        by the prompt + parse step rather than a Gemini response_schema, to
        stay provider-agnostic). Mirrors generate()'s error mapping, retry
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
                response = self._client.models.generate_content(
                    model=settings.gemini_model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                )
            except httpx.TimeoutException as exc:
                raise LLMTimeoutError(
                    f"Gemini request timed out after {settings.gemini_timeout_seconds}s: {exc}"
                ) from exc
            except genai_errors.APIError as exc:
                raise LLMAPIError(f"Gemini API request failed: {exc}") from exc
            except Exception as exc:
                raise LLMAPIError(f"Unexpected error calling Gemini: {exc}") from exc

            text = (response.text or "").strip()
            if not text:
                raise LLMEmptyResponseError("Gemini returned an empty response.")
            return text, response

        text, response = _generate_json(prompt)

        processing_duration = time.perf_counter() - start

        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", None) or 0
        completion_tokens = getattr(usage, "candidates_token_count", None) or 0
        total_tokens = prompt_tokens + completion_tokens
        estimated_cost_usd = round((total_tokens / 1000) * settings.cost_per_1k_tokens, 6)

        record_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )

        get_metrics().record_llm_generation(
            provider="gemini",
            model=settings.gemini_model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )

        logger.info(
            "llm_generation_completed",
            extra={
                "extra_fields": {
                    "provider": "gemini",
                    "model_name": settings.gemini_model_name,
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((LLMTimeoutError, LLMAPIError)),
        reraise=True,
        before_sleep=_log_retry,
    )
    def generate_with_image(
        self, prompt: str, image_bytes: bytes, mime_type: str = "image/png"
    ) -> str:
        """Multi-modal counterpart to generate(): a prompt plus one image
        sent to Gemini's native vision input (an inline_data part), used
        by image_captioning_service and vision_qa_service.

        Mirrors generate() exactly — same retry policy, same error
        mapping, same usage logging and cost tracking — the only
        difference is the request shape (a Content with the image part
        alongside the text prompt instead of a bare prompt string).
        """
        start = time.perf_counter()

        # SDK-declared param type is a Union whose list members are
        # invariant (list[Content | ContentDict | ...]), so a plain
        # list[Content] — the only type mypy can infer for a single-item
        # literal like this — never structurally matches even though every
        # element in it is a valid Content. The cast documents that this is
        # exactly the invariance false-positive mypy itself warns about,
        # not a real type mismatch.
        contents: list[types.Content] = [
            types.Content(
                parts=[
                    types.Part(inline_data=types.Blob(mime_type=mime_type, data=image_bytes)),
                    types.Part(text=prompt),
                ]
            )
        ]

        try:
            response = self._client.models.generate_content(
                model=settings.gemini_model_name,
                contents=cast("types.ContentListUnionDict", contents),
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"Gemini request timed out after {settings.gemini_timeout_seconds}s: {exc}"
            ) from exc
        except genai_errors.APIError as exc:
            raise LLMAPIError(f"Gemini API request failed: {exc}") from exc
        except Exception as exc:
            raise LLMAPIError(f"Unexpected error calling Gemini: {exc}") from exc

        text = (response.text or "").strip()
        if not text:
            raise LLMEmptyResponseError("Gemini returned an empty response to image input.")

        processing_duration = time.perf_counter() - start

        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", None) or 0
        completion_tokens = getattr(usage, "candidates_token_count", None) or 0
        total_tokens = prompt_tokens + completion_tokens
        estimated_cost_usd = round((total_tokens / 1000) * settings.cost_per_1k_tokens, 6)

        record_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )

        get_metrics().record_llm_generation(
            provider="gemini",
            model=settings.gemini_model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )

        logger.info(
            "llm_generation_completed",
            extra={
                "extra_fields": {
                    "provider": "gemini",
                    "model_name": settings.gemini_model_name,
                    "prompt_length": len(prompt),
                    "response_length": len(text),
                    "processing_duration": round(processing_duration, 4),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "estimated_cost_usd": estimated_cost_usd,
                    "image_input": True,
                    "mime_type": mime_type,
                }
            },
        )

        return text

    # Deliberately NOT wrapped in @retry like generate() is: tenacity retries
    # by re-calling the decorated function, but this is a generator — calling
    # it doesn't execute any code until iterated, so a mid-stream failure
    # happens outside the retry-wrapped call and @retry couldn't see it
    # anyway. More importantly, once tokens have started reaching a live SSE
    # client, silently restarting the whole generation would mean replaying
    # content the client already rendered — that's a caller/UX decision, not
    # something to do transparently at this layer. A failure here propagates
    # once, the same exception types generate() raises.
    def generate_stream(self, prompt: str) -> Iterator[str]:
        start = time.perf_counter()
        text_parts: list[str] = []
        last_chunk = None

        try:
            stream = self._client.models.generate_content_stream(
                model=settings.gemini_model_name,
                contents=prompt,
            )
            for chunk in stream:
                last_chunk = chunk
                piece = chunk.text or ""
                if piece:
                    text_parts.append(piece)
                    yield piece
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"Gemini request timed out after {settings.gemini_timeout_seconds}s: {exc}"
            ) from exc
        except genai_errors.APIError as exc:
            raise LLMAPIError(f"Gemini API request failed: {exc}") from exc
        except Exception as exc:
            raise LLMAPIError(f"Unexpected error calling Gemini: {exc}") from exc

        text = "".join(text_parts).strip()
        if not text:
            raise LLMEmptyResponseError("Gemini returned an empty streamed response.")

        processing_duration = time.perf_counter() - start

        # Streaming responses may only carry usage_metadata on the final
        # chunk (or not at all, depending on SDK/model) — same defensive
        # "default to 0" as the non-streaming path above, not an assumption
        # it's populated.
        usage = getattr(last_chunk, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", None) or 0
        completion_tokens = getattr(usage, "candidates_token_count", None) or 0
        total_tokens = prompt_tokens + completion_tokens
        estimated_cost_usd = round((total_tokens / 1000) * settings.cost_per_1k_tokens, 6)

        record_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )

        logger.info(
            "llm_generation_completed",
            extra={
                "extra_fields": {
                    "provider": "gemini",
                    "model_name": settings.gemini_model_name,
                    "prompt_length": len(prompt),
                    "response_length": len(text),
                    "processing_duration": round(processing_duration, 4),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "estimated_cost_usd": estimated_cost_usd,
                    "streamed": True,
                }
            },
        )
