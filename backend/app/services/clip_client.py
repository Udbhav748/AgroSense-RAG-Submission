"""HTTP client for the CLIP embedding microservice (clip_service/).

The CLIP service keeps its own transformers/torch stack in its own
process; AgroSense-RAG never imports torch or CLIP code — this module makes
plain HTTP calls, the same isolation vision_client.py uses for LeafSense.

Contract (docs/MULTIUSER_MULTIMODAL_PLAN.md Phase 4):

- POST /embed/text   body {"text": "..."} -> {"embedding": [...], "dimension": N}
- POST /embed/image  multipart field "file" -> {"embedding": [...], "dimension": N}

Both embeddings are L2-normalized by the service, so inner product across
text <-> image vectors equals cosine similarity — which is what lets a
CLIP text embedding of a query directly rank CLIP image embeddings in the
image FAISS index. This client does no normalization of its own; the
service owns the model and its output contract.
"""

import logging
import time
from typing import Any

import httpx
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from app.core.config import settings
from app.core.exceptions import ClipServiceError
from app.models.document import ClipEmbedding

logger = logging.getLogger(__name__)


def _log_retry(retry_state: RetryCallState) -> None:
    # .outcome is a property, not a plain attribute -- assign it to a local
    # so mypy can narrow the None check (it won't narrow across repeated
    # property reads, since a property isn't guaranteed to return the same
    # value each access).
    outcome = retry_state.outcome
    exception = outcome.exception() if outcome is not None else None
    logger.warning(
        "clip_request_retrying",
        extra={
            "extra_fields": {
                "attempt": retry_state.attempt_number,
                "exception": str(exception),
            }
        },
    )


def _collect_embedding(payload: dict[str, Any]) -> ClipEmbedding:
    """Extract and validate a {embedding, dimension} payload. Returns a
    ClipEmbedding; raises ClipServiceError for anything outside the
    documented shape (missing field, non-numeric values, dimension that
    doesn't match the actual embedding length)."""
    try:
        embedding = payload["embedding"]
        dimension = int(payload["dimension"])
    except (ValueError, KeyError, TypeError) as exc:
        raise ClipServiceError(
            f"CLIP service returned an unexpected response shape: {exc}"
        ) from exc
    try:
        vector = [float(v) for v in embedding]
    except (TypeError, ValueError) as exc:
        raise ClipServiceError(
            f"CLIP service embedding contained non-numeric values: {exc}"
        ) from exc
    if len(vector) != dimension:
        raise ClipServiceError(
            f"CLIP service embedding length ({len(vector)}) does not match declared "
            f"dimension ({dimension})."
        )
    return ClipEmbedding(embedding=vector, dimension=dimension)


def _request_headers() -> dict[str, str]:
    if settings.clip_service_api_key:
        return {"X-API-Key": settings.clip_service_api_key}
    return {}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    retry=retry_if_exception_type(ClipServiceError),
    reraise=True,
    before_sleep=_log_retry,
)
def embed_text(text: str) -> ClipEmbedding:
    """Embed a text query/string in CLIP space. Raises ClipServiceError
    after 3 attempts if the service is unreachable, times out, or returns
    an out-of-contract response."""
    start = time.perf_counter()
    try:
        response = httpx.post(
            f"{settings.clip_service_url}/embed/text",
            json={"text": text},
            headers=_request_headers(),
            timeout=settings.clip_service_timeout_seconds,
        )
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise ClipServiceError(
            f"CLIP service timed out after {settings.clip_service_timeout_seconds}s: {exc}"
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise ClipServiceError(
            f"CLIP service returned {exc.response.status_code}: {exc.response.text}"
        ) from exc
    except httpx.RequestError as exc:
        raise ClipServiceError(
            f"Could not reach CLIP service at {settings.clip_service_url}: {exc}"
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise ClipServiceError(f"CLIP service returned non-JSON body: {exc}") from exc

    result = _collect_embedding(payload)
    logger.info(
        "clip_text_embedded",
        extra={
            "extra_fields": {
                "text_length": len(text),
                "dimension": result.dimension,
                "processing_duration": round(time.perf_counter() - start, 4),
            }
        },
    )
    return result


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    retry=retry_if_exception_type(ClipServiceError),
    reraise=True,
    before_sleep=_log_retry,
)
def embed_image(contents: bytes, filename: str, content_type: str) -> ClipEmbedding:
    """Embed an image's bytes in CLIP space. Same contract and failure
    behavior as embed_text."""
    start = time.perf_counter()
    try:
        response = httpx.post(
            f"{settings.clip_service_url}/embed/image",
            files={"file": (filename, contents, content_type)},
            headers=_request_headers(),
            timeout=settings.clip_service_timeout_seconds,
        )
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise ClipServiceError(
            f"CLIP service timed out after {settings.clip_service_timeout_seconds}s: {exc}"
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise ClipServiceError(
            f"CLIP service returned {exc.response.status_code}: {exc.response.text}"
        ) from exc
    except httpx.RequestError as exc:
        raise ClipServiceError(
            f"Could not reach CLIP service at {settings.clip_service_url}: {exc}"
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise ClipServiceError(f"CLIP service returned non-JSON body: {exc}") from exc

    result = _collect_embedding(payload)
    logger.info(
        "clip_image_embedded",
        extra={
            "extra_fields": {
                "filename": filename,
                "byte_size": len(contents),
                "dimension": result.dimension,
                "processing_duration": round(time.perf_counter() - start, 4),
            }
        },
    )
    return result
