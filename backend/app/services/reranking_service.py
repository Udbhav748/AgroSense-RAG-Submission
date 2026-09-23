"""Cross-encoder re-ranking of a retrieval candidate pool.

Unlike the bi-encoder used for retrieval (embedding_service.py, which
embeds the query and each chunk independently so they can be compared by
vector similarity), a cross-encoder scores (query, chunk) pairs jointly —
slower per pair, but typically more accurate at judging relevance, which
is why it's used to re-order an already-narrowed candidate pool rather
than to search the whole corpus.

Lazy-loaded and cached the same way embedding_service.get_embedding_model()
is: a CrossEncoder is a real model load, done once per process and reused.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import TYPE_CHECKING, cast

from app.core.config import settings
from app.core.exceptions import RerankingError

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

    from app.models.document import RetrievedChunk

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Load the cross-encoder once and reuse it on every subsequent call.

    Same lru_cache rationale as get_embedding_model(): serializes
    concurrent first-call loads instead of racing, and a failed load isn't
    cached, so the next call retries rather than staying permanently
    broken.

    Imported here rather than at module top for the same reason as
    embedding_service.get_embedding_model: `sentence_transformers` costs
    30-45s of import before uvicorn can bind the port. With reranking off
    by default (settings.reranking_enabled=False) that import is pure waste
    on every startup — deferred to first actual use, when a retrieve-action
    request actually needs the cross-encoder.
    """
    from sentence_transformers import CrossEncoder

    try:
        # sentence_transformers ships no py.typed marker, so its constructor
        # resolves to Any under mypy --strict; cast back to the declared
        # return type rather than let Any leak into every caller.
        return cast("CrossEncoder", CrossEncoder(settings.reranking_model_name))
    except Exception as exc:
        raise RerankingError(
            f"Failed to load reranking model '{settings.reranking_model_name}': {exc}"
        ) from exc


def rerank(query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Re-score candidates against query with the cross-encoder and return
    the best top_k, re-ordered.

    Each returned chunk keeps its ORIGINAL .score (semantic or fused)
    rather than the raw cross-encoder score: cross-encoder scores aren't
    on the same roughly-0-1 scale the rest of the system treats .score as
    being on (retrieval_min_score filtering, ChatService._grade_retrieval's
    threshold) — swapping it in would silently break both. The
    cross-encoder only decides *selection and order* here; its raw score
    is still logged (see "reranking_applied") and stashed in
    metadata["rerank_score"] for transparency.

    Raises RerankingError on failure — callers are expected to catch it
    and fall back to the unreranked candidates (see retrieval_service.py)
    rather than fail retrieval outright over a quality enhancement.
    """
    if not candidates:
        return candidates

    before_top_chunk_ids = [chunk.chunk_id for chunk in candidates[:top_k]]

    start = time.perf_counter()
    model = get_reranker()
    pairs = [(query, chunk.text) for chunk in candidates]
    try:
        # CrossEncoder.predict's stub type covers the newer multimodal
        # (text/image/audio/video) input shapes; list[tuple[str, str]] is
        # the documented, correct call for plain text reranking.
        raw_scores = model.predict(pairs)  # type: ignore[arg-type]
    except Exception as exc:
        raise RerankingError(f"Failed to re-rank candidates: {exc}") from exc
    processing_duration = time.perf_counter() - start

    ranked = sorted(
        zip(candidates, raw_scores, strict=False), key=lambda pair: pair[1], reverse=True
    )
    results = [
        chunk.model_copy(
            update={"metadata": {**chunk.metadata, "rerank_score": float(rerank_score)}}
        )
        for chunk, rerank_score in ranked[:top_k]
    ]

    logger.info(
        "reranking_applied",
        extra={
            "extra_fields": {
                "query_length": len(query),
                "candidate_count": len(candidates),
                "top_k": top_k,
                "before_top_chunk_ids": before_top_chunk_ids,
                "after_top_chunk_ids": [chunk.chunk_id for chunk in results],
                "processing_duration": round(processing_duration, 4),
            }
        },
    )

    return results
