"""Cross-Encoder Reranker & Scoring Service for AgroSense-RAG.

Provides precision re-ranking of retrieval candidate pools using:
1. Neural Cross-Encoder models (e.g. sentence-transformers/bge-reranker-small
   or cross-encoder/ms-marco-MiniLM-L-6-v2) with lazy-loaded singleton caching.
2. Fast heuristic token-overlap/exact-term alignment fallback for lightweight
   CPU environments or when torch/transformers are unavailable, guaranteeing
   100% zero-failure execution.
3. Logit calibration into normalized [0.0, 1.0] confidence scores stored in
   chunk metadata.
"""

from __future__ import annotations

import contextlib
import logging
import math
import re
import time
from functools import lru_cache
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from app.core.config import settings

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

TChunk = TypeVar("TChunk", bound=Any)

_TOKEN_RE = re.compile(r"\w+")
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "of", "while", "during",
    "to", "from", "in", "out", "on", "off", "again", "further", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "each",
    "few", "more", "most", "other", "some", "no", "nor", "not", "only",
    "own", "same", "too", "very", "s", "t", "can", "will", "don",
    "should", "now",
}


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _heuristic_score(query: str, text: str) -> float:
    """Compute an exact-term and token-alignment overlap score in [0.0, 1.0].

    Combines:
    - Content query term coverage / recall (filtered for stopwords)
    - Overall query term coverage
    - Jaccard token similarity
    - Exact phrase match bonus
    - Consecutive bigram match bonus
    """
    if not query.strip() or not text.strip():
        return 0.0

    q_tokens = _tokenize(query)
    t_tokens = _tokenize(text)
    if not q_tokens or not t_tokens:
        return 0.0

    q_content = [t for t in q_tokens if t not in _STOPWORDS]
    if not q_content:
        q_content = q_tokens

    q_set = set(q_tokens)
    q_content_set = set(q_content)
    t_set = set(t_tokens)

    # 1. Content query term coverage (weight: 0.45)
    content_overlap = q_content_set.intersection(t_set)
    content_coverage = len(content_overlap) / len(q_content_set)

    # 2. Overall token recall (weight: 0.15)
    token_overlap = q_set.intersection(t_set)
    token_coverage = len(token_overlap) / len(q_set)

    # 3. Jaccard similarity (weight: 0.15)
    jaccard = len(token_overlap) / len(q_set.union(t_set))

    # 4. Exact phrase match bonus (weight: 0.15)
    clean_q = " ".join(q_tokens)
    clean_t = " ".join(t_tokens)
    phrase_match = 1.0 if clean_q in clean_t else 0.0

    # 5. Consecutive bigram match bonus (weight: 0.10)
    if len(q_tokens) > 1:
        q_bigrams = {(q_tokens[i], q_tokens[i + 1]) for i in range(len(q_tokens) - 1)}
        t_bigrams = {(t_tokens[i], t_tokens[i + 1]) for i in range(len(t_tokens) - 1)}
        bigram_overlap = (
            len(q_bigrams.intersection(t_bigrams)) / len(q_bigrams) if q_bigrams else 0.0
        )
    else:
        bigram_overlap = 1.0 if content_coverage == 1.0 else 0.0

    score = (
        0.45 * content_coverage
        + 0.15 * token_coverage
        + 0.15 * jaccard
        + 0.15 * phrase_match
        + 0.10 * bigram_overlap
    )
    return max(0.0, min(1.0, float(score)))


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid mapping real logits to [0.0, 1.0]."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _calibrate_logits(raw_scores: Sequence[float]) -> list[float]:
    """Calibrate cross-encoder output into normalized [0.0, 1.0] scores."""
    if not raw_scores:
        return []
    calibrated = []
    for s in raw_scores:
        val = float(s)
        score = _sigmoid(val)
        calibrated.append(max(0.0, min(1.0, round(score, 6))))
    return calibrated


class CrossEncoderReranker:
    """Neural cross-encoder reranker with automatic heuristic fallback."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or getattr(
            settings, "reranking_model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self._model: Any = None
        self._model_load_attempted: bool = False
        self._model_load_failed: bool = False

    def _get_model(self) -> Any:
        """Lazy load sentence_transformers CrossEncoder."""
        if self._model_load_attempted:
            return self._model

        self._model_load_attempted = True
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
            logger.info(
                "cross_encoder_model_loaded",
                extra={"extra_fields": {"model_name": self._model_name}},
            )
        except Exception as exc:
            self._model_load_failed = True
            self._model = None
            logger.warning(
                "cross_encoder_load_failed_falling_back_to_heuristic",
                extra={"extra_fields": {"model_name": self._model_name, "error": str(exc)}},
            )
        return self._model

    def _extract_text(self, chunk: Any) -> str:
        """Safely extract text from chunk (BaseModel, dict, or object)."""
        if hasattr(chunk, "text") and isinstance(chunk.text, str):
            return chunk.text
        if isinstance(chunk, dict):
            if "text" in chunk:
                return str(chunk["text"])
            if (
                "metadata" in chunk
                and isinstance(chunk["metadata"], dict)
                and "text" in chunk["metadata"]
            ):
                return str(chunk["metadata"]["text"])
        if (
            hasattr(chunk, "metadata")
            and isinstance(chunk.metadata, dict)
            and "text" in chunk.metadata
        ):
            return str(chunk.metadata["text"])
        return str(chunk)

    def _attach_score(self, chunk: TChunk, score: float) -> TChunk:
        """Store rerank_score in metadata and on the chunk object."""
        score_val = max(0.0, min(1.0, float(score)))
        if isinstance(chunk, BaseModel):
            meta = dict(getattr(chunk, "metadata", {}) or {})
            meta["rerank_score"] = score_val
            update_dict: dict[str, Any] = {"metadata": meta}
            try:
                copied = chunk.model_copy(update=update_dict)
                with contextlib.suppress(Exception):
                    object.__setattr__(copied, "rerank_score", score_val)
                return copied
            except Exception:
                pass

        if isinstance(chunk, dict):
            chunk_copy = dict(chunk)
            if "metadata" in chunk_copy and isinstance(chunk_copy["metadata"], dict):
                chunk_copy["metadata"] = {**chunk_copy["metadata"], "rerank_score": score_val}
            else:
                chunk_copy["metadata"] = {"rerank_score": score_val}
            chunk_copy["rerank_score"] = score_val
            return chunk_copy  # type: ignore[return-value]

        if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
            chunk.metadata["rerank_score"] = score_val
        with contextlib.suppress(Exception):
            chunk.rerank_score = score_val
        return chunk

    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
        """Compute scores for a query and a sequence of texts."""
        if not texts:
            return []

        model = self._get_model()
        if model is not None:
            pairs = [(query, text) for text in texts]
            try:
                raw_scores = model.predict(pairs)
                return _calibrate_logits(raw_scores)
            except Exception as exc:
                logger.warning(
                    "cross_encoder_inference_failed_using_heuristic",
                    extra={"extra_fields": {"error": str(exc), "query": query[:50]}},
                )

        return [_heuristic_score(query, text) for text in texts]

    def rerank(
        self,
        query: str,
        chunks: Sequence[TChunk],
        top_k: int = 5,
    ) -> list[TChunk]:
        """Rerank chunks for the given query and return top_k precision results.

        Zero-failure guarantee: uses neural cross-encoder if available, falling back
        to heuristic alignment scoring on any failure or lightweight environment.
        """
        if not chunks or top_k <= 0:
            return []

        start = time.perf_counter()
        texts = [self._extract_text(chunk) for chunk in chunks]
        scores = self.score_pairs(query, texts)

        scored_pairs = [
            (self._attach_score(chunk, score), score)
            for chunk, score in zip(chunks, scores, strict=False)
        ]
        scored_pairs.sort(key=lambda pair: pair[1], reverse=True)

        results = [chunk for chunk, _ in scored_pairs[:top_k]]
        duration = time.perf_counter() - start

        logger.info(
            "rerank_completed",
            extra={
                "extra_fields": {
                    "query_length": len(query),
                    "input_candidates": len(chunks),
                    "output_chunks": len(results),
                    "top_k": top_k,
                    "top_score": round(scores[0], 4) if scores else 0.0,
                    "model_used": (
                        self._model_name
                        if (self._model is not None and not self._model_load_failed)
                        else "heuristic_fallback"
                    ),
                    "duration_seconds": round(duration, 4),
                }
            },
        )
        return results


@lru_cache(maxsize=1)
def get_reranker(model_name: str | None = None) -> CrossEncoderReranker:
    """Return a singleton instance of CrossEncoderReranker."""
    return CrossEncoderReranker(model_name=model_name)


def rerank(
    query: str,
    chunks: Sequence[TChunk],
    top_k: int = 5,
    model_name: str | None = None,
) -> list[TChunk]:
    """Module-level convenience function for reranking."""
    reranker_instance = get_reranker(model_name=model_name)
    return reranker_instance.rerank(query, chunks, top_k=top_k)
