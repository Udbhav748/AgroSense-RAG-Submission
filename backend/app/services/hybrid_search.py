"""Hybrid search: fuses FAISS semantic search with BM25 lexical search.

BM25Index wraps rank_bm25.BM25Okapi and is owned by FAISSVectorStore (see
faiss_vector_store.py), rebuilt from scratch on every
add_embeddings/delete_document/load/create_index — BM25Okapi has no
incremental update API, so a full rebuild is the only option, and this
project's scale makes that cheap enough not to matter.

This is a FAISSVectorStore-specific capability, not part of the portable
VectorStore interface (app/services/vector_store.py): swapping in a
different VectorStore backend later would need its own BM25 (or
equivalent) if hybrid search is still wanted, since lexical indexing over
an arbitrary backend isn't something the abstract interface promises.
retrieval_service.py checks for it explicitly (isinstance) rather than
assuming every VectorStore has it.
"""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Any

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from app.core.config import settings
from app.core.exceptions import ClipServiceError
from app.services import clip_client
from app.services.embedding_service import embed_query

if TYPE_CHECKING:
    from app.models.document import RetrievedChunk
    from app.services.faiss_vector_store import FAISSVectorStore
    from app.services.reranker import CrossEncoderReranker
    from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Lexical (BM25) index over a corpus of FAISSVectorStore metadata
    records. Not thread-safe against concurrent rebuild + search — the
    same caveat FAISSVectorStore itself already has around concurrent
    writes (see docs/ARCHITECTURE.md)."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._records: list[dict[str, Any]] = []

    def rebuild(self, records: list[dict[str, Any]]) -> None:
        """records is FAISSVectorStore._metadata — each a dict with
        chunk_id, document_id, and metadata (which carries the chunk
        text under metadata["text"]). Rebuilds from scratch every time;
        BM25Okapi has no incremental add."""
        self._records = records
        if not records or BM25Okapi is None:
            self._bm25 = None
            return

        corpus = [_tokenize(record["metadata"].get("text", "")) for record in records]
        self._bm25 = BM25Okapi(corpus)

    def search(
        self,
        query: str,
        top_k: int,
        tenant_id: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        """Return up to top_k (record, score) pairs, highest score first.
        score is BM25Okapi's raw score: unbounded above, exactly 0.0 for a
        document with no query-term overlap at all.

        tenant_id, when given, restricts candidates to records tagged with
        that tenant (see FAISSVectorStore.search's docstring for the same
        semantics) — applied before ranking, not after, so a wrong-tenant
        record can never crowd out a same-tenant one within top_k.
        document_ids, when given, restricts candidates to records whose
        document_id is in the list (Agent 3.1 collection-scoped chat)."""
        if self._bm25 is None or top_k <= 0:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        candidate_positions: list[int] = list(range(len(scores)))
        if tenant_id is not None:
            candidate_positions = [
                i
                for i in candidate_positions
                if self._records[i]["metadata"].get("tenant_id") in (None, tenant_id)
            ]
        if document_ids is not None:
            allowed = set(document_ids)
            candidate_positions = [
                i for i in candidate_positions if self._records[i]["document_id"] in allowed
            ]
        ranked_positions = sorted(candidate_positions, key=lambda i: scores[i], reverse=True)[
            :top_k
        ]
        return [(self._records[i], float(scores[i])) for i in ranked_positions]


def _min_max_normalize(scores: list[float]) -> list[float]:
    """Scale to [0, 1]. If every score is identical — including the
    degenerate empty-list case — there's no discriminative signal to
    preserve by dividing by a zero range: returns 1.0 for a positive tie
    (every candidate is equally "best") or 0.0 for an all-zero tie (BM25's
    "no lexical overlap at all" case)."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0 if hi > 0 else 0.0 for _ in scores]
    return [(score - lo) / (hi - lo) for score in scores]


def reciprocal_rank_fusion(
    ranked_lists: list[tuple[list[RetrievedChunk], float]],
    k: int = 60,
    top_k: int = 10,
) -> list[RetrievedChunk]:
    """Fuse multiple ranked lists of RetrievedChunk using Reciprocal Rank Fusion (RRF):

    RRF(d) = sum_{m in M} (w_m / (k + rank_m(d)))

    where:
    - ranked_lists: list of (candidate_chunks, weight) tuples for each retrieval modality m in M
    - rank_m(d): 1-based rank position of chunk d in list m (1, 2, ...)
    - k: RRF smoothing constant (default 60)
    - w_m: weight assigned to modality m
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    for results, weight in ranked_lists:
        if weight <= 0.0 or not results:
            continue
        for rank, chunk in enumerate(results, start=1):
            chunk_id = chunk.chunk_id
            rrf_score = weight / (k + rank)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_score
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = chunk

    sorted_chunk_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    return [
        chunk_map[cid].model_copy(update={"score": scores[cid]}) for cid in sorted_chunk_ids[:top_k]
    ]


def hybrid_search(
    query: str,
    vector_store: FAISSVectorStore,
    top_k: int,
    candidate_k: int | None = None,
    tenant_id: int | None = None,
    document_ids: list[str] | None = None,
    image_vector_store: VectorStore | None = None,
    rrf_k: int | None = None,
) -> list[RetrievedChunk]:
    """Fuse FAISS semantic search with BM25 lexical search, and (when CLIP
    cross-modal retrieval is enabled) a third CLIP image-similarity signal
    using Reciprocal Rank Fusion (RRF).

    vector_store must be a FAISSVectorStore (or anything exposing the same
    .search()/.search_bm25() pair) — see the module docstring on why this
    isn't typed against the VectorStore ABC.

    Pulls candidate_k results from each retriever (Settings.retrieval_candidate_k
    by default), calculates RRF rank scores per modality with smoothing constant
    k (Settings.hybrid_rrf_k or rrf_k parameter, defaulting to 60), fuses
    scores weighted by configured modality weights, dedupes by chunk_id, and
    returns the top_k fused results.

    Two-signal fusion (CLIP off / image_vector_store absent):
    w_sem = Settings.hybrid_semantic_weight, w_bm25 = 1 - w_sem.
    RRF(d) = w_sem / (k + rank_sem(d)) + w_bm25 / (k + rank_bm25(d))

    Three-signal fusion (Settings.clip_embedding_enabled and
    image_vector_store given):
    w_clip = Settings.hybrid_clip_weight, w_sem = (1 - w_clip) * hybrid_semantic_weight,
    w_bm25 = (1 - w_clip) * (1 - hybrid_semantic_weight).
    RRF(d) = w_clip / (k + rank_clip(d)) + w_sem / (k + rank_sem(d)) + w_bm25 / (k + rank_bm25(d))
    """
    resolved_candidate_k = (
        candidate_k if candidate_k is not None else settings.retrieval_candidate_k
    )
    resolved_rrf_k = rrf_k if rrf_k is not None else getattr(settings, "hybrid_rrf_k", 60)
    semantic_weight = settings.hybrid_semantic_weight

    start = time.perf_counter()

    query_vector = embed_query(query)
    search_kwargs = {}
    if document_ids is not None:
        search_kwargs["document_ids"] = document_ids
    semantic_results = vector_store.search(
        query_vector, resolved_candidate_k, tenant_id=tenant_id, **search_kwargs
    )
    bm25_results = vector_store.search_bm25(
        query, resolved_candidate_k, tenant_id=tenant_id, **search_kwargs
    )

    # CLIP cross-modal signal (Phase 4) — opt-in, degrade-don't-fail.
    clip_weight = 0.0
    clip_results: list[RetrievedChunk] = []
    if settings.clip_embedding_enabled and image_vector_store is not None:
        clip_weight = settings.hybrid_clip_weight
        try:
            clip_query = clip_client.embed_text(query)
            clip_results = image_vector_store.search(
                clip_query.embedding, resolved_candidate_k, tenant_id=tenant_id, **search_kwargs
            )
        except ClipServiceError as exc:
            logger.warning(
                "clip_query_degraded",
                extra={"extra_fields": {"error": str(exc), "query_length": len(query)}},
            )
            clip_results = []

    if clip_weight > 0.0:
        w_clip = clip_weight
        w_sem = (1.0 - clip_weight) * semantic_weight
        w_bm25 = (1.0 - clip_weight) * (1.0 - semantic_weight)
        ranked_lists: list[tuple[list[RetrievedChunk], float]] = [
            (semantic_results, w_sem),
            (bm25_results, w_bm25),
            (clip_results, w_clip),
        ]
    else:
        w_sem = semantic_weight
        w_bm25 = 1.0 - semantic_weight
        ranked_lists = [
            (semantic_results, w_sem),
            (bm25_results, w_bm25),
        ]

    results = reciprocal_rank_fusion(ranked_lists, k=resolved_rrf_k, top_k=top_k)

    processing_duration = time.perf_counter() - start
    logger.info(
        "hybrid_search_completed",
        extra={
            "extra_fields": {
                "query_length": len(query),
                "candidate_k": resolved_candidate_k,
                "rrf_k": resolved_rrf_k,
                "top_k": top_k,
                "semantic_candidate_count": len(semantic_results),
                "bm25_candidate_count": len(bm25_results),
                "clip_candidate_count": len(clip_results),
                "clip_weight": clip_weight,
                "fused_result_count": len(results),
                "processing_duration": round(processing_duration, 4),
            }
        },
    )

    return results


def hybrid_search_with_rerank(
    query: str,
    vector_store: FAISSVectorStore,
    top_k: int = 5,
    candidate_pool: int = 20,
    tenant_id: int | None = None,
    document_ids: list[str] | None = None,
    image_vector_store: VectorStore | None = None,
    rrf_k: int = 60,
    reranker: CrossEncoderReranker | None = None,
    **kwargs: Any,
) -> list[RetrievedChunk]:
    """Retrieve top candidates using hybrid search (RRF over dense FAISS + sparse BM25)
    and precision re-rank them using a CrossEncoderReranker.

    1. Uses Reciprocal Rank Fusion (RRF k=60) over dense FAISS + sparse BM25
       to retrieve the top candidate_pool (default 20) documents.
    2. Passes the top candidates into CrossEncoderReranker.rerank(query, candidates, top_k=top_k).
    3. Returns the top top_k precision-ranked chunks.
    """
    resolved_candidate_pool = candidate_pool if candidate_pool > 0 else 20
    resolved_top_k = top_k if top_k > 0 else 5

    candidates = hybrid_search(
        query=query,
        vector_store=vector_store,
        top_k=resolved_candidate_pool,
        candidate_k=resolved_candidate_pool,
        tenant_id=tenant_id,
        document_ids=document_ids,
        image_vector_store=image_vector_store,
        rrf_k=rrf_k,
    )

    if not candidates:
        return []

    if reranker is None:
        from app.services.reranker import get_reranker

        active_reranker = get_reranker()
    else:
        active_reranker = reranker

    reranked = active_reranker.rerank(
        query=query,
        chunks=candidates,
        top_k=resolved_top_k,
    )

    return reranked
