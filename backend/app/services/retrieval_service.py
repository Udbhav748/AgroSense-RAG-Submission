"""Retrieves the most relevant chunks for a natural-language query.

The baseline path depends only on the VectorStore interface and
embedding_service.embed_query. Two opt-in enhancements layer on top,
each config-gated and defaulting to False so the baseline is unchanged
unless explicitly enabled (see docs/OPERATIONS.md's "Retrieval ablation"
for how to A/B them against each other):

- Settings.hybrid_search_enabled — fuse FAISS semantic search with a BM25
  lexical index (hybrid_search.py). Requires a FAISSVectorStore
  specifically (checked via isinstance, not assumed) since BM25 is a
  FAISSVectorStore-specific capability, not part of the VectorStore ABC.
- Settings.reranking_enabled — re-score a wider candidate pool with a
  cross-encoder (reranking_service.py) before narrowing to top_k.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import TYPE_CHECKING, Any

from app.core.config import settings
from app.core.exceptions import RerankingError
from app.core.metrics import get_metrics
from app.services.embedding_service import embed_query
from app.services.faiss_vector_store import FAISSVectorStore
from app.services.hybrid_search import hybrid_search
from app.services.prompt_injection_service import detect_possible_injection
from app.services.reranking_service import rerank
from app.services.tool_registry import track_tool

if TYPE_CHECKING:
    from app.models.document import RetrievedChunk
    from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _search_with_timeout(
    query: str,
    vector_store: VectorStore,
    fetch_k: int,
    resolved_top_k: int,
    tenant_id: int | None = None,
    document_ids: list[str] | None = None,
    image_vector_store: VectorStore | None = None,
) -> list[RetrievedChunk]:
    """Run the retrieval search block (hybrid or plain semantic) under a
    configurable wall-clock timeout.

    The search + optional reranking is CPU-bound in-process work, so the
    only portable way to bound it is a worker thread whose result is
    waited on with a timeout. If the timeout fires, the search thread keeps
    running (Python threads can't be killed) but this request degrades to
    an empty result list instead of hanging — which flows through the
    existing retrieval-grading path as "insufficient" and, if enabled, the
    web-search fallback, so a pathological index never wedges a request.

    With settings.retrieval_timeout_seconds == 0 (the default), the search
    runs inline with no thread overhead at all — behavior unchanged.

    image_vector_store is forwarded to hybrid_search for the Phase 4 CLIP
    cross-modal signal; None (or the flag off) keeps retrieval text-only.
    """

    def _do_search() -> list[RetrievedChunk]:
        if settings.hybrid_search_enabled and isinstance(vector_store, FAISSVectorStore):
            if document_ids is not None:
                return hybrid_search(
                    query,
                    vector_store,
                    top_k=fetch_k,
                    tenant_id=tenant_id,
                    document_ids=document_ids,
                    image_vector_store=image_vector_store,
                )
            return hybrid_search(
                query,
                vector_store,
                top_k=fetch_k,
                tenant_id=tenant_id,
                image_vector_store=image_vector_store,
            )
        query_vector = embed_query(query)
        # document_ids filtering is only available at the FAISS search
        # layer; for any other VectorStore implementation the caller's
        # exact post-filter in retrieve() is the enforcement point.
        if isinstance(vector_store, FAISSVectorStore):
            if document_ids is not None:
                return vector_store.search(
                    query_vector, fetch_k, tenant_id=tenant_id, document_ids=document_ids
                )
            return vector_store.search(query_vector, fetch_k, tenant_id=tenant_id)
        return vector_store.search(query_vector, fetch_k, tenant_id=tenant_id)

    timeout = settings.retrieval_timeout_seconds
    if timeout <= 0:
        return _do_search()

    # No `with` block here: ThreadPoolExecutor.__exit__ calls
    # shutdown(wait=True), which would block until the hung worker thread
    # finishes — exactly what the timeout exists to avoid. shutdown
    # (wait=False) lets the orphaned thread drain in the background while
    # this request already moved on.
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_do_search)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            get_metrics().inc_counter("retrieval_timeouts_total")
            logger.warning(
                "retrieval_timed_out",
                extra={"extra_fields": {"timeout_seconds": timeout, "top_k": resolved_top_k}},
            )
            return []
    finally:
        executor.shutdown(wait=False)


def _retrieve_core(
    query: str,
    vector_store: VectorStore,
    top_k: int | None = None,
    min_score: float | None = None,
    tenant_id: int | None = None,
    document_ids: list[str] | None = None,
    image_vector_store: VectorStore | None = None,
    collection: str | None = None,
    rerank_candidates: bool | None = None,
) -> list[RetrievedChunk]:
    """Inner retrieval implementation shared by retrieve() and the dynamic
    tool registry.
    """
    resolved_top_k = top_k if top_k is not None else settings.retrieval_top_k
    resolved_min_score = min_score if min_score is not None else settings.retrieval_min_score

    start = time.perf_counter()

    resolved_doc_ids = list(document_ids) if document_ids is not None else None
    if collection is not None and resolved_doc_ids is None:
        try:
            from app.services.document_repository import list_documents

            docs = list_documents(tenant_id=tenant_id, collection=collection)
            if docs:
                resolved_doc_ids = [doc["document_id"] for doc in docs]
        except Exception:
            resolved_doc_ids = None

    reranking_active = (
        rerank_candidates if rerank_candidates is not None else settings.reranking_enabled
    )
    fetch_k = settings.retrieval_candidate_k if reranking_active else resolved_top_k

    results = _search_with_timeout(
        query,
        vector_store,
        fetch_k,
        resolved_top_k,
        tenant_id=tenant_id,
        document_ids=resolved_doc_ids,
        image_vector_store=image_vector_store,
    )

    reranked = False
    if reranking_active:
        try:
            results = rerank(query, results, top_k=resolved_top_k)
            reranked = True
        except RerankingError as exc:
            logger.warning("reranking_failed", extra={"extra_fields": {"error": str(exc)}})
            results = results[:resolved_top_k]

    if reranked or settings.hybrid_search_enabled:
        filtered = results
    else:
        filtered = [chunk for chunk in results if chunk.score >= resolved_min_score]

    if resolved_doc_ids:
        allowed = set(resolved_doc_ids)
        filtered = [chunk for chunk in filtered if chunk.document_id in allowed]
    elif collection:
        # Fallback: if collection couldn't be resolved via DB, filter by chunk metadata when present
        collection_filtered = [
            chunk
            for chunk in filtered
            if chunk.metadata.get("collection") in (collection, "plant_disease_docs", "general", None)
            or chunk.metadata.get("crop") == collection
        ]
        if collection_filtered:
            filtered = collection_filtered

    log_extra: dict[str, Any]
    if resolved_doc_ids:
        log_extra = {
            "document_id_count": len(resolved_doc_ids),
            "results_after_doc_filter": len(filtered),
        }
    elif collection:
        log_extra = {"collection": collection, "results_after_collection_filter": len(filtered)}
    else:
        log_extra = {}

    processing_duration = time.perf_counter() - start

    logger.info(
        "retrieval_completed",
        extra={
            "extra_fields": {
                "query_length": len(query),
                "top_k": resolved_top_k,
                "min_score": resolved_min_score,
                "hybrid_search_enabled": settings.hybrid_search_enabled,
                "reranking_enabled": settings.reranking_enabled,
                "results_before_threshold": len(results),
                "results_returned": len(filtered),
                "processing_duration": round(processing_duration, 4),
            }
            | log_extra
        },
    )

    flagged = [
        {"chunk_id": chunk.chunk_id, "document_id": chunk.document_id, "categories": categories}
        for chunk in filtered
        if (categories := detect_possible_injection(chunk.text))
    ]
    if flagged:
        logger.warning(
            "possible_injection_detected",
            extra={"extra_fields": {"source": "chunk", "flagged_chunks": flagged}},
        )

    return filtered


@track_tool("retrieval")
def retrieve(
    query: str,
    vector_store: VectorStore,
    top_k: int | None = None,
    min_score: float | None = None,
    tenant_id: int | None = None,
    document_ids: list[str] | None = None,
    image_vector_store: VectorStore | None = None,
    collection: str | None = None,
    rerank: bool | None = None,
) -> list[RetrievedChunk]:
    """Public, instrumented retrieval entry point — the inline ChatService
    path. Validates args, runs _retrieve_core, and logs one
    tool_invocation event (see tool_registry.py's @track_tool). The
    dynamic tool registry calls _retrieve_core directly instead, so a
    single retrieval call is never counted twice.
    """
    return _retrieve_core(
        query,
        vector_store,
        top_k=top_k,
        min_score=min_score,
        tenant_id=tenant_id,
        document_ids=document_ids,
        image_vector_store=image_vector_store,
        collection=collection,
        rerank_candidates=rerank,
    )
