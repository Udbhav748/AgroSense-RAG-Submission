"""Generates embeddings for DocumentChunks using Sentence Transformers.

Independent of FAISS, retrieval, and Gemini: this service only turns chunk
text into embedding vectors.
"""

from __future__ import annotations

import logging
import time
import uuid
from functools import lru_cache
from typing import TYPE_CHECKING, cast

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.exceptions import (
    EmbeddingGenerationError,
    EmbeddingModelLoadError,
)
from app.models.document import DocumentChunk, EmbeddedChunk

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Tokens reserved for the [CLS]/[SEP] special tokens SentenceTransformer
# adds around the content when it actually encodes — model.max_seq_length
# is the *total* budget including those, so content tokens get this much
# less room than the raw max_seq_length.
_SPECIAL_TOKEN_BUDGET = 2


def _log_retry(retry_state: RetryCallState) -> None:
    # .outcome is a property, not a plain attribute -- assign it to a local
    # so mypy can narrow the None check (it won't narrow across repeated
    # property reads, since a property isn't guaranteed to return the same
    # value each access).
    outcome = retry_state.outcome
    exception = outcome.exception() if outcome is not None else None
    logger.warning(
        "embed_query_retrying",
        extra={
            "extra_fields": {
                "attempt": retry_state.attempt_number,
                "exception": str(exception),
            }
        },
    )


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it on every subsequent call.

    lru_cache serializes concurrent cache misses (CPython locks around the
    cache), so concurrent first calls block on one load instead of racing
    to load the model independently. A failed load isn't cached, so the
    next call retries rather than staying permanently broken.

    Imported here rather than at module top: `sentence_transformers` pulls
    in torch/transformers, a 30-45s cold import that used to block uvicorn
    from binding the port on every single startup. Deferring it into this
    function means the model loads only on first real use (embedding a
    chunk/query), the same once-per-process lru_cache lifecycle as before.
    """
    from sentence_transformers import SentenceTransformer

    from app.core.config import _sanitize_hf_cache_dirs

    _sanitize_hf_cache_dirs()
    try:
        # Fast path: load directly from local cache without HTTP network checks
        return cast(
            "SentenceTransformer",
            SentenceTransformer(settings.embedding_model_name, local_files_only=True),
        )
    except Exception:
        try:
            return cast("SentenceTransformer", SentenceTransformer(settings.embedding_model_name))
        except Exception as exc:
            raise EmbeddingModelLoadError(
                f"Failed to load embedding model '{settings.embedding_model_name}': {exc}"
            ) from exc


# Query embeddings are a pure function of (normalized query text, model
# weights) -- the model is loaded once per process (get_embedding_model's
# own lru_cache) and never changes mid-process, so this can never go
# stale the way caching retrieval results or generated answers would
# (both depend on the vector store's mutable state -- documents can be
# uploaded/deleted between requests, which the embedding step doesn't
# care about at all). Deliberately a *separate*, smaller cache from
# get_embedding_model's maxsize=1 -- this one is keyed per distinct
# query, not per model.
#
# Returns a tuple, not a list: lru_cache shares the exact same return
# object across every cache hit, and a list is mutable -- if this were
# embed_query itself, a caller mutating its result in place (nothing
# does today, but nothing stops a future one) would silently corrupt the
# cached value for every other caller sharing that entry. embed_query()
# below always returns a fresh list built from this, so its return-type
# contract doesn't change at all; only this private helper deals in the
# cache-safe immutable form.
@lru_cache(maxsize=256)
def _embed_query_cached(normalized_query: str) -> tuple[float, ...]:
    model = get_embedding_model()
    try:
        vector = model.encode(normalized_query, normalize_embeddings=True)
    except Exception as exc:
        raise EmbeddingGenerationError(f"Failed to generate embedding for query: {exc}") from exc
    return tuple(vector.tolist())


# Scoped to the exceptions this function actually raises: EmbeddingGenerationError
# (from model.encode failures) and EmbeddingModelLoadError (from get_embedding_model
# failures). The earlier version incorrectly caught LLMTimeoutError/LLMAPIError
# which this function never raises — so the retry was effectively dead code.
# lru_cache doesn't cache exceptions, so a failed call retries the real
# model.encode() on the next attempt rather than retrying against a cached
# failure.
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((EmbeddingGenerationError, EmbeddingModelLoadError)),
    reraise=True,
    before_sleep=_log_retry,
)
def embed_query(query: str) -> list[float]:
    """Embed a single query string, using the same model and normalization
    as generate_embeddings, so query and chunk vectors share one space.

    Cached (see _embed_query_cached) keyed on the query normalized by
    .strip() alone -- deliberately not case-folded or otherwise altered,
    since that's also what actually gets encoded on a cache miss (the
    normalized string, not the raw input), so a cache hit can never
    return an embedding for text other than what this exact call would
    have encoded itself.
    """
    normalized_query = query.strip()
    vector = list(_embed_query_cached(normalized_query))
    # embed_query had no success-path log line at all before this — its
    # retry events (embed_query_retrying, above) had nothing to correlate
    # against, so its retry outcomes were invisible to Retry Success Rate
    # (see monitoring/log_aggregate.py, eval/metrics_report.py).
    logger.info("embed_query_completed", extra={"extra_fields": {"query_length": len(query)}})
    return vector


def _split_oversized_chunk(chunk: DocumentChunk, model: SentenceTransformer) -> list[DocumentChunk]:
    """Split chunk.text into token-safe pieces if it would exceed the
    embedding model's max_seq_length.

    chunk_size (chars) was chosen without regard to the model's *token*
    budget — chunking_service is deliberately independent of the
    embedding model (see its module docstring). Ordinary prose at 1000
    chars stays well under all-MiniLM-L6-v2's 256-token limit, but dense
    technical text (chemical names, measurements, symptom lists — the
    shape of this app's own crop-disease content) can exceed it by 50%+.
    SentenceTransformer.encode() truncates silently past that limit, no
    error, no warning — so without this, the tail of an overlong chunk
    would just never make it into its embedding vector at all, even
    though the full text still reaches the LLM prompt unchanged.

    Uses the tokenizer's own offset mapping to find exact character
    cutoffs, so pieces are token-exact rather than a guessed character
    count. Returns [chunk] unchanged when it's already within budget —
    the overwhelmingly common case.
    """
    # max_seq_length is typed as int | None by sentence_transformers, though
    # a loaded model always sets it; 256 matches all-MiniLM-L6-v2's actual
    # limit (see docstring above) and is only a fallback for that None case.
    max_seq_length = model.max_seq_length or 256
    budget = max_seq_length - _SPECIAL_TOKEN_BUDGET
    encoding = model.tokenizer(chunk.text, add_special_tokens=False, return_offsets_mapping=True)
    token_count = len(encoding["input_ids"])
    if token_count <= budget:
        return [chunk]

    offsets = encoding["offset_mapping"]
    piece_texts = []
    for start_idx in range(0, token_count, budget):
        token_slice = offsets[start_idx : start_idx + budget]
        char_start, char_end = token_slice[0][0], token_slice[-1][1]
        piece_texts.append(chunk.text[char_start:char_end])

    logger.info(
        "chunk_split_for_token_budget",
        extra={
            "extra_fields": {
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "token_count": token_count,
                "token_budget": budget,
                "pieces": len(piece_texts),
            }
        },
    )

    return [
        DocumentChunk(
            # A fresh id per piece — chunk_id must be unique per vector
            # store row (see faiss_vector_store.py), and reusing the
            # original for one piece while suffixing the rest would be an
            # arbitrary asymmetry for no benefit.
            chunk_id=str(uuid.uuid4()),
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            text=piece_text,
            metadata={**chunk.metadata, "split_part": i, "split_of": len(piece_texts)},
        )
        for i, piece_text in enumerate(piece_texts)
    ]


def generate_embeddings(chunks: list[DocumentChunk]) -> list[EmbeddedChunk]:
    """Generate one embedding per chunk, preserving chunk_id, document_id, and metadata.

    A chunk whose token count exceeds the embedding model's max_seq_length
    is split into multiple token-safe pieces first (see
    _split_oversized_chunk), so the output can have *more* entries than
    the input — the same shape DocumentProcessingResponse already expects
    (total_chunks vs total_embeddings are tracked as separate fields).
    """
    if not chunks:
        return []

    model = get_embedding_model()
    document_id = chunks[0].document_id
    start = time.perf_counter()

    safe_chunks = [piece for chunk in chunks for piece in _split_oversized_chunk(chunk, model)]

    try:
        vectors = model.encode(
            [chunk.text for chunk in safe_chunks],
            normalize_embeddings=True,
            batch_size=settings.embedding_batch_size,
        )
    except Exception as exc:
        raise EmbeddingGenerationError(
            f"Failed to generate embeddings for document {document_id}: {exc}"
        ) from exc

    embedded_chunks = [
        EmbeddedChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            embedding=vector.tolist(),
            # Chunk text rides along in metadata so the vector store can
            # persist and later return it, without adding a field to
            # EmbeddedChunk itself.
            metadata={**chunk.metadata, "text": chunk.text},
        )
        for chunk, vector in zip(safe_chunks, vectors, strict=False)
    ]

    processing_duration = time.perf_counter() - start
    embedding_dimension = len(embedded_chunks[0].embedding)

    logger.info(
        "embeddings_generated",
        extra={
            "extra_fields": {
                "document_id": document_id,
                "embedding_model": settings.embedding_model_name,
                "total_chunks": len(chunks),
                "total_embeddings": len(embedded_chunks),
                "embedding_dimension": embedding_dimension,
                "processing_duration": round(processing_duration, 4),
            }
        },
    )

    return embedded_chunks
