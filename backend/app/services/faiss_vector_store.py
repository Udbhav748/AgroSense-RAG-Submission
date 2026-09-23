"""FAISS-backed implementation of the VectorStore interface.

Uses IndexFlatIP (exact inner-product search). Callers are responsible for
passing L2-normalized embeddings (see embedding_service), so that inner
product is equivalent to cosine similarity. Metadata is persisted
separately as JSON, positionally aligned with vector rows in the index
(row i <-> metadata[i]).

Also owns a BM25Index (see hybrid_search.py) alongside the FAISS index —
a second, lexical view over the same chunk texts, rebuilt from self._metadata
every time it changes (create_index/add_embeddings/delete_document/load),
since BM25Okapi has no incremental update API. search_bm25() exposes it.
This BM25 pairing is a FAISSVectorStore-specific capability, not part of
the VectorStore ABC — see hybrid_search.py's module docstring for why.
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, cast

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore[assignment]
import numpy as np

from app.core.config import settings
from app.core.encryption import decrypt_text_field, encrypt_text_field
from app.core.exceptions import (
    CorruptedVectorStoreError,
    EmbeddingDimensionMismatchError,
    MetadataSyncError,
    VectorStoreNotFoundError,
)
from app.models.document import EmbeddedChunk, RetrievedChunk
from app.services import s3_sync_service
from app.services.hybrid_search import BM25Index
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

_VECTOR_STORE_DIR = settings.data_dir(settings.vector_store_dir_name)
DEFAULT_INDEX_PATH = _VECTOR_STORE_DIR / settings.vector_index_filename
DEFAULT_METADATA_PATH = _VECTOR_STORE_DIR / settings.vector_metadata_filename


def _encrypt_metadata_for_disk(metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Module 10 gap-closure: encrypt each record's chunk `text` before
    writing metadata.json to disk. Returns a NEW list -- the in-memory
    self._metadata this process continues to search/serve from is never
    mutated, since BM25 (a lexical index over the raw terms) and every
    other in-memory consumer need real plaintext to keep working; only
    the on-disk copy is protected. `chunk_id` is bound as associated
    data, so a ciphertext can't be silently swapped onto a different
    chunk's row without detection -- the same pattern already used for
    ChatTurn.content/ChatSession.title/feedback comments."""
    encrypted = []
    for record in metadata:
        record_metadata = record.get("metadata", {})
        text = record_metadata.get("text")
        if text is None:
            encrypted.append(record)
            continue
        new_record_metadata = dict(record_metadata)
        new_record_metadata["text"] = encrypt_text_field(
            text, associated_data=record["chunk_id"], key_b64=settings.encryption_key_b64
        )
        encrypted.append({**record, "metadata": new_record_metadata})
    return encrypted


def _decrypt_metadata_from_disk(metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Decrypt each record's chunk `text` after reading metadata.json
    from disk, back into the plaintext form every in-memory consumer
    (BM25, prompt building, citations, ...) already expects. A record
    without the encryption marker is a legacy plaintext row (written
    before this change) and passes through unchanged -- no forced
    migration, no plaintext row mistaken for ciphertext."""
    decrypted = []
    for record in metadata:
        record_metadata = record.get("metadata", {})
        text = record_metadata.get("text")
        if text is None:
            decrypted.append(record)
            continue
        new_record_metadata = dict(record_metadata)
        new_record_metadata["text"] = decrypt_text_field(
            text, associated_data=record["chunk_id"], key_b64=settings.encryption_key_b64
        )
        decrypted.append({**record, "metadata": new_record_metadata})
    return decrypted


class FAISSVectorStore(VectorStore):
    def __init__(self, index_path: Path | None = None, metadata_path: Path | None = None):
        self.index_path = index_path or DEFAULT_INDEX_PATH
        self.metadata_path = metadata_path or DEFAULT_METADATA_PATH
        self._index: faiss.Index | None = None
        self._metadata: list[dict[str, Any]] = []
        self._bm25_index = BM25Index()
        # Guards every method that reads or writes self._index/self._metadata
        # against a writer (add_embeddings/delete_document/load) swapping
        # them mid-read. threading.Lock rather than asyncio.Lock: FastAPI
        # runs async routes on the event loop but sync `def` routes (like
        # DELETE /documents/{id}) in a worker thread pool, so a lock scoped
        # to one event loop wouldn't actually cover both call paths — this
        # one does, at the cost of briefly blocking whichever thread is
        # waiting, which these already-synchronous, already-fast index/
        # metadata operations make an acceptable trade.
        #
        # Reads (search/search_bm25/get_chunks_by_document/list_document_ids/
        # first_chunk_vector/total_vectors) share this same lock rather than
        # a separate read/write lock: delete_document() replaces both
        # self._index and self._metadata with new objects, not in one atomic
        # step, so a read interleaved between those two reassignments could
        # search the new (rebuilt, differently-ordered) index but resolve
        # positions against the old metadata list — silently returning the
        # wrong document's chunk, or an IndexError, rather than a crash loud
        # enough to notice. A plain Lock (not RLock) is safe here because no
        # locked method ever calls another locked method while already
        # holding it — verified by inspection, not just assumed; a nested
        # acquire would deadlock instead of erroring, which is exactly the
        # kind of bug that stays invisible until real concurrent traffic
        # finds it, so this is worth stating explicitly rather than leaving
        # implicit.
        self._lock = threading.Lock()

    def _record_to_chunk(self, record: dict[str, Any], score: float) -> RetrievedChunk:
        record_metadata = record["metadata"]
        text = record_metadata.get("text", "")
        metadata = {key: value for key, value in record_metadata.items() if key != "text"}
        return RetrievedChunk(
            chunk_id=record["chunk_id"],
            document_id=record["document_id"],
            text=text,
            score=score,
            metadata=metadata,
        )

    def create_index(self, dimension: int) -> None:
        self._index = faiss.IndexFlatIP(dimension)
        self._metadata = []
        self._bm25_index.rebuild(self._metadata)

    def add_embeddings(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        with self._lock:
            if self._index is None:
                raise VectorStoreNotFoundError(
                    "No index to add to. Call create_index() or load() first."
                )
            if not embedded_chunks:
                return

            dimension = self._index.d
            for chunk in embedded_chunks:
                if len(chunk.embedding) != dimension:
                    raise EmbeddingDimensionMismatchError(
                        f"Embedding for chunk {chunk.chunk_id} has dimension "
                        f"{len(chunk.embedding)}, but the index expects {dimension}."
                    )

            start = time.perf_counter()

            # Agent 3.2 — semantic chunk dedup: skip chunks that are
            # near-duplicates of one already indexed for the SAME document
            # (this is what prevents near-identical chunks from starving
            # each other out via hybrid search's min-max normalization).
            # Only chunks already in the index are compared against, so the
            # filtering is independent of batch composition. A batch where
            # everything is filtered out is a valid no-op, not an error.
            deduped = 0
            if settings.chunk_dedup_enabled:
                surviving: list[EmbeddedChunk] = []
                for chunk in embedded_chunks:
                    if self._is_duplicate_of_existing(chunk.document_id, chunk.embedding):
                        deduped += 1
                        logger.info(
                            "chunk_deduplicated",
                            extra={
                                "extra_fields": {
                                    "document_id": chunk.document_id,
                                    "chunk_id": chunk.chunk_id,
                                }
                            },
                        )
                    else:
                        surviving.append(chunk)
                embedded_chunks = surviving

            if not embedded_chunks:
                logger.info(
                    "chunk_batch_fully_deduplicated",
                    extra={"extra_fields": {"chunks_skipped": deduped}},
                )
                return

            vectors = np.array([chunk.embedding for chunk in embedded_chunks], dtype="float32")
            self._index.add(vectors)
            self._metadata.extend(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "metadata": chunk.metadata,
                }
                for chunk in embedded_chunks
            )

            if self._index.ntotal != len(self._metadata):
                raise MetadataSyncError(
                    f"Vector count ({self._index.ntotal}) and metadata count "
                    f"({len(self._metadata)}) diverged after add_embeddings()."
                )

            self._bm25_index.rebuild(self._metadata)

            processing_duration = time.perf_counter() - start

            logger.info(
                "vectors_added",
                extra={
                    "extra_fields": {
                        "vectors_added": len(embedded_chunks),
                        "chunks_deduplicated": deduped,
                        "total_vectors": self._index.ntotal,
                        "embedding_dimension": dimension,
                        "processing_duration": round(processing_duration, 4),
                    }
                },
            )

    def _is_duplicate_of_existing(self, document_id: str, vector: list[float]) -> bool:
        """True if `vector` is a near-duplicate (cosine similarity >=
        settings.chunk_dedup_similarity_threshold) of any already-indexed
        chunk belonging to `document_id`. Vectors are assumed
        L2-normalized, so inner product == cosine similarity. Linear scan
        over self._metadata restricted to this document, same pattern
        delete_document already uses — fine at this app's per-document
        scale."""
        if not settings.chunk_dedup_enabled:
            return False
        if self._index is None:
            # Only called from add_embeddings(), which already guarantees
            # self._index is not None before reaching this helper — this is
            # a defensive fallback for mypy's benefit and for any future
            # caller, not a path expected to actually run: with no index,
            # nothing indexed yet counts as "not a duplicate".
            return False
        positions = [
            i for i, record in enumerate(self._metadata) if record["document_id"] == document_id
        ]
        for i in positions:
            existing_vector = self._index.reconstruct(i)
            similarity = float(np.dot(existing_vector, vector))
            if similarity >= settings.chunk_dedup_similarity_threshold:
                return True
        return False

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        tenant_id: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        with self._lock:
            if self._index is None:
                raise VectorStoreNotFoundError(
                    "No index to search. Call create_index() or load() first."
                )

            dimension = self._index.d
            if len(query_vector) != dimension:
                raise EmbeddingDimensionMismatchError(
                    f"Query vector has dimension {len(query_vector)}, but the index "
                    f"expects {dimension}."
                )

            if self._index.ntotal == 0 or top_k <= 0:
                return []

            query = np.array([query_vector], dtype="float32")
            # IndexFlatIP is exact brute-force: it scores every vector in the
            # index regardless of k, then returns the top k of those already-
            # computed scores. So when filtering by tenant, fetching *every*
            # score (k=ntotal) costs essentially the same as fetching top_k —
            # it just lets the tenant filter below see the whole ranked list
            # instead of risking top_k being filled with another tenant's
            # chunks before this tenant's actually-best matches are reached.
            k = self._index.ntotal if tenant_id is not None else min(top_k, self._index.ntotal)
            scores, positions = self._index.search(query, k)

            allowed_document_ids = set(document_ids) if document_ids is not None else None

            results = []
            for score, position in zip(scores[0], positions[0], strict=False):
                if position == -1:
                    continue
                record = self._metadata[position]
                if (
                    tenant_id is not None
                    and record["metadata"].get("tenant_id") not in (None, tenant_id)
                ):
                    continue
                if (
                    allowed_document_ids is not None
                    and record["document_id"] not in allowed_document_ids
                ):
                    continue
                results.append(self._record_to_chunk(record, float(score)))
                if len(results) >= top_k:
                    break

            return results

    def search_bm25(
        self,
        query: str,
        top_k: int,
        tenant_id: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """BM25 lexical search — a FAISSVectorStore-specific capability
        alongside search() (semantic), not part of the VectorStore ABC.
        Used by hybrid_search.hybrid_search() when
        Settings.hybrid_search_enabled. score is BM25's raw score, not a
        0-1 similarity — callers that fuse it with search()'s scores are
        responsible for normalizing first (see hybrid_search.py).

        tenant_id and document_ids have the same filtering semantics as
        search()."""
        with self._lock:
            return [
                self._record_to_chunk(record, score)
                for record, score in self._bm25_index.search(
                    query, top_k, tenant_id=tenant_id, document_ids=document_ids
                )
            ]

    def delete_document(self, document_id: str, tenant_id: int | None = None) -> int:
        with self._lock:
            if self._index is None:
                raise VectorStoreNotFoundError(
                    "No index to delete from. Call create_index() or load() first."
                )

            keep_positions = [
                i
                for i, record in enumerate(self._metadata)
                if record["document_id"] != document_id
                or (
                    tenant_id is not None
                    and record["metadata"].get("tenant_id") is not None
                    and record["metadata"].get("tenant_id") != tenant_id
                )
            ]
            removed_count = len(self._metadata) - len(keep_positions)
            if removed_count == 0:
                return 0

            start = time.perf_counter()
            dimension = self._index.d

            # IndexFlatIP has no native remove-by-id, but it stores raw vectors,
            # so rebuilding from the vectors we want to keep is exact (not an
            # approximation) and cheap at this project's scale.
            if keep_positions:
                all_vectors = self._index.reconstruct_n(0, self._index.ntotal)
                kept_vectors = all_vectors[keep_positions]
            else:
                kept_vectors = np.empty((0, dimension), dtype="float32")

            new_index = faiss.IndexFlatIP(dimension)
            if len(kept_vectors) > 0:
                new_index.add(kept_vectors)

            self._index = new_index
            self._metadata = [self._metadata[i] for i in keep_positions]
            self._bm25_index.rebuild(self._metadata)

            processing_duration = time.perf_counter() - start

            logger.info(
                "document_deleted",
                extra={
                    "extra_fields": {
                        "document_id": document_id,
                        "vectors_removed": removed_count,
                        "total_vectors": self._index.ntotal,
                        "processing_duration": round(processing_duration, 4),
                    }
                },
            )

        return removed_count

    def get_chunks_by_document(
        self, document_id: str, tenant_id: int | None = None
    ) -> list[RetrievedChunk]:
        with self._lock:
            if self._index is None:
                return []

            # score=1.0 isn't a similarity score — there's no query here,
            # this is a full-document fetch. It just signals "included".
            matches = [
                self._record_to_chunk(record, 1.0)
                for record in self._metadata
                if record["document_id"] == document_id
                and (tenant_id is None or record["metadata"].get("tenant_id") == tenant_id)
            ]

            matches.sort(key=lambda chunk: chunk.metadata.get("chunk_index", 0))
            return matches

    def list_document_ids(self, tenant_id: int | None = None) -> list[str]:
        """Distinct document_ids currently in the index, optionally
        restricted to one tenant. Used by duplicate-document detection
        (Agent 3.3) to enumerate peer documents to compare against — the
        vector store is the source of truth for what actually exists, so
        this works even when the metadata DB is disabled."""
        with self._lock:
            if self._index is None:
                return []

            document_ids: list[str] = []
            seen: set[str] = set()
            for record in self._metadata:
                if tenant_id is not None and record["metadata"].get("tenant_id") != tenant_id:
                    continue
                document_id = record["document_id"]
                if document_id not in seen:
                    seen.add(document_id)
                    document_ids.append(document_id)
            return document_ids

    def first_chunk_vector(
        self, document_id: str, tenant_id: int | None = None
    ) -> list[float] | None:
        """Raw embedding vector of a document's first chunk (in
        chunk_index order), or None if the document has no chunks. Vectors
        are L2-normalized, so inner product == cosine similarity — the
        cheap "fingerprint" duplicate-document detection compares."""
        with self._lock:
            if self._index is None:
                return None

            positions = [
                i
                for i, record in enumerate(self._metadata)
                if record["document_id"] == document_id
                and (tenant_id is None or record["metadata"].get("tenant_id") == tenant_id)
            ]
            if not positions:
                return None
            positions.sort(key=lambda i: self._metadata[i]["metadata"].get("chunk_index", 0))
            # faiss has no type stubs, so reconstruct(...).tolist() is
            # typed Any all the way through; reconstruct() returns a 1-D
            # float32 ndarray and .tolist() genuinely produces list[float]
            # at runtime, so this cast reflects reality rather than
            # papering over a real gap.
            return cast("list[float]", self._index.reconstruct(positions[0]).tolist())

    def save(self) -> None:
        with self._lock:
            if self._index is None:
                raise VectorStoreNotFoundError(
                    "No index to save. Call create_index() or load() first."
                )
            if self._index.ntotal != len(self._metadata):
                raise MetadataSyncError(
                    f"Refusing to save: vector count ({self._index.ntotal}) and metadata "
                    f"count ({len(self._metadata)}) are out of sync."
                )

            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self.index_path))
            self.metadata_path.write_text(json.dumps(_encrypt_metadata_for_disk(self._metadata)))

        # Outside the lock: network I/O has no reason to hold it, and this
        # covers both of save()'s call sites (upload, delete) automatically.
        s3_sync_service.upload_dir(self.index_path.parent, settings.vector_store_dir_name)

    def load(self) -> None:
        if not self.index_path.is_file():
            raise VectorStoreNotFoundError(f"No index file at {self.index_path}")
        if not self.metadata_path.is_file():
            raise VectorStoreNotFoundError(f"No metadata file at {self.metadata_path}")

        try:
            index = faiss.read_index(str(self.index_path))
        except Exception as exc:
            raise CorruptedVectorStoreError(
                f"Failed to read FAISS index at {self.index_path}: {exc}"
            ) from exc

        try:
            raw_metadata = json.loads(self.metadata_path.read_text())
        except Exception as exc:
            raise CorruptedVectorStoreError(
                f"Failed to read metadata at {self.metadata_path}: {exc}"
            ) from exc

        # Deliberately NOT inside the try/except above: a missing/wrong
        # encryption key or tampered ciphertext is a real, specific
        # failure (EncryptionKeyMissingError/EncryptionIntegrityError)
        # -- it must surface as that exact type, not be relabeled as a
        # generic "corrupted vector store" error, matching every other
        # encrypted-field call site in this codebase (ChatTurn.content,
        # ChatSession.title, feedback comments).
        metadata = _decrypt_metadata_from_disk(raw_metadata)

        if index.ntotal != len(metadata):
            raise MetadataSyncError(
                f"Loaded index has {index.ntotal} vectors but metadata has {len(metadata)} entries."
            )

        with self._lock:
            self._index = index
            self._metadata = metadata
            self._bm25_index.rebuild(self._metadata)

    def total_vectors(self) -> int:
        with self._lock:
            return self._index.ntotal if self._index is not None else 0
