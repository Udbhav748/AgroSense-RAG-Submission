"""Abstract interface for a vector store.

The rest of the application should depend on this interface, not on any
specific backing implementation (e.g. FAISS), so the store can be swapped
(e.g. for Chroma) without touching calling code.
"""

from abc import ABC, abstractmethod

from app.models.document import EmbeddedChunk, RetrievedChunk


class VectorStore(ABC):
    @abstractmethod
    def create_index(self, dimension: int) -> None:
        """Create a new, empty index for vectors of the given dimension."""

    @abstractmethod
    def add_embeddings(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """Add embeddings to the index, keeping metadata in sync with vector positions."""

    @abstractmethod
    def search(
        self, query_vector: list[float], top_k: int, tenant_id: int | None = None
    ) -> list[RetrievedChunk]:
        """Return up to top_k chunks most similar to query_vector, ranked by score.

        Similarity thresholding is not applied here; callers filter results
        by score themselves.

        tenant_id, when given, restricts results to chunks tagged with that
        tenant at ingest (see chunking_service.chunk_document) — chunks
        with no tenant tag (uploaded before tenant tracking existed, or
        with the DB disabled) never match a specific tenant_id, so they're
        excluded rather than leaked. None (the default) means no
        filtering, i.e. today's single-tenant behavior.
        """

    @abstractmethod
    def delete_document(self, document_id: str, tenant_id: int | None = None) -> int:
        """Remove every vector belonging to document_id for the given tenant. Returns count removed.

        Returns 0 (rather than raising) if the document isn't present —
        callers decide whether that's an error.
        """

    @abstractmethod
    def get_chunks_by_document(
        self, document_id: str, tenant_id: int | None = None
    ) -> list[RetrievedChunk]:
        """Return every stored chunk for document_id, ordered by chunk_index.

        Not a similarity search — score is meaningless here and callers
        (e.g. summarization_service) should ignore it. Returns an empty
        list (rather than raising) if the document isn't present.

        tenant_id has the same filtering semantics as search(): when
        given, chunks not tagged with that tenant (including untagged
        ones) are excluded — so a caller can't summarize a document by ID
        alone regardless of who it actually belongs to.
        """

    @abstractmethod
    def save(self) -> None:
        """Persist the index and its metadata to storage."""

    @abstractmethod
    def load(self) -> None:
        """Load a previously persisted index and its metadata from storage."""

    @abstractmethod
    def total_vectors(self) -> int:
        """Return the number of vectors currently in the index."""
