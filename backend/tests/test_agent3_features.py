"""Tests for Agent 3 features: collection-scoped retrieval (3.1), semantic
chunk dedup at indexing (3.2), and duplicate-document detection on upload
(3.3). Uses real FAISSVectorStore instances backed by tmp_path, with no
embedding-model or network calls."""

import asyncio

import pytest

from app.core.config import settings
from app.models.document import DocumentChunk, EmbeddedChunk
from app.services.document_processing_service import DocumentProcessingService
from app.services.faiss_vector_store import FAISSVectorStore
from app.services.retrieval_service import retrieve

DIM = 8


def make_embedded(chunk_id, document_id, vector, chunk_index=0):
    return EmbeddedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        embedding=vector,
        metadata={"chunk_index": chunk_index, "text": f"text for {chunk_id}"},
    )


def make_store(tmp_path):
    store = FAISSVectorStore(
        index_path=tmp_path / "index.faiss", metadata_path=tmp_path / "metadata.json"
    )
    store.create_index(dimension=DIM)
    return store


async def _fake_save_uploaded_file(file):
    """Stand-in for the real async save_uploaded_file: no disk write, a
    fixed upload record."""
    return {
        "document_id": "new-doc",
        "original_filename": "new.pdf",
        "stored_filename": "new.pdf",
        "file_size": 100,
    }


class TestDocumentIdsScoping:
    def test_faiss_search_restricts_to_document_ids(self, tmp_path):
        store = make_store(tmp_path)
        v = [1.0] + [0.0] * (DIM - 1)
        store.add_embeddings([make_embedded("a1", "doc-a", v), make_embedded("b1", "doc-b", v)])

        results = store.search(v, top_k=10, document_ids=["doc-a"])
        assert {c.document_id for c in results} == {"doc-a"}

    def test_retrieve_passes_document_ids_through(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "hybrid_search_enabled", False)
        monkeypatch.setattr(settings, "reranking_enabled", False)
        monkeypatch.setattr("app.services.retrieval_service.embed_query", lambda q: [1.0] + [0.0] * (DIM - 1))
        store = make_store(tmp_path)
        v = [1.0] + [0.0] * (DIM - 1)
        store.add_embeddings([make_embedded("a1", "doc-a", v), make_embedded("b1", "doc-b", v)])

        results = retrieve("query", store, top_k=5, min_score=0.0, document_ids=["doc-a"])
        assert {c.document_id for c in results} == {"doc-a"}


class TestChunkDedup:
    def test_near_duplicate_from_same_document_is_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "chunk_dedup_enabled", True)
        monkeypatch.setattr(settings, "chunk_dedup_similarity_threshold", 0.97)
        store = make_store(tmp_path)

        base = [1.0] + [0.0] * (DIM - 1)
        store.add_embeddings([make_embedded("c1", "doc-1", base)])
        assert store.total_vectors() == 1

        near = [0.99] + [0.0] * (DIM - 1)  # dot(base, near) = 0.99 >= 0.97
        store.add_embeddings([make_embedded("c2", "doc-1", near)])
        assert store.total_vectors() == 1  # silently skipped

    def test_distinct_chunk_from_same_document_is_added(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "chunk_dedup_enabled", True)
        monkeypatch.setattr(settings, "chunk_dedup_similarity_threshold", 0.97)
        store = make_store(tmp_path)

        base = [1.0] + [0.0] * (DIM - 1)
        store.add_embeddings([make_embedded("c1", "doc-1", base)])

        diff = [0.0, 1.0] + [0.0] * (DIM - 2)  # orthogonal → dot 0
        store.add_embeddings([make_embedded("c3", "doc-1", diff)])
        assert store.total_vectors() == 2

    def test_same_vector_from_different_document_is_not_deduped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "chunk_dedup_enabled", True)
        monkeypatch.setattr(settings, "chunk_dedup_similarity_threshold", 0.97)
        store = make_store(tmp_path)

        base = [1.0] + [0.0] * (DIM - 1)
        store.add_embeddings([make_embedded("c1", "doc-1", base)])
        store.add_embeddings([make_embedded("c2", "doc-2", base)])
        assert store.total_vectors() == 2  # different document, not a duplicate

    def test_disabled_never_dedupes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "chunk_dedup_enabled", False)
        store = make_store(tmp_path)

        base = [1.0] + [0.0] * (DIM - 1)
        store.add_embeddings([make_embedded("c1", "doc-1", base)])
        store.add_embeddings([make_embedded("c2", "doc-1", base)])
        assert store.total_vectors() == 2


class TestDuplicateDocumentDetection:
    @pytest.fixture(autouse=True)
    def enable_flag(self, monkeypatch):
        monkeypatch.setattr(settings, "duplicate_document_detection_enabled", True)
        monkeypatch.setattr(settings, "duplicate_document_similarity_threshold", 0.95)

    def test_near_identical_upload_names_the_existing_document(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)
        vector = [1.0] + [0.0] * (DIM - 1)
        store.add_embeddings([make_embedded("existing-chunk", "existing-doc", vector)])

        monkeypatch.setattr(
            "app.services.document_processing_service.save_uploaded_file", _fake_save_uploaded_file
        )
        # Module 10 gap-closure: the real save_uploaded_file now encrypts
        # and writes the file to UPLOAD_DIR, but this test's fake stand-in
        # doesn't touch disk at all -- point UPLOAD_DIR at tmp_path (where
        # the UploadFile's own backing file below is actually created) so
        # document_processing_service's post-save decrypt-to-tempfile step
        # finds a real (here, empty) file instead of one that doesn't
        # exist. Extraction itself is separately mocked below and ignores
        # file content either way.
        monkeypatch.setattr("app.services.document_processing_service.UPLOAD_DIR", tmp_path)
        monkeypatch.setattr(
            "app.services.document_processing_service.extract_text_from_pdf",
            lambda document_id, file_path: {
                "total_pages": 1,
                "pages_ocred": 0,
                "extracted_text": "new text",
                "extracted_characters": 8,
            },
        )
        monkeypatch.setattr("app.services.document_processing_service.detect_pii", lambda text: {})
        monkeypatch.setattr(
            "app.services.document_processing_service.extract_images_from_pdf",
            lambda document_id, file_path: [],
        )
        monkeypatch.setattr(
            "app.services.document_processing_service.extract_tables_from_pdf",
            lambda document_id, file_path: [],
        )
        monkeypatch.setattr(
            "app.services.document_processing_service.chunk_document",
            lambda extracted_document, tenant_id=None: [
                DocumentChunk(chunk_id="nc1", document_id="new-doc", chunk_index=0, text="new text")
            ],
        )
        monkeypatch.setattr(
            "app.services.document_processing_service.generate_embeddings",
            lambda chunks: [make_embedded("nc1", "new-doc", vector)],
        )

        from fastapi import UploadFile

        upload = UploadFile(filename="new.pdf", file=(tmp_path / "new.pdf").open("w+b"))
        try:
            response = asyncio.run(
                DocumentProcessingService(store).process(upload, tenant_id=None, collection="c")
            )
        finally:
            upload.file.close()

        assert response.collection == "c"
        assert response.possible_duplicate_of == "existing-doc"

    def test_different_document_is_not_flagged(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)
        store.add_embeddings([make_embedded("existing-chunk", "existing-doc", [1.0] + [0.0] * (DIM - 1))])

        other_vector = [0.0, 1.0] + [0.0] * (DIM - 2)
        monkeypatch.setattr(
            "app.services.document_processing_service.save_uploaded_file", _fake_save_uploaded_file
        )
        # Module 10 gap-closure: the real save_uploaded_file now encrypts
        # and writes the file to UPLOAD_DIR, but this test's fake stand-in
        # doesn't touch disk at all -- point UPLOAD_DIR at tmp_path (where
        # the UploadFile's own backing file below is actually created) so
        # document_processing_service's post-save decrypt-to-tempfile step
        # finds a real (here, empty) file instead of one that doesn't
        # exist. Extraction itself is separately mocked below and ignores
        # file content either way.
        monkeypatch.setattr("app.services.document_processing_service.UPLOAD_DIR", tmp_path)
        monkeypatch.setattr(
            "app.services.document_processing_service.extract_text_from_pdf",
            lambda document_id, file_path: {
                "total_pages": 1,
                "pages_ocred": 0,
                "extracted_text": "new text",
                "extracted_characters": 8,
            },
        )
        monkeypatch.setattr("app.services.document_processing_service.detect_pii", lambda text: {})
        monkeypatch.setattr(
            "app.services.document_processing_service.extract_images_from_pdf",
            lambda document_id, file_path: [],
        )
        monkeypatch.setattr(
            "app.services.document_processing_service.extract_tables_from_pdf",
            lambda document_id, file_path: [],
        )
        monkeypatch.setattr(
            "app.services.document_processing_service.chunk_document",
            lambda extracted_document, tenant_id=None: [
                DocumentChunk(chunk_id="nc1", document_id="new-doc", chunk_index=0, text="new text")
            ],
        )
        monkeypatch.setattr(
            "app.services.document_processing_service.generate_embeddings",
            lambda chunks: [make_embedded("nc1", "new-doc", other_vector)],
        )

        from fastapi import UploadFile

        upload = UploadFile(filename="new.pdf", file=(tmp_path / "new.pdf").open("w+b"))
        try:
            response = asyncio.run(DocumentProcessingService(store).process(upload, tenant_id=None))
        finally:
            upload.file.close()

        assert response.possible_duplicate_of is None

    def test_flag_off_never_flags(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "duplicate_document_detection_enabled", False)
        store = make_store(tmp_path)
        vector = [1.0] + [0.0] * (DIM - 1)
        store.add_embeddings([make_embedded("existing-chunk", "existing-doc", vector)])

        monkeypatch.setattr(
            "app.services.document_processing_service.save_uploaded_file", _fake_save_uploaded_file
        )
        # Module 10 gap-closure: the real save_uploaded_file now encrypts
        # and writes the file to UPLOAD_DIR, but this test's fake stand-in
        # doesn't touch disk at all -- point UPLOAD_DIR at tmp_path (where
        # the UploadFile's own backing file below is actually created) so
        # document_processing_service's post-save decrypt-to-tempfile step
        # finds a real (here, empty) file instead of one that doesn't
        # exist. Extraction itself is separately mocked below and ignores
        # file content either way.
        monkeypatch.setattr("app.services.document_processing_service.UPLOAD_DIR", tmp_path)
        monkeypatch.setattr(
            "app.services.document_processing_service.extract_text_from_pdf",
            lambda document_id, file_path: {
                "total_pages": 1,
                "pages_ocred": 0,
                "extracted_text": "new text",
                "extracted_characters": 8,
            },
        )
        monkeypatch.setattr("app.services.document_processing_service.detect_pii", lambda text: {})
        monkeypatch.setattr(
            "app.services.document_processing_service.extract_images_from_pdf",
            lambda document_id, file_path: [],
        )
        monkeypatch.setattr(
            "app.services.document_processing_service.extract_tables_from_pdf",
            lambda document_id, file_path: [],
        )
        monkeypatch.setattr(
            "app.services.document_processing_service.chunk_document",
            lambda extracted_document, tenant_id=None: [
                DocumentChunk(chunk_id="nc1", document_id="new-doc", chunk_index=0, text="new text")
            ],
        )
        monkeypatch.setattr(
            "app.services.document_processing_service.generate_embeddings",
            lambda chunks: [make_embedded("nc1", "new-doc", vector)],
        )

        from fastapi import UploadFile

        upload = UploadFile(filename="new.pdf", file=(tmp_path / "new.pdf").open("w+b"))
        try:
            response = asyncio.run(DocumentProcessingService(store).process(upload, tenant_id=None))
        finally:
            upload.file.close()

        assert response.possible_duplicate_of is None
