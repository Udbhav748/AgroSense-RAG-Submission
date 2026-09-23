"""Tests for cross-tenant data isolation in the retrieval, summarization,
and session paths.

Before this, tenant_id was tracked in Postgres (document_repository.py)
purely for the /documents listing endpoint, but never reached chunk
metadata, the vector store's search, document summarization, or session
lookups — any authenticated client could retrieve or summarize another
tenant's uploaded documents, and anyone who obtained another tenant's
session_id could read/extend their conversation history. These tests
cover the fix at each layer: chunking tags tenant_id into chunk metadata,
FAISSVectorStore.search()/search_bm25()/get_chunks_by_document() filter
by it, hybrid_search() and retrieve() forward it end to end,
summarize_document() refuses to summarize another tenant's document, and
InMemorySessionStore refuses to hand a session to a different tenant than
the one that created it. Offline throughout — real
FAISSVectorStore/BM25Index/InMemorySessionStore, no network calls.
"""

import pytest

from app.core.exceptions import DocumentNotFoundError
from app.models.document import EmbeddedChunk, ExtractedDocument
from app.services.chunking_service import chunk_document
from app.services.faiss_vector_store import FAISSVectorStore
from app.services.hybrid_search import hybrid_search
from app.services.retrieval_service import retrieve
from app.services.session_store import InMemorySessionStore
from app.services.summarization_service import summarize_document

FAKE_EMBEDDING_DIM = 4
# Both tenants' chunks share the *same* embedding, so without tenant
# filtering they'd tie for top score — this makes the tests airtight:
# what excludes the other tenant's chunk is the filter, not similarity.
SHARED_EMBEDDING = [1.0, 0.0, 0.0, 0.0]


def _store_with_two_tenants(tmp_path):
    store = FAISSVectorStore(index_path=tmp_path / "i.faiss", metadata_path=tmp_path / "m.json")
    store.create_index(dimension=FAKE_EMBEDDING_DIM)
    store.add_embeddings(
        [
            EmbeddedChunk(
                chunk_id="tenant-1-chunk",
                document_id="doc-1",
                embedding=SHARED_EMBEDDING,
                metadata={"chunk_index": 0, "total_chunks": 1, "source": "pdf", "tenant_id": 1, "text": "t1 text"},
            ),
            EmbeddedChunk(
                chunk_id="tenant-2-chunk",
                document_id="doc-2",
                embedding=SHARED_EMBEDDING,
                metadata={"chunk_index": 0, "total_chunks": 1, "source": "pdf", "tenant_id": 2, "text": "t2 text"},
            ),
            EmbeddedChunk(
                chunk_id="untagged-chunk",
                document_id="doc-3",
                embedding=SHARED_EMBEDDING,
                metadata={"chunk_index": 0, "total_chunks": 1, "source": "pdf", "text": "legacy text"},
            ),
        ]
    )
    return store


class TestChunkDocumentTagsTenant:
    def test_tenant_id_lands_in_every_chunk_metadata(self):
        text = "a" * 3000
        document = ExtractedDocument(
            document_id="doc-1", extracted_text=text, total_pages=1, extracted_characters=len(text), pages_ocred=0
        )
        chunks = chunk_document(document, tenant_id=42)
        assert chunks
        assert all(chunk.metadata["tenant_id"] == 42 for chunk in chunks)

    def test_no_tenant_stores_none_not_a_missing_key(self):
        text = "short text"
        document = ExtractedDocument(
            document_id="doc-1", extracted_text=text, total_pages=1, extracted_characters=len(text), pages_ocred=0
        )
        chunks = chunk_document(document)
        assert chunks[0].metadata["tenant_id"] is None


class TestFAISSVectorStoreTenantFiltering:
    def test_search_without_tenant_id_returns_everyone(self, tmp_path):
        store = _store_with_two_tenants(tmp_path)
        results = store.search(SHARED_EMBEDDING, top_k=10)
        assert {r.chunk_id for r in results} == {"tenant-1-chunk", "tenant-2-chunk", "untagged-chunk"}

    def test_search_with_tenant_id_excludes_other_tenants(self, tmp_path):
        store = _store_with_two_tenants(tmp_path)
        results = store.search(SHARED_EMBEDDING, top_k=10, tenant_id=1)
        assert {r.chunk_id for r in results} == {"tenant-1-chunk", "untagged-chunk"}

    def test_search_with_tenant_id_includes_untagged_public_chunks(self, tmp_path):
        # An untagged chunk is treated as public/shared (e.g. the app's own
        # ingested reference corpus, not a private per-tenant upload) —
        # visible to every tenant's filtered search, not excluded.
        store = _store_with_two_tenants(tmp_path)
        results = store.search(SHARED_EMBEDDING, top_k=10, tenant_id=2)
        assert {r.chunk_id for r in results} == {"tenant-2-chunk", "untagged-chunk"}

    def test_search_respects_top_k_after_filtering(self, tmp_path):
        # A tied-score other-tenant chunk must not consume a top_k slot
        # that belongs to this tenant (or to public/untagged content) --
        # which of the two legitimately-visible chunks wins the tie is
        # implementation-defined and not what this test is about.
        store = _store_with_two_tenants(tmp_path)
        results = store.search(SHARED_EMBEDDING, top_k=1, tenant_id=1)
        assert len(results) == 1
        assert results[0].chunk_id in {"tenant-1-chunk", "untagged-chunk"}

    def test_search_bm25_filters_by_tenant(self, tmp_path):
        store = _store_with_two_tenants(tmp_path)
        results = store.search_bm25("text", top_k=10, tenant_id=1)
        assert {r.chunk_id for r in results} == {"tenant-1-chunk", "untagged-chunk"}

    def test_search_bm25_without_tenant_id_returns_everyone(self, tmp_path):
        store = _store_with_two_tenants(tmp_path)
        results = store.search_bm25("text", top_k=10)
        assert {r.chunk_id for r in results} == {"tenant-1-chunk", "tenant-2-chunk", "untagged-chunk"}


class TestHybridSearchForwardsTenantId:
    def test_only_requesting_tenants_chunk_survives_fusion(self, tmp_path, monkeypatch):
        store = _store_with_two_tenants(tmp_path)
        monkeypatch.setattr("app.services.hybrid_search.embed_query", lambda query: SHARED_EMBEDDING)

        results = hybrid_search("text", store, top_k=10, tenant_id=1)

        assert {r.chunk_id for r in results} == {"tenant-1-chunk", "untagged-chunk"}


class TestRetrieveForwardsTenantId:
    def test_retrieve_filters_by_tenant_end_to_end(self, tmp_path, monkeypatch):
        store = _store_with_two_tenants(tmp_path)
        monkeypatch.setattr("app.services.retrieval_service.embed_query", lambda query: SHARED_EMBEDDING)
        monkeypatch.setattr("app.services.hybrid_search.embed_query", lambda query: SHARED_EMBEDDING)

        results = retrieve("text", store, top_k=10, min_score=0.0, tenant_id=2)

        assert {r.chunk_id for r in results} == {"tenant-2-chunk", "untagged-chunk"}

    def test_retrieve_without_tenant_id_is_unfiltered(self, tmp_path, monkeypatch):
        store = _store_with_two_tenants(tmp_path)
        monkeypatch.setattr("app.services.retrieval_service.embed_query", lambda query: SHARED_EMBEDDING)
        monkeypatch.setattr("app.services.hybrid_search.embed_query", lambda query: SHARED_EMBEDDING)

        results = retrieve("text", store, top_k=10, min_score=0.0)

        assert {r.chunk_id for r in results} == {"tenant-1-chunk", "tenant-2-chunk", "untagged-chunk"}


class TestGetChunksByDocumentTenantFiltering:
    def test_wrong_tenant_gets_no_chunks(self, tmp_path):
        store = _store_with_two_tenants(tmp_path)
        assert store.get_chunks_by_document("doc-1", tenant_id=2) == []

    def test_matching_tenant_gets_its_chunks(self, tmp_path):
        store = _store_with_two_tenants(tmp_path)
        chunks = store.get_chunks_by_document("doc-1", tenant_id=1)
        assert [c.chunk_id for c in chunks] == ["tenant-1-chunk"]

    def test_no_tenant_id_is_unfiltered(self, tmp_path):
        store = _store_with_two_tenants(tmp_path)
        chunks = store.get_chunks_by_document("doc-3")
        assert [c.chunk_id for c in chunks] == ["untagged-chunk"]

    def test_untagged_document_excluded_from_specific_tenant(self, tmp_path):
        store = _store_with_two_tenants(tmp_path)
        assert store.get_chunks_by_document("doc-3", tenant_id=1) == []


class FakeLLMClient:
    def generate(self, prompt: str) -> str:
        return "a concise summary"


class TestSummarizeDocumentTenantFiltering:
    def test_wrong_tenant_raises_document_not_found(self, tmp_path):
        store = _store_with_two_tenants(tmp_path)
        with pytest.raises(DocumentNotFoundError):
            summarize_document("doc-1", store, FakeLLMClient(), tenant_id=2)

    def test_matching_tenant_succeeds(self, tmp_path):
        store = _store_with_two_tenants(tmp_path)
        summary, chunks = summarize_document("doc-1", store, FakeLLMClient(), tenant_id=1)
        assert summary == "a concise summary"
        assert [c.chunk_id for c in chunks] == ["tenant-1-chunk"]

    def test_no_tenant_id_is_unfiltered(self, tmp_path):
        store = _store_with_two_tenants(tmp_path)
        summary, chunks = summarize_document("doc-2", store, FakeLLMClient())
        assert [c.chunk_id for c in chunks] == ["tenant-2-chunk"]


class TestSessionStoreTenantBinding:
    def test_same_tenant_reuses_its_own_session(self):
        store = InMemorySessionStore()
        session_id = store.create_session(tenant_id=1)
        assert store.get_or_create_session(session_id, tenant_id=1) == session_id

    def test_different_known_tenant_gets_a_fresh_session(self):
        store = InMemorySessionStore()
        session_id = store.create_session(tenant_id=1)

        result = store.get_or_create_session(session_id, tenant_id=2)

        assert result != session_id
        assert store.get_session_count() == 2

    def test_unknown_requester_tenant_is_permissive(self):
        # DB disabled / no auth context (tenant_id=None) — can't verify
        # ownership either way, so the existing session is still returned.
        store = InMemorySessionStore()
        session_id = store.create_session(tenant_id=1)
        assert store.get_or_create_session(session_id, tenant_id=None) == session_id

    def test_session_with_unknown_owner_is_permissive(self):
        # A session created before tenant tracking existed (or with the
        # DB disabled) has tenant_id=None — unknown ownership isn't
        # treated as a mismatch.
        store = InMemorySessionStore()
        session_id = store.create_session()  # tenant_id defaults to None
        assert store.get_or_create_session(session_id, tenant_id=1) == session_id

    def test_unknown_session_id_still_creates_new_session(self):
        store = InMemorySessionStore()
        result = store.get_or_create_session("does-not-exist", tenant_id=1)
        assert result != "does-not-exist"
        assert store.session_exists(result)
