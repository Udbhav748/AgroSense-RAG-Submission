"""Concurrency tests for FAISSVectorStore's shared threading.Lock.

Before this, only add_embeddings/delete_document/save held the lock —
search()/search_bm25()/get_chunks_by_document()/list_document_ids()/
first_chunk_vector()/total_vectors()/load() didn't. delete_document()
reassigns self._index and self._metadata as two separate steps, not one
atomic swap, so a read landing between them could score against the new,
rebuilt index but resolve positions against the old, now-mismatched
metadata list.

These tests don't try to reproduce that race by chance (thread scheduling
makes that flaky); they force the exact interleaving deterministically by
making delete_document() block mid-critical-section on a
threading.Event, then assert a concurrent search() can't complete until
delete_document() releases the lock — the actual guarantee the fix
provides, regardless of how the GIL happens to schedule anything.
"""

import threading

from app.models.document import EmbeddedChunk
from app.services.faiss_vector_store import FAISSVectorStore

FAKE_EMBEDDING_DIM = 4
EMBEDDING_A = [1.0, 0.0, 0.0, 0.0]
EMBEDDING_B = [0.0, 1.0, 0.0, 0.0]


def _store_with_two_documents(tmp_path):
    store = FAISSVectorStore(index_path=tmp_path / "i.faiss", metadata_path=tmp_path / "m.json")
    store.create_index(dimension=FAKE_EMBEDDING_DIM)
    store.add_embeddings(
        [
            EmbeddedChunk(
                chunk_id="doc-1-chunk",
                document_id="doc-1",
                embedding=EMBEDDING_A,
                metadata={"chunk_index": 0, "total_chunks": 1, "source": "pdf", "text": "doc one text"},
            ),
            EmbeddedChunk(
                chunk_id="doc-2-chunk",
                document_id="doc-2",
                embedding=EMBEDDING_B,
                metadata={"chunk_index": 0, "total_chunks": 1, "source": "pdf", "text": "doc two text"},
            ),
        ]
    )
    return store


class TestSearchBlocksOnConcurrentDeleteDocument:
    def test_search_does_not_complete_until_delete_document_releases_the_lock(self, tmp_path):
        store = _store_with_two_documents(tmp_path)

        delete_holds_lock = threading.Event()
        release_delete = threading.Event()

        # _bm25_index.rebuild() is a plain Python call delete_document()
        # makes while still holding self._lock, right before returning —
        # the natural point to pause it mid-critical-section without
        # touching faiss's C++-backed Index object.
        original_rebuild = store._bm25_index.rebuild

        def blocking_rebuild(metadata):
            delete_holds_lock.set()
            release_delete.wait(timeout=5)
            return original_rebuild(metadata)

        store._bm25_index.rebuild = blocking_rebuild

        delete_thread = threading.Thread(target=store.delete_document, args=("doc-1",))
        delete_thread.start()
        assert delete_holds_lock.wait(timeout=2), "delete_document never reached the blocking point"

        search_completed = threading.Event()
        search_results = []

        def do_search():
            search_results.extend(store.search(EMBEDDING_A, top_k=5))
            search_completed.set()

        search_thread = threading.Thread(target=do_search)
        search_thread.start()

        # delete_document is still holding the lock inside blocking_rebuild
        # — a search that could run concurrently would finish almost
        # instantly, so a short wait here is enough to prove it's blocked,
        # not just slow.
        assert not search_completed.wait(timeout=0.3), "search() ran concurrently with delete_document() instead of blocking"

        release_delete.set()
        delete_thread.join(timeout=5)
        search_thread.join(timeout=5)

        assert search_completed.is_set()
        # By the time search() actually ran, delete_document had already
        # committed — doc-1 is gone, so its chunk must not appear.
        assert all(chunk.document_id != "doc-1" for chunk in search_results)

    def test_get_chunks_by_document_does_not_complete_until_delete_document_releases_the_lock(self, tmp_path):
        store = _store_with_two_documents(tmp_path)

        delete_holds_lock = threading.Event()
        release_delete = threading.Event()
        original_rebuild = store._bm25_index.rebuild

        def blocking_rebuild(metadata):
            delete_holds_lock.set()
            release_delete.wait(timeout=5)
            return original_rebuild(metadata)

        store._bm25_index.rebuild = blocking_rebuild

        delete_thread = threading.Thread(target=store.delete_document, args=("doc-1",))
        delete_thread.start()
        assert delete_holds_lock.wait(timeout=2)

        read_completed = threading.Event()

        def do_read():
            store.get_chunks_by_document("doc-2")
            read_completed.set()

        read_thread = threading.Thread(target=do_read)
        read_thread.start()

        assert not read_completed.wait(timeout=0.3), "get_chunks_by_document() ran concurrently with delete_document()"

        release_delete.set()
        delete_thread.join(timeout=5)
        read_thread.join(timeout=5)
        assert read_completed.is_set()
