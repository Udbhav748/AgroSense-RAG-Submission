"""Regression test for a real bug found during Module 10 gap-closure
(the eval-potato-02 faithfulness investigation): scripts/run_rag_eval.py's
execute_retrieval() called retrieval_service.retrieve() with a keyword
argument (rerank_candidates) that does not exist on the real function --
the actual parameter is `rerank`. This raised a TypeError on every single
call, silently caught by execute_retrieval's broad `except Exception`,
degrading every retrieval in this eval script's history to a raw
vector-only fallback (no hybrid BM25, no collection filter, no
reranking) without ever surfacing an error.

This test deliberately does NOT mock retrieval_service.retrieve -- a
mock would happily accept any keyword argument and hide exactly the bug
this test exists to catch. It uses the REAL function (with a minimal
real FAISSVectorStore) so a signature mismatch fails loudly here, not
silently in a `except Exception: logger.debug(...)` block three call
frames away.
"""

from __future__ import annotations

import inspect

import pytest

from app.services.retrieval_service import retrieve
from scripts.run_rag_eval import execute_retrieval


class _FakeVectorStore:
    """Minimal vector store stand-in: only needs to exist so
    execute_retrieval's `if vector_store is not None` branch is taken and
    the real retrieve() function is actually invoked."""

    def search(self, *args, **kwargs):
        return []


class TestExecuteRetrievalCallsRealRetrieveSignatureCorrectly:
    def test_execute_retrieval_never_hits_the_exception_fallback_path(self, monkeypatch, caplog):
        """The real bug: execute_retrieval's kwargs to retrieve() didn't
        match retrieve()'s real signature, so every call TypeError'd and
        was silently caught by `except Exception: logger.debug(...)`,
        degrading every retrieval in this eval script's history to the
        raw-vector/file-based fallback below without ever surfacing an
        error. Patch embed_query (the only real network/model dependency
        retrieve() has) so this test runs fully offline, but call the
        REAL retrieve() function -- not a mock -- so an incorrect keyword
        argument raises here exactly as it did in production, instead of
        being hidden behind a mock that accepts any keyword.

        The precise regression pin: the "Live retrieve() call failed"
        warning (emitted only when retrieve() itself raised) must never
        fire for a call using valid, current keyword arguments.
        """
        # retrieval_service.py does `from app.services.embedding_service
        # import embed_query` -- patching the origin module doesn't touch
        # that already-bound name, so the patch target must be the
        # importing module's own namespace.
        monkeypatch.setattr(
            "app.services.retrieval_service.embed_query", lambda query: [1.0] + [0.0] * 383
        )

        from app.services.faiss_vector_store import FAISSVectorStore

        store = FAISSVectorStore(index_path=None, metadata_path=None)  # type: ignore[arg-type]
        store.create_index(dimension=384)

        with caplog.at_level("WARNING"):
            execute_retrieval(
                query="test query",
                top_k=5,
                crop="potato",
                vector_store=store,
                hybrid=True,
                rerank_flag=True,
            )

        assert "Live retrieve() call failed" not in caplog.text

    def test_retrieve_signature_has_rerank_not_rerank_candidates(self):
        """Direct signature pin: if retrieve()'s parameter is ever renamed
        again, this fails immediately and explicitly, instead of via a
        silently-swallowed TypeError three layers away."""
        params = inspect.signature(retrieve).parameters
        assert "rerank" in params
        assert "rerank_candidates" not in params

    def test_execute_retrieval_source_uses_the_real_parameter_name(self):
        """Belt-and-suspenders: read execute_retrieval's own source and
        confirm it passes `rerank=`, not the old broken `rerank_candidates=`,
        to retrieve(). Pins the actual fix, not just the callee's shape."""
        source = inspect.getsource(execute_retrieval)
        assert "rerank_candidates=" not in source
        assert "rerank=" in source
