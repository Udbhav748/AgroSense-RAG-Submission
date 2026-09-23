"""Unit test suite for SemanticQueryCache and low-latency cache integrations."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.models.document import RetrievedChunk
from app.models.schemas import ChatResponse, DiagnosisInfo, SourceReference
from app.services.cache_service import (
    AdaptiveCacheService,
    SemanticQueryCache,
    cache_service,
    cached,
    cosine_similarity,
    normalize_text,
)

VALID_HEADERS = {"X-API-Key": settings.api_key}


class TestSemanticQueryCacheExactHits:
    """Test exact match caching, normalization, and agricultural scoping."""

    def test_exact_query_hit_and_metadata(self) -> None:
        cache = SemanticQueryCache(capacity=50, ttl=3600)
        query = "What are the symptoms of tomato early blight?"
        payload = {
            "answer": "Symptoms include concentric brown rings on lower leaves.",
            "sources": [{"document_id": "doc1", "filename": "tomato_guide.pdf", "page": 2}],
        }

        # Cache miss before setting
        assert cache.get(query, crop="tomato", disease="early blight") is None
        assert cache.misses == 1
        assert cache.hits == 0

        # Store in cache
        cache.set(query, payload, crop="tomato", disease="early blight")
        assert cache.size == 1

        # Cache hit
        cached_result = cache.get(query, crop="tomato", disease="early blight")
        assert cached_result is not None
        assert cached_result["answer"] == payload["answer"]
        assert cached_result["metadata"]["cached"] is True
        assert cache.hits == 1
        assert cache.hit_ratio == 0.5  # 1 hit, 1 miss

    def test_exact_query_normalization_invariance(self) -> None:
        cache = SemanticQueryCache(capacity=50)
        base_query = "How to treat Tomato Early Blight?"
        payload = {"answer": "Apply copper fungicides weekly."}

        cache.set(base_query, payload, crop="tomato", disease="early blight")

        # Test various case, punctuation, and whitespace variations
        variations = [
            "how to treat tomato early blight",
            "  HOW TO TREAT TOMATO EARLY BLIGHT!  ",
            "How to treat, tomato: early blight???",
            "how   to   treat   tomato   early   blight",
            "How to treat tomato early-blight",
        ]

        for variant in variations:
            res = cache.get(variant, crop="tomato", disease="early blight")
            assert res is not None, f"Failed match for variant: {variant}"
            assert res["answer"] == "Apply copper fungicides weekly."
            assert res["metadata"]["cached"] is True

    def test_crop_and_disease_isolation(self) -> None:
        cache = SemanticQueryCache(capacity=50)
        query = "Recommended fungicide dosage"

        cache.set(query, {"answer": "Tomato: 2.5g/L"}, crop="tomato", disease="early blight")
        cache.set(query, {"answer": "Potato: 3.0g/L"}, crop="potato", disease="late blight")

        # Querying tomato early blight
        res_tomato = cache.get(query, crop="tomato", disease="early blight")
        assert res_tomato is not None
        assert res_tomato["answer"] == "Tomato: 2.5g/L"

        # Querying potato late blight
        res_potato = cache.get(query, crop="potato", disease="late blight")
        assert res_potato is not None
        assert res_potato["answer"] == "Potato: 3.0g/L"

        # Querying mismatched crop/disease returns None
        assert cache.get(query, crop="tomato", disease="late blight") is None
        assert cache.get(query, crop="apple", disease="scab") is None


class TestSemanticQueryCacheSemanticHits:
    """Test semantic vector similarity matching and threshold bounds."""

    def test_semantic_similarity_hit(self) -> None:
        # Mock embedding mapping
        embedding_dict = {
            "what are the symptoms of tomato leaf curl virus": [1.0, 0.0, 0.0, 0.0],
            "tell me tomato leaf curl symptoms": [0.98, 0.02, 0.0, 0.0],  # Cosine sim > 0.99
            "how do i harvest sweet corn": [0.0, 0.0, 1.0, 0.0],  # Orthogonal vector (0.0)
        }

        def mock_embed(text: str) -> list[float]:
            norm = normalize_text(text)
            return embedding_dict.get(norm, [0.5, 0.5, 0.5, 0.5])

        cache = SemanticQueryCache(
            capacity=50,
            similarity_threshold=0.96,
            embedding_service=mock_embed,
        )

        q1 = "What are the symptoms of tomato leaf curl virus?"
        payload = {"answer": "Upward curling and yellowing of leaf margins."}

        cache.set(q1, payload, crop="tomato", disease="leaf curl")

        # Query 2 is semantically similar (> 0.96 threshold)
        q2 = "Tell me tomato leaf curl symptoms"
        res_semantic = cache.get(q2, crop="tomato", disease="leaf curl")

        assert res_semantic is not None
        assert res_semantic["answer"] == payload["answer"]
        assert res_semantic["metadata"]["cached"] is True
        assert res_semantic["metadata"]["semantic_hit"] is True
        assert res_semantic["metadata"]["similarity_score"] >= 0.96
        assert cache.semantic_hits == 1
        assert cache.hits == 1

        # Query 3 is unrelated -> cache miss
        q3 = "How do I harvest sweet corn?"
        res_unrelated = cache.get(q3, crop="corn")
        assert res_unrelated is None
        assert cache.misses == 1

    def test_semantic_similarity_threshold_boundary(self) -> None:
        # Sim = (1*0.959) = 0.959 (below 0.96) vs (1*0.961) = 0.961 (above 0.96)
        embed_map = {
            "query base": [1.0, 0.0],
            "query below threshold": [0.955, 0.296],  # cos sim ~ 0.955
            "query above threshold": [0.970, 0.243],  # cos sim ~ 0.970
        }

        def mock_embed(text: str) -> list[float]:
            return embed_map.get(normalize_text(text), [0.0, 1.0])

        cache = SemanticQueryCache(
            capacity=10,
            similarity_threshold=0.96,
            embedding_service=mock_embed,
        )

        cache.set("query base", {"answer": "Base answer"})

        # Below threshold -> miss
        assert cache.get("query below threshold") is None

        # Above threshold -> hit
        res = cache.get("query above threshold")
        assert res is not None
        assert res["answer"] == "Base answer"
        assert res["metadata"]["cached"] is True


class TestSemanticQueryCacheTTLExpiration:
    """Test TTL expiration and cleanup."""

    def test_ttl_expiration(self) -> None:
        cache = SemanticQueryCache(capacity=10, ttl=1)
        cache.set("ephemeral query", {"data": "test_data"}, ttl=1)

        # Immediate retrieval succeeds
        assert cache.get("ephemeral query") is not None

        # Simulate time moving past TTL
        current_time = time.monotonic()
        with patch("time.monotonic", return_value=current_time + 2.0):
            assert cache.get("ephemeral query") is None
            assert cache.size == 0


class TestSemanticQueryCacheLRUEviction:
    """Test capacity enforcement and least-recently-used eviction."""

    def test_lru_eviction_order(self) -> None:
        cache = SemanticQueryCache(capacity=3)

        cache.set("q1", {"id": 1})
        cache.set("q2", {"id": 2})
        cache.set("q3", {"id": 3})

        # Access q1 to make it most recently used (q2 becomes oldest)
        _ = cache.get("q1")

        # Insert q4 -> should evict q2
        cache.set("q4", {"id": 4})

        assert cache.get("q1") is not None
        assert cache.get("q2") is None  # Evicted
        assert cache.get("q3") is not None
        assert cache.get("q4") is not None
        assert cache.size == 3


class TestSemanticQueryCacheInvalidation:
    """Test cache invalidation and targeted item deletion."""

    def test_invalidate_and_clear(self) -> None:
        cache = SemanticQueryCache(capacity=10)
        cache.set("q1", "ans1")
        cache.set("q2", "ans2")
        cache.set("q3", "ans3")
        assert cache.size == 3

        cache.invalidate()
        assert cache.size == 0
        assert cache.get("q1") is None
        assert cache.get("q2") is None

    def test_delete_specific_entry(self) -> None:
        cache = SemanticQueryCache(capacity=10)
        cache.set("tomato blight", "ans_tomato", crop="tomato", disease="blight")
        cache.set("potato scab", "ans_potato", crop="potato", disease="scab")

        # Delete tomato blight
        deleted = cache.delete("tomato blight", crop="tomato", disease="blight")
        assert deleted is True

        assert cache.get("tomato blight", crop="tomato", disease="blight") is None
        assert cache.get("potato scab", crop="potato", disease="scab") is not None


class TestSemanticQueryCacheConcurrencySafety:
    """Test multi-threaded concurrent access safety."""

    def test_concurrent_reads_writes_and_invalidations(self) -> None:
        cache = SemanticQueryCache(capacity=100, ttl=60)
        errors: list[Exception] = []

        def worker(thread_id: int) -> None:
            try:
                for i in range(50):
                    q = f"query_{thread_id}_{i % 10}"
                    crop = f"crop_{thread_id % 3}"
                    disease = f"disease_{i % 5}"
                    cache.set(
                        q,
                        {"thread": thread_id, "iter": i},
                        crop=crop,
                        disease=disease,
                        query_embedding=[float(thread_id), float(i)],
                    )

                    _ = cache.get(q, crop=crop, disease=disease)
                    _ = cache.get_metrics()

                    if i == 25 and thread_id == 0:
                        cache.invalidate()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrency errors occurred: {errors}"
        metrics = cache.get_metrics()
        assert isinstance(metrics["hits"], int)
        assert isinstance(metrics["misses"], int)
        assert isinstance(metrics["hit_ratio"], float)


class TestAdaptiveCacheServiceCompatibility:
    """Test backward compatibility of AdaptiveCacheService wrapper."""

    def test_adaptive_cache_service_in_memory_interface(self) -> None:
        cache = AdaptiveCacheService(max_in_memory_items=20)
        assert cache.active_engine == "in_memory"

        key = cache._generate_key("chat", query="organic pesticide")
        cache.set(key, {"answer": "Neem oil spray"})

        cached_data = cache.get(key)
        assert cached_data is not None
        assert cached_data["answer"] == "Neem oil spray"

        stats = cache.get_stats()
        assert stats["engine"] == "in_memory"
        assert stats["hits"] == 1


class TestCacheAdminRoute:
    """Test DELETE /cache admin endpoint."""

    def test_delete_cache_admin_endpoint(self) -> None:
        client = TestClient(app)

        # Seed global cache
        cache_service.set("test_endpoint_query", {"answer": "Cached answer"})
        assert cache_service.size > 0

        # Invalidate via DELETE /cache with admin permission
        with patch("app.core.permissions.check_permission"):
            response = client.delete("/cache", headers=VALID_HEADERS)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "invalidated"

        # Verify cache is cleared
        assert cache_service.size == 0
        assert cache_service.get("test_endpoint_query") is None

    def test_get_cache_metrics_endpoint(self) -> None:
        client = TestClient(app)
        response = client.get("/cache/metrics", headers=VALID_HEADERS)
        assert response.status_code == 200
        metrics = response.json()
        assert "hits" in metrics
        assert "misses" in metrics
        assert "hit_ratio" in metrics
        assert "size" in metrics


class TestRAGServiceCacheIntegration:
    """Test handle_query and handle_diagnose caching behavior."""

    def test_handle_query_cache_hit_returns_sub_50ms_cached_response(self) -> None:
        from app.services.rag_service import ChatService

        mock_vector_store = MagicMock()
        mock_vector_store.total_vectors.return_value = 10
        mock_llm_client = MagicMock()
        mock_llm_client.generate.return_value = "Tomato blight is treated with copper sulfate."

        chat_service = ChatService(mock_vector_store, mock_llm_client)

        # Invalidate cache before testing
        cache_service.invalidate()

        mock_chunk = RetrievedChunk(
            text="Tomato early blight treatment details...",
            document_id="doc-123",
            chunk_id="chunk-1",
            score=0.92,
            metadata={"source": "tomato_manual.pdf", "page": 1},
        )

        with patch("app.services.rag_service.retrieve", return_value=[mock_chunk]):
            # 1. First invocation: cache miss, runs LLM generation
            res1 = chat_service.handle_query("What controls tomato early blight?")
            assert res1.answer == "Tomato blight is treated with copper sulfate."
            assert mock_llm_client.generate.call_count == 1

            # 2. Second invocation: instant cache hit, bypasses LLM generation
            res2 = chat_service.handle_query("What controls tomato early blight?")
            assert res2.answer == "Tomato blight is treated with copper sulfate."
            assert res2.metadata.get("cached") is True
            # LLM call count must NOT have increased!
            assert mock_llm_client.generate.call_count == 1

    def test_handle_diagnose_cache_hit_returns_instant_cached_response(self) -> None:
        from app.services.vision_client import VisionPrediction
        from app.services.rag_service import ChatService

        mock_vector_store = MagicMock()
        mock_vector_store.total_vectors.return_value = 10
        mock_llm_client = MagicMock()
        mock_llm_client.generate.return_value = "Apply Daconil fungicide."

        chat_service = ChatService(mock_vector_store, mock_llm_client)
        cache_service.invalidate()

        mock_prediction = VisionPrediction(
            crop="Tomato",
            disease="Early Blight",
            confidence=0.97,
            raw_class="Tomato___Early_blight",
            low_confidence=False,
        )

        mock_chunk = RetrievedChunk(
            text="Early blight management recommendations...",
            document_id="doc-999",
            chunk_id="chunk-9",
            score=0.95,
            metadata={"source": "plant_pathology.pdf", "page": 5},
        )

        with (
            patch(
                "app.services.rag_service.diagnose_image",
                return_value=mock_prediction,
            ),
            patch("app.services.rag_service.retrieve", return_value=[mock_chunk]),
        ):
            fake_image_bytes = b"fake_leaf_image_bytes"

            # First run -> generates
            diag_res1 = chat_service.handle_diagnose(
                fake_image_bytes, "leaf.jpg", "image/jpeg", query="How to treat this?"
            )
            assert diag_res1.answer == "Apply Daconil fungicide."
            assert mock_llm_client.generate.call_count == 1

            # Second run -> served from cache
            diag_res2 = chat_service.handle_diagnose(
                fake_image_bytes, "leaf.jpg", "image/jpeg", query="How to treat this?"
            )
            assert diag_res2.answer == "Apply Daconil fungicide."
            assert diag_res2.metadata.get("cached") is True
            assert mock_llm_client.generate.call_count == 1
