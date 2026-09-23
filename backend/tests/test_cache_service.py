"""Unit test suite for the AdaptiveCacheService and Redis semantic cache."""

import time
from unittest.mock import MagicMock, patch

import pytest

from app.models.schemas import ChatResponse
from app.services.cache_service import AdaptiveCacheService, cached


class TestAdaptiveCacheServiceInMemory:
    """Test in-memory fallback behavior of AdaptiveCacheService."""

    def test_in_memory_set_and_get(self) -> None:
        cache = AdaptiveCacheService(max_in_memory_items=10)
        assert cache.active_engine == "in_memory"
        assert not cache.is_redis_active

        key = cache._generate_key("test", query="tomato blight", tenant_id=1)
        cache.set(key, {"answer": "Apply Copper Hydroxide", "confidence": 0.95})

        cached = cache.get(key)
        assert cached is not None
        assert cached["answer"] == "Apply Copper Hydroxide"
        assert cached["confidence"] == 0.95

    def test_in_memory_ttl_expiration(self) -> None:
        cache = AdaptiveCacheService(max_in_memory_items=10)
        key = cache._generate_key("test", query="short ttl")
        # 1 second TTL
        cache.set(key, {"data": 123}, expire=1)

        # Immediate get
        assert cache.get(key) == {"data": 123}

        # Mock time forward
        with patch("time.monotonic", return_value=time.monotonic() + 2.0):
            assert cache.get(key) is None

    def test_in_memory_lru_eviction(self) -> None:
        cache = AdaptiveCacheService(max_in_memory_items=3)

        key1 = cache._generate_key("test", id=1)
        key2 = cache._generate_key("test", id=2)
        key3 = cache._generate_key("test", id=3)
        key4 = cache._generate_key("test", id=4)

        cache.set(key1, "val1")
        cache.set(key2, "val2")
        cache.set(key3, "val3")

        # Touch key1 so key2 becomes oldest LRU
        _ = cache.get(key1)

        # Insert key4 -> should evict key2
        cache.set(key4, "val4")

        assert cache.get(key1) == "val1"
        assert cache.get(key2) is None
        assert cache.get(key3) == "val3"
        assert cache.get(key4) == "val4"

    def test_in_memory_delete_and_clear(self) -> None:
        cache = AdaptiveCacheService(max_in_memory_items=10)
        key = cache._generate_key("test", action="delete_me")
        cache.set(key, "data")
        assert cache.get(key) == "data"

        assert cache.delete(key) is True
        assert cache.get(key) is None

        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.clear()
        assert cache.get("k1") is None
        assert cache.get("k2") is None

    def test_get_stats_telemetry(self) -> None:
        cache = AdaptiveCacheService(max_in_memory_items=50)
        key = cache._generate_key("test", param="a")

        _ = cache.get(key)  # Miss
        cache.set(key, "hello")
        _ = cache.get(key)  # Hit

        stats = cache.get_stats()
        assert stats["engine"] == "in_memory"
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate_pct"] == 50.0
        assert stats["in_memory_items"] == 1
        assert stats["max_in_memory_capacity"] == 50


class TestAdaptiveCacheServiceRedisMock:
    """Test Redis integration when Redis is available or fails."""

    def test_redis_connected_mode(self) -> None:
        cache = AdaptiveCacheService(max_in_memory_items=10)
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.get.return_value = '{"answer": "Redis Cached Plan"}'
        mock_redis.setex.return_value = True

        cache._redis_client = mock_redis
        assert cache.active_engine == "redis"
        assert cache.is_redis_active is True

        res = cache.get("redis_key")
        assert res == {"answer": "Redis Cached Plan"}
        mock_redis.get.assert_called_once_with("redis_key")

        ok = cache.set("redis_key", {"answer": "New Plan"}, expire=3600)
        assert ok is True
        mock_redis.setex.assert_called_once()

    def test_redis_failure_falls_back_to_in_memory(self) -> None:
        cache = AdaptiveCacheService(max_in_memory_items=10)
        mock_redis = MagicMock()
        mock_redis.ping.side_effect = ConnectionError("Redis down")
        cache._redis_client = mock_redis

        assert cache.is_redis_active is False
        assert cache.active_engine == "in_memory"

        # Should store and retrieve from in-memory store smoothly
        cache.set("fallback_key", "saved_locally")
        assert cache.get("fallback_key") == "saved_locally"


@pytest.mark.asyncio
async def test_cached_decorator() -> None:
    call_count = 0

    @cached(expire=60, key_prefix="test_fn")
    async def expensive_computation(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2

    res1 = await expensive_computation(5)
    assert res1 == 10
    assert call_count == 1

    # Second call should be served from cache
    res2 = await expensive_computation(5)
    assert res2 == 10
    assert call_count == 1

    # Different argument calls function
    res3 = await expensive_computation(6)
    assert res3 == 12
    assert call_count == 2
