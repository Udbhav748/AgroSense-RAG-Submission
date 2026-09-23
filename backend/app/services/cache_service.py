"""Semantic Query Cache & Low-Latency Engine for AgroSense-RAG.

Provides thread-safe in-memory LRU caching with:
1. Exact normalized query matching (lowercased, stripped punctuation, crop/disease tuple keying).
2. Semantic similarity matching via cosine similarity over query embeddings (threshold >= 0.96).
3. Configurable capacity and TTL with automatic LRU eviction.
4. Comprehensive cache metrics (hits, misses, semantic_hits, hit_ratio).
5. Redis integration fallback for distributed deployments.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import re
import threading
import time
from collections import OrderedDict
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from app.core.config import settings

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


def normalize_text(text: str | None) -> str:
    """Normalize text by lowercasing, stripping punctuation, and collapsing whitespace."""
    if not text:
        return ""
    # Lowercase
    cleaned = text.lower().strip()
    # Replace punctuation characters with spaces to preserve word boundaries
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    # Collapse multiple whitespaces
    return re.sub(r"\s+", " ", cleaned).strip()


def cosine_similarity(
    vec1: list[float] | tuple[float, ...] | Any,
    vec2: list[float] | tuple[float, ...] | Any,
) -> float:
    """Compute cosine similarity between two float vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    try:
        import numpy as np

        v1 = np.asarray(vec1, dtype=np.float32)
        v2 = np.asarray(vec2, dtype=np.float32)
        norm1 = float(np.linalg.norm(v1))
        norm2 = float(np.linalg.norm(v2))
        if norm1 <= 0.0 or norm2 <= 0.0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))
    except Exception:
        # Pure Python fallback
        dot = 0.0
        norm1_sq = 0.0
        norm2_sq = 0.0
        for a, b in zip(vec1, vec2, strict=False):
            dot += a * b
            norm1_sq += a * a
            norm2_sq += b * b
        if norm1_sq <= 0.0 or norm2_sq <= 0.0:
            return 0.0
        return dot / (math.sqrt(norm1_sq) * math.sqrt(norm2_sq))


class CacheEntry:
    """A single cached item with metadata, TTL, and optional embedding."""

    __slots__ = (
        "key",
        "query",
        "raw_query",
        "response",
        "crop",
        "disease",
        "embedding",
        "tenant_id",
        "document_ids",
        "expires_at",
        "created_at",
    )

    def __init__(
        self,
        key: str,
        query: str,
        raw_query: str,
        response: dict[str, Any],
        crop: str | None,
        disease: str | None,
        embedding: list[float] | None,
        tenant_id: int | None,
        document_ids: tuple[str, ...] | None,
        expires_at: float,
        created_at: float,
    ) -> None:
        self.key = key
        self.query = query
        self.raw_query = raw_query
        self.response = response
        self.crop = crop
        self.disease = disease
        self.embedding = embedding
        self.tenant_id = tenant_id
        self.document_ids = document_ids
        self.expires_at = expires_at
        self.created_at = created_at

    def is_expired(self) -> bool:
        """Check if the cache entry has exceeded its time-to-live."""
        return time.monotonic() > self.expires_at


class SemanticQueryCache:
    """In-memory thread-safe LRU cache with semantic similarity matching and TTL expiration."""

    def __init__(
        self,
        capacity: int = 500,
        ttl: int = 3600,
        similarity_threshold: float = 0.96,
        embedding_service: Any | None = None,
    ) -> None:
        self.capacity = capacity
        self.ttl = ttl
        self.similarity_threshold = similarity_threshold
        self._embedding_service = embedding_service
        self._lock = threading.RLock()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0
        self._semantic_hits: int = 0

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def semantic_hits(self) -> int:
        return self._semantic_hits

    @property
    def hit_ratio(self) -> float:
        total = self._hits + self._misses
        return (self._hits / total) if total > 0 else 0.0

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    def _get_embedding(self, text: str) -> list[float] | None:
        """Obtain an embedding vector for the query string."""
        if not text:
            return None

        if self._embedding_service is not None:
            try:
                if callable(self._embedding_service):
                    return self._embedding_service(text)
                if hasattr(self._embedding_service, "embed_query"):
                    return self._embedding_service.embed_query(text)
                if hasattr(self._embedding_service, "get_embedding"):
                    return self._embedding_service.get_embedding(text)
            except Exception as exc:
                logger.debug("Failed to embed query via custom embedding service: %s", exc)
                return None

        # Fallback to default application embedding service
        try:
            from app.services.embedding_service import embed_query

            return embed_query(text)
        except Exception as exc:
            logger.debug("Default embedding service unavailable: %s", exc)
            return None

    def make_cache_key(
        self,
        query: str,
        crop: str | None = None,
        disease: str | None = None,
        tenant_id: int | None = None,
        document_ids: list[str] | tuple[str, ...] | None = None,
    ) -> str:
        """Create a deterministic hash key from normalized query and agricultural context."""
        norm_query = normalize_text(query)
        norm_crop = normalize_text(crop)
        norm_disease = normalize_text(disease)
        tenant = str(tenant_id) if tenant_id is not None else "no-tenant"
        doc_scope = ",".join(sorted(document_ids)) if document_ids else "no-docs"
        raw_key = f"{norm_crop}|{norm_disease}|{norm_query}|{tenant}|{doc_scope}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:32]

    def get(
        self,
        query: str,
        crop: str | None = None,
        disease: str | None = None,
        tenant_id: int | None = None,
        document_ids: list[str] | tuple[str, ...] | None = None,
        query_embedding: list[float] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Retrieve response from cache via exact normalized hash or cosine similarity match."""
        # 1. Exact match lookup
        exact_key = self.make_cache_key(
            query,
            crop=crop,
            disease=disease,
            tenant_id=tenant_id,
            document_ids=document_ids,
        )

        with self._lock:
            lookup_key = None
            if exact_key in self._cache:
                lookup_key = exact_key
            elif query in self._cache:
                lookup_key = query

            if lookup_key is not None:
                entry = self._cache[lookup_key]
                if entry.is_expired():
                    del self._cache[lookup_key]
                else:
                    self._cache.move_to_end(lookup_key, last=True)
                    self._hits += 1
                    response = copy.deepcopy(entry.response)
                    if isinstance(response, dict) and ("answer" in response or "metadata" in response or "diagnosis" in response):
                        if "metadata" not in response or not isinstance(response["metadata"], dict):
                            response["metadata"] = {}
                        response["metadata"]["cached"] = True
                    return response

        # 2. Semantic similarity lookup
        target_embedding = query_embedding or self._get_embedding(query)
        if target_embedding is not None:
            norm_crop = normalize_text(crop)
            norm_disease = normalize_text(disease)
            doc_tuple = tuple(sorted(document_ids)) if document_ids else None

            best_entry: CacheEntry | None = None
            best_key: str | None = None
            best_score: float = -1.0

            with self._lock:
                expired_keys: list[str] = []
                for k, entry in list(self._cache.items()):
                    if entry.is_expired():
                        expired_keys.append(k)
                        continue

                    # Context filters must match
                    if entry.crop != norm_crop or entry.disease != norm_disease:
                        continue
                    if entry.tenant_id != tenant_id or entry.document_ids != doc_tuple:
                        continue
                    if entry.embedding is None:
                        continue

                    similarity = cosine_similarity(target_embedding, entry.embedding)
                    if similarity >= self.similarity_threshold and similarity > best_score:
                        best_score = similarity
                        best_key = k
                        best_entry = entry

                # Purge expired entries encountered during iteration
                for exp_k in expired_keys:
                    self._cache.pop(exp_k, None)

                if best_entry is not None and best_key is not None:
                    self._cache.move_to_end(best_key, last=True)
                    self._hits += 1
                    self._semantic_hits += 1
                    response = copy.deepcopy(best_entry.response)
                    if isinstance(response, dict):
                        if "metadata" not in response or not isinstance(response["metadata"], dict):
                            response["metadata"] = {}
                        response["metadata"]["cached"] = True
                        response["metadata"]["semantic_hit"] = True
                        response["metadata"]["similarity_score"] = round(best_score, 4)
                    return response

        # 3. Cache Miss
        with self._lock:
            self._misses += 1
        return None

    def set(
        self,
        query: str,
        response: dict[str, Any] | Any,
        crop: str | None = None,
        disease: str | None = None,
        query_embedding: list[float] | None = None,
        tenant_id: int | None = None,
        document_ids: list[str] | tuple[str, ...] | None = None,
        ttl: int | None = None,
        **kwargs: Any,
    ) -> bool:
        """Store query response in cache with normalized key and optional embedding."""
        if not query:
            return False

        norm_query = normalize_text(query)
        norm_crop = normalize_text(crop)
        norm_disease = normalize_text(disease)
        doc_tuple = tuple(sorted(document_ids)) if document_ids else None
        exact_key = self.make_cache_key(
            query,
            crop=crop,
            disease=disease,
            tenant_id=tenant_id,
            document_ids=document_ids,
        )

        # Prepare serializable response
        if hasattr(response, "model_dump"):
            stored_response = response.model_dump()
        else:
            stored_response = copy.deepcopy(response)

        # Obtain query embedding if not explicitly provided
        embedding = query_embedding if query_embedding is not None else self._get_embedding(query)

        effective_ttl = ttl if ttl is not None else self.ttl
        expires_at = time.monotonic() + effective_ttl
        created_at = time.monotonic()

        entry = CacheEntry(
            key=exact_key,
            query=norm_query,
            raw_query=query,
            response=stored_response,
            crop=norm_crop,
            disease=norm_disease,
            embedding=embedding,
            tenant_id=tenant_id,
            document_ids=doc_tuple,
            expires_at=expires_at,
            created_at=created_at,
        )

        with self._lock:
            if exact_key in self._cache:
                self._cache.move_to_end(exact_key, last=True)
            else:
                # Evict oldest LRU entry if at capacity
                while len(self._cache) >= self.capacity:
                    self._cache.popitem(last=False)

            self._cache[exact_key] = entry
            return True

    def delete(
        self,
        key_or_query: str,
        crop: str | None = None,
        disease: str | None = None,
        tenant_id: int | None = None,
        document_ids: list[str] | tuple[str, ...] | None = None,
    ) -> bool:
        """Delete an entry by exact key or normalized query."""
        with self._lock:
            if key_or_query in self._cache:
                del self._cache[key_or_query]
                return True

            exact_key = self.make_cache_key(
                key_or_query,
                crop=crop,
                disease=disease,
                tenant_id=tenant_id,
                document_ids=document_ids,
            )
            if exact_key in self._cache:
                del self._cache[exact_key]
                return True
            return False

    def invalidate(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()

    def clear(self) -> None:
        """Alias for invalidate()."""
        self.invalidate()

    def get_metrics(self) -> dict[str, Any]:
        """Return cache performance metrics and current state."""
        with self._lock:
            total = self._hits + self._misses
            ratio = (self._hits / total) if total > 0 else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "semantic_hits": self._semantic_hits,
                "hit_ratio": round(ratio, 4),
                "hit_rate_pct": round(ratio * 100.0, 2),
                "size": len(self._cache),
                "capacity": self.capacity,
                "ttl": self.ttl,
                "similarity_threshold": self.similarity_threshold,
            }

    def get_stats(self) -> dict[str, Any]:
        """Telemetry dictionary format matching existing health endpoints."""
        metrics = self.get_metrics()
        return {
            "engine": "in_memory",
            "is_redis_active": False,
            "hits": metrics["hits"],
            "misses": metrics["misses"],
            "semantic_hits": metrics["semantic_hits"],
            "hit_ratio": metrics["hit_ratio"],
            "hit_rate_pct": metrics["hit_rate_pct"],
            "in_memory_items": metrics["size"],
            "max_in_memory_capacity": metrics["capacity"],
            "ttl": metrics["ttl"],
            "similarity_threshold": metrics["similarity_threshold"],
        }


class AdaptiveCacheService:
    """Adaptive Cache Service supporting Redis with automatic in-memory fallback."""

    def __init__(
        self,
        max_in_memory_items: int = 500,
        similarity_threshold: float = 0.96,
        ttl: int = 3600,
        embedding_service: Any | None = None,
    ) -> None:
        self.max_in_memory_items = max_in_memory_items
        self._semantic_cache = SemanticQueryCache(
            capacity=max_in_memory_items,
            ttl=ttl,
            similarity_threshold=similarity_threshold,
            embedding_service=embedding_service,
        )
        self._redis_client: Any = None
        self._connect_redis()

    def _connect_redis(self) -> None:
        """Initialize Redis connection if REDIS_URL is configured."""
        if not settings.redis_url:
            self._redis_client = None
            return

        try:
            import redis

            client = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=2.0,
                socket_timeout=2.0,
                retry_on_timeout=False,
            )
            client.ping()
            self._redis_client = client
            logger.info("Connected to Redis distributed cache (%s)", settings.redis_url)
        except Exception as e:
            logger.warning("Redis connection failed (%s). Falling back to in-memory cache.", e)
            self._redis_client = None

    @property
    def is_redis_active(self) -> bool:
        if self._redis_client is None:
            return False
        try:
            self._redis_client.ping()
            return True
        except Exception:
            return False

    @property
    def active_engine(self) -> str:
        return "redis" if self.is_redis_active else "in_memory"

    @property
    def hits(self) -> int:
        return self._semantic_cache.hits

    @property
    def misses(self) -> int:
        return self._semantic_cache.misses

    @property
    def hit_ratio(self) -> float:
        return self._semantic_cache.hit_ratio

    def _generate_key(self, prefix: str, *args: Any, **kwargs: Any) -> str:
        """Generate a deterministic hashed cache key."""
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
        key_string = ":".join(key_parts)
        hash_key = hashlib.sha256(key_string.encode("utf-8")).hexdigest()[:32]
        return f"{prefix}:{hash_key}"

    def get(
        self,
        key_or_query: str,
        crop: str | None = None,
        disease: str | None = None,
        tenant_id: int | None = None,
        document_ids: list[str] | tuple[str, ...] | None = None,
        query_embedding: list[float] | None = None,
        **kwargs: Any,
    ) -> Any | None:
        """Get value from active cache (Redis or In-Memory Semantic Cache)."""
        # 1. Check Redis if active
        if self.is_redis_active:
            try:
                raw = self._redis_client.get(key_or_query)
                if raw is not None:
                    try:
                        return json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        return raw
                return None
            except Exception as e:
                logger.debug("Redis get failed (%s), falling back to in-memory", e)

        # 2. Check Semantic In-Memory Cache
        return self._semantic_cache.get(
            query=key_or_query,
            crop=crop,
            disease=disease,
            tenant_id=tenant_id,
            document_ids=document_ids,
            query_embedding=query_embedding,
            **kwargs,
        )

    def set(
        self,
        key_or_query: str,
        value: Any,
        expire: int | timedelta | None = None,
        crop: str | None = None,
        disease: str | None = None,
        query_embedding: list[float] | None = None,
        tenant_id: int | None = None,
        document_ids: list[str] | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> bool:
        """Set value in cache with expiration."""
        ttl_seconds = 3600
        if isinstance(expire, timedelta):
            ttl_seconds = int(expire.total_seconds())
        elif isinstance(expire, int):
            ttl_seconds = expire

        if ttl_seconds <= 0:
            return False

        # 1. Set in Redis if active
        if self.is_redis_active:
            try:
                if isinstance(value, dict | list | int | float | bool):
                    serialized = json.dumps(value, default=str)
                elif hasattr(value, "model_dump"):
                    serialized = json.dumps(value.model_dump(), default=str)
                else:
                    serialized = str(value)
                self._redis_client.setex(key_or_query, ttl_seconds, serialized)
            except Exception as e:
                logger.debug("Redis set failed (%s), proceeding to in-memory", e)

        # 2. Set in In-Memory Semantic Cache
        return self._semantic_cache.set(
            query=key_or_query,
            response=value,
            crop=crop,
            disease=disease,
            query_embedding=query_embedding,
            tenant_id=tenant_id,
            document_ids=document_ids,
            ttl=ttl_seconds,
            **kwargs,
        )

    def delete(self, key_or_query: str, crop: str | None = None, disease: str | None = None) -> bool:
        """Delete key from both Redis and in-memory cache."""
        deleted = False
        if self.is_redis_active:
            try:
                deleted = bool(self._redis_client.delete(key_or_query))
            except Exception as e:
                logger.debug("Redis delete failed (%s)", e)

        mem_deleted = self._semantic_cache.delete(key_or_query, crop=crop, disease=disease)
        return deleted or mem_deleted

    def invalidate(self) -> None:
        """Flush the active cache."""
        if self.is_redis_active:
            try:
                self._redis_client.flushdb()
            except Exception as e:
                logger.debug("Redis flushdb failed (%s)", e)

        self._semantic_cache.invalidate()

    def clear(self) -> None:
        self.invalidate()

    def get_metrics(self) -> dict[str, Any]:
        return self._semantic_cache.get_metrics()

    def get_stats(self) -> dict[str, Any]:
        stats = self._semantic_cache.get_stats()
        stats["engine"] = self.active_engine
        stats["is_redis_active"] = self.is_redis_active
        return stats


# Global singleton instance
cache_service = SemanticQueryCache(capacity=500, ttl=3600, similarity_threshold=0.96)


def cached(
    expire: int | timedelta = 3600,
    key_prefix: str = "cache",
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Decorator for caching async function results."""

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            key_parts = [str(arg) for arg in args]
            key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
            key_string = ":".join(key_parts)
            hash_key = hashlib.sha256(key_string.encode("utf-8")).hexdigest()[:32]
            cache_key = f"{key_prefix}:{func.__name__}:{hash_key}"

            cached_result = cache_service.get(cache_key)
            if cached_result is not None:
                logger.debug("Cache hit for key %s", cache_key)
                return cached_result

            result = await func(*args, **kwargs)
            ttl = int(expire.total_seconds()) if isinstance(expire, timedelta) else int(expire)
            cache_service.set(cache_key, result, ttl=ttl)
            logger.debug("Cached result for key %s", cache_key)
            return result

        return wrapper

    return decorator
