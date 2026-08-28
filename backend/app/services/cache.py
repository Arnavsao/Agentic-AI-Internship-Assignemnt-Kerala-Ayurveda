"""
Response Cache — Two-Layer Caching Strategy
=============================================

WHY CACHING MATTERS:
Without caching, every identical query costs:
  - ~50ms for embedding the query
  - ~100ms for vector search
  - ~20ms for BM25 search
  - ~200ms for cross-encoder reranking
  - ~2000ms for LLM generation
  - ~$0.001 in LLM API cost (per query with Gemini Flash)

With caching, a repeated query costs ~1ms (dict lookup) and $0.00.

At 1000 queries/day with 30% repeat rate, caching saves:
  - 300 × 2 seconds = 10 minutes of compute time
  - 300 × $0.001 = $0.30/day = $9/month in API costs

This seems small, but with expensive models (GPT-4, Claude) and higher
volume, caching savings become significant.

TWO LAYERS:
1. In-memory LRU cache (always available, fast, process-local)
2. Redis (optional, survives restarts, shared across workers)

If Redis isn't configured, we fall back to in-memory only.
This means:
  - Development: Just works (no Redis needed)
  - Production: Add REDIS_URL for persistence and multi-worker sharing
"""

import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Any, Optional

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)


class LRUCache:
    """
    Simple in-memory LRU (Least Recently Used) cache.

    WHY not just use functools.lru_cache:
    lru_cache works on function arguments, but our cache key depends on
    query + context content (which changes as documents are updated).
    We need explicit control over cache invalidation.

    HOW LRU WORKS:
    When the cache is full and a new item arrives:
    1. Evict the Least Recently Used item (the one not accessed the longest)
    2. Add the new item

    This is implemented using an OrderedDict — items are ordered by access time.
    Accessing an item moves it to the end. Eviction removes from the front.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 86400):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache. Returns None if not found or expired."""
        if key not in self._cache:
            self._misses += 1
            return None

        value, timestamp = self._cache[key]

        # Check TTL expiration
        if time.time() - timestamp > self.ttl_seconds:
            del self._cache[key]
            self._misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        return value

    def set(self, key: str, value: Any) -> None:
        """Set a value in cache, evicting LRU items if at capacity."""
        # If key exists, update it
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = (value, time.time())
            return

        # Evict LRU items if at capacity
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)  # Remove oldest

        self._cache[key] = (value, time.time())

    def invalidate(self, key: str) -> None:
        """Remove a specific key from cache."""
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear entire cache (e.g., after re-indexing documents)."""
        self._cache.clear()
        logger.info("Cache cleared", extra={"component": "cache"})

    @property
    def stats(self) -> dict:
        """Cache statistics for monitoring."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }


class ResponseCache:
    """
    Two-layer response cache with in-memory LRU and optional Redis.

    Usage:
        cache = ResponseCache()

        # Check cache
        cached = cache.get_response("What is Ashwagandha?")
        if cached:
            return cached  # Skip RAG pipeline entirely

        # Run RAG pipeline...
        response = rag_pipeline(query)

        # Cache the response
        cache.set_response("What is Ashwagandha?", response)
    """

    def __init__(self):
        settings = get_settings()
        self._memory_cache = LRUCache(
            max_size=settings.cache_max_size,
            ttl_seconds=settings.cache_ttl_seconds,
        )
        self._redis = None
        self._index_version = 0

        # Try to connect to Redis if configured
        if settings.redis_url:
            try:
                import redis
                self._redis = redis.from_url(settings.redis_url)
                self._redis.ping()  # Test connection
                logger.info("Redis cache connected", extra={"component": "cache"})
            except Exception as e:
                logger.warning(
                    f"Redis not available, using in-memory cache only: {e}",
                    extra={"component": "cache"}
                )
                self._redis = None
        else:
            logger.info(
                "No REDIS_URL configured, using in-memory cache only",
                extra={"component": "cache"}
            )

    def set_index_version(self, version: int) -> None:
        """
        Point the cache at an index generation.

        Cache keys embed this version, so re-indexing automatically orphans
        every previously cached answer instead of serving stale results built
        from documents that have since changed. Old entries age out via TTL.
        """
        if version != self._index_version:
            logger.info(
                f"Cache namespace moved to index v{version}",
                extra={"component": "cache", "index_version": version},
            )
            self._index_version = version
            self._memory_cache.clear()

    def _make_key(self, query: str) -> str:
        """
        Build a cache key.

        The key covers everything that changes what a correct answer looks
        like: the index generation, the embedding and reranker models, and the
        retrieval depth — not just the query text. Keying on query text alone
        (the previous behaviour) meant that swapping the embedding model or
        re-indexing kept serving answers produced by the old configuration.
        """
        settings = get_settings()
        normalized = query.strip().lower()
        fingerprint = "|".join([
            settings.embedding_model,
            settings.reranker_model,
            str(settings.retrieval_top_k),
            str(settings.retrieval_top_n),
            str(settings.retrieval_context_n),
            normalized,
        ])
        digest = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
        return f"rag:v{self._index_version}:{digest}"

    def get_response(self, query: str) -> Optional[dict]:
        """
        Look up a cached response for this query.
        Checks in-memory first, then Redis.
        """
        key = self._make_key(query)

        # Layer 1: In-memory (fastest)
        cached = self._memory_cache.get(key)
        if cached is not None:
            logger.debug(f"Cache HIT (memory): {query[:50]}", extra={"component": "cache"})
            return cached

        # Layer 2: Redis (if available)
        if self._redis:
            try:
                redis_val = self._redis.get(key)
                if redis_val:
                    data = json.loads(redis_val)
                    # Promote to memory cache for faster subsequent access
                    self._memory_cache.set(key, data)
                    logger.debug(f"Cache HIT (Redis): {query[:50]}", extra={"component": "cache"})
                    return data
            except Exception as e:
                logger.warning(f"Redis get failed: {e}", extra={"component": "cache"})

        return None

    def set_response(self, query: str, response: dict) -> None:
        """
        Cache a response in both layers.
        The response dict should be JSON-serializable.
        """
        key = self._make_key(query)
        settings = get_settings()

        # Layer 1: In-memory
        self._memory_cache.set(key, response)

        # Layer 2: Redis (if available)
        if self._redis:
            try:
                self._redis.setex(
                    key,
                    settings.cache_ttl_seconds,
                    json.dumps(response, default=str),
                )
            except Exception as e:
                logger.warning(f"Redis set failed: {e}", extra={"component": "cache"})

    def invalidate_all(self) -> None:
        """
        Clear all cached responses.
        Call this when documents are re-indexed (cached answers may be stale).
        """
        self._memory_cache.clear()
        if self._redis:
            try:
                # SCAN, not KEYS: KEYS blocks the Redis event loop for the
                # duration of a full keyspace walk, which stalls every other
                # client. SCAN yields in bounded batches.
                for key in self._redis.scan_iter(match="rag:v*", count=500):
                    self._redis.delete(key)
            except Exception as e:
                logger.warning(f"Redis invalidation failed: {e}", extra={"component": "cache"})

        logger.info("All cached responses invalidated", extra={"component": "cache"})

    @property
    def stats(self) -> dict:
        """Combined cache statistics."""
        return {
            "memory": self._memory_cache.stats,
            "redis_available": self._redis is not None,
        }
