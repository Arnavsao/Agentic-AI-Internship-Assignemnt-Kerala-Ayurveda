"""
RAG Pipeline — Complete Orchestration
========================================

WHY THIS EXISTS:
This is the "conductor" that orchestrates the entire RAG process:
  1. Check cache → return immediately if hit
  2. Run hybrid retrieval (semantic + BM25 + reranking)
  3. Generate answer from retrieved context
  4. Cache the response
  5. Log the query for analytics

In the original codebase, all this was mixed into AyurvedaRAGSystem.
By separating it, we can:
  - Test each component independently
  - Swap components (e.g., different retriever, different LLM)
  - Add new steps (e.g., guardrails, content filtering) without touching others
  - Monitor each step's latency independently

TEACHING POINT — Dependency Injection:
Notice how RAGPipeline.__init__ takes all its dependencies as parameters.
It doesn't create its own ChromaDB client or LLM provider — those are
passed in from the outside. This is called Dependency Injection, and it's
the single most important pattern for testable, maintainable code.

In tests, you pass mock objects. In production, you pass real ones.
The pipeline doesn't know or care which.
"""

import asyncio
import logging
import time
from typing import Optional

from backend.app.core.config import get_settings
from backend.app.core.logging import LogTimer
from backend.app.services.llm import LLMProvider
from backend.app.services.cache import ResponseCache
from backend.app.services.ingestion.service import ingest
from backend.app.services.rag.embeddings import get_dense_embedder
from backend.app.services.rag.retriever import HybridRetriever
from backend.app.services.rag.vectorstore import QdrantStore, get_qdrant_client
from backend.app.services.rag.generator import (
    agenerate_answer, generate_answer, QueryResponse, Citation,
)

logger = logging.getLogger(__name__)

NO_RESULTS_ANSWER = (
    "I couldn't find relevant information in the knowledge base to answer this "
    "question. Please try rephrasing or ask about Kerala Ayurveda products, "
    "treatments, or concepts."
)


class RAGPipeline:
    """
    Production RAG pipeline with hybrid search, caching, and observability.

    Lifecycle:
      1. initialize() — Load embeddings, build/load index, create retriever
      2. query() — Answer user questions (the hot path)
      3. reindex() — Rebuild index when documents change

    The original AyurvedaRAGSystem.answer_user_query() is now
    RAGPipeline.query() — same logic, better architecture.
    """

    def __init__(self):
        self.settings = get_settings()
        self.llm_provider = LLMProvider()
        self.cache = ResponseCache()
        self.store: Optional[QdrantStore] = None
        self.retriever: Optional[HybridRetriever] = None
        self.index_version: int = 0
        self.chunk_count: int = 0
        self._initialized = False

    def initialize(self, force_reindex: bool = False) -> None:
        """
        Initialize the RAG pipeline.

        1. Load the dense embedding model (determines the vector dimension)
        2. Ensure the Qdrant collection exists with dense + sparse vectors
        3. Assert the live collection matches the configured model
        4. Run incremental ingestion (unchanged files are skipped)
        5. Create the hybrid retriever

        Called once at application startup via FastAPI lifespan.

        Note there is no "reuse existing index" fast path any more. Ingestion
        is now incremental and hash-based, so an unchanged corpus costs a few
        file hashes rather than a full re-embed — and unlike the old fast path,
        it can't silently serve an index built by a different model.
        """
        with LogTimer(logger, "rag_initialization"):
            embedder = get_dense_embedder()

            self.store = QdrantStore(
                client=get_qdrant_client(),
                collection=self.settings.qdrant_collection,
                dense_dim=embedder.dim,
                embedding_model=self.settings.embedding_model,
                sparse_model=self.settings.sparse_model,
            )

            if force_reindex:
                logger.info(
                    "Force reindex requested — dropping collection",
                    extra={"component": "rag"},
                )
                self.store.drop_collection()

            created = self.store.ensure_collection()
            if not created:
                # Fails loudly rather than serving a mismatched vector space.
                self.store.assert_compatible()

            stats = ingest(
                store=self.store,
                content_dir=self.settings.content_path,
                force=force_reindex,
            )

            self.index_version = stats.index_version
            self.chunk_count = self.store.count()
            self.cache.set_index_version(self.index_version)

            self.retriever = HybridRetriever(store=self.store)
            self._initialized = True

            logger.info(
                f"RAG pipeline ready: {self.chunk_count} chunks, index v{self.index_version}",
                extra={
                    "component": "rag",
                    "chunk_count": self.chunk_count,
                    "index_version": self.index_version,
                },
            )

    async def aquery(self, user_query: str, use_cache: bool = True) -> QueryResponse:
        """
        Async entry point — what the API routes call.

        Retrieval is CPU-bound (embedding the query, then cross-encoder
        reranking) and runs on a worker thread; generation is network-bound
        and uses the provider's native async path. The previous `query()` did
        all of this inline in an async route, so a single request blocked the
        event loop for the whole LLM round trip and no other request could be
        served meanwhile.
        """
        if not self._initialized:
            raise RuntimeError("RAG pipeline not initialized. Call initialize() first.")

        start_time = time.perf_counter()

        if use_cache:
            cached = self.cache.get_response(user_query)
            if cached:
                logger.info(
                    f"Cache HIT for query: {user_query[:60]}",
                    extra={"component": "rag", "cache_hit": True},
                )
                return QueryResponse(
                    answer=cached["answer"],
                    citations=[Citation(**c) for c in cached["citations"]],
                    retrieved_chunks=cached.get("retrieved_chunks", []),
                    cache_hit=True,
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                )

        with LogTimer(logger, "retrieval", query=user_query[:100]):
            chunks = await asyncio.to_thread(self.retriever.retrieve, user_query)

        if not chunks:
            logger.warning(
                f"No chunks retrieved for query: {user_query[:60]}",
                extra={"component": "rag"},
            )
            return QueryResponse(
                answer=NO_RESULTS_ANSWER,
                citations=[],
                retrieved_chunks=[],
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )

        with LogTimer(logger, "generation", query=user_query[:100]):
            response = await agenerate_answer(user_query, chunks, self.llm_provider)

        response.latency_ms = (time.perf_counter() - start_time) * 1000

        if use_cache:
            self.cache.set_response(user_query, self._cache_payload(response))

        logger.info(
            f"Query answered in {response.latency_ms:.0f}ms: {user_query[:60]}",
            extra={
                "component": "rag",
                "latency_ms": response.latency_ms,
                "chunks_retrieved": len(chunks),
                "citations": len(response.citations),
            },
        )
        return response

    @staticmethod
    def _cache_payload(response: QueryResponse) -> dict:
        return {
            "answer": response.answer,
            "citations": [
                {
                    "doc_id": c.doc_id,
                    "section_id": c.section_id,
                    "content_snippet": c.content_snippet,
                    "relevance_score": c.relevance_score,
                }
                for c in response.citations
            ],
            "retrieved_chunks": response.retrieved_chunks[:5],
        }

    def query(self, user_query: str, use_cache: bool = True) -> QueryResponse:
        """
        Synchronous query path.

        Retained for scripts and evaluation that run outside an event loop.
        Request handlers should call `aquery` instead.

        Args:
            user_query: The user's question
            use_cache: Whether to check/set cache (disable for evaluation)
        """
        if not self._initialized:
            raise RuntimeError("RAG pipeline not initialized. Call initialize() first.")

        start_time = time.perf_counter()

        # ── Step 1: Check cache ──
        if use_cache:
            cached = self.cache.get_response(user_query)
            if cached:
                logger.info(
                    f"Cache HIT for query: {user_query[:60]}",
                    extra={"component": "rag", "cache_hit": True}
                )
                return QueryResponse(
                    answer=cached["answer"],
                    citations=[Citation(**c) for c in cached["citations"]],
                    retrieved_chunks=cached.get("retrieved_chunks", []),
                    cache_hit=True,
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                )

        # ── Step 2: Hybrid retrieval ──
        with LogTimer(logger, "retrieval", query=user_query[:100]):
            chunks = self.retriever.retrieve(user_query)

        if not chunks:
            logger.warning(
                f"No chunks retrieved for query: {user_query[:60]}",
                extra={"component": "rag"}
            )
            return QueryResponse(
                answer=NO_RESULTS_ANSWER,
                citations=[],
                retrieved_chunks=[],
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )

        # ── Step 3: Generate answer ──
        with LogTimer(logger, "generation", query=user_query[:100]):
            response = generate_answer(user_query, chunks, self.llm_provider)

        response.latency_ms = (time.perf_counter() - start_time) * 1000

        # ── Step 4: Cache the response ──
        if use_cache:
            self.cache.set_response(user_query, self._cache_payload(response))

        logger.info(
            f"Query answered in {response.latency_ms:.0f}ms: {user_query[:60]}",
            extra={
                "component": "rag",
                "latency_ms": response.latency_ms,
                "chunks_retrieved": len(chunks),
                "citations": len(response.citations),
            }
        )

        return response

    def sync(self, force: bool = False) -> dict:
        """
        Bring the index in sync with the content directory.

        By default this is incremental: files whose hash is unchanged are
        skipped without re-embedding. Pass force=True to rebuild everything
        from scratch (needed after changing the embedding model).
        """
        if force:
            logger.info("Force reindexing all documents...", extra={"component": "rag"})
            self.initialize(force_reindex=True)
            return {
                "status": "completed",
                "mode": "full_rebuild",
                "chunks_indexed": self.chunk_count,
                "index_version": self.index_version,
            }

        if not self._initialized:
            raise RuntimeError("RAG pipeline not initialized. Call initialize() first.")

        logger.info("Incremental reindex...", extra={"component": "rag"})
        stats = ingest(store=self.store, content_dir=self.settings.content_path)

        self.index_version = stats.index_version
        self.chunk_count = self.store.count()
        self.cache.set_index_version(self.index_version)

        return {"status": "completed", "mode": "incremental", **stats.as_dict()}

    # Kept for backwards compatibility with the existing /reindex route.
    def reindex(self) -> dict:
        return self.sync(force=True)

    @property
    def stats(self) -> dict:
        """Pipeline statistics for health checks."""
        return {
            "initialized": self._initialized,
            "chunks_indexed": self.chunk_count,
            "index_version": self.index_version,
            "collection": self.settings.qdrant_collection,
            "embedding_model": self.settings.embedding_model,
            "cache": self.cache.stats,
            "llm": self.llm_provider.status(),
        }


# ── Global singleton ──
# The RAG pipeline is expensive to initialize (loads embedding model,
# builds indices). We create it once and reuse across all requests.
_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create the global RAG pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
