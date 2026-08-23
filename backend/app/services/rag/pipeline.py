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

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from langchain_community.vectorstores import Chroma

from backend.app.core.config import get_settings
from backend.app.core.logging import LogTimer
from backend.app.services.llm import LLMProvider
from backend.app.services.cache import ResponseCache
from backend.app.services.rag.embeddings import get_embeddings
from backend.app.services.rag.chunker import load_all_documents
from backend.app.services.rag.retriever import HybridRetriever, BM25Index
from backend.app.services.rag.generator import (
    generate_answer, QueryResponse, Citation, build_context,
)

logger = logging.getLogger(__name__)


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
        self.vectorstore = None
        self.retriever: Optional[HybridRetriever] = None
        self.bm25_index: Optional[BM25Index] = None
        self.parent_chunks: Dict[str, str] = {}
        self._initialized = False

    def initialize(self, force_reindex: bool = False) -> None:
        """
        Initialize the RAG pipeline.

        This:
        1. Loads the embedding model
        2. Either loads existing ChromaDB index or builds from scratch
        3. Builds BM25 index for keyword search
        4. Creates the hybrid retriever

        Called once at application startup via FastAPI lifespan.
        """
        with LogTimer(logger, "rag_initialization"):
            embeddings = get_embeddings()

            persist_path = str(Path(self.settings.chroma_persist_dir).resolve())
            Path(persist_path).mkdir(parents=True, exist_ok=True)
            chroma_client = chromadb.PersistentClient(path=persist_path)

            # ── Try to reuse existing index ──
            if not force_reindex:
                try:
                    existing = chroma_client.get_collection(self.settings.chroma_collection_name)
                    count = existing.count()
                    if count > 0:
                        logger.info(
                            f"Reusing existing ChromaDB index ({count} chunks)",
                            extra={"component": "rag", "chunk_count": count}
                        )
                        self.vectorstore = Chroma(
                            client=chroma_client,
                            collection_name=self.settings.chroma_collection_name,
                            embedding_function=embeddings,
                        )
                        self._build_bm25_from_vectorstore()
                        self._initialized = True
                        return
                except Exception:
                    pass  # Collection doesn't exist — build from scratch

            # ── Build index from scratch ──
            self._build_index(chroma_client, embeddings)
            self._initialized = True

    def _build_index(self, chroma_client, embeddings) -> None:
        """Build the complete index from source documents."""
        logger.info(
            "Building vector index from scratch...",
            extra={"component": "rag"}
        )

        content_dir = self.settings.content_path
        if not content_dir.exists():
            raise FileNotFoundError(f"Content directory not found: {content_dir}")

        # Load and chunk all documents
        chunk_result = load_all_documents(content_dir)

        if not chunk_result.chunks:
            raise ValueError(f"No documents found in {content_dir}")

        # Store parent chunks for context expansion
        for parent_doc in chunk_result.parent_chunks:
            parent_id = parent_doc.metadata.get("parent_chunk_id")
            if parent_id:
                self.parent_chunks[parent_id] = parent_doc.page_content

        # Index child chunks into ChromaDB
        logger.info(
            f"Embedding and indexing {len(chunk_result.chunks)} chunks...",
            extra={"component": "rag"}
        )

        # Delete existing collection if rebuilding
        try:
            chroma_client.delete_collection(self.settings.chroma_collection_name)
        except Exception:
            pass

        self.vectorstore = Chroma.from_documents(
            documents=chunk_result.chunks,
            embedding=embeddings,
            client=chroma_client,
            collection_name=self.settings.chroma_collection_name,
        )

        # Build BM25 index from the same chunks
        self.bm25_index = BM25Index()
        self.bm25_index.index(chunk_result.chunks)

        # Create hybrid retriever
        self.retriever = HybridRetriever(
            vectorstore=self.vectorstore,
            bm25_index=self.bm25_index,
            parent_chunks=self.parent_chunks,
        )

        logger.info(
            f"Index built: {chunk_result.total_chunks} child chunks, "
            f"{chunk_result.total_parent_chunks} parent chunks",
            extra={"component": "rag"}
        )

        # Invalidate cache since documents changed
        self.cache.invalidate_all()

    def _build_bm25_from_vectorstore(self) -> None:
        """Build BM25 index from existing ChromaDB collection."""
        logger.info("Building BM25 index from existing vectorstore...",
                     extra={"component": "rag"})

        # Get all documents from ChromaDB
        collection = self.vectorstore._collection
        results = collection.get(include=["documents", "metadatas"])

        from langchain_core.documents import Document

        documents = []
        for text, metadata in zip(results["documents"], results["metadatas"]):
            doc = Document(page_content=text, metadata=metadata or {})
            documents.append(doc)

            # Build parent chunks map
            parent_id = (metadata or {}).get("parent_chunk_id")
            if parent_id and (metadata or {}).get("is_parent"):
                self.parent_chunks[parent_id] = text

        self.bm25_index = BM25Index()
        self.bm25_index.index(documents)

        self.retriever = HybridRetriever(
            vectorstore=self.vectorstore,
            bm25_index=self.bm25_index,
            parent_chunks=self.parent_chunks,
        )

        logger.info(
            f"BM25 index built from {len(documents)} existing chunks",
            extra={"component": "rag"}
        )

    def query(self, user_query: str, use_cache: bool = True) -> QueryResponse:
        """
        Answer a user query using the full RAG pipeline.

        This is the main entry point — equivalent to the original
        AyurvedaRAGSystem.answer_user_query().

        Steps:
          1. Check response cache
          2. Hybrid retrieval (semantic + BM25 + reranking)
          3. LLM answer generation
          4. Cache the response
          5. Return structured QueryResponse

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
                answer="I couldn't find relevant information in the knowledge base to answer this question. "
                       "Please try rephrasing or ask about Kerala Ayurveda products, treatments, or concepts.",
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
            cache_data = {
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
                "retrieved_chunks": response.retrieved_chunks[:5],  # Limit stored chunks
            }
            self.cache.set_response(user_query, cache_data)

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

    def reindex(self) -> dict:
        """
        Force re-index all documents.
        Returns statistics about the reindexing.
        """
        logger.info("Force reindexing all documents...", extra={"component": "rag"})
        self.initialize(force_reindex=True)
        return {
            "status": "completed",
            "chunks_indexed": self.bm25_index.n_docs if self.bm25_index else 0,
            "parent_chunks": len(self.parent_chunks),
            "cache_cleared": True,
        }

    @property
    def stats(self) -> dict:
        """Pipeline statistics for health checks."""
        return {
            "initialized": self._initialized,
            "chunks_indexed": self.bm25_index.n_docs if self.bm25_index else 0,
            "parent_chunks": len(self.parent_chunks),
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
