"""
Hybrid Retriever — Qdrant Fusion + Cross-Encoder Reranking
============================================================

WHY HYBRID SEARCH:
Pure semantic search handles conceptual queries well ("How does Ayurveda view
stress?") but misses the cases that matter most in a product/health corpus:

  - Exact identifiers: "product KA-P001" — embeddings encode IDs poorly
  - Rare terms: "Shirodhara" — may be out-of-distribution for the encoder
  - Precise safety keywords, where a near-miss is worse than no answer

Keyword (BM25) search covers exactly those. Running both and fusing the
rankings gets both behaviours.

WHAT CHANGED:
Fusion used to happen client-side — a Python BM25 index plus a hand-written
RRF, with the BM25 index rebuilt at every startup by scanning the entire
collection through Chroma's private API. Qdrant stores a sparse vector next
to the dense one and fuses them internally, so this module now issues one
query and startup touches nothing.

RECIPROCAL RANK FUSION:
  score(doc) = Σ 1 / (k + rank_i(doc))
Rank-based, so scores from different systems never need to be comparable, and
parameter-free in practice. Qdrant computes it server-side.

WHY RERANK AFTER FUSION:
The embedding model is a bi-encoder — query and document are encoded
separately and compared as vectors, which is fast but blind to their
interaction. A cross-encoder reads query and document together and can judge
"does this passage about Ashwagandha actually answer the pregnancy-safety
question?" It is far slower per pair, so it only ever sees the ~10 fused
candidates, never the full collection.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from backend.app.core.config import get_settings
from backend.app.core.logging import LogTimer
from backend.app.services.rag.embeddings import get_dense_embedder, get_sparse_embedder
from backend.app.services.rag.vectorstore import QdrantStore

logger = logging.getLogger(__name__)

# Cross-encoder singleton, loaded on first use.
_reranker = None


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its scores and provenance."""
    document: Document
    fusion_score: float = 0.0        # RRF score from Qdrant
    rerank_score: float = 0.0        # Cross-encoder score
    final_score: float = 0.0         # Score used for final ranking

    doc_id: str = ""
    section_id: str = ""
    parent_content: Optional[str] = None  # Larger surrounding chunk, if any

    def __post_init__(self):
        if self.document and self.document.metadata:
            self.doc_id = self.document.metadata.get("doc_id", "")
            self.section_id = self.document.metadata.get("section_id", "")

    @property
    def context_content(self) -> str:
        """Parent chunk when available, else the chunk itself."""
        return self.parent_content or self.document.page_content


def get_reranker():
    """
    Load the cross-encoder reranker (lazy singleton).

    ms-marco-MiniLM-L-6-v2 scores ~10 pairs in 50-100ms on CPU. Larger
    rerankers such as bge-reranker-base are more accurate but take 0.5-1.5s
    for the same batch, which is a real hit on a hot path serving a corpus
    this small. Override with RERANKER_MODEL if that trade changes.
    """
    global _reranker
    if _reranker is not None:
        return _reranker

    settings = get_settings()
    logger.info(
        f"Loading reranker: {settings.reranker_model}",
        extra={"component": "retriever"},
    )

    from sentence_transformers import CrossEncoder
    _reranker = CrossEncoder(settings.reranker_model)

    logger.info("Reranker loaded", extra={"component": "retriever"})
    return _reranker


def reset_reranker() -> None:
    """Drop the cached reranker. Used by tests."""
    global _reranker
    _reranker = None


def _payload_to_document(payload: Dict[str, Any]) -> Document:
    """Rebuild a LangChain Document from a Qdrant payload."""
    content = payload.get("content", "")
    metadata = {k: v for k, v in payload.items() if k not in ("content", "parent_content")}
    return Document(page_content=content, metadata=metadata)


class HybridRetriever:
    """
    Two-stage retrieval:
      1. Qdrant fuses dense + sparse candidates with RRF (server-side)
      2. Cross-encoder reranks the survivors
      3. Each result carries its parent chunk for richer LLM context
    """

    def __init__(self, store: QdrantStore):
        self.store = store

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
        use_reranking: bool = True,
    ) -> List[RetrievedChunk]:
        """
        Args:
            query: the user's question
            top_k: candidates to fuse from each branch (default: config)
            top_n: results to keep after reranking (default: config)
            use_reranking: disable to inspect raw fusion ordering
        """
        settings = get_settings()
        top_k = top_k or settings.retrieval_top_k
        top_n = top_n or settings.retrieval_top_n

        with LogTimer(logger, "hybrid_retrieval", query=query[:100]):
            dense = get_dense_embedder()
            sparse = get_sparse_embedder()

            with LogTimer(logger, "query_embedding"):
                dense_vec = dense.encode_query(query)
                sparse_vec = sparse.encode_query(query)

            with LogTimer(logger, "qdrant_hybrid_search"):
                try:
                    fused = self.store.hybrid_query(dense_vec, sparse_vec, limit=top_k)
                except Exception as e:
                    logger.error(
                        f"Hybrid search failed: {e}",
                        extra={"component": "retriever"},
                    )
                    return []

            logger.info(
                f"Hybrid search returned {len(fused)} fused candidates",
                extra={"component": "retriever"},
            )

            if not fused:
                return []

            if use_reranking:
                with LogTimer(logger, "cross_encoder_reranking"):
                    ranked = self._rerank(query, fused, top_n)
            else:
                ranked = [(p, s, s) for p, s in fused[:top_n]]

            chunks = []
            for payload, fusion_score, final_score in ranked:
                chunk = RetrievedChunk(
                    document=_payload_to_document(payload),
                    fusion_score=fusion_score,
                    rerank_score=final_score if use_reranking else 0.0,
                    final_score=final_score,
                    parent_content=payload.get("parent_content"),
                )
                chunks.append(chunk)

            return chunks

    def _rerank(
        self,
        query: str,
        candidates: List[Tuple[Dict[str, Any], float]],
        top_n: int,
    ) -> List[Tuple[Dict[str, Any], float, float]]:
        """
        Rerank with the cross-encoder.

        Returns (payload, fusion_score, rerank_score) sorted by rerank score.
        Falls back to fusion order if the reranker fails — a degraded ranking
        still answers the question, an exception does not.
        """
        try:
            reranker = get_reranker()
            pairs = [(query, p.get("content", "")) for p, _ in candidates]
            scores = reranker.predict(pairs)

            scored = [
                (payload, fusion_score, float(rerank_score))
                for (payload, fusion_score), rerank_score in zip(candidates, scores)
            ]
            scored.sort(key=lambda x: x[2], reverse=True)

            logger.debug(
                f"Reranked {len(candidates)} candidates → top {top_n}",
                extra={"component": "retriever"},
            )
            return scored[:top_n]

        except Exception as e:
            logger.warning(
                f"Reranking failed, falling back to fusion order: {e}",
                extra={"component": "retriever"},
            )
            return [(p, s, s) for p, s in candidates[:top_n]]
