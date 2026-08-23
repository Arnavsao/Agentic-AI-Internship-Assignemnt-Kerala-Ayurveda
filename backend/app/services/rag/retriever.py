"""
Hybrid Retriever — Semantic + BM25 with Cross-Encoder Reranking
==================================================================

WHY HYBRID SEARCH:
Your original retrieval is semantic-only (cosine similarity on embeddings).
This works great for conceptual queries ("How does Ayurveda view stress?")
but fails on:
  - Exact keyword queries: "product KA-P001" → embeddings don't encode IDs well
  - Rare terms: "Shirodhara" → the embedding model may not have seen this word
  - Boolean queries: "Ashwagandha AND pregnancy" → embeddings don't do AND

BM25 (keyword search) excels at exactly these cases. By combining both:
  - Semantic search finds conceptually related content
  - BM25 finds exact keyword matches
  - Reciprocal Rank Fusion (RRF) merges both result sets

TEACHING POINT — Reciprocal Rank Fusion (RRF):
Given results from N ranking systems, RRF combines them using:
  score(doc) = Σ 1 / (k + rank_i(doc))
where rank_i is the rank in system i, and k=60 is a smoothing constant.

This is elegant because:
  - It doesn't require the scores from different systems to be comparable
  - A document ranked high by BOTH systems gets a much higher combined score
  - A document ranked high by ONE system still gets some credit
  - It's parameter-free (k=60 works well in practice)

WHY RERANKING:
After hybrid search gives us ~10 candidates, we rerank with a cross-encoder.
A bi-encoder (like our embedding model) encodes query and document SEPARATELY,
then compares their vectors. A cross-encoder encodes query AND document TOGETHER,
attending to their interactions. This is much more accurate but 100x slower,
which is why we only use it for 10 candidates, not 10,000.
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from langchain_core.documents import Document

from backend.app.core.config import get_settings
from backend.app.core.logging import LogTimer

logger = logging.getLogger(__name__)

# Global reranker singleton (loaded lazily)
_reranker = None


@dataclass
class RetrievedChunk:
    """A retrieved chunk with all its scores and provenance."""
    document: Document
    semantic_score: float = 0.0       # Cosine similarity from vector search
    bm25_score: float = 0.0          # BM25 keyword match score
    rrf_score: float = 0.0           # Combined Reciprocal Rank Fusion score
    rerank_score: float = 0.0        # Cross-encoder reranking score
    final_score: float = 0.0         # The score used for final ranking

    # Provenance
    doc_id: str = ""
    section_id: str = ""
    parent_content: Optional[str] = None  # Parent chunk content (richer context)

    def __post_init__(self):
        if self.document and self.document.metadata:
            self.doc_id = self.document.metadata.get("doc_id", "")
            self.section_id = self.document.metadata.get("section_id", "")


class BM25Index:
    """
    Simple BM25 implementation for keyword search.

    WHY not use a library:
    For <10,000 documents, a simple Python BM25 is fast enough (<10ms)
    and avoids adding another dependency. If you scale to millions of
    documents, you'd switch to Elasticsearch or Meilisearch.

    HOW BM25 WORKS:
    BM25 scores a document based on:
    1. Term Frequency (TF): How often does the query term appear in this doc?
    2. Inverse Document Frequency (IDF): How rare is this term across all docs?
    3. Document Length normalization: Don't favor long documents just because
       they contain more words.

    A document scores high if it contains rare query terms frequently.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # Term frequency saturation parameter
        self.b = b     # Length normalization parameter
        self.documents: List[Document] = []
        self.doc_freqs: Dict[str, int] = defaultdict(int)  # term → num docs containing it
        self.doc_term_freqs: List[Dict[str, int]] = []     # per-doc term frequencies
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.n_docs: int = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercase tokenizer."""
        import re
        # Split on non-alphanumeric, lowercase, filter short tokens
        tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
        return [t for t in tokens if len(t) > 1]

    def index(self, documents: List[Document]) -> None:
        """Build the BM25 index from a list of documents."""
        self.documents = documents
        self.n_docs = len(documents)
        self.doc_term_freqs = []
        self.doc_lengths = []
        self.doc_freqs = defaultdict(int)

        for doc in documents:
            tokens = self._tokenize(doc.page_content)
            self.doc_lengths.append(len(tokens))

            # Count term frequencies for this document
            tf = defaultdict(int)
            for token in tokens:
                tf[token] += 1
            self.doc_term_freqs.append(dict(tf))

            # Count document frequency (how many docs contain each term)
            for term in set(tokens):
                self.doc_freqs[term] += 1

        self.avg_doc_length = (
            sum(self.doc_lengths) / self.n_docs if self.n_docs > 0 else 1.0
        )

        logger.info(
            f"BM25 index built: {self.n_docs} documents, {len(self.doc_freqs)} unique terms",
            extra={"component": "retriever"}
        )

    def search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Search the BM25 index and return top-k results with scores."""
        if not self.documents:
            return []

        query_tokens = self._tokenize(query)
        scores = []

        for idx in range(self.n_docs):
            score = 0.0
            doc_len = self.doc_lengths[idx]
            doc_tf = self.doc_term_freqs[idx]

            for term in query_tokens:
                if term not in self.doc_freqs:
                    continue

                # IDF: log((N - df + 0.5) / (df + 0.5))
                df = self.doc_freqs[term]
                idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

                # TF with saturation and length normalization
                tf = doc_tf.get(term, 0)
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                )

                score += idf * tf_norm

            scores.append((self.documents[idx], score))

        # Sort by score descending, return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


def get_reranker():
    """
    Load cross-encoder reranker (lazy singleton).

    WHY cross-encoder for reranking:
    A bi-encoder (embedding model) encodes query and document independently,
    then compares vectors. It's fast (encode once, compare many) but misses
    nuanced query-document interactions.

    A cross-encoder takes BOTH query and document as input together, allowing
    attention across them. "Does this passage about Ashwagandha actually answer
    the question about pregnancy safety?" — the cross-encoder can reason about
    this, the bi-encoder can't.

    The trade-off: cross-encoders are ~100x slower per comparison.
    So we use bi-encoder to find 10 candidates, then cross-encoder to pick
    the best 3-5. Two-stage retrieval.
    """
    global _reranker
    if _reranker is not None:
        return _reranker

    settings = get_settings()
    logger.info(
        f"Loading reranker: {settings.reranker_model}",
        extra={"component": "retriever"}
    )

    from sentence_transformers import CrossEncoder
    _reranker = CrossEncoder(settings.reranker_model)

    logger.info("Reranker loaded", extra={"component": "retriever"})
    return _reranker


def reciprocal_rank_fusion(
    result_lists: List[List[Tuple[Document, float]]],
    k: int = 60,
) -> List[Tuple[Document, float]]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    Each document gets a score of Σ 1/(k + rank_i) across all lists.
    k=60 is the standard smoothing constant from the original RRF paper.
    """
    # Use content hash as document key (since the same chunk might appear
    # in both semantic and BM25 results)
    doc_scores: Dict[str, float] = defaultdict(float)
    doc_map: Dict[str, Tuple[Document, float, float]] = {}

    for list_idx, results in enumerate(result_lists):
        for rank, (doc, score) in enumerate(results):
            key = doc.metadata.get("content_hash", doc.page_content[:100])
            rrf_contribution = 1.0 / (k + rank + 1)
            doc_scores[key] += rrf_contribution

            # Keep track of individual scores
            if key not in doc_map:
                doc_map[key] = (doc, 0.0, 0.0)
            existing = doc_map[key]
            if list_idx == 0:  # Semantic search
                doc_map[key] = (doc, score, existing[2])
            else:  # BM25
                doc_map[key] = (doc, existing[1], score)

    # Sort by RRF score
    sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)

    results = []
    for key in sorted_keys:
        doc, sem_score, bm25_score = doc_map[key]
        results.append((doc, doc_scores[key]))

    return results


class HybridRetriever:
    """
    Two-stage hybrid retriever:
    1. Retrieve candidates via semantic search (ChromaDB) + BM25 (keyword)
    2. Merge with Reciprocal Rank Fusion
    3. Rerank with cross-encoder
    4. Optionally expand to parent chunks for richer context

    This replaces the simple similarity_search_with_relevance_scores() call
    in the original rag_system.py.
    """

    def __init__(self, vectorstore, bm25_index: BM25Index, parent_chunks: Optional[Dict[str, str]] = None):
        """
        Args:
            vectorstore: ChromaDB vector store (for semantic search)
            bm25_index: BM25 keyword index
            parent_chunks: Map of parent_chunk_id → parent content (for context expansion)
        """
        self.vectorstore = vectorstore
        self.bm25_index = bm25_index
        self.parent_chunks = parent_chunks or {}

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
        use_reranking: bool = True,
        expand_to_parent: bool = True,
    ) -> List[RetrievedChunk]:
        """
        Full hybrid retrieval pipeline.

        Args:
            query: User's search query
            top_k: Number of candidates from each search method (default: from config)
            top_n: Number of results after reranking (default: from config)
            use_reranking: Whether to apply cross-encoder reranking
            expand_to_parent: Whether to expand child chunks to parent chunks

        Returns:
            List of RetrievedChunk with scores and provenance
        """
        settings = get_settings()
        top_k = top_k or settings.retrieval_top_k
        top_n = top_n or settings.retrieval_top_n

        with LogTimer(logger, "hybrid_retrieval", query=query[:100]):
            # ── Stage 1: Dual retrieval ──
            # Run semantic search and BM25 in parallel (they're independent)

            # Semantic search (ChromaDB)
            with LogTimer(logger, "semantic_search"):
                try:
                    semantic_results = self.vectorstore.similarity_search_with_relevance_scores(
                        query, k=top_k
                    )
                except Exception as e:
                    logger.error(f"Semantic search failed: {e}", extra={"component": "retriever"})
                    semantic_results = []

            # BM25 keyword search
            with LogTimer(logger, "bm25_search"):
                bm25_results = self.bm25_index.search(query, k=top_k)

            # ── Stage 2: Reciprocal Rank Fusion ──
            fused_results = reciprocal_rank_fusion(
                [semantic_results, bm25_results]
            )

            logger.info(
                f"Hybrid search: {len(semantic_results)} semantic + {len(bm25_results)} BM25 "
                f"→ {len(fused_results)} fused candidates",
                extra={"component": "retriever"}
            )

            # ── Stage 3: Cross-encoder reranking ──
            if use_reranking and fused_results:
                with LogTimer(logger, "cross_encoder_reranking"):
                    reranked = self._rerank(query, fused_results, top_n)
            else:
                reranked = fused_results[:top_n]

            # ── Stage 4: Build RetrievedChunk objects with parent expansion ──
            retrieved_chunks = []
            for doc, score in reranked:
                chunk = RetrievedChunk(
                    document=doc,
                    final_score=score,
                )

                # Expand to parent chunk for richer context
                if expand_to_parent:
                    parent_id = doc.metadata.get("parent_chunk_id")
                    if parent_id and parent_id in self.parent_chunks:
                        chunk.parent_content = self.parent_chunks[parent_id]

                retrieved_chunks.append(chunk)

            return retrieved_chunks

    def _rerank(
        self,
        query: str,
        candidates: List[Tuple[Document, float]],
        top_n: int,
    ) -> List[Tuple[Document, float]]:
        """Rerank candidates using cross-encoder."""
        try:
            reranker = get_reranker()

            # Prepare query-document pairs for cross-encoder
            pairs = [(query, doc.page_content) for doc, _ in candidates]
            scores = reranker.predict(pairs)

            # Combine with documents
            scored = list(zip([doc for doc, _ in candidates], scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            logger.debug(
                f"Reranked {len(candidates)} candidates → top {top_n}",
                extra={"component": "retriever"}
            )

            return [(doc, float(score)) for doc, score in scored[:top_n]]

        except Exception as e:
            logger.warning(
                f"Reranking failed, using RRF scores: {e}",
                extra={"component": "retriever"}
            )
            return candidates[:top_n]
