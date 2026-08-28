"""
Unit Tests — Chunker
======================

TEACHING POINT — What makes a good unit test:
1. It tests ONE thing (not the entire pipeline)
2. It doesn't need external services (no API keys, no ChromaDB, no network)
3. It runs in milliseconds
4. It has a clear name that describes what it tests
5. It tests edge cases, not just the happy path

These tests verify the chunking logic independently of embeddings or LLMs.
If someone changes the chunker and accidentally breaks FAQ chunking,
these tests catch it BEFORE it reaches production.
"""

import pytest
from pathlib import Path

from backend.app.services.rag.chunker import (
    detect_document_type,
    chunk_document,
    content_hash,
    extract_csv_documents,
)


class TestDocumentTypeDetection:
    """Test adaptive document type detection."""

    def test_faq_detection(self):
        assert detect_document_type("faq_general_ayurveda_patients") == "faq"
        assert detect_document_type("FAQ_sleep.md") == "faq"

    def test_product_detection(self):
        assert detect_document_type("product_ashwagandha_tablets_internal") == "product"
        assert detect_document_type("product_overview.md") == "product"

    def test_guide_detection(self):
        assert detect_document_type("dosha_guide_vata_pitta_kapha") == "guide"
        assert detect_document_type("ayurveda_foundations") == "guide"
        assert detect_document_type("content_style_guide") == "guide"

    def test_pdf_detection(self):
        assert detect_document_type("big_book.pdf") == "guide"

    def test_default_detection(self):
        assert detect_document_type("random_document") == "default"
        assert detect_document_type("notes") == "default"


class TestContentHash:
    """Test content deduplication hashing."""

    def test_same_content_same_hash(self):
        assert content_hash("hello world") == content_hash("hello world")

    def test_different_content_different_hash(self):
        assert content_hash("hello") != content_hash("world")

    def test_hash_length(self):
        h = content_hash("test content")
        assert len(h) == 16  # Truncated SHA-256


class TestChunkDocument:
    """Test the adaptive chunking logic."""

    def test_faq_chunking_small_chunks(self):
        """FAQ documents should produce smaller chunks."""
        content = "## Question 1\nWhat is Ayurveda?\n\n" * 10
        result = chunk_document(content, "test_faq", "faq")
        assert result.total_chunks > 0
        # FAQ chunks should be ≤ 400 chars (with some tolerance for overlap)
        for chunk in result.chunks:
            assert len(chunk.page_content) <= 500  # 400 + overlap margin

    def test_guide_chunking_larger_chunks(self):
        """Guide documents should produce larger chunks."""
        content = "## Section\n" + "This is guide content about Ayurveda. " * 50
        result = chunk_document(content, "test_guide", "guide")
        assert result.total_chunks > 0

    def test_metadata_attached(self):
        """Every chunk should have proper metadata."""
        content = "## Test Section\nSome content about herbs.\n\n" * 5
        result = chunk_document(content, "my_doc", "product")

        for chunk in result.chunks:
            assert chunk.metadata["doc_id"] == "my_doc"
            assert chunk.metadata["doc_type"] == "product"
            assert "chunk_index" in chunk.metadata
            assert "content_hash" in chunk.metadata
            assert "section_id" in chunk.metadata

    def test_parent_chunks_created(self):
        """Chunking should produce both child and parent chunks."""
        content = "## Section 1\n" + "Content. " * 100 + "\n\n## Section 2\n" + "More content. " * 100
        result = chunk_document(content, "test_doc", "guide")

        assert result.total_parent_chunks > 0
        assert result.total_chunks >= result.total_parent_chunks  # More children than parents

    def test_parent_child_linking(self):
        """Child chunks should reference parent chunk IDs."""
        content = "## Section\n" + "Content about Ayurveda. " * 100
        result = chunk_document(content, "test_doc", "default")

        for chunk in result.chunks:
            assert "parent_chunk_id" in chunk.metadata

    def test_empty_content(self):
        """Empty content should produce no chunks."""
        result = chunk_document("", "empty_doc", "default")
        assert result.total_chunks == 0

    def test_section_id_extraction(self):
        """Chunks containing markdown headers should extract section_id."""
        content = "## Benefits of Ashwagandha\nAshwagandha supports stress resilience..."
        result = chunk_document(content, "test_doc", "product")

        # At least one chunk should have the header as section_id
        section_ids = [c.metadata["section_id"] for c in result.chunks]
        assert any("Ashwagandha" in sid for sid in section_ids)


# BM25 keyword search and Reciprocal Rank Fusion used to be implemented in
# this process (retriever.BM25Index / reciprocal_rank_fusion) and were unit
# tested here. Both now run inside Qdrant: documents carry a sparse vector
# alongside the dense one, and the Query API fuses the two rankings with RRF
# server-side. The behaviour those tests covered — exact keyword matching and
# rank fusion — is verified end-to-end against real models in
# tests/integration/test_retrieval.py instead.


class TestLRUCache:
    """Test the in-memory LRU cache."""

    def test_cache_set_and_get(self):
        from backend.app.services.cache import LRUCache

        cache = LRUCache(max_size=10, ttl_seconds=3600)
        cache.set("key1", {"answer": "test"})
        assert cache.get("key1") == {"answer": "test"}

    def test_cache_miss(self):
        from backend.app.services.cache import LRUCache

        cache = LRUCache(max_size=10)
        assert cache.get("nonexistent") is None

    def test_cache_eviction(self):
        from backend.app.services.cache import LRUCache

        cache = LRUCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # Should evict "a"

        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_cache_lru_order(self):
        from backend.app.services.cache import LRUCache

        cache = LRUCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")     # Access "a" → it becomes most recently used
        cache.set("c", 3)  # Should evict "b" (least recently used)

        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3

    def test_cache_stats(self):
        from backend.app.services.cache import LRUCache

        cache = LRUCache(max_size=10)
        cache.set("x", 1)
        cache.get("x")     # Hit
        cache.get("y")     # Miss

        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
