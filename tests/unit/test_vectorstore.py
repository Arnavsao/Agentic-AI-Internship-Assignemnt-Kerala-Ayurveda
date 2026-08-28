"""
Tests for the Qdrant adapter.

The compatibility assertions get the most attention here: the failure they
prevent (a 768-d model querying a 384-d index, returning plausible-looking
nonsense) is silent, so a regression would not announce itself.
"""

import pytest
from qdrant_client import models

from backend.app.services.rag.vectorstore import (
    IndexCompatibilityError,
    QdrantStore,
    point_id_for,
)


def make_point(store, doc_id, content_hash, content, **payload):
    return models.PointStruct(
        id=point_id_for(doc_id, content_hash),
        vector={
            "dense": [0.1] * store.dense_dim,
            "sparse": models.SparseVector(indices=[1, 2], values=[0.5, 0.5]),
        },
        payload={
            "content": content,
            "doc_id": doc_id,
            "content_hash": content_hash,
            **payload,
        },
    )


class TestPointIds:
    def test_deterministic(self):
        assert point_id_for("doc", "abc123") == point_id_for("doc", "abc123")

    def test_differs_by_content(self):
        assert point_id_for("doc", "abc123") != point_id_for("doc", "def456")

    def test_differs_by_document(self):
        assert point_id_for("doc_a", "abc123") != point_id_for("doc_b", "abc123")

    def test_is_valid_uuid(self):
        import uuid
        uuid.UUID(point_id_for("doc", "abc123"))  # raises if malformed


class TestCollectionLifecycle:
    def test_ensure_creates_once(self, store):
        assert store.ensure_collection() is True
        assert store.collection_exists()
        assert store.ensure_collection() is False

    def test_named_vectors_configured(self, store):
        store.ensure_collection()
        info = store.client.get_collection(store.collection)
        assert "dense" in info.config.params.vectors
        assert info.config.params.vectors["dense"].size == store.dense_dim
        assert "sparse" in info.config.params.sparse_vectors

    def test_sparse_uses_idf_modifier(self, store):
        """IDF must be server-side or sparse scoring isn't real BM25."""
        store.ensure_collection()
        info = store.client.get_collection(store.collection)
        assert info.config.params.sparse_vectors["sparse"].modifier == models.Modifier.IDF

    def test_drop_removes_both_collections(self, store):
        store.ensure_collection()
        store.write_meta({"embedding_model": "test/fake-embedder"})
        store.drop_collection()
        assert not store.collection_exists()
        assert not store.client.collection_exists(store.meta_collection)


class TestMetadata:
    def test_roundtrip(self, store):
        store.ensure_collection()
        store.write_meta({"embedding_model": "m", "index_version": 3})
        meta = store.read_meta()
        assert meta["embedding_model"] == "m"
        assert meta["index_version"] == 3

    def test_empty_before_write(self, store):
        store.ensure_collection()
        assert store.read_meta() == {}


class TestCompatibilityAssertion:
    def test_passes_when_matching(self, store):
        store.ensure_collection()
        store.write_meta({"embedding_model": "test/fake-embedder"})
        store.assert_compatible()  # must not raise

    def test_rejects_dimension_mismatch(self, qdrant_client, store):
        """The exact hazard the Chroma implementation shipped with."""
        store.ensure_collection()
        store.write_meta({"embedding_model": "test/fake-embedder"})

        wider = QdrantStore(
            client=qdrant_client,
            collection=store.collection,
            dense_dim=768,                       # collection holds 8-d
            embedding_model="test/fake-embedder",
            sparse_model="test/fake-sparse",
        )
        with pytest.raises(IndexCompatibilityError, match="dimension mismatch"):
            wider.assert_compatible()

    def test_rejects_model_mismatch_at_same_dimension(self, qdrant_client, store):
        """Same width, unrelated vector space — still meaningless."""
        store.ensure_collection()
        store.write_meta({"embedding_model": "test/fake-embedder"})

        other = QdrantStore(
            client=qdrant_client,
            collection=store.collection,
            dense_dim=8,
            embedding_model="some/other-model",
            sparse_model="test/fake-sparse",
        )
        with pytest.raises(IndexCompatibilityError, match="model mismatch"):
            other.assert_compatible()

    def test_error_names_the_remedy(self, qdrant_client, store):
        store.ensure_collection()
        store.write_meta({"embedding_model": "test/fake-embedder"})
        other = QdrantStore(
            client=qdrant_client,
            collection=store.collection,
            dense_dim=8,
            embedding_model="some/other-model",
            sparse_model="test/fake-sparse",
        )
        with pytest.raises(IndexCompatibilityError, match="scripts.ingest --rebuild"):
            other.assert_compatible()


class TestWritesAndDeletes:
    def test_upsert_and_count(self, store):
        store.ensure_collection()
        store.upsert_points([
            make_point(store, "doc_a", "h1", "first"),
            make_point(store, "doc_a", "h2", "second"),
        ])
        assert store.count() == 2

    def test_reupsert_is_idempotent(self, store):
        """
        Re-ingesting identical content must not duplicate points.
        The Chroma index had all 8 catalog products stored twice because
        nothing keyed writes on content.
        """
        store.ensure_collection()
        points = [make_point(store, "doc_a", "h1", "first")]
        store.upsert_points(points)
        store.upsert_points(points)
        assert store.count() == 1

    def test_delete_by_doc_id(self, store):
        store.ensure_collection()
        store.upsert_points([
            make_point(store, "doc_a", "h1", "a1"),
            make_point(store, "doc_b", "h2", "b1"),
        ])
        store.delete_by_doc_id("doc_a")
        assert store.count() == 1

    def test_point_ids_for_doc(self, store):
        store.ensure_collection()
        store.upsert_points([
            make_point(store, "doc_a", "h1", "a1"),
            make_point(store, "doc_a", "h2", "a2"),
            make_point(store, "doc_b", "h3", "b1"),
        ])
        assert len(store.point_ids_for_doc("doc_a")) == 2
        assert len(store.point_ids_for_doc("doc_b")) == 1

    def test_delete_specific_points(self, store):
        store.ensure_collection()
        store.upsert_points([
            make_point(store, "doc_a", "h1", "a1"),
            make_point(store, "doc_a", "h2", "a2"),
        ])
        store.delete_points([point_id_for("doc_a", "h1")])
        assert store.count() == 1


class TestHybridQuery:
    def test_returns_payloads_and_scores(self, store):
        store.ensure_collection()
        store.upsert_points([
            make_point(store, "doc_a", "h1", "ashwagandha supports stress"),
            make_point(store, "doc_b", "h2", "triphala supports digestion"),
        ])
        results = store.hybrid_query(
            dense_vector=[0.1] * 8,
            sparse_vector=([1, 2], [0.5, 0.5]),
            limit=5,
        )
        assert len(results) == 2
        payload, score = results[0]
        assert "content" in payload
        assert isinstance(score, float)

    def test_dense_only_when_sparse_empty(self, store):
        """A query with no indexable terms must still return dense hits."""
        store.ensure_collection()
        store.upsert_points([make_point(store, "doc_a", "h1", "content")])
        results = store.hybrid_query([0.1] * 8, ([], []), limit=5)
        assert len(results) == 1

    def test_respects_limit(self, store):
        store.ensure_collection()
        store.upsert_points([
            make_point(store, "doc", f"h{i}", f"chunk {i}") for i in range(10)
        ])
        assert len(store.hybrid_query([0.1] * 8, ([1], [0.5]), limit=3)) == 3
