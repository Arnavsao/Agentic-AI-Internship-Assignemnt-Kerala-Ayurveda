"""
Tests for incremental ingestion.

The behaviour worth protecting is that a second ingest of an unchanged corpus
does no work. The previous implementation re-embedded everything on every
rebuild and deduplicated nothing, which is how the live index ended up holding
each catalog product twice.

Embedders are stubbed: these tests are about the diffing logic, and loading
real models would make them slow without testing anything extra.
"""

import pytest

from backend.app.services.ingestion import service as ingestion
from backend.app.services.rag.vectorstore import QdrantStore


class FakeDense:
    """Deterministic 8-d vectors derived from text length."""
    model_name = "test/fake-embedder"
    dim = 8

    def __init__(self):
        self.calls = 0

    def encode_documents(self, texts):
        self.calls += len(texts)
        return [[(len(t) % 10) / 10.0] * self.dim for t in texts]

    def encode_query(self, text):
        return [(len(text) % 10) / 10.0] * self.dim


class FakeSparse:
    model_name = "test/fake-sparse"

    def encode_documents(self, texts):
        return [([1, 2, 3], [0.5, 0.3, 0.2]) for _ in texts]

    def encode_query(self, text):
        return ([1, 2, 3], [0.5, 0.3, 0.2])


@pytest.fixture
def fake_embedders(monkeypatch):
    dense, sparse = FakeDense(), FakeSparse()
    monkeypatch.setattr(ingestion, "get_dense_embedder", lambda: dense)
    monkeypatch.setattr(ingestion, "get_sparse_embedder", lambda: sparse)
    return dense, sparse


@pytest.fixture
def ingest_store(qdrant_client):
    return QdrantStore(
        client=qdrant_client,
        collection="ingest_test",
        dense_dim=8,
        embedding_model="test/fake-embedder",
        sparse_model="test/fake-sparse",
    )


class TestFirstIngest:
    def test_indexes_every_file(self, ingest_store, content_dir, fake_embedders):
        stats = ingestion.ingest(ingest_store, content_dir)
        assert stats.files_scanned == 3          # 2 markdown + 1 csv
        assert stats.files_indexed == 3
        assert stats.files_skipped == 0
        assert stats.points_upserted > 0
        assert ingest_store.count() == stats.points_upserted

    def test_records_manifest_and_version(self, ingest_store, content_dir, fake_embedders):
        stats = ingestion.ingest(ingest_store, content_dir)
        meta = ingest_store.read_meta()
        assert meta["embedding_model"] == "test/fake-embedder"
        assert meta["embedding_dim"] == 8
        assert stats.index_version == 1
        assert set(meta["manifest"]) == {
            "faq_general.md",
            "product_ashwagandha_internal.md",
            "products_catalog.csv",
        }

    def test_payload_carries_provenance(self, ingest_store, content_dir, fake_embedders):
        ingestion.ingest(ingest_store, content_dir)
        points, _ = ingest_store.client.scroll(
            collection_name=ingest_store.collection, limit=100, with_payload=True
        )
        for p in points:
            assert p.payload["content"]
            assert p.payload["doc_id"]
            assert p.payload["content_hash"]
            assert p.payload["source_file"]

    def test_parent_content_inlined(self, ingest_store, content_dir, fake_embedders):
        """Parent text rides along in the payload instead of a separate map."""
        ingestion.ingest(ingest_store, content_dir)
        points, _ = ingest_store.client.scroll(
            collection_name=ingest_store.collection, limit=200, with_payload=True
        )
        md_points = [p for p in points if p.payload.get("file_type") == "md"]
        assert any(p.payload.get("parent_content") for p in md_points)

    def test_csv_rows_indexed_once_each(self, ingest_store, content_dir, fake_embedders):
        """Regression: the Chroma index stored every catalog product twice."""
        ingestion.ingest(ingest_store, content_dir)
        points, _ = ingest_store.client.scroll(
            collection_name=ingest_store.collection, limit=200, with_payload=True
        )
        catalog = [p for p in points if p.payload.get("file_type") == "csv"]
        assert len(catalog) == 2
        assert len({p.payload["product_id"] for p in catalog}) == 2


class TestIncrementalIngest:
    def test_second_run_is_a_no_op(self, ingest_store, content_dir, fake_embedders):
        dense, _ = fake_embedders
        ingestion.ingest(ingest_store, content_dir)
        first_count = ingest_store.count()
        embed_calls = dense.calls

        stats = ingestion.ingest(ingest_store, content_dir)

        assert stats.files_skipped == 3
        assert stats.files_indexed == 0
        assert stats.points_upserted == 0
        assert ingest_store.count() == first_count
        assert dense.calls == embed_calls, "unchanged files must not be re-embedded"

    def test_version_unchanged_when_nothing_changed(self, ingest_store, content_dir, fake_embedders):
        v1 = ingestion.ingest(ingest_store, content_dir).index_version
        v2 = ingestion.ingest(ingest_store, content_dir).index_version
        assert v1 == v2

    def test_only_the_edited_file_is_reindexed(self, ingest_store, content_dir, fake_embedders):
        ingestion.ingest(ingest_store, content_dir)

        (content_dir / "faq_general.md").write_text(
            "# FAQ\n\n## New question\nA fresh answer about Shirodhara therapy.\n",
            encoding="utf-8",
        )
        stats = ingestion.ingest(ingest_store, content_dir)

        assert stats.files_indexed == 1
        assert stats.files_skipped == 2
        assert stats.points_deleted > 0, "superseded chunks should be removed"
        assert stats.index_version == 2

    def test_new_file_is_picked_up(self, ingest_store, content_dir, fake_embedders):
        before = ingest_store.count() if ingest_store.collection_exists() else 0
        ingestion.ingest(ingest_store, content_dir)
        mid = ingest_store.count()

        (content_dir / "guide_dosha.md").write_text(
            "# Dosha Guide\n\n## Vata\nVata governs movement and is light and dry.\n",
            encoding="utf-8",
        )
        stats = ingestion.ingest(ingest_store, content_dir)

        assert stats.files_indexed == 1
        assert stats.files_scanned == 4
        assert ingest_store.count() > mid >= before

    def test_deleted_file_is_purged(self, ingest_store, content_dir, fake_embedders):
        ingestion.ingest(ingest_store, content_dir)
        before = ingest_store.count()

        (content_dir / "faq_general.md").unlink()
        stats = ingestion.ingest(ingest_store, content_dir)

        assert stats.files_removed == 1
        assert ingest_store.count() < before
        remaining, _ = ingest_store.client.scroll(
            collection_name=ingest_store.collection, limit=200, with_payload=True
        )
        assert all(p.payload["doc_id"] != "faq_general" for p in remaining)
        assert "faq_general.md" not in ingest_store.read_meta()["manifest"]

    def test_force_reembeds_everything(self, ingest_store, content_dir, fake_embedders):
        dense, _ = fake_embedders
        ingestion.ingest(ingest_store, content_dir)
        calls_after_first = dense.calls

        stats = ingestion.ingest(ingest_store, content_dir, force=True)

        assert stats.files_indexed == 3
        assert stats.files_skipped == 0
        assert dense.calls > calls_after_first

    def test_force_does_not_duplicate_points(self, ingest_store, content_dir, fake_embedders):
        """Deterministic IDs mean even a forced re-ingest upserts in place."""
        ingestion.ingest(ingest_store, content_dir)
        count = ingest_store.count()
        ingestion.ingest(ingest_store, content_dir, force=True)
        assert ingest_store.count() == count


class TestErrors:
    def test_missing_content_dir(self, ingest_store, tmp_path, fake_embedders):
        with pytest.raises(FileNotFoundError):
            ingestion.ingest(ingest_store, tmp_path / "nope")
