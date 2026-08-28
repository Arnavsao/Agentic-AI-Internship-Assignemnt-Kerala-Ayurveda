"""
End-to-end retrieval against real embedding models.

Unlike the ingestion unit tests, these load the actual bge encoder, the
FastEmbed BM25 model, and the cross-encoder — the point is to prove the whole
chain returns the right document, not just that the plumbing connects. First
run downloads model weights.

Run with:  pytest tests/integration/ -m integration
Skip with: pytest -m "not integration"
"""

import pytest

from backend.app.services.ingestion.service import ingest
from backend.app.services.rag.embeddings import get_dense_embedder, get_sparse_embedder
from backend.app.services.rag.retriever import HybridRetriever
from backend.app.services.rag.vectorstore import QdrantStore

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def indexed(tmp_path_factory):
    """Ingest a small corpus once and reuse it across the module."""
    from qdrant_client import QdrantClient

    d = tmp_path_factory.mktemp("content")
    (d / "faq_general.md").write_text(
        "# FAQ\n\n"
        "## Can Ayurveda help with stress and sleep?\n"
        "Ayurvedic routines are traditionally used to support restful sleep "
        "and healthy stress response. Consult a qualified practitioner.\n",
        encoding="utf-8",
    )
    (d / "product_ashwagandha_internal.md").write_text(
        "# Ashwagandha Stress Balance Tablets\n\n"
        "## Traditional Positioning\n"
        "Ashwagandha is traditionally used to support the body's ability to "
        "adapt to stress and promote calmness and emotional balance.\n\n"
        "## Contraindications and Safety\n"
        "Ashwagandha is not recommended during pregnancy and may interact "
        "with thyroid medication.\n",
        encoding="utf-8",
    )
    (d / "treatment_stress_program.md").write_text(
        "# Stress Support Program\n\n"
        "## Program Structure\n"
        "The program begins with a practitioner consultation, then combines "
        "Abhyanga warm oil massage and Shirodhara oil-flow therapy across "
        "several sessions. It is not a substitute for medical care.\n",
        encoding="utf-8",
    )
    (d / "products_catalog.csv").write_text(
        "product_id,name,category,format,target_concerns,key_herbs,"
        "contains_animal_products,contraindications_short,internal_tags\n"
        "KA-P001,Ashwagandha Stress Balance Tablets,Supplement,Tablet,"
        "Stress;Sleep,Ashwagandha,No,Pregnancy,stress\n"
        "KA-P002,Triphala Digestive Capsules,Supplement,Capsule,"
        "Digestion,Amalaki;Bibhitaki;Haritaki,No,Pregnancy,digestion\n",
        encoding="utf-8",
    )

    client = QdrantClient(":memory:")
    dense = get_dense_embedder()
    store = QdrantStore(
        client=client,
        collection="integration_test",
        dense_dim=dense.dim,
        embedding_model=dense.model_name,
        sparse_model=get_sparse_embedder().model_name,
    )
    stats = ingest(store, d)
    yield store, stats
    client.close()


class TestIndexing:
    def test_corpus_indexed(self, indexed):
        store, stats = indexed
        assert stats.files_indexed == 4
        assert store.count() > 0

    def test_dimension_matches_model(self, indexed):
        store, _ = indexed
        info = store.client.get_collection(store.collection)
        assert info.config.params.vectors["dense"].size == get_dense_embedder().dim

    def test_compatibility_assertion_passes(self, indexed):
        store, _ = indexed
        store.assert_compatible()


class TestRetrievalQuality:
    def test_semantic_query_finds_right_document(self, indexed):
        store, _ = indexed
        chunks = HybridRetriever(store).retrieve("Is Ashwagandha safe during pregnancy?")
        assert chunks
        assert any("ashwagandha" in c.doc_id.lower() for c in chunks[:3])
        assert any("pregnan" in c.document.page_content.lower() for c in chunks[:3])

    def test_rare_term_found_via_keyword_branch(self, indexed):
        """
        'Shirodhara' is the kind of rare term dense retrieval alone tends to
        miss — this is the case hybrid search exists for, and the one behind
        the q005 benchmark failure.
        """
        store, _ = indexed
        chunks = HybridRetriever(store).retrieve("Shirodhara")
        assert chunks
        joined = " ".join(c.document.page_content for c in chunks[:3]).lower()
        assert "shirodhara" in joined

    def test_product_id_lookup(self, indexed):
        """Exact identifiers are a keyword-search strength, not a dense one."""
        store, _ = indexed
        chunks = HybridRetriever(store).retrieve("KA-P002")
        assert chunks
        joined = " ".join(c.document.page_content for c in chunks[:3])
        assert "KA-P002" in joined

    def test_program_query_surfaces_specific_treatments(self, indexed):
        """
        The q005 regression: 'How does the Stress Support Program work?'
        previously returned an answer with no mention of Abhyanga or
        Shirodhara, scoring 0.00 coverage.
        """
        store, _ = indexed
        chunks = HybridRetriever(store).retrieve("How does the Stress Support Program work?")
        assert chunks
        top = " ".join(c.context_content for c in chunks[:3]).lower()
        assert "abhyanga" in top or "shirodhara" in top

    def test_parent_context_is_available(self, indexed):
        store, _ = indexed
        chunks = HybridRetriever(store).retrieve("Ashwagandha benefits for stress")
        assert any(c.parent_content for c in chunks)
        for c in chunks:
            assert len(c.context_content) >= len(c.document.page_content)


class TestRetrievalMechanics:
    def test_respects_top_n(self, indexed):
        store, _ = indexed
        assert len(HybridRetriever(store).retrieve("stress", top_k=10, top_n=2)) == 2

    def test_reranking_can_reorder(self, indexed):
        store, _ = indexed
        retriever = HybridRetriever(store)
        query = "What helps with restful sleep?"
        with_rr = retriever.retrieve(query, use_reranking=True)
        without = retriever.retrieve(query, use_reranking=False)
        assert with_rr and without
        assert all(c.rerank_score != 0.0 for c in with_rr)

    def test_scores_are_ordered(self, indexed):
        store, _ = indexed
        chunks = HybridRetriever(store).retrieve("Ashwagandha")
        scores = [c.final_score for c in chunks]
        assert scores == sorted(scores, reverse=True)

    def test_provenance_populated(self, indexed):
        store, _ = indexed
        for c in HybridRetriever(store).retrieve("Ashwagandha"):
            assert c.doc_id
            assert c.section_id


class TestQueryInstruction:
    def test_bge_prefix_applied_to_queries_only(self):
        """
        bge is trained asymmetrically. The previous code claimed LangChain
        applied the query prefix automatically; it does not, so every query
        was embedded in the document distribution.
        """
        dense = get_dense_embedder()
        if not dense.uses_query_instruction:
            pytest.skip("configured model is not bge-family")

        text = "Ashwagandha for stress"
        assert dense.encode_query(text) != dense.encode_documents([text])[0]
