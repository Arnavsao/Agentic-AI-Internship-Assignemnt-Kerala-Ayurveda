"""
Shared test fixtures.

Every test runs against an in-process Qdrant (`QdrantClient(":memory:")`),
which supports the same Query API as the server — named vectors, sparse
vectors, and RRF fusion included. No container needed to run the suite.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient

from backend.app.services.rag.vectorstore import QdrantStore


@pytest.fixture
def qdrant_client():
    """A fresh in-memory Qdrant per test."""
    client = QdrantClient(":memory:")
    yield client
    client.close()


@pytest.fixture
def store(qdrant_client):
    """A QdrantStore over a small fake 8-dim vector space."""
    return QdrantStore(
        client=qdrant_client,
        collection="test_collection",
        dense_dim=8,
        embedding_model="test/fake-embedder",
        sparse_model="test/fake-sparse",
    )


@pytest.fixture
def content_dir(tmp_path):
    """
    A miniature knowledge base mirroring the real corpus shape:
    one FAQ, one product doc, and a CSV catalog.
    """
    d = tmp_path / "content"
    d.mkdir()

    (d / "faq_general.md").write_text(
        "# Frequently Asked Questions\n\n"
        "## Can Ayurveda help with sleep?\n"
        "Ayurvedic routines are traditionally used to support restful sleep. "
        "Consult a qualified practitioner before starting anything new.\n\n"
        "## Is Ayurveda safe during pregnancy?\n"
        "Many herbs are not recommended during pregnancy. Always consult a "
        "qualified practitioner first.\n",
        encoding="utf-8",
    )

    (d / "product_ashwagandha_internal.md").write_text(
        "# Ashwagandha Stress Balance Tablets\n\n"
        "## Traditional Positioning\n"
        "Ashwagandha is traditionally used to support the body's ability to "
        "adapt to stress and to promote calmness.\n\n"
        "## Contraindications and Safety\n"
        "Not recommended during pregnancy. May interact with thyroid "
        "medication. Consult a qualified practitioner.\n",
        encoding="utf-8",
    )

    (d / "products_catalog.csv").write_text(
        "product_id,name,category,format,target_concerns,key_herbs,"
        "contains_animal_products,contraindications_short,internal_tags\n"
        "KA-P001,Ashwagandha Stress Balance Tablets,Supplement,Tablet,"
        "Stress;Sleep,Ashwagandha,No,Pregnancy,stress;sleep\n"
        "KA-P002,Triphala Digestive Capsules,Supplement,Capsule,"
        "Digestion,Amalaki;Bibhitaki;Haritaki,No,Pregnancy;Diarrhea,digestion\n",
        encoding="utf-8",
    )

    return d
