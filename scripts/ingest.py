"""
Ingestion CLI
===============

Indexing used to happen implicitly at app startup, which meant the only way to
rebuild was to delete the store and restart the server. Ingestion is now a
standalone step you can run and verify on its own:

    python -m scripts.ingest             # incremental — skips unchanged files
    python -m scripts.ingest --rebuild   # drop the collection and start over
    python -m scripts.ingest --status    # report what's indexed, change nothing

Run --rebuild after changing the embedding model; the vector space changes and
old vectors become meaningless (the app refuses to start in that state rather
than serving nonsense).
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python -m scripts.ingest` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.core.config import get_settings
from backend.app.core.logging import setup_logging
from backend.app.services.ingestion.service import ingest
from backend.app.services.rag.embeddings import get_dense_embedder
from backend.app.services.rag.vectorstore import QdrantStore, get_qdrant_client

logger = logging.getLogger(__name__)


def build_store() -> QdrantStore:
    settings = get_settings()
    embedder = get_dense_embedder()
    return QdrantStore(
        client=get_qdrant_client(),
        collection=settings.qdrant_collection,
        dense_dim=embedder.dim,
        embedding_model=settings.embedding_model,
        sparse_model=settings.sparse_model,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Index the knowledge base into Qdrant")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop the collection and re-embed everything from scratch",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show what is currently indexed without modifying anything",
    )
    parser.add_argument(
        "--content-dir",
        type=Path,
        default=None,
        help="Override the content directory (defaults to CONTENT_DIR)",
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(debug=settings.debug)

    store = build_store()

    if args.status:
        if not store.collection_exists():
            print(f"Collection '{store.collection}' does not exist. Run: python -m scripts.ingest")
            return 1
        meta = store.read_meta()
        manifest = meta.get("manifest") or {}
        print(f"Collection:      {store.collection}")
        print(f"Points:          {store.count()}")
        print(f"Index version:   {meta.get('index_version', 0)}")
        print(f"Embedding model: {meta.get('embedding_model', '?')} ({meta.get('embedding_dim', '?')}d)")
        print(f"Sparse model:    {meta.get('sparse_model', '?')}")
        print(f"Indexed files:   {len(manifest)}")
        for name, record in sorted(manifest.items()):
            print(f"  - {name}: {record.get('chunk_count', 0)} chunks")
        return 0

    if args.rebuild:
        print(f"Dropping collection '{store.collection}'...")
        store.drop_collection()

    store.ensure_collection()
    if not args.rebuild:
        # Surfaces a model/dimension mismatch as a clear error rather than
        # letting the upsert scatter incompatible vectors into the collection.
        try:
            store.assert_compatible()
        except RuntimeError as e:
            print(f"\nERROR: {e}\n", file=sys.stderr)
            return 1

    stats = ingest(store=store, content_dir=args.content_dir, force=args.rebuild)

    print("\nIngestion complete")
    print(f"  files scanned:   {stats.files_scanned}")
    print(f"  files indexed:   {stats.files_indexed}")
    print(f"  files unchanged: {stats.files_skipped}")
    print(f"  files removed:   {stats.files_removed}")
    print(f"  points upserted: {stats.points_upserted}")
    print(f"  points deleted:  {stats.points_deleted}")
    print(f"  total points:    {store.count()}")
    print(f"  index version:   {stats.index_version}")
    print(f"  duration:        {stats.duration_seconds:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
