"""
Incremental Ingestion
=======================

The previous behaviour was delete-everything-and-rebuild on any change, which
meant re-embedding the entire corpus to add one file. `content_hash` was
already computed and stored on every chunk but nothing ever read it, so
nothing was deduplicated — the live Chroma index held all 8 catalog products
twice.

This service closes that gap:

  * Each source file's SHA-256 is recorded in a manifest stored alongside the
    index. Unchanged files are skipped without embedding anything.
  * Changed files have their points re-embedded and upserted, then any stale
    points (ones whose content_hash no longer appears) are deleted.
  * Files removed from disk have all their points deleted by doc_id filter.
  * Point IDs are uuid5(doc_id + content_hash), so an unchanged chunk always
    lands on the same ID — re-ingest is an idempotent upsert, never an append.

Parent chunks are inlined into each child point's payload as `parent_content`
rather than stored as their own points. Parents are only ever looked up by ID
for context expansion, never searched, so giving them vectors would just add
noise to retrieval. Inlining also removes the in-memory parent map that the
old pipeline rebuilt at every startup by scanning the whole collection.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from qdrant_client import models

from backend.app.core.config import get_settings
from backend.app.core.logging import LogTimer
from backend.app.services.rag.chunker import chunk_document, extract_csv_documents, extract_pdf_text, detect_document_type
from backend.app.services.rag.embeddings import get_dense_embedder, get_sparse_embedder
from backend.app.services.rag.vectorstore import QdrantStore, point_id_for

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    """What an ingest run actually did."""
    files_scanned: int = 0
    files_skipped: int = 0
    files_indexed: int = 0
    files_removed: int = 0
    points_upserted: int = 0
    points_deleted: int = 0
    index_version: int = 0
    duration_seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "files_scanned": self.files_scanned,
            "files_skipped": self.files_skipped,
            "files_indexed": self.files_indexed,
            "files_removed": self.files_removed,
            "points_upserted": self.points_upserted,
            "points_deleted": self.points_deleted,
            "index_version": self.index_version,
            "duration_seconds": round(self.duration_seconds, 2),
        }


def file_hash(path: Path) -> str:
    """SHA-256 of a file's bytes, used to detect content changes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _source_files(content_dir: Path) -> List[Path]:
    """Every file the chunker knows how to parse."""
    files = sorted(content_dir.glob("*.md"))
    files += sorted(content_dir.glob("*.pdf"))
    csv_file = content_dir / "products_catalog.csv"
    if csv_file.exists():
        files.append(csv_file)
    return files


def _chunk_file(path: Path):
    """
    Parse and chunk one source file.

    Returns (child_docs, parent_docs). The CSV catalog is a special case:
    each row is already small enough to be its own parent.
    """
    suffix = path.suffix.lower()

    if suffix == ".csv":
        docs = extract_csv_documents(path)
        return docs, docs

    if suffix == ".pdf":
        content = extract_pdf_text(path)
        if not content.strip():
            return [], []
        result = chunk_document(content, path.stem, "guide", file_type="pdf")
        return result.chunks, result.parent_chunks

    content = path.read_text(encoding="utf-8")
    doc_type = detect_document_type(path.stem)
    result = chunk_document(content, path.stem, doc_type, file_type="md")
    return result.chunks, result.parent_chunks


def _doc_ids_for_file(path: Path, child_docs) -> List[str]:
    """
    The doc_ids a file contributes.

    Usually one (the file stem), but the CSV catalog produces one doc_id per
    product row, so deletion has to cover all of them.
    """
    return sorted({d.metadata.get("doc_id", path.stem) for d in child_docs})


def ingest(
    store: QdrantStore,
    content_dir: Optional[Path] = None,
    force: bool = False,
) -> IngestionStats:
    """
    Bring the index in sync with the content directory.

    Args:
        store: target Qdrant collection adapter
        content_dir: source documents (defaults to configured content_dir)
        force: re-embed every file even if its hash is unchanged
    """
    settings = get_settings()
    content_dir = content_dir or settings.content_path
    started = time.perf_counter()

    if not content_dir.exists():
        raise FileNotFoundError(f"Content directory not found: {content_dir}")

    stats = IngestionStats()

    store.ensure_collection()
    meta = store.read_meta()
    manifest: Dict[str, dict] = dict(meta.get("manifest") or {})
    index_version = int(meta.get("index_version") or 0)

    dense = get_dense_embedder()
    sparse = get_sparse_embedder()

    files = _source_files(content_dir)
    stats.files_scanned = len(files)
    seen_names: set = set()
    changed = False

    with LogTimer(logger, "ingestion"):
        for path in files:
            name = path.name
            seen_names.add(name)
            current_hash = file_hash(path)
            recorded = manifest.get(name) or {}

            if not force and recorded.get("file_hash") == current_hash:
                stats.files_skipped += 1
                continue

            child_docs, parent_docs = _chunk_file(path)
            if not child_docs:
                logger.warning(
                    f"No chunks produced for {name}; skipping",
                    extra={"component": "ingestion", "doc_id": path.stem},
                )
                continue

            # parent_chunk_id → parent text, for inlining into child payloads
            parent_text: Dict[str, str] = {}
            for pdoc in parent_docs:
                pid = pdoc.metadata.get("parent_chunk_id")
                if pid:
                    parent_text[pid] = pdoc.page_content

            texts = [d.page_content for d in child_docs]
            dense_vectors = dense.encode_documents(texts)
            sparse_vectors = sparse.encode_documents(texts)

            points: List[models.PointStruct] = []
            new_ids: set = set()

            for doc, dvec, (sidx, sval) in zip(child_docs, dense_vectors, sparse_vectors):
                md = doc.metadata
                doc_id = md.get("doc_id", path.stem)
                c_hash = md.get("content_hash", "")
                pid = point_id_for(doc_id, c_hash)
                new_ids.add(pid)

                payload = {
                    "content": doc.page_content,
                    "doc_id": doc_id,
                    "section_id": md.get("section_id", ""),
                    "doc_type": md.get("doc_type", "default"),
                    "chunk_index": md.get("chunk_index", 0),
                    "file_type": md.get("file_type", "md"),
                    "content_hash": c_hash,
                    "source_file": name,
                }
                if md.get("product_id"):
                    payload["product_id"] = str(md["product_id"])

                parent_id = md.get("parent_chunk_id")
                if parent_id and parent_id in parent_text:
                    payload["parent_content"] = parent_text[parent_id]

                points.append(
                    models.PointStruct(
                        id=pid,
                        vector={
                            "dense": dvec,
                            "sparse": models.SparseVector(indices=sidx, values=sval),
                        },
                        payload=payload,
                    )
                )

            # Points that belonged to this file before but no longer exist.
            previous_ids = set(recorded.get("point_ids") or [])
            stale = previous_ids - new_ids

            stats.points_upserted += store.upsert_points(points)
            if stale:
                store.delete_points(sorted(stale))
                stats.points_deleted += len(stale)

            manifest[name] = {
                "file_hash": current_hash,
                "point_ids": sorted(new_ids),
                "doc_ids": _doc_ids_for_file(path, child_docs),
                "chunk_count": len(points),
            }
            stats.files_indexed += 1
            changed = True

            logger.info(
                f"Indexed {name}: {len(points)} chunks"
                + (f", {len(stale)} stale removed" if stale else ""),
                extra={"component": "ingestion", "doc_id": path.stem},
            )

        # Files that disappeared from disk.
        for name in sorted(set(manifest) - seen_names):
            record = manifest.pop(name)
            for doc_id in record.get("doc_ids") or []:
                store.delete_by_doc_id(doc_id)
            removed = len(record.get("point_ids") or [])
            stats.points_deleted += removed
            stats.files_removed += 1
            changed = True
            logger.info(
                f"Removed {name} from index ({removed} chunks)",
                extra={"component": "ingestion"},
            )

    if changed:
        index_version += 1

    store.write_meta({
        "embedding_model": dense.model_name,
        "embedding_dim": dense.dim,
        "sparse_model": sparse.model_name,
        "index_version": index_version,
        "manifest": manifest,
        "updated_at": time.time(),
    })

    stats.index_version = index_version
    stats.duration_seconds = time.perf_counter() - started

    logger.info(
        f"Ingestion complete: {stats.files_indexed} indexed, "
        f"{stats.files_skipped} unchanged, {stats.files_removed} removed, "
        f"{stats.points_upserted} points upserted, {stats.points_deleted} deleted "
        f"(index v{index_version})",
        extra={"component": "ingestion", **stats.as_dict()},
    )

    return stats
