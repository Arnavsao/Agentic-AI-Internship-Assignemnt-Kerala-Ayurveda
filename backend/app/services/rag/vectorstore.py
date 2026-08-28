"""
Qdrant Vector Store Adapter
=============================

Replaces the embedded ChromaDB store. Three things drove the switch:

1. SERVER-SIDE HYBRID SEARCH.
   Chroma has no sparse-vector support, so keyword search was a pure-Python
   BM25 index rebuilt at every warm start by scanning the whole collection
   through a private API (`vectorstore._collection.get()`). Qdrant stores a
   sparse vector alongside the dense one and fuses both with RRF inside the
   engine, so startup no longer touches every point.

2. IDEMPOTENT UPSERTS.
   Point IDs are derived from content (uuid5 of doc_id + content_hash), so
   re-ingesting unchanged text writes to the same ID instead of appending a
   duplicate. The Chroma index had all 8 catalog products stored twice.

3. AN EXPLICIT COMPATIBILITY CONTRACT.
   `assert_compatible()` refuses to serve a collection whose stored embedding
   model or dimension doesn't match the running config. The previous code
   reused any non-empty collection, which meant a 768-d model could silently
   query a 384-d index and return nonsense.

The adapter deliberately wraps `qdrant-client` directly rather than
`langchain-qdrant`: we need named vectors, deterministic IDs, payload indexes,
scroll, filtered deletes, and the metadata assertion — all of which the
LangChain wrapper hides.
"""

import logging
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from qdrant_client import QdrantClient, models

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)

# Namespace for deterministic point IDs. Fixed constant — changing it would
# orphan every existing point.
_POINT_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")

# Reserved ID of the single point holding index metadata in the meta collection.
_META_POINT_ID = "00000000-0000-0000-0000-000000000001"

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"


def point_id_for(doc_id: str, content_hash: str) -> str:
    """
    Deterministic point ID.

    Same document + same text always produces the same ID, which turns
    re-ingestion into an idempotent upsert rather than an append.
    """
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{doc_id}:{content_hash}"))


class IndexCompatibilityError(RuntimeError):
    """Raised when the live collection doesn't match the running config."""


class QdrantStore:
    """Thin adapter over a Qdrant collection holding dense + sparse vectors."""

    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        dense_dim: int,
        embedding_model: str,
        sparse_model: str,
    ):
        self.client = client
        self.collection = collection
        self.meta_collection = f"{collection}__meta"
        self.dense_dim = dense_dim
        self.embedding_model = embedding_model
        self.sparse_model = sparse_model

    # ── Collection lifecycle ────────────────────────────────────

    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def ensure_collection(self) -> bool:
        """
        Create the collection and its payload indexes if absent.

        Returns True if a new collection was created.
        """
        if self.collection_exists():
            return False

        logger.info(
            f"Creating Qdrant collection '{self.collection}' "
            f"(dense={self.dense_dim}d, sparse={self.sparse_model})",
            extra={"component": "vectorstore"},
        )

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=self.dense_dim,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                # IDF modifier makes Qdrant compute inverse document frequency
                # server-side, which is what turns raw term weights into real
                # BM25 scoring.
                SPARSE_VECTOR: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )

        # Payload indexes. doc_id is load-bearing — incremental re-ingest
        # deletes a document's old points by filtering on it.
        for field in ("doc_id", "doc_type", "product_id", "content_hash"):
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

        self._ensure_meta_collection()
        return True

    def _ensure_meta_collection(self) -> None:
        """Tiny sidecar collection holding one point of index metadata."""
        if not self.client.collection_exists(self.meta_collection):
            self.client.create_collection(
                collection_name=self.meta_collection,
                vectors_config=models.VectorParams(
                    size=1, distance=models.Distance.COSINE
                ),
            )

    # ── Index metadata ──────────────────────────────────────────

    def read_meta(self) -> Dict[str, Any]:
        """Read the index metadata payload. Empty dict if never written."""
        self._ensure_meta_collection()
        try:
            points = self.client.retrieve(
                collection_name=self.meta_collection,
                ids=[_META_POINT_ID],
                with_payload=True,
            )
        except Exception:
            return {}
        if not points:
            return {}
        return points[0].payload or {}

    def write_meta(self, meta: Dict[str, Any]) -> None:
        """Overwrite the index metadata payload."""
        self._ensure_meta_collection()
        self.client.upsert(
            collection_name=self.meta_collection,
            points=[
                models.PointStruct(
                    id=_META_POINT_ID,
                    vector=[0.0],
                    payload=meta,
                )
            ],
        )

    def assert_compatible(self) -> None:
        """
        Refuse to serve an index built with a different model or dimension.

        This is the guard the Chroma implementation lacked: it reused any
        non-empty collection, so switching embedding models silently produced
        garbage retrieval instead of an error.
        """
        info = self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        dense_cfg = vectors.get(DENSE_VECTOR) if isinstance(vectors, dict) else None

        if dense_cfg is None:
            raise IndexCompatibilityError(
                f"Collection '{self.collection}' has no '{DENSE_VECTOR}' named vector. "
                f"It was probably created by an older build. "
                f"Re-create it with: python -m scripts.ingest --rebuild"
            )

        if dense_cfg.size != self.dense_dim:
            raise IndexCompatibilityError(
                f"Vector dimension mismatch: collection '{self.collection}' stores "
                f"{dense_cfg.size}-d vectors but the configured embedding model "
                f"'{self.embedding_model}' produces {self.dense_dim}-d vectors. "
                f"Re-index with: python -m scripts.ingest --rebuild"
            )

        stored_model = self.read_meta().get("embedding_model")
        if stored_model and stored_model != self.embedding_model:
            raise IndexCompatibilityError(
                f"Embedding model mismatch: collection '{self.collection}' was built "
                f"with '{stored_model}' but the running config says "
                f"'{self.embedding_model}'. Dimensions happen to match, but the vector "
                f"spaces are unrelated and retrieval would be meaningless. "
                f"Re-index with: python -m scripts.ingest --rebuild"
            )

    # ── Writes ──────────────────────────────────────────────────

    def upsert_points(self, points: Sequence[models.PointStruct]) -> int:
        """Upsert points in batches. Returns the number written."""
        if not points:
            return 0

        batch_size = 128
        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection,
                points=list(points[start:start + batch_size]),
                wait=True,
            )
        return len(points)

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Remove every point belonging to a document."""
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def delete_points(self, point_ids: Sequence[str]) -> None:
        """Remove specific points by ID."""
        if not point_ids:
            return
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.PointIdsList(points=list(point_ids)),
            wait=True,
        )

    def drop_collection(self) -> None:
        """Delete the collection and its metadata sidecar."""
        for name in (self.collection, self.meta_collection):
            if self.client.collection_exists(name):
                self.client.delete_collection(name)

    # ── Reads ───────────────────────────────────────────────────

    def count(self) -> int:
        if not self.collection_exists():
            return 0
        return self.client.count(self.collection, exact=True).count

    def point_ids_for_doc(self, doc_id: str) -> List[str]:
        """All point IDs belonging to one document (used by incremental ingest)."""
        ids: List[str] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id),
                        )
                    ]
                ),
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            ids.extend(str(p.id) for p in points)
            if offset is None:
                break
        return ids

    def hybrid_query(
        self,
        dense_vector: List[float],
        sparse_vector: Optional[Tuple[List[int], List[float]]],
        limit: int,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Dense + sparse retrieval fused server-side with Reciprocal Rank Fusion.

        Each branch retrieves `limit` candidates independently; Qdrant merges
        them with RRF and returns the top `limit`. This replaces the previous
        client-side semantic search + Python BM25 + manual RRF.
        """
        prefetch = [
            models.Prefetch(
                query=dense_vector,
                using=DENSE_VECTOR,
                limit=limit,
            )
        ]

        if sparse_vector is not None:
            indices, values = sparse_vector
            if indices:
                prefetch.append(
                    models.Prefetch(
                        query=models.SparseVector(indices=indices, values=values),
                        using=SPARSE_VECTOR,
                        limit=limit,
                    )
                )

        response = self.client.query_points(
            collection_name=self.collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

        return [(p.payload or {}, float(p.score)) for p in response.points]


# ── Client factory ──────────────────────────────────────────────

_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """
    Qdrant client singleton. `QDRANT_URL` selects one of three modes:

      http://host:6333   server mode — the deployment target, and the only
                         mode that supports payload indexes and concurrent
                         access from multiple workers
      :memory:           in-process, discarded on exit — used by the tests
      ./some/path        in-process, file-backed — lets you run the app
                         without Docker, at the cost of a single-process lock

    The local modes implement the same Query API (named vectors, sparse
    vectors, RRF fusion), so retrieval behaves identically; they simply don't
    scale past one process.
    """
    global _client
    if _client is not None:
        return _client

    settings = get_settings()
    url = settings.qdrant_url

    if url == ":memory:":
        _client = QdrantClient(":memory:")
        logger.info("Qdrant running in-memory", extra={"component": "vectorstore"})
    elif url.startswith(("http://", "https://")):
        _client = QdrantClient(url=url, timeout=30)
        logger.info(f"Qdrant client connected: {url}", extra={"component": "vectorstore"})
    else:
        # Filesystem path — local persistent mode.
        _client = QdrantClient(path=url)
        logger.warning(
            f"Qdrant running in local file mode at '{url}'. This locks the "
            f"directory to a single process; use a Qdrant server for anything "
            f"beyond local development.",
            extra={"component": "vectorstore"},
        )

    return _client


def reset_qdrant_client() -> None:
    """Drop the cached client. Used by tests."""
    global _client
    _client = None
