"""
Embedding Models — Dense (bge) and Sparse (BM25)
==================================================

DENSE: BAAI/bge-base-en-v1.5, 768-d, local, free.

The bge family is trained asymmetrically: queries must carry the instruction
prefix "Represent this sentence for searching relevant passages:" while
documents are embedded bare. The previous implementation used
`langchain_huggingface.HuggingFaceEmbeddings` with a comment claiming the
prefix was "handled automatically" — it is not. LangChain only applies it if
you pass `query_instruction` explicitly, so every query was being embedded in
the document distribution. Wrapping SentenceTransformer directly makes the
asymmetry explicit and visible.

SPARSE: Qdrant/bm25 via FastEmbed.

Produces term-weight vectors that Qdrant scores with real BM25 (the IDF half
is computed server-side via the collection's IDF modifier). This replaces the
hand-rolled Python BM25 index, and more importantly removes the reason that
index existed — it had to be rebuilt on every startup by scanning the whole
collection.

Both models are local and free: no API keys, no per-query cost, no rate limits.
"""

import contextlib
import logging
import os
import warnings
from typing import List, Optional, Tuple

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)

# bge asymmetric retrieval prefix, applied to queries only.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_dense: Optional["DenseEmbedder"] = None
_sparse: Optional["SparseEmbedder"] = None


def _quiet_hf_env() -> None:
    """Silence HuggingFace/PyTorch startup chatter."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "true")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)


class DenseEmbedder:
    """
    SentenceTransformer wrapper with correct query/document asymmetry.

    `encode_query` applies the instruction prefix for bge-family models;
    `encode_documents` never does. Both L2-normalize so cosine distance in
    Qdrant behaves as expected.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        _quiet_hf_env()
        from sentence_transformers import SentenceTransformer

        logger.info(
            f"Loading dense embedding model: {model_name}",
            extra={"component": "embeddings", "model": model_name},
        )

        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                self.model = SentenceTransformer(model_name, device=device)

        self.model_name = model_name
        self.dim = int(self.model.get_sentence_embedding_dimension())
        # Only bge-family models were trained with the retrieval instruction.
        self.uses_query_instruction = "bge" in model_name.lower()

        logger.info(
            f"Dense embedder ready: {model_name} ({self.dim}d, "
            f"query_instruction={self.uses_query_instruction})",
            extra={"component": "embeddings", "model": model_name, "dim": self.dim},
        )

    def encode_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return [v.tolist() for v in vectors]

    def encode_query(self, text: str) -> List[float]:
        if self.uses_query_instruction:
            text = BGE_QUERY_INSTRUCTION + text
        vector = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()


class SparseEmbedder:
    """
    FastEmbed BM25 term-weight vectors.

    Emits (indices, values) pairs. Qdrant applies IDF server-side, so what
    ships here is term frequency information only.
    """

    def __init__(self, model_name: str):
        _quiet_hf_env()
        from fastembed import SparseTextEmbedding

        logger.info(
            f"Loading sparse embedding model: {model_name}",
            extra={"component": "embeddings", "model": model_name},
        )

        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                self.model = SparseTextEmbedding(model_name=model_name)

        self.model_name = model_name
        logger.info("Sparse embedder ready", extra={"component": "embeddings"})

    def encode_documents(self, texts: List[str]) -> List[Tuple[List[int], List[float]]]:
        return [
            (e.indices.tolist(), e.values.tolist())
            for e in self.model.embed(texts)
        ]

    def encode_query(self, text: str) -> Tuple[List[int], List[float]]:
        embeddings = list(self.model.query_embed(text))
        if not embeddings:
            return ([], [])
        e = embeddings[0]
        return (e.indices.tolist(), e.values.tolist())


def get_dense_embedder() -> DenseEmbedder:
    """
    Dense embedder singleton.

    The model is ~220 MB resident and takes a few seconds to load. Loading it
    per request would exhaust memory under any real concurrency.
    """
    global _dense
    if _dense is None:
        settings = get_settings()
        _dense = DenseEmbedder(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
        )
    return _dense


def get_sparse_embedder() -> SparseEmbedder:
    """Sparse embedder singleton."""
    global _sparse
    if _sparse is None:
        settings = get_settings()
        _sparse = SparseEmbedder(model_name=settings.sparse_model)
    return _sparse


def reset_embedders() -> None:
    """Drop cached embedders. Used by tests that switch models."""
    global _dense, _sparse
    _dense = None
    _sparse = None
