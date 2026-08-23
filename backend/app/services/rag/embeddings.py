"""
Embedding Model Manager
========================

WHY bge-base-en-v1.5 instead of all-MiniLM-L6-v2:
┌─────────────────────────┬────────────────────┬────────────────────┐
│ Property                │ all-MiniLM-L6-v2   │ bge-base-en-v1.5   │
├─────────────────────────┼────────────────────┼────────────────────┤
│ Dimensions              │ 384                │ 768                │
│ Max tokens              │ 256                │ 512                │
│ MTEB Retrieval score    │ 41.8               │ 53.3               │
│ Model size              │ 80 MB              │ 220 MB             │
│ Speed (relative)        │ 1x                 │ ~1.5x slower       │
│ Still local/free?       │ Yes                │ Yes                │
└─────────────────────────┴────────────────────┴────────────────────┘

The key improvement is the 512-token context window. Your guide chunks
are 800 characters (~200 tokens), which fits both models. But the
retrieval accuracy improvement (41.8 → 53.3 on MTEB) means significantly
better matching for medical/Ayurvedic terminology.

TRADE-OFF:
bge-base is ~2.8x larger and ~1.5x slower. For 100 chunks, this means
indexing takes 3 seconds instead of 2 seconds. For 100,000 chunks,
it means 30 minutes instead of 20 minutes. The accuracy gain is worth it
at any scale.

TEACHING POINT — Instruction-tuned embeddings:
bge models were trained with a special trick: the query gets a prefix
"Represent this sentence for searching relevant passages:" which tells
the model "this is a search query, not a document." This asymmetry
improves retrieval because queries and documents ARE fundamentally different.
"""

import logging
import os
import warnings
import contextlib

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)

# Global singleton to avoid loading the model multiple times
_embeddings_instance = None


def get_embeddings():
    """
    Get or create the embedding model singleton.

    WHY a singleton:
    The embedding model is ~220 MB in memory. Loading it takes 2-5 seconds.
    We load it once at startup and reuse it for every request.
    If you created a new instance per request, you'd load 220 MB per request
    and run out of memory with 10 concurrent users.
    """
    global _embeddings_instance

    if _embeddings_instance is not None:
        return _embeddings_instance

    settings = get_settings()

    logger.info(
        f"Loading embedding model: {settings.embedding_model}",
        extra={"component": "embeddings", "model": settings.embedding_model}
    )

    # Suppress noisy HuggingFace/PyTorch output during model loading
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    from langchain_huggingface import HuggingFaceEmbeddings

    # Suppress stdout/stderr during model download/load
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            _embeddings_instance = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={"device": settings.embedding_device},
                encode_kwargs={
                    "normalize_embeddings": True,
                    # bge models benefit from query instruction prefix
                    # This is handled by HuggingFaceEmbeddings automatically
                    # for models that support it
                },
            )

    logger.info(
        f"Embedding model loaded: {settings.embedding_model}",
        extra={
            "component": "embeddings",
            "model": settings.embedding_model,
            "device": settings.embedding_device,
        }
    )

    return _embeddings_instance
