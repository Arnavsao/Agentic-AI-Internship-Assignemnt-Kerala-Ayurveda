"""
Centralized Configuration — Pydantic Settings
===============================================

WHY THIS EXISTS:
The original codebase had os.getenv() scattered across 4 different files,
each with its own default handling. This creates several production problems:
  1. Typo in env var name → silent None → crash at runtime instead of startup
  2. No type validation → "true" vs True vs 1 are all different
  3. No single place to see what config the app needs
  4. Tests can't easily override config

HOW IT WORKS:
Pydantic Settings loads from environment variables automatically.
If a required variable is missing, the app won't even start — you get
a clear error message listing exactly what's missing.

TEACHING POINT:
In production, you want to "fail fast." If your database URL is wrong,
you want to know at startup (in the deploy logs), not 3 hours later when
the first user tries to query. This class enforces that.
"""

from pathlib import Path
from typing import Optional, List
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    Pydantic Settings automatically reads from:
    1. Environment variables (highest priority)
    2. .env file (if it exists)
    3. Default values defined here (lowest priority)
    """

    # ── Application ────────────────────────────────────────────
    app_name: str = "Kerala Ayurveda AI"
    app_version: str = "2.0.0"
    debug: bool = Field(default=False, description="Enable debug mode (verbose logging, auto-reload)")
    environment: str = Field(default="development", description="development | staging | production")

    # ── API Server ─────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    api_prefix: str = "/api/v1"
    cors_origins: List[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins. Vite dev server runs on 5173."
    )

    # ── LLM Providers ─────────────────────────────────────────
    # MegaLLM (primary provider — OpenAI-compatible endpoint)
    mega_api_key: Optional[str] = Field(default=None, description="MegaLLM API key")
    mega_base_url: str = "https://ai.megallm.io/v1"
    mega_model: str = "gemini-3-pro-preview"

    # Gemini (fallback provider — supports key rotation)
    google_api_key: Optional[str] = Field(default=None, description="Single Gemini API key (fallback)")
    google_api_key_1: Optional[str] = Field(default=None, description="Gemini key 1 for rotation")
    google_api_key_2: Optional[str] = Field(default=None, description="Gemini key 2 for rotation")
    google_api_key_3: Optional[str] = Field(default=None, description="Gemini key 3 for rotation")
    gemini_model: str = Field(
        default="gemini-3.6-flash",
        description=(
            "Gemini fallback model. Was gemini-2.5-flash here while the legacy "
            "path defaulted to gemini-3.6-flash; the two stacks answered the "
            "same questions with different models. Unified on 3.6-flash. Note "
            "2.5-flash still resolves for older API projects but is not served "
            "to projects created after its deprecation."
        )
    )

    # LLM behavior
    llm_temperature: float = Field(default=0.1, description="Temperature for RAG answers (low = consistent)")
    llm_max_retries: int = Field(default=3, description="Max LLM API retries before giving up")
    llm_retry_delay: float = Field(default=2.0, description="Seconds between LLM retries")
    llm_max_concurrency: int = Field(
        default=3,
        description=(
            "Max simultaneous LLM calls. The article pipeline writes sections "
            "in parallel; without a cap it can burn the Gemini free tier's "
            "15 requests/minute in a single burst."
        )
    )

    # ── Embeddings ─────────────────────────────────────────────
    embedding_model: str = Field(
        default="BAAI/bge-base-en-v1.5",
        description=(
            "HuggingFace embedding model. bge-base-en-v1.5 is 768-dim with "
            "512-token context — significantly better than MiniLM for retrieval. "
            "Still runs locally, no API needed."
        )
    )
    embedding_device: str = Field(default="cpu", description="Device for embeddings: 'cpu' or 'cuda'")
    sparse_model: str = Field(
        default="Qdrant/bm25",
        description=(
            "FastEmbed sparse model for keyword matching. Qdrant scores these "
            "with real BM25 (IDF applied server-side via the collection's IDF "
            "modifier), replacing the in-process Python BM25 index."
        )
    )

    # ── Vector Store ───────────────────────────────────────────
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description=(
            "Qdrant server URL. Use ':memory:' for an in-process instance "
            "(tests) — it supports the same Query API including sparse vectors."
        )
    )
    qdrant_collection: str = Field(
        default="ayurveda_rag_v2",
        description=(
            "Collection name. Suffixed v2 to avoid colliding with the old "
            "384-d Chroma-era collection during migration."
        )
    )

    # ── Retrieval ──────────────────────────────────────────────
    retrieval_top_k: int = Field(default=10, description="How many chunks to retrieve from vector search")
    retrieval_top_n: int = Field(default=5, description="How many chunks to keep after reranking")
    retrieval_context_n: int = Field(default=3, description="How many chunks to put in LLM context")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description=(
            "Cross-encoder model for reranking. Cross-encoders compare query+doc "
            "jointly (not separately like bi-encoders), so they're much more accurate "
            "for ranking but too slow for initial retrieval."
        )
    )
    # Note: hybrid fusion is Reciprocal Rank Fusion, computed by Qdrant.
    # RRF is rank-based and parameter-free, so there are no dense/sparse
    # weights to tune. (The former bm25_weight/semantic_weight settings were
    # never read by any code path.)

    # ── Chunking ───────────────────────────────────────────────
    chunk_size_faq: int = 400
    chunk_size_product: int = 500
    chunk_size_guide: int = 800
    chunk_size_default: int = 600
    chunk_overlap: int = 100

    # ── Data Paths ─────────────────────────────────────────────
    content_dir: str = Field(default="data", description="Directory containing knowledge base documents")
    upload_dir: str = Field(default="uploads", description="Directory for user-uploaded documents")

    # ── Database ───────────────────────────────────────────────
    database_url: str = Field(
        default="sqlite+aiosqlite:///./kerala_ayurveda.db",
        description=(
            "Database connection string. SQLite for local dev, PostgreSQL for production. "
            "Example PostgreSQL: postgresql+asyncpg://user:pass@localhost:5432/kerala_ayurveda"
        )
    )

    # ── Cache ──────────────────────────────────────────────────
    redis_url: Optional[str] = Field(
        default=None,
        description=(
            "Redis connection URL. If not set, falls back to in-memory LRU cache. "
            "In-memory cache is fine for single-process dev but doesn't survive restarts "
            "and can't be shared across workers."
        )
    )
    cache_ttl_seconds: int = Field(default=86400, description="Cache TTL in seconds (default: 24 hours)")
    cache_max_size: int = Field(default=1000, description="Max entries in in-memory cache fallback")

    # ── Agent Workflow ─────────────────────────────────────────
    agent_max_iterations: int = Field(default=2, description="Max fact-check → revision cycles")
    agent_grounding_threshold: float = Field(default=0.7, description="Minimum grounding score to pass")
    agent_style_threshold: float = Field(default=0.7, description="Minimum style score to pass")
    agent_timeout_seconds: int = Field(default=120, description="Timeout per agent step")

    # ── Rate Limiting ──────────────────────────────────────────
    rate_limit_requests: int = Field(default=60, description="Max requests per minute per IP")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",  # Don't crash on unexpected env vars
    }

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"environment must be one of {allowed}, got '{v}'")
        return v

    @property
    def gemini_keys(self) -> List[str]:
        """Collect all available Gemini API keys for rotation."""
        keys = []
        for key in [self.google_api_key_1, self.google_api_key_2, self.google_api_key_3]:
            if key:
                keys.append(key.strip())
        if not keys and self.google_api_key:
            keys.append(self.google_api_key.strip())
        return keys

    @property
    def has_llm_keys(self) -> bool:
        """Check if any LLM provider is configured."""
        return bool(self.mega_api_key or self.gemini_keys)

    @property
    def chunk_sizes(self) -> dict:
        """Chunk size mapping by document type."""
        return {
            "faq": self.chunk_size_faq,
            "product": self.chunk_size_product,
            "guide": self.chunk_size_guide,
            "default": self.chunk_size_default,
        }

    @property
    def content_path(self) -> Path:
        return Path(self.content_dir)

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir)


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings singleton.

    WHY lru_cache:
    Settings reads from disk (.env file) and environment variables.
    We only want to do this once, not on every request. lru_cache
    ensures the same Settings object is returned every time.
    This is a common FastAPI pattern for dependency injection.
    """
    return Settings()
