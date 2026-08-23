"""
Database Models — The Metadata Schema
========================================

WHY THESE MODELS EXIST:
ChromaDB stores vectors and raw text, but it's a terrible metadata store.
You can't efficiently answer questions like:
  - "Which documents were uploaded in the last week?"
  - "What's the average query latency?"
  - "Show me the article generation job that failed yesterday"

These SQLAlchemy models define the relational structure for all metadata
that ISN'T vector embeddings. Think of it as:
  - ChromaDB = "find similar documents" (vector search)
  - PostgreSQL/SQLite = "manage everything else" (CRUD, analytics, audit trail)

TEACHING POINT — Why separate tables instead of one big table:
Each table represents a distinct entity with its own lifecycle:
  - Documents exist independently of queries
  - Chunks belong to documents (if a document is deleted, its chunks go too)
  - QueryLogs are append-only analytics (never edited)
  - ArticleJobs track async workflows (have state transitions)

This is called "normalization" — it prevents data duplication and makes
updates safe. If you store document metadata inside every chunk row,
updating a document's title means updating 50+ chunk rows.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum as PyEnum
from typing import Optional, List

from sqlalchemy import (
    String, Text, Float, Integer, Boolean, DateTime,
    ForeignKey, JSON, Enum,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.core.database import Base


def utcnow() -> datetime:
    """UTC timestamp factory for default values."""
    return datetime.now(timezone.utc)


def generate_uuid() -> str:
    """Generate a short, readable UUID."""
    return str(uuid.uuid4())[:12]


# ── Document Model ─────────────────────────────────────────────

class DocumentStatus(str, PyEnum):
    """
    Document lifecycle states.

    TEACHING POINT — Why a state machine:
    A document goes through stages: uploaded → processing → indexed.
    Without explicit states, you'd check "does it have chunks?" to know
    if it's indexed, "is there an error message?" to know if it failed.
    Explicit states make the logic clear and queryable.
    """
    PENDING = "pending"        # Uploaded, waiting for processing
    PROCESSING = "processing"  # Being chunked and embedded
    INDEXED = "indexed"        # Successfully added to vector store
    FAILED = "failed"          # Processing failed
    DELETED = "deleted"        # Soft-deleted (chunks removed from vector store)


class Document(Base):
    """
    Represents an uploaded knowledge base document.

    This tracks the source file, its processing status, and metadata.
    The actual text content lives in Chunk rows; vector embeddings
    live in ChromaDB.
    """
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(12), primary_key=True, default=generate_uuid)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(20), nullable=False)  # md, pdf, csv, docx
    doc_type: Mapped[str] = mapped_column(String(50), nullable=False)   # faq, product, guide, etc.
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=True, unique=True)  # SHA-256 for dedup

    status: Mapped[str] = mapped_column(
        Enum(DocumentStatus),
        default=DocumentStatus.PENDING,
        nullable=False,
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationship: a document has many chunks
    chunks: Mapped[List["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Document {self.id}: {self.filename} ({self.status})>"


# ── Chunk Model ────────────────────────────────────────────────

class Chunk(Base):
    """
    Represents a single chunk of a document.

    WHY store chunks in SQL when they're also in ChromaDB?
    ChromaDB stores the vector + raw text for similarity search.
    SQL stores the metadata for management:
      - Which document does this chunk belong to?
      - What's the section heading?
      - What's its position in the document?
    This lets us do things like "delete all chunks for document X"
    without scanning the entire ChromaDB collection.
    """
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(12), primary_key=True, default=generate_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    content: Mapped[str] = mapped_column(Text, nullable=False)
    section_id: Mapped[str] = mapped_column(String(255), nullable=False)
    doc_type: Mapped[str] = mapped_column(String(50), nullable=False)

    # Parent chunk ID for parent-child retrieval strategy
    parent_chunk_id: Mapped[Optional[str]] = mapped_column(String(12), nullable=True)

    # ChromaDB reference — the ID used in the vector store
    chroma_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    char_count: Mapped[int] = mapped_column(Integer, default=0)
    token_count_approx: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # Relationship back to document
    document: Mapped["Document"] = relationship(back_populates="chunks")

    def __repr__(self) -> str:
        return f"<Chunk {self.id}: doc={self.document_id} idx={self.chunk_index}>"


# ── Query Log Model ────────────────────────────────────────────

class QueryLog(Base):
    """
    Audit trail for every RAG query.

    WHY log queries:
    1. Debugging: "User got a wrong answer" → find the exact query,
       retrieved chunks, and generated answer
    2. Analytics: "What are users asking about most?" → product decisions
    3. Evaluation: Build golden sets from real user queries
    4. Cost tracking: How many LLM tokens are we using?
    """
    __tablename__ = "query_logs"

    id: Mapped[str] = mapped_column(String(12), primary_key=True, default=generate_uuid)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)

    # Retrieval metadata
    chunks_retrieved: Mapped[int] = mapped_column(Integer, default=0)
    chunks_used: Mapped[int] = mapped_column(Integer, default=0)
    avg_relevance_score: Mapped[float] = mapped_column(Float, default=0.0)

    # Citations
    citations_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON string

    # Performance
    retrieval_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    generation_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    total_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)

    # LLM metadata
    model_used: Mapped[str] = mapped_column(String(100), nullable=True)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)

    # Cache
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    def __repr__(self) -> str:
        return f"<QueryLog {self.id}: '{self.query[:50]}...'>"


# ── Article Job Model ──────────────────────────────────────────

class ArticleJobStatus(str, PyEnum):
    """Article generation pipeline states."""
    QUEUED = "queued"
    OUTLINING = "outlining"
    WRITING = "writing"
    FACT_CHECKING = "fact_checking"
    TONE_EDITING = "tone_editing"
    COMPLETED = "completed"
    FAILED = "failed"


class ArticleJob(Base):
    """
    Tracks async article generation jobs.

    WHY a separate table:
    Article generation takes 2-4 minutes with 4 sequential LLM calls.
    If this were synchronous, the user's browser connection would timeout.
    Instead:
      1. User submits a brief → API returns job_id immediately
      2. Background worker runs the pipeline → updates job status
      3. Frontend polls GET /articles/{job_id} for progress
      4. When status=completed, frontend shows the result
    """
    __tablename__ = "article_jobs"

    id: Mapped[str] = mapped_column(String(12), primary_key=True, default=generate_uuid)

    # Brief
    topic: Mapped[str] = mapped_column(Text, nullable=False)
    target_audience: Mapped[str] = mapped_column(Text, nullable=False)
    key_points_json: Mapped[str] = mapped_column(Text, nullable=False)  # JSON array
    word_count_target: Mapped[int] = mapped_column(Integer, default=800)
    products_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array

    # Status
    status: Mapped[str] = mapped_column(
        Enum(ArticleJobStatus),
        default=ArticleJobStatus.QUEUED,
        nullable=False,
    )
    current_step: Mapped[int] = mapped_column(Integer, default=0)  # 0-4
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Results (populated as pipeline progresses)
    outline_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    draft_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    final_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Scores
    fact_check_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    style_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    citations_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    editor_notes_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ready_for_editor: Mapped[bool] = mapped_column(Boolean, default=False)

    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    def __repr__(self) -> str:
        return f"<ArticleJob {self.id}: '{self.topic[:40]}...' ({self.status})>"
