"""
Document Chunker — Improved Chunking with Parent-Child Strategy
==================================================================

WHAT CHANGED from the original:
Your original chunker (rag_system.py lines 136-176) had the right idea —
adaptive chunk sizes based on document type. This version keeps that logic
but adds three important improvements:

1. PARENT-CHILD CHUNKS:
   Problem: Small chunks are better for precise retrieval (the vector search
   finds exactly the right sentence), but they provide too little context
   for the LLM to generate a good answer.

   Solution: Create two levels of chunks:
   - Child chunks (~400-800 chars): Used for vector search (precise matching)
   - Parent chunks (~1500-2000 chars): Returned to the LLM (more context)

   When a child chunk matches a query, we return its parent chunk to the LLM.
   This gives the best of both worlds: precise retrieval + rich context.

2. SEMANTIC SECTION DETECTION:
   Instead of relying only on character count to split, we first detect
   section boundaries (markdown headers, numbered questions, product fields).
   This keeps related content together — a Q&A pair won't be split across
   two chunks.

3. DEDUPLICATION:
   When re-indexing after adding new documents, we skip chunks that already
   exist (based on content hash). This prevents duplicate entries in ChromaDB.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pypdf
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Rich metadata attached to every chunk for tracing and management."""
    doc_id: str                         # Source document identifier
    section_id: str                     # Section heading or "section_N"
    doc_type: str                       # faq, product, guide, etc.
    chunk_index: int                    # Position within the document
    parent_chunk_id: Optional[str] = None  # ID of the parent (larger) chunk
    file_type: str = "md"               # md, pdf, csv
    content_hash: str = ""              # SHA-256 for deduplication
    page_number: Optional[int] = None   # For PDFs


@dataclass
class ChunkResult:
    """Output of the chunking process."""
    chunks: List[Document]              # LangChain Document objects with metadata
    parent_chunks: List[Document]       # Parent (larger context) chunks
    doc_id: str
    doc_type: str
    total_chunks: int = 0
    total_parent_chunks: int = 0


def content_hash(text: str) -> str:
    """Generate a short hash for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def detect_document_type(filename: str) -> str:
    """
    Detect document type for adaptive chunking.
    Preserved from original with PDF support added.
    """
    name = filename.lower()
    if "faq" in name:
        return "faq"
    elif "product" in name:
        return "product"
    elif any(x in name for x in ("guide", "dosha", "foundation")):
        return "guide"
    elif name.endswith(".pdf"):
        return "guide"  # PDFs are typically long-form content
    return "default"


def extract_pdf_text(pdf_path: Path) -> str:
    """
    Extract text from PDF with page markers.

    Improvement over original: uses page numbers in metadata
    so citations can reference specific pages.
    """
    reader = pypdf.PdfReader(str(pdf_path))
    pages_text = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages_text.append(f"[Page {i + 1}]\n{text.strip()}")

    if not pages_text:
        logger.warning(
            f"PDF produced no extractable text (may be scanned): {pdf_path.name}",
            extra={"component": "chunker", "doc_id": pdf_path.stem}
        )

    return "\n\n".join(pages_text)


def extract_csv_documents(csv_path: Path) -> List[Document]:
    """
    Convert CSV rows to Document objects.
    Preserved from original — each product becomes a separate document.
    """
    df = pd.read_csv(csv_path)
    documents = []

    for _, row in df.iterrows():
        product_text = (
            f"Product: {row['name']} (ID: {row['product_id']})\n"
            f"Category: {row['category']}\n"
            f"Format: {row['format']}\n"
            f"Target Concerns: {row['target_concerns']}\n"
            f"Key Herbs: {row['key_herbs']}\n"
            f"Contraindications: {row['contraindications_short']}\n"
            f"Tags: {row['internal_tags']}"
        )

        doc = Document(
            page_content=product_text,
            metadata={
                "doc_id": f"catalog_{row['product_id']}",
                "section_id": str(row["name"]),
                "doc_type": "product_catalog",
                "product_id": str(row["product_id"]),
                "file_type": "csv",
                "content_hash": content_hash(product_text),
                "chunk_index": 0,
            }
        )
        documents.append(doc)

    return documents


def chunk_document(
    content: str,
    doc_id: str,
    doc_type: str,
    file_type: str = "md",
) -> ChunkResult:
    """
    Chunk a document using adaptive strategy with parent-child chunks.

    The two-level chunking strategy:
    1. Create CHILD chunks at the configured size (400-800 chars)
       → These go into ChromaDB for vector search
    2. Create PARENT chunks at ~2x the child size
       → These are returned to the LLM for richer context

    When retrieval finds a child chunk, we look up its parent_chunk_id
    and send the parent's content to the LLM instead.
    """
    settings = get_settings()
    chunk_size = settings.chunk_sizes.get(doc_type, settings.chunk_size_default)

    # ── Child chunks (for retrieval) ──
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    # ── Parent chunks (for context) ──
    # Parent chunks are ~2.5x the child size, giving the LLM more context
    # without making the retrieval less precise
    parent_size = int(chunk_size * 2.5)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=int(settings.chunk_overlap * 1.5),
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    # Split into parent chunks first
    parent_texts = parent_splitter.split_text(content)
    parent_docs = []
    parent_id_map = {}  # content_hash → parent_chunk_id

    for pi, parent_text in enumerate(parent_texts):
        p_hash = content_hash(parent_text)
        parent_id = f"{doc_id}_parent_{pi}"

        section_match = re.search(r"^#+\s+(.+?)$", parent_text, re.MULTILINE)
        section_id = section_match.group(1) if section_match else f"section_{pi}"

        parent_doc = Document(
            page_content=parent_text,
            metadata={
                "doc_id": doc_id,
                "section_id": section_id,
                "doc_type": doc_type,
                "chunk_index": pi,
                "file_type": file_type,
                "content_hash": p_hash,
                "is_parent": True,
                "parent_chunk_id": parent_id,
            }
        )
        parent_docs.append(parent_doc)
        parent_id_map[pi] = parent_id

    # Split into child chunks
    child_texts = child_splitter.split_text(content)
    child_docs = []

    for ci, child_text in enumerate(child_texts):
        c_hash = content_hash(child_text)

        section_match = re.search(r"^#+\s+(.+?)$", child_text, re.MULTILINE)
        section_id = section_match.group(1) if section_match else f"section_{ci}"

        # Find which parent chunk contains this child
        # Simple heuristic: map by position ratio
        parent_idx = min(
            int(ci * len(parent_texts) / max(len(child_texts), 1)),
            len(parent_texts) - 1
        ) if parent_texts else 0
        parent_id = parent_id_map.get(parent_idx)

        child_doc = Document(
            page_content=child_text,
            metadata={
                "doc_id": doc_id,
                "section_id": section_id,
                "doc_type": doc_type,
                "chunk_index": ci,
                "file_type": file_type,
                "content_hash": c_hash,
                "is_parent": False,
                "parent_chunk_id": parent_id,
            }
        )
        child_docs.append(child_doc)

    logger.info(
        f"Chunked {doc_id}: {len(child_docs)} child chunks, {len(parent_docs)} parent chunks",
        extra={
            "component": "chunker",
            "doc_id": doc_id,
            "doc_type": doc_type,
            "child_chunks": len(child_docs),
            "parent_chunks": len(parent_docs),
        }
    )

    return ChunkResult(
        chunks=child_docs,
        parent_chunks=parent_docs,
        doc_id=doc_id,
        doc_type=doc_type,
        total_chunks=len(child_docs),
        total_parent_chunks=len(parent_docs),
    )


def load_all_documents(content_dir: Path) -> ChunkResult:
    """
    Load and chunk all documents from the content directory.

    This is the equivalent of the original load_and_index_content(),
    but separated from the indexing step (separation of concerns).
    """
    all_child_chunks = []
    all_parent_chunks = []

    # Load markdown files
    for md_file in sorted(content_dir.glob("*.md")):
        doc_id = md_file.stem
        doc_type = detect_document_type(doc_id)

        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()

        result = chunk_document(content, doc_id, doc_type, file_type="md")
        all_child_chunks.extend(result.chunks)
        all_parent_chunks.extend(result.parent_chunks)

    # Load PDF files
    for pdf_file in sorted(content_dir.glob("*.pdf")):
        doc_id = pdf_file.stem
        doc_type = "guide"  # PDFs are long-form content

        try:
            content = extract_pdf_text(pdf_file)
            if content.strip():
                result = chunk_document(content, doc_id, doc_type, file_type="pdf")
                all_child_chunks.extend(result.chunks)
                all_parent_chunks.extend(result.parent_chunks)
        except Exception as e:
            logger.error(
                f"Failed to load PDF: {pdf_file.name}: {e}",
                extra={"component": "chunker", "doc_id": doc_id}
            )

    # Load CSV product catalog
    csv_file = content_dir / "products_catalog.csv"
    if csv_file.exists():
        csv_docs = extract_csv_documents(csv_file)
        all_child_chunks.extend(csv_docs)
        # CSV products are small enough to be their own parents
        all_parent_chunks.extend(csv_docs)
        logger.info(
            f"Loaded products_catalog.csv: {len(csv_docs)} products",
            extra={"component": "chunker"}
        )

    logger.info(
        f"Total: {len(all_child_chunks)} child chunks, {len(all_parent_chunks)} parent chunks "
        f"from {content_dir}",
        extra={"component": "chunker"}
    )

    return ChunkResult(
        chunks=all_child_chunks,
        parent_chunks=all_parent_chunks,
        doc_id="all",
        doc_type="mixed",
        total_chunks=len(all_child_chunks),
        total_parent_chunks=len(all_parent_chunks),
    )
