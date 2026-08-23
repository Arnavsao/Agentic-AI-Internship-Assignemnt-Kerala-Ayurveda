"""
Document Management API Routes
=================================

These endpoints let users upload new documents and manage the knowledge base
without restarting the application — something impossible in the original system.
"""

import logging
from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel

from backend.app.api.deps import get_rag
from backend.app.services.rag.pipeline import RAGPipeline
from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


class DocumentInfo(BaseModel):
    """Information about an indexed document."""
    filename: str
    doc_type: str
    file_type: str
    size_bytes: int


class UploadResponse(BaseModel):
    """Response after uploading a document."""
    filename: str
    status: str
    message: str


@router.get("", response_model=List[DocumentInfo])
async def list_documents():
    """List all documents in the knowledge base."""
    settings = get_settings()
    content_dir = settings.content_path
    documents = []

    if content_dir.exists():
        for f in sorted(content_dir.iterdir()):
            if f.is_file() and f.suffix in ('.md', '.pdf', '.csv'):
                from backend.app.services.rag.chunker import detect_document_type
                documents.append(DocumentInfo(
                    filename=f.name,
                    doc_type=detect_document_type(f.name),
                    file_type=f.suffix.lstrip('.'),
                    size_bytes=f.stat().st_size,
                ))

    return documents


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    rag: RAGPipeline = Depends(get_rag),
):
    """
    Upload a new document to the knowledge base.

    Accepts: .md, .pdf, .csv files
    After upload, the document is saved to the data directory.
    Call POST /api/v1/query/reindex to include it in search results.
    """
    # Validate file type
    allowed_extensions = {'.md', '.pdf', '.csv', '.txt'}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {allowed_extensions}"
        )

    settings = get_settings()
    content_dir = settings.content_path
    content_dir.mkdir(parents=True, exist_ok=True)

    # Save file
    dest_path = content_dir / file.filename
    try:
        content = await file.read()
        with open(dest_path, 'wb') as f:
            f.write(content)

        logger.info(
            f"Document uploaded: {file.filename} ({len(content)} bytes)",
            extra={"component": "documents", "doc_id": file.filename}
        )

        return UploadResponse(
            filename=file.filename,
            status="uploaded",
            message=f"File saved. Call POST /api/v1/query/reindex to include in search results.",
        )

    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True, extra={"component": "documents"})
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.delete("/{filename}")
async def delete_document(
    filename: str,
    rag: RAGPipeline = Depends(get_rag),
):
    """
    Delete a document from the knowledge base.

    After deletion, call POST /api/v1/query/reindex to update search results.
    """
    settings = get_settings()
    file_path = settings.content_path / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Document not found: {filename}")

    try:
        file_path.unlink()
        logger.info(f"Document deleted: {filename}", extra={"component": "documents"})
        return {
            "status": "deleted",
            "filename": filename,
            "message": "Call POST /api/v1/query/reindex to update search results.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")
