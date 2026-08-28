"""
RAG Q&A API Routes
====================

WHY REST API instead of Streamlit callbacks:
Streamlit re-runs the ENTIRE Python script on every user interaction.
With a REST API:
  - Multiple users send requests concurrently (Streamlit is single-threaded)
  - Any frontend (React, mobile app, Slack bot) can call the same API
  - Each request is isolated — one user's error doesn't crash another's session
  - You can load-test the API with tools like k6 or locust
  - The API generates OpenAPI docs automatically (visit /docs)

TEACHING POINT — Request/Response Models:
Pydantic models (QueryRequest, QueryResponseModel) serve as:
  1. Input validation — malformed requests get a 422 error with details
  2. Documentation — FastAPI auto-generates API docs from these models
  3. Serialization — Python objects → JSON automatically
  4. IDE support — autocomplete and type checking in your editor
"""

import time
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.app.api.deps import get_rag
from backend.app.services.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["RAG Q&A"])


# ── Request/Response Models ──

class QueryRequest(BaseModel):
    """Request body for RAG Q&A queries."""
    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The user's question about Kerala Ayurveda",
        examples=["What are the benefits of Ashwagandha for stress?"],
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to use response cache (disable for evaluation)"
    )


class CitationModel(BaseModel):
    """A single citation in the response."""
    doc_id: str
    section_id: str
    content_snippet: str
    relevance_score: float


class QueryResponseModel(BaseModel):
    """Response body for RAG Q&A queries."""
    answer: str
    citations: List[CitationModel]
    chunks_retrieved: int
    cache_hit: bool = False
    latency_ms: float = 0.0
    model_used: str = ""


# ── Routes ──

@router.post(
    "",
    response_model=QueryResponseModel,
    summary="Ask the Kerala Ayurveda knowledge base",
    description=(
        "Submit a question and receive an answer grounded in the Kerala Ayurveda "
        "knowledge base, with citations linking back to source documents."
    ),
)
async def query_knowledge_base(
    request: QueryRequest,
    rag: RAGPipeline = Depends(get_rag),
):
    """
    Main RAG Q&A endpoint.

    This replaces what was previously done inside the Streamlit callback
    (streamlit_app.py lines 182-204). The logic is the same:
      1. Take user query
      2. Retrieve relevant chunks
      3. Generate answer with citations
      4. Return structured response

    But now it's:
      - Concurrent (multiple users at once)
      - Cacheable (same query = instant response)
      - Testable (just send an HTTP request)
      - Documented (visit /docs to see the API schema)
    """
    try:
        response = await rag.aquery(request.query, use_cache=request.use_cache)

        return QueryResponseModel(
            answer=response.answer,
            citations=[
                CitationModel(
                    doc_id=c.doc_id,
                    section_id=c.section_id,
                    content_snippet=c.content_snippet,
                    relevance_score=c.relevance_score,
                )
                for c in response.citations
            ],
            chunks_retrieved=len(response.retrieved_chunks),
            cache_hit=response.cache_hit,
            latency_ms=response.latency_ms,
            model_used=response.model_used,
        )

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True, extra={"component": "api"})
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )


@router.post(
    "/reindex",
    summary="Re-index documents",
    description=(
        "Syncs the vector index with the content directory. Incremental by "
        "default — files whose content is unchanged are skipped without "
        "re-embedding. Pass force=true to drop the collection and rebuild "
        "everything (required after changing the embedding model)."
    ),
)
async def reindex(
    force: bool = False,
    rag: RAGPipeline = Depends(get_rag),
):
    """Sync the index with the content directory."""
    try:
        stats = rag.sync(force=force)
        return {"status": "success", **stats}
    except Exception as e:
        logger.error(f"Reindex failed: {e}", exc_info=True, extra={"component": "api"})
        raise HTTPException(
            status_code=500,
            detail=f"Reindexing failed: {str(e)}"
        )


@router.get(
    "/stats",
    summary="Get RAG pipeline statistics",
)
async def get_stats(
    rag: RAGPipeline = Depends(get_rag),
):
    """Return pipeline statistics including cache performance and index size."""
    return rag.stats
