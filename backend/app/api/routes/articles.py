"""
Article Generation API Routes
================================

WHY async job model:
Article generation takes 2-4 minutes (4 sequential LLM calls).
Making this synchronous would:
  1. Block the server for 4 minutes per request
  2. Timeout most HTTP clients (30-60s default)
  3. Leave the user staring at a spinner with no progress info

Instead, we use a job model:
  1. POST /articles/generate → returns job_id immediately (~50ms)
  2. Background task runs the pipeline, updating status in the job dict
  3. GET /articles/{job_id} → returns current status + partial results
  4. Frontend polls this endpoint every 2-3 seconds for progress

In a full production setup, the background task would run in a separate
worker process via Celery/Redis. For this implementation, we use
FastAPI's BackgroundTasks which runs in the same process but doesn't
block the response.
"""

import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import List, Optional, Dict
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from backend.app.api.deps import get_rag
from backend.app.services.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/articles", tags=["Article Generation"])

# In-memory job store (production would use Redis or database)
_jobs: Dict[str, dict] = {}


# ── Request/Response Models ──

class ArticleBriefRequest(BaseModel):
    """Request body for article generation."""
    topic: str = Field(..., min_length=5, max_length=500)
    target_audience: str = Field(..., min_length=5, max_length=500)
    key_points: List[str] = Field(..., min_length=1)
    word_count_target: int = Field(default=800, ge=300, le=2000)
    must_include_products: List[str] = Field(default_factory=list)


class ArticleJobResponse(BaseModel):
    """Current state of an article generation job."""
    job_id: str
    status: str  # queued, outlining, writing, fact_checking, tone_editing, completed, failed
    current_step: int  # 0-4
    total_steps: int = 4

    # Partial results (populated as pipeline progresses)
    outline: Optional[dict] = None
    fact_check_score: Optional[float] = None
    style_score: Optional[float] = None
    final_content: Optional[str] = None
    citations: Optional[List[dict]] = None
    editor_notes: Optional[List[str]] = None
    ready_for_editor: bool = False
    error_message: Optional[str] = None

    created_at: str = ""
    completed_at: Optional[str] = None


# ── Background task ──

def run_article_pipeline(job_id: str, brief: ArticleBriefRequest, rag: RAGPipeline):
    """
    Run the multi-agent article pipeline as a background task.

    This function updates _jobs[job_id] as it progresses through each step,
    so the polling endpoint can report real-time status.
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

    job = _jobs[job_id]

    try:
        # Import agent modules
        from src.agent_workflow import (
            ArticleWorkflowOrchestrator, ArticleBrief,
        )
        from src.rag_system import AyurvedaRAGSystem

        # We need the original RAG system for the agents (they depend on its interface)
        # This is a transitional bridge — eventually agents will use the new RAG pipeline directly
        original_rag = AyurvedaRAGSystem()
        original_rag.load_and_index_content()
        orchestrator = ArticleWorkflowOrchestrator(original_rag)

        # Build the brief
        agent_brief = ArticleBrief(
            topic=brief.topic,
            target_audience=brief.target_audience,
            key_points=brief.key_points,
            word_count_target=brief.word_count_target,
            must_include_products=brief.must_include_products,
        )

        # Step 1: Outline
        job["status"] = "outlining"
        job["current_step"] = 1
        outline = orchestrator.outline_agent.generate_outline(agent_brief)
        job["outline"] = {
            "title": outline.title,
            "sections": outline.sections,
        }

        # Step 2: Write
        job["status"] = "writing"
        job["current_step"] = 2
        draft = orchestrator.writer_agent.write_draft(agent_brief, outline)

        # Step 3: Fact-check
        job["status"] = "fact_checking"
        job["current_step"] = 3
        fact_check = orchestrator.fact_checker.fact_check(draft)
        job["fact_check_score"] = fact_check.grounding_score

        # Step 4: Tone edit
        job["status"] = "tone_editing"
        job["current_step"] = 4
        tone = orchestrator.tone_editor.edit_tone(draft, fact_check)
        job["style_score"] = tone.style_score

        # Final result
        final_content = tone.revised_content if tone.revised_content != "NO CHANGES" else draft.content
        ready = (
            fact_check.grounding_score >= 0.7
            and tone.style_score >= 0.7
            and len(draft.citations) > 0
        )

        job["status"] = "completed"
        job["final_content"] = final_content
        job["citations"] = draft.citations
        job["ready_for_editor"] = ready
        job["editor_notes"] = []

        if fact_check.grounding_score < 0.9:
            job["editor_notes"].append(
                f"Fact-check: Some claims may need verification (score: {fact_check.grounding_score:.2f})"
            )
        if tone.style_score < 0.85:
            job["editor_notes"].append(
                f"Style: Minor tone adjustments may be needed (score: {tone.style_score:.2f})"
            )

        job["completed_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(
            f"Article pipeline completed: job={job_id}, topic='{brief.topic[:40]}'",
            extra={"component": "articles"}
        )

    except Exception as e:
        logger.error(f"Article pipeline failed: {e}", exc_info=True, extra={"component": "articles"})
        job["status"] = "failed"
        job["error_message"] = str(e)


# ── Routes ──

@router.post(
    "/generate",
    response_model=ArticleJobResponse,
    summary="Start article generation",
    description="Kicks off the multi-agent article pipeline and returns a job ID for tracking.",
)
async def generate_article(
    brief: ArticleBriefRequest,
    background_tasks: BackgroundTasks,
    rag: RAGPipeline = Depends(get_rag),
):
    """Start async article generation and return job ID immediately."""
    job_id = str(uuid4())[:12]

    _jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "current_step": 0,
        "total_steps": 4,
        "outline": None,
        "fact_check_score": None,
        "style_score": None,
        "final_content": None,
        "citations": None,
        "editor_notes": None,
        "ready_for_editor": False,
        "error_message": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
    }

    # Run pipeline in background
    background_tasks.add_task(run_article_pipeline, job_id, brief, rag)

    logger.info(
        f"Article job created: {job_id}, topic='{brief.topic[:40]}'",
        extra={"component": "articles"}
    )

    return ArticleJobResponse(**_jobs[job_id])


@router.get(
    "/{job_id}",
    response_model=ArticleJobResponse,
    summary="Get article generation status",
    description="Poll this endpoint to track progress of article generation.",
)
async def get_article_status(job_id: str):
    """Get the current status and results of an article generation job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return ArticleJobResponse(**_jobs[job_id])


@router.get(
    "",
    summary="List all article jobs",
)
async def list_article_jobs():
    """List all article generation jobs (recent first)."""
    jobs = sorted(
        _jobs.values(),
        key=lambda j: j.get("created_at", ""),
        reverse=True,
    )
    return [ArticleJobResponse(**j) for j in jobs[:20]]  # Limit to 20 most recent
