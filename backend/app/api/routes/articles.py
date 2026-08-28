"""
Article Generation Routes
===========================

Article generation takes roughly a minute, so it runs as a background job:
POST returns a job ID immediately and the client polls for progress.

WHAT CHANGED:

1. JOBS SURVIVE RESTARTS.
   State lived in a module-level `_jobs` dict — process-local, unbounded, and
   erased on every reload. A user polling across a deploy got a 404 for work
   that had actually completed. Jobs now persist to the `article_jobs` table,
   which was defined in the schema but never used. A startup sweep marks jobs
   left mid-run by a crash as failed, so nothing polls forever.

2. NO SECOND RAG SYSTEM PER JOB.
   The old task built a fresh `AyurvedaRAGSystem()` inside every job —
   re-loading the embedding model and opening a second client against the
   same store — while ignoring the pipeline that was already injected into
   the route. Jobs now use the injected pipeline.

3. GENUINELY ASYNC.
   The task was a sync `def` running the pipeline's blocking calls in the
   threadpool. It's now async and awaits the LangGraph run, so section writes
   proceed concurrently and progress is recorded as each node completes.
"""

import json
import logging
from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, update

from backend.app.api.deps import get_rag
from backend.app.core.database import get_session_factory
from backend.app.models.schemas import ArticleJob, ArticleJobStatus
from backend.app.services.agents.graph import generate_article as run_graph
from backend.app.services.agents.models import ArticleBrief
from backend.app.services.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/articles", tags=["Article Generation"])

# Node name → (job status, step number) for progress reporting.
NODE_PROGRESS = {
    "outline": (ArticleJobStatus.OUTLINING, 1),
    "write_section": (ArticleJobStatus.WRITING, 2),
    "assemble_draft": (ArticleJobStatus.WRITING, 2),
    "fact_check": (ArticleJobStatus.FACT_CHECKING, 3),
    "revise": (ArticleJobStatus.FACT_CHECKING, 3),
    "tone_edit": (ArticleJobStatus.TONE_EDITING, 4),
}

TERMINAL_STATUSES = {ArticleJobStatus.COMPLETED, ArticleJobStatus.FAILED}


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
    status: str
    current_step: int
    total_steps: int = 4

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


def _status_value(status) -> str:
    return status.value if hasattr(status, "value") else str(status)


def _to_response(job: ArticleJob) -> ArticleJobResponse:
    """Map a job row to the API shape."""
    return ArticleJobResponse(
        job_id=job.id,
        status=_status_value(job.status),
        current_step=job.current_step,
        outline=json.loads(job.outline_json) if job.outline_json else None,
        fact_check_score=job.fact_check_score,
        style_score=job.style_score,
        final_content=job.final_content,
        citations=json.loads(job.citations_json) if job.citations_json else None,
        editor_notes=json.loads(job.editor_notes_json) if job.editor_notes_json else None,
        ready_for_editor=job.ready_for_editor,
        error_message=job.error_message,
        created_at=job.created_at.isoformat() if job.created_at else "",
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )


async def _update_job(job_id: str, **fields) -> None:
    """Patch a job row in its own short transaction."""
    factory = get_session_factory()
    async with factory() as session:
        await session.execute(
            update(ArticleJob).where(ArticleJob.id == job_id).values(**fields)
        )
        await session.commit()


async def sweep_stale_jobs() -> int:
    """
    Mark jobs left in a non-terminal state as failed.

    Called at startup: a job that was mid-run when the process died has no
    worker any more, and without this it would sit at "writing" forever while
    a client polls it.
    """
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            update(ArticleJob)
            .where(ArticleJob.status.not_in(list(TERMINAL_STATUSES)))
            .values(
                status=ArticleJobStatus.FAILED,
                error_message="Server restarted while this job was running.",
                completed_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()
        count = result.rowcount or 0

    if count:
        logger.warning(
            f"Marked {count} interrupted article job(s) as failed",
            extra={"component": "articles"},
        )
    return count


# ── Background task ──

async def run_article_pipeline(job_id: str, brief: ArticleBriefRequest, rag: RAGPipeline):
    """
    Run the LangGraph article pipeline, recording progress as it goes.

    Uses the injected pipeline's retriever and LLM gateway rather than
    constructing its own RAG stack.
    """
    await _update_job(
        job_id,
        status=ArticleJobStatus.OUTLINING,
        current_step=1,
        started_at=datetime.now(timezone.utc),
    )

    async def on_step(node_name: str, update_payload: dict) -> None:
        """Persist whatever this node produced so polling reflects real progress."""
        fields = {}
        status_step = NODE_PROGRESS.get(node_name)
        if status_step:
            fields["status"], fields["current_step"] = status_step

        if outline := update_payload.get("outline"):
            fields["outline_json"] = json.dumps({
                "title": outline.title,
                "sections": [s.model_dump() for s in outline.sections],
            })
        if draft := update_payload.get("draft"):
            fields["draft_content"] = draft.content
        if fact_check := update_payload.get("fact_check"):
            fields["fact_check_score"] = fact_check.grounding_score
        if final := update_payload.get("final"):
            fields["style_score"] = final.style_score
            fields["final_content"] = final.content
            fields["citations_json"] = json.dumps(final.citations)
            fields["editor_notes_json"] = json.dumps(final.editor_notes)
            fields["ready_for_editor"] = final.ready_for_editor

        if fields:
            await _update_job(job_id, **fields)

    try:
        state = await run_graph(
            brief=ArticleBrief(**brief.model_dump()),
            retriever=rag.retriever,
            llm_provider=rag.llm_provider,
            on_step=on_step,
        )

        final = state.final
        if final is None:
            raise RuntimeError("Pipeline finished without producing an article")

        await _update_job(
            job_id,
            status=ArticleJobStatus.COMPLETED,
            current_step=4,
            final_content=final.content,
            citations_json=json.dumps(final.citations),
            editor_notes_json=json.dumps(final.editor_notes),
            fact_check_score=final.fact_check_score,
            style_score=final.style_score,
            ready_for_editor=final.ready_for_editor,
            completed_at=datetime.now(timezone.utc),
        )

        logger.info(
            f"Article job {job_id} completed: grounding={final.fact_check_score:.2f}, "
            f"style={final.style_score:.2f}, ready={final.ready_for_editor}",
            extra={"component": "articles"},
        )

    except Exception as e:
        logger.error(
            f"Article job {job_id} failed: {e}",
            exc_info=True,
            extra={"component": "articles"},
        )
        await _update_job(
            job_id,
            status=ArticleJobStatus.FAILED,
            error_message=str(e),
            completed_at=datetime.now(timezone.utc),
        )


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
    """Start article generation and return the job ID immediately."""
    job_id = str(uuid4())[:12]

    factory = get_session_factory()
    async with factory() as session:
        session.add(ArticleJob(
            id=job_id,
            topic=brief.topic,
            target_audience=brief.target_audience,
            key_points_json=json.dumps(brief.key_points),
            word_count_target=brief.word_count_target,
            products_json=json.dumps(brief.must_include_products),
            status=ArticleJobStatus.QUEUED,
            current_step=0,
        ))
        await session.commit()

    background_tasks.add_task(run_article_pipeline, job_id, brief, rag)

    logger.info(
        f"Article job created: {job_id}, topic='{brief.topic[:40]}'",
        extra={"component": "articles"},
    )

    async with factory() as session:
        job = await session.get(ArticleJob, job_id)
        return _to_response(job)


@router.get(
    "/{job_id}",
    response_model=ArticleJobResponse,
    summary="Get article generation status",
    description="Poll this endpoint to track progress of article generation.",
)
async def get_article_status(job_id: str):
    """Current status and results of one job."""
    factory = get_session_factory()
    async with factory() as session:
        job = await session.get(ArticleJob, job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return _to_response(job)


@router.get(
    "",
    response_model=List[ArticleJobResponse],
    summary="List article jobs",
)
async def list_article_jobs(limit: int = 20):
    """Recent jobs, newest first."""
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(ArticleJob).order_by(ArticleJob.created_at.desc()).limit(limit)
        )
        jobs = result.scalars().all()

    return [_to_response(j) for j in jobs]
