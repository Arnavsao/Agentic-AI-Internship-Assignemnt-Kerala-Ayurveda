"""
FastAPI Application — Production Backend Entry Point
=======================================================

WHY FastAPI instead of Streamlit as the backend:

┌────────────────────────┬──────────────────┬────────────────────┐
│ Capability             │ Streamlit        │ FastAPI            │
├────────────────────────┼──────────────────┼────────────────────┤
│ Concurrent users       │ 1 (single thread)│ 100+ (async)       │
│ Request isolation      │ Shared state     │ Isolated per req   │
│ API documentation      │ None             │ Auto-generated     │
│ Input validation       │ Manual           │ Pydantic (auto)    │
│ Background tasks       │ Not supported    │ Built-in           │
│ WebSocket streaming    │ Not supported    │ Built-in           │
│ Frontend flexibility   │ Streamlit only   │ Any frontend       │
│ Testing                │ Difficult        │ TestClient (easy)  │
│ Deployment scaling     │ 1 instance       │ N workers (gunicorn)│
│ CORS / auth / rate lim │ Not built-in     │ Middleware          │
└────────────────────────┴──────────────────┴────────────────────┘

TEACHING POINT — Lifespan Events:
FastAPI's lifespan context manager lets you define startup/shutdown logic.
The RAG pipeline initialization (loading embeddings, building index) happens
ONCE at startup, not on the first request. This means:
  - The first user doesn't wait 15 seconds for model loading
  - If initialization fails, the server doesn't start (fail fast)
  - Cleanup (closing DB connections) happens reliably on shutdown
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import get_settings
from backend.app.core.logging import setup_logging, set_correlation_id, get_correlation_id
from backend.app.core.database import init_db, close_db
from backend.app.services.rag.pipeline import get_rag_pipeline

# Import route modules
from backend.app.api.routes import health, query, documents, articles


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Startup: Initialize database, load RAG pipeline
    Shutdown: Close database connections, cleanup

    Everything in the 'yield' block is startup.
    Everything after 'yield' is shutdown.
    """
    settings = get_settings()
    setup_logging(environment=settings.environment, debug=settings.debug)
    logger = logging.getLogger("app")

    logger.info(
        f"Starting {settings.app_name} v{settings.app_version}",
        extra={"component": "app", "environment": settings.environment}
    )

    # ── Startup ──
    # 1. Initialize database
    await init_db()

    # 2. Fail any jobs that were mid-flight when the process last stopped.
    # Their worker is gone, so without this a client polls a job that will
    # never move off "writing".
    from backend.app.api.routes.articles import sweep_stale_jobs
    await sweep_stale_jobs()

    # 3. Initialize RAG pipeline. Loads the embedding model, ensures the
    # Qdrant collection exists and matches this model, and runs incremental
    # ingestion. Model/dimension mismatches raise here rather than degrading
    # retrieval silently at query time.
    logger.info("Initializing RAG pipeline...", extra={"component": "app"})
    rag = get_rag_pipeline()
    await asyncio.to_thread(rag.initialize)
    logger.info("RAG pipeline ready", extra={"component": "app"})

    logger.info(
        f"{settings.app_name} ready on http://{settings.host}:{settings.port}",
        extra={"component": "app"}
    )

    yield  # Application is running

    # ── Shutdown ──
    logger.info("Shutting down...", extra={"component": "app"})
    await close_db()
    logger.info("Shutdown complete", extra={"component": "app"})


def create_app() -> FastAPI:
    """
    Application factory.

    WHY a factory function:
    Instead of creating the app at module level (app = FastAPI()),
    a factory lets you create multiple app instances with different
    configurations (e.g., one for production, one for testing).
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production RAG system for Kerala Ayurveda with hybrid search, "
            "multi-agent article generation, and comprehensive observability. "
            "Visit /docs for interactive API documentation."
        ),
        lifespan=lifespan,
        docs_url="/docs",     # Swagger UI
        redoc_url="/redoc",   # ReDoc (alternative docs)
    )

    # ── CORS Middleware ──
    # WHY: Without CORS, a React app on localhost:5173 can't call
    # the API on localhost:8000. The browser blocks it as a security measure.
    # CORS headers explicitly allow cross-origin requests from our frontend.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request Logging Middleware ──
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """
        Log every request with timing and correlation ID.

        This middleware:
        1. Assigns a unique correlation ID to each request
        2. Measures request processing time
        3. Logs the request and response metadata

        The correlation ID appears in ALL log entries for this request,
        so you can trace "what happened during request abc123?" across
        every component (retriever, generator, cache, etc.).
        """
        # Generate correlation ID
        cid = str(uuid.uuid4())[:8]
        set_correlation_id(cid)

        start_time = time.perf_counter()
        logger = logging.getLogger("app.requests")

        # Log request
        logger.info(
            f"→ {request.method} {request.url.path}",
            extra={"component": "http", "status_code": None}
        )

        # Process request
        response = await call_next(request)

        # Log response
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"← {request.method} {request.url.path} → {response.status_code} ({latency_ms:.0f}ms)",
            extra={
                "component": "http",
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            }
        )

        # Add correlation ID to response headers (useful for debugging)
        response.headers["X-Correlation-ID"] = cid
        return response

    # ── Register Routes ──
    app.include_router(health.router)
    app.include_router(query.router, prefix=settings.api_prefix)
    app.include_router(documents.router, prefix=settings.api_prefix)
    app.include_router(articles.router, prefix=settings.api_prefix)

    return app


# Create the application instance
app = create_app()
