"""Health check endpoints for monitoring and load balancers."""

from fastapi import APIRouter, Depends
from backend.app.api.deps import get_rag
from backend.app.services.rag.pipeline import RAGPipeline

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """
    Basic health check.

    WHY THIS EXISTS:
    Load balancers (nginx, AWS ALB, Kubernetes) poll this endpoint
    to determine if the server is alive. If it returns non-200,
    the load balancer stops sending traffic to this instance.

    This is also what Docker HEALTHCHECK uses (replaces the curl
    to Streamlit's /_stcore/health in the original Dockerfile).
    """
    return {"status": "healthy", "service": "kerala-ayurveda-api"}


@router.get("/health/ready")
async def readiness_check():
    """
    Readiness check — is the system ready to serve requests?

    The difference between health and ready:
    - /health: "Is the process running?" (always true if you can respond)
    - /health/ready: "Can I handle user requests?" (RAG pipeline loaded?)

    Kubernetes uses this to determine when to start routing traffic to
    a newly started pod. Without it, users hit a pod that hasn't finished
    loading its embedding model and get errors.
    """
    try:
        rag = get_rag()
        stats = rag.stats
        return {
            "status": "ready",
            "rag_initialized": stats["initialized"],
            "chunks_indexed": stats["chunks_indexed"],
            "llm_providers": stats["llm"],
            "cache": stats["cache"],
        }
    except Exception as e:
        return {
            "status": "not_ready",
            "reason": str(e),
        }
