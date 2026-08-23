"""
FastAPI Dependency Injection — Shared Resources
=================================================

WHY THIS EXISTS:
FastAPI's dependency injection system lets you define shared resources
(database sessions, RAG pipeline, LLM provider) ONCE and inject them
into any route that needs them.

Without DI:
  @router.post("/query")
  async def query(request):
      rag = RAGPipeline()  # Created every request! Loads 220MB model each time!
      db = connect_to_db()  # Creates new connection each time!
      ...

With DI:
  @router.post("/query")
  async def query(rag: RAGPipeline = Depends(get_rag)):
      ...  # Same pipeline instance reused across all requests

TEACHING POINT:
Dependency injection is NOT about making code "more abstract." It's about:
1. Resource management — expensive resources are created once, shared across requests
2. Testability — tests can inject mock dependencies
3. Lifecycle — FastAPI manages creation/cleanup automatically
"""

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db_session
from backend.app.services.rag.pipeline import get_rag_pipeline, RAGPipeline
from backend.app.services.llm import LLMProvider


async def get_db() -> AsyncSession:
    """Provide a database session to route handlers."""
    async for session in get_db_session():
        yield session


def get_rag() -> RAGPipeline:
    """Provide the RAG pipeline singleton to route handlers."""
    pipeline = get_rag_pipeline()
    if not pipeline._initialized:
        raise RuntimeError(
            "RAG pipeline not initialized. The server is still starting up."
        )
    return pipeline


def get_llm_provider() -> LLMProvider:
    """Provide the LLM provider to route handlers."""
    return get_rag_pipeline().llm_provider
