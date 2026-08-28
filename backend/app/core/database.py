"""
Database Layer — SQLAlchemy Async Engine
==========================================

WHY THIS EXISTS:
The original system has NO persistent metadata storage. Everything lives in
memory or ChromaDB. This means:
  - No record of which documents were uploaded, when, or by whom
  - No query logs for debugging or analytics
  - No way to track article generation jobs
  - If the app crashes mid-article-generation, the work is lost

HOW IT WORKS:
SQLAlchemy is an ORM (Object-Relational Mapper) — it lets you define database
tables as Python classes and query them with Python instead of raw SQL.

We use the ASYNC version because:
  - FastAPI is async — blocking database calls would defeat the purpose
  - An async DB call lets the server handle other requests while waiting for DB I/O
  - Example: 100 concurrent users, each query takes 5ms of DB time. Sync = 500ms total
    (sequential). Async = ~5ms total (concurrent).

TEACHING POINT — Connection Pooling:
A database connection is expensive to create (~50-100ms). Connection pooling
keeps a set of pre-created connections ready to use, so each request just
borrows one instead of creating a new one. pool_size=5 means 5 ready connections,
max_overflow=10 means up to 15 total under load.
"""

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy models.

    Every model class inherits from this. SQLAlchemy uses this to
    track all your table definitions and generate CREATE TABLE statements.
    """
    pass


# These are module-level but initialized lazily via init_db()
_engine = None
_session_factory = None


async def init_db() -> None:
    """
    Initialize database engine and create tables.

    Called once at application startup (FastAPI lifespan).
    Creates all tables if they don't exist (safe for development).
    In production, you'd use Alembic migrations instead.
    """
    global _engine, _session_factory

    settings = get_settings()

    logger.info(f"Initializing database: {settings.database_url.split('://')[0]}",
                extra={"component": "database"})

    _engine = create_async_engine(
        settings.database_url,
        echo=settings.debug,  # Log SQL queries in debug mode
        # Connection pool settings
        # pool_size: number of persistent connections kept open
        # max_overflow: additional connections allowed under load
        pool_size=5,
        max_overflow=10,
        # pool_recycle: recreate connections after this many seconds
        # (prevents stale connections after DB restarts)
        pool_recycle=3600,
    )

    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,  # Don't lazy-load after commit (async-unsafe)
    )

    # Create tables (dev only — production should use Alembic migrations)
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database initialized and tables created",
                extra={"component": "database"})


async def close_db() -> None:
    """Close database engine. Called at application shutdown."""
    global _engine
    if _engine:
        await _engine.dispose()
        logger.info("Database connections closed", extra={"component": "database"})


def get_session_factory() -> async_sessionmaker:
    """
    The raw session factory, for code outside the request cycle.

    `get_db_session` is a FastAPI dependency and only works where dependency
    injection applies. Background tasks outlive the request that started them,
    so they open their own short-lived sessions through this instead of
    holding a request-scoped one open for the length of the job.
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a database session.

    Usage in a route:
        @router.post("/query")
        async def query(session: AsyncSession = Depends(get_db_session)):
            ...

    WHY a context manager (yield):
    The session is automatically committed on success and rolled back on
    exception. Without this, every route would need try/except/finally
    boilerplate for database cleanup.
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
