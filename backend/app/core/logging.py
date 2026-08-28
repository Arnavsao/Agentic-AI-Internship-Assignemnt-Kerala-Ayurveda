"""
Structured Logging — Production-Grade Observability
=====================================================

WHY THIS EXISTS:
The original codebase uses print() statements for all output. In production,
this creates several problems:
  1. print() output in a Docker container goes to stdout but has no structure —
     you can't search, filter, or aggregate logs
  2. No way to correlate logs from a single request across multiple function calls
  3. No severity levels — a warning looks the same as a critical error
  4. No timestamps in a consistent format

HOW IT WORKS:
We configure Python's standard logging library with:
  - JSON formatter for production (machines can parse it)
  - Human-readable formatter for development (you can read it)
  - Correlation IDs: each request gets a unique ID that appears in every log
    entry for that request, so you can trace "what happened to request X?"
  - Module-specific log levels (e.g., suppress noisy HuggingFace logs)

TEACHING POINT:
Structured logging is the foundation of observability. When your system
is running on a cloud server and a user reports "I got a wrong answer,"
the ONLY way to debug it is through logs. If your logs are unstructured
print statements, you're debugging blind.
"""

import logging
import sys
import json
import time
import uuid
from contextvars import ContextVar
from typing import Optional

# Context variable to track correlation IDs across async calls
# ContextVar is the async-safe version of thread-local storage
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> str:
    """Get or create a correlation ID for the current request."""
    cid = correlation_id_var.get()
    if cid is None:
        cid = str(uuid.uuid4())[:8]  # Short ID for readability
        correlation_id_var.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    """Set correlation ID (typically called by middleware at request start)."""
    correlation_id_var.set(cid)


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for production.

    Why JSON:
    Cloud logging services (CloudWatch, Stackdriver, Datadog) can
    automatically parse JSON logs and let you search by any field.
    "Show me all ERROR logs where component=rag and latency > 5s"
    is trivial with structured logs, impossible with print().
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_var.get(),
        }

        # Add exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add any extra fields (e.g., latency, model, tokens)
        for key in ("latency_ms", "model", "tokens_used", "component",
                     "query", "doc_id", "chunk_count", "status_code"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        return json.dumps(log_entry, default=str)


class DevFormatter(logging.Formatter):
    """
    Human-readable formatter for development.
    Color-coded by severity level for terminal readability.
    """

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        cid = correlation_id_var.get() or "------"

        # Format: [TIME] LEVEL [CID] logger: message
        time_str = self.formatTime(record, "%H:%M:%S")
        msg = f"{color}[{time_str}] {record.levelname:<8}{self.RESET} [{cid}] {record.name}: {record.getMessage()}"

        if record.exc_info and record.exc_info[0] is not None:
            msg += f"\n{self.formatException(record.exc_info)}"

        return msg


def setup_logging(environment: str = "development", debug: bool = False) -> None:
    """
    Configure application-wide logging.

    Args:
        environment: "development" uses colored console output;
                     "production" uses JSON format
        debug: If True, set root level to DEBUG
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Remove existing handlers (prevents duplicate logs on re-init)
    root_logger.handlers.clear()

    # Create handler
    handler = logging.StreamHandler(sys.stdout)

    if environment == "production":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(DevFormatter())

    root_logger.addHandler(handler)

    # ── Suppress noisy third-party loggers ──────────────────────
    # These libraries log at INFO/WARNING level by default, which
    # floods your output. We keep them at ERROR so only real problems
    # come through.
    noisy_loggers = [
        "transformers",
        "huggingface_hub",
        "sentence_transformers",
        "qdrant_client",
        "fastembed",
        "httpx",
        "httpcore",
        "urllib3",
        "asyncio",
        "uvicorn.access",  # Access logs (handled by middleware instead)
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    # Uvicorn's error logger should still show
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)

    logging.getLogger("app").info(
        "Logging configured",
        extra={"component": "logging", "environment": environment}
    )


class LogTimer:
    """
    Context manager to log execution time of a block of code.

    Usage:
        with LogTimer(logger, "rag_retrieval", query="What is Vata?"):
            results = retriever.search(query)

    This logs:
        [12:34:56] INFO     [abc123] rag: rag_retrieval completed in 234ms
    """

    def __init__(self, logger: logging.Logger, operation: str, **extra):
        self.logger = logger
        self.operation = operation
        self.extra = extra
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        if exc_type:
            self.logger.error(
                f"{self.operation} failed after {elapsed_ms:.0f}ms: {exc_val}",
                extra={"latency_ms": elapsed_ms, **self.extra}
            )
        else:
            self.logger.info(
                f"{self.operation} completed in {elapsed_ms:.0f}ms",
                extra={"latency_ms": elapsed_ms, **self.extra}
            )
        return False  # Don't suppress exceptions
