# ── Backend Dockerfile ──
# Multi-stage build for production deployment
#
# WHY multi-stage:
# Stage 1 (builder): Installs all build tools and compiles dependencies
# Stage 2 (runtime): Only copies the compiled packages — no build tools
# Result: Final image is ~1.5GB instead of ~3GB
#
# WHY non-root user:
# If an attacker exploits the app, they get the permissions of the process owner.
# Running as root = they own the entire container. Running as 'appuser' = they
# can only access /app. This is a defense-in-depth measure.

FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ──
FROM python:3.11-slim

# Install runtime-only dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application code
COPY backend/ ./backend/
COPY src/ ./src/
COPY data/ ./data/
COPY .env.example ./.env.example

# Create necessary directories
RUN mkdir -p chroma_db uploads && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Run with uvicorn
# --workers 1: Single worker since we're CPU-bound (embedding model)
# For I/O-bound apps you'd use more workers, but ML models need shared memory
CMD ["uvicorn", "backend.app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
