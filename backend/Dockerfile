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
COPY scripts/ ./scripts/
COPY .env.example ./.env.example

# Create necessary directories
RUN mkdir -p uploads && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Pre-download the local models at build time.
# Without this, the first container start blocks for a minute or two fetching
# ~400 MB from HuggingFace — and fails outright in an air-gapped deploy.
# Costs roughly 1GB of image size to make startup deterministic and offline.
ENV HF_HOME=/home/appuser/.cache/huggingface
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
from fastembed import SparseTextEmbedding; \
SentenceTransformer('BAAI/bge-base-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
list(SparseTextEmbedding('Qdrant/bm25').embed(['warmup'])); \
print('models cached')"

# Mount point for the SQLite database volume.
# It must exist in the image AND be owned by appuser: Docker creates a missing
# mount point as root, and the container runs unprivileged, so without this the
# app cannot open its database file. When the path does exist, a fresh named
# volume inherits its ownership.
# Placed after the model layer so editing it doesn't invalidate that cache.
USER root
RUN mkdir -p /app/var && chown appuser:appuser /app/var
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Run with uvicorn
# --workers 1: each worker loads its own copy of the embedding and reranker
# models (~500 MB), so workers cost memory linearly. Now that the vector store
# is a separate Qdrant service rather than an embedded SQLite file, multiple
# workers are safe to run — raise this when the box has the RAM for it.
CMD ["uvicorn", "backend.app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
