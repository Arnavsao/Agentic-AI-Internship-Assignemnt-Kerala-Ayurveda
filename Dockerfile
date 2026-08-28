# ── Streamlit UI Dockerfile ──
#
# The UI is a thin HTTP client over the API, so this image carries only
# streamlit and requests. It used to install the full ML stack (torch,
# sentence-transformers, chromadb, LangChain) because the UI ran retrieval
# in-process — that was roughly 2.5 GB of image for a page of widgets.
#
# Point it at the backend with API_BASE_URL (docker-compose.yml sets it).

FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ──
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

COPY --from=builder /install /usr/local

WORKDIR /app

# Only the UI entrypoint and its Streamlit config — no data, no source tree.
COPY streamlit_app.py ./
COPY .streamlit/ ./.streamlit/

RUN chown -R appuser:appuser /app
USER appuser

ENV API_BASE_URL=http://api:8000

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
