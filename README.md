# 🌿 Kerala Ayurveda RAG System

<p align="center">
  <a href="https://www.youtube.com/watch?v=7lbHq9pGP-Y">
    <img src="https://github.com/user-attachments/assets/f80e03c7-c902-44b1-9bc6-8e6572434389" alt="AI Tools Thumbnail" width="800"/>
  </a>
</p>


> **Agentic AI Internship Assignment** — A production-ready Retrieval-Augmented Generation system with a multi-agent article generation pipeline for Kerala Ayurveda content.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi)
![LangGraph](https://img.shields.io/badge/LangGraph-agents-green?logo=chainlink)
![Qdrant](https://img.shields.io/badge/Qdrant-hybrid%20search-red)
![Gemini](https://img.shields.io/badge/Google%20Gemini-3.6%20Flash-orange?logo=google)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [System Architecture](#-system-architecture)
3. [Tech Stack](#-tech-stack)
4. [Project Structure](#-project-structure)
5. [Quick Start](#-quick-start)
6. [Part A — RAG System](#-part-a--rag-system)
7. [Part B — Multi-Agent Workflow](#-part-b--multi-agent-workflow)
8. [Evaluation Framework](#-evaluation-framework)
9. [Benchmark Report](#-benchmark-report)
10. [Running the Project](#-running-the-project)
11. [Key Design Decisions](#-key-design-decisions)

---

## 🎯 Overview

This project builds an end-to-end **AI content pipeline** for Kerala Ayurveda:

| Capability | Description |
|---|---|
| 🔍 **RAG Q&A** | Hybrid retrieval (dense + sparse, RRF-fused, cross-encoder reranked) with structured citations on every answer |
| 🤖 **Agentic Article Generation** | LangGraph pipeline — outline → parallel section writes → fact-check → revise → tone edit — run as a persisted async job |
| 📊 **Evaluation Framework** | 18-question golden set with semantic coverage, citation accuracy, LLM-judge grounding, and tone compliance |

**What makes it production-shaped:**
- **Hybrid retrieval** — dense embeddings for concepts, BM25 sparse vectors for exact terms like `KA-P001` and `Shirodhara`, fused server-side by Qdrant
- **Incremental indexing** — content-hashed; unchanged files are skipped, so re-indexing costs a few file hashes rather than a full re-embed
- **Fail-loud index contract** — the app refuses to start against a collection built by a different embedding model instead of silently returning nonsense
- **A real revision loop** — a rejected draft is rewritten before being re-checked, not simply re-checked
- **Traceable citations** on every answer (`doc_id` + `section_id` + score)
- **Grounding guardrail** — fact-check ≥ 0.7, and an unparseable fact-check fails closed
- **Free local models** for embedding, sparse retrieval, and reranking — only generation calls an API

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    KERALA AYURVEDA RAG SYSTEM                        │
│                                                                      │
│  ┌───────────────────────────────────────────┐                       │
│  │              LLM PROVIDER LAYER           │                       │
│  │                                           │                       │
│  │   ┌─────────────────────────────────┐     │                       │
│  │   │  PRIMARY: MegaLLM               │     │                       │
│  │   │  model: gemini-3-pro-preview     │     │                       │
│  │   │  base:  https://ai.megallm.io/v1│     │                       │
│  │   └────────────────┬────────────────┘     │                       │
│  │                    │ fails?                │                       │
│  │                    ▼                       │                       │
│  │   ┌─────────────────────────────────┐     │                       │
│  │   │  FALLBACK: Google Gemini        │     │                       │
│  │   │  model: gemini-3.6-flash        │     │                       │
│  │   │  keys: auto-rotated (3 keys)    │     │                       │
│  │   └─────────────────────────────────┘     │                       │
│  │    Managed by the LLM gateway (llm.py)    │                       │
│  └───────────────────────────────────────────┘                       │
│                          │                                           │
│                          ▼                                           │
│  ┌───────────────────────────────────────────┐                       │
│  │              DATA LAYER                   │                       │
│  │  8 Markdown docs + 1 CSV + 1 PDF book     │                       │
│  │  → Adaptive parent/child chunking         │                       │
│  │  → bge-base-en-v1.5 dense (768d, local)   │                       │
│  │  → Qdrant/bm25 sparse (local)             │                       │
│  │  → 130 chunks in Qdrant                   │                       │
│  │  → Incremental: unchanged files skipped   │                       │
│  └───────────────────┬───────────────────────┘                       │
│                      │                                               │
│                      ▼                                               │
│  ┌───────────────────────────────────────────┐                       │
│  │           PART A — RAG SYSTEM             │                       │
│  │                                           │                       │
│  │  User Query                               │                       │
│  │     → Dense + sparse retrieval (Qdrant)   │                       │
│  │     → RRF fusion, server-side             │                       │
│  │     → Cross-encoder rerank → top-5        │                       │
│  │     → Top-3 (parent chunks) into prompt   │                       │
│  │     → LLM generation (MegaLLM → Gemini)   │                       │
│  │     → Structured citations returned       │                       │
│  └───────────────────┬───────────────────────┘                       │
│                      │                                               │
│                      ▼                                               │
│  ┌───────────────────────────────────────────┐                       │
│  │   PART B — LANGGRAPH AGENT PIPELINE       │                       │
│  │                                           │                       │
│  │  Article Brief                            │                       │
│  │    → [1] Outline        (temp 0.3)        │                       │
│  │    → [2] Write sections (temp 0.2) ║ par. │                       │
│  │    → [3] Fact-Check     (temp 0.0)        │                       │
│  │    → [4] Revise         (temp 0.2) ↺ loop │                       │
│  │    → [5] Tone Edit      (temp 0.2)        │                       │
│  │    → Final Article  ✓ grounding ≥ 0.7     │                       │
│  │  Runs as a persisted async job            │                       │
│  └───────────────────┬───────────────────────┘                       │
│                      │                                               │
│                      ▼                                               │
│  ┌───────────────────────────────────────────┐                       │
│  │          EVALUATION FRAMEWORK             │                       │
│  │  • 18-question golden set (+ negatives)   │                       │
│  │  • Semantic coverage (embedding cosine)   │                       │
│  │  • Citation accuracy, LLM-judge grounding │                       │
│  │  • Tone check (word-boundary, negation)   │                       │
│  └───────────────────────────────────────────┘                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Service topology

```
  Streamlit UI  ──HTTP──►  FastAPI backend  ──►  Qdrant  (vectors)
  (thin client)                    │            └─►  Redis   (cache)
                                   └─────────────►  SQLite  (jobs, docs)
```

The UI holds no ML dependencies; everything runs behind the API.

---

## 🛠️ Tech Stack

Everything in the retrieval path is local and free — no per-query embedding or reranking cost. Only generation calls an API.

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **API** | FastAPI + Uvicorn | Backend serving retrieval, agents, and jobs |
| **UI** | Streamlit | Thin HTTP client — no ML dependencies |
| **Vector DB** | Qdrant | Dense + sparse vectors, server-side RRF fusion |
| **Dense embeddings** | `BAAI/bge-base-en-v1.5` (768d) | Local semantic embeddings — free |
| **Sparse embeddings** | `Qdrant/bm25` via FastEmbed | Local keyword matching — free |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local cross-encoder — free |
| **Agents** | LangGraph | Article graph with parallel writes + revision loop |
| **LLM (Primary)** | MegaLLM — `gemini-3-pro-preview` | Via `https://ai.megallm.io/v1` |
| **LLM (Fallback)** | Google Gemini — `gemini-3.6-flash` | Automatic key rotation on quota exhaustion |
| **Cache** | Redis (optional) + in-process LRU | Response cache, keyed by index version |
| **Metadata** | SQLAlchemy + SQLite | Article jobs, documents, query logs |
| **Documents** | pypdf, pandas | PDF and CSV parsing |
| **Runtime** | Python 3.11+ | |

**Why these models.** `bge-base-en-v1.5` scores substantially better than `all-MiniLM-L6-v2` on retrieval benchmarks (MTEB 53.3 vs 41.8) and has a 512-token window, at 768 dimensions and ~220 MB. `BGE-M3` would be stronger still but is 2.2 GB and roughly 8× slower on CPU, and its native sparse output is redundant next to Qdrant's BM25. The reranker stays on the small MiniLM cross-encoder because it scores 10 pairs in 50–100 ms on CPU; `bge-reranker-base` takes 0.5–1.5 s for the same batch, which is a poor trade on a hot path over a corpus this size. Override either with `EMBEDDING_MODEL` / `RERANKER_MODEL`.

---

## 📁 Project Structure

```
.
├── backend/app/
│   ├── main.py                   # FastAPI app, lifespan, middleware
│   ├── core/                     # config (pydantic-settings), logging, database
│   ├── models/schemas.py         # SQLAlchemy: documents, chunks, query logs, jobs
│   ├── api/routes/               # health, query, documents, articles
│   └── services/
│       ├── llm.py                # LLM gateway: failover + key rotation
│       ├── cache.py              # Two-layer response cache (LRU + Redis)
│       ├── evaluation.py         # Metric scorers
│       ├── agents/               # LangGraph article pipeline
│       │   ├── models.py         #   ArticleState and the inter-node contracts
│       │   ├── prompts.py        #   Agent prompts and guardrails
│       │   ├── nodes.py          #   Node implementations
│       │   └── graph.py          #   Graph wiring, fan-out, revision loop
│       ├── ingestion/service.py  # Incremental, hash-diffed indexing
│       └── rag/
│           ├── chunker.py        #   Adaptive parent/child chunking
│           ├── embeddings.py     #   Dense (bge) + sparse (bm25) encoders
│           ├── vectorstore.py    #   Qdrant adapter
│           ├── retriever.py      #   Hybrid retrieval + reranking
│           ├── generator.py      #   Answer generation with citations
│           └── pipeline.py       #   Orchestration
│
├── scripts/
│   ├── ingest.py                 # CLI: index the knowledge base
│   └── evaluate.py               # CLI: run the golden-set benchmark
│
├── eval/golden_set.json          # 18 benchmark questions incl. negative cases
├── streamlit_app.py              # UI — thin HTTP client over the API
│
├── src/                          # LEGACY single-process stack (superseded)
│
├── data/                         # Kerala Ayurveda knowledge base (10 documents)
│   ├── ayurveda_foundations.md
│   ├── content_style_and_tone_guide.md
│   ├── dosha_guide_vata_pitta_kapha.md
│   ├── faq_general_ayurveda_patients.md
│   ├── product_ashwagandha_tablets_internal.md
│   ├── product_brahmi_tailam_internal.md
│   ├── product_triphala_capsules_internal.md
│   ├── treatment_stress_support_program.md
│   ├── products_catalog.csv      # 8-product structured catalog
│   └── ayurveda_chapter_1_and_2.pdf  # Astanga Hridaya (Vagbhat) — 24-page PDF book
│
├── tests/
│   ├── unit/                     # chunker, vectorstore, ingestion, agents,
│   │                             #   llm gateway, evaluation scorers
│   └── integration/              # retrieval against real models
│
├── evaluation_results/           # Timestamped evaluation JSON outputs
├── docker-compose.yml            # Production: qdrant + redis + api + streamlit
├── docker-compose.dev.yml        # Development: qdrant + redis + api (hot reload)
├── Dockerfile                    # Streamlit UI image (thin)
├── backend/Dockerfile            # API image (models baked in)
├── requirements.txt              # UI dependencies only
├── backend/requirements.txt      # Backend dependencies
├── .env.example                  # Documented environment variables
└── .env                          # API keys (gitignored — never committed)
```

---

## ⚡ Quick Start

### Option A — Docker (recommended)

Brings up Qdrant, Redis, the API, and the UI together.

```bash
git clone https://github.com/Arnavsao/Agentic-AI-System-Kerala-Ayurveda.git
cd Agentic-AI-System-Kerala-Ayurveda

cp .env.example .env      # add at least one LLM key
docker compose up --build
```

- UI: [http://localhost:8501](http://localhost:8501)
- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Qdrant dashboard: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

The index builds on first API start (about 10 s for this corpus) and persists
in a Docker volume. Later starts skip files whose contents haven't changed.

### Option B — Local development

```bash
python3 -m venv venv
source venv/bin/activate
./venv/bin/python -m pip install -r backend/requirements.txt
./venv/bin/python -m pip install -r requirements.txt

# Vector store + cache
docker compose -f docker-compose.dev.yml up -d qdrant redis

# Index the knowledge base
./venv/bin/python -m scripts.ingest

# API, then UI in a second shell
./venv/bin/python -m uvicorn backend.app.main:app --reload
./venv/bin/python -m streamlit run streamlit_app.py
```

> Invoke pip as `./venv/bin/python -m pip`, not `./venv/bin/pip` — the `pip`
> shebang hardcodes an absolute path and breaks if the folder is moved.

**No Docker at all?** Point Qdrant at a local directory instead of a server:

```bash
QDRANT_URL=./qdrant_local ./venv/bin/python -m scripts.ingest
```

Local file mode implements the same Query API but locks the directory to one
process, so use a server for anything beyond single-process development.

### Configure API keys

Only generation needs an API key — embeddings and reranking run locally.

```env
# Primary (optional)
MEGA_API_KEY=sk-mega-your_megallm_key_here

# Fallback, rotated automatically on quota exhaustion
GOOGLE_API_KEY_1=your_google_api_key_1
GOOGLE_API_KEY_2=
GOOGLE_API_KEY_3=
```

Free Gemini keys: [Google AI Studio](https://aistudio.google.com/app/apikey) ·
MegaLLM keys: [megallm.io](https://megallm.io)

Keys are tried in order and **the first empty slot ends the list**, so don't
leave a gap. See `.env.example` for every setting.

### Managing the index

```bash
python -m scripts.ingest              # incremental — skips unchanged files
python -m scripts.ingest --rebuild    # full re-embed (after a model change)
python -m scripts.ingest --status     # what's indexed, changes nothing
```

Changing `EMBEDDING_MODEL` invalidates the index. The app refuses to start
against a mismatched collection rather than querying one vector space with
another's vectors, and the error names the fix.

---

## 🔍 Part A — RAG System

**Files:** `backend/app/services/rag/` · **Endpoint:** `POST /api/v1/query`

### How It Works

```
User Query
   ↓  Embed as a dense vector (bge-base-en-v1.5, 768d, with query prefix)
   ↓  Embed as a sparse vector (Qdrant/bm25 term weights)
   ↓  Qdrant retrieves top-10 by each and fuses them with RRF (server-side)
   ↓  Cross-encoder reranks the fused candidates → top-5
   ↓  Take top-3, expanding each to its parent chunk for context
   ↓  Build prompt with [Source X: doc_id - section_id] labels
   ↓  MegaLLM generates the answer (fallback: Gemini, with key rotation)
   ↓  Return QueryResponse(answer, citations, retrieved_chunks)
```

### Why hybrid search

Dense retrieval handles conceptual questions well but misses the cases that
matter most in a product and safety corpus:

| Query | Dense alone | With sparse |
|---|---|---|
| `KA-P001` | Embeddings encode identifiers poorly | Exact term match |
| `Shirodhara` | Rare term, likely out-of-distribution | Exact term match |
| "How does Ayurveda view stress?" | Strong | Strong |

Both branches run inside Qdrant and are merged with Reciprocal Rank Fusion:

```
score(doc) = Σ 1 / (k + rank_i(doc))        k = 60
```

RRF is rank-based, so scores from the two systems never need to be on a
comparable scale, and it has no weights to tune. A document ranked well by
both branches scores far higher than one ranked well by only one.

The cross-encoder then rescores the survivors. The embedding model is a
bi-encoder — it encodes query and document separately, which is fast but blind
to their interaction. A cross-encoder reads both together and can judge
"does this passage about Ashwagandha actually answer the pregnancy question?"
It is far slower per pair, so it only ever sees ~10 candidates.

### Adaptive Chunking Strategy

Different document types are chunked differently — a single size doesn't fit all:

| Document Type | Chunk Size | Overlap | Reason |
|---|---|---|---|
| FAQ (`faq_*.md`) | 400 chars | 100 | Keep Q&A pairs together |
| Product (`product_*.md`) | 500 chars | 100 | Preserve product sections |
| Guide (`*guide*.md`, `dosha*.md`) | 800 chars | 100 | Conceptual content needs context |
| PDF (`.pdf`) | 800 chars | 100 | Long-form content needs larger context |
| Default | 600 chars | 100 | General articles |

Splitters try to break at `## headers → ### headers → paragraphs → sentences` before falling back to characters, preserving semantic boundaries.

### Data Sources

| Document | Type | Description |
|---|---|---|
| `ayurveda_foundations.md` | Guide | Core Ayurvedic philosophy and positioning |
| `dosha_guide_vata_pitta_kapha.md` | Guide | Vata, Pitta, Kapha dosha characteristics |
| `content_style_and_tone_guide.md` | Guide | Brand voice and content guidelines |
| `faq_general_ayurveda_patients.md` | FAQ | Common patient questions and answers |
| `product_ashwagandha_tablets_internal.md` | Product | Ashwagandha product dossier |
| `product_brahmi_tailam_internal.md` | Product | Brahmi Tailam product dossier |
| `product_triphala_capsules_internal.md` | Product | Triphala product dossier |
| `treatment_stress_support_program.md` | Treatment | Stress Support Program details |
| `products_catalog.csv` | Catalog | 8-product structured catalog with metadata |
| `ayurveda_chapter_1_and_2.pdf` | PDF Book | Astanga Hridaya (Vagbhat) — 24 pages |

**Total indexed chunks: 130** (persisted in Qdrant; unchanged files are skipped on re-index).

### Citation Structure

Every answer includes structured citations:

```python
Citation(
    doc_id="product_ashwagandha_tablets_internal",
    section_id="Traditional Positioning",
    content_snippet="In Ayurveda, Ashwagandha is traditionally...",
    relevance_score=0.534   # cosine similarity score
)
```

### Example Output

```
Query: "What are the benefits of Ashwagandha?"

Answer: "Ashwagandha is traditionally used to support the body's ability
to adapt to stress, promote calmness and emotional balance, support strength
and stamina, and help maintain restful sleep [Source 1, Source 2, Source 3].
Always consult a qualified Ayurvedic practitioner before starting any new herb."

Sources:
  [1] product_ashwagandha_tablets_internal — Traditional Positioning (53.4%)
  [2] product_ashwagandha_tablets_internal — Contraindications & Safety (51.2%)
  [3] products_catalog — Ashwagandha Stress Balance Tablets (49.8%)
```

---

## 🤖 Part B — Multi-Agent Workflow

**Files:** `backend/app/services/agents/` · **Endpoint:** `POST /api/v1/articles/generate`

Built on **LangGraph** — the agents are nodes in a state machine rather than a
straight-line sequence of Python calls.

### Agent Graph

```
   outline  ── plans against what the corpus can actually support
      │
      ├─ Send ─→ write_section  ┐
      ├─ Send ─→ write_section  ├─ run concurrently, each with its own retrieval
      └─ Send ─→ write_section  ┘
                    │
                    ▼
             assemble_draft
                    │
                    ▼
               fact_check ◄────────────┐
                    │                  │
          grounded? │                  │ revised draft
             ┌──────┴───────┐          │
             │ no, budget   ├──────────┘
             │ remaining    │   revise
             │              │
             │ yes / spent  │
             └──────┬───────┘
                    ▼
                tone_edit
                    │
                   END
```

Runs as a persisted async job: `POST` returns a job ID immediately and the
client polls `GET /api/v1/articles/{job_id}`. Progress is written to the
database as each node completes, so a job survives a browser refresh — and a
job interrupted by a restart is marked failed rather than polling forever.

### What each node does

| Node | Temp | Behaviour |
|---|---|---|
| **outline** | 0.3 | Retrieves corpus evidence, plans only sections the data supports |
| **write_section** | 0.2 | One per section, **in parallel**, each retrieving its own evidence |
| **assemble_draft** | — | Stitches sections in order, extracts citations |
| **fact_check** | 0.0 | Extracts claims, scores grounding against retrieved evidence |
| **revise** | 0.2 | Rewrites flagged claims — adds citations, softens, or removes |
| **tone_edit** | 0.2 | Aligns voice with the style guide without touching facts |

### Design notes

**Agents retrieve raw chunks.** Every agent previously reached the corpus
through `answer_user_query()` — a retrieval *plus a full LLM generation* — so
each lookup cost a round trip and returned a paraphrase. The writer then
grounded its citations on a summary of a summary. Nodes now call the retriever
directly: lookups are vector queries in milliseconds, and the text an agent
cites is the text actually in the knowledge base.

**The revision loop is real.** The previous retry re-ran the fact-checker on
the *same unmodified draft*, so a rejected article could only be rejected again
— the source even carried a comment reading "In production, would have revision
agent here." `revise` is that agent.

**Cost.** Roughly 5–7 sequential LLM steps (sections run in parallel) rather
than 11–15 strictly sequential ones.

### LLM Provider Strategy

```
Request → MegaLLM (gemini-3-pro-preview)
           │ fails? (timeout / quota / error)
           ▼
          Gemini key 1 → key 2 → key 3
           │ all exhausted?
           ▼
          LLMProviderError
```

Concurrent calls are capped (`LLM_MAX_CONCURRENCY`, default 3) so parallel
section writes can't exhaust the Gemini free tier's 15 requests/minute in one
burst. Rotation is lock-guarded and retry backoff is non-blocking on the async
path.

## 📊 Evaluation Framework

**Files:** `backend/app/services/evaluation.py`, `scripts/evaluate.py`, `eval/golden_set.json`

```bash
python -m scripts.evaluate                  # full golden set
python -m scripts.evaluate --no-judge       # local metrics only, no API calls
python -m scripts.evaluate --ids q005 q013  # specific questions
```

### Golden Set — 18 questions

Grown from 5. Every corpus document is now covered, plus multi-document
synthesis and negative cases.

| Category | Count | Purpose |
|---|---|---|
| Product | 4 | Product dossiers, including safety-critical questions |
| Concept | 4 | Doshas and foundations |
| FAQ / style | 2 | Patient FAQ and the brand style guide |
| Treatment | 1 | Stress Support Program |
| Catalog | 2 | CSV catalog, including exact-ID lookup |
| Multi-document | 2 | Answers requiring synthesis across sources |
| Classical text | 1 | The 24-page Astanga Hridaya PDF |
| **Negative** | **2** | **Must refuse — out of scope or absent from the corpus** |

Negative cases matter most here: a system that fabricates a price or claims a
cure fails in a way that no coverage score would catch.

### Metrics

| Metric | How it is computed | Target |
|---|---|---|
| **Coverage** | Embedding cosine similarity between the answer's best-matching sentence and each expected point | ≥ 0.60 |
| **Citation accuracy** | Fraction of expected source documents actually cited | ≥ 0.50 |
| **Faithfulness / hallucination** | LLM judge comparing the answer against retrieved sources | ≤ 0.20 hallucination |
| **Tone compliance** | Word-boundary red-flag scan + hedged-phrasing check | ≥ 0.80 |
| **Negative refusal** | Did the system decline to answer out-of-scope questions? | 1.00 |

### Three scorers were rewritten, because they were measuring themselves

The earlier benchmark numbers were partly artifacts of the metric code:

**Coverage was a substring test.** A correct answer phrased differently scored
zero. This is precisely why q005 reported **0.00 coverage** while the report's
own analysis conceded the answer was "correct but overly general." It is now
embedding similarity, so paraphrase is credited the way a human grader would —
with a literal match kept as a floor.

**The hallucination detector was `"YES" in response.upper()`.** A judge
replying *"Yes, this is well grounded"* was recorded as a hallucination,
because the substring appears regardless of what follows. The judge now
returns a structured verdict, and an unparseable reply counts as *unknown*
rather than silently passing or failing.

**The tone checker matched red-flag words as substrings.** `"cure"` matched
`"procedure"` — and, worse, matched `"does not cure"`, the exact hedged
phrasing the style guide instructs writers to use. It now uses word boundaries
with negation-aware context.

Citation accuracy was sound and is unchanged.

---

## 📈 Benchmark Report

Run on **2026-08-29** against the 18-question golden set via
`python -m scripts.evaluate`, with LLM-judge grounding enabled.

### Aggregate Results

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| **Average Coverage** | **0.94** (94%) | ≥ 0.60 | ✅ Exceeded |
| **Average Citation Accuracy** | **0.92** (92%) | ≥ 0.50 | ✅ Exceeded |
| **Hallucination Rate** | **0.50** (50%) | ≤ 0.20 | ❌ Above target |
| **Tone Compliance** | **1.00** (100%) | ≥ 0.80 | ✅ Exceeded |
| **Negative Refusal Rate** | **1.00** (100%) | 1.00 | ✅ Met |
| Average faithfulness | 0.81 | — | — |
| Average latency | 17.3 s | — | dominated by generation |

**9 of 18 questions passed all four gates.**

### Comparison with the previous benchmark

Not directly comparable — the golden set grew from 5 questions to 18, and
three scorers were rewritten because they were measuring their own
implementation rather than the system. Both columns are shown for orientation,
not as a controlled A/B.

| Metric | Before (5q, old scorers) | After (18q, fixed scorers) |
|---|---|---|
| Coverage | 0.60 | **0.94** |
| Citation accuracy | 1.00 | 0.92 |
| Hallucination rate | 0.80 | **0.50** |
| Tone compliance | 0.60 | **1.00** |
| Negative cases | none tested | 2/2 refused |

### The q005 regression is fixed

"How does the Stress Support Program work?" previously scored **0.00 coverage**
— the answer never mentioned Abhyanga or Shirodhara. It now scores **1.00**,
and both terms plus the consultation step appear in the retrieved context.

Two independent causes, both addressed:

1. **Retrieval.** "Shirodhara" is a rare term that dense embeddings handle
   poorly. Sparse BM25 retrieval now matches it exactly.
2. **Scoring.** The old substring metric would have scored a correct paraphrase
   zero regardless of retrieval quality.

### Retrieval is strong; generation is the weak link

Coverage (0.94) and citation accuracy (0.92) say the right chunks are being
found and cited. The hallucination rate says the model then embellishes them.

The judge's reasons are specific and checkable — for q016 it flagged "the four
purposes of life (Dharma, Artha…)", which is accurate Ayurvedic knowledge that
simply is not in the retrieved sources. The failure mode is the model drawing
on training knowledge to produce a *richer* answer than the corpus supports.

**This is an open issue, not a solved one.** Tightening the generation prompt
to forbid outside knowledge was tried: it improved three of the six worst
cases and worsened three others — net neutral on faithfulness, though coverage
rose to 100% on that subset. The clearer prompt was kept, but it did not move
the metric.

Approaches worth trying next, roughly in order of expected value:

- **A grounding pass over the draft answer** — the same shape as the article
  pipeline's `revise` node, which does have a real feedback loop, applied to
  Q&A. This is the most likely fix, since it corrects rather than instructs.
- **Sentence-level citation enforcement** — programmatically strip sentences
  that carry no `[Source X]` marker.
- **Judge calibration** — the current judge counts any unsupported clause
  against the answer. Some flagged items are benign connective phrasing; a
  claim-weighted rubric would separate embellishment from framing.

A caveat on measurement: these are single-run numbers, and both generation and
judging vary between runs. Differences of ±0.1 on a single question should not
be read as signal.

### Benchmark Environment

| Parameter | Value |
|-----------|-------|
| Date | 2026-08-29 |
| Vector store | Qdrant, 130 chunks, collection `ayurveda_rag_v2` |
| Dense embeddings | `BAAI/bge-base-en-v1.5` (768d, local) |
| Sparse embeddings | `Qdrant/bm25` (local) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| Generation | MegaLLM `gemini-3-pro-preview` → Gemini `gemini-3.6-flash` |
| Golden set | 18 questions incl. 2 negative cases |
| Retrieval funnel | top-10 fused → rerank 5 → 3 in prompt |

### A note on the "264 chunks" in earlier documentation

The previous index reported 264 chunks. It actually held **130 unique chunks,
each stored twice** — every document's count in the old store is exactly double
the current one (PDF 150→75, FAQ 20→10, each catalog product 2→1). Indexing
appended with random IDs, so re-running it duplicated the entire corpus rather
than updating it. Point IDs are now derived from content, making re-ingestion
idempotent.

---

## 🚀 Running the Project

Run everything from the project root.

### Full stack

```bash
docker compose up --build
```

UI at :8501, API docs at :8000/docs, Qdrant dashboard at :6333/dashboard.

### Backend only (development)

```bash
docker compose -f docker-compose.dev.yml up -d qdrant redis
./venv/bin/python -m uvicorn backend.app.main:app --reload
```

### Query the API directly

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H 'content-type: application/json' \
  -d '{"query": "Is Ashwagandha safe during pregnancy?"}'
```

### Generate an article

```bash
JOB=$(curl -s -X POST http://localhost:8000/api/v1/articles/generate \
  -H 'content-type: application/json' \
  -d '{"topic": "Ayurvedic Support for Stress and Better Sleep",
       "target_audience": "Busy professionals",
       "key_points": ["How Ayurveda views stress", "Herbs that support resilience"],
       "word_count_target": 800}' | jq -r .job_id)

curl -s http://localhost:8000/api/v1/articles/$JOB | jq '{status, current_step}'
```

The job runs server-side and its progress is persisted, so polling can stop
and resume freely.

### Index management

```bash
python -m scripts.ingest              # incremental
python -m scripts.ingest --rebuild    # full re-embed
python -m scripts.ingest --status     # inspect, change nothing
```

### Evaluation

```bash
python -m scripts.evaluate             # full golden set, with LLM judge
python -m scripts.evaluate --no-judge  # local metrics only, no API calls
```

### Tests

```bash
pytest tests/ -m "not integration"   # 138 unit tests, no models loaded (~13s)
pytest tests/ -m integration         # 13 tests against real models (~1min)
pytest tests/                        # everything
```

| Suite | Covers |
|---|---|
| `test_chunker.py` | Adaptive chunking, content hashing |
| `test_vectorstore.py` | Qdrant adapter, index compatibility guards |
| `test_ingestion.py` | Incremental diffing, idempotent upserts |
| `test_agent_graph.py` | Graph routing, revision loop, failure handling |
| `test_llm_provider.py` | Parameter pass-through, failover, rotation, concurrency |
| `test_evaluation.py` | Metric scorers and their former bugs |
| `integration/test_retrieval.py` | End-to-end retrieval with real models |

---

## 🎯 Key Design Decisions

**1. Qdrant with dense + sparse vectors**
Keyword search used to be a Python BM25 index rebuilt at every startup by
scanning the whole collection through a private API. Qdrant stores a sparse
vector beside the dense one and fuses both with RRF internally, so startup
touches nothing and retrieval is one query.

**2. Local models throughout the retrieval path**
`bge-base-en-v1.5` for dense, `Qdrant/bm25` for sparse, and a MiniLM
cross-encoder for reranking — all free, no rate limits, no per-query cost.
Only generation calls an API.

**3. Content-addressed point IDs**
Point IDs are `uuid5(doc_id + content_hash)`, so re-ingesting unchanged text
upserts in place. This is what makes incremental indexing possible, and it is
why the corpus is no longer stored twice.

**4. A vector-space contract, enforced at startup**
The collection records which embedding model built it. Booting with a
different model raises immediately instead of querying one vector space with
another's vectors — the previous code reused any non-empty collection, so a
768-d model could silently query a 384-d index.

**5. Parent-child chunks, parents inlined**
Retrieval matches small precise chunks; the LLM receives the larger parent for
context. Parents live in the child's payload rather than as their own points —
they are only ever looked up by ID, never searched.

**6. Agents retrieve chunks, not generated summaries**
Every agent previously reached the corpus through a full RAG call, so each
lookup cost an LLM round trip and returned a paraphrase. Nodes now query the
retriever directly, which is both faster and better grounded.

**7. A revision node on the fact-check loop**
The retry previously re-checked the same unmodified draft, which could only
ever produce the same verdict. `revise` rewrites the flagged claims first.

**8. Failing closed on medical content**
An unparseable fact-check response used to default to 0.75 and "grounded",
silently clearing the safety gate. It now scores zero and flags for review.

**9. Jobs in the database, not in a dict**
Article jobs persist, survive restarts, and are swept to `failed` if the
process died mid-run — rather than leaving a client polling forever.

---

## 📦 Assignment Deliverables

| Requirement | Implementation |
|---|---|
| Part A: RAG with chunking | `backend/app/services/rag/` — adaptive parent/child chunking, hybrid retrieval, reranking |
| Part A: Q&A with citations | `POST /api/v1/query` → `doc_id` + `section_id` + score per citation |
| Part B: Multi-agent pipeline | `backend/app/services/agents/` — LangGraph graph with parallel section writes |
| Part B: Fact-checking guardrail | `fact_check` node, 0.7 grounding threshold, real revision loop |
| Evaluation framework | `scripts/evaluate.py`, 18-question golden set, four metrics |
| Benchmark report | [Benchmark Report](#-benchmark-report) — full results and open issues |
| Web UI | `streamlit_app.py` — thin client over the API |
| Tests | 151 tests across unit and integration suites |

---

<div align="center">

**Kerala Ayurveda RAG System** · Built for the Agentic AI Internship Assignment

*Stack: Python · FastAPI · LangGraph · Qdrant · bge-base-en-v1.5 · Gemini · Streamlit*

</div>
