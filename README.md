# 🌿 Kerala Ayurveda RAG System

<p align="center">
  <a href="https://www.youtube.com/watch?v=7lbHq9pGP-Y">
    <img src="https://github.com/user-attachments/assets/f80e03c7-c902-44b1-9bc6-8e6572434389" alt="AI Tools Thumbnail" width="800"/>
  </a>
</p>


> **Agentic AI Internship Assignment** — A production-ready Retrieval-Augmented Generation system with a multi-agent article generation pipeline for Kerala Ayurveda content.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.2.8-green?logo=chainlink)
![MegaLLM](https://img.shields.io/badge/MegaLLM-gemini--3--pro--preview-blueviolet)
![Gemini](https://img.shields.io/badge/Google%20Gemini-3.6%20Flash-orange?logo=google)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.4.1-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-1.54.0-red?logo=streamlit)

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
| 🔍 **RAG Q&A** | Answers user questions with structured citations sourced from the Ayurveda knowledge base |
| 🤖 **Agentic Article Generation** | 4-agent pipeline (Outline → Write → Fact-Check → Tone Edit) produces publication-ready articles |
| 📊 **Evaluation Framework** | Golden set benchmarking tracks coverage, citation accuracy, hallucination rate & tone compliance |

**Why it's production-ready:**
- Adaptive chunking (not one-size-fits-all — 400-800 chars by document type)
- Traceable citations on every answer (`doc_id` + `section_id` + relevance score)
- Automatic hallucination guardrails (fact-check score ≥ 0.7 required)
- Continuous evaluation with a golden test set
- Clean, modular architecture with a Streamlit web UI

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
│  │         Managed by GeminiKeyManager        │                       │
│  └───────────────────────────────────────────┘                       │
│                          │                                           │
│                          ▼                                           │
│  ┌───────────────────────────────────────────┐                       │
│  │              DATA LAYER                   │                       │
│  │  8 Markdown docs + 1 CSV + 1 PDF book     │                       │
│  │  → Adaptive chunking (400–800 chars)      │                       │
│  │  → HuggingFace all-MiniLM-L6-v2 embeddings│                       │
│  │  → 264 chunks persisted in ChromaDB       │                       │
│  └───────────────────┬───────────────────────┘                       │
│                      │                                               │
│                      ▼                                               │
│  ┌───────────────────────────────────────────┐                       │
│  │           PART A — RAG SYSTEM             │                       │
│  │                                           │                       │
│  │  User Query                               │                       │
│  │     → Semantic Search (ChromaDB)          │                       │
│  │     → Top-5 retrieved, Top-3 used         │                       │
│  │     → LLM generation (MegaLLM → Gemini)   │                       │
│  │     → Structured citations returned       │                       │
│  └───────────────────┬───────────────────────┘                       │
│                      │                                               │
│                      ▼                                               │
│  ┌───────────────────────────────────────────┐                       │
│  │       PART B — MULTI-AGENT WORKFLOW       │                       │
│  │                                           │                       │
│  │  Article Brief                            │                       │
│  │    → [1] Outline Agent   (temp 0.3)       │                       │
│  │    → [2] Writer Agent    (temp 0.2)       │                       │
│  │    → [3] Fact-Checker    (temp 0.0)       │                       │
│  │    → [4] Tone Editor     (temp 0.2)       │                       │
│  │    → Final Article  ✓ grounding ≥ 0.7     │                       │
│  └───────────────────┬───────────────────────┘                       │
│                      │                                               │
│                      ▼                                               │
│  ┌───────────────────────────────────────────┐                       │
│  │          EVALUATION FRAMEWORK             │                       │
│  │  • 5-example golden set                   │                       │
│  │  • Metrics: Coverage, Citations,          │                       │
│  │    Hallucination Rate, Tone               │                       │
│  │  • Results persisted to JSONL             │                       │
│  └───────────────────────────────────────────┘                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Installed Version | Purpose |
|-------|-----------|---------|---------| 
| **LLM (Primary)** | MegaLLM — `gemini-3-pro-preview` | OpenAI-compatible API | Primary LLM provider via `https://ai.megallm.io/v1` |
| **LLM (Fallback)** | Google Gemini 3.6 Flash | `gemini-3.6-flash` | Fallback with automatic key rotation; override via `GEMINI_MODEL` |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | `sentence-transformers 5.2.2` | Local semantic embeddings — no API cost |
| **Vector DB** | ChromaDB | `1.4.1` | Persistent vector store with 264 indexed chunks |
| **Framework** | LangChain | `1.2.8` | Chains, prompts, document processing |
| **LangChain Core** | `langchain-core` | `1.2.22` | Runnables, prompt templates, document models |
| **LangChain Google GenAI** | `langchain-google-genai` | `4.2.0` | Gemini LLM integration via LangChain |
| **LangChain OpenAI** | `langchain-openai` | `1.1.7` | MegaLLM integration (OpenAI-compatible) |
| **LangChain HuggingFace** | `langchain-huggingface` | `1.2.1` | HuggingFace embedding integration |
| **LangChain Community** | `langchain-community` | `0.4.1` | Community vector store integrations (Chroma) |
| **Text Splitters** | `langchain-text-splitters` | `1.1.0` | Recursive character text splitting |
| **Google GenAI SDK** | `google-generativeai` | `0.8.6` | Google Generative AI Python SDK |
| **Data Processing** | Pandas | `2.3.3` | CSV product catalog processing |
| **Numerics** | NumPy | `2.4.2` | Numerical operations for embeddings |
| **Validation** | Pydantic | `2.12.5` | Data model validation |
| **PDF Extraction** | pypdf | `6.9.2` | PDF text extraction for book documents |
| **Environment** | python-dotenv | `1.2.1` | `.env` file loading for API keys |
| **UI** | Streamlit | `1.54.0` | Interactive web demo |
| **Runtime** | Python | `3.13.7` | Core language (via Homebrew) |

---

## 📁 Project Structure

```
.
├── src/
│   ├── __init__.py               # Package exports for all public classes
│   ├── rag_system.py             # Part A — RAG: chunking, retrieval, Q&A with citations
│   ├── agent_workflow.py         # Part B — 4-agent pipeline for article generation
│   ├── evaluation.py             # Evaluation framework: golden set, metrics, tracking
│   └── key_manager.py            # LLM provider manager: MegaLLM → Gemini key rotation
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
├── chroma_db/                    # Persisted ChromaDB vector index (264 chunks)
├── evaluation_results/           # Timestamped evaluation JSON outputs
├── golden_set.json               # 5 benchmark Q&A pairs
├── metrics_history.jsonl         # Continuous metrics log
│
├── streamlit_app.py              # Web UI entrypoint
├── run.sh                        # Launcher — pins the venv interpreter
├── test_project.py               # Project validation tests
├── requirements.txt              # Python dependencies
├── .streamlit/config.toml        # Streamlit Cloud / local server config
├── .env.example                  # Example environment variables
└── .env                          # API keys (gitignored — never committed)
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Arnavsao/Agentic-AI-System-Kerala-Ayurveda.git
cd Agentic-AI-System-Kerala-Ayurveda

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
./venv/bin/python -m pip install -r requirements.txt
```

> Invoke pip as `./venv/bin/python -m pip`, not `./venv/bin/pip`. The `pip`
> shebang hardcodes an absolute path, so it breaks if the project folder is
> ever moved or renamed.

### 2. Configure API Keys

Create a `.env` file in the project root:

```env
# Primary LLM — MegaLLM (gemini-3-pro-preview via OpenAI-compatible API)
MEGA_API_KEY=sk-mega-your_megallm_key_here

# Fallback — Google Gemini (auto-rotated on quota exhaustion)
GOOGLE_API_KEY_1=your_google_api_key_1
GOOGLE_API_KEY_2=your_google_api_key_2   # optional
GOOGLE_API_KEY_3=your_google_api_key_3   # optional

# Legacy single-key fallback (used if numbered keys not found)
GOOGLE_API_KEY=your_google_api_key
```

- Get a MegaLLM key at [megallm.io](https://megallm.io)
- Get a free Gemini key at [Google AI Studio](https://aistudio.google.com/app/apikey)

The system tries MegaLLM first; if it fails for any reason it automatically falls back to Gemini with key rotation.

### 3. Run the Streamlit App

```bash
./run.sh
```

Open [http://localhost:8501](http://localhost:8501) — the knowledge base loads automatically
(~15s on a warm ChromaDB index, ~30s longer if it must be rebuilt).

`run.sh` pins the project virtualenv. Avoid a bare `streamlit run streamlit_app.py`:
that resolves `streamlit` from `PATH`, and if it picks a system Python without this
project's dependencies the app stalls on startup instead of reporting an error.
The equivalent explicit command is:

```bash
./venv/bin/python -m streamlit run streamlit_app.py
```

---

## 🔍 Part A — RAG System

**File:** `src/rag_system.py`

### How It Works

```
User Query
   ↓  Convert to embedding vector (all-MiniLM-L6-v2, 384 dimensions)
   ↓  Semantic search → retrieve 5 most relevant chunks from 264 indexed
   ↓  Select top 3 for generation (balance relevance vs. context length)
   ↓  Build prompt with [Source X: doc_id - section_id] labels
   ↓  MegaLLM generates answer (fallback: Gemini 3.6 Flash)
   ↓  Return QueryResponse(answer, citations, retrieved_chunks)
```

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

**Total indexed chunks: 264** (persisted in ChromaDB for instant startup on subsequent runs).

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

**File:** `src/agent_workflow.py`

### Agent Pipeline

```
ArticleBrief
    ↓
[Agent 1: OutlineAgent]
    • Queries RAG to verify corpus coverage
    • Generates JSON-structured outline (sections + key points)
    • Guardrail: only creates sections that can be supported by data
    ↓
[Agent 2: WriterAgent]
    • Retrieves RAG context per section (not generic article-level)
    • Writes full draft with [Source: doc_id - section_id] citations
    • Enforces Kerala Ayurveda brand voice
    ↓
[Agent 3: FactCheckerAgent]  ← Most critical step
    • Extracts all factual claims from draft
    • Scores grounding: supported_claims / total_claims
    • Auto-rejects if grounding_score < 0.7
    • Suggests RAG sources for unsupported claims
    ↓
[Agent 4: ToneEditorAgent]
    • Loads style guide from RAG corpus
    • Scores style adherence (0–1)
    • Revises content for brand voice — never removes citations
    ↓
FinalArticle (with fact_check_score, style_score, editor_notes)
```

### LLM Provider Strategy (via `key_manager.py`)

All four agents share a single `GeminiKeyManager` instance for coordinated key rotation:

```
Request → MegaLLM (gemini-3-pro-preview via ai.megallm.io/v1)
           │
           │ fails? (timeout / quota / error)
           ▼
          Gemini Key 1 (GOOGLE_API_KEY_1)
           │
           │ 429 / quota exhausted?
           ▼
          Gemini Key 2 (GOOGLE_API_KEY_2)
           │
           │ 429 / quota exhausted?
           ▼
          Gemini Key 3 (GOOGLE_API_KEY_3)
           │
           │ all exhausted?
           ▼
          RuntimeError raised
```

### Article Brief Example

```python
brief = ArticleBrief(
    topic="Ayurvedic Support for Stress and Better Sleep",
    target_audience="Busy professionals experiencing stress and sleep issues",
    key_points=[
        "How Ayurveda views stress and sleep",
        "Practical lifestyle approaches",
        "Herbs that support stress resilience",
        "Evening routines for better sleep"
    ],
    word_count_target=800,
    must_include_products=["Ashwagandha Stress Balance Tablets", "Brahmi Tailam"]
)
```

### Guardrails

| Agent | Failure Mode | Guardrail |
|-------|------------|---------|
| Outline | Topics not in corpus | Corpus coverage check before outlining |
| Writer | Hallucinated claims | Per-section RAG retrieval enforced |
| Fact-Checker | Missed unsupported claims | 0.7 grounding threshold — auto-reject below |
| Tone Editor | Removes safety disclaimers | Must preserve all citations & medical caveats |

---

## 📊 Evaluation Framework

**File:** `src/evaluation.py`

### Golden Set

5 benchmark questions covering the system's main use cases:

| ID | Query | Category |
|----|-------|---------|
| q001 | Benefits of Ashwagandha for stress? | Product |
| q002 | Contraindications for Triphala? | Product (safety-critical) |
| q003 | Can Ayurveda help with stress and sleep? | FAQ |
| q004 | What is Vata dosha? | Concept |
| q005 | How does the Stress Support Program work? | Treatment |

### Metrics Tracked

| Metric | Description | Target |
|--------|------------|--------|
| **Coverage Score** | % of expected key points in answer | ≥ 0.60 |
| **Citation Accuracy** | Expected sources actually cited | ≥ 0.50 |
| **Hallucination Rate** | % of answers with unsupported claims | ≤ 0.20 |
| **Tone Compliance** | % of answers using proper brand voice | ≥ 0.80 |

Results are saved to `evaluation_results/` with timestamps and appended to `metrics_history.jsonl` for trend tracking.

---

## 📈 Benchmark Report

The following benchmark was run against the 5-question golden set on **2026-03-23** using the full evaluation pipeline (`src/evaluation.py`). The RAG system answered each question using its standard retrieval + generation flow, and the evaluation framework scored each answer on four metrics.

### Aggregate Results

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| **Average Coverage Score** | **0.60** (60%) | ≥ 0.60 | ✅ Met |
| **Average Citation Accuracy** | **1.00** (100%) | ≥ 0.50 | ✅ Exceeded |
| **Hallucination Rate** | **0.80** (80%) | ≤ 0.20 | ❌ Above target |
| **Tone Compliance Rate** | **0.60** (60%) | ≥ 0.80 | ⚠️ Below target |

### Per-Query Breakdown

| ID | Query | Coverage | Citation Acc. | Hallucination? | Tone OK? |
|----|-------|----------|---------------|----------------|----------|
| q001 | Benefits of Ashwagandha for stress? | **0.75** | **1.00** | ⚠️ Yes | ❌ No |
| q002 | Contraindications for Triphala? | **0.75** | **1.00** | ⚠️ Yes | ❌ No |
| q003 | Can Ayurveda help with stress and sleep? | **0.50** | **1.00** | ✅ No | ✅ Yes |
| q004 | What is Vata dosha? | **1.00** | **1.00** | ⚠️ Yes | ✅ Yes |
| q005 | How does the Stress Support Program work? | **0.00** | **1.00** | ⚠️ Yes | ✅ Yes |

### Detailed Analysis

#### ✅ Strengths

- **Citation accuracy is 100%** across all 5 queries — every answer correctly cited the expected source document, demonstrating the retrieval pipeline reliably surfaces the right content.
- **q004 (Vata dosha)** achieved **perfect coverage** (1.00) — all four expected key points ("movement", "light", "dry", "tendencies") appeared in the answer.
- **q001 and q002** both scored **0.75 coverage**, successfully covering 3 out of 4 expected key points each.

#### ⚠️ Areas for Improvement

- **Hallucination rate (80%)** is significantly above the 20% target. The hallucination detector (LLM-as-judge) flagged 4 of 5 answers as containing claims not directly in the source chunks. This is partially a strictness issue — the detector is conservative, flagging reasonable inferences (e.g., "daily support for stress resilience" when the source says "traditionally used to support the body's ability to adapt to stress"). Tuning the hallucination detector prompt would reduce false positives.

- **q005 coverage (0.00)** — the answer for "How does the Stress Support Program work?" failed to mention specific treatments ("Abhyanga", "Shirodhara") that were expected. The answer was correct but overly general, focusing on the program's complementary nature rather than specific treatment steps. This suggests the retrieval stage may not have surfaced the most detail-rich chunks for this query.

- **Tone compliance (60%)** — the tone checker uses keyword-based heuristics (looking for ≥ 2 of: "traditionally used", "may help", "support", "consult"). Two answers (q001, q002) scored as non-compliant despite using appropriate cautious language, because they used slightly different phrasing. Enhancing the tone checker with semantic matching would improve this metric.

### Benchmark Environment

| Parameter | Value |
|-----------|-------|
| Date | 2026-03-23T17:34:43 |
| LLM Model | `gemini-2.5-flash` (via MegaLLM → Gemini fallback) |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` (384d, local) |
| Vector DB | ChromaDB (264 chunks indexed) |
| Knowledge Base | 10 documents (8 Markdown + 1 CSV + 1 PDF) |
| Golden Set Size | 5 queries |
| Evaluation Output | `evaluation_results/rag_eval_20260323_173443.json` |

---

## 🚀 Running the Project

Run every command from the project root. The examples use `./venv/bin/python`
explicitly so they work whether or not the virtualenv is activated — the modules
use absolute `from src.… import`, so they must be run as `python -m src.<module>`.

### Streamlit Web UI (Recommended)

```bash
./run.sh
```

### RAG System Demo (Terminal)

```bash
./venv/bin/python -m src.rag_system
```

Runs 3 example queries and prints answers with full citation details.

### Agent Workflow Demo (Terminal)

```bash
./venv/bin/python -m src.agent_workflow
```

Runs the full 4-agent pipeline on a sample stress & sleep article brief. Takes ~2-3 minutes.

### Evaluation Suite

```bash
./venv/bin/python -m src.evaluation
```

Evaluates all 5 golden examples and saves results to `evaluation_results/`.

### Rebuild the Vector Index

```bash
rm -rf chroma_db
```

The index is reused on startup when present. Delete it to force a re-embed
(~30s) after changing anything in `data/`.

### Override the Gemini Model

```bash
GEMINI_MODEL=gemini-3.5-flash-lite ./run.sh
```

The Gemini fallback model defaults to `gemini-3.6-flash`. Point `GEMINI_MODEL`
at a different model to use a separate free-tier quota pool if the default is
exhausted.

### Project Tests

```bash
./venv/bin/python test_project.py
```

Validates that all imports, the RAG system, and the evaluation framework load correctly.

---

## 🎯 Key Design Decisions

**1. Local Embeddings (HuggingFace `all-MiniLM-L6-v2`)**
Using local embeddings instead of an API means no extra cost, no rate limits, and faster indexing. The 384-dimension model is well-suited for semantic medical Q&A.

**2. Adaptive Chunking by Document Type**
FAQ documents need small chunks (400 chars) to keep Q&A pairs together. Guides and PDFs need larger chunks (800 chars) to maintain conceptual context. A single chunk size would degrade retrieval quality.

**3. Retrieve 5, Use Top 3**
Retrieving 5 chunks casts a wide semantic net while passing only 3 to the LLM keeps the prompt focused and avoids context dilution. All 5 are returned in the response for transparency.

**4. Per-Section RAG in Writer Agent**
The Writer Agent queries RAG once per outline section rather than once for the whole article. This ensures each section gets the most relevant sources, not a one-size-fits-all context block.

**5. 0.7 Grounding Threshold**
The Fact-Checker auto-rejects articles below 70% grounding. Medical content has zero tolerance for hallucination — this threshold triggers a revision loop (up to 2 iterations) before escalating to an editor note.

**6. MegaLLM-First Provider Strategy**
The `key_manager.py` tries MegaLLM (OpenAI-compatible API at `ai.megallm.io/v1`) first for every call, then falls back to Gemini with automatic key rotation. This maximizes availability — if MegaLLM is down or rate-limited, the system transparently switches to Gemini without any user intervention.

**7. Persistent ChromaDB Index**
On first run, all documents are embedded and indexed (~30s). On subsequent startups, the persisted ChromaDB collection is reused instantly, avoiding the re-embedding cost that would otherwise cause Streamlit reloads.

---

## 📦 Assignment Deliverables

| Requirement | Implementation |
|---|---|
| Part A: RAG with chunking | `src/rag_system.py` — adaptive chunking, ChromaDB (264 chunks), structured citations |
| Part A: Q&A with citations | `answer_user_query()` returns `QueryResponse` with doc_id + section_id |
| Part B: Multi-agent pipeline | `src/agent_workflow.py` — 4-agent orchestrator with shared key manager |
| Part B: Fact-checking guardrail | `FactCheckerAgent` with 0.7 grounding threshold |
| Evaluation framework | `src/evaluation.py` — golden set (5 queries), 4 metrics, JSONL history |
| Benchmark report | See [Benchmark Report](#-benchmark-report) — full results with per-query analysis |
| Web UI | `streamlit_app.py` — interactive demo with citations |

---

<div align="center">

**Kerala Ayurveda RAG System** · Built for the Agentic AI Internship Assignment

*Stack: Python 3.13 · MegaLLM (gemini-3-pro-preview) · Google Gemini 3.6 Flash · LangChain 1.2.8 · ChromaDB 1.4.1 · HuggingFace · Streamlit 1.54.0*

</div>
