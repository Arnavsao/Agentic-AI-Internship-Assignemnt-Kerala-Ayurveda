# 🌿 Kerala Ayurveda RAG System

<p align="center">
  <a href="https://www.youtube.com/watch?v=7lbHq9pGP-Y">
    <img src="https://github.com/user-attachments/assets/f80e03c7-c902-44b1-9bc6-8e6572434389" alt="AI Tools Thumbnail" width="800"/>
  </a>
</p>


> **Agentic AI Internship Assignment** — A production-ready Retrieval-Augmented Generation system with a multi-agent article generation pipeline for Kerala Ayurveda content.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-green?logo=chainlink)
![MegaLLM](https://img.shields.io/badge/MegaLLM-gemini--3--pro--preview-blueviolet)
![Gemini](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-orange?logo=google)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-purple)
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
9. [Running the Project](#-running-the-project)
10. [Key Design Decisions](#-key-design-decisions)

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
│  │   │  model: gemini-2.5-flash        │     │                       │
│  │   │  keys: auto-rotated (3 keys)    │     │                       │
│  │   └─────────────────────────────────┘     │                       │
│  │         Managed by GeminiKeyManager        │                       │
│  └───────────────────────────────────────────┘                       │
│                          │                                           │
│                          ▼                                           │
│  ┌───────────────────────────────────────────┐                       │
│  │              DATA LAYER                   │                       │
│  │  8 Markdown docs + 1 CSV catalog          │                       │
│  │  → Adaptive chunking (400–800 chars)      │                       │
│  │  → HuggingFace all-MiniLM-L6-v2 embeddings│                       │
│  │  → 57+ chunks persisted in ChromaDB       │                       │
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

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **LLM (Primary)** | MegaLLM — `gemini-3-pro-preview` | OpenAI-compatible API | Primary LLM provider via `https://ai.megallm.io/v1` |
| **LLM (Fallback)** | Google Gemini 2.5 Flash | `gemini-2.5-flash` | Fallback with automatic 3-key rotation on quota exhaustion |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | via `langchain-huggingface` | Local semantic embeddings — no API cost |
| **Vector DB** | ChromaDB | `≥0.4.0` | Persistent vector store with similarity scores |
| **Framework** | LangChain | `≥0.1.0` | Chains, prompts, document processing |
| **UI** | Streamlit | `≥1.28.0` | Interactive web demo |
| **Runtime** | Python | 3.13 | Core language |

---

## 📁 Project Structure

```
.
├── src/
│   ├── rag_system.py          # Part A — RAG: chunking, retrieval, Q&A with citations
│   ├── agent_workflow.py      # Part B — 4-agent pipeline for article generation
│   ├── evaluation.py          # Evaluation framework: golden set, metrics, tracking
│   └── demo_examples.py       # Pre-built demo examples
│
├── data/                      # Kerala Ayurveda knowledge base
│   ├── ayurveda_foundations.md
│   ├── content_style_and_tone_guide.md
│   ├── dosha_guide_vata_pitta_kapha.md
│   ├── faq_general_ayurveda_patients.md
│   ├── product_ashwagandha_tablets_internal.md
│   ├── product_brahmi_tailam_internal.md
│   ├── product_triphala_capsules_internal.md
│   ├── treatment_stress_support_program.md
│   └── products_catalog.csv   # 8-product structured catalog
│
├── chroma_db/                 # Persisted ChromaDB vector index
├── evaluation_results/        # Timestamped evaluation JSON outputs
├── golden_set.json            # 5 benchmark Q&A pairs
├── metrics_history.jsonl      # Continuous metrics log
│
├── streamlit_app.py           # Web UI entrypoint
├── demo.py                    # RAG system demo script
├── test_project.py            # Project validation tests
├── requirements.txt           # Python dependencies
└── .env                       # API keys (MEGA_API_KEY + GOOGLE_API_KEY_1/2/3)
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Arnavsao/Agentic-AI-Internship-Assignemnt-Kerala-Ayurveda.git
cd Agentic-AI-Internship-Assignemnt-Kerala-Ayurveda

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```env
# Primary LLM — MegaLLM (gemini-3-pro-preview)
MEGA_API_KEY=sk-mega-your_megallm_key_here

# Fallback — Google Gemini (auto-rotated on quota exhaustion)
GOOGLE_API_KEY_1=your_google_api_key_1
GOOGLE_API_KEY_2=your_google_api_key_2   # optional
GOOGLE_API_KEY_3=your_google_api_key_3   # optional
```

- Get a MegaLLM key at [megallm.io](https://megallm.io)
- Get a free Gemini key at [Google AI Studio](https://aistudio.google.com/app/apikey)

The system tries MegaLLM first; if it fails for any reason it automatically falls back to Gemini.

### 3. Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) — the knowledge base loads automatically.

---

## 🔍 Part A — RAG System

**File:** `src/rag_system.py`

### How It Works

```
User Query
   ↓  Convert to embedding vector (all-MiniLM-L6-v2)
   ↓  Semantic search → retrieve 5 most relevant chunks
   ↓  Select top 3 for generation (balance relevance vs. context length)
   ↓  Build prompt with [Source X: doc_id - section_id] labels
   ↓  Gemini 2.5 Flash generates answer with brand voice
   ↓  Return QueryResponse(answer, citations, retrieved_chunks)
```

### Adaptive Chunking Strategy

Different document types are chunked differently — a single size doesn't fit all:

| Document Type | Chunk Size | Overlap | Reason |
|---|---|---|---|
| FAQ (`faq_*.md`) | 400 chars | 100 | Keep Q&A pairs together |
| Product (`product_*.md`) | 500 chars | 100 | Preserve product sections |
| Guide (`*guide*.md`, `dosha*.md`) | 800 chars | 100 | Conceptual content needs context |
| Default | 600 chars | 100 | General articles |

Splitters try to break at `## headers → ### headers → paragraphs → sentences` before falling back to characters, preserving semantic boundaries.

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

## 🚀 Running the Project

### Streamlit Web UI (Recommended)

```bash
streamlit run streamlit_app.py
```

### RAG System Demo (Terminal)

```bash
python -m src.rag_system
```

Runs 3 example queries and prints answers with full citation details.

### Agent Workflow Demo (Terminal)

```bash
python -m src.agent_workflow
```

Runs the full 4-agent pipeline on a sample stress & sleep article brief. Takes ~2-3 minutes.

### Evaluation Suite

```bash
python -m src.evaluation
```

Evaluates all 5 golden examples and saves results to `evaluation_results/`.

### Project Tests

```bash
python test_project.py
```

Validates that all imports, the RAG system, and the evaluation framework load correctly.

---

## 🎯 Key Design Decisions

**1. Local Embeddings (HuggingFace `all-MiniLM-L6-v2`)**
Using local embeddings instead of an API means no extra cost, no rate limits, and faster indexing. The 384-dimension model is well-suited for semantic medical Q&A.

**2. Adaptive Chunking by Document Type**
FAQ documents need small chunks (400 chars) to keep Q&A pairs together. Guides need larger chunks (800 chars) to maintain conceptual context. A single chunk size would degrade retrieval quality.

**3. Retrieve 5, Use Top 3**
Retrieving 5 chunks casts a wide semantic net while passing only 3 to the LLM keeps the prompt focused and avoids context dilution. All 5 are returned in the response for transparency.

**4. Per-Section RAG in Writer Agent**
The Writer Agent queries RAG once per outline section rather than once for the whole article. This ensures each section gets the most relevant sources, not a one-size-fits-all context block.

**5. 0.7 Grounding Threshold**
The Fact-Checker auto-rejects articles below 70% grounding. Medical content has zero tolerance for hallucination — this threshold triggers a revision loop (up to 2 iterations) before escalating to an editor note.

---

## 📦 Assignment Deliverables

| Requirement | Implementation |
|---|---|
| Part A: RAG with chunking | `src/rag_system.py` — adaptive chunking, ChromaDB, structured citations |
| Part A: Q&A with citations | `answer_user_query()` returns `QueryResponse` with doc_id + section_id |
| Part B: Multi-agent pipeline | `src/agent_workflow.py` — 4-agent orchestrator |
| Part B: Fact-checking guardrail | `FactCheckerAgent` with 0.7 grounding threshold |
| Evaluation framework | `src/evaluation.py` — golden set, 4 metrics, JSONL history |
| Web UI | `streamlit_app.py` — interactive demo with citations |

---

<div align="center">

**Kerala Ayurveda RAG System** · Built for the Agentic AI Internship Assignment · March 2026

*Stack: Python 3.13 · MegaLLM (gemini-3-pro-preview) · Google Gemini 2.5 Flash · LangChain · ChromaDB · HuggingFace · Streamlit*

</div>
