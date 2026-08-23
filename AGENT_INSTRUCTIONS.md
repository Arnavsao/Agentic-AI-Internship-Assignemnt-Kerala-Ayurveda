# 🤖 System Instructions for AI Agents & Models

**Target Audience:** AI Models (Claude, Gemini, ChatGPT, etc.) acting as coding assistants, code reviewers, or maintainers for this repository.

---

## 1. Project Context
You are interacting with the **Kerala Ayurveda RAG System**, a production-ready Retrieval-Augmented Generation (RAG) system with a multi-agent article generation pipeline.

**Core Capabilities:**
1. **RAG Q&A (`src/rag_system.py`):** Answers questions using semantic search over local medical/ayurvedic documents with structured citations.
2. **Multi-Agent Workflow (`src/agent_workflow.py`):** A 4-agent pipeline (Outline → Write → Fact-Check → Tone Edit) that drafts publication-ready articles.
3. **Evaluation Framework (`src/evaluation.py`):** Benchmarks system performance on coverage, citations, hallucination, and tone against a golden dataset.

---

## 2. Codebase Architecture & Key Files

When making changes or debugging, you must respect the following architectural decisions:

### A. RAG System (`src/rag_system.py`)
- **Embeddings:** Uses local HuggingFace `all-MiniLM-L6-v2` (384 dimensions). Do NOT try to switch this to an API-based embedding unless explicitly instructed.
- **Vector DB:** `ChromaDB`. It persists locally in `./chroma_db`.
- **Adaptive Chunking:** We do not use a single chunk size. Documents are chunked dynamically based on type (FAQ=400, Product=500, Guide/PDF=800).
- **Citations:** Every generated answer MUST include traceable citations using `[Source: doc_id - section_id]`.

### B. Agentic Workflow (`src/agent_workflow.py`)
The pipeline runs strictly in this order:
1. **OutlineAgent:** Creates a JSON outline. Queries RAG to ensure topics are covered.
2. **WriterAgent:** Writes the draft section-by-section. Queries RAG for *each* section independently.
3. **FactCheckerAgent:** Extracts claims and scores grounding. **Crucial Guardrail:** Automatically rejects drafts with a grounding score below `0.7`.
4. **ToneEditorAgent:** Adjusts brand tone (warm, reassuring, precise) but is strictly forbidden from altering facts or removing citations/safety notes.

### C. LLM Key Manager (`src/key_manager.py`)
- We use a custom `GeminiKeyManager` to handle API limits.
- **Primary Provider:** MegaLLM (`gemini-3-pro-preview` via `https://ai.megallm.io/v1`).
- **Fallback Provider:** Google Gemini API (`gemini-2.5-flash`), utilizing automatic key rotation (`GOOGLE_API_KEY_1`, `GOOGLE_API_KEY_2`, etc.) upon 429/Quota Exhausted errors.
- **Rule:** Whenever you initialize an LLM in this codebase, you MUST use `self.key_manager.invoke_with_rotation` to ensure high availability. Do not make direct LLM calls that bypass the key manager.

---

## 3. Important Rules & Guardrails for AI Modifying this Code

If the user asks you to modify, extend, or debug this code, adhere strictly to these rules:

1. **Medical Safety First:** This project deals with Ayurveda and health. Never remove medical disclaimers, safety guidelines, or the requirement to "consult a qualified practitioner" from the prompts or code logic.
2. **No Hallucinations in the Code:** The `FactCheckerAgent` acts as an LLM-as-a-judge. If you modify its prompt, ensure it remains extremely strict about verifying claims against the provided context.
3. **Preserve JSON Extraction Robustness:** The `_extract_json()` function in `agent_workflow.py` handles LLMs wrapping JSON in markdown blocks. Do not replace this with simple `json.loads` as it will break the pipeline.
4. **Dependencies:** Use the exact versions specified in `requirements.txt`. (e.g., `sentence-transformers 5.2.2`, `chromadb 1.4.1`, `streamlit 1.54.0`). Do not introduce unnecessary heavy libraries.
5. **Warning Suppression:** `rag_system.py` explicitly suppresses HuggingFace and PyTorch warnings to keep the terminal output clean for the user. Preserve these suppression blocks.

---

## 4. How to Run / Test Your Changes

When asked to verify your work, guide the user to run these commands:
- **Web UI:** `streamlit run streamlit_app.py`
- **RAG Demo:** `python -m src.rag_system`
- **Agent Workflow:** `python -m src.agent_workflow`
- **Evaluation Suite:** `python -m src.evaluation`

If you are writing tests, ensure they fit within the existing `test_project.py` structure or use the established golden set in `golden_set.json`.
