# How to Run - Kerala Ayurveda RAG + Agentic AI System

## Prerequisites

- Python 3.8 or higher
- Google Gemini API key (already configured in `.env`)

## Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
pip3 install -r requirements.txt
```

This will install:
- Google Gemini API (`google-generativeai`)
- LangChain with Gemini support
- ChromaDB (vector database)
- Sentence Transformers (local embeddings)
- Other required packages

**Installation time**: 2-5 minutes depending on your internet speed

### Step 2: Verify API Key

Make sure your `.env` file contains:
```
GOOGLE_API_KEY=AIzaSyCrJhSPnv1-2JjuV1w3pH2u0l9R5lElyl0
```

(Already configured - no action needed)

### Step 3: Run the Demo

```bash
python3 demo.py
```

## What You'll See

The demo script will show you a menu:

```
Choose a demo:
1. RAG Q&A System (with interactive mode)
2. Multi-Agent Blog Generation
3. Both
4. Exit
```

### Option 1: RAG Q&A System

This demonstrates:
- Loading and indexing Kerala Ayurveda content (from `context/` folder)
- Answering 3 pre-configured questions about Ayurveda
- **Interactive mode** where you can ask your own questions

Example flow:
1. System loads 8 markdown files from `context/`
2. Creates embeddings and stores in ChromaDB
3. Answers sample questions with citations
4. Opens interactive prompt for you to ask questions

**Try asking**:
- "What are the benefits of Ashwagandha?"
- "Tell me about the Stress Support Program"
- "Are there contraindications for Triphala?"
- "What is Vata dosha?"

### Option 2: Multi-Agent Blog Generation

This demonstrates the 5-agent pipeline:
1. **Outline Agent** - Creates article structure
2. **Writer Agent** - Writes content based on outline
3. **RAG Retrieval** - Pulls Kerala Ayurveda knowledge
4. **Fact-Checker Agent** - Verifies accuracy
5. **Tone Editor Agent** - Adjusts brand voice

Example output:
- Full generated article on "Benefits of Ashwagandha for Stress Management"
- Workflow metadata showing time for each step

### Option 3: Both

Runs both demos sequentially for a complete walkthrough.

## System Architecture Overview

```
Project Structure:
├── context/              # Kerala Ayurveda knowledge base (8 MD files)
├── src/
│   ├── rag_system.py    # RAG Q&A with citations
│   ├── agent_workflow.py # 5-agent blog generation
│   └── evaluation.py    # Golden set evaluation
├── demo.py              # Local demo script (RUN THIS)
├── requirements.txt     # Python dependencies
└── .env                 # Google Gemini API key
```

## Key Features Demonstrated

### 1. Adaptive Chunking
- Products: 400 chars with 100 char overlap
- Programs: 600 chars with 150 char overlap
- Other content: 800 chars with 200 char overlap

### 2. Semantic Search
- Local embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- ChromaDB vector database
- Top-k retrieval with relevance scoring

### 3. Structured Citations
- `doc_id` + `section_id` for traceability
- Relevance scores for each citation
- Content snippets from source material

### 4. Multi-Agent Workflow
- 5 specialized agents with different temperatures
- Sequential processing with state management
- Timing metadata for performance analysis

### 5. Google Gemini Integration
- Using `gemini-pro` model for all LLM calls
- Temperature settings: 0.1 (RAG), 0.2 (Writer/Tone), 0.3 (Outline), 0 (Fact-Checker)
- Local embeddings (no API calls for embeddings)

## Troubleshooting

### Import Errors
If you see `ModuleNotFoundError`:
```bash
pip3 install -r requirements.txt
```

### API Key Errors
If you see "GOOGLE_API_KEY not found":
1. Check `.env` file exists
2. Verify it contains: `GOOGLE_API_KEY=your_key_here`
3. Make sure no spaces around the `=` sign

### ChromaDB Errors
If you see ChromaDB initialization errors:
```bash
pip3 install --upgrade chromadb
```

### Slow First Run
First run will be slower because:
1. Downloading sentence-transformers model (~90MB)
2. Creating embeddings for all content
3. Building ChromaDB index

Subsequent runs will be faster due to caching.

## Expected Runtime

- **First time setup**: 2-5 minutes (installing dependencies)
- **RAG Q&A Demo**: 30-60 seconds (loading + 3 questions)
- **Interactive Q&A**: 3-5 seconds per question
- **Agent Blog Generation**: 30-90 seconds (depends on article length)

## Demo Output Examples

### RAG Q&A Output
```
Question: What are the benefits of Ashwagandha?

Answer:
Ashwagandha (Withania somnifera) is a cornerstone herb in Ayurvedic medicine,
known for its adaptogenic properties. Key benefits include:
- Stress reduction and cortisol regulation
- Enhanced energy and vitality
- Improved cognitive function
[...more details...]

Sources (3 citations):
1. ashwagandha_product.md (Section: benefits)
   Relevance: 94.2%
2. stress_support_program.md (Section: key-ingredients)
   Relevance: 87.6%
3. adaptogenic_herbs.md (Section: ashwagandha)
   Relevance: 85.3%
```

### Multi-Agent Output
```
GENERATED ARTICLE
═══════════════════════════════════════════════════

[Full article with proper Kerala Ayurveda brand voice]

WORKFLOW METADATA
═══════════════════════════════════════════════════
Total processing time: 45.32s
Outline generation: 8.12s
Writing: 22.45s
Fact checking: 9.67s
Tone editing: 5.08s
```

## Interview Tip

When presenting this project:
1. Start with **Option 1** to show RAG capabilities
2. Ask a custom question in interactive mode to show live demo
3. Then run **Option 2** to show multi-agent workflow
4. Explain the temperature settings and why each agent uses different values
5. Highlight the structured citations and fact-checking guardrails

## Next Steps

For evaluation metrics and testing:
```python
from src.evaluation import RAGEvaluator, ArticleEvaluator

# Create evaluators
rag_eval = RAGEvaluator()
article_eval = ArticleEvaluator()

# Run golden set tests
rag_results = rag_eval.run_golden_set("data/golden_set_rag.json")
article_results = article_eval.evaluate_article(article, references)
```

See [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md) for detailed code explanations.
