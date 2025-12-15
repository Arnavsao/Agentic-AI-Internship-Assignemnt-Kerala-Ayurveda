# Kerala Ayurveda RAG + Agentic AI System

**Assignment submission for Agentic AI Internship at Kerala Ayurveda**

A production-ready system demonstrating RAG (Retrieval-Augmented Generation) for Q&A and multi-agent workflows for article generation, with comprehensive quality checks and evaluation framework.

---

## 🎯 Project Overview

This project implements two main systems:

1. **Part A: RAG System** - Answers user questions about Ayurveda with structured citations
2. **Part B: Multi-Agent Workflow** - Generates complete articles from briefs with fact-checking and style validation
3. **Evaluation Framework** - Golden set testing with automated metrics

---

## 📁 Project Structure

```
kerala-ayurveda-rag/
├── data/                                  # Content corpus (Kerala Ayurveda materials)
│   ├── ayurveda_foundations.md
│   ├── content_style_and_tone_guide.md
│   ├── dosha_guide_vata_pitta_kapha.md
│   ├── faq_general_ayurveda_patients.md
│   ├── product_ashwagandha_tablets_internal.md
│   ├── product_brahmi_tailam_internal.md
│   ├── product_triphala_capsules_internal.md
│   ├── products_catalog.csv
│   └── treatment_stress_support_program.md
│
├── src/                                   # Core implementation
│   ├── __init__.py                        # Package initialization
│   ├── rag_system.py                      # RAG Q&A system (Part A)
│   ├── agent_workflow.py                  # Multi-agent article generation (Part B)
│   ├── evaluation.py                      # Evaluation framework
│   └── demo_examples.py                   # Example query demonstrations
│
├── docs/                                  # Documentation
│   ├── README.md                          # Comprehensive documentation
│   └── QUICKSTART.md                      # Quick start guide
│
├── streamlit_app.py                       # Web UI for demo
├── requirements.txt                       # Python dependencies
├── .env.example                           # Environment variables template
├── .gitignore                             # Git ignore rules
├── .python-version                        # Python version specification
└── .streamlit/                            # Streamlit configuration
    └── config.toml
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <your-repo-url>
cd kerala-ayurveda-rag

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your MegaLLM API key
# MEGALLM_API_KEY=your_key_here
```

### 3. Run the System

**Option A: Q&A System (CLI)**
```bash
python -m src.rag_system
```

**Option B: Article Generation (CLI)**
```bash
python -m src.agent_workflow
```

**Option C: Web Interface**
```bash
streamlit run streamlit_app.py
```

---

## 🔑 Key Features

### Part A: RAG System

- **Adaptive Chunking**: Different chunk sizes (400-800 chars) based on document type
  - FAQs: 400 chars (keep Q&A pairs intact)
  - Products: 500 chars (preserve sections)
  - Guides: 800 chars (maintain conceptual context)

- **Semantic Retrieval**: OpenAI embeddings via MegaLLM
  - Retrieve k=5, use top 3 for generation
  - Balances relevance and token budget

- **Structured Citations**: Every answer includes:
  - `doc_id` - Source document
  - `section_id` - Specific section
  - `relevance_score` - Semantic similarity (0-1)

- **Brand Voice Enforcement**: Kerala Ayurveda style in system prompt
  - "traditionally used to support..."
  - "may help maintain..."
  - Never claims to diagnose, treat, cure, prevent

### Part B: Multi-Agent Workflow

**5-Agent Pipeline**:
1. **Outline Agent** - Creates structured outline (verifies corpus coverage)
2. **Writer Agent** - Drafts article with RAG context per section
3. **Fact-Checker Agent** - Verifies grounding (auto-rejects <0.7)
4. **Tone Editor** - Ensures brand voice compliance
5. **Final Review** - Generates editor notes

**Guardrails**:
- Medical claims without sources = auto-reject
- Grounding score < 0.7 = auto-reject
- Citation preservation verified during editing
- Safety language protected

### Evaluation Framework

**Golden Set**: 5 benchmark questions covering:
- Product benefits (Ashwagandha)
- Safety/contraindications (Triphala)
- General FAQs (stress & sleep)
- Conceptual knowledge (doshas)
- Treatment programs

**Metrics Tracked**:
- Coverage Score (target >0.80)
- Citation Accuracy (target >0.85)
- Hallucination Rate (target <0.10)
- Tone Compliance (target >0.90)
- Grounding Score (target >0.85)

---

## 📊 Technical Architecture

### Tech Stack
- **LangChain**: Document processing, prompting, chains
- **ChromaDB**: Vector database for embeddings
- **OpenAI Embeddings**: text-embedding-3-small
- **MegaLLM API**: Unified gateway (70+ models)
- **Streamlit**: Web UI
- **Python 3.11**: Core implementation

### Design Decisions

**Why Adaptive Chunking?**
- Medical content varies in structure
- FAQs need small chunks, guides need large chunks
- One-size-fits-all loses critical context

**Why Retrieve 5, Use 3?**
- Cast wide net (k=5) to avoid missing info
- Use top 3 to manage token costs
- Empirically optimal balance

**Why Low Temperature (0.1-0.2)?**
- Medical content prioritizes accuracy over creativity
- Consistency critical for guardrail effectiveness
- Reduces hallucination risk

**Why Fact-Check Threshold 0.7?**
- <70% grounding = significant unsupported claims
- Medical content has zero tolerance for hallucination
- Editor still reviews, but catches major issues

---

## 💻 Usage Examples

### Q&A System

```python
from src.rag_system import AyurvedaRAGSystem

# Initialize and load content
rag = AyurvedaRAGSystem()
rag.load_and_index_content()

# Ask a question
response = rag.answer_user_query("What are the benefits of Ashwagandha?")

print(response.answer)
for citation in response.citations:
    print(f"[{citation.doc_id}] {citation.section_id} ({citation.relevance_score:.2%})")
```

### Article Generation

```python
from src.agent_workflow import ArticleWorkflowOrchestrator, ArticleBrief

# Initialize
orchestrator = ArticleWorkflowOrchestrator(rag)

# Create brief
brief = ArticleBrief(
    topic="Ayurvedic Support for Stress and Better Sleep",
    target_audience="Busy professionals",
    key_points=["Ayurveda view", "Lifestyle", "Herbs"],
    word_count_target=800,
    must_include_products=["Ashwagandha Tablets"]
)

# Generate article
article = orchestrator.generate_article(brief)
print(f"Ready: {article.ready_for_editor}")
print(f"Fact-Check: {article.fact_check_score:.2%}")
print(f"Style: {article.style_score:.2%}")
```

### Evaluation

```python
from src.evaluation import RAGEvaluator, GoldenSetManager

# Run evaluation
golden_manager = GoldenSetManager()
evaluator = RAGEvaluator(rag)
metrics = evaluator.evaluate_golden_set(golden_manager.examples)

print(f"Coverage: {metrics['avg_coverage_score']:.2%}")
print(f"Hallucination: {metrics['hallucination_rate']:.2%}")
```

---

## 📚 Documentation

- **[Full Documentation](docs/README.md)** - Comprehensive guide with design rationale
- **[Quick Start Guide](docs/QUICKSTART.md)** - Setup and usage instructions
- **[Demo Examples](src/demo_examples.py)** - Example queries with failure mode analysis

---

## 🧪 Testing

```bash
# Test RAG system
python -m src.rag_system

# Test agent workflow
python -m src.agent_workflow

# Run evaluation
python -m src.evaluation
```

---

## 🌐 Deployment

The system is deployed on Streamlit Cloud:
- **URL**: [Add your deployment URL]
- **Configuration**: Python 3.11, MegaLLM API
- **Secrets**: MEGALLM_API_KEY configured in Streamlit settings

---

## 📝 Assignment Requirements Met

### Part A ✅
- [x] Adaptive chunking strategy (400-800 chars)
- [x] Embeddings-based retrieval with semantic search
- [x] `answer_user_query()` function with structured citations
- [x] 3 example queries with outputs & failure modes

### Part B ✅
- [x] 5-step agent workflow (Outline → Writer → Fact-Check → Tone → Review)
- [x] Input/output schemas for each agent
- [x] Failure modes identified per agent
- [x] Guardrails implemented per agent
- [x] Evaluation framework with golden set
- [x] Metrics tracking (coverage, citations, hallucination, tone)
- [x] 2-week prioritization plan

### Reflection ✅
- Time spent: ~3-5 hours focused work
- Most interesting: Multi-agent orchestration, fact-checking guardrails
- AI tools used: Claude Code for implementation, debugging, documentation

---

## 🔧 Future Enhancements

**Explicitly Postponed (2-week plan)**:
- Hybrid BM25 + embeddings retrieval
- Fine-tuned embeddings for medical domain
- Automated evaluation dashboard
- Multi-lingual support
- Human-in-the-loop editing UI
- CMS integration

---

## 📄 License

This project is submitted as part of the Agentic AI Internship assignment for Kerala Ayurveda.

---

## 👤 Author

**Arnav Sao**
- Assignment: Agentic AI Internship @ Kerala Ayurveda
- Date: December 2025

---

## 🙏 Acknowledgments

- Kerala Ayurveda for the internship opportunity
- Content corpus provided by Kerala Ayurveda
- Built with LangChain, ChromaDB, and MegaLLM
