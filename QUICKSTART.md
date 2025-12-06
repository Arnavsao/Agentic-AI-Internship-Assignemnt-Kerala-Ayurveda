# Quick Start Guide - Kerala Ayurveda RAG System

## 🚀 Getting Started in 5 Minutes

### Step 1: Prerequisites

Make sure you have:
- **Python 3.9+** installed
- **MegaLLM API key** (get it from https://megallm.io)

### Step 2: Installation

```bash
# Navigate to project directory
cd "Assignement Agentic AI"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure API Key

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env file and add your MegaLLM API key
# Replace 'your_megallm_api_key_here' with your actual key
```

**Edit `.env` file:**
```bash
MEGALLM_API_KEY=sk-mega-your-actual-key-here
```

### Step 4: Run the Examples

#### Option A: Run RAG Q&A System (Part A)

```bash
python rag_system.py
```

**What this does:**
- Loads and indexes all Ayurveda content files
- Runs 3 example queries:
  1. "What are the key benefits of Ashwagandha tablets?"
  2. "Are there any contraindications for Triphala?"
  3. "Can Ayurveda help with stress and sleep?"
- Shows answers with citations

**Expected output:**
```
Loading and indexing content...
  Loaded ayurveda_foundations.md: 8 chunks (default type)
  Loaded content_style_and_tone_guide.md: 6 chunks (default type)
  ...

Building vector index with 45 total chunks...
Index built successfully!

================================================================================
TESTING QUERIES
================================================================================

Query: What are the key benefits of Ashwagandha tablets?
--------------------------------------------------------------------------------

Answer:
Ashwagandha Stress Balance Tablets are traditionally used to support...
[Full answer with citations]

Citations:
  [1] product_ashwagandha_tablets_internal - Traditional Positioning
      Relevance: 0.952
      ...
```

---

#### Option B: Run Agentic Article Generation (Part B)

```bash
python agent_workflow.py
```

**What this does:**
- Initializes the 5-agent workflow system
- Generates a full article on "Ayurvedic Support for Stress and Better Sleep"
- Runs through: Outline → Writer → Fact-Checker → Tone Editor → Final Review
- Shows the final article with quality scores

**Expected output:**
```
================================================================================
GENERATING ARTICLE
================================================================================

Step 1: Generating outline...
Step 2: Writing draft...
Step 3: Fact-checking (iteration 1)...
  Grounding score too low (0.65), revising...
Step 3: Fact-checking (iteration 2)...
Step 4: Editing tone and style...
Step 5: Final review...

================================================================================
FINAL ARTICLE
================================================================================

Ready for editor: True
Fact-check score: 0.87
Style score: 0.91

Editor notes:
  - None (article passed all checks!)

Article content:
[Full generated article with citations]
```

---

#### Option C: Run Evaluation Framework

```bash
python evaluation.py
```

**What this does:**
- Loads the golden set of test queries
- Evaluates RAG system on 5 benchmark examples
- Computes metrics: Coverage Score, Citation Accuracy, Hallucination Rate, Tone Compliance
- Saves results to `evaluation_results/` directory

**Expected output:**
```
================================================================================
RUNNING EVALUATION ON GOLDEN SET
================================================================================

Evaluating: q001 - What are the benefits of Ashwagandha for stress?
Evaluating: q002 - Are there any contraindications for Triphala?
...

================================================================================
EVALUATION RESULTS
================================================================================

Average Coverage Score: 85.00%
Average Citation Accuracy: 90.00%
Hallucination Rate: 8.00%
Tone Compliance Rate: 92.00%
```

---

#### Option D: View Example Query Analysis

```bash
python demo_examples.py
```

**What this does:**
- Shows detailed analysis of 3 example queries
- Displays expected retrieved documents
- Shows expected answers
- Lists potential failure modes and mitigations

**This is a documentation script** - no API calls, runs instantly.

---

## 🧪 Testing Your Own Queries

### Interactive Python Session

```bash
python
```

Then in Python:

```python
from rag_system import AyurvedaRAGSystem

# Initialize
rag = AyurvedaRAGSystem()
rag.load_and_index_content()

# Ask your own questions
response = rag.answer_user_query("Your question here")
print(response.answer)

# See citations
for citation in response.citations:
    print(f"{citation.doc_id} - {citation.section_id}")
```

### Custom Article Generation

```python
from agent_workflow import ArticleWorkflowOrchestrator, ArticleBrief
from rag_system import AyurvedaRAGSystem

# Initialize
rag = AyurvedaRAGSystem()
rag.load_and_index_content()
orchestrator = ArticleWorkflowOrchestrator(rag)

# Create your own brief
brief = ArticleBrief(
    topic="Your Topic Here",
    target_audience="Your Target Audience",
    key_points=["Point 1", "Point 2", "Point 3"],
    word_count_target=800,
    must_include_products=[]  # Optional
)

# Generate article
article = orchestrator.generate_article(brief)
print(article.content)
```

---

## 📁 Project Files Overview

```
Assignement Agentic AI/
├── rag_system.py              # Core RAG implementation (Part A)
├── agent_workflow.py          # Multi-agent article generation (Part B)
├── evaluation.py              # Evaluation framework
├── demo_examples.py           # Example analysis (no API calls)
├── requirements.txt           # Python dependencies
├── README.md                  # Full documentation
├── QUICKSTART.md              # This file
├── .env.example               # Environment variables template
├── .env                       # Your actual API key (git-ignored)
│
├── Content Files (provided by Kerala Ayurveda):
├── ayurveda_foundations.md
├── content_style_and_tone_guide.md
├── dosha_guide_vata_pitta_kapha.md
├── faq_general_ayurveda_patients.md
├── product_ashwagandha_tablets_internal.md
├── product_brahmi_tailam_internal.md
├── product_triphala_capsules_internal.md
├── treatment_stress_support_program.md
└── products_catalog.csv
```

---

## 🔧 Troubleshooting

### Error: "No module named 'openai'"

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Error: "MEGALLM_API_KEY not found"

```bash
# Check if .env file exists
ls -la .env

# If not, copy from example
cp .env.example .env

# Edit .env and add your key
nano .env  # or use any text editor
```

### Error: "Invalid API key"

- Make sure your MegaLLM API key is correct
- Get a new key from https://megallm.io
- Update your `.env` file with the new key

### ChromaDB Permission Errors

```bash
# Delete the old ChromaDB directory and recreate
rm -rf chroma_db
python rag_system.py
```

### Slow Performance

**First run is always slower** (building vector index). Subsequent runs use cached index.

To rebuild index:
```bash
rm -rf chroma_db
```

---

## 📊 Expected Performance

### RAG System (Part A)
- **First run:** ~30-60 seconds (building index)
- **Subsequent runs:** ~5-10 seconds per query
- **Index size:** ~2-3 MB for provided content

### Agent System (Part B)
- **Article generation:** ~2-5 minutes (multiple agent calls)
- **Quality scores:** Typically 0.7-0.9 (fact-check and style)

### Evaluation
- **Golden set evaluation:** ~2-3 minutes (5 examples)
- **Results saved to:** `evaluation_results/`

---

## 🎯 What to Try First

1. **Start simple:** Run `python demo_examples.py` (no API calls, instant)
2. **Test RAG:** Run `python rag_system.py` (1-2 minutes)
3. **Generate article:** Run `python agent_workflow.py` (3-5 minutes)
4. **Evaluate:** Run `python evaluation.py` (2-3 minutes)

---

## 💡 Tips

1. **First time setup takes longer** - Index building is one-time
2. **Check .env file** - Most errors are due to missing/wrong API key
3. **Virtual environment** - Always activate before running
4. **Monitor costs** - Each query uses MegaLLM tokens (track usage at megallm.io)
5. **Experiment** - Modify queries, briefs, and parameters to explore

---

## 🆘 Need Help?

- Check [README.md](README.md) for full documentation
- Review error messages carefully
- Ensure all prerequisites are met
- Try rebuilding virtual environment if issues persist

---

## ✅ Success Checklist

- [ ] Python 3.9+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file created with valid `MEGALLM_API_KEY`
- [ ] Can run `python demo_examples.py` successfully
- [ ] Can run `python rag_system.py` successfully

Once all checked, you're ready to go! 🚀
