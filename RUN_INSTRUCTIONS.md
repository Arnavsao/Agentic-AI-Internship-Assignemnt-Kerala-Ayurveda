# 🚀 How to Run the Kerala Ayurveda RAG System

## Quick Setup (5 minutes)

### Step 1: Activate Virtual Environment

```bash
cd "/Users/arnavsao/Desktop/Assignement Agentic AI"
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages (LangChain, ChromaDB, Streamlit, etc.)

### Step 3: Verify API Key

Check that your `.env` file contains your MegaLLM API key:

```bash
cat .env
```

Should show:
```
MEGALLM_API_KEY=sk-mega-your-key-here
```

If not, edit `.env` and add your key.

---

## 🎯 Ways to Run the Project

### Option 1: RAG Q&A System (CLI) - **RECOMMENDED TO START HERE**

```bash
# Make sure venv is activated
source venv/bin/activate

# Run the RAG system
python -m src.rag_system
```

**What it does:**
- Loads all Ayurveda content files (9 markdown files + CSV)
- Builds vector index (first time: ~30-60 seconds)
- Runs 3 example queries:
  1. "What are the key benefits of Ashwagandha tablets?"
  2. "Are there any contraindications for Triphala?"
  3. "Can Ayurveda help with stress and sleep?"
- Shows answers with structured citations

**Expected Output:**
```
Loading and indexing content...
  Loaded ayurveda_foundations.md: 4 chunks (default type)
  Loaded product_ashwagandha_tablets_internal.md: 5 chunks (product type)
  ...
Building vector index with 47 total chunks...
Index built successfully!

================================================================================
TESTING QUERIES
================================================================================

Query: What are the key benefits of Ashwagandha tablets?
--------------------------------------------------------------------------------

Answer:
Ashwagandha tablets are traditionally used to support the body's ability
to adapt to stress and promote emotional balance [Source 1]...

Citations:
  [1] product_ashwagandha_tablets_internal - Traditional Positioning
      Relevance: 0.952
```

---

### Option 2: Web Interface (Streamlit) - **BEST FOR DEMO**

```bash
# Make sure venv is activated
source venv/bin/activate

# Run Streamlit app
streamlit run streamlit_app.py
```

**What it does:**
- Opens a web browser interface
- Interactive Q&A system
- Shows citations with expandable sections
- Displays relevance scores

**Access:** Browser will open automatically at `http://localhost:8501`

---

### Option 3: Multi-Agent Article Generation (CLI)

```bash
# Make sure venv is activated
source venv/bin/activate

# Run agent workflow
python -m src.agent_workflow
```

**What it does:**
- Generates a complete article from a brief
- Runs through 5-agent pipeline:
  1. Outline Agent (verifies corpus coverage)
  2. Writer Agent (drafts with RAG context)
  3. Fact-Checker Agent (validates grounding)
  4. Tone Editor (ensures brand voice)
  5. Final Review (generates editor notes)
- Shows quality scores and final article

**Expected Output:**
```
Step 1: Generating outline...
Step 2: Writing draft...
Step 3: Fact-checking (iteration 1)...
  Grounding score: 0.85 ✓
Step 4: Editing tone and style...
Step 5: Final review...

Final Article:
  Ready for editor: True
  Fact-check score: 0.85
  Style score: 0.92
  Word count: 823
  Citations: 12
```

**Note:** This takes 2-5 minutes (multiple API calls)

---

### Option 4: Evaluation Framework

```bash
# Make sure venv is activated
source venv/bin/activate

# Run evaluation
python -m src.evaluation
```

**What it does:**
- Tests system on 5 golden set questions
- Computes metrics:
  - Coverage Score
  - Citation Accuracy
  - Hallucination Rate
  - Tone Compliance
- Saves results to `evaluation_results/` directory

---

### Option 5: Interactive Python Session

```bash
# Make sure venv is activated
source venv/bin/activate

# Start Python
python
```

Then in Python:

```python
from src.rag_system import AyurvedaRAGSystem

# Initialize and load content
rag = AyurvedaRAGSystem()
rag.load_and_index_content()  # First time: ~30-60 seconds

# Ask your own questions
response = rag.answer_user_query("What is Vata dosha?")
print(response.answer)

# See citations
for citation in response.citations:
    print(f"{citation.doc_id} - {citation.section_id} ({citation.relevance_score:.2%})")
```

---

## 🔧 Troubleshooting

### Error: "No module named 'X'"

**Solution:** Install dependencies
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Error: "MEGALLM_API_KEY not found"

**Solution:** Check `.env` file exists and has your API key
```bash
cat .env
# Should show: MEGALLM_API_KEY=sk-mega-...
```

### Error: "ChromaDB permission error"

**Solution:** Delete and rebuild index
```bash
rm -rf chroma_db
python -m src.rag_system
```

### Slow Performance

**First run is always slower** (~30-60 seconds) because it builds the vector index.
Subsequent runs use the cached index and are much faster (~5-10 seconds per query).

---

## 📊 Performance Expectations

| Component | First Run | Subsequent Runs |
|-----------|-----------|-----------------|
| RAG System | 30-60 sec | 5-10 sec/query |
| Article Generation | 2-5 minutes | 2-5 minutes |
| Streamlit App | 30-60 sec startup | Instant queries |
| Evaluation | 2-3 minutes | 2-3 minutes |

---

## ✅ Quick Start Checklist

- [ ] Virtual environment activated (`source venv/bin/activate`)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file has valid `MEGALLM_API_KEY`
- [ ] Can run `python -m src.rag_system` successfully

---

## 🎯 Recommended First Steps

1. **Start with RAG System:**
   ```bash
   python -m src.rag_system
   ```

2. **Then try Web Interface:**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Explore Article Generation:**
   ```bash
   python -m src.agent_workflow
   ```

---

## 💡 Tips

- **Always activate venv first:** `source venv/bin/activate`
- **First run builds index:** Be patient, it's one-time
- **Check API key:** Most errors are due to missing/wrong API key
- **Monitor costs:** Each query uses MegaLLM tokens
- **Experiment:** Try different queries and briefs

---

## 🆘 Need Help?

- Check `README.md` for full documentation
- Review `docs/QUICKSTART.md` for detailed setup
- Check error messages carefully
- Ensure Python 3.11+ is installed
