# Kerala Ayurveda RAG & Agentic AI System

**Assignment Submission for Agentic AI Internship Role**

A production-ready RAG system and multi-agent workflow for Kerala Ayurveda content generation, with built-in fact-checking, style validation, and evaluation framework.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Part A: RAG System Design](#part-a-rag-system-design)
- [Part B: Agentic Workflow](#part-b-agentic-workflow)
- [Installation & Setup](#installation--setup)
- [Usage Examples](#usage-examples)
- [2-Week Implementation Plan](#2-week-implementation-plan)
- [Reflection](#reflection)

---

## 🎯 Project Overview

This system addresses Kerala Ayurveda's need for:
1. **Accurate Q&A** on Ayurveda products, treatments, and concepts
2. **Content generation** that's grounded, brand-safe, and editor-ready
3. **Quality assurance** through automated evaluation and fact-checking

**Key Features:**
- Adaptive document chunking based on content type
- Hybrid retrieval with semantic search
- Multi-agent article generation pipeline
- Automated fact-checking and tone validation
- Comprehensive evaluation framework with golden set testing

---

## 📚 Part A: RAG System Design

### High-Level Approach (10 bullet points)

1. **Adaptive chunking by document type:**
   - FAQs: 400 chars (keep Q&A pairs together)
   - Products: 500 chars (structured sections)
   - Guides: 800 chars (conceptual content needs context)
   - 100 char overlap to preserve context at boundaries

2. **Markdown-aware splitting:**
   - Split on headers first (`## `, `### `), then paragraphs
   - Preserves document structure and semantic coherence

3. **Embeddings-based semantic retrieval (not BM25):**
   - Better for medical content where semantic similarity > keyword matching
   - Handles synonyms naturally (stress/anxiety, digestion/gut health)
   - OpenAI embeddings with ChromaDB vector store

4. **Retrieve k=5, use top 3 in prompt:**
   - Cast wider net (5) ensures critical info not missed
   - Use top 3 balances relevance vs. context length
   - Manages token budget (~450-600 tokens for context)

5. **Structured citations with doc_id + section_id:**
   - Format: `[Source: product_ashwagandha - Traditional Positioning]`
   - Enables traceability back to original content
   - Includes relevance scores for quality assessment

6. **Low temperature (0.1) for consistency:**
   - Prioritizes factual accuracy over creativity
   - Critical for medical/health content

7. **Kerala Ayurveda style in system prompt:**
   - "Traditionally used to support..." language
   - No disease claims (diagnose/treat/cure)
   - Always include safety disclaimers

8. **Separate retrieval for product catalog CSV:**
   - Products converted to rich text representation
   - Indexed alongside markdown documents
   - Enables product-specific queries

9. **Metadata-rich chunks:**
   - Every chunk tagged with: doc_id, section_id, doc_type, chunk_index
   - Enables filtering, analysis, debugging

10. **Graceful degradation:**
    - If no relevant docs found, explicitly say "not in knowledge base"
    - Never fabricate information

### Function Design

**Core Implementation:**

```python
def answer_user_query(query: str) -> QueryResponse:
    # 1. Retrieve relevant chunks with scores
    retrieved = vectorstore.similarity_search_with_relevance_scores(query, k=5)

    # 2. Select top 3 for context
    top_chunks = retrieved[:3]

    # 3. Build context with source attribution
    context = build_context_with_sources(top_chunks)

    # 4. Generate answer (temp=0.1, Kerala Ayurveda style)
    answer = llm.invoke(prompt.format(context=context, query=query))

    # 5. Extract and structure citations
    citations = [
        Citation(doc_id, section_id, snippet, score)
        for doc, score in top_chunks
    ]

    # 6. Return structured response
    return QueryResponse(answer, citations, retrieved_chunks)
```

See [rag_system.py](rag_system.py) for full implementation.

### Example Query Analysis

**See [demo_examples.py](demo_examples.py) for detailed analysis.**

#### Example 1: "What are the key benefits of Ashwagandha tablets?"

**Retrieved:** `product_ashwagandha_tablets_internal` (sections: Traditional Positioning, Key Messages, Safety)

**Answer includes:**
- Stress resilience support
- Emotional balance promotion
- Restful sleep support
- Safety warnings for thyroid conditions

**Failure mode:** Could claim to "cure anxiety" → **Mitigation:** Strict prompt + fact-checker flags disease claims

#### Example 2: "Are there any contraindications for Triphala?"

**Retrieved:** `product_triphala_capsules_internal - Safety & Precautions`, `catalog_KA-P001`

**Answer includes:**
- Chronic digestive disease warning
- Pregnancy/breastfeeding caution
- Post-surgery consultation need

**Failure mode:** Too generic ("consult doctor" without specifics) → **Mitigation:** Prompt emphasizes specificity

#### Example 3: "Can Ayurveda help with stress and sleep?"

**Retrieved:** `faq_general_ayurveda_patients - Q3`, `treatment_stress_support_program`, `product_ashwagandha`

**Answer includes:**
- Daily routines, herbs, therapies
- Complementary (not replacement) framing
- Mental health professional recommendation

**Failure mode:** Overpromising results → **Mitigation:** Style guide + tone checker enforce "may help" language

---

## 🤖 Part B: Agentic Workflow

### Agent Step Graph (5 Agents)

```
Brief → [1] Outline → [2] Writer → [3] Fact-Checker → [4] Tone Editor → [5] Final Review
```

### Agent Specifications

#### [1] Outline Agent

**Input:** ArticleBrief (topic, audience, key_points, word_count, products)

**Output:** Outline (title, sections, estimated_word_count, key_sources_needed)

**Failure mode:** Creates outline with topics not in corpus

**Guardrail:** Query RAG first to verify topic coverage; only create sections that can be grounded

---

#### [2] Writer Agent

**Input:** ArticleBrief + Outline

**Output:** Draft (content, word_count, citations, sections)

**Failure modes:**
- Adds claims not in corpus (hallucination)
- Forgets citations
- Wrong tone (marketing speak)

**Guardrails:**
- Retrieve RAG context for EACH section before writing
- Require ≥1 citation per section
- Use Kerala Ayurveda style guide in prompt

---

#### [3] Fact-Checker Agent

**Input:** Draft

**Output:** FactCheckResult (is_grounded, grounding_score, unsupported_claims, suggested_fixes)

**Process:**
1. Extract all factual claims
2. Verify each has citation
3. Compute grounding_score = supported / total
4. REJECT if score < 0.7

**Failure modes:**
- Misses subtle unsupported claims
- Over-flags reasonable inferences

**Guardrails:**
- **CRITICAL:** Auto-reject medical claims without source
- Break compound sentences into atomic claims
- For unsupported claims, query RAG for potential sources

---

#### [4] Tone Editor Agent

**Input:** Draft + FactCheckResult

**Output:** ToneCheckResult (style_score, issues, revised_content)

**Checks:**
- ✅ Uses "traditionally used to support" language
- ✅ Safety disclaimers present
- ✅ Warm, reassuring tone
- ❌ No disease claims
- ❌ No guarantees

**Failure modes:**
- Changes meaning while editing
- Removes safety warnings

**Guardrails:**
- Explicit instruction: preserve all citations and disclaimers
- Compare revised vs original for semantic drift
- Flag if safety language weakened

---

#### [5] Final Reviewer (Implicit)

**Output:** FinalArticle (content, citations, scores, ready_for_editor, editor_notes)

**Ready criteria:**
```python
ready = (
    fact_check_score >= 0.7 AND
    style_score >= 0.7 AND
    citation_count > 0
)
```

**Editor notes:** Auto-generated alerts for low scores, missing disclaimers, etc.

---

### Evaluation Framework

#### Golden Set Structure

**5 initial examples across categories:**

| ID | Query | Category | Tests |
|----|-------|----------|-------|
| q001 | Benefits of Ashwagandha | Product | Knowledge + tone |
| q002 | Contraindications for Triphala | Product | Safety info |
| q003 | Can Ayurveda help stress/sleep? | FAQ | Balanced positioning |
| q004 | What is Vata dosha? | Concept | Conceptual explanation |
| q005 | Stress Support Program details | Treatment | Clinic offerings |

**Growth strategy:** Add real user queries, edge cases, failures from production.

#### Metrics Tracked Over Time

**RAG System:**
- Coverage Score (>0.80): % of expected points covered
- Citation Accuracy (>0.85): % of correct sources cited
- Hallucination Rate (<0.10): % with unsupported claims
- Tone Compliance (>0.90): % with appropriate tone

**Agent System:**
- Grounding Score (>0.85): % of article claims with sources
- Style Score (>0.85): Brand voice alignment
- Citation Density (1-3 per 100 words): Not too sparse/overwhelming
- Safety Disclaimer Rate (100%): Legal compliance
- Editor Acceptance Rate (>0.70): % accepted without major changes

**Continuous Loop:**
1. Run evaluation on golden set
2. Log metrics to time-series
3. Alert on regression
4. Human review of failures
5. Update golden set with learnings

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9+
- MegaLLM API key (provides access to 70+ AI models including OpenAI GPT-4, Claude, Gemini, etc.)
  - Sign up at: https://megallm.io

### Installation

```bash
cd "Assignement Agentic AI"

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your MEGALLM_API_KEY
```

### Why MegaLLM?

This project uses **MegaLLM** as the unified API gateway:
- **One API for all models**: Access GPT-4, Claude, Gemini, Llama through a single endpoint
- **Cost optimization**: Automatic fallbacks and model selection
- **Simple integration**: Uses standard OpenAI SDK with custom base URL
- **No changes needed**: Code works with OpenAI SDK you already know

```python
from openai import OpenAI

# Point to MegaLLM endpoint
client = OpenAI(
    base_url="https://ai.megallm.io/v1",
    api_key="your-megallm-api-key"
)
```

### Project Structure

```
.
├── rag_system.py              # Core RAG (Part A)
├── agent_workflow.py          # Multi-agent system (Part B)
├── evaluation.py              # Evaluation framework
├── demo_examples.py           # Example query analysis
├── requirements.txt
├── README.md
│
├── [9 content files from Kerala Ayurveda]
└── products_catalog.csv
```

---

## 🚀 Usage Examples

### RAG Q&A

```python
from rag_system import AyurvedaRAGSystem

rag = AyurvedaRAGSystem()
rag.load_and_index_content()

response = rag.answer_user_query("What are the benefits of Ashwagandha?")
print(response.answer)
```

**Run demo:** `python rag_system.py`

### Article Generation

```python
from agent_workflow import ArticleWorkflowOrchestrator, ArticleBrief
from rag_system import AyurvedaRAGSystem

rag = AyurvedaRAGSystem()
rag.load_and_index_content()
orchestrator = ArticleWorkflowOrchestrator(rag)

brief = ArticleBrief(
    topic="Ayurvedic Support for Stress and Sleep",
    target_audience="Busy professionals",
    key_points=["Ayurveda view of stress", "Lifestyle approaches", "Supportive herbs"],
    word_count_target=800,
    must_include_products=["Ashwagandha Stress Balance Tablets"]
)

article = orchestrator.generate_article(brief)
print(f"Ready: {article.ready_for_editor}")
print(f"Scores: {article.fact_check_score:.2f} / {article.style_score:.2f}")
```

**Run demo:** `python agent_workflow.py`

### Evaluation

```python
from evaluation import GoldenSetManager, RAGEvaluator

golden_manager = GoldenSetManager()
evaluator = RAGEvaluator(rag_system)
metrics = evaluator.evaluate_golden_set(golden_manager.examples)

print(f"Coverage: {metrics['avg_coverage_score']:.2%}")
print(f"Hallucination Rate: {metrics['hallucination_rate']:.2%}")
```

**Run demo:** `python evaluation.py`

**View example analysis:** `python demo_examples.py`

---

## 📅 2-Week Implementation Plan

### DEFINITELY Ship in 2 Weeks

**Week 1: Core RAG (70% effort)**

- **Days 1-2:** Data ingestion & chunking
  - Adaptive chunking by doc type
  - Build vector index
  - **Deliverable:** Can retrieve chunks

- **Days 3-4:** RAG Q&A
  - Implement `answer_user_query()`
  - Kerala Ayurveda system prompt
  - Citation formatting
  - **Deliverable:** Working Q&A for testing

- **Day 5:** Basic evaluation
  - Create golden set (5-10 examples)
  - Manual testing
  - **Deliverable:** Quality baseline

**Week 2: Minimal Agent + Polish (30% effort)**

- **Days 6-7:** 2-agent workflow
  - Writer agent (brief → draft)
  - Fact-checker (draft → scores)
  - **Deliverable:** Grounded drafts

- **Days 8-9:** Safety guardrails
  - Reject grounding < 0.7
  - Verify safety disclaimers
  - Flag uncited medical claims
  - **Deliverable:** Safe content generation

- **Day 10:** Testing & docs
  - Test on 3-5 real briefs
  - Document usage + limits
  - **Deliverable:** Growth team can use it

### EXPLICITLY Postpone

**Weeks 3-4:**
- Full 5-agent workflow (outline + tone agents)
- Hybrid BM25+embeddings retrieval
- Automated evaluation dashboard
- Multi-lingual support

**Month 2+:**
- Human-in-the-loop editing UI
- Fine-tuned embeddings
- Production deployment (API, caching)
- CMS integration

### Success Criteria (Week 2)

**Minimum Viable Product:**
- ✅ Growth team gets grounded answers with citations
- ✅ Growth team gets 70%+ editor-ready drafts
- ✅ All content has safety disclaimers
- ✅ Hallucination rate < 10%
- ✅ "Saves me 60% of time vs. writing from scratch"

**Not Required Yet:**
- ❌ Production infrastructure
- ❌ Perfect first drafts (70% is enough)
- ❌ 100% edge case coverage

### Rationale: Value Over Perfection

**80/20 rule:** Basic RAG + fact-checking = 80% of value

**Human-in-the-loop:** Start with AI drafts + human refine, automate refinement later

**Learn before optimizing:** Need real usage data to know what to optimize

---

## 💭 Reflection

### Time Spent: ~4.5 hours

| Activity | Time |
|----------|------|
| Reading assignment + content | 45 min |
| Designing RAG architecture | 60 min |
| Coding RAG (Part A) | 75 min |
| Designing agent workflow | 30 min |
| Coding agents (Part B) | 60 min |
| Evaluation framework | 30 min |
| Documentation | 60 min |

### Most Interesting

1. **Adaptive chunking trade-offs:** FAQ Q&A pairs need 400 chars, guides need 800. No one-size-fits-all.

2. **Fact-checking as first-class agent:** Medical content has zero tolerance for hallucination. Auto-reject < 0.7 is non-negotiable.

3. **Citation density as quality signal:** 1-3 per 100 words is sweet spot. Too few = hallucinating, too many = research paper.

### Unclear / Would Ask Team

1. **Average article length?** Assumed 800, but might prefer 1200+ for SEO
2. **Dosha personalization?** Generate per-dosha variants (Vata/Pitta/Kapha)?
3. **Editor vs. self-service?** AI→editor→publish or AI→auto-publish?
4. **CMS integration?** Existing tools that would influence architecture?

### AI Tool Usage

**Tools:**
- **Claude Code (this session):** 100% of code generation
- **Used for:** Writing Python, structuring project, drafting README
- **Process:** Design architecture first, then prompt for implementation, then review

**Not used AI for:**
- System design decisions (chunk sizes, k=5→3, threshold 0.7)
- Failure mode analysis and guardrails
- Prioritization strategy
- These require domain judgment

**Philosophy:** AI excellent for implementation, humans essential for architecture.

---

## 🎓 Key Takeaways

**Technical:**
- Retrieval is the bottleneck
- Grounding > Fluency for medical content
- Adaptive strategies beat one-size-fits-all
- Citations are features, not footnotes

**Product:**
- Ship 70% perfect in 2 weeks > 95% perfect in 2 months
- Automate tedious, not creative
- Golden sets are invaluable
- Design for distrust initially

**Process:**
- Start with design docs, not code
- Test with real content early
- Think in user workflows

---

## 📞 Next Steps

**If selected for interview:**
- Can demo live system with real API calls
- Happy to discuss technical trade-offs
- Can present alternative architectures

**This submission includes:**
- ✅ Complete working code (Parts A & B)
- ✅ Detailed documentation
- ✅ Example query analysis with failure modes
- ✅ Evaluation framework design
- ✅ 2-week implementation plan
- ✅ Reflection on process and AI usage

---

**Date:** December 6, 2025

Thank you for this opportunity! Excited to potentially build AI systems for Kerala Ayurveda.

---

## 📄 License

Submitted as part of internship application. Kerala Ayurveda content remains property of Kerala Ayurveda Ltd.