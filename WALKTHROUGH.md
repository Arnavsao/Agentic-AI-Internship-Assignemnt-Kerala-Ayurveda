# Kerala Ayurveda RAG — Complete Walkthrough & Interview Prep

> One document covering: how the system works end to end, everything that
> changed in the migration, what measurably improved, what did *not*, and how
> to talk about all of it under questioning.

---

## Table of Contents

1. [The 60-Second Summary](#1-the-60-second-summary)
2. [Architecture: Before and After](#2-architecture-before-and-after)
3. [The Three Flows, Step by Step](#3-the-three-flows-step-by-step)
4. [Every Change, and Why](#4-every-change-and-why)
5. [Measured Improvements](#5-measured-improvements)
6. [What Is Still Broken](#6-what-is-still-broken)
7. [Interview Prep: Core Questions](#7-interview-prep-core-questions)
8. [Interview Prep: Deep Dives](#8-interview-prep-deep-dives)
9. [The Bug Stories](#9-the-bug-stories-your-best-material)
10. [Weak Spots and How to Handle Them](#10-weak-spots-and-how-to-handle-them)
11. [Cheat Sheet](#11-cheat-sheet)

---

## 1. The 60-Second Summary

**What it is.** A Retrieval-Augmented Generation system over a Kerala Ayurveda
knowledge base (10 documents: 8 markdown, 1 CSV catalog, 1 24-page PDF), plus a
multi-agent pipeline that writes publication-ready articles grounded in that
corpus.

**What changed.** The system was migrated from a single-process Streamlit app
with an embedded ChromaDB to a service architecture: Qdrant for vectors,
FastAPI for the backend, LangGraph for the agents, Streamlit as a thin client.

**The headline numbers.**

| | Before | After |
|---|---|---|
| Coverage | 0.60 | **0.94** |
| Tone compliance | 0.60 | **1.00** |
| Hallucination rate | 0.80 | **0.50** (target 0.20 — still missed) |
| Golden set size | 5 | **18** (incl. 2 negative cases) |
| Tests | 24 | **151** |
| Corpus integrity | 130 chunks stored **twice** | 130, deduplicated |

**The one-line pitch.** *"I took a working prototype and made it correct under
production conditions — hybrid retrieval so exact terms stop getting missed,
content-addressed indexing so re-ingestion is idempotent, a fail-loud contract
on the vector space, and a real revision loop in the agent pipeline. Along the
way the evaluation harness turned out to be measuring itself, so I rewrote
three of the four scorers."*

---

## 2. Architecture: Before and After

### Before — one process, two competing stacks

```
  Streamlit app (single process)
     │
     ├── loads embedding model in-process
     ├── opens embedded ChromaDB (SQLite file)
     ├── runs retrieval
     └── calls 4 agents sequentially, blocking the request thread

  backend/ FastAPI (parallel reimplementation, incomplete)
     └── same ChromaDB directory, DIFFERENT embedding model
```

The critical flaw: **both stacks wrote to the same Chroma collection with
different embedding dimensions** (384-d MiniLM vs 768-d bge). Whichever built
the index first won; the other silently queried a foreign vector space.

### After — services with clear boundaries

```
  Streamlit UI  ──HTTP──►  FastAPI backend  ──►  Qdrant   (dense + sparse vectors)
  (no ML deps)                   │             ├─►  Redis    (response cache)
                                 │             └─►  SQLite   (jobs, documents, logs)
                                 │
                                 ├─ RAG pipeline    (retrieval → generation)
                                 └─ LangGraph agents (article generation as async jobs)
```

**Why services and not one process:**

- The embedded store held a SQLite file a second process couldn't safely open,
  which is why the API was pinned to `--workers 1`. That constraint is now
  only about model memory, not correctness.
- Streamlit re-runs its whole script on every widget interaction. Anything not
  wrapped in `@st.cache_resource` was rebuilt constantly, and a cache failure
  meant re-embedding the corpus.
- Two implementations of retrieval had already drifted apart. One is now the
  only one.

---

## 3. The Three Flows, Step by Step

### 3.1 Ingestion — `python -m scripts.ingest`

```
For each file in data/:
  1. SHA-256 the file bytes
  2. Compare against the manifest stored in Qdrant's metadata sidecar
     ├─ unchanged  → SKIP (no embedding, no network)
     └─ changed    → continue
  3. Parse       (markdown / pypdf / pandas)
  4. Chunk       adaptive size by doc type, parent + child levels
  5. Embed       dense (bge-base, 768-d) AND sparse (Qdrant/bm25)
  6. Point ID    uuid5(NAMESPACE, f"{doc_id}:{content_hash}")
  7. Upsert      → same content lands on the same ID, so this is idempotent
  8. Delete      any previous point IDs for this file that no longer appear
Finally:
  - files deleted from disk → purge their points by doc_id filter
  - bump index_version (invalidates the response cache)
```

**Adaptive chunk sizes** (a chunk size that suits an FAQ ruins a PDF):

| Doc type | Child | Parent | Why |
|---|---|---|---|
| FAQ | 400 | 1000 | Keep one Q&A pair together |
| Product | 500 | 1250 | Preserve product sections |
| Guide / PDF | 800 | 2000 | Conceptual content needs context |
| Default | 600 | 1500 | General articles |

**Parent-child**: retrieval matches the *small* chunk (precise), but the LLM
receives the *parent* (context). Parents are inlined into the child's payload
rather than stored as their own searchable points — they're only ever looked
up by ID, so giving them vectors would just add noise to retrieval.

### 3.2 Query — `POST /api/v1/query`

```
User query
  │
  ├─ 1. Cache check       key = f(index_version, embed_model, reranker, top_k/n, query)
  │                       HIT → return, skipping retrieval AND generation
  │
  ├─ 2. Embed the query   dense (with the bge instruction prefix)
  │                       sparse (BM25 term weights)
  │
  ├─ 3. Qdrant query_points
  │        prefetch: dense branch (top-10)  ─┐
  │                  sparse branch (top-10) ─┴─► RRF fusion, SERVER-SIDE
  │        returns top-10 fused                    ~5-15 ms
  │
  ├─ 4. Cross-encoder rerank  10 candidates → top-5        ~440 ms
  │
  ├─ 5. Take top-3, expand each to its parent chunk
  │
  ├─ 6. Build prompt: [Source i: doc_id - section_id] blocks
  │
  ├─ 7. Generate      MegaLLM → Gemini fallback           ~10-20 s
  │
  └─ 8. Cache + return QueryResponse(answer, citations, retrieved_chunks)
```

**Latency profile (measured):** retrieval ~490 ms total, of which the Qdrant
hybrid search is 5–15 ms and the cross-encoder is ~440 ms. Generation is
10–20 s. **The LLM dominates by a factor of ~30** — which is why optimising
retrieval further would be pointless and why caching and async matter.

### 3.3 Article generation — `POST /api/v1/articles/generate`

```
POST → job row written (status=queued), job_id returned IMMEDIATELY
        background task starts; client polls GET /api/v1/articles/{job_id}

   outline ── retrieves corpus evidence, plans only supported sections
      │
      ├─ Send ─→ write_section  ┐
      ├─ Send ─→ write_section  ├─ CONCURRENT, each retrieves its own evidence
      └─ Send ─→ write_section  ┘  (capped at 3 in flight by a semaphore)
                    │
                    ▼
             assemble_draft  ── stitch in order, extract citations
                    │
                    ▼
               fact_check ◄──────────────┐
                    │                    │
          grounded? │                    │  revised draft
             ┌──────┴───────┐            │
             │ no + budget  ├────────────┘
             │  remaining   │   revise ── rewrites flagged claims
             │              │
             │ yes / spent  │
             └──────┬───────┘
                    ▼
                tone_edit ── style guide alignment, citations protected
                    │
                   END
```

Every node completion writes progress to the job row, so polling reflects
real state and the work survives a browser refresh — or a server restart
(interrupted jobs are swept to `failed`).

---

## 4. Every Change, and Why

### A. Retrieval layer

| # | Change | The problem it solves |
|---|---|---|
| A1 | **Chroma → Qdrant** | Chroma has no sparse vectors, so keyword search was a Python BM25 index rebuilt at *every startup* by scanning the whole collection through a private API (`vectorstore._collection`) |
| A2 | **Server-side hybrid + RRF** | Dense retrieval alone misses exact terms — `KA-P001`, `Shirodhara`. Both branches now run in Qdrant and fuse internally |
| A3 | **Deterministic point IDs** | Indexing appended with random IDs, so re-running duplicated the corpus. IDs are now `uuid5(doc_id + content_hash)` |
| A4 | **Incremental ingestion** | Re-index was delete-all-and-rebuild. Now hash-diffed: unchanged files cost a file hash |
| A5 | **`assert_compatible()`** | `initialize()` reused any non-empty collection without checking which model built it — a 768-d model could query a 384-d index and return plausible nonsense |
| A6 | **bge query prefix** | `HuggingFaceEmbeddings` does *not* apply the bge instruction prefix despite a code comment claiming it did, so every query was embedded in the document distribution |
| A7 | **Parent chunks inlined** | Removes the in-memory parent map and its startup rebuild |
| A8 | **Cache key versioning** | Key was the query text alone, so re-indexing or changing models kept serving old answers. Now includes index version + model fingerprint |

### B. LLM access

| # | Change | The problem it solves |
|---|---|---|
| B1 | **Parameter pass-through** | `invoke_with_rotation` built the MegaLLM client with **no arguments**, silently discarding every caller's temperature and model. The agents' tuned temperatures only applied on the Gemini fallback path |
| B2 | **Lock-guarded rotation** | `_gemini_index` was mutated without a lock; concurrent requests could skip a key |
| B3 | **Async + `asyncio.sleep`** | Retry backoff used `time.sleep`, stalling the whole event loop |
| B4 | **Concurrency semaphore** | Nothing bounded in-flight calls; parallel section writes could burn the free tier's 15 RPM in one burst |

### C. Agent pipeline

| # | Change | The problem it solves |
|---|---|---|
| C1 | **LangGraph state machine** | Agents were a straight-line sequence of Python calls with no state object |
| C2 | **A real `revise` node** | The retry re-ran the fact-checker on the **identical draft** — nothing in between could change the text, so a rejected article could only be rejected again |
| C3 | **Counter fix** | The loop incremented twice per pass, so `max_iterations=2` bought exactly one extra check |
| C4 | **Parallel section writes** | Sections were written one at a time despite sharing no state |
| C5 | **Agents read raw chunks** | Every agent reached the corpus through `answer_user_query()` — a retrieval *plus a full LLM generation* — so the writer grounded citations on a summary of a summary |
| C6 | **Fact-check fails closed** | An unparseable response defaulted to `0.75 / grounded=True`, silently clearing the safety gate on medical content |
| C7 | **Citations protected** | A tone edit that drops citations is now rejected — that's the one rule the agent is told never to break |
| C8 | **Batched claim verification** | One LLM call per unsupported claim, unbounded → one batched call, capped at 8 |

### D. Jobs and API

| # | Change | The problem it solves |
|---|---|---|
| D1 | **Jobs in SQL** | State lived in a module-level dict: process-local, unbounded, erased on reload. Polling across a deploy 404'd on completed work |
| D2 | **Startup sweep** | A job interrupted by a restart sat at "writing" forever |
| D3 | **Injected pipeline** | The task constructed a *new* RAG system per job — reloading the model, opening a second store client — while ignoring the pipeline already injected into the route |
| D4 | **Async query path** | `query()` blocked the event loop for the entire LLM call inside an async route |

### E. Evaluation

| # | Change | The problem it solves |
|---|---|---|
| E1 | **Semantic coverage** | Was a substring test; a correct paraphrase scored zero |
| E2 | **Structured judge verdict** | Was `"YES" in response.upper()` — a judge replying *"Yes, this is well grounded"* scored as a hallucination |
| E3 | **Word-boundary tone check** | `"cure"` matched `"procedure"` and `"does not cure"` — the exact phrasing the style guide prescribes |
| E4 | **18-question golden set** | 5 questions is noise; no negative cases at all |

### F. Infrastructure

| # | Change |
|---|---|
| F1 | Qdrant + Redis as compose services; production compose brings up all four containers |
| F2 | Models baked into the API image at build time (offline start, deterministic) |
| F3 | Streamlit reduced to `streamlit` + `requests` — dropped ~2.5 GB of ML deps |
| F4 | `.env.example` documents every setting (it previously omitted `MEGA_API_KEY`, which the code required) |

---

## 5. Measured Improvements

### Evaluation (18-question golden set, live run)

| Metric | Before (5q, old scorers) | After (18q, fixed scorers) | Target | Status |
|---|---|---|---|---|
| Coverage | 0.60 | **0.94** | ≥ 0.60 | Exceeded |
| Citation accuracy | 1.00 | 0.92 | ≥ 0.50 | Exceeded |
| Hallucination rate | 0.80 | **0.50** | ≤ 0.20 | **Missed** |
| Tone compliance | 0.60 | **1.00** | ≥ 0.80 | Exceeded |
| Negative refusal | not tested | **1.00** | 1.00 | Met |
| Avg faithfulness | — | 0.81 | — | — |

> **Say this out loud in the interview:** these columns are *not* a controlled
> A/B. The golden set grew and three scorers were rewritten. Both are shown for
> orientation, and the honest framing is "the old numbers were partly measuring
> the metric code."

### Operational (verified on the running stack)

| Behaviour | Result |
|---|---|
| Startup ingest | 130 chunks, 10 files |
| **Re-ingest (unchanged)** | **0 upserted, 10 unchanged, 0.1 s** (vs full re-embed before) |
| Qdrant hybrid search | 5–15 ms |
| Cross-encoder rerank | ~440 ms |
| Total retrieval | ~490 ms |
| Generation | 10–20 s |
| Article pipeline | 3m40s, grounding **0.91**, style **0.95**, 1246 words, 42 citation markers |
| Job survives restart | Completed job retrievable with scores intact |
| Interrupted job | Swept to `failed`, not a 404 |
| Model mismatch | Fails loudly with an actionable message |

### Corpus integrity

The old index reported **264 chunks**. It actually held **130 unique chunks,
each stored twice** — every document's count is exactly double the current one
(PDF 150→75, FAQ 20→10, each catalog product 2→1). Indexing appended with
random IDs, so a second run duplicated the entire corpus rather than updating
it.

### Test coverage

| | Before | After |
|---|---|---|
| Tests | 24 (chunker only) | **151** |
| Unit | 24 | 138 (~9 s, no models loaded) |
| Integration | 0 | 13 (real models, ~1 min) |

---

## 6. What Is Still Broken

**Lead with this section if asked "what would you do next?" — owning the gaps
is stronger than being caught by them.**

### 6.1 Hallucination rate: 0.50 against a 0.20 target

Improved from 0.80, but unresolved. The judge's reasons are specific and
checkable — for q016 it flagged *"the four purposes of life (Dharma, Artha…)"*,
which is accurate Ayurvedic knowledge that simply is not in the retrieved
sources.

**The diagnosis matters:** coverage is 0.94 and citation accuracy 0.92, so
retrieval is finding and citing the right chunks. The model then *embellishes*
them from training knowledge. **This is a generation problem, not a retrieval
problem.**

I tried tightening the generation prompt to forbid outside knowledge. It
improved three of the six worst cases and worsened three others — net neutral
on faithfulness, though coverage rose to 100% on that subset. I kept the
clearer prompt but it did not move the metric, and I did not claim it did.

**What I would try next, in order:**
1. **A grounding pass over the draft answer** — the same shape as the article
   pipeline's `revise` node, which *does* have a real feedback loop. Correcting
   beats instructing.
2. **Sentence-level citation enforcement** — programmatically strip sentences
   carrying no `[Source X]` marker.
3. **Judge calibration** — the current judge counts any unsupported clause
   against the answer; some flagged items are benign connective phrasing.

### 6.2 Article latency did not improve

3m40s measured, versus 2–3 minutes before. Fewer LLM calls did **not** buy
wall-clock: each `gemini-3-pro-preview` call costs 20–60 s, and the concurrency
cap that keeps the free tier happy serialises parallel section writes into
batches. The rewrite bought grounding quality and cost, not latency. The lever
for latency is a faster model on the section-write node.

### 6.3 Smaller items

- `src/` (the legacy stack) is still in the tree. Nothing imports it; it's dead
  weight awaiting a delete commit.
- Only one Gemini key is configured, so `gemini_keys_total: 1` — the rotation
  path works but has nowhere to rotate to.
- Single uvicorn worker. Now a memory choice (each worker loads its own models),
  not a correctness constraint.
- Requirements are unpinned floors; no lockfile.
- No auth or rate limiting on the API — fine for an assignment, not for public
  deployment.

---

## 7. Interview Prep: Core Questions

### Q: Walk me through what happens when a user asks a question.

> Query comes in, hits the cache first — the key covers the index version and
> model fingerprint, not just the query text, so re-indexing can't serve stale
> answers. On a miss, the query gets embedded twice: dense with bge, and sparse
> as BM25 term weights. Both go to Qdrant in a single `query_points` call with
> two prefetch branches, and Qdrant fuses the rankings with Reciprocal Rank
> Fusion server-side — that's about 5–15 ms. The top 10 go to a cross-encoder
> for reranking, roughly 440 ms, and the top 3 survivors get expanded to their
> parent chunks so the LLM sees more context than the matched fragment.
> Then generation, 10–20 seconds, which dominates everything else.

### Q: Why Qdrant? Why not stay on Chroma, or use pgvector or Pinecone?

> Three reasons, in order of weight.
>
> **Sparse vectors.** Chroma has none, so keyword search had to be a Python
> BM25 index — and because it lived in the process, it was rebuilt at *every
> startup* by scanning the entire collection through a private API. Qdrant
> stores a sparse vector next to the dense one and fuses them internally, so
> startup touches nothing.
>
> **Idempotent writes.** Qdrant upserts by ID. Combined with content-derived
> IDs, re-ingestion becomes a no-op instead of a duplication event.
>
> **Operational shape.** As a separate service it lifts the single-worker
> constraint the embedded SQLite file imposed.
>
> pgvector would have been the pick if we already needed Postgres — one less
> service. Pinecone if we didn't want to run infrastructure at all. For this
> project, self-hosted and free mattered, and Qdrant's native hybrid search was
> the deciding feature.

### Q: Explain Reciprocal Rank Fusion.

> `score(doc) = Σ 1 / (k + rank_i(doc))`, where k is a smoothing constant
> (60 in the original paper; Qdrant applies its own internally — we don't
> pass one).
>
> The point is that it's **rank-based, not score-based**. Cosine similarity and
> BM25 produce numbers on completely different scales — you can't average them
> meaningfully, and normalising them requires assumptions that break as the
> corpus changes. RRF only looks at position. A document ranked well by both
> branches scores much higher than one ranked well by a single branch, and
> there are no weights to tune.
>
> Worth noting: the old config had `bm25_weight` and `semantic_weight`
> settings that were never read by any code path. I removed them rather than
> wire them up, because RRF doesn't take weights.

### Q: Why hybrid search at all? Isn't semantic retrieval better?

> Semantic is better for conceptual questions. It's actively bad at three
> things that matter in this corpus:
>
> - **Exact identifiers.** `KA-P001` — embeddings encode IDs poorly.
> - **Rare terms.** `Shirodhara`, `Abhyanga` — likely out-of-distribution for
>   the encoder.
> - **Safety keywords**, where a near-miss is worse than no answer.
>
> That's not hypothetical here. The `q005` benchmark question — "How does the
> Stress Support Program work?" — scored **0.00 coverage** because the answer
> never mentioned Abhyanga or Shirodhara. It now scores 1.00.

### Q: Why a cross-encoder after retrieval?

> The embedding model is a **bi-encoder**: query and document are encoded
> separately and compared as vectors. Fast, because you embed documents once —
> but it never sees the query and document together, so it can't reason about
> their interaction.
>
> A **cross-encoder** takes both as one input and can judge "does this passage
> about Ashwagandha actually answer the pregnancy-safety question?" It's
> roughly 100× slower per pair, so it only ever sees the ~10 fused candidates,
> never the corpus. Classic two-stage retrieval: cheap recall, expensive
> precision.

### Q: How does incremental indexing work, and why does it matter?

> Each file's SHA-256 is recorded in a manifest stored alongside the index.
> On ingest, unchanged files are skipped before any embedding happens. Changed
> files are re-chunked and upserted, and any point IDs that used to belong to
> that file but no longer appear are deleted. Files removed from disk have
> their points purged by a `doc_id` filter.
>
> The enabling trick is the **point ID**: `uuid5(namespace, doc_id + content_hash)`.
> Identical content always produces the same ID, so an upsert overwrites in
> place instead of appending.
>
> Measured: a re-ingest of an unchanged corpus is **0.1 seconds and zero
> writes**, versus a full re-embed before. At this corpus size that's
> convenience; at 100× it's the difference between a deploy step and an outage.

### Q: Why LangGraph instead of just calling the agents in sequence?

> The sequence was the bug. On a failed fact-check the old code called
> `fact_check` again on the *identical draft* — there was no step in between
> that could change the text, so a rejected article could only be rejected
> again. The source even carried a comment reading *"In production, would have
> revision agent here."*
>
> LangGraph gave me three things: a **state object** so nodes have a defined
> contract; **conditional edges** so the fact-check outcome routes to either
> `revise` or `tone_edit`; and the **Send API** so sections — which share no
> state — are written concurrently instead of in a loop.
>
> I deliberately **did not** add a checkpointer. It's six nodes and about a
> minute; job progress already persists in the database. A second persistence
> layer would be overhead. `SqliteSaver` is the documented upgrade path if
> mid-run resume is ever wanted.

### Q: How do you stop the system hallucinating?

> Layered, and I'll be upfront that the top layer isn't good enough yet.
>
> - **Retrieval quality** — hybrid search means the right evidence is actually
>   present. Coverage 0.94.
> - **Prompt constraints** — answer only from context, cite every claim.
> - **The fact-check node** scores grounding and rejects below 0.7.
> - **A revision node** rewrites flagged claims and re-checks.
> - **Failing closed** — an unparseable fact-check response used to default to
>   0.75 and "grounded", silently clearing the gate on *medical content*. It
>   now scores zero and flags for review.
> - **The tone editor can't drop citations** — if a revision reduces the
>   citation count, it's rejected.
>
> Measured hallucination rate is 0.50 against a 0.20 target. Better than the
> 0.80 it started at, but not solved — and it's a generation problem, not a
> retrieval one.

### Q: How do you evaluate this?

> An 18-question golden set covering every corpus document, plus multi-document
> synthesis, exact-ID lookup, the PDF, and — importantly — **two negative cases
> that must be refused**. A system that invents a price or claims a cure fails
> in a way no coverage score catches.
>
> Four metrics: semantic coverage, citation accuracy, LLM-judge faithfulness,
> and tone compliance. Everything but the judge runs locally and deterministically.
>
> The part worth telling: **three of the four original scorers were measuring
> themselves.** Coverage was a substring test, so correct paraphrase scored
> zero. The hallucination detector was `"YES" in response.upper()`, so a judge
> replying "Yes, this is well grounded" counted as a hallucination. The tone
> checker matched `"cure"` as a substring, so it fired on `"procedure"` and on
> `"does not cure"` — the exact phrasing the style guide *tells writers to use*.
> The benchmark numbers were partly artifacts of the metric code.

### Q: What are the trade-offs in your design?

| Decision | Gained | Gave up |
|---|---|---|
| Qdrant as a service | Hybrid search, multi-worker capability | One more container to run |
| Parent chunks inlined in payload | No startup rebuild, no second store | Text duplication (trivial at 600 KB, revisit at 100×) |
| `ms-marco-MiniLM` reranker | 50–100 ms on CPU | `bge-reranker-base` is more accurate but 0.5–1.5 s |
| `bge-base` over `BGE-M3` | 220 MB, fast on CPU | M3 is stronger and multilingual, but 2.2 GB and ~8× slower |
| No LangGraph checkpointer | Less machinery | No mid-run resume |
| BackgroundTasks over Celery | No worker container, no broker | Won't survive scaling past one box |
| Fixed the gateway, no LiteLLM | ~80 lines instead of a dependency | LiteLLM's provider breadth |

---

## 8. Interview Prep: Deep Dives

### 8.1 Parent-child chunking

**The tension.** Small chunks retrieve precisely — the vector matches the exact
sentence. But a 400-character fragment gives the LLM too little to work with.
Large chunks give context but dilute the embedding, so retrieval gets vaguer.

**The resolution.** Chunk twice. Index the small ones; return the large ones.
When a child matches, the LLM receives its parent.

**The implementation detail worth knowing:** parent-child linkage is a
positional heuristic — child index mapped proportionally onto the parent list.
It does not verify the child text is actually contained in that parent, so
drift is possible where the two splitters diverge. A real containment/offset
mapping would be the correct fix. *Volunteering this is a strength — it shows
you read your own code critically.*

### 8.2 The bge query instruction prefix

bge models are trained **asymmetrically**: queries carry the prefix
`"Represent this sentence for searching relevant passages: "` while documents
are embedded bare. This matters because queries and documents genuinely are
different kinds of text — a question and an answer shouldn't be embedded by the
same function if you want them to align.

The previous code used `langchain_huggingface.HuggingFaceEmbeddings` with a
comment claiming the prefix was "handled automatically." **It isn't** —
LangChain only applies it if you pass `query_instruction` explicitly. Every
query was being embedded in the document distribution. Wrapping
`SentenceTransformer` directly made the asymmetry visible and testable; there's
a unit test asserting `encode_query(t) != encode_documents([t])[0]`.

### 8.3 Why the cache key changed

The old key was `sha256(query)`. That's wrong the moment anything else changes:
re-index the corpus, or swap the embedding model, and the cache happily serves
answers built from the old configuration.

The key now covers index version, embedding model, reranker model, and the
retrieval depths. Because the version is *in the key*, re-indexing orphans old
entries automatically — they age out via TTL, and there's no invalidation
sweep to get wrong. (There's still a SCAN-based purge for the admin path; the
original used `KEYS`, which blocks Redis across a full keyspace walk.)

### 8.4 The concurrency semaphore

Gemini's free tier allows 15 requests/minute. Once section writes fan out in
parallel, nothing structurally prevents six simultaneous calls, and a burst
exhausts the quota for everyone including the fact-checker later in the same
run.

`asyncio.Semaphore(llm_max_concurrency)`, default 3. It's created lazily and
rebound per event loop, because a semaphore binds to the loop that created it.

**The honest cost:** this is also why article generation didn't get faster.
The cap serialises six parallel sections into two batches. Concurrency and
rate-limit compliance are in direct tension here, and I chose compliance.

### 8.5 Fail-loud versus fail-silent

This is the theme worth naming explicitly if you get a "what's your philosophy"
question.

Three of the bugs shared one shape: **the failure produced plausible output
instead of an error.**

| Failure | Old behaviour | New behaviour |
|---|---|---|
| Model/dimension mismatch | Served nonsense from a foreign vector space | `IndexCompatibilityError` at startup, naming the fix |
| Unparseable fact-check | Scored 0.75, `grounded=True` | Scores 0.0, flags for review |
| Duplicate ingestion | Silently doubled the corpus | Upsert in place |

A crash is a *good* outcome compared to confidently wrong medical content. The
guiding question: **if this breaks, will anyone find out?**

### 8.6 Why agents reading raw chunks matters

Every agent used to reach the corpus through `answer_user_query()` — which is
retrieval **plus a full LLM generation**. Two consequences:

1. **Cost.** Each "lookup" was an LLM round trip. That's most of the 11–15
   sequential calls in the original pipeline.
2. **Grounding.** The agent received a *paraphrase*, not source text. The
   writer then produced citations against a summary of a summary. Every hop
   is a chance to drift.

Nodes now call `retriever.retrieve()` and read chunks directly. The text an
agent cites is the text actually in the knowledge base.

---

## 9. The Bug Stories (Your Best Material)

Interviewers remember specifics. Each of these is a complete story with a
diagnosis and a fix.

### The corpus was stored twice

**Symptom:** documentation said 264 chunks; the rebuilt index had 130.

**Investigation:** queried the old Chroma SQLite directly and grouped by
`doc_id`. Every single document was exactly double: PDF 150→75, FAQ 20→10,
each catalog product 2→1.

**Root cause:** `Chroma.from_documents` assigns random UUIDs. Running ingestion
twice appends the whole corpus again. A `content_hash` field existed in the
metadata and a docstring claimed deduplication — but nothing ever read it.

**Fix:** content-derived point IDs make upserts idempotent.

**The lesson:** a metric in the README ("264 chunks") had been quoted for
months and was simply wrong. Numbers in docs need a source.

### The temperature that never arrived

**Symptom:** none. That's what makes it interesting.

**Investigation:** reading `invoke_with_rotation`, the MegaLLM branch called
`self.create_mega_llm()` — no arguments. The Gemini branch called
`create_llm_fn(key)`, which *did* carry the caller's settings.

**Root cause:** MegaLLM was the default provider. So every carefully chosen
temperature — 0.0 for the fact-checker, 0.3 for the outline agent — was
discarded on the primary path and only applied on fallback.

**Fix:** a `generate()` API that threads parameters to whichever provider wins,
plus explicit `llm_kwargs` on the lower-level call. There's a regression test
asserting the temperature reaches the client.

**The lesson:** a bug that changes output quality but never raises is the
hardest kind to notice. Tests that assert on *arguments*, not just results.

### The metric that graded itself

**Symptom:** `q005` scored 0.00 coverage, and the report's own analysis said
the answer was "correct but overly general."

**Investigation:** `evaluate_coverage` was a case-insensitive substring check
against expected phrases. The answer said "practitioner consultation"; the
expected literal was "consultation" — that one passed. But paraphrases of
"Abhyanga" and "Shirodhara" couldn't possibly match, because those are proper
nouns the answer never reached.

Two independent causes, and this is the part worth telling: **retrieval really
had missed those chunks** (rare terms, dense-only search) **and** the metric
would have scored a correct paraphrase zero regardless. Fixing either alone
would have left the other hidden.

**Fix:** hybrid retrieval for the first, embedding-similarity coverage for the
second. q005 now scores 1.00 and the retrieved context contains both terms.

### The volume Docker owned

**Symptom:** API container crash-looped on first boot —
`sqlite3.OperationalError: unable to open database file`.

**Investigation:** `DATABASE_URL` pointed into `/app/var`, which the `api_data`
volume mounts over. That path didn't exist in the image.

**Root cause:** when a named volume's mount point doesn't exist in the image,
Docker creates it **root-owned** — and the container runs as `appuser`.

**Fix:** `mkdir -p /app/var && chown appuser` in the Dockerfile. A fresh volume
inherits the ownership of the path it mounts over. Placed *after* the
model-download layer so editing it doesn't invalidate a 10-minute cache step.

**The lesson:** no unit test reaches this. It only appears when you actually
run the thing. Which is why "it passes CI" isn't "it works."

---

## 10. Weak Spots and How to Handle Them

### "Your hallucination rate misses the target."

Don't get defensive; you diagnosed it precisely.

> Correct — 0.50 against 0.20. It came down from 0.80, but it's not solved.
> What I can tell you is *where* the problem is: coverage is 0.94 and citation
> accuracy 0.92, so retrieval is finding and citing the right material. The
> model then adds accurate Ayurvedic knowledge that isn't in the sources. It's
> a generation problem. I tried prompt tightening — it helped three cases and
> hurt three, net neutral, and I didn't ship it as a fix because it wasn't one.
> The approach I'd take next is a grounding pass over the draft answer, the
> same shape as the revise node in the article pipeline, because correcting
> works better than instructing.

### "You claimed the pipeline got faster and it didn't."

> I did, and the measured run corrected me — 3m40s, essentially unchanged. I
> updated the docs and the code comments rather than leave the claim standing.
> Fewer LLM calls didn't buy wall-clock because each call to a reasoning-tier
> model is 20–60 seconds and my own rate-limit semaphore serialises the
> parallel writes. What the rewrite actually bought was grounding quality and
> cost. The lever for latency would be a cheaper model on the section-write
> node.

### "Why didn't you use Ragas / LiteLLM / Celery?"

> Each was a real consideration.
>
> **Ragas** — I implemented faithfulness with the same LLM-judge approach it
> uses, through the existing gateway. Adding it would have meant a heavy
> dependency whose judge calls hit the same free-tier quota anyway, for metrics
> I could compute directly. At a larger scale I'd take the validated
> implementation over mine.
>
> **LiteLLM** — two providers, one already OpenAI-compatible. Free-tier key
> rotation would still be custom logic layered on its router. It was about 80
> lines to fix properly.
>
> **Celery** — single box, one worker, low traffic. It adds a broker and a
> worker container for no benefit at this scale. The failure mode it protects
> against (losing jobs) I handled by persisting job state to the database
> instead. `arq` is the noted upgrade path.

### "Is 18 questions enough to evaluate on?"

> No. It's enough to catch regressions and to cover every document, which is
> what it's for. Under about 100 you're largely measuring noise on the
> aggregate. I'd also flag that these are single-run numbers — both generation
> and judging vary between runs, so a ±0.1 difference on one question isn't
> signal. Growing the set from real user queries in the query log would be the
> right next step; that table already exists.

### "What if the corpus grows 1000×?"

> Several things change, in this order:
>
> - **Parent chunks inlined in payloads** stop being free. They'd move to the
>   SQL chunk table, fetched by ID after reranking.
> - **Ingestion** would need to parallelise and probably move off the API
>   process entirely.
> - **Payload indexes** on `doc_id`/`doc_type` start mattering for filtered
>   search; they're already created.
> - **Qdrant** handles the vector side to tens of millions before sharding is
>   the conversation.
> - **The reranker** becomes the bottleneck if `top_k` grows; it's fixed at 10
>   candidates so it actually scales fine.
>
> The design decision that ages best is content-addressed IDs — incremental
> ingestion is what makes a large corpus maintainable at all.

---

## 11. Cheat Sheet

### Numbers to have ready

| | |
|---|---|
| Corpus | 10 documents, 130 chunks |
| Dense model | `BAAI/bge-base-en-v1.5`, 768-d, ~220 MB |
| Sparse model | `Qdrant/bm25` via FastEmbed |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Funnel | top-10 fused → rerank 5 → 3 in prompt |
| Qdrant search | 5–15 ms |
| Rerank | ~440 ms |
| Generation | 10–20 s |
| Re-ingest (no changes) | 0.1 s, zero writes |
| Article run | 3m40s, grounding 0.91, style 0.95 |
| Coverage / citations / tone | 0.94 / 0.92 / 1.00 |
| Hallucination | 0.50 (target 0.20) |
| Tests | 151 (138 unit + 13 integration) |

### Commands

```bash
docker compose up --build -d                                 # full stack
docker compose logs -f api                                   # follow logs
docker compose exec api python -m scripts.ingest --status    # index state
docker compose exec api python -m scripts.ingest             # incremental
docker compose exec api python -m scripts.ingest --rebuild   # full re-embed
docker compose exec api python -m scripts.evaluate           # benchmark
./venv/bin/python -m pytest tests/ -m "not integration"      # 138 tests, ~9s
```

- UI http://localhost:8501 · API docs http://localhost:8000/docs · Qdrant http://localhost:6333/dashboard

### Where things live

| Concern | File |
|---|---|
| Qdrant adapter | `backend/app/services/rag/vectorstore.py` |
| Incremental ingest | `backend/app/services/ingestion/service.py` |
| Hybrid retrieval | `backend/app/services/rag/retriever.py` |
| Chunking | `backend/app/services/rag/chunker.py` |
| LLM gateway | `backend/app/services/llm.py` |
| Agent graph | `backend/app/services/agents/graph.py` |
| Agent nodes | `backend/app/services/agents/nodes.py` |
| Job routes | `backend/app/api/routes/articles.py` |
| Metric scorers | `backend/app/services/evaluation.py` |

### Three sentences that carry the whole interview

1. *"The old index reported 264 chunks but actually held 130 stored twice —
   indexing appended with random IDs, so I made point IDs content-derived and
   re-ingestion idempotent."*
2. *"Three of the four evaluation scorers were measuring their own
   implementation — coverage was a substring test, so a correct paraphrase
   scored zero."*
3. *"Retrieval is strong at 0.94 coverage; the remaining hallucination is the
   model embellishing correct sources, which is a generation problem, and I
   didn't ship the prompt fix because it didn't actually work."*

---

*Companion to `README.md`. The README documents the system as it stands; this
document explains how it got there and what to say about it.*
