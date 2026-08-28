"""
Kerala Ayurveda — Streamlit UI
================================

A thin HTTP client over the FastAPI backend.

WHY IT CHANGED:
This file used to import and run the whole RAG stack in-process: it loaded the
embedding model, opened the vector store, and called the agents directly. That
had three consequences.

  * Two copies of everything. The Streamlit path and the API path each had
    their own retrieval, chunking, and prompts, which drifted apart — they used
    different embedding models against the same collection.

  * Streamlit's execution model fought it. Every widget interaction re-runs the
    script top to bottom, so anything not wrapped in @st.cache_resource was
    rebuilt constantly, and a crashed cache meant re-embedding the corpus.

  * The article tab called the four agents directly instead of the orchestrator,
    so the fact-check retry loop never ran in the UI at all.

Now the UI renders and the backend computes. Article generation posts a job and
polls it, so a browser refresh no longer loses in-flight work.
"""

import os
import time

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
API = f"{API_BASE_URL}/api/v1"

REQUEST_TIMEOUT = 120       # generous: a cold query waits on the LLM
POLL_INTERVAL = 2.0
POLL_TIMEOUT = 600

st.set_page_config(
    page_title="Kerala Ayurveda AI",
    page_icon="🌿",
    layout="wide",
)


# ── API helpers ─────────────────────────────────────────────────

def api_get(path: str, **kwargs):
    r = requests.get(f"{API}{path}", timeout=REQUEST_TIMEOUT, **kwargs)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict):
    r = requests.post(f"{API}{path}", json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=30)
def backend_health() -> dict:
    """Backend status, cached briefly so reruns don't hammer it."""
    r = requests.get(f"{API_BASE_URL}/health/ready", timeout=10)
    r.raise_for_status()
    return r.json()


# ── Sidebar ─────────────────────────────────────────────────────

with st.sidebar:
    st.title("Kerala Ayurveda AI")
    st.markdown("*Agentic AI*")
    st.divider()

    st.header("Backend")
    st.caption(API_BASE_URL)

    try:
        health = backend_health()
        rag_stats = health.get("rag", {}) or {}
        st.success("Connected")

        chunks = rag_stats.get("chunks_indexed")
        if chunks is not None:
            st.metric("Indexed chunks", chunks)
        if version := rag_stats.get("index_version"):
            st.caption(f"Index version {version}")
        if model := rag_stats.get("embedding_model"):
            st.caption(f"Embeddings: `{model}`")

        cache_stats = (rag_stats.get("cache") or {}).get("memory", {})
        if cache_stats.get("hits") is not None:
            st.caption(
                f"Cache: {cache_stats['hits']} hits / "
                f"{cache_stats.get('misses', 0)} misses"
            )
        BACKEND_UP = True

    except Exception as e:
        BACKEND_UP = False
        st.error("Backend unreachable")
        st.caption(str(e)[:200])
        st.markdown(
            "Start it with:\n"
            "```bash\n"
            "docker compose -f docker-compose.dev.yml up\n"
            "```\n"
            "or set `API_BASE_URL` if it runs elsewhere."
        )

    st.divider()
    st.header("Knowledge Base")
    st.markdown("""
    | Document | Type |
    |---|---|
    | Ayurveda Foundations | Guide |
    | Dosha Guide (V/P/K) | Guide |
    | Style & Tone Guide | Guide |
    | Patient FAQs | FAQ |
    | Ashwagandha Tablets | Product |
    | Brahmi Tailam | Product |
    | Triphala Capsules | Product |
    | Stress Support Program | Treatment |
    | Products Catalog (8) | CSV |
    | Astanga Hridaya (Vagbhat, 24pp) | PDF Book |
    """)

if not BACKEND_UP:
    st.title("Kerala Ayurveda AI")
    st.warning("The backend API is not reachable. See the sidebar for details.")
    st.stop()


# ── Tabs ────────────────────────────────────────────────────────

tab_qa, tab_agent = st.tabs(["RAG Q&A", "Article Generator"])


# ══════════════════════════════════════════════════════════════
# TAB 1 — RAG Q&A
# ══════════════════════════════════════════════════════════════
with tab_qa:
    st.header("Ask the Knowledge Base")
    st.markdown(
        "Ask any question about Kerala Ayurveda — products, doshas, "
        "treatments, or wellness concepts."
    )

    if "qa_query" not in st.session_state:
        st.session_state.qa_query = ""

    col_input, col_examples = st.columns([3, 1])

    with col_input:
        query = st.text_input(
            "Your question:",
            value=st.session_state.qa_query,
            placeholder="e.g. What are the key benefits of Ashwagandha tablets?",
            label_visibility="collapsed",
        )
        search_btn = st.button("Get Answer", type="primary", use_container_width=True)

    with col_examples:
        st.markdown("**Try these:**")
        for ex in [
            "What are the benefits of Ashwagandha?",
            "Contraindications for Triphala?",
            "Can Ayurveda help with stress?",
            "What is Vata dosha?",
            "How does the Stress Support Program work?",
            "What is product KA-P001?",
        ]:
            if st.button(ex, key=f"ex_{ex[:24]}", use_container_width=True):
                st.session_state.qa_query = ex
                st.rerun()

    if search_btn and query:
        with st.spinner("Searching knowledge base..."):
            try:
                data = api_post("/query", {"query": query, "use_cache": True})

                st.markdown("### Answer")
                st.markdown(data["answer"])

                citations = data.get("citations") or []
                if citations:
                    st.markdown("### Sources")
                    for i, c in enumerate(citations, 1):
                        score = c.get("relevance_score", 0.0)
                        with st.expander(f"Source {i}: {c['doc_id']} — score {score:.2f}"):
                            st.markdown(f"**Section:** {c.get('section_id', '—')}")
                            st.text(c.get("content_snippet", ""))

                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Sources cited", len(citations))
                c2.metric("Latency", f"{data.get('latency_ms', 0):.0f} ms")
                c3.metric("Cache", "hit" if data.get("cache_hit") else "miss")

            except requests.HTTPError as e:
                st.error(f"API error: {e.response.status_code}")
                st.caption(e.response.text[:400])
            except Exception as e:
                st.error(f"Request failed: {e}")

    elif search_btn:
        st.warning("Please enter a question first.")


# ══════════════════════════════════════════════════════════════
# TAB 2 — ARTICLE GENERATOR
# ══════════════════════════════════════════════════════════════
with tab_agent:
    st.header("Multi-Agent Article Generator")
    st.markdown(
        "Submit a brief and the backend runs the agent graph: "
        "**outline → write sections (in parallel) → fact-check → revise → tone edit**. "
        "Usually three to four minutes — the job runs server-side, so you can "
        "leave this page and come back."
    )

    with st.form("article_brief"):
        topic = st.text_input(
            "Topic",
            value="Ayurvedic Support for Stress and Better Sleep",
        )
        audience = st.text_input(
            "Target audience",
            value="Busy professionals experiencing stress and sleep issues",
        )
        key_points_raw = st.text_area(
            "Key points (one per line)",
            value=(
                "How Ayurveda views stress and sleep\n"
                "Practical lifestyle approaches\n"
                "Herbs that support stress resilience\n"
                "Evening routines for better sleep"
            ),
            height=120,
        )
        col_a, col_b = st.columns(2)
        word_count = col_a.number_input(
            "Word count target", min_value=300, max_value=2000, value=800, step=100
        )
        products_raw = col_b.text_input(
            "Must include products (comma-separated)",
            value="Ashwagandha Stress Balance Tablets, Brahmi Tailam",
        )
        submitted = st.form_submit_button("Generate Article", type="primary")

    if submitted:
        key_points = [p.strip() for p in key_points_raw.splitlines() if p.strip()]
        products = [p.strip() for p in products_raw.split(",") if p.strip()]

        if not key_points:
            st.warning("Add at least one key point.")
            st.stop()

        try:
            job = api_post("/articles/generate", {
                "topic": topic,
                "target_audience": audience,
                "key_points": key_points,
                "word_count_target": int(word_count),
                "must_include_products": products,
            })
        except Exception as e:
            st.error(f"Could not start the job: {e}")
            st.stop()

        job_id = job["job_id"]
        st.caption(f"Job `{job_id}` started")

        progress = st.progress(0.0)
        status_line = st.empty()
        deadline = time.time() + POLL_TIMEOUT

        # The job runs server-side, so a refresh here doesn't kill it —
        # this loop only follows along.
        while time.time() < deadline:
            try:
                job = api_get(f"/articles/{job_id}")
            except Exception as e:
                status_line.warning(f"Polling error (retrying): {e}")
                time.sleep(POLL_INTERVAL)
                continue

            status = job.get("status", "unknown")
            step = job.get("current_step", 0)
            total = job.get("total_steps", 4) or 4

            progress.progress(min(step / total, 1.0))
            status_line.info(f"**{status.replace('_', ' ').title()}** — step {step}/{total}")

            if status in ("completed", "failed"):
                break
            time.sleep(POLL_INTERVAL)
        else:
            st.error("Timed out waiting for the job. It may still be running server-side.")
            st.stop()

        if job.get("status") == "failed":
            st.error(f"Generation failed: {job.get('error_message', 'unknown error')}")
            st.stop()

        progress.progress(1.0)
        status_line.success("Completed")

        c1, c2, c3 = st.columns(3)
        c1.metric("Grounding", f"{(job.get('fact_check_score') or 0):.2f}")
        c2.metric("Style", f"{(job.get('style_score') or 0):.2f}")
        c3.metric("Ready for editor", "Yes" if job.get("ready_for_editor") else "Needs review")

        if notes := job.get("editor_notes"):
            with st.expander(f"Editor notes ({len(notes)})", expanded=not job.get("ready_for_editor")):
                for n in notes:
                    st.markdown(f"- {n}")

        if outline := job.get("outline"):
            with st.expander("Outline"):
                st.markdown(f"**{outline.get('title', '')}**")
                for s in outline.get("sections", []):
                    st.markdown(f"- **{s.get('heading', '')}** — {s.get('key_points', '')}")

        st.divider()
        st.markdown("### Article")
        st.markdown(job.get("final_content") or "_No content returned._")

        if citations := job.get("citations"):
            with st.expander(f"Citations ({len(citations)})"):
                for c in citations:
                    st.markdown(f"- `{c.get('citation', '')}`")

        st.download_button(
            "Download as Markdown",
            data=job.get("final_content") or "",
            file_name=f"article_{job_id}.md",
            mime="text/markdown",
        )
