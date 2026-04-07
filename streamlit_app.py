"""
Streamlit Web UI for Kerala Ayurveda RAG System
Demo interface for the assignment submission
"""

import streamlit as st
import os
import json
from dotenv import load_dotenv
load_dotenv()

from src.rag_system import AyurvedaRAGSystem
from src.agent_workflow import ArticleWorkflowOrchestrator, ArticleBrief

# Page configuration
st.set_page_config(
    page_title="Kerala Ayurveda AI System",
    page_icon="🌿",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("Kerala Ayurveda AI")
    st.markdown("*Agentic AI*")
    st.divider()

    st.header("Knowledge Base")
    st.markdown("""
    **10 documents · 132 chunks**

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

    st.divider()

    # Key pool status
    st.header("API Key Pool")
    try:
        from src.key_manager import GeminiKeyManager
        km = GeminiKeyManager()
        status = km.status()
        st.markdown(f"**{status['total_keys']} key(s) loaded**")
        st.caption(f"Active: key {status['active_key_index']} of {status['total_keys']}")
    except Exception as e:
        st.warning(f"Key manager: {e}")

    st.divider()
    st.caption("Stack: Google Gemini 2.5 Flash · ChromaDB · LangChain · HuggingFace Embeddings")

    st.divider()
    if st.button("Clear Cache & Reload", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

# API Key check
if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GOOGLE_API_KEY_1"):
    st.error("GOOGLE_API_KEY not configured. Please add it in .env file.")
    st.stop()

# Shared RAG system (cached)
@st.cache_resource(show_spinner=False)
def load_rag_system():
    try:
        rag = AyurvedaRAGSystem()
        rag.load_and_index_content()
        return rag, None
    except Exception as e:
        error_str = str(e).lower()
        # Auto-recover from stale/locked ChromaDB files
        if "unable to open database" in error_str or "database error" in error_str:
            import shutil
            chroma_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "chroma_db"
            )
            if os.path.exists(chroma_path):
                shutil.rmtree(chroma_path)
            try:
                rag = AyurvedaRAGSystem()
                rag.load_and_index_content()
                return rag, None
            except Exception as e2:
                import traceback
                return None, traceback.format_exc()
        import traceback
        return None, traceback.format_exc()

with st.spinner("Loading Kerala Ayurveda knowledge base..."):
    rag, error = load_rag_system()

if rag is None:
    st.error("Failed to initialize RAG system.")
    if error:
        with st.expander("Show Error Details"):
            st.code(error, language="python")
    st.stop()

# Tabs
tab_qa, tab_agent = st.tabs(["RAG Q&A", "Article Generator"])


# ══════════════════════════════════════════════════════════════
# TAB 1 — RAG Q&A
# ══════════════════════════════════════════════════════════════
with tab_qa:
    st.header("Ask the Knowledge Base")
    st.markdown("Ask any question about Kerala Ayurveda — products, doshas, treatments, or wellness concepts.")

    col_input, col_examples = st.columns([3, 1])
    with col_input:
        query = st.text_input(
            "Your question:",
            placeholder="e.g. What are the key benefits of Ashwagandha tablets?",
            label_visibility="collapsed"
        )
        search_btn = st.button("Get Answer", type="primary", use_container_width=True)

    with col_examples:
        st.markdown("**Try these:**")
        examples = [
            "What are the benefits of Ashwagandha?",
            "Contraindications for Triphala?",
            "Can Ayurveda help with stress?",
            "What is Vata dosha?",
            "Tell me about the Stress Support Program",
            "What does Astanga Hridaya say about daily routine?",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
                query = ex
                search_btn = True

    if search_btn and query:
        with st.spinner("Searching knowledge base..."):
            try:
                response = rag.answer_user_query(query)

                st.markdown("### Answer")
                st.markdown(response.answer)

                st.markdown("### Sources")
                for i, citation in enumerate(response.citations, 1):
                    with st.expander(f"Source {i}: {citation.doc_id}  —  {citation.relevance_score:.1%} relevance"):
                        st.markdown(f"**Section:** {citation.section_id}")
                        st.text(citation.content_snippet)

                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Sources Retrieved", len(response.citations))
                c2.metric("Avg Relevance", f"{sum(c.relevance_score for c in response.citations) / len(response.citations):.1%}")
                c3.metric("Chunks Searched", len(response.retrieved_chunks))

            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)
    elif search_btn:
        st.warning("Please enter a question first.")


# ══════════════════════════════════════════════════════════════
# TAB 2 — AGENT WORKFLOW
# ══════════════════════════════════════════════════════════════
with tab_agent:
    st.header("Multi-Agent Article Generator")
    st.markdown(
        "Fill in an article brief and run the full **4-agent pipeline**: "
        "Outline → Write → Fact-Check → Tone Edit → Final Article."
    )

    with st.expander("How the pipeline works", expanded=False):
        st.markdown("""
        | Step | Agent | What it does |
        |------|-------|-------------|
        | 1 | **Outline Agent** | Queries the corpus, creates a structured section outline |
        | 2 | **Writer Agent** | Retrieves RAG context per section, writes the full draft with citations |
        | 3 | **Fact-Checker Agent** | Scores grounding (>=0.7 required), flags unsupported claims |
        | 4 | **Tone Editor Agent** | Checks brand voice, revises if needed — never removes citations |

        Takes approx. 2–4 minutes — each agent makes separate Gemini API calls.
        Keys rotate automatically if a quota limit is hit.
        """)

    # Article Brief Form
    st.subheader("Article Brief")

    with st.form("article_brief_form"):
        col1, col2 = st.columns(2)

        with col1:
            topic = st.text_input(
                "Topic *",
                value="Ayurvedic Support for Stress and Better Sleep",
                help="The main subject of the article"
            )
            target_audience = st.text_input(
                "Target Audience *",
                value="Busy professionals experiencing stress and sleep issues",
                help="Who is this article written for?"
            )
            word_count = st.slider(
                "Target Word Count",
                min_value=400,
                max_value=1200,
                value=800,
                step=100
            )

        with col2:
            key_points_raw = st.text_area(
                "Key Points to Cover *",
                value="How Ayurveda views stress and sleep\nPractical lifestyle approaches\nHerbs that support stress resilience\nEvening routines for better sleep",
                height=130,
                help="One point per line"
            )
            products_raw = st.text_input(
                "Products to Mention (optional)",
                value="Ashwagandha Stress Balance Tablets, Brahmi Tailam",
                help="Comma-separated product names from the catalog"
            )

        submitted = st.form_submit_button("Generate Article", type="primary", use_container_width=True)

    # Pipeline Execution
    if submitted:
        if not topic or not key_points_raw:
            st.error("Topic and Key Points are required.")
        else:
            key_points = [p.strip() for p in key_points_raw.strip().splitlines() if p.strip()]
            products = [p.strip() for p in products_raw.split(",") if p.strip()]

            brief = ArticleBrief(
                topic=topic,
                target_audience=target_audience,
                key_points=key_points,
                word_count_target=word_count,
                must_include_products=products
            )

            st.divider()
            st.subheader("Pipeline Progress")
            progress = st.progress(0, text="Starting pipeline...")
            status_box = st.empty()

            step_col1, step_col2, step_col3, step_col4 = st.columns(4)
            s1 = step_col1.empty()
            s2 = step_col2.empty()
            s3 = step_col3.empty()
            s4 = step_col4.empty()

            def update_step(n):
                labels = ["Outline", "Writer", "Fact-Check", "Tone Edit"]
                cols = [s1, s2, s3, s4]
                for i, (col, lbl) in enumerate(zip(cols, labels)):
                    if i < n:
                        col.success(f"{lbl} - Done")
                    elif i == n:
                        col.info(f"{lbl} - Running")
                    else:
                        col.empty()

            try:
                orchestrator = ArticleWorkflowOrchestrator(rag)

                update_step(0)
                status_box.info("**Step 1 / 4** — Outline Agent: verifying corpus coverage and building article structure...")
                progress.progress(10, text="Generating outline...")

                outline = orchestrator.outline_agent.generate_outline(brief)
                update_step(1)
                progress.progress(30, text="Outline complete — writing draft...")
                status_box.info("**Step 2 / 4** — Writer Agent: fetching RAG context per section and writing the full draft...")

                draft = orchestrator.writer_agent.write_draft(brief, outline)
                update_step(2)
                progress.progress(55, text="Draft complete — fact-checking...")
                status_box.info("**Step 3 / 4** — Fact-Checker Agent: verifying all claims are grounded in the knowledge base...")

                fact_check = orchestrator.fact_checker.fact_check(draft)
                update_step(3)
                progress.progress(80, text="Fact-check complete — editing tone...")
                status_box.info("**Step 4 / 4** — Tone Editor Agent: reviewing brand voice and style compliance...")

                tone = orchestrator.tone_editor.edit_tone(draft, fact_check)
                progress.progress(100, text="Done!")
                status_box.success("All agents complete!")

                final_content = tone.revised_content if tone.revised_content != "NO CHANGES" else draft.content
                ready = (
                    fact_check.grounding_score >= 0.7
                    and tone.style_score >= 0.7
                    and len(draft.citations) > 0
                )

                # Results
                st.divider()
                st.subheader("Generated Article")

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Fact-Check Score", f"{fact_check.grounding_score:.0%}",
                           delta="Pass" if fact_check.is_grounded else "Review needed",
                           delta_color="normal" if fact_check.is_grounded else "inverse")
                mc2.metric("Style Score", f"{tone.style_score:.0%}",
                           delta="Pass" if tone.style_score >= 0.7 else "Review needed",
                           delta_color="normal" if tone.style_score >= 0.7 else "inverse")
                mc3.metric("Citations Found", len(draft.citations))
                mc4.metric("Ready for Editor", "Yes" if ready else "Needs Review")

                st.markdown(final_content)

                with st.expander("Outline Used"):
                    st.markdown(f"**{outline.title}**")
                    for section in outline.sections:
                        st.markdown(f"- **{section['heading']}** — {section['key_points']}")

                with st.expander(f"Fact-Check Details  (score: {fact_check.grounding_score:.0%})"):
                    if fact_check.unsupported_claims:
                        st.warning(f"{len(fact_check.unsupported_claims)} unsupported claim(s) found:")
                        for claim in fact_check.unsupported_claims:
                            st.markdown(f"- {claim}")
                        if fact_check.suggested_fixes:
                            st.markdown("**Suggested fixes:**")
                            for fix in fact_check.suggested_fixes:
                                st.markdown(f"- **Claim:** {fix['claim']}")
                                st.markdown(f"  **Source:** `{fix['suggested_source']}`")
                    else:
                        st.success("All claims are grounded in the knowledge base.")

                with st.expander(f"Tone Edit Details  (score: {tone.style_score:.0%})"):
                    if tone.issues:
                        for issue in tone.issues:
                            st.markdown(f"- **{issue.get('location', 'General')}:** {issue.get('issue', '')} — *{issue.get('suggestion', '')}*")
                    else:
                        st.success("Brand voice and style are fully compliant.")

                st.divider()
                st.download_button(
                    label="Download Article (Markdown)",
                    data=final_content,
                    file_name=f"kerala_ayurveda_{topic[:30].replace(' ', '_').lower()}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

            except Exception as e:
                progress.empty()
                status_box.error(f"Pipeline failed: {e}")
                with st.expander("Error Details"):
                    st.exception(e)

# Footer
st.divider()
st.markdown(
    "<div style='text-align:center;color:gray;font-size:0.85em'>"
    "Kerala Ayurveda AI System · Agentic AI Internship Assignment · March 2026<br>"
    "Google Gemini 2.5 Flash · LangChain · ChromaDB · HuggingFace · Streamlit"
    "</div>",
    unsafe_allow_html=True
)
