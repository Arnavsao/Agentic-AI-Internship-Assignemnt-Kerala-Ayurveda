"""
Streamlit Web UI for Kerala Ayurveda RAG System
Demo interface for the assignment submission
"""

import streamlit as st
import os
from src.rag_system import AyurvedaRAGSystem

# Page configuration
st.set_page_config(
    page_title="Kerala Ayurveda RAG Assistant",
    page_icon="🌿",
    layout="wide"
)

# Title and description
st.title("🌿 Kerala Ayurveda RAG Assistant")
st.markdown("**AI-powered Q&A system for Kerala Ayurveda content**")
st.markdown("*Assignment submission for Agentic AI Internship*")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This RAG system demonstrates:
    - **Adaptive chunking** (400-800 chars)
    - **Semantic retrieval** with embeddings
    - **Structured citations** (doc_id + section_id)
    - **Kerala Ayurveda brand voice**

    Built with:
    - LangChain
    - ChromaDB
    - Google Gemini API
    """)

    st.header("Example Questions")
    st.markdown("""
    - What are the benefits of Ashwagandha?
    - Are there contraindications for Triphala?
    - Can Ayurveda help with stress?
    - What is Vata dosha?
    - Tell me about the Stress Support Program
    """)

# Check for API key
if not os.getenv("GOOGLE_API_KEY"):
    st.error("⚠️ GOOGLE_API_KEY not configured. Please add it in Streamlit Cloud settings.")
    st.info("Go to: App settings → Secrets → Add GOOGLE_API_KEY")
    st.stop()

# Initialize RAG system with caching
@st.cache_resource(show_spinner=False)
def load_rag_system():
    """Load and cache the RAG system"""
    try:
        rag = AyurvedaRAGSystem()
        rag.load_and_index_content()
        return rag, None
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return None, error_details

with st.spinner("Loading Kerala Ayurveda knowledge base..."):
    rag, error = load_rag_system()

if rag is None:
    st.error("Failed to initialize RAG system.")
    if error:
        with st.expander("Show Error Details"):
            st.code(error, language="python")
    st.info("💡 Try: Clear cache in Streamlit Cloud and redeploy")
    st.stop()

# Main interface
st.markdown("---")
st.subheader("Ask a Question")

# Query input
query = st.text_input(
    "Enter your question about Ayurveda:",
    placeholder="e.g., What are the key benefits of Ashwagandha tablets?",
    help="Ask about products, treatments, doshas, or Ayurvedic concepts"
)

# Search button
if st.button("Get Answer", type="primary") or query:
    if query:
        with st.spinner("Searching knowledge base..."):
            try:
                # Get response
                response = rag.answer_user_query(query)

                # Display answer
                st.markdown("### Answer")
                st.markdown(response.answer)

                # Display citations
                st.markdown("### 📚 Sources")
                for i, citation in enumerate(response.citations, 1):
                    with st.expander(f"Source {i}: {citation.doc_id}"):
                        st.markdown(f"**Section:** {citation.section_id}")
                        st.markdown(f"**Relevance:** {citation.relevance_score:.2%}")
                        st.markdown(f"**Snippet:**")
                        st.text(citation.content_snippet)

                # Show retrieval stats
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sources Retrieved", len(response.citations))
                with col2:
                    st.metric("Avg Relevance", f"{sum(c.relevance_score for c in response.citations) / len(response.citations):.1%}")
                with col3:
                    st.metric("Chunks Searched", len(response.retrieved_chunks))

            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                st.exception(e)
    else:
        st.warning("Please enter a question.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>Kerala Ayurveda RAG System | Assignment Submission | December 2025</p>
    <p>Built with adaptive chunking, semantic retrieval, and fact-checking guardrails</p>
</div>
""", unsafe_allow_html=True)
