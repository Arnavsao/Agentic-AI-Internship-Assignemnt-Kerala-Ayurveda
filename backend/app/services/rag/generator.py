"""
LLM Answer Generator — Context-Grounded Response Generation
==============================================================

WHY SEPARATE from retrieval:
The original rag_system.py mixes retrieval, prompt construction, and
LLM invocation in a single method. Separating them enables:
  1. Testing the generator with mock retrieval results
  2. Swapping the LLM without touching retrieval logic
  3. Different generation strategies (e.g., streaming vs. batch)
  4. Caching at the generation level independently of retrieval

This module is ONLY responsible for:
  - Taking retrieved chunks + query → building a prompt → calling LLM → formatting response

The retrieval logic lives in retriever.py.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate

from backend.app.core.config import get_settings
from backend.app.core.logging import LogTimer
from backend.app.services.llm import LLMProvider, response_text
from backend.app.services.rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Structured citation with provenance information."""
    doc_id: str
    section_id: str
    content_snippet: str
    relevance_score: float


@dataclass
class QueryResponse:
    """Complete response to a user query."""
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[str]
    model_used: str = ""
    cache_hit: bool = False
    latency_ms: float = 0.0


# ── System prompt ──
# Preserved from original with minor improvements for clarity
SYSTEM_PROMPT = """You are an expert assistant for Kerala Ayurveda. Answer questions using ONLY the provided context.

Style guidelines:
- Warm & reassuring, like a calm practitioner
- Grounded & precise — no vague claims
- Use phrases like "traditionally used to support...", "may help maintain..."
- NEVER claim to diagnose, treat, cure, or prevent diseases
- Always include gentle safety notes when relevant
- Encourage consultation with qualified practitioners

IMPORTANT:
- Only use information from the provided sources
- If the answer isn't in the sources, say so clearly
- Include specific citations in your answer using [Source X] notation
- Be concise but complete"""

USER_PROMPT = """Context from Kerala Ayurveda knowledge base:

{context}

Question: {query}

Please provide a helpful answer based on the context above. Include [Source X] citations in your response."""


def build_context(chunks: List[RetrievedChunk], use_parent: bool = True) -> str:
    """
    Build the context string from retrieved chunks.

    If use_parent=True and parent content is available, we use the parent
    chunk (richer context) but cite the child chunk (precise reference).
    """
    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        doc_id = chunk.doc_id
        section_id = chunk.section_id

        # Use parent content if available (richer context for LLM)
        content = chunk.parent_content if (use_parent and chunk.parent_content) else chunk.document.page_content

        context_parts.append(
            f"[Source {i}: {doc_id} - {section_id}]\n{content}\n"
        )

    return "\n---\n".join(context_parts)


def generate_answer(
    query: str,
    chunks: List[RetrievedChunk],
    llm_provider: LLMProvider,
    context_n: Optional[int] = None,
) -> QueryResponse:
    """
    Generate an answer from retrieved chunks using the LLM.

    Args:
        query: User's original question
        chunks: Retrieved and ranked chunks from the hybrid retriever
        llm_provider: LLM provider with key rotation
        context_n: How many chunks to include in context (default: from config)

    Returns:
        QueryResponse with answer, citations, and metadata
    """
    settings = get_settings()
    context_n = context_n or settings.retrieval_context_n

    # Use top-N chunks for context
    top_chunks = chunks[:context_n]

    # Build context string
    context = build_context(top_chunks, use_parent=True)

    # Create prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ])

    # Generate answer with key rotation
    with LogTimer(logger, "llm_generation", query=query[:100]):
        def create_llm(api_key):
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=settings.gemini_model,
                temperature=settings.llm_temperature,
                google_api_key=api_key,
            )

        def invoke(llm):
            chain = prompt_template | llm
            return chain.invoke({"context": context, "query": query})

        response = llm_provider.invoke_with_rotation(create_llm, invoke)

    answer = response_text(response)

    # Build citation objects
    citations = []
    for chunk in top_chunks:
        citation = Citation(
            doc_id=chunk.doc_id,
            section_id=chunk.section_id,
            content_snippet=chunk.document.page_content[:200] + "...",
            relevance_score=chunk.final_score,
        )
        citations.append(citation)

    # Collect all retrieved chunk texts for evaluation/debugging
    retrieved_texts = [c.document.page_content for c in chunks]

    return QueryResponse(
        answer=answer,
        citations=citations,
        retrieved_chunks=retrieved_texts,
        model_used=settings.gemini_model,
    )
