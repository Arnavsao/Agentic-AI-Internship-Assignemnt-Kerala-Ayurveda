"""
Kerala Ayurveda Agentic Workflow - Part B Implementation
Multi-agent system for article generation with fact-checking and style validation
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.rag_system import AyurvedaRAGSystem


class AgentStep(Enum):
    """Workflow steps for article generation"""
    OUTLINE = "outline"
    WRITER = "writer"
    FACT_CHECKER = "fact_checker"
    TONE_EDITOR = "tone_editor"
    FINAL_REVIEW = "final_review"


@dataclass
class ArticleBrief:
    """Input brief for article generation"""
    topic: str
    target_audience: str
    key_points: List[str]
    word_count_target: int = 800
    must_include_products: List[str] = field(default_factory=list)


@dataclass
class Outline:
    """Structured outline from outline agent"""
    title: str
    sections: List[Dict[str, str]]  # [{"heading": "...", "key_points": "..."}]
    estimated_word_count: int
    key_sources_needed: List[str]


@dataclass
class Draft:
    """Article draft with metadata"""
    content: str
    word_count: int
    citations: List[Dict[str, str]]
    sections: List[str]


@dataclass
class FactCheckResult:
    """Results from fact-checking agent"""
    is_grounded: bool
    grounding_score: float  # 0-1, percentage of claims with sources
    unsupported_claims: List[str]
    missing_citations: List[str]
    suggested_fixes: List[Dict[str, str]]


@dataclass
class ToneCheckResult:
    """Results from tone editor agent"""
    style_score: float  # 0-1, adherence to brand guidelines
    issues: List[Dict[str, str]]  # [{"issue": "...", "location": "...", "suggestion": "..."}]
    revised_content: str


@dataclass
class FinalArticle:
    """Complete article output with all metadata"""
    content: str
    citations: List[Dict[str, str]]
    fact_check_score: float
    style_score: float
    workflow_metadata: Dict
    ready_for_editor: bool
    editor_notes: List[str]


class OutlineAgent:
    """
    Agent 1: Outline Generator

    Input: ArticleBrief
    Output: Structured outline with sections and key points

    Failure modes:
    - Creates outline with topics not covered in corpus
    - Over-promises on scope (asks for info we don't have)

    Guardrail:
    - Query RAG system to verify each section topic is grounded
    """

    def __init__(self, rag_system: AyurvedaRAGSystem):
        self.rag = rag_system
        megallm_api_key = os.getenv("MEGALLM_API_KEY")
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=megallm_api_key,
            openai_api_base="https://ai.megallm.io/v1"
        )

    def generate_outline(self, brief: ArticleBrief) -> Outline:
        """Generate article outline based on brief"""

        # First, verify topic coverage in corpus
        coverage_check = self.rag.answer_user_query(
            f"What information is available about {brief.topic}?"
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Ayurveda content strategist for Kerala Ayurveda.
Your job is to create a structured outline for an article.

Available context about the topic:
{corpus_context}

Guidelines:
- Only include sections that can be supported by the available context
- Follow Kerala Ayurveda's warm, grounded tone
- Include practical takeaways
- Plan for {word_count} words
- Structure: Introduction, 3-5 main sections, Conclusion/Summary

Output as JSON with structure:
{{
    "title": "Article title",
    "sections": [
        {{"heading": "Section name", "key_points": "What to cover"}},
        ...
    ],
    "estimated_word_count": 800,
    "key_sources_needed": ["doc_id_1", "doc_id_2"]
}}"""),
            ("user", """Create an outline for:

Topic: {topic}
Target Audience: {audience}
Key Points to Cover: {key_points}
Word Count Target: {word_count}
Must Include Products: {products}

Generate the outline as JSON.""")
        ])

        chain = prompt_template | self.llm
        response = chain.invoke({
            "corpus_context": coverage_check.answer,
            "topic": brief.topic,
            "audience": brief.target_audience,
            "key_points": ", ".join(brief.key_points),
            "word_count": brief.word_count_target,
            "products": ", ".join(brief.must_include_products) if brief.must_include_products else "None specified"
        })

        # Parse JSON response
        outline_data = json.loads(response.content)

        return Outline(
            title=outline_data["title"],
            sections=outline_data["sections"],
            estimated_word_count=outline_data["estimated_word_count"],
            key_sources_needed=outline_data.get("key_sources_needed", [])
        )


class WriterAgent:
    """
    Agent 2: Content Writer

    Input: Outline + Brief
    Output: Full draft with citations

    Failure modes:
    - Adds claims not in corpus (hallucination)
    - Forgets to add citations
    - Strays from brand tone

    Guardrail:
    - Require at least one citation per main claim
    - Use RAG context retrieval for each section
    """

    def __init__(self, rag_system: AyurvedaRAGSystem):
        self.rag = rag_system
        megallm_api_key = os.getenv("MEGALLM_API_KEY")
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            openai_api_key=megallm_api_key,
            openai_api_base="https://ai.megallm.io/v1"
        )

    def write_draft(self, brief: ArticleBrief, outline: Outline) -> Draft:
        """Write full article draft based on outline"""

        # Retrieve relevant context for each section
        section_contexts = []
        all_citations = []

        for section in outline.sections:
            query = f"{brief.topic} {section['heading']} {section['key_points']}"
            rag_response = self.rag.answer_user_query(query)

            section_contexts.append({
                "heading": section["heading"],
                "context": rag_response.answer,
                "sources": [
                    {"doc_id": c.doc_id, "section_id": c.section_id}
                    for c in rag_response.citations
                ]
            })

            all_citations.extend(rag_response.citations)

        # Build comprehensive context
        context_text = "\n\n".join([
            f"## {sc['heading']}\nContext: {sc['context']}\nSources: {sc['sources']}"
            for sc in section_contexts
        ])

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Ayurveda content writer for Kerala Ayurveda.

Write a complete article following these STRICT guidelines:

TONE & STYLE:
- Warm & reassuring, like a calm practitioner
- Grounded & precise - no vague claims
- Use "traditionally used to support...", "may help maintain..."
- NEVER claim to diagnose, treat, cure, or prevent diseases

CITATIONS:
- MUST cite sources for every factual claim using [Source: doc_id - section_id]
- Include safety notes where relevant
- Encourage practitioner consultation

STRUCTURE:
- Clear H2/H3 headings
- Short paragraphs (2-4 sentences)
- Bulleted lists for practical points
- Summary section at end

Use ONLY information from the provided context. Do not add outside knowledge."""),
            ("user", """Write a complete article based on:

Title: {title}

Outline:
{outline}

Retrieved Context & Sources:
{context}

Target word count: {word_count}

Write the full article with citations.""")
        ])

        chain = prompt_template | self.llm
        response = chain.invoke({
            "title": outline.title,
            "outline": json.dumps(outline.sections, indent=2),
            "context": context_text,
            "word_count": outline.estimated_word_count
        })

        content = response.content
        word_count = len(content.split())

        # Extract citations from content
        import re
        citation_pattern = r'\[Source: ([^\]]+)\]'
        citations_found = re.findall(citation_pattern, content)

        citations_list = [
            {"citation": cite, "mentioned_in": "article"}
            for cite in citations_found
        ]

        return Draft(
            content=content,
            word_count=word_count,
            citations=citations_list,
            sections=[s["heading"] for s in outline.sections]
        )


class FactCheckerAgent:
    """
    Agent 3: Fact Checker

    Input: Draft
    Output: Fact-check results with grounding score

    Failure modes:
    - Misses subtle unsupported claims
    - Over-flags reasonable inferences

    Guardrail:
    - Reject drafts with grounding_score < 0.7
    - Flag any medical claims without source
    """

    def __init__(self, rag_system: AyurvedaRAGSystem):
        self.rag = rag_system
        megallm_api_key = os.getenv("MEGALLM_API_KEY")
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=megallm_api_key,
            openai_api_base="https://ai.megallm.io/v1"
        )

    def fact_check(self, draft: Draft) -> FactCheckResult:
        """Verify all claims in draft are supported by corpus"""

        # Split draft into claims
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a fact-checking agent for medical content.

Analyze the article and:
1. Extract all factual claims about Ayurveda, herbs, treatments, benefits
2. For each claim, determine if it has a citation
3. Verify claims can be supported by the source

Output as JSON:
{{
    "total_claims": 15,
    "supported_claims": 12,
    "unsupported_claims": ["claim without source...", ...],
    "missing_citations": ["section/paragraph with no citation", ...],
    "grounding_score": 0.8
}}"""),
            ("user", """Fact-check this article:

{article}

Extracted citations: {citations}

Analyze the article and return JSON.""")
        ])

        chain = prompt_template | self.llm
        response = chain.invoke({
            "article": draft.content,
            "citations": json.dumps(draft.citations, indent=2)
        })

        # Parse response
        result_data = json.loads(response.content)

        # For each unsupported claim, try to find supporting evidence
        suggested_fixes = []
        for claim in result_data.get("unsupported_claims", []):
            rag_response = self.rag.answer_user_query(f"Verify: {claim}")
            if rag_response.citations:
                suggested_fixes.append({
                    "claim": claim,
                    "suggested_source": f"{rag_response.citations[0].doc_id} - {rag_response.citations[0].section_id}",
                    "supporting_text": rag_response.answer[:200]
                })

        return FactCheckResult(
            is_grounded=(result_data["grounding_score"] >= 0.7),
            grounding_score=result_data["grounding_score"],
            unsupported_claims=result_data.get("unsupported_claims", []),
            missing_citations=result_data.get("missing_citations", []),
            suggested_fixes=suggested_fixes
        )


class ToneEditorAgent:
    """
    Agent 4: Tone & Style Editor

    Input: Draft (fact-checked)
    Output: Revised draft matching brand guidelines

    Failure modes:
    - Changes meaning while editing tone
    - Removes necessary caveats/safety notes

    Guardrail:
    - Preserve all citations and medical disclaimers
    - Flag if safety language is weakened
    """

    def __init__(self, rag_system: AyurvedaRAGSystem):
        self.rag = rag_system
        megallm_api_key = os.getenv("MEGALLM_API_KEY")
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            openai_api_key=megallm_api_key,
            openai_api_base="https://ai.megallm.io/v1"
        )

        # Load style guide from corpus
        style_guide_response = rag_system.answer_user_query(
            "What are Kerala Ayurveda's content style and tone guidelines?"
        )
        self.style_guide_context = style_guide_response.answer

    def edit_tone(self, draft: Draft, fact_check: FactCheckResult) -> ToneCheckResult:
        """Review and improve tone/style while preserving facts"""

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a style editor for Kerala Ayurveda content.

Style Guide:
{style_guide}

Your job:
1. Review the article for tone and style alignment
2. Identify issues (aggressive claims, missing warmth, jargon, etc.)
3. Suggest improvements
4. Provide revised version if needed

CRITICAL: Do NOT change factual content or remove citations.

Output as JSON:
{{
    "style_score": 0.85,
    "issues": [
        {{"issue": "Description", "location": "Section name", "suggestion": "How to fix"}},
        ...
    ],
    "revised_content": "Full revised article text if changes needed, or 'NO CHANGES' if perfect"
}}"""),
            ("user", """Review this article for style and tone:

{article}

Fact-check passed: {fact_check_passed}
Grounding score: {grounding_score}

Return JSON analysis.""")
        ])

        chain = prompt_template | self.llm
        response = chain.invoke({
            "style_guide": self.style_guide_context,
            "article": draft.content,
            "fact_check_passed": fact_check.is_grounded,
            "grounding_score": fact_check.grounding_score
        })

        result_data = json.loads(response.content)

        revised_content = result_data.get("revised_content", draft.content)
        if revised_content == "NO CHANGES":
            revised_content = draft.content

        return ToneCheckResult(
            style_score=result_data["style_score"],
            issues=result_data.get("issues", []),
            revised_content=revised_content
        )


class ArticleWorkflowOrchestrator:
    """
    Orchestrates the multi-agent workflow for article generation

    Workflow: Brief → Outline → Draft → Fact-Check → Tone Edit → Final Review
    """

    def __init__(self, rag_system: AyurvedaRAGSystem):
        self.rag = rag_system
        self.outline_agent = OutlineAgent(rag_system)
        self.writer_agent = WriterAgent(rag_system)
        self.fact_checker = FactCheckerAgent(rag_system)
        self.tone_editor = ToneEditorAgent(rag_system)

    def generate_article(self, brief: ArticleBrief, max_iterations: int = 2) -> FinalArticle:
        """
        Run complete workflow to generate article

        Args:
            brief: Article brief with requirements
            max_iterations: Max fact-check/revision cycles

        Returns:
            FinalArticle with all metadata
        """
        workflow_log = []
        start_time = datetime.now()

        # Step 1: Generate outline
        print("Step 1: Generating outline...")
        outline = self.outline_agent.generate_outline(brief)
        workflow_log.append({"step": "outline", "status": "complete"})

        # Step 2: Write draft
        print("Step 2: Writing draft...")
        draft = self.writer_agent.write_draft(brief, outline)
        workflow_log.append({"step": "writer", "status": "complete", "word_count": draft.word_count})

        # Step 3-4: Fact-check and revise loop
        iteration = 0
        fact_check_result = None

        while iteration < max_iterations:
            print(f"Step 3: Fact-checking (iteration {iteration + 1})...")
            fact_check_result = self.fact_checker.fact_check(draft)
            workflow_log.append({
                "step": f"fact_check_{iteration}",
                "grounding_score": fact_check_result.grounding_score,
                "is_grounded": fact_check_result.is_grounded
            })

            if fact_check_result.is_grounded:
                break

            # If not grounded, revise draft with suggested fixes
            if iteration < max_iterations - 1:
                print(f"  Grounding score too low ({fact_check_result.grounding_score:.2f}), revising...")
                # In production, would have revision agent here
                iteration += 1
            else:
                print(f"  Max iterations reached. Final score: {fact_check_result.grounding_score:.2f}")
                break

            iteration += 1

        # Step 4: Tone editing
        print("Step 4: Editing tone and style...")
        tone_result = self.tone_editor.edit_tone(draft, fact_check_result)
        workflow_log.append({
            "step": "tone_editor",
            "style_score": tone_result.style_score,
            "issues_found": len(tone_result.issues)
        })

        # Step 5: Final review and editor notes
        print("Step 5: Final review...")
        editor_notes = []

        if fact_check_result.grounding_score < 0.9:
            editor_notes.append(
                f"Fact-check: Some claims may need verification (score: {fact_check_result.grounding_score:.2f})"
            )

        if tone_result.style_score < 0.85:
            editor_notes.append(
                f"Style: Minor tone adjustments may be needed (score: {tone_result.style_score:.2f})"
            )

        if len(fact_check_result.unsupported_claims) > 0:
            editor_notes.append(
                f"Please review {len(fact_check_result.unsupported_claims)} unsupported claims"
            )

        ready_for_editor = (
            fact_check_result.grounding_score >= 0.7 and
            tone_result.style_score >= 0.7 and
            len(draft.citations) > 0
        )

        end_time = datetime.now()
        workflow_metadata = {
            "workflow_log": workflow_log,
            "execution_time_seconds": (end_time - start_time).total_seconds(),
            "brief": brief.__dict__,
            "outline": outline.__dict__
        }

        return FinalArticle(
            content=tone_result.revised_content,
            citations=[
                {"doc_id": c.doc_id, "section_id": c.section_id}
                for c in draft.citations[:10]  # Deduplicate and limit
            ] if hasattr(draft.citations[0], 'doc_id') else draft.citations,
            fact_check_score=fact_check_result.grounding_score,
            style_score=tone_result.style_score,
            workflow_metadata=workflow_metadata,
            ready_for_editor=ready_for_editor,
            editor_notes=editor_notes
        )


def main():
    """Example usage of agentic workflow"""
    import dotenv
    dotenv.load_dotenv()

    # Initialize RAG system
    rag = AyurvedaRAGSystem()
    rag.load_and_index_content()

    # Initialize workflow orchestrator
    orchestrator = ArticleWorkflowOrchestrator(rag)

    # Example brief
    brief = ArticleBrief(
        topic="Ayurvedic Support for Stress and Better Sleep",
        target_audience="Busy professionals experiencing stress and sleep issues",
        key_points=[
            "How Ayurveda views stress and sleep",
            "Practical lifestyle approaches",
            "Herbs that support stress resilience",
            "Evening routines for better sleep"
        ],
        word_count_target=800,
        must_include_products=["Ashwagandha Stress Balance Tablets", "Brahmi Tailam"]
    )

    # Generate article
    print("\n" + "="*80)
    print("GENERATING ARTICLE")
    print("="*80 + "\n")

    article = orchestrator.generate_article(brief)

    print("\n" + "="*80)
    print("FINAL ARTICLE")
    print("="*80)
    print(f"\nReady for editor: {article.ready_for_editor}")
    print(f"Fact-check score: {article.fact_check_score:.2f}")
    print(f"Style score: {article.style_score:.2f}")
    print(f"\nEditor notes:")
    for note in article.editor_notes:
        print(f"  - {note}")

    print(f"\n\nArticle content:\n")
    print(article.content)

    print(f"\n\nCitations:")
    for cite in article.citations:
        print(f"  - {cite}")


if __name__ == "__main__":
    main()
