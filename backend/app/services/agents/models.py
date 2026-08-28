"""
Article Pipeline State
========================

Pydantic versions of the dataclasses the original workflow passed between
agents (src/agent_workflow.py). Field names are preserved deliberately — the
API response shape and the Streamlit UI both read them.

The one structural addition is `ArticleState`, the object LangGraph threads
through the graph. The previous implementation had no state object at all:
each agent took the previous agent's return value as a positional argument,
which is why the retry loop could re-run the fact-checker against a draft that
no revision step had touched.

`sections` uses a reducer because section writing fans out — several nodes
write to the same dict concurrently and LangGraph needs to know how to merge
those partial updates rather than have the last writer win.
"""

from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, Field


def merge_sections(
    left: Optional[Dict[int, str]],
    right: Optional[Dict[int, str]],
) -> Dict[int, str]:
    """
    Reducer for concurrent section writes.

    Each parallel `write_section` node returns a single-entry dict keyed by
    section index. LangGraph folds them together with this instead of letting
    one overwrite the rest.
    """
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class ArticleBrief(BaseModel):
    """Input brief for article generation."""
    topic: str
    target_audience: str
    key_points: List[str] = Field(default_factory=list)
    word_count_target: int = 800
    must_include_products: List[str] = Field(default_factory=list)


class OutlineSection(BaseModel):
    heading: str
    key_points: str = ""


class Outline(BaseModel):
    """Structured outline from the outline agent."""
    title: str
    sections: List[OutlineSection] = Field(default_factory=list)
    estimated_word_count: int = 0
    key_sources_needed: List[str] = Field(default_factory=list)


class Draft(BaseModel):
    """Assembled article draft."""
    content: str
    word_count: int = 0
    citations: List[Dict[str, str]] = Field(default_factory=list)
    sections: List[str] = Field(default_factory=list)


class FactCheckResult(BaseModel):
    """Output of the fact-checking agent."""
    is_grounded: bool = False
    grounding_score: float = 0.0
    total_claims: int = 0
    supported_claims: int = 0
    unsupported_claims: List[str] = Field(default_factory=list)
    missing_citations: List[str] = Field(default_factory=list)
    suggested_fixes: List[Dict[str, str]] = Field(default_factory=list)
    parse_failed: bool = False


class ToneCheckResult(BaseModel):
    """Output of the tone editor agent."""
    style_score: float = 0.0
    issues: List[Dict[str, str]] = Field(default_factory=list)
    revised_content: str = ""


class FinalArticle(BaseModel):
    """Complete article with all provenance."""
    content: str
    citations: List[Dict[str, str]] = Field(default_factory=list)
    fact_check_score: float = 0.0
    style_score: float = 0.0
    workflow_metadata: Dict[str, Any] = Field(default_factory=dict)
    ready_for_editor: bool = False
    editor_notes: List[str] = Field(default_factory=list)


class ArticleState(BaseModel):
    """
    State threaded through the article graph.

    Every node receives this and returns a partial update.
    """
    brief: ArticleBrief

    outline: Optional[Outline] = None
    sections: Annotated[Dict[int, str], merge_sections] = Field(default_factory=dict)
    draft: Optional[Draft] = None
    fact_check: Optional[FactCheckResult] = None
    tone_result: Optional[ToneCheckResult] = None
    final: Optional[FinalArticle] = None

    # Counts completed revision passes. The original incremented its loop
    # counter twice per iteration, so max_iterations=2 actually allowed one.
    revision_count: int = 0

    editor_notes: List[str] = Field(default_factory=list)
    status: str = "queued"

    model_config = {"arbitrary_types_allowed": True}
