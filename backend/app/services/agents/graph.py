"""
Article Generation Graph
==========================

    outline
       │
       ├─ Send ─→ write_section (one per section, concurrent)
       │              │
       └──────────────┴─→ assemble_draft
                              │
                              ▼
                         fact_check ←──────┐
                              │            │
                    grounded? │            │ revised draft
                     ┌────────┴────────┐   │
                     │ no, retries left├───┘
                     │                 │  revise
                     │ yes / exhausted │
                     └────────┬────────┘
                              ▼
                          tone_edit
                              │
                             END

WHAT THIS REPLACES:
`ArticleWorkflowOrchestrator.generate_article` ran the four agents as a
straight-line sequence of Python calls. Three problems came out of that shape:

1. The retry loop was inert. On a failed fact-check it called `fact_check` again
   on the identical draft, with no step in between that could change the text —
   and it incremented its counter twice per pass, so `max_iterations=2` bought
   exactly one extra check of the same content. Both are fixed here: `revise`
   sits on the loop edge, and the counter advances once per revision.

2. Sections were written one at a time even though they share no state.

3. There was no state object, so no way to see or resume a partial run.

CHECKPOINTING: deliberately omitted. A checkpointer buys resume-after-crash and
time-travel debugging, which matter for long or human-in-the-loop graphs. This
one is six nodes and about a minute end to end, and job progress is already
persisted in the ArticleJob table — a second persistence layer would be pure
overhead. `graph.compile(checkpointer=SqliteSaver(...))` is the upgrade path if
mid-run resume is ever wanted.
"""

import logging
from typing import Any, List

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from backend.app.core.config import get_settings
from backend.app.services.agents.models import ArticleBrief, ArticleState
from backend.app.services.agents.nodes import ArticleNodes

logger = logging.getLogger(__name__)


def fan_out_sections(state: ArticleState) -> List[Send]:
    """
    Dispatch one concurrent `write_section` branch per outline section.

    Each branch carries only what it needs. They run in parallel and their
    results are folded back together by the `sections` reducer.
    """
    outline = state.outline
    if not outline or not outline.sections:
        return []

    # Leave room for the title and any framing; divide the rest evenly.
    per_section = max(120, outline.estimated_word_count // max(len(outline.sections), 1))

    return [
        Send(
            "write_section",
            {
                "index": i,
                "section": section,
                "title": outline.title,
                "audience": state.brief.target_audience,
                "word_budget": per_section,
            },
        )
        for i, section in enumerate(outline.sections)
    ]


def route_after_fact_check(state: ArticleState) -> str:
    """
    Decide whether the draft goes back for revision or moves to tone editing.

    Revision only happens when there is something actionable to revise and
    budget left to do it — otherwise the article proceeds with an editor note
    explaining why, rather than looping pointlessly.
    """
    settings = get_settings()
    fc = state.fact_check

    if fc is None:
        return "tone_edit"

    if fc.is_grounded:
        return "tone_edit"

    if state.revision_count >= settings.agent_max_iterations:
        logger.warning(
            f"Grounding {fc.grounding_score:.2f} still below threshold after "
            f"{state.revision_count} revision(s); escalating to editor",
            extra={"component": "agents"},
        )
        return "tone_edit"

    if not fc.unsupported_claims and not fc.suggested_fixes:
        # Nothing concrete to act on — another pass would just resend the same
        # text and get the same verdict.
        logger.warning(
            "Draft is ungrounded but no specific claims were flagged; "
            "skipping revision",
            extra={"component": "agents"},
        )
        return "tone_edit"

    logger.info(
        f"Grounding {fc.grounding_score:.2f} below threshold — "
        f"revision {state.revision_count + 1}",
        extra={"component": "agents"},
    )
    return "revise"


def build_article_graph(retriever: Any, llm_provider: Any):
    """
    Compile the article generation graph.

    Args:
        retriever: something with .retrieve(query, top_n) -> chunks
        llm_provider: something with async .agenerate(messages, temperature=...)
    """
    nodes = ArticleNodes(retriever=retriever, llm_provider=llm_provider)

    graph = StateGraph(ArticleState)
    graph.add_node("outline", nodes.outline)
    graph.add_node("write_section", nodes.write_section)
    graph.add_node("assemble_draft", nodes.assemble_draft)
    graph.add_node("fact_check", nodes.fact_check)
    graph.add_node("revise", nodes.revise)
    graph.add_node("tone_edit", nodes.tone_edit)

    graph.add_edge(START, "outline")
    graph.add_conditional_edges("outline", fan_out_sections, ["write_section"])
    graph.add_edge("write_section", "assemble_draft")
    graph.add_edge("assemble_draft", "fact_check")
    graph.add_conditional_edges(
        "fact_check", route_after_fact_check, ["revise", "tone_edit"]
    )
    graph.add_edge("revise", "fact_check")
    graph.add_edge("tone_edit", END)

    return graph.compile()


async def generate_article(
    brief: ArticleBrief,
    retriever: Any,
    llm_provider: Any,
    on_step: Any = None,
) -> ArticleState:
    """
    Run the pipeline end to end.

    Args:
        brief: what to write
        retriever: hybrid retriever
        llm_provider: LLM gateway
        on_step: optional async callback(node_name, state_dict) invoked as each
                 node completes — used to stream progress into the job record

    Returns the final ArticleState.
    """
    app = build_article_graph(retriever, llm_provider)
    initial = ArticleState(brief=brief, status="outlining")

    latest_values: dict = {}

    # Two stream modes at once:
    #   "updates" — each node's own output, for the progress callback
    #   "values"  — the full reduced state, which is what we return
    #
    # The final state must come from "values". Folding the per-node "updates"
    # together by hand would apply plain dict.update() and lose concurrent
    # writes: the parallel write_section branches each emit {"sections": {i: ...}},
    # and only the graph's reducer merges those correctly rather than letting
    # the last branch win.
    async for mode, chunk in app.astream(initial, stream_mode=["updates", "values"]):
        if mode == "values":
            latest_values = chunk
        elif mode == "updates" and on_step is not None:
            for node_name, update in (chunk or {}).items():
                await on_step(node_name, update or {})

    if not latest_values:
        return initial

    if isinstance(latest_values, ArticleState):
        return latest_values
    return ArticleState(**latest_values)
