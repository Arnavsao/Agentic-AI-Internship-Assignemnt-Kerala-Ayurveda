"""
Article Pipeline Nodes
========================

THE CHANGE THAT MATTERS: agents now retrieve raw chunks.

Every agent in the original pipeline reached the corpus through
`rag.answer_user_query()` — a full retrieval *plus an LLM generation* — so
each "lookup" cost a round trip and handed back a paraphrase rather than the
source text. The writer then grounded its citations on a summary of a summary.
That indirection accounted for most of the 11-15 sequential LLM calls behind
the 2-3 minute runtime, and it degraded grounding at the same time.

These nodes call `retriever.retrieve()` and read the chunks directly. Lookups
become vector queries measured in milliseconds, and the text an agent cites is
the text that is actually in the knowledge base.

Other fixes:
  * Sections are written concurrently (LangGraph Send fan-out), not in a loop.
  * The fact-checker verifies all flagged claims in one batched call instead of
    one unbounded LLM call per claim.
  * The tone editor reads the style guide from disk. It is a static file in
    data/; the original fetched it through RAG + LLM on every single run.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.app.core.config import get_settings
from backend.app.services.agents import prompts
from backend.app.services.agents.models import (
    ArticleState, Draft, FactCheckResult, FinalArticle, Outline, OutlineSection,
    ToneCheckResult,
)

logger = logging.getLogger(__name__)

# Claims verified per fact-check run. The original issued one LLM call per
# unsupported claim with no ceiling, so a bad draft could fan out indefinitely.
MAX_CLAIMS_VERIFIED = 8

_style_guide_cache: Optional[str] = None


def extract_json(text: str) -> dict:
    """
    Pull a JSON object out of an LLM response.

    Models routinely wrap JSON in markdown fences or pad it with prose. Strips
    fences, then falls back to brace matching. Returns {} if nothing parses —
    callers must treat that as a failure, not as a default-shaped result.
    """
    if not text or not text.strip():
        return {}

    stripped = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped.strip())

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    return {}


def load_style_guide() -> str:
    """
    Read the brand style guide from disk (cached).

    This is a static file in the content directory. The original tone editor
    retrieved it through the RAG pipeline on every run, spending a retrieval
    and an LLM call to fetch text that never changes.
    """
    global _style_guide_cache
    if _style_guide_cache is not None:
        return _style_guide_cache

    settings = get_settings()
    path = settings.content_path / "content_style_and_tone_guide.md"
    try:
        _style_guide_cache = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning(
            f"Style guide not readable at {path}: {e}. Falling back to built-in rules.",
            extra={"component": "agents"},
        )
        _style_guide_cache = (
            "Kerala Ayurveda voice: warm, grounded, precise. Use "
            "'traditionally used to support...' phrasing. Never claim to "
            "diagnose, treat, cure, or prevent disease. Always encourage "
            "consultation with a qualified practitioner."
        )
    return _style_guide_cache


def format_evidence(chunks: List[Any], limit: Optional[int] = None) -> str:
    """Render retrieved chunks in the [Source: doc_id - section_id] form agents cite."""
    parts = []
    for chunk in chunks[:limit] if limit else chunks:
        parts.append(
            f"[Source: {chunk.doc_id} - {chunk.section_id}]\n{chunk.context_content}"
        )
    return "\n\n---\n\n".join(parts)


def extract_citations(text: str) -> List[Dict[str, str]]:
    """Collect [Source: ...] markers from generated text."""
    seen, citations = set(), []
    for match in re.findall(r"\[Source:\s*([^\]]+)\]", text):
        marker = match.strip()
        if marker in seen:
            continue
        seen.add(marker)
        doc_id, _, section_id = marker.partition(" - ")
        citations.append({
            "citation": marker,
            "doc_id": doc_id.strip(),
            "section_id": section_id.strip(),
        })
    return citations


class ArticleNodes:
    """
    The graph's nodes, bound to a retriever and an LLM provider.

    Dependencies are injected rather than constructed here so tests can drive
    the whole graph with fakes.
    """

    def __init__(self, retriever, llm_provider):
        self.retriever = retriever
        self.llm = llm_provider
        self.settings = get_settings()

    def _retrieve(self, query: str, top_n: Optional[int] = None) -> List[Any]:
        try:
            return self.retriever.retrieve(query, top_n=top_n)
        except Exception as e:
            logger.warning(
                f"Retrieval failed for '{query[:60]}': {e}",
                extra={"component": "agents"},
            )
            return []

    # ── 1. Outline ──────────────────────────────────────────────

    async def outline(self, state: ArticleState) -> dict:
        """
        Plan the article against what the corpus can actually support.

        The coverage check is now a plain vector query. The original spent a
        full RAG generation here just to summarize what the corpus contained.
        """
        brief = state.brief
        chunks = self._retrieve(
            f"{brief.topic} {' '.join(brief.key_points)}",
            top_n=self.settings.retrieval_top_n,
        )
        corpus_context = format_evidence(chunks) or "No relevant context found."

        text = await self.llm.agenerate(
            [
                ("system", prompts.OUTLINE_SYSTEM.format(
                    corpus_context=corpus_context,
                    word_count=brief.word_count_target,
                )),
                ("user", prompts.OUTLINE_USER.format(
                    topic=brief.topic,
                    audience=brief.target_audience,
                    key_points=", ".join(brief.key_points),
                    word_count=brief.word_count_target,
                    products=", ".join(brief.must_include_products) or "None specified",
                )),
            ],
            temperature=0.3,
        )

        data = extract_json(text)
        sections = [
            OutlineSection(
                heading=str(s.get("heading", f"Section {i + 1}")),
                key_points=str(s.get("key_points", "")),
            )
            for i, s in enumerate(data.get("sections") or [])
            if isinstance(s, dict)
        ]

        if not sections:
            # Fall back to the brief's own key points rather than failing the
            # run. Uses .get throughout — the original indexed the parsed dict
            # directly and raised KeyError on a partial parse.
            logger.warning(
                "Outline parse produced no sections; falling back to brief key points",
                extra={"component": "agents"},
            )
            sections = [OutlineSection(heading=kp, key_points=kp) for kp in brief.key_points]

        outline = Outline(
            title=str(data.get("title") or brief.topic),
            sections=sections,
            estimated_word_count=int(data.get("estimated_word_count") or brief.word_count_target),
            key_sources_needed=[str(s) for s in (data.get("key_sources_needed") or [])],
        )

        logger.info(
            f"Outline: '{outline.title}' with {len(sections)} sections",
            extra={"component": "agents"},
        )
        return {"outline": outline, "status": "writing"}

    # ── 2. Section writer (fanned out) ──────────────────────────

    async def write_section(self, payload: dict) -> dict:
        """
        Write one section against its own retrieved evidence.

        Receives a plain dict rather than ArticleState because LangGraph's Send
        API delivers a per-branch payload. Returns a single-entry dict that the
        `sections` reducer merges with its siblings.
        """
        index: int = payload["index"]
        section: OutlineSection = payload["section"]
        title: str = payload["title"]
        audience: str = payload["audience"]
        word_budget: int = payload["word_budget"]

        chunks = self._retrieve(
            f"{section.heading} {section.key_points}",
            top_n=self.settings.retrieval_top_n,
        )
        evidence = format_evidence(chunks) or "No specific context retrieved."

        text = await self.llm.agenerate(
            [
                ("system", prompts.SECTION_SYSTEM),
                ("user", prompts.SECTION_USER.format(
                    title=title,
                    audience=audience,
                    heading=section.heading,
                    key_points=section.key_points,
                    word_budget=word_budget,
                    context=evidence,
                )),
            ],
            temperature=0.2,
        )

        logger.info(
            f"Section {index + 1} written: '{section.heading}' ({len(text.split())} words)",
            extra={"component": "agents"},
        )
        return {"sections": {index: text}}

    # ── 3. Assemble ─────────────────────────────────────────────

    async def assemble_draft(self, state: ArticleState) -> dict:
        """Stitch the parallel sections back into one document, in order."""
        outline = state.outline
        ordered = [state.sections[i] for i in sorted(state.sections) if state.sections.get(i)]
        body = "\n\n".join(ordered)
        content = f"# {outline.title}\n\n{body}" if outline else body

        draft = Draft(
            content=content,
            word_count=len(content.split()),
            citations=extract_citations(content),
            sections=[s.heading for s in outline.sections] if outline else [],
        )

        logger.info(
            f"Draft assembled: {draft.word_count} words, {len(draft.citations)} citations",
            extra={"component": "agents"},
        )
        return {"draft": draft, "status": "fact_checking"}

    # ── 4. Fact check ───────────────────────────────────────────

    async def fact_check(self, state: ArticleState) -> dict:
        """
        Score how much of the draft the corpus actually supports.

        Evidence for every cited document is gathered in one retrieval pass and
        checked in a single LLM call. The original made one call per flagged
        claim, unbounded.
        """
        draft = state.draft

        queries = [c["citation"] for c in draft.citations[:MAX_CLAIMS_VERIFIED]]
        if not queries and state.outline:
            queries = [s.heading for s in state.outline.sections[:MAX_CLAIMS_VERIFIED]]

        evidence_chunks, seen = [], set()
        for q in queries:
            for chunk in self._retrieve(q, top_n=3):
                key = (chunk.doc_id, chunk.section_id)
                if key not in seen:
                    seen.add(key)
                    evidence_chunks.append(chunk)

        text = await self.llm.agenerate(
            [
                ("system", prompts.FACT_CHECK_SYSTEM),
                ("user", prompts.FACT_CHECK_USER.format(
                    article=draft.content,
                    citations=json.dumps(draft.citations, indent=2),
                    evidence=format_evidence(evidence_chunks) or "No evidence retrieved.",
                )),
            ],
            temperature=0.0,
        )

        data = extract_json(text)

        if not data:
            # A parse failure is an unknown result, not a pass. The original
            # defaulted to 0.75 / grounded=True here, so a malformed response
            # silently cleared the safety gate on medical content.
            logger.error(
                "Fact-check response did not parse; treating as ungrounded",
                extra={"component": "agents"},
            )
            return {
                "fact_check": FactCheckResult(
                    is_grounded=False,
                    grounding_score=0.0,
                    parse_failed=True,
                    unsupported_claims=["Fact-check response could not be parsed"],
                ),
                "editor_notes": ["Fact-check output was unparseable — manual review required."],
            }

        total = int(data.get("total_claims") or 0)
        supported = int(data.get("supported_claims") or 0)
        raw_score = data.get("grounding_score")
        score = float(raw_score) if raw_score is not None else (supported / total if total else 0.0)
        score = max(0.0, min(1.0, score))

        fixes = [f for f in (data.get("suggested_fixes") or []) if isinstance(f, dict)]
        result = FactCheckResult(
            is_grounded=score >= self.settings.agent_grounding_threshold,
            grounding_score=score,
            total_claims=total,
            supported_claims=supported,
            unsupported_claims=[str(c) for c in (data.get("unsupported_claims") or [])],
            missing_citations=[str(c) for c in (data.get("missing_citations") or [])],
            suggested_fixes=[{str(k): str(v) for k, v in f.items()} for f in fixes],
        )

        logger.info(
            f"Fact check: {score:.2f} grounding "
            f"({supported}/{total} claims), grounded={result.is_grounded}",
            extra={"component": "agents", "grounding_score": score},
        )
        return {"fact_check": result}

    # ── 5. Revise ───────────────────────────────────────────────

    async def revise(self, state: ArticleState) -> dict:
        """
        Rewrite flagged claims so the next fact-check sees different text.

        This node did not exist. The original loop re-ran the fact-checker
        against the unchanged draft, so a rejected article could only be
        rejected again — the retry was structurally incapable of helping.
        """
        draft = state.draft
        fc = state.fact_check

        evidence_chunks, seen = [], set()
        for claim in fc.unsupported_claims[:MAX_CLAIMS_VERIFIED]:
            for chunk in self._retrieve(claim, top_n=3):
                key = (chunk.doc_id, chunk.section_id)
                if key not in seen:
                    seen.add(key)
                    evidence_chunks.append(chunk)

        revised = await self.llm.agenerate(
            [
                ("system", prompts.REVISE_SYSTEM),
                ("user", prompts.REVISE_USER.format(
                    article=draft.content,
                    unsupported="\n".join(f"- {c}" for c in fc.unsupported_claims) or "None listed",
                    fixes=json.dumps(fc.suggested_fixes, indent=2) or "None suggested",
                    evidence=format_evidence(evidence_chunks) or "No evidence retrieved.",
                )),
            ],
            temperature=0.2,
        )

        if not revised.strip():
            logger.warning(
                "Revision returned empty content; keeping previous draft",
                extra={"component": "agents"},
            )
            return {"revision_count": state.revision_count + 1}

        new_draft = Draft(
            content=revised,
            word_count=len(revised.split()),
            citations=extract_citations(revised),
            sections=draft.sections,
        )

        logger.info(
            f"Revision {state.revision_count + 1}: "
            f"{new_draft.word_count} words, {len(new_draft.citations)} citations",
            extra={"component": "agents"},
        )
        return {
            "draft": new_draft,
            "revision_count": state.revision_count + 1,
            "status": "fact_checking",
        }

    # ── 6. Tone edit ────────────────────────────────────────────

    async def tone_edit(self, state: ArticleState) -> dict:
        """Align voice with the brand style guide without touching facts."""
        draft = state.draft
        fc = state.fact_check

        text = await self.llm.agenerate(
            [
                ("system", prompts.TONE_SYSTEM.format(style_guide=load_style_guide())),
                ("user", prompts.TONE_USER.format(
                    article=draft.content,
                    fact_check_passed=fc.is_grounded if fc else False,
                    grounding_score=f"{fc.grounding_score:.2f}" if fc else "0.00",
                )),
            ],
            temperature=0.2,
        )

        data = extract_json(text)
        raw_score = data.get("style_score")
        style_score = max(0.0, min(1.0, float(raw_score))) if raw_score is not None else 0.0
        issues = [i for i in (data.get("issues") or []) if isinstance(i, dict)]

        revised = str(data.get("revised_content") or "").strip()
        # "NO CHANGES" is the documented signal that the draft already complies.
        keep_original = (not revised) or revised.upper().startswith("NO CHANGES")
        final_content = draft.content if keep_original else revised

        # A revision that drops citations has broken the one rule this agent is
        # told never to break, so it doesn't get to ship.
        if not keep_original:
            before = len(extract_citations(draft.content))
            after = len(extract_citations(final_content))
            if before and after < before:
                logger.warning(
                    f"Tone edit dropped citations ({before} -> {after}); keeping original",
                    extra={"component": "agents"},
                )
                final_content = draft.content
                keep_original = True

        tone_result = ToneCheckResult(
            style_score=style_score,
            issues=[{str(k): str(v) for k, v in i.items()} for i in issues],
            revised_content=final_content,
        )

        notes = list(state.editor_notes)
        if fc and not fc.is_grounded:
            notes.append(
                f"Grounding {fc.grounding_score:.2f} is below the "
                f"{self.settings.agent_grounding_threshold} threshold after "
                f"{state.revision_count} revision(s) — needs editor review."
            )
        if style_score < self.settings.agent_style_threshold:
            notes.append(f"Style score {style_score:.2f} is below threshold.")
        if not keep_original:
            notes.append("Tone editor revised the draft.")

        final = FinalArticle(
            content=final_content,
            citations=extract_citations(final_content),
            fact_check_score=fc.grounding_score if fc else 0.0,
            style_score=style_score,
            ready_for_editor=bool(fc and fc.is_grounded)
                             and style_score >= self.settings.agent_style_threshold,
            editor_notes=notes,
            workflow_metadata={
                "revisions": state.revision_count,
                "sections": len(state.sections),
                "word_count": len(final_content.split()),
                "grounding_score": fc.grounding_score if fc else 0.0,
                "style_score": style_score,
                "fact_check_parse_failed": bool(fc and fc.parse_failed),
            },
        )

        logger.info(
            f"Tone edit complete: style={style_score:.2f}, "
            f"ready_for_editor={final.ready_for_editor}",
            extra={"component": "agents"},
        )
        return {
            "tone_result": tone_result,
            "final": final,
            "editor_notes": notes,
            "status": "completed",
        }
