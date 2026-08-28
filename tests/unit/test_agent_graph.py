"""
Tests for the LangGraph article pipeline.

Driven entirely by fakes — no network, no models. The properties under test
are the ones the original sequential implementation got wrong:

  * a failed fact-check must produce a *different* draft before re-checking
  * the revision counter must advance once per revision, not twice
  * a fact-check whose output doesn't parse must not be scored as a pass
  * sections must be written concurrently and reassembled in order
"""

import asyncio
import json

import pytest

from backend.app.services.agents.graph import (
    fan_out_sections, generate_article, route_after_fact_check,
)
from backend.app.services.agents.models import (
    ArticleBrief, ArticleState, Draft, FactCheckResult, Outline, OutlineSection,
)
from backend.app.services.agents.nodes import extract_citations, extract_json


class FakeChunk:
    def __init__(self, doc_id="doc", section_id="sec", content="Evidence text."):
        self.doc_id = doc_id
        self.section_id = section_id
        self.context_content = content
        self.document = type("D", (), {"page_content": content})()


class FakeRetriever:
    def __init__(self, chunks=None):
        self.chunks = chunks if chunks is not None else [FakeChunk()]
        self.queries = []

    def retrieve(self, query, top_n=None, **kwargs):
        self.queries.append(query)
        return self.chunks


class ScriptedLLM:
    """
    Returns canned responses in order, keyed by the kind of call.

    Records the temperature it was handed so tests can assert per-agent
    settings actually reach the provider — the exact thing the old MegaLLM
    path silently dropped.
    """

    def __init__(self, outline=None, section=None, fact_checks=None,
                 revision=None, tone=None):
        self.outline_response = outline or json.dumps({
            "title": "Ayurvedic Support for Stress",
            "sections": [
                {"heading": "Understanding Stress", "key_points": "How Ayurveda views stress"},
                {"heading": "Herbs That Help", "key_points": "Ashwagandha and Brahmi"},
            ],
            "estimated_word_count": 800,
            "key_sources_needed": ["faq_general"],
        })
        self.section_response = section or (
            "## Section\n\nAshwagandha is traditionally used to support the "
            "body's response to stress [Source: product_ashwagandha - Positioning]."
        )
        self.fact_check_responses = list(fact_checks or [json.dumps({
            "total_claims": 4, "supported_claims": 4,
            "unsupported_claims": [], "missing_citations": [],
            "grounding_score": 0.9,
        })])
        self.revision_response = revision or (
            "# Revised\n\nA corrected and properly grounded article "
            "[Source: product_ashwagandha - Positioning]."
        )
        self.tone_response = tone or json.dumps({
            "style_score": 0.9, "issues": [], "revised_content": "NO CHANGES",
        })

        self.calls = []          # (kind, temperature)
        self.concurrent_peak = 0
        self._in_flight = 0

    async def agenerate(self, messages, *, temperature=None, model=None, **kwargs):
        system = messages[0][1] if messages else ""

        if "content strategist" in system:
            kind, response = "outline", self.outline_response
        elif "ONE section" in system:
            kind, response = "section", self.section_response
        elif "fact-checking agent" in system:
            kind = "fact_check"
            response = (self.fact_check_responses.pop(0)
                        if len(self.fact_check_responses) > 1
                        else self.fact_check_responses[0])
        elif "revision agent" in system:
            kind, response = "revise", self.revision_response
        elif "style editor" in system:
            kind, response = "tone", self.tone_response
        else:
            kind, response = "unknown", "{}"

        self.calls.append((kind, temperature))

        self._in_flight += 1
        self.concurrent_peak = max(self.concurrent_peak, self._in_flight)
        await asyncio.sleep(0.01)   # yield so parallelism is observable
        self._in_flight -= 1

        return response

    def kinds(self):
        return [k for k, _ in self.calls]

    def temperature_for(self, kind):
        return next(t for k, t in self.calls if k == kind)


@pytest.fixture
def brief():
    return ArticleBrief(
        topic="Ayurvedic Support for Stress and Better Sleep",
        target_audience="Busy professionals",
        key_points=["How Ayurveda views stress", "Herbs that support resilience"],
        word_count_target=800,
    )


class TestHelpers:
    def test_extract_json_plain(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_extract_json_fenced(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_extract_json_with_prose(self):
        assert extract_json('Here you go:\n{"a": 1}\nHope that helps') == {"a": 1}

    def test_extract_json_unparseable_returns_empty(self):
        assert extract_json("not json at all") == {}

    def test_extract_citations(self):
        text = "Claim one [Source: doc_a - Section One]. Claim two [Source: doc_b - Section Two]."
        cites = extract_citations(text)
        assert len(cites) == 2
        assert cites[0]["doc_id"] == "doc_a"
        assert cites[0]["section_id"] == "Section One"

    def test_extract_citations_deduplicates(self):
        text = "A [Source: doc - Sec]. B [Source: doc - Sec]."
        assert len(extract_citations(text)) == 1


class TestRouting:
    def _state(self, **kw):
        return ArticleState(brief=ArticleBrief(topic="t", target_audience="a"), **kw)

    def test_grounded_goes_to_tone(self):
        state = self._state(fact_check=FactCheckResult(is_grounded=True, grounding_score=0.9))
        assert route_after_fact_check(state) == "tone_edit"

    def test_ungrounded_with_claims_revises(self):
        state = self._state(fact_check=FactCheckResult(
            is_grounded=False, grounding_score=0.4,
            unsupported_claims=["unsupported claim"],
        ))
        assert route_after_fact_check(state) == "revise"

    def test_revision_budget_is_respected(self):
        state = self._state(
            fact_check=FactCheckResult(
                is_grounded=False, grounding_score=0.4,
                unsupported_claims=["claim"],
            ),
            revision_count=2,   # equals default agent_max_iterations
        )
        assert route_after_fact_check(state) == "tone_edit"

    def test_no_actionable_claims_skips_revision(self):
        """Revising with nothing flagged would resend identical text."""
        state = self._state(fact_check=FactCheckResult(
            is_grounded=False, grounding_score=0.4, unsupported_claims=[],
        ))
        assert route_after_fact_check(state) == "tone_edit"


class TestFanOut:
    def test_one_send_per_section(self, brief):
        state = ArticleState(brief=brief, outline=Outline(
            title="T",
            sections=[OutlineSection(heading=f"S{i}") for i in range(3)],
            estimated_word_count=900,
        ))
        sends = fan_out_sections(state)
        assert len(sends) == 3
        assert {s.arg["index"] for s in sends} == {0, 1, 2}

    def test_word_budget_divided(self, brief):
        state = ArticleState(brief=brief, outline=Outline(
            title="T",
            sections=[OutlineSection(heading=f"S{i}") for i in range(3)],
            estimated_word_count=900,
        ))
        assert fan_out_sections(state)[0].arg["word_budget"] == 300

    def test_empty_outline_produces_no_sends(self, brief):
        assert fan_out_sections(ArticleState(brief=brief)) == []


@pytest.mark.asyncio
class TestHappyPath:
    async def test_completes(self, brief):
        llm = ScriptedLLM()
        state = await generate_article(brief, FakeRetriever(), llm)
        assert state.status == "completed"
        assert state.final is not None
        assert state.final.content

    async def test_runs_each_stage(self, brief):
        llm = ScriptedLLM()
        await generate_article(brief, FakeRetriever(), llm)
        kinds = llm.kinds()
        assert kinds.count("outline") == 1
        assert kinds.count("section") == 2      # two outline sections
        assert kinds.count("fact_check") == 1
        assert kinds.count("tone") == 1
        assert "revise" not in kinds            # grounded draft needs no revision

    async def test_sections_written_concurrently(self, brief):
        """The original wrote sections in a sequential loop."""
        llm = ScriptedLLM()
        await generate_article(brief, FakeRetriever(), llm)
        assert llm.concurrent_peak > 1

    async def test_sections_assembled_in_order(self, brief):
        llm = ScriptedLLM()
        state = await generate_article(brief, FakeRetriever(), llm)
        assert sorted(state.sections) == [0, 1]

    async def test_per_agent_temperatures_reach_provider(self, brief):
        """
        Regression for the dropped-kwargs bug: the MegaLLM path built its
        client with no arguments, so these values never left the caller.
        """
        llm = ScriptedLLM()
        await generate_article(brief, FakeRetriever(), llm)
        assert llm.temperature_for("outline") == 0.3
        assert llm.temperature_for("section") == 0.2
        assert llm.temperature_for("fact_check") == 0.0
        assert llm.temperature_for("tone") == 0.2

    async def test_agents_read_raw_chunks(self, brief):
        """Corpus access is retrieval, not a nested RAG generation."""
        retriever = FakeRetriever()
        await generate_article(brief, retriever, ScriptedLLM())
        assert len(retriever.queries) > 0

    async def test_citations_collected(self, brief):
        state = await generate_article(brief, FakeRetriever(), ScriptedLLM())
        assert state.final.citations

    async def test_ready_for_editor_when_scores_pass(self, brief):
        state = await generate_article(brief, FakeRetriever(), ScriptedLLM())
        assert state.final.ready_for_editor is True


@pytest.mark.asyncio
class TestRevisionLoop:
    def _ungrounded_then_grounded(self):
        return [
            json.dumps({
                "total_claims": 4, "supported_claims": 1,
                "unsupported_claims": ["Ashwagandha cures anxiety"],
                "missing_citations": ["Section 2"],
                "grounding_score": 0.25,
                "suggested_fixes": [{"claim": "cures anxiety", "fix": "soften"}],
            }),
            json.dumps({
                "total_claims": 4, "supported_claims": 4,
                "unsupported_claims": [], "missing_citations": [],
                "grounding_score": 0.95,
            }),
        ]

    async def test_low_grounding_triggers_revision(self, brief):
        llm = ScriptedLLM(fact_checks=self._ungrounded_then_grounded())
        await generate_article(brief, FakeRetriever(), llm)
        assert "revise" in llm.kinds()

    async def test_revision_actually_changes_the_draft(self, brief):
        """
        The core fix. The original re-ran the fact-checker on the identical
        draft, so a rejected article could only ever be rejected again.
        """
        llm = ScriptedLLM(fact_checks=self._ungrounded_then_grounded())
        state = await generate_article(brief, FakeRetriever(), llm)
        assert state.revision_count == 1
        assert "Revised" in state.draft.content
        assert state.draft.content == llm.revision_response

    async def test_fact_check_reruns_after_revision(self, brief):
        llm = ScriptedLLM(fact_checks=self._ungrounded_then_grounded())
        await generate_article(brief, FakeRetriever(), llm)
        assert llm.kinds().count("fact_check") == 2

    async def test_counter_increments_once_per_revision(self, brief):
        """The original incremented twice per pass, halving the real budget."""
        llm = ScriptedLLM(fact_checks=self._ungrounded_then_grounded())
        state = await generate_article(brief, FakeRetriever(), llm)
        assert state.revision_count == 1

    async def test_persistently_ungrounded_stops_at_the_cap(self, brief):
        always_bad = json.dumps({
            "total_claims": 4, "supported_claims": 0,
            "unsupported_claims": ["bad claim"], "missing_citations": [],
            "grounding_score": 0.1,
            "suggested_fixes": [{"claim": "bad", "fix": "remove"}],
        })
        llm = ScriptedLLM(fact_checks=[always_bad])
        state = await generate_article(brief, FakeRetriever(), llm)

        assert state.revision_count == 2                    # agent_max_iterations
        assert llm.kinds().count("revise") == 2
        assert state.status == "completed"                  # escalates, not hangs
        assert state.final.ready_for_editor is False
        assert any("below" in n for n in state.final.editor_notes)


@pytest.mark.asyncio
class TestFailureHandling:
    async def test_unparseable_fact_check_is_not_a_pass(self, brief):
        """
        The original defaulted an unparseable response to 0.75/grounded=True,
        so a malformed reply silently cleared the safety gate.
        """
        llm = ScriptedLLM(fact_checks=["I could not analyze this article."])
        state = await generate_article(brief, FakeRetriever(), llm)

        assert state.fact_check.parse_failed is True
        assert state.fact_check.is_grounded is False
        assert state.fact_check.grounding_score == 0.0
        assert state.final.ready_for_editor is False

    async def test_unparseable_outline_falls_back_to_brief(self, brief):
        llm = ScriptedLLM(outline="Sorry, I can't do that.")
        state = await generate_article(brief, FakeRetriever(), llm)
        assert state.outline is not None
        assert len(state.outline.sections) == len(brief.key_points)
        assert state.final is not None

    async def test_tone_edit_cannot_drop_citations(self, brief):
        """The tone editor is told never to remove citations; enforce it."""
        stripped = json.dumps({
            "style_score": 0.9, "issues": [],
            "revised_content": "A prettier article with no citations at all.",
        })
        llm = ScriptedLLM(tone=stripped)
        state = await generate_article(brief, FakeRetriever(), llm)
        assert state.final.citations, "citations must survive tone editing"
        assert "[Source:" in state.final.content

    async def test_retrieval_failure_does_not_crash_run(self, brief):
        class BrokenRetriever:
            def retrieve(self, *a, **kw):
                raise RuntimeError("qdrant down")

        state = await generate_article(brief, BrokenRetriever(), ScriptedLLM())
        assert state.final is not None      # degrades, doesn't explode

    async def test_no_changes_keeps_original_draft(self, brief):
        llm = ScriptedLLM()
        state = await generate_article(brief, FakeRetriever(), llm)
        assert state.final.content == state.draft.content


@pytest.mark.asyncio
class TestProgressCallback:
    async def test_on_step_fires_per_node(self, brief):
        seen = []

        async def on_step(node, update):
            seen.append(node)

        await generate_article(brief, FakeRetriever(), ScriptedLLM(), on_step=on_step)
        assert "outline" in seen
        assert "write_section" in seen
        assert "fact_check" in seen
        assert "tone_edit" in seen
