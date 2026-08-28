"""
Tests for the evaluation scorers.

Each class here pins down one of the metric bugs that made the original
benchmark report misleading — the numbers in the README were partly measuring
the scorers, not the system.
"""

import pytest

from backend.app.services.evaluation import (
    aggregate, check_refusal, check_tone, parse_judge_verdict, QueryEvaluation,
    score_citations, score_coverage,
)


class FakeEmbedder:
    """
    Toy embedder: a text maps to a normalized bag-of-words vector over a fixed
    vocabulary, so overlapping wording yields high cosine similarity.
    """
    VOCAB = [
        "abhyanga", "shirodhara", "consultation", "massage", "oil", "therapy",
        "stress", "sleep", "ashwagandha", "program", "sessions", "practitioner",
        "digestion", "triphala", "price", "dollars",
    ]

    def encode_documents(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            v = [1.0 if word in lowered else 0.0 for word in self.VOCAB]
            norm = sum(x * x for x in v) ** 0.5
            vectors.append([x / norm for x in v] if norm else v)
        return vectors

    def encode_query(self, text):
        return self.encode_documents([text])[0]


@pytest.fixture
def embedder():
    return FakeEmbedder()


class TestCoverage:
    def test_literal_match_scores_full(self, embedder):
        score, _ = score_coverage(
            "The program includes Abhyanga and Shirodhara.",
            ["Abhyanga", "Shirodhara"],
            embedder,
        )
        assert score == 1.0

    def test_paraphrase_is_credited(self, embedder):
        """
        The q005 failure. The old scorer required the literal token; a correct
        answer using different wording scored 0.00.
        """
        answer = "Treatment begins with a practitioner consultation."
        score, detail = score_coverage(answer, ["consultation"], embedder)
        assert score == 1.0
        assert detail["consultation"] > 0.5

    def test_missing_content_scores_zero(self, embedder):
        score, _ = score_coverage(
            "This is about something entirely unrelated.",
            ["Abhyanga", "Shirodhara"],
            embedder,
        )
        assert score == 0.0

    def test_partial_coverage(self, embedder):
        score, _ = score_coverage(
            "The program includes Abhyanga massage.",
            ["Abhyanga", "price"],
            embedder,
        )
        assert score == 0.5

    def test_no_expected_points_is_vacuously_full(self, embedder):
        assert score_coverage("anything", [], embedder)[0] == 1.0

    def test_empty_answer_scores_zero(self, embedder):
        assert score_coverage("", ["Abhyanga"], embedder)[0] == 0.0

    def test_detail_is_reported_per_point(self, embedder):
        _, detail = score_coverage("Abhyanga therapy.", ["Abhyanga", "price"], embedder)
        assert set(detail) == {"Abhyanga", "price"}


class TestCitations:
    def test_all_expected_cited(self):
        assert score_citations(["doc_a", "doc_b"], ["doc_a", "doc_b"]) == 1.0

    def test_partial(self):
        assert score_citations(["doc_a"], ["doc_a", "doc_b"]) == 0.5

    def test_none_expected_is_full(self):
        assert score_citations([], []) == 1.0

    def test_substring_match_counts(self):
        assert score_citations(["product_ashwagandha_internal"], ["ashwagandha"]) == 1.0


class TestTone:
    def test_compliant_hedged_language(self):
        ok, flags = check_tone(
            "Ashwagandha is traditionally used to support the stress response. "
            "Consult a qualified practitioner."
        )
        assert ok is True
        assert flags == []

    def test_procedure_does_not_trip_the_cure_flag(self):
        """
        The original matched "cure" as a substring, so "procedure" was a red
        flag. This is the tone-compliance false-negative from the benchmark.
        """
        ok, flags = check_tone(
            "This procedure is traditionally used to support balance. "
            "Please consult a practitioner."
        )
        assert flags == []
        assert ok is True

    def test_does_not_cure_is_compliant(self):
        """
        The style guide asks writers to say a product "does not cure" — the
        old checker flagged that exact compliant phrasing as a violation.
        """
        ok, flags = check_tone(
            "Ayurveda does not cure disease; it is traditionally used to "
            "support balance. Consult a qualified practitioner."
        )
        assert flags == []
        assert ok is True

    def test_genuine_cure_claim_is_flagged(self):
        ok, flags = check_tone("This herb cures your anxiety and will cure insomnia.")
        assert ok is False
        assert flags

    def test_guarantee_is_flagged(self):
        ok, flags = check_tone("Guaranteed results in one week.")
        assert ok is False

    def test_miracle_is_flagged(self):
        ok, _ = check_tone("A miracle herb, 100% safe for everyone.")
        assert ok is False

    def test_unhedged_answer_is_not_compliant(self):
        ok, flags = check_tone("Take three tablets daily.")
        assert flags == []
        assert ok is False   # no red flags, but no hedging either


class TestRefusal:
    def test_detects_not_in_knowledge_base(self):
        assert check_refusal(
            "I couldn't find relevant information in the knowledge base."
        ) is True

    def test_detects_sources_do_not_contain(self):
        assert check_refusal(
            "The sources do not contain information about pricing."
        ) is True

    def test_normal_answer_is_not_a_refusal(self):
        assert check_refusal(
            "Ashwagandha is traditionally used to support the stress response."
        ) is False


class TestJudgeParsing:
    def test_parses_grounded(self):
        grounded, score, reason = parse_judge_verdict(
            "VERDICT: GROUNDED\nSCORE: 0.9\nREASON: All claims cite sources."
        )
        assert grounded is True
        assert score == 0.9
        assert "cite sources" in reason

    def test_parses_ungrounded(self):
        grounded, score, _ = parse_judge_verdict(
            "VERDICT: UNGROUNDED\nSCORE: 0.3\nREASON: Invented a dosage."
        )
        assert grounded is False
        assert score == 0.3

    def test_yes_in_prose_does_not_flip_the_verdict(self):
        """
        The original test was `"YES" in response.upper()`, so a judge replying
        "Yes, this is well grounded" was recorded as a hallucination.
        """
        grounded, _, _ = parse_judge_verdict(
            "VERDICT: GROUNDED\nSCORE: 0.95\nREASON: Yes, every claim is supported."
        )
        assert grounded is True

    def test_unparseable_reply_is_unknown_not_pass(self):
        grounded, score, _ = parse_judge_verdict("I'm not sure how to evaluate this.")
        assert grounded is None
        assert score is None

    def test_score_alone_infers_verdict(self):
        grounded, score, _ = parse_judge_verdict("SCORE: 0.85")
        assert grounded is True
        assert score == 0.85


class TestAggregate:
    def _result(self, **kw):
        base = dict(id="q", query="q", category="general", answer="a")
        base.update(kw)
        return QueryEvaluation(**base)

    def test_negative_cases_excluded_from_coverage(self):
        """
        A negative case has no expected content, so folding its coverage into
        the average would distort the number in either direction.
        """
        metrics = aggregate([
            self._result(coverage_score=0.8),
            self._result(coverage_score=0.0, is_negative_case=True),
        ])
        assert metrics["avg_coverage"] == 0.8

    def test_unjudged_queries_excluded_from_hallucination_rate(self):
        metrics = aggregate([
            self._result(hallucinated=False),
            self._result(hallucinated=True),
            self._result(hallucinated=None),   # judge failed
        ])
        assert metrics["hallucination_rate"] == 0.5
        assert metrics["judged_queries"] == 2
        assert metrics["unjudged_queries"] == 1

    def test_hallucination_rate_none_when_nothing_judged(self):
        metrics = aggregate([self._result(hallucinated=None)])
        assert metrics["hallucination_rate"] is None

    def test_negative_refusal_rate(self):
        metrics = aggregate([
            self._result(is_negative_case=True, refused=True),
            self._result(is_negative_case=True, refused=False),
        ])
        assert metrics["negative_refusal_rate"] == 0.5
