"""
Evaluation Metrics
====================

Three of the four original scorers were measuring their own implementation
rather than the system:

1. COVERAGE was a case-insensitive substring test against expected phrases.
   A correct answer phrased differently scored zero. This is why q005 ("How
   does the Stress Support Program work?") reported 0.00 coverage in the
   benchmark while the README's own analysis noted the answer was "correct but
   overly general". Coverage is now embedding cosine similarity between the
   answer and each expected point — free, local, deterministic, and it credits
   paraphrase the way a human grader would.

2. HALLUCINATION was `"YES" in response.upper()`, so a judge replying "Yes,
   this is well grounded" scored as a hallucination — the substring appears in
   the word "Yes" regardless of what follows. It now asks for a bare verdict
   token and parses the first line strictly.

3. TONE looked for red-flag words with plain substring matching, so "cure"
   matched "procedure" and, worse, matched the compliant phrase "does not
   cure" that the style guide explicitly asks writers to use. Now word-boundary
   regex with negation-aware context.

CITATION ACCURACY was sound and is kept as-is.

Everything runs on the local embedding model and the existing LLM gateway —
no new dependencies, and the only API calls are the judge passes.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Claims that no phrasing makes acceptable for health content.
RED_FLAG_PATTERNS = [
    r"\bguarantee(?:d|s)?\b",
    r"\b100%\s+safe\b",
    r"\bmiracle\b",
    r"\bscientifically proven to cure\b",
    r"\bwill cure\b",
    r"\bcures?\s+(?:your|the|all)\b",
]

# Hedged, non-diagnostic phrasing the style guide asks for.
COMPLIANT_PATTERNS = [
    r"\btraditionally used\b",
    r"\bmay help\b",
    r"\bmay support\b",
    r"\bsupports?\b",
    r"\bconsult\b",
    r"\bpractitioner\b",
    r"\bnot a substitute\b",
]

# Phrases that legitimately contain a red-flag word while denying it.
NEGATION_CONTEXTS = [
    r"\b(?:do(?:es)? not|don't|doesn't|cannot|can't|never|no)\s+\w{0,12}\s?cure",
    r"\bnot (?:a )?(?:cure|guaranteed|miracle)\b",
    r"\bwithout guarantee",
]

REFUSAL_PATTERNS = [
    r"\b(?:not|no|couldn't|could not|cannot|can't|don't|do not)\b[^.]{0,60}"
    r"\b(?:find|have|contain|include|information|available|knowledge base|mention)\b",
    r"\bisn't (?:in|covered)\b",
    r"\bnot (?:in|covered by|available in) the (?:sources|knowledge base|context)\b",
]


@dataclass
class QueryEvaluation:
    """Scores for a single golden-set question."""
    id: str
    query: str
    category: str
    answer: str

    coverage_score: float = 0.0
    coverage_detail: Dict[str, float] = field(default_factory=dict)
    citation_accuracy: float = 0.0
    cited_sources: List[str] = field(default_factory=list)
    expected_sources: List[str] = field(default_factory=list)

    faithfulness: Optional[float] = None
    hallucinated: Optional[bool] = None
    judge_reason: str = ""

    tone_compliant: bool = False
    tone_red_flags: List[str] = field(default_factory=list)

    refused: bool = False
    is_negative_case: bool = False
    passed: bool = False
    latency_ms: float = 0.0

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "query": self.query,
            "category": self.category,
            "answer": self.answer,
            "coverage_score": round(self.coverage_score, 3),
            "coverage_detail": {k: round(v, 3) for k, v in self.coverage_detail.items()},
            "citation_accuracy": round(self.citation_accuracy, 3),
            "cited_sources": self.cited_sources,
            "expected_sources": self.expected_sources,
            "faithfulness": round(self.faithfulness, 3) if self.faithfulness is not None else None,
            "hallucinated": self.hallucinated,
            "judge_reason": self.judge_reason,
            "tone_compliant": self.tone_compliant,
            "tone_red_flags": self.tone_red_flags,
            "refused": self.refused,
            "is_negative_case": self.is_negative_case,
            "passed": self.passed,
            "latency_ms": round(self.latency_ms, 1),
        }


# ── Coverage ────────────────────────────────────────────────────

def score_coverage(
    answer: str,
    expected_points: List[str],
    embedder: Any,
    threshold: float = 0.55,
) -> tuple:
    """
    Semantic coverage: how many expected points the answer actually conveys.

    Each expected point is compared against the answer's most similar sentence
    rather than the whole answer — otherwise a long answer dilutes the vector
    of any single point and everything scores mid-range.

    Returns (score, {point: similarity}).
    """
    if not expected_points:
        return 1.0, {}
    if not answer.strip():
        return 0.0, {point: 0.0 for point in expected_points}

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if s.strip()]
    if not sentences:
        sentences = [answer]

    sentence_vectors = embedder.encode_documents(sentences)
    point_vectors = embedder.encode_documents(list(expected_points))

    detail: Dict[str, float] = {}
    for point, pvec in zip(expected_points, point_vectors):
        # Vectors are already L2-normalized, so a dot product is the cosine.
        best = max(
            (sum(a * b for a, b in zip(pvec, svec)) for svec in sentence_vectors),
            default=0.0,
        )
        # A literal appearance is unambiguous evidence — keep the old
        # substring check as a floor rather than replacing it outright.
        if point.lower() in answer.lower():
            best = max(best, 1.0)
        detail[point] = float(best)

    covered = sum(1 for v in detail.values() if v >= threshold)
    return covered / len(expected_points), detail


# ── Citations ───────────────────────────────────────────────────

def score_citations(cited: List[str], expected: List[str]) -> float:
    """
    Recall of the expected source documents.

    Unchanged from the original — this metric was correct.
    """
    if not expected:
        return 1.0
    cited_set = {c.strip() for c in cited}
    hits = sum(
        1 for exp in expected
        if any(exp in c or c in exp for c in cited_set)
    )
    return hits / len(expected)


# ── Tone ────────────────────────────────────────────────────────

def check_tone(answer: str) -> tuple:
    """
    Brand-voice compliance.

    Returns (compliant, red_flags_found).

    The original used substring matching on a red-flag list containing "cure",
    which fired on "procedure" and on "does not cure" — the exact hedged
    phrasing the style guide prescribes. Word boundaries plus negation
    awareness fix both.
    """
    lowered = answer.lower()

    negated_spans = []
    for pattern in NEGATION_CONTEXTS:
        negated_spans.extend(m.span() for m in re.finditer(pattern, lowered))

    def inside_negation(span) -> bool:
        return any(ns <= span[0] and span[1] <= ne + 12 for ns, ne in negated_spans)

    red_flags = []
    for pattern in RED_FLAG_PATTERNS:
        for match in re.finditer(pattern, lowered):
            if not inside_negation(match.span()):
                red_flags.append(match.group(0))

    if red_flags:
        return False, red_flags

    hedges = sum(1 for p in COMPLIANT_PATTERNS if re.search(p, lowered))
    return hedges >= 2, []


def check_refusal(answer: str) -> bool:
    """Does the answer decline to answer from the knowledge base?"""
    lowered = answer.lower()
    return any(re.search(p, lowered) for p in REFUSAL_PATTERNS)


# ── Faithfulness (LLM judge) ────────────────────────────────────

JUDGE_SYSTEM = """You are evaluating whether an AI answer is grounded in its source material.

Compare the ANSWER against the SOURCES. Judge only grounding — not whether the
answer is well written, and not whether it is true in general.

Rules:
- A claim restating or paraphrasing the sources is GROUNDED.
- A reasonable summary or synthesis of the sources is GROUNDED.
- A specific factual claim absent from the sources is NOT grounded.
- Generic safety advice ("consult a practitioner") is always GROUNDED.
- An answer that declines because the sources lack the information is GROUNDED.

Respond in exactly this format, nothing else:
VERDICT: GROUNDED or UNGROUNDED
SCORE: a number from 0.0 to 1.0 (fraction of claims supported)
REASON: one short sentence"""

JUDGE_USER = """SOURCES:
{sources}

ANSWER:
{answer}

Evaluate the grounding of the answer."""


def parse_judge_verdict(text: str) -> tuple:
    """
    Parse the judge's structured reply.

    The original did `"YES" in response.upper()`, which matched the "Yes" in
    "Yes, this is grounded" and scored a passing answer as a hallucination.
    This reads the labelled fields and treats an unparseable reply as unknown
    rather than guessing.
    """
    verdict, score, reason = None, None, ""

    for line in text.strip().splitlines():
        line = line.strip()
        if m := re.match(r"^VERDICT:\s*(GROUNDED|UNGROUNDED)\b", line, re.IGNORECASE):
            verdict = m.group(1).upper() == "GROUNDED"
        elif m := re.match(r"^SCORE:\s*([01](?:\.\d+)?)", line, re.IGNORECASE):
            score = float(m.group(1))
        elif m := re.match(r"^REASON:\s*(.+)", line, re.IGNORECASE):
            reason = m.group(1).strip()

    if verdict is None and score is not None:
        verdict = score >= 0.7
    if score is None and verdict is not None:
        score = 1.0 if verdict else 0.0

    return verdict, score, reason


async def judge_faithfulness(answer: str, sources: List[str], llm_provider: Any) -> tuple:
    """
    Ask the LLM whether the answer is grounded in the retrieved sources.

    Returns (is_grounded, score, reason). All None/empty if the judge fails —
    an unavailable judge is an unknown, not a pass and not a failure.
    """
    if not sources:
        return None, None, "no sources retrieved"

    try:
        text = await llm_provider.agenerate(
            [
                ("system", JUDGE_SYSTEM),
                ("user", JUDGE_USER.format(
                    sources="\n\n---\n\n".join(sources[:5]),
                    answer=answer,
                )),
            ],
            temperature=0.0,
        )
    except Exception as e:
        logger.warning(f"Judge call failed: {e}", extra={"component": "eval"})
        return None, None, f"judge unavailable: {e}"

    grounded, score, reason = parse_judge_verdict(text)
    if grounded is None:
        logger.warning(
            f"Judge reply did not parse: {text[:120]!r}",
            extra={"component": "eval"},
        )
        return None, None, "judge reply unparseable"
    return grounded, score, reason


# ── Aggregation ─────────────────────────────────────────────────

def aggregate(results: List[QueryEvaluation]) -> dict:
    """Roll per-question scores into the reported metrics."""
    if not results:
        return {}

    n = len(results)
    judged = [r for r in results if r.hallucinated is not None]
    positives = [r for r in results if not r.is_negative_case]
    negatives = [r for r in results if r.is_negative_case]

    return {
        "total_queries": n,
        "avg_coverage": sum(r.coverage_score for r in positives) / len(positives) if positives else 0.0,
        "avg_citation_accuracy": sum(r.citation_accuracy for r in results) / n,
        "hallucination_rate": (
            sum(1 for r in judged if r.hallucinated) / len(judged) if judged else None
        ),
        "avg_faithfulness": (
            sum(r.faithfulness for r in judged if r.faithfulness is not None) / len(judged)
            if judged else None
        ),
        "tone_compliance_rate": sum(1 for r in results if r.tone_compliant) / n,
        "negative_refusal_rate": (
            sum(1 for r in negatives if r.refused) / len(negatives) if negatives else None
        ),
        "judged_queries": len(judged),
        "unjudged_queries": n - len(judged),
        "avg_latency_ms": sum(r.latency_ms for r in results) / n,
    }
