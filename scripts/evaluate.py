"""
Evaluation Runner
===================

    python -m scripts.evaluate                  # full golden set
    python -m scripts.evaluate --no-judge       # skip LLM judging (no API calls)
    python -m scripts.evaluate --ids q005 q013  # just these questions
    python -m scripts.evaluate --json out.json  # also write raw results

Runs each golden-set question through the RAG pipeline with caching disabled
and scores the answers. Coverage and tone are computed locally; only the
faithfulness judge makes API calls, and those are paced to stay inside the
Gemini free tier.
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.core.config import get_settings
from backend.app.core.logging import setup_logging
from backend.app.services.evaluation import (
    QueryEvaluation, aggregate, check_refusal, check_tone, judge_faithfulness,
    score_citations, score_coverage,
)
from backend.app.services.rag.embeddings import get_dense_embedder
from backend.app.services.rag.pipeline import get_rag_pipeline

logger = logging.getLogger(__name__)

DEFAULT_GOLDEN_SET = Path("eval/golden_set.json")

# Pause between judge calls. The free tier allows 15 requests/minute.
JUDGE_DELAY_SECONDS = 4.5


async def evaluate_one(item: dict, pipeline, embedder, use_judge: bool) -> QueryEvaluation:
    """Answer one golden-set question and score it."""
    is_negative = item.get("category") == "negative"

    response = await pipeline.aquery(item["query"], use_cache=False)
    answer = response.answer

    coverage, detail = score_coverage(
        answer, item.get("expected_answer_contains", []), embedder
    )
    cited = [c.doc_id for c in response.citations]
    citation_accuracy = score_citations(cited, item.get("expected_sources", []))
    tone_ok, red_flags = check_tone(answer)
    refused = check_refusal(answer)

    result = QueryEvaluation(
        id=item["id"],
        query=item["query"],
        category=item.get("category", "general"),
        answer=answer,
        coverage_score=coverage,
        coverage_detail=detail,
        citation_accuracy=citation_accuracy,
        cited_sources=cited,
        expected_sources=item.get("expected_sources", []),
        tone_compliant=tone_ok,
        tone_red_flags=red_flags,
        refused=refused,
        is_negative_case=is_negative,
        latency_ms=response.latency_ms,
    )

    if use_judge:
        grounded, score, reason = await judge_faithfulness(
            answer, response.retrieved_chunks, pipeline.llm_provider
        )
        if grounded is not None:
            result.hallucinated = not grounded
            result.faithfulness = score
        result.judge_reason = reason

    # A negative case passes by refusing; a positive one by covering the
    # expected points with correct citations and compliant tone.
    if is_negative:
        result.passed = refused and not red_flags
    else:
        result.passed = (
            coverage >= 0.6
            and citation_accuracy >= 0.5
            and tone_ok
            and result.hallucinated is not True
        )

    return result


async def run(items: List[dict], use_judge: bool) -> tuple:
    settings = get_settings()
    pipeline = get_rag_pipeline()

    print(f"Initializing pipeline (collection: {settings.qdrant_collection})...")
    await asyncio.to_thread(pipeline.initialize)
    embedder = get_dense_embedder()
    print(f"Ready: {pipeline.chunk_count} chunks indexed\n")

    results = []
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item['id']}: {item['query'][:60]}")
        try:
            result = await evaluate_one(item, pipeline, embedder, use_judge)
        except Exception as e:
            logger.error(f"{item['id']} failed: {e}", exc_info=True)
            print(f"    ERROR: {e}")
            continue

        results.append(result)
        flag = "PASS" if result.passed else "FAIL"
        print(
            f"    {flag}  coverage={result.coverage_score:.2f} "
            f"citations={result.citation_accuracy:.2f} "
            f"tone={'ok' if result.tone_compliant else 'NO'}"
            + (f" faithful={result.faithfulness:.2f}"
               if result.faithfulness is not None else "")
        )
        if result.tone_red_flags:
            print(f"          red flags: {', '.join(result.tone_red_flags)}")

        if use_judge and i < len(items):
            await asyncio.sleep(JUDGE_DELAY_SECONDS)

    return results, aggregate(results)


def print_report(results: List[QueryEvaluation], metrics: dict) -> None:
    def pct(v):
        return "n/a" if v is None else f"{v:.0%}"

    targets = {
        "avg_coverage": (0.60, "higher"),
        "avg_citation_accuracy": (0.50, "higher"),
        "hallucination_rate": (0.20, "lower"),
        "tone_compliance_rate": (0.80, "higher"),
    }

    print("\n" + "=" * 66)
    print("EVALUATION REPORT")
    print("=" * 66)
    print(f"{'Metric':<28} {'Score':>10} {'Target':>10} {'Status':>12}")
    print("-" * 66)

    for key, (target, direction) in targets.items():
        value = metrics.get(key)
        if value is None:
            print(f"{key:<28} {'n/a':>10} {target:>10.0%} {'skipped':>12}")
            continue
        ok = value >= target if direction == "higher" else value <= target
        print(f"{key:<28} {pct(value):>10} {target:>10.0%} "
              f"{('MET' if ok else 'MISSED'):>12}")

    if (v := metrics.get("avg_faithfulness")) is not None:
        print(f"{'avg_faithfulness':<28} {v:>10.2f} {'—':>10} {'':>12}")
    if (v := metrics.get("negative_refusal_rate")) is not None:
        print(f"{'negative_refusal_rate':<28} {pct(v):>10} {1.0:>10.0%} "
              f"{('MET' if v >= 1.0 else 'MISSED'):>12}")

    print("-" * 66)
    print(f"{'avg_latency_ms':<28} {metrics.get('avg_latency_ms', 0):>10.0f}")
    if metrics.get("unjudged_queries"):
        print(f"{'unjudged (judge failed)':<28} {metrics['unjudged_queries']:>10}")

    print("\nPER-QUERY")
    print("-" * 66)
    print(f"{'ID':<6} {'Cat':<14} {'Cov':>5} {'Cite':>5} {'Tone':>5} {'Faith':>6}  Result")
    for r in results:
        print(
            f"{r.id:<6} {r.category:<14} "
            f"{r.coverage_score:>5.2f} {r.citation_accuracy:>5.2f} "
            f"{'ok' if r.tone_compliant else 'NO':>5} "
            f"{(f'{r.faithfulness:.2f}' if r.faithfulness is not None else '—'):>6}  "
            f"{'PASS' if r.passed else 'FAIL'}"
        )

    passed = sum(1 for r in results if r.passed)
    print("-" * 66)
    print(f"{passed}/{len(results)} queries passed")
    print("=" * 66)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline")
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET)
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM faithfulness judging (no API calls)")
    parser.add_argument("--ids", nargs="*", help="Only evaluate these question IDs")
    parser.add_argument("--json", type=Path, help="Write raw results here")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(environment=settings.environment, debug=False)

    if not args.golden_set.exists():
        print(f"Golden set not found: {args.golden_set}", file=sys.stderr)
        return 1

    items = json.loads(args.golden_set.read_text())
    if args.ids:
        items = [i for i in items if i["id"] in set(args.ids)]
        if not items:
            print(f"No questions matched: {args.ids}", file=sys.stderr)
            return 1

    results, metrics = asyncio.run(run(items, use_judge=not args.no_judge))
    if not results:
        print("No results produced.", file=sys.stderr)
        return 1

    print_report(results, metrics)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "collection": settings.qdrant_collection,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "gemini_model": settings.gemini_model,
        "judge_enabled": not args.no_judge,
        "metrics": metrics,
        "results": [r.as_dict() for r in results],
    }

    out_dir = Path("evaluation_results")
    out_dir.mkdir(exist_ok=True)
    stamped = out_dir / f"rag_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    stamped.write_text(json.dumps(payload, indent=2))
    print(f"\nResults written to {stamped}")

    if args.json:
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"Results written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
