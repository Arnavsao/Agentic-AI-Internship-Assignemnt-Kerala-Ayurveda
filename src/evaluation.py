"""
Evaluation Framework for Kerala Ayurveda RAG & Agent System

Implements:
- Golden set management
- Automated scoring (grounding, citations, tone)
- Metrics tracking over time
- Human-in-the-loop evaluation interface
"""

import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import statistics

from langchain_google_genai import ChatGoogleGenerativeAI
from src.rag_system import AyurvedaRAGSystem, QueryResponse
from src.agent_workflow import FinalArticle


@dataclass
class GoldenExample:
    """A golden set example for evaluation"""
    id: str
    query: str
    expected_answer_contains: List[str]  # Key points that should be in answer
    expected_sources: List[str]  # Doc IDs that should be cited
    category: str  # "product", "treatment", "concept", "faq"
    notes: str = ""


@dataclass
class EvaluationResult:
    """Result of evaluating a single example"""
    example_id: str
    query: str
    answer: str
    citations: List[str]

    # Automated scores
    coverage_score: float  # 0-1: How many expected points were covered
    citation_accuracy: float  # 0-1: Correct sources cited
    hallucination_detected: bool
    tone_appropriate: bool

    # Human ratings (optional)
    human_usefulness: Optional[int] = None  # 1-5 scale
    human_safety: Optional[int] = None  # 1-5 scale
    human_notes: str = ""

    timestamp: str = ""


@dataclass
class ArticleEvaluation:
    """Evaluation results for generated article"""
    article_id: str
    brief_topic: str

    # Automated scores
    grounding_score: float
    style_score: float
    citation_count: int
    has_safety_disclaimer: bool

    # Structure checks
    has_clear_sections: bool
    appropriate_length: bool  # Within 20% of target

    # Human editor ratings
    editor_usefulness: Optional[int] = None  # 1-5: Ready to publish with minor edits?
    editor_accuracy: Optional[int] = None  # 1-5: Facts correct?
    editor_notes: str = ""

    timestamp: str = ""


class GoldenSetManager:
    """Manages golden examples for evaluation"""

    def __init__(self, golden_set_path: str = "golden_set.json"):
        self.golden_set_path = Path(golden_set_path)
        self.examples: List[GoldenExample] = []
        self.load_golden_set()

    def load_golden_set(self):
        """Load golden set from disk"""
        if self.golden_set_path.exists():
            with open(self.golden_set_path, 'r') as f:
                data = json.load(f)
                self.examples = [GoldenExample(**ex) for ex in data]
        else:
            # Create default golden set
            self.examples = self.create_default_golden_set()
            self.save_golden_set()

    def save_golden_set(self):
        """Save golden set to disk"""
        with open(self.golden_set_path, 'w') as f:
            json.dump([asdict(ex) for ex in self.examples], f, indent=2)

    def create_default_golden_set(self) -> List[GoldenExample]:
        """Create initial golden set for Kerala Ayurveda"""
        return [
            GoldenExample(
                id="q001",
                query="What are the benefits of Ashwagandha for stress?",
                expected_answer_contains=[
                    "adapt to stress",
                    "emotional balance",
                    "restful sleep",
                    "traditionally used"
                ],
                expected_sources=["product_ashwagandha_tablets_internal"],
                category="product",
                notes="Should mention stress resilience without claiming to cure anxiety"
            ),
            GoldenExample(
                id="q002",
                query="Are there any contraindications for Triphala?",
                expected_answer_contains=[
                    "chronic digestive disease",
                    "pregnancy",
                    "consult",
                    "healthcare provider"
                ],
                expected_sources=["product_triphala_capsules_internal"],
                category="product",
                notes="Must include safety warnings"
            ),
            GoldenExample(
                id="q003",
                query="Can Ayurveda help with stress and sleep?",
                expected_answer_contains=[
                    "daily routines",
                    "herbs",
                    "not replace",
                    "complement"
                ],
                expected_sources=["faq_general_ayurveda_patients"],
                category="faq",
                notes="Should be balanced - helps but doesn't replace medical care"
            ),
            GoldenExample(
                id="q004",
                query="What is Vata dosha?",
                expected_answer_contains=[
                    "movement",
                    "light",
                    "dry",
                    "tendencies"
                ],
                expected_sources=["dosha_guide_vata_pitta_kapha"],
                category="concept",
                notes="Should avoid rigid labels, emphasize patterns"
            ),
            GoldenExample(
                id="q005",
                query="How does the Stress Support Program work at Kerala Ayurveda clinics?",
                expected_answer_contains=[
                    "consultation",
                    "Abhyanga",
                    "Shirodhara",
                    "not a substitute"
                ],
                expected_sources=["treatment_stress_support_program"],
                category="treatment",
                notes="Must clarify this is complementary, not psychiatric treatment"
            ),
        ]

    def add_example(self, example: GoldenExample):
        """Add new example to golden set"""
        self.examples.append(example)
        self.save_golden_set()


class RAGEvaluator:
    """Evaluates RAG system performance"""

    def __init__(self, rag_system: AyurvedaRAGSystem):
        self.rag = rag_system
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            google_api_key=gemini_api_key
        )
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)

    def evaluate_coverage(self, answer: str, expected_points: List[str]) -> float:
        """Score how many expected points are covered in answer"""
        covered = 0
        for point in expected_points:
            # Simple substring match (could be improved with semantic similarity)
            if point.lower() in answer.lower():
                covered += 1
        return covered / len(expected_points) if expected_points else 0.0

    def evaluate_citations(self, cited_docs: List[str], expected_docs: List[str]) -> float:
        """Score citation accuracy"""
        if not expected_docs:
            return 1.0

        cited_set = set(cited_docs)
        expected_set = set(expected_docs)

        # Precision: what % of cited docs are in expected?
        if not cited_set:
            return 0.0

        correct_citations = len(cited_set.intersection(expected_set))
        return correct_citations / len(expected_set)

    def detect_hallucination(self, answer: str, retrieved_chunks: List[str]) -> bool:
        """Use LLM to detect if answer contains info not in sources"""
        context = "\n\n".join(retrieved_chunks)

        prompt = f"""Compare the answer to the source context.

Source Context:
{context}

Answer:
{answer}

Does the answer contain any factual claims that are NOT supported by the source context?
Consider:
- Made-up statistics or numbers
- Benefits/effects not mentioned in sources
- Products/treatments not in sources

Respond with just: YES (hallucination detected) or NO (answer is grounded)"""

        response = self.llm.invoke(prompt)
        return "YES" in response.content.upper()

    def check_tone(self, answer: str) -> bool:
        """Check if answer follows Kerala Ayurveda tone guidelines"""
        # Simple rule-based checks (could be enhanced with LLM)
        red_flags = [
            "guaranteed",
            "cure",
            "100% safe",
            "miracle",
            "scientifically proven to cure"
        ]

        for flag in red_flags:
            if flag.lower() in answer.lower():
                return False

        # Positive signals
        good_phrases = [
            "traditionally used",
            "may help",
            "support",
            "consult"
        ]

        found_good = sum(1 for phrase in good_phrases if phrase in answer.lower())
        return found_good >= 2

    def evaluate_example(self, example: GoldenExample) -> EvaluationResult:
        """Evaluate RAG system on a single golden example"""
        # Get answer from RAG system
        response = self.rag.answer_user_query(example.query)

        # Extract citation doc IDs
        cited_docs = [c.doc_id for c in response.citations]

        # Compute scores
        coverage = self.evaluate_coverage(response.answer, example.expected_answer_contains)
        citation_acc = self.evaluate_citations(cited_docs, example.expected_sources)
        hallucination = self.detect_hallucination(response.answer, response.retrieved_chunks)
        tone_ok = self.check_tone(response.answer)

        return EvaluationResult(
            example_id=example.id,
            query=example.query,
            answer=response.answer,
            citations=cited_docs,
            coverage_score=coverage,
            citation_accuracy=citation_acc,
            hallucination_detected=hallucination,
            tone_appropriate=tone_ok,
            timestamp=datetime.now().isoformat()
        )

    def evaluate_golden_set(self, golden_set: List[GoldenExample]) -> Dict:
        """Evaluate entire golden set and return aggregate metrics"""
        results = []

        for example in golden_set:
            print(f"Evaluating: {example.id} - {example.query}")
            result = self.evaluate_example(example)
            results.append(result)

        # Aggregate metrics
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "total_examples": len(results),
            "avg_coverage_score": statistics.mean([r.coverage_score for r in results]),
            "avg_citation_accuracy": statistics.mean([r.citation_accuracy for r in results]),
            "hallucination_rate": sum([r.hallucination_detected for r in results]) / len(results),
            "tone_compliance_rate": sum([r.tone_appropriate for r in results]) / len(results),
            "detailed_results": [asdict(r) for r in results]
        }

        # Save results
        result_file = self.results_dir / f"rag_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\nResults saved to: {result_file}")
        return metrics


class ArticleEvaluator:
    """Evaluates generated articles"""

    def __init__(self):
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            google_api_key=gemini_api_key
        )
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)

    def evaluate_article(self, article: FinalArticle, target_word_count: int) -> ArticleEvaluation:
        """Evaluate a generated article"""
        content = article.content
        word_count = len(content.split())

        # Structure checks
        has_sections = bool(len([line for line in content.split('\n') if line.startswith('##')]) >= 3)
        length_ok = 0.8 <= (word_count / target_word_count) <= 1.2

        # Safety disclaimer check
        safety_keywords = ["consult", "healthcare provider", "not a substitute", "informational purposes"]
        has_disclaimer = any(kw in content.lower() for kw in safety_keywords)

        return ArticleEvaluation(
            article_id=f"article_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            brief_topic=article.workflow_metadata.get('brief', {}).get('topic', 'Unknown'),
            grounding_score=article.fact_check_score,
            style_score=article.style_score,
            citation_count=len(article.citations),
            has_safety_disclaimer=has_disclaimer,
            has_clear_sections=has_sections,
            appropriate_length=length_ok,
            timestamp=datetime.now().isoformat()
        )


class MetricsTracker:
    """Track metrics over time for continuous improvement"""

    def __init__(self, metrics_file: str = "metrics_history.jsonl"):
        self.metrics_file = Path(metrics_file)

    def log_metrics(self, metrics: Dict, system: str = "rag"):
        """Append metrics to history"""
        metrics_entry = {
            "timestamp": datetime.now().isoformat(),
            "system": system,
            **metrics
        }

        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics_entry) + '\n')

    def get_metrics_history(self, system: str = "rag", last_n: int = 10) -> List[Dict]:
        """Retrieve recent metrics history"""
        if not self.metrics_file.exists():
            return []

        history = []
        with open(self.metrics_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                if entry.get('system') == system:
                    history.append(entry)

        return history[-last_n:]

    def print_metrics_summary(self, system: str = "rag"):
        """Print summary of recent metrics"""
        history = self.get_metrics_history(system)

        if not history:
            print(f"No metrics history for {system}")
            return

        print(f"\n{'='*60}")
        print(f"Metrics Summary - {system.upper()}")
        print(f"{'='*60}")
        print(f"Total runs: {len(history)}")

        if system == "rag":
            recent = history[-1]
            print(f"\nLatest Results:")
            print(f"  Coverage Score: {recent.get('avg_coverage_score', 0):.2f}")
            print(f"  Citation Accuracy: {recent.get('avg_citation_accuracy', 0):.2f}")
            print(f"  Hallucination Rate: {recent.get('hallucination_rate', 0):.2%}")
            print(f"  Tone Compliance: {recent.get('tone_compliance_rate', 0):.2%}")

        print(f"{'='*60}\n")


def main():
    """Run evaluation on RAG system"""
    import dotenv
    dotenv.load_dotenv()

    # Initialize systems
    print("Initializing RAG system...")
    rag = AyurvedaRAGSystem()
    rag.load_and_index_content()

    # Load golden set
    print("\nLoading golden set...")
    golden_manager = GoldenSetManager()
    print(f"Loaded {len(golden_manager.examples)} golden examples")

    # Evaluate
    print("\n" + "="*80)
    print("RUNNING EVALUATION ON GOLDEN SET")
    print("="*80 + "\n")

    evaluator = RAGEvaluator(rag)
    metrics = evaluator.evaluate_golden_set(golden_manager.examples)

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nAverage Coverage Score: {metrics['avg_coverage_score']:.2%}")
    print(f"Average Citation Accuracy: {metrics['avg_citation_accuracy']:.2%}")
    print(f"Hallucination Rate: {metrics['hallucination_rate']:.2%}")
    print(f"Tone Compliance Rate: {metrics['tone_compliance_rate']:.2%}")

    # Log to metrics tracker
    tracker = MetricsTracker()
    tracker.log_metrics(metrics, system="rag")
    tracker.print_metrics_summary("rag")


if __name__ == "__main__":
    main()
