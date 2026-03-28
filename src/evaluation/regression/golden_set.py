"""Golden test set management — run ground-truth queries and score responses."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from src.models.queries import ResearchQuery
from src.retrieval.pipeline import ResearchPipeline

try:
    from rouge_score import rouge_scorer

    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False


@dataclass
class GoldenResult:
    """Result of evaluating a single golden test case."""

    test_id: str
    query: str
    query_type: str
    difficulty: str
    answer_relevance: float  # ROUGE-L against expected
    source_accuracy: float
    completeness: float  # How many expected_answer_contains were found
    confidence_calibration: float
    latency_ms: float
    passed: bool

    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "query": self.query,
            "query_type": self.query_type,
            "difficulty": self.difficulty,
            "answer_relevance": round(self.answer_relevance, 4),
            "source_accuracy": round(self.source_accuracy, 4),
            "completeness": round(self.completeness, 4),
            "confidence_calibration": round(self.confidence_calibration, 4),
            "latency_ms": round(self.latency_ms, 2),
            "passed": self.passed,
        }


@dataclass
class GoldenSetReport:
    """Aggregate results from a golden test set run."""

    pipeline_type: str  # "rag" or "agent"
    total: int = 0
    passed: int = 0
    failed: int = 0
    results: list[GoldenResult] = field(default_factory=list)
    avg_relevance: float = 0.0
    avg_completeness: float = 0.0
    avg_source_accuracy: float = 0.0
    avg_confidence_calibration: float = 0.0
    avg_latency_ms: float = 0.0
    pass_rate: float = 0.0

    # Per-type breakdown
    by_type: dict[str, dict] = field(default_factory=dict)
    by_difficulty: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "pipeline_type": self.pipeline_type,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.pass_rate, 4),
            "avg_relevance": round(self.avg_relevance, 4),
            "avg_completeness": round(self.avg_completeness, 4),
            "avg_source_accuracy": round(self.avg_source_accuracy, 4),
            "avg_confidence_calibration": round(self.avg_confidence_calibration, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "by_type": self.by_type,
            "by_difficulty": self.by_difficulty,
            "results": [r.to_dict() for r in self.results],
        }


class GoldenSetRunner:
    """Run the golden test set and score each response on multiple dimensions."""

    def __init__(
        self,
        pipeline: ResearchPipeline | None = None,
        pass_threshold: float = 0.5,
    ):
        self.pipeline = pipeline or ResearchPipeline()
        self.pass_threshold = pass_threshold
        self._rouge = None
        if _ROUGE_AVAILABLE:
            self._rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def run(
        self,
        golden_path: Path | None = None,
        pipeline_type: str = "rag",
    ) -> GoldenSetReport:
        """Run all golden queries and produce a scored report.

        Args:
            golden_path: Path to golden_queries.json.
            pipeline_type: "rag" or "agent".
        """
        from config.settings import settings

        if golden_path is None:
            golden_path = settings.data_dir / "evaluation" / "golden_queries.json"

        test_cases = json.loads(golden_path.read_text())
        results: list[GoldenResult] = []

        for case in test_cases:
            logger.info(f"Golden test [{case['id']}]: {case['query'][:60]}...")
            if pipeline_type == "agent":
                result = self._evaluate_agent(case)
            else:
                result = self._evaluate_rag(case)
            results.append(result)

        return self._aggregate(results, pipeline_type)

    def _evaluate_rag(self, case: dict) -> GoldenResult:
        """Evaluate a single golden case against the RAG pipeline."""
        t0 = time.perf_counter()
        try:
            query = ResearchQuery(question=case["query"])
            response = self.pipeline.query(query)
            latency = (time.perf_counter() - t0) * 1000

            answer = response.answer
            expected_contains = case.get("expected_answer_contains", [])

            completeness = self._score_completeness(answer, expected_contains)
            relevance = self._score_relevance(answer, case)
            source_accuracy = self._score_sources(response.sources, case)
            calibration = self._score_calibration(response.confidence, completeness)

            overall = (relevance + completeness + source_accuracy) / 3
            passed = overall >= self.pass_threshold

            return GoldenResult(
                test_id=case["id"],
                query=case["query"],
                query_type=case.get("query_type", "unknown"),
                difficulty=case.get("difficulty", "unknown"),
                answer_relevance=relevance,
                source_accuracy=source_accuracy,
                completeness=completeness,
                confidence_calibration=calibration,
                latency_ms=latency,
                passed=passed,
            )
        except Exception as e:
            logger.error(f"Golden test failed [{case['id']}]: {e}")
            return GoldenResult(
                test_id=case["id"],
                query=case["query"],
                query_type=case.get("query_type", "unknown"),
                difficulty=case.get("difficulty", "unknown"),
                answer_relevance=0.0,
                source_accuracy=0.0,
                completeness=0.0,
                confidence_calibration=0.0,
                latency_ms=(time.perf_counter() - t0) * 1000,
                passed=False,
            )

    def _evaluate_agent(self, case: dict) -> GoldenResult:
        """Evaluate a golden case against the agent pipeline."""
        from src.agents.orchestrator import run_agent_query

        t0 = time.perf_counter()
        try:
            result = run_agent_query(case["query"])
            latency = (time.perf_counter() - t0) * 1000

            answer = result.get("synthesis", "")
            expected_contains = case.get("expected_answer_contains", [])

            completeness = self._score_completeness(answer, expected_contains)
            relevance = self._score_relevance(answer, case)

            # Source accuracy from corpus results
            corpus_results = result.get("corpus_results", [])
            source_accuracy = min(len(corpus_results) / 3, 1.0) if corpus_results else 0.0

            confidence = result.get("confidence", 0.0)
            calibration = self._score_calibration(confidence, completeness)

            overall = (relevance + completeness + source_accuracy) / 3
            passed = overall >= self.pass_threshold

            return GoldenResult(
                test_id=case["id"],
                query=case["query"],
                query_type=case.get("query_type", "unknown"),
                difficulty=case.get("difficulty", "unknown"),
                answer_relevance=relevance,
                source_accuracy=source_accuracy,
                completeness=completeness,
                confidence_calibration=calibration,
                latency_ms=latency,
                passed=passed,
            )
        except Exception as e:
            logger.error(f"Agent golden test failed [{case['id']}]: {e}")
            return GoldenResult(
                test_id=case["id"],
                query=case["query"],
                query_type=case.get("query_type", "unknown"),
                difficulty=case.get("difficulty", "unknown"),
                answer_relevance=0.0,
                source_accuracy=0.0,
                completeness=0.0,
                confidence_calibration=0.0,
                latency_ms=(time.perf_counter() - t0) * 1000,
                passed=False,
            )

    def _score_completeness(self, answer: str, expected_contains: list[str]) -> float:
        """Score what fraction of expected terms appear in the answer."""
        if not expected_contains:
            return 1.0
        answer_lower = answer.lower()
        hits = sum(1 for term in expected_contains if term.lower() in answer_lower)
        return hits / len(expected_contains)

    def _score_relevance(self, answer: str, case: dict) -> float:
        """Score answer relevance using ROUGE-L if available, else keyword overlap."""
        if self._rouge and case.get("expected_answer_contains"):
            reference = " ".join(case["expected_answer_contains"])
            scores = self._rouge.score(reference, answer)
            return scores["rougeL"].fmeasure
        # Fallback: check if query keywords appear in answer
        query_words = set(case["query"].lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(query_words & answer_words)
        return min(overlap / max(len(query_words), 1), 1.0)

    def _score_sources(self, sources, case: dict) -> float:
        """Score source quality — checks if retrieved sources match expected types."""
        expected_source_types = case.get("expected_sources_contain", [])
        if not expected_source_types:
            return 1.0  # No source verification needed — pass by default
        if not sources:
            return 0.0  # Sources expected but none retrieved — fail

        source_files = [s.source_file.lower() for s in sources]
        source_types = [
            s.source_type.value.lower()
            if hasattr(s.source_type, "value")
            else str(s.source_type).lower()
            for s in sources
        ]
        all_source_text = " ".join(source_files + source_types)

        hits = sum(1 for t in expected_source_types if t.lower() in all_source_text)
        return hits / len(expected_source_types)

    def _score_calibration(self, confidence: float, actual_quality: float) -> float:
        """Score confidence calibration — how well confidence predicts actual quality."""
        # Perfect calibration: confidence ≈ actual quality
        # Score = 1 - |confidence - quality|
        return max(0.0, 1.0 - abs(confidence - actual_quality))

    def _aggregate(self, results: list[GoldenResult], pipeline_type: str) -> GoldenSetReport:
        """Compute aggregate statistics from golden results."""
        n = len(results) or 1
        passed_list = [r for r in results if r.passed]

        # By type breakdown
        by_type: dict[str, dict] = {}
        for r in results:
            if r.query_type not in by_type:
                by_type[r.query_type] = {"total": 0, "passed": 0, "avg_completeness": 0.0}
            by_type[r.query_type]["total"] += 1
            if r.passed:
                by_type[r.query_type]["passed"] += 1
            by_type[r.query_type]["avg_completeness"] += r.completeness

        for qt in by_type:
            total = by_type[qt]["total"]
            by_type[qt]["avg_completeness"] = round(by_type[qt]["avg_completeness"] / total, 4)
            by_type[qt]["pass_rate"] = round(by_type[qt]["passed"] / total, 4)

        # By difficulty
        by_difficulty: dict[str, dict] = {}
        for r in results:
            if r.difficulty not in by_difficulty:
                by_difficulty[r.difficulty] = {"total": 0, "passed": 0}
            by_difficulty[r.difficulty]["total"] += 1
            if r.passed:
                by_difficulty[r.difficulty]["passed"] += 1

        for d in by_difficulty:
            total = by_difficulty[d]["total"]
            by_difficulty[d]["pass_rate"] = round(by_difficulty[d]["passed"] / total, 4)

        return GoldenSetReport(
            pipeline_type=pipeline_type,
            total=len(results),
            passed=len(passed_list),
            failed=len(results) - len(passed_list),
            results=results,
            avg_relevance=sum(r.answer_relevance for r in results) / n,
            avg_completeness=sum(r.completeness for r in results) / n,
            avg_source_accuracy=sum(r.source_accuracy for r in results) / n,
            avg_confidence_calibration=sum(r.confidence_calibration for r in results) / n,
            avg_latency_ms=sum(r.latency_ms for r in results) / n,
            pass_rate=len(passed_list) / n,
            by_type=by_type,
            by_difficulty=by_difficulty,
        )
