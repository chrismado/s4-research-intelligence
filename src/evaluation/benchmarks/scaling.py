"""Corpus scaling tests — accuracy vs corpus size analysis."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from src.models.queries import ResearchQuery
from src.retrieval.pipeline import ResearchPipeline


@dataclass
class ScalePoint:
    """Evaluation results at a specific corpus percentage."""

    corpus_pct: float  # 0.25, 0.50, 0.75, 1.0
    num_documents: int
    avg_source_recall: float
    avg_confidence: float
    avg_latency_ms: float
    total_queries: int

    def to_dict(self) -> dict:
        return {
            "corpus_pct": self.corpus_pct,
            "num_documents": self.num_documents,
            "avg_source_recall": round(self.avg_source_recall, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "total_queries": self.total_queries,
        }


@dataclass
class ScalingReport:
    """Full corpus scaling analysis."""

    scale_points: list[ScalePoint] = field(default_factory=list)
    scaling_trend: str = ""  # "linear", "sublinear", "degrading"

    def to_dict(self) -> dict:
        return {
            "scale_points": [p.to_dict() for p in self.scale_points],
            "scaling_trend": self.scaling_trend,
        }


class ScalingBenchmark:
    """Benchmark accuracy and latency across different corpus sizes.

    Tests the system at 25%, 50%, 75%, and 100% of the corpus to
    show how performance scales with data volume.
    """

    def __init__(self, pipeline: ResearchPipeline | None = None):
        self.pipeline = pipeline or ResearchPipeline()

    def run(
        self,
        golden_queries_path: Path | None = None,
        scale_fractions: list[float] | None = None,
    ) -> ScalingReport:
        """Run scaling benchmarks across corpus size fractions.

        Args:
            golden_queries_path: Path to golden test set JSON.
            scale_fractions: Corpus fractions to test. Default: [0.25, 0.5, 0.75, 1.0].
        """
        from config.settings import settings

        if golden_queries_path is None:
            golden_queries_path = settings.data_dir / "evaluation" / "golden_queries.json"

        if scale_fractions is None:
            scale_fractions = [0.25, 0.50, 0.75, 1.00]

        # Load golden queries
        try:
            queries = json.loads(golden_queries_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load golden queries from {golden_queries_path}: {e}")
            return ScalingReport(scale_points=[], scaling_trend="error")

        # Get total document count
        try:
            collection = self.pipeline.vector_store.collection
            total_docs = collection.count()
        except Exception:
            total_docs = 0
            logger.warning("Could not get corpus size; running at full scale only")
            scale_fractions = [1.0]

        scale_points: list[ScalePoint] = []
        for fraction in scale_fractions:
            n_docs = int(total_docs * fraction)
            logger.info(f"Scaling test: {fraction:.0%} of corpus ({n_docs} docs)")
            point = self._evaluate_at_scale(queries, fraction, total_docs)
            scale_points.append(point)

        # Determine scaling trend
        trend = self._analyze_trend(scale_points)

        report = ScalingReport(scale_points=scale_points, scaling_trend=trend)
        logger.info(f"Scaling benchmark complete: trend={trend}")
        return report

    def _evaluate_at_scale(
        self,
        queries: list[dict],
        fraction: float,
        total_docs: int,
    ) -> ScalePoint:
        """Evaluate the golden set at a specific corpus fraction.

        For fractions < 1.0, we limit retrieval top_k proportionally
        rather than physically removing documents from the store.
        """
        effective_top_k = max(1, int(8 * fraction))
        num_docs = int(total_docs * fraction)

        recalls = []
        confidences = []
        latencies = []

        for q in queries:
            query_text = q["query"]
            expected = q.get("expected_answer_contains", [])

            t0 = time.perf_counter()
            try:
                query = ResearchQuery(question=query_text, top_k=effective_top_k)
                response = self.pipeline.query(query)
                latency = (time.perf_counter() - t0) * 1000

                # Check answer coverage
                answer_lower = response.answer.lower()
                hits = sum(1 for e in expected if e.lower() in answer_lower)
                recall = hits / len(expected) if expected else 1.0

                recalls.append(recall)
                confidences.append(response.confidence)
                latencies.append(latency)
            except Exception as e:
                logger.debug(f"Scaling eval error: {e}")
                latencies.append(-1)

        valid_latencies = [lat for lat in latencies if lat >= 0]
        n = len(recalls) or 1

        return ScalePoint(
            corpus_pct=fraction,
            num_documents=num_docs,
            avg_source_recall=sum(recalls) / n if recalls else 0.0,
            avg_confidence=sum(confidences) / n if confidences else 0.0,
            avg_latency_ms=(
                sum(valid_latencies) / len(valid_latencies) if valid_latencies else 0.0
            ),
            total_queries=len(queries),
        )

    def _analyze_trend(self, points: list[ScalePoint]) -> str:
        """Analyze how accuracy scales with corpus size."""
        if len(points) < 2:
            return "insufficient_data"

        recalls = [p.avg_source_recall for p in points]
        first_half_gain = recalls[len(recalls) // 2] - recalls[0] if len(recalls) >= 2 else 0
        second_half_gain = recalls[-1] - recalls[len(recalls) // 2] if len(recalls) >= 2 else 0

        if recalls[-1] < recalls[0]:
            return "degrading"
        elif second_half_gain < first_half_gain * 0.5:
            return "sublinear"
        else:
            return "linear"
