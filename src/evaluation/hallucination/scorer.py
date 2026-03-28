"""Aggregate hallucination scoring across multiple queries with trend tracking."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from src.evaluation.hallucination.detector import HallucinationReport


@dataclass
class AggregateHallucinationScore:
    """Aggregate hallucination metrics across a batch of queries."""

    total_queries: int = 0
    total_claims: int = 0
    total_supported: int = 0
    total_unsupported: int = 0
    total_contradicted: int = 0
    avg_hallucination_rate: float = 0.0
    avg_fabrication_rate: float = 0.0
    avg_grounding_score: float = 0.0
    worst_queries: list[dict] = field(default_factory=list)
    per_query: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "total_claims": self.total_claims,
            "total_supported": self.total_supported,
            "total_unsupported": self.total_unsupported,
            "total_contradicted": self.total_contradicted,
            "avg_hallucination_rate": round(self.avg_hallucination_rate, 4),
            "avg_fabrication_rate": round(self.avg_fabrication_rate, 4),
            "avg_grounding_score": round(self.avg_grounding_score, 4),
            "worst_queries": self.worst_queries,
            "per_query": self.per_query,
        }


class HallucinationScorer:
    """Compute aggregate hallucination metrics and track trends over time."""

    def __init__(self, history_dir: Path | None = None):
        from config.settings import settings

        self.history_dir = history_dir or (settings.data_dir / "evaluation" / "benchmark_history")
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def score_batch(
        self, reports: list[HallucinationReport], top_worst: int = 5
    ) -> AggregateHallucinationScore:
        """Compute aggregate scores from a list of hallucination reports."""
        if not reports:
            return AggregateHallucinationScore()

        n = len(reports)
        total_claims = sum(len(r.claims) for r in reports)
        total_supported = sum(len(r.supported_claims) for r in reports)
        total_unsupported = sum(len(r.unsupported_claims) for r in reports)
        total_contradicted = sum(len(r.contradicted_claims) for r in reports)

        # Sort by worst hallucination rate to find problem queries
        sorted_reports = sorted(reports, key=lambda r: r.hallucination_rate, reverse=True)
        worst = [
            {
                "query": r.query,
                "hallucination_rate": round(r.hallucination_rate, 4),
                "unsupported_claims": [c.text for c in r.unsupported_claims],
            }
            for r in sorted_reports[:top_worst]
            if r.hallucination_rate > 0
        ]

        return AggregateHallucinationScore(
            total_queries=n,
            total_claims=total_claims,
            total_supported=total_supported,
            total_unsupported=total_unsupported,
            total_contradicted=total_contradicted,
            avg_hallucination_rate=sum(r.hallucination_rate for r in reports) / n,
            avg_fabrication_rate=sum(r.fabrication_rate for r in reports) / n,
            avg_grounding_score=sum(r.grounding_score for r in reports) / n,
            worst_queries=worst,
            per_query=[r.to_dict() for r in reports],
        )

    def save_score(self, score: AggregateHallucinationScore, label: str = "hallucination") -> Path:
        """Save score to history for trend tracking."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.history_dir / f"{label}_{timestamp}.json"
        path.write_text(json.dumps(score.to_dict(), indent=2))
        logger.info(f"Hallucination score saved: {path}")
        return path

    def load_history(self, label: str = "hallucination") -> list[dict]:
        """Load historical scores for trend analysis."""
        files = sorted(self.history_dir.glob(f"{label}_*.json"))
        history = []
        for f in files:
            try:
                data = json.loads(f.read_text())
                data["_file"] = f.name
                data["_timestamp"] = f.stem.split("_", 1)[1] if "_" in f.stem else ""
                history.append(data)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load history file {f}: {e}")
        return history

    def compute_trend(self, label: str = "hallucination") -> dict:
        """Analyze trend in hallucination scores over time."""
        history = self.load_history(label)
        if len(history) < 2:
            return {"trend": "insufficient_data", "runs": len(history)}

        recent = history[-1]
        previous = history[-2]
        delta_grounding = recent["avg_grounding_score"] - previous["avg_grounding_score"]
        delta_hallucination = recent["avg_hallucination_rate"] - previous["avg_hallucination_rate"]

        if delta_grounding > 0.02:
            trend = "improving"
        elif delta_grounding < -0.02:
            trend = "degrading"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "runs": len(history),
            "latest_grounding": recent["avg_grounding_score"],
            "previous_grounding": previous["avg_grounding_score"],
            "delta_grounding": round(delta_grounding, 4),
            "delta_hallucination": round(delta_hallucination, 4),
        }
