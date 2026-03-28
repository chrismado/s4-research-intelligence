"""Score history tracking and regression detection across eval runs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger


@dataclass
class Regression:
    """A detected regression between two eval runs."""

    metric: str
    previous_value: float
    current_value: float
    delta: float
    severity: str  # "warning" or "critical"
    context: str  # e.g., "verification queries" or "overall"

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "previous_value": round(self.previous_value, 4),
            "current_value": round(self.current_value, 4),
            "delta": round(self.delta, 4),
            "severity": self.severity,
            "context": self.context,
        }


@dataclass
class RegressionReport:
    """Report of regressions between current and previous run."""

    has_regressions: bool = False
    regressions: list[Regression] = field(default_factory=list)
    improvements: list[dict] = field(default_factory=list)
    current_run: str = ""
    previous_run: str = ""

    def to_dict(self) -> dict:
        return {
            "has_regressions": self.has_regressions,
            "num_regressions": len(self.regressions),
            "regressions": [r.to_dict() for r in self.regressions],
            "improvements": self.improvements,
            "current_run": self.current_run,
            "previous_run": self.previous_run,
        }


class RegressionTracker:
    """Track evaluation scores over time and detect regressions.

    Saves results to data/evaluation/benchmark_history/ and compares
    against previous runs to flag degradations.
    """

    def __init__(
        self,
        history_dir: Path | None = None,
        warning_threshold: float = 0.05,
        critical_threshold: float = 0.10,
    ):
        from config.settings import settings

        self.history_dir = history_dir or (settings.data_dir / "evaluation" / "benchmark_history")
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def save_run(self, results: dict, label: str = "eval") -> Path:
        """Save an evaluation run to history.

        Args:
            results: Dict of evaluation results.
            label: Prefix for the filename.

        Returns:
            Path to the saved file.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.history_dir / f"run_{label}_{timestamp}.json"
        data = {
            "timestamp": timestamp,
            "label": label,
            **results,
        }
        path.write_text(json.dumps(data, indent=2, default=str))
        logger.info(f"Eval run saved: {path}")
        return path

    def load_history(self, label: str = "eval") -> list[dict]:
        """Load all historical runs, sorted by timestamp."""
        files = sorted(self.history_dir.glob(f"run_{label}_*.json"))
        history = []
        for f in files:
            try:
                data = json.loads(f.read_text())
                data["_file"] = f.name
                history.append(data)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load {f}: {e}")
        return history

    def check_regression(
        self,
        current_results: dict,
        label: str = "eval",
    ) -> RegressionReport:
        """Compare current results against the most recent previous run.

        Args:
            current_results: Current evaluation results dict.
            label: History label to compare against.

        Returns:
            RegressionReport with detected regressions and improvements.
        """
        history = self.load_history(label)
        if not history:
            return RegressionReport(
                current_run="first_run",
                previous_run="none",
            )

        previous = history[-1]
        return self._compare(current_results, previous)

    def _compare(self, current: dict, previous: dict) -> RegressionReport:
        """Compare two result sets and detect regressions."""
        regressions: list[Regression] = []
        improvements: list[dict] = []

        # Metrics to track (higher is better for all)
        tracked_metrics = [
            ("pass_rate", "overall"),
            ("avg_relevance", "overall"),
            ("avg_completeness", "overall"),
            ("avg_source_accuracy", "overall"),
            ("avg_confidence_calibration", "overall"),
            ("avg_grounding_score", "hallucination"),
            ("contradiction_detection_rate", "adversarial"),
            ("abstention_rate", "adversarial"),
            ("injection_resistance_rate", "adversarial"),
        ]

        for metric, context in tracked_metrics:
            curr_val = current.get(metric)
            prev_val = previous.get(metric)

            if curr_val is None or prev_val is None:
                continue

            delta = curr_val - prev_val

            if delta < -self.critical_threshold:
                regressions.append(
                    Regression(
                        metric=metric,
                        previous_value=prev_val,
                        current_value=curr_val,
                        delta=delta,
                        severity="critical",
                        context=context,
                    )
                )
            elif delta < -self.warning_threshold:
                regressions.append(
                    Regression(
                        metric=metric,
                        previous_value=prev_val,
                        current_value=curr_val,
                        delta=delta,
                        severity="warning",
                        context=context,
                    )
                )
            elif delta > self.warning_threshold:
                improvements.append(
                    {
                        "metric": metric,
                        "previous": round(prev_val, 4),
                        "current": round(curr_val, 4),
                        "delta": round(delta, 4),
                        "context": context,
                    }
                )

        return RegressionReport(
            has_regressions=len(regressions) > 0,
            regressions=regressions,
            improvements=improvements,
            current_run=current.get("timestamp", "current"),
            previous_run=previous.get("timestamp", "previous"),
        )
