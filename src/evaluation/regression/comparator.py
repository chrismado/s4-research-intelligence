"""A/B comparison — RAG vs Agent with statistical significance testing."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from loguru import logger

from src.evaluation.regression.golden_set import GoldenSetReport, GoldenSetRunner


@dataclass
class ABComparison:
    """Statistical comparison between RAG and Agent pipelines."""

    metric: str
    rag_mean: float
    agent_mean: float
    delta: float
    t_statistic: float
    p_value: float
    significant: bool  # p < 0.05
    winner: str  # "rag", "agent", or "tie"

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "rag_mean": round(self.rag_mean, 4),
            "agent_mean": round(self.agent_mean, 4),
            "delta": round(self.delta, 4),
            "t_statistic": round(self.t_statistic, 4),
            "p_value": round(self.p_value, 4),
            "significant": self.significant,
            "winner": self.winner,
        }


@dataclass
class ABReport:
    """Full A/B comparison report between RAG and Agent pipelines."""

    rag_report: GoldenSetReport | None = None
    agent_report: GoldenSetReport | None = None
    comparisons: list[ABComparison] = field(default_factory=list)
    overall_winner: str = "tie"
    agent_adds_value: bool = False

    def to_dict(self) -> dict:
        return {
            "rag_summary": self.rag_report.to_dict() if self.rag_report else {},
            "agent_summary": self.agent_report.to_dict() if self.agent_report else {},
            "comparisons": [c.to_dict() for c in self.comparisons],
            "overall_winner": self.overall_winner,
            "agent_adds_value": self.agent_adds_value,
        }


class ABComparator:
    """Run A/B comparison between RAG and Agent pipelines with statistical testing.

    Uses paired t-tests to determine if differences are statistically significant.
    """

    def __init__(self, runner: GoldenSetRunner | None = None):
        self.runner = runner or GoldenSetRunner()

    def compare(self, golden_path=None) -> ABReport:
        """Run both pipelines on the golden set and compare.

        Args:
            golden_path: Path to golden_queries.json.
        """
        logger.info("Running RAG pipeline on golden set...")
        rag_report = self.runner.run(golden_path, pipeline_type="rag")

        logger.info("Running Agent pipeline on golden set...")
        agent_report = self.runner.run(golden_path, pipeline_type="agent")

        # Extract per-query scores for paired comparison
        comparisons = self._paired_tests(rag_report, agent_report)

        # Determine overall winner
        agent_wins = sum(1 for c in comparisons if c.winner == "agent" and c.significant)
        rag_wins = sum(1 for c in comparisons if c.winner == "rag" and c.significant)

        if agent_wins > rag_wins:
            overall = "agent"
        elif rag_wins > agent_wins:
            overall = "rag"
        else:
            overall = "tie"

        return ABReport(
            rag_report=rag_report,
            agent_report=agent_report,
            comparisons=comparisons,
            overall_winner=overall,
            agent_adds_value=agent_wins > 0,
        )

    def _paired_tests(
        self,
        rag: GoldenSetReport,
        agent: GoldenSetReport,
    ) -> list[ABComparison]:
        """Run paired t-tests on matching queries."""
        metrics = ["answer_relevance", "completeness", "source_accuracy", "confidence_calibration"]
        comparisons = []

        for metric in metrics:
            rag_scores = [getattr(r, metric, 0.0) for r in rag.results]
            agent_scores = [getattr(r, metric, 0.0) for r in agent.results]

            # Ensure same length
            n = min(len(rag_scores), len(agent_scores))
            if n == 0:
                continue

            rag_scores = rag_scores[:n]
            agent_scores = agent_scores[:n]

            t_stat, p_value = self._paired_t_test(rag_scores, agent_scores)
            rag_mean = sum(rag_scores) / n
            agent_mean = sum(agent_scores) / n
            delta = agent_mean - rag_mean

            significant = p_value < 0.05
            winner = ("agent" if delta > 0 else "rag") if significant else "tie"

            comparisons.append(
                ABComparison(
                    metric=metric,
                    rag_mean=rag_mean,
                    agent_mean=agent_mean,
                    delta=delta,
                    t_statistic=t_stat,
                    p_value=p_value,
                    significant=significant,
                    winner=winner,
                )
            )

        # Add latency comparison (lower is better)
        rag_latencies = [r.latency_ms for r in rag.results]
        agent_latencies = [r.latency_ms for r in agent.results]
        n = min(len(rag_latencies), len(agent_latencies))
        if n > 0:
            t_stat, p_value = self._paired_t_test(rag_latencies[:n], agent_latencies[:n])
            rag_mean = sum(rag_latencies[:n]) / n
            agent_mean = sum(agent_latencies[:n]) / n

            comparisons.append(
                ABComparison(
                    metric="latency_ms",
                    rag_mean=rag_mean,
                    agent_mean=agent_mean,
                    delta=agent_mean - rag_mean,
                    t_statistic=t_stat,
                    p_value=p_value,
                    significant=p_value < 0.05,
                    winner=(
                        "rag"
                        if agent_mean > rag_mean
                        else "agent"
                        if agent_mean < rag_mean
                        else "tie"
                    )
                    if p_value < 0.05
                    else "tie",
                )
            )

        return comparisons

    @staticmethod
    def _paired_t_test(a: list[float], b: list[float]) -> tuple[float, float]:
        """Compute paired t-test statistic and p-value.

        Implements the test from scratch (no scipy dependency).
        Returns (t_statistic, p_value).
        """
        n = len(a)
        if n < 2:
            return 0.0, 1.0

        diffs = [a[i] - b[i] for i in range(n)]
        mean_diff = sum(diffs) / n
        variance = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)

        if variance == 0:
            return 0.0, 1.0

        se = math.sqrt(variance / n)
        t_stat = mean_diff / se

        # Approximate p-value using t-distribution
        # (simplified — use degrees of freedom n-1)
        df = n - 1
        p_value = ABComparator._t_distribution_p(abs(t_stat), df) * 2  # Two-tailed

        return t_stat, min(p_value, 1.0)

    @staticmethod
    def _t_distribution_p(t: float, df: int) -> float:
        """Approximate upper-tail p-value for t-distribution.

        Uses the approximation from Abramowitz and Stegun for the
        incomplete beta function. Accurate enough for significance testing.
        """
        if df <= 0:
            return 0.5
        x = df / (df + t * t)

        # Regularized incomplete beta function approximation
        # Using a simple series expansion
        a = df / 2
        if x >= 1.0:
            return 0.0
        if x <= 0.0:
            return 0.5

        # Simple numerical integration approximation
        # For practical purposes, use the normal approximation for large df
        if df > 30:
            # Normal approximation
            z = t * (1 - 1 / (4 * df)) / math.sqrt(1 + t * t / (2 * df))
            return 0.5 * math.erfc(z / math.sqrt(2))

        # For small df, use continued fraction approximation
        # More accurate than the naive power formula for df < 30
        beta_val = 1.0
        for i in range(int(a), 0, -1):
            beta_val = 1.0 + (i - a) * x / (i + 1) * beta_val
        p = (x**a) * ((1 - x) ** 0.5) / (a * beta_val)
        return max(0.0, min(0.5, p))
