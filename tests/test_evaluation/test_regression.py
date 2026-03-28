"""Tests for regression tracking and A/B comparison."""

from src.evaluation.regression.comparator import ABComparator, ABComparison
from src.evaluation.regression.golden_set import GoldenResult, GoldenSetReport
from src.evaluation.regression.tracker import RegressionReport, RegressionTracker


class TestRegressionTracker:
    def test_compare_no_regression(self):
        tracker = RegressionTracker.__new__(RegressionTracker)
        tracker.warning_threshold = 0.05
        tracker.critical_threshold = 0.10

        current = {"pass_rate": 0.85, "avg_relevance": 0.80}
        previous = {"pass_rate": 0.82, "avg_relevance": 0.78, "timestamp": "prev"}

        report = tracker._compare(current, previous)
        assert report.has_regressions is False

    def test_compare_with_regression(self):
        tracker = RegressionTracker.__new__(RegressionTracker)
        tracker.warning_threshold = 0.05
        tracker.critical_threshold = 0.10

        current = {"pass_rate": 0.60, "avg_relevance": 0.50}
        previous = {"pass_rate": 0.85, "avg_relevance": 0.80, "timestamp": "prev"}

        report = tracker._compare(current, previous)
        assert report.has_regressions is True
        assert len(report.regressions) >= 1

        # Check severity
        severities = {r.metric: r.severity for r in report.regressions}
        assert severities.get("pass_rate") == "critical"

    def test_compare_with_improvement(self):
        tracker = RegressionTracker.__new__(RegressionTracker)
        tracker.warning_threshold = 0.05
        tracker.critical_threshold = 0.10

        current = {"pass_rate": 0.95, "timestamp": "now"}
        previous = {"pass_rate": 0.80, "timestamp": "prev"}

        report = tracker._compare(current, previous)
        assert report.has_regressions is False
        assert len(report.improvements) >= 1

    def test_save_and_load(self, tmp_path):
        tracker = RegressionTracker(history_dir=tmp_path)
        tracker.save_run({"pass_rate": 0.85}, label="test")

        history = tracker.load_history(label="test")
        assert len(history) == 1
        assert history[0]["pass_rate"] == 0.85

    def test_report_to_dict(self):
        report = RegressionReport(
            has_regressions=True,
            current_run="now",
            previous_run="before",
        )
        d = report.to_dict()
        assert d["has_regressions"] is True


class TestABComparator:
    def test_paired_t_test_identical(self):
        a = [0.8, 0.8, 0.8, 0.8]
        b = [0.8, 0.8, 0.8, 0.8]
        t, p = ABComparator._paired_t_test(a, b)
        assert t == 0.0
        assert p == 1.0

    def test_paired_t_test_different(self):
        a = [0.9, 0.85, 0.88, 0.92, 0.87]
        b = [0.6, 0.55, 0.58, 0.62, 0.57]
        t, p = ABComparator._paired_t_test(a, b)
        assert t > 0  # a is significantly higher
        assert p < 0.05

    def test_paired_t_test_short(self):
        t, p = ABComparator._paired_t_test([0.5], [0.5])
        assert p == 1.0

    def test_latency_winner_agent_faster(self):
        """Agent wins latency when it's faster (lower latency = better)."""
        # Agent is faster (lower latency) with significant difference
        a = [500.0, 520.0, 510.0, 530.0, 490.0]  # RAG latencies
        b = [200.0, 210.0, 190.0, 220.0, 195.0]  # Agent latencies (much faster)
        t, p = ABComparator._paired_t_test(a, b)
        assert t > 0  # RAG - Agent > 0 means agent is faster
        assert p < 0.05  # Significant

    def test_comparison_to_dict(self):
        c = ABComparison(
            metric="answer_relevance",
            rag_mean=0.75,
            agent_mean=0.85,
            delta=0.10,
            t_statistic=2.5,
            p_value=0.03,
            significant=True,
            winner="agent",
        )
        d = c.to_dict()
        assert d["winner"] == "agent"
        assert d["significant"] is True


class TestGoldenResult:
    def test_to_dict(self):
        result = GoldenResult(
            test_id="factual-001",
            query="What did Lazar say?",
            query_type="factual",
            difficulty="easy",
            answer_relevance=0.85,
            source_accuracy=0.70,
            completeness=0.90,
            confidence_calibration=0.80,
            latency_ms=150.0,
            passed=True,
        )
        d = result.to_dict()
        assert d["test_id"] == "factual-001"
        assert d["passed"] is True
        assert d["answer_relevance"] == 0.85


class TestGoldenSetReport:
    def test_to_dict(self):
        report = GoldenSetReport(
            pipeline_type="rag",
            total=40,
            passed=35,
            failed=5,
            pass_rate=0.875,
        )
        d = report.to_dict()
        assert d["total"] == 40
        assert d["pass_rate"] == 0.875
