"""Main evaluation suite orchestrator — runs all eval modules and produces reports."""

from __future__ import annotations

from loguru import logger

from src.evaluation.adversarial.generator import AdversarialGenerator, AdversarialReport
from src.evaluation.benchmarks.latency import LatencyProfiler
from src.evaluation.benchmarks.memory import MemoryProfiler
from src.evaluation.benchmarks.quantization import QuantizationBenchmark, QuantizationReport
from src.evaluation.benchmarks.scaling import ScalingBenchmark, ScalingReport
from src.evaluation.benchmarks.throughput import ThroughputBenchmark
from src.evaluation.datasets.loader import DatasetLoader
from src.evaluation.hallucination.detector import HallucinationDetector
from src.evaluation.hallucination.scorer import HallucinationScorer
from src.evaluation.regression.comparator import ABComparator, ABReport
from src.evaluation.regression.golden_set import GoldenSetReport, GoldenSetRunner
from src.evaluation.regression.tracker import RegressionReport, RegressionTracker
from src.evaluation.report.json_export import JSONExporter
from src.evaluation.report.markdown import MarkdownReporter
from src.evaluation.report.terminal import TerminalDashboard
from src.retrieval.pipeline import ResearchPipeline


class EvalSuite:
    """Orchestrate the complete evaluation pipeline.

    Coordinates hallucination detection, adversarial testing, benchmarking,
    regression tracking, and report generation.
    """

    def __init__(self, pipeline: ResearchPipeline | None = None):
        self.pipeline = pipeline or ResearchPipeline()
        self.loader = DatasetLoader()
        self.dashboard = TerminalDashboard()
        self._results: dict = {}

    def run_all(self, show_dashboard: bool = True) -> dict:
        """Run the complete evaluation suite.

        Runs: golden set, hallucination, adversarial, benchmarks,
        quantization, regression, and A/B comparison.
        """
        logger.info("Starting full evaluation suite...")

        progress = self.dashboard.create_progress()
        with progress:
            task = progress.add_task("Running evaluation suite...", total=8)

            # 1. Golden set
            progress.update(task, description="Running golden set evaluation...")
            self._results["golden_set"] = self.run_golden_set().to_dict()
            progress.advance(task)

            # 2. Hallucination
            progress.update(task, description="Running hallucination detection...")
            self._results["hallucination"] = self.run_hallucination().to_dict()
            progress.advance(task)

            # 3. Adversarial
            progress.update(task, description="Running adversarial tests...")
            self._results["adversarial"] = self.run_adversarial().to_dict()
            progress.advance(task)

            # 4. Benchmarks
            progress.update(task, description="Running benchmarks...")
            self._results["benchmarks"] = self.run_benchmarks()
            progress.advance(task)

            # 5. Quantization
            progress.update(task, description="Running quantization benchmarks...")
            self._results["quantization"] = self.run_quantization().to_dict()
            progress.advance(task)

            # 6. A/B comparison
            progress.update(task, description="Running A/B comparison...")
            self._results["ab_comparison"] = self.run_compare().to_dict()
            progress.advance(task)

            # 7. Regression check
            progress.update(task, description="Checking for regressions...")
            self._results["regression"] = self.run_regression().to_dict()
            progress.advance(task)

            # 8. Save and report
            progress.update(task, description="Generating reports...")
            self._save_history()
            progress.advance(task)

        if show_dashboard:
            self.dashboard.display(self._results)

        return self._results

    def run_golden_set(self, pipeline_type: str = "rag") -> GoldenSetReport:
        """Run golden test set evaluation."""
        from config.settings import settings

        logger.info(f"Running golden set ({pipeline_type})...")
        runner = GoldenSetRunner(
            pipeline=self.pipeline,
            pass_threshold=settings.eval_pass_threshold,
        )
        return runner.run(pipeline_type=pipeline_type)

    def run_hallucination(self):
        """Run hallucination detection on golden queries."""
        from config.settings import settings

        logger.info("Running hallucination detection...")

        detector = HallucinationDetector(
            support_threshold=settings.eval_support_threshold,
            contradiction_threshold=settings.eval_contradiction_threshold,
        )
        scorer = HallucinationScorer()

        golden = self.loader.load_golden_queries()
        reports = []

        from src.models.queries import ResearchQuery

        for case in golden:
            try:
                response = self.pipeline.query(ResearchQuery(question=case["query"]))
                report = detector.analyze(case["query"], response.answer)
                reports.append(report)
            except Exception as e:
                logger.warning(f"Hallucination check failed for {case.get('id')}: {e}")

        aggregate = scorer.score_batch(reports)
        scorer.save_score(aggregate)
        return aggregate

    def run_adversarial(self) -> AdversarialReport:
        """Run all adversarial tests."""
        logger.info("Running adversarial tests...")
        generator = AdversarialGenerator(pipeline=self.pipeline)
        return generator.run_all()

    def run_benchmarks(self) -> dict:
        """Run performance benchmarks (latency, throughput, memory)."""
        logger.info("Running performance benchmarks...")
        golden = self.loader.load_golden_queries()
        queries = [q["query"] for q in golden[:10]]  # Use first 10 for speed

        result: dict = {}

        # Latency
        profiler = LatencyProfiler(pipeline=self.pipeline)
        latency = profiler.profile_rag(queries)
        result["latency"] = latency.to_dict()

        # Throughput
        throughput = ThroughputBenchmark(pipeline=self.pipeline)
        tp_report = throughput.run(queries[:5], concurrency_levels=[1, 2, 4])
        result["throughput"] = tp_report.to_dict()

        # Memory
        mem = MemoryProfiler()
        if mem.is_available():
            from src.models.queries import ResearchQuery

            profile = mem.profile_query(
                lambda: self.pipeline.query(ResearchQuery(question=queries[0]))
            )
            result["memory"] = profile.to_dict()
        else:
            result["memory"] = {}

        return result

    def run_quantization(self, **kwargs) -> QuantizationReport:
        """Run KV cache quantization benchmarks."""
        logger.info("Running KV cache quantization benchmarks...")
        bench = QuantizationBenchmark(**kwargs)
        return bench.run()

    def run_scaling(self) -> ScalingReport:
        """Run corpus scaling benchmarks."""
        logger.info("Running corpus scaling benchmarks...")
        bench = ScalingBenchmark(pipeline=self.pipeline)
        return bench.run()

    def run_regression(self) -> RegressionReport:
        """Check for regressions against previous run."""
        from config.settings import settings

        logger.info("Checking for regressions...")
        tracker = RegressionTracker(
            warning_threshold=settings.eval_regression_warning,
            critical_threshold=settings.eval_regression_critical,
        )
        # Flatten key metrics for comparison
        flat = {}
        golden = self._results.get("golden_set", {})
        flat.update(
            {
                "pass_rate": golden.get("pass_rate", 0),
                "avg_relevance": golden.get("avg_relevance", 0),
                "avg_completeness": golden.get("avg_completeness", 0),
                "avg_source_accuracy": golden.get("avg_source_accuracy", 0),
                "avg_confidence_calibration": golden.get("avg_confidence_calibration", 0),
            }
        )
        halluc = self._results.get("hallucination", {})
        flat["avg_grounding_score"] = halluc.get("avg_grounding_score", 0)

        adv = self._results.get("adversarial", {})
        flat.update(
            {
                "contradiction_detection_rate": adv.get("contradiction_detection_rate", 0),
                "abstention_rate": adv.get("abstention_rate", 0),
                "injection_resistance_rate": adv.get("injection_resistance_rate", 0),
            }
        )

        return tracker.check_regression(flat)

    def run_compare(self) -> ABReport:
        """Run A/B comparison between RAG and Agent pipelines."""
        logger.info("Running A/B comparison...")
        runner = GoldenSetRunner(pipeline=self.pipeline)
        comparator = ABComparator(runner=runner)
        return comparator.compare()

    def generate_report(self, format: str = "markdown") -> str | dict:
        """Generate evaluation report.

        Args:
            format: "markdown" or "json".
        """
        if format == "json":
            exporter = JSONExporter()
            return exporter.export(self._results)
        else:
            reporter = MarkdownReporter()
            return reporter.generate(self._results)

    def _save_history(self) -> None:
        """Save current run to history for regression tracking."""
        tracker = RegressionTracker()
        flat = {}
        golden = self._results.get("golden_set", {})
        flat.update(
            {
                "pass_rate": golden.get("pass_rate", 0),
                "avg_relevance": golden.get("avg_relevance", 0),
                "avg_completeness": golden.get("avg_completeness", 0),
            }
        )
        halluc = self._results.get("hallucination", {})
        flat["avg_grounding_score"] = halluc.get("avg_grounding_score", 0)
        tracker.save_run(flat)
