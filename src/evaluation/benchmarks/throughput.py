"""Throughput benchmarking — concurrent query load testing."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from loguru import logger

from src.models.queries import ResearchQuery
from src.retrieval.pipeline import ResearchPipeline


@dataclass
class ThroughputResult:
    """Results from a throughput benchmark run."""

    concurrency: int
    total_queries: int
    successful: int
    failed: int
    total_time_s: float
    queries_per_second: float
    avg_latency_ms: float
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "concurrency": self.concurrency,
            "total_queries": self.total_queries,
            "successful": self.successful,
            "failed": self.failed,
            "total_time_s": round(self.total_time_s, 2),
            "queries_per_second": round(self.queries_per_second, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "errors": self.errors[:10],  # Cap error list
        }


@dataclass
class ThroughputReport:
    """Multi-concurrency throughput report."""

    results: list[ThroughputResult] = field(default_factory=list)
    peak_qps: float = 0.0
    optimal_concurrency: int = 0

    def to_dict(self) -> dict:
        return {
            "peak_qps": round(self.peak_qps, 2),
            "optimal_concurrency": self.optimal_concurrency,
            "results": [r.to_dict() for r in self.results],
        }


class ThroughputBenchmark:
    """Benchmark system throughput under varying concurrency levels."""

    def __init__(self, pipeline: ResearchPipeline | None = None):
        self.pipeline = pipeline or ResearchPipeline()

    def run(
        self,
        queries: list[str],
        concurrency_levels: list[int] | None = None,
    ) -> ThroughputReport:
        """Run throughput tests at multiple concurrency levels.

        Args:
            queries: List of query strings to send.
            concurrency_levels: List of thread counts to test.
                Default: [1, 2, 4, 8].
        """
        if concurrency_levels is None:
            concurrency_levels = [1, 2, 4, 8]

        results: list[ThroughputResult] = []
        for level in concurrency_levels:
            logger.info(f"Throughput test: concurrency={level}")
            result = self._run_at_concurrency(queries, level)
            results.append(result)

        # Find peak
        peak = max(results, key=lambda r: r.queries_per_second) if results else None

        return ThroughputReport(
            results=results,
            peak_qps=peak.queries_per_second if peak else 0.0,
            optimal_concurrency=peak.concurrency if peak else 0,
        )

    def _run_at_concurrency(self, queries: list[str], concurrency: int) -> ThroughputResult:
        """Execute queries at a specific concurrency level."""
        latencies: list[float] = []
        errors: list[str] = []

        start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(self._execute_query, q): q for q in queries}
            for future in as_completed(futures):
                try:
                    latency_ms = future.result()
                    latencies.append(latency_ms)
                except Exception as e:
                    errors.append(str(e))

        total_time = time.perf_counter() - start
        successful = len(latencies)
        failed = len(errors)

        return ThroughputResult(
            concurrency=concurrency,
            total_queries=len(queries),
            successful=successful,
            failed=failed,
            total_time_s=total_time,
            queries_per_second=successful / total_time if total_time > 0 else 0,
            avg_latency_ms=(sum(latencies) / len(latencies) if latencies else 0.0),
            errors=errors,
        )

    def _execute_query(self, query_text: str) -> float:
        """Execute a single query and return latency in ms."""
        t0 = time.perf_counter()
        self.pipeline.query(ResearchQuery(question=query_text))
        return (time.perf_counter() - t0) * 1000
