"""Latency profiling — per-component and end-to-end timing with percentile analysis."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

from loguru import logger

from src.models.queries import ResearchQuery
from src.retrieval.pipeline import ResearchPipeline


@dataclass
class LatencyResult:
    """Latency measurements for a single query."""

    query: str
    total_ms: float = 0.0
    embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    llm_inference_ms: float = 0.0
    agent_orchestration_ms: float = 0.0
    time_to_first_token_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "total_ms": round(self.total_ms, 2),
            "embedding_ms": round(self.embedding_ms, 2),
            "retrieval_ms": round(self.retrieval_ms, 2),
            "llm_inference_ms": round(self.llm_inference_ms, 2),
            "agent_orchestration_ms": round(self.agent_orchestration_ms, 2),
            "time_to_first_token_ms": round(self.time_to_first_token_ms, 2),
        }


@dataclass
class LatencyReport:
    """Aggregate latency statistics with percentile analysis."""

    pipeline_type: str  # "rag" or "agent"
    num_queries: int = 0
    results: list[LatencyResult] = field(default_factory=list)

    # Percentile stats (computed)
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0

    # Component breakdown
    avg_embedding_ms: float = 0.0
    avg_retrieval_ms: float = 0.0
    avg_llm_ms: float = 0.0
    avg_orchestration_ms: float = 0.0
    avg_ttft_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "pipeline_type": self.pipeline_type,
            "num_queries": self.num_queries,
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "avg_embedding_ms": round(self.avg_embedding_ms, 2),
            "avg_retrieval_ms": round(self.avg_retrieval_ms, 2),
            "avg_llm_ms": round(self.avg_llm_ms, 2),
            "avg_orchestration_ms": round(self.avg_orchestration_ms, 2),
            "avg_ttft_ms": round(self.avg_ttft_ms, 2),
            "results": [r.to_dict() for r in self.results],
        }


class LatencyProfiler:
    """Profile latency across RAG and agent pipelines with per-component breakdown."""

    def __init__(self, pipeline: ResearchPipeline | None = None):
        self.pipeline = pipeline or ResearchPipeline()

    def profile_rag(self, queries: list[str], warmup: int = 1) -> LatencyReport:
        """Profile RAG pipeline latency across multiple queries.

        Args:
            queries: List of query strings to benchmark.
            warmup: Number of warmup queries to run before timing.
        """
        import contextlib

        # Warmup
        for q in queries[:warmup]:
            with contextlib.suppress(Exception):
                self.pipeline.query(ResearchQuery(question=q))

        results: list[LatencyResult] = []
        for q in queries:
            result = self._time_rag_query(q)
            results.append(result)

        return self._compute_report(results, "rag")

    def profile_agent(self, queries: list[str], warmup: int = 1) -> LatencyReport:
        """Profile agent pipeline latency across multiple queries."""
        import contextlib

        from src.agents.orchestrator import run_agent_query

        # Warmup
        for q in queries[:warmup]:
            with contextlib.suppress(Exception):
                run_agent_query(q)

        results: list[LatencyResult] = []
        for q in queries:
            result = self._time_agent_query(q)
            results.append(result)

        return self._compute_report(results, "agent")

    def _time_rag_query(self, query_text: str) -> LatencyResult:
        """Time individual components of a RAG query."""
        result = LatencyResult(query=query_text)

        try:
            # Time embedding
            t0 = time.perf_counter()
            self.pipeline.vector_store.embed_text(query_text)
            result.embedding_ms = (time.perf_counter() - t0) * 1000

            # Time retrieval
            t0 = time.perf_counter()
            query = ResearchQuery(question=query_text)
            self.pipeline.vector_store.search(query_text, top_k=8)
            result.retrieval_ms = (time.perf_counter() - t0) * 1000

            # Time full query (includes LLM)
            t0 = time.perf_counter()
            self.pipeline.query(query)
            total = (time.perf_counter() - t0) * 1000
            result.total_ms = total
            result.llm_inference_ms = total - result.embedding_ms - result.retrieval_ms
            result.time_to_first_token_ms = result.embedding_ms + result.retrieval_ms
        except Exception as e:
            logger.warning(f"Latency profiling failed for query: {e}")
            # Still record what we got
            result.total_ms = -1

        return result

    def _time_agent_query(self, query_text: str) -> LatencyResult:
        """Time agent pipeline execution."""
        from src.agents.orchestrator import run_agent_query

        result = LatencyResult(query=query_text)

        try:
            t0 = time.perf_counter()
            agent_result = run_agent_query(query_text)
            total = (time.perf_counter() - t0) * 1000
            result.total_ms = total
            result.agent_orchestration_ms = total

            # Extract component timings from trace if available
            trace = agent_result.get("trace", [])
            for entry in trace:
                duration = entry.get("duration_ms", 0)
                node = entry.get("node", "")
                if node == "corpus_search":
                    result.retrieval_ms += duration
                elif node in ("cross_reference", "timeline", "fact_check") or node == "synthesize":
                    result.llm_inference_ms += duration
        except Exception as e:
            logger.warning(f"Agent latency profiling failed: {e}")
            result.total_ms = -1

        return result

    def _compute_report(self, results: list[LatencyResult], pipeline_type: str) -> LatencyReport:
        """Compute percentile statistics from raw results."""
        valid = [r for r in results if r.total_ms >= 0]
        if not valid:
            return LatencyReport(pipeline_type=pipeline_type, num_queries=0)

        totals = sorted(r.total_ms for r in valid)
        n = len(valid)

        report = LatencyReport(
            pipeline_type=pipeline_type,
            num_queries=n,
            results=results,
            mean_ms=statistics.mean(totals),
            min_ms=totals[0],
            max_ms=totals[-1],
            p50_ms=self._percentile(totals, 50),
            p95_ms=self._percentile(totals, 95),
            p99_ms=self._percentile(totals, 99),
            avg_embedding_ms=statistics.mean(r.embedding_ms for r in valid),
            avg_retrieval_ms=statistics.mean(r.retrieval_ms for r in valid),
            avg_llm_ms=statistics.mean(r.llm_inference_ms for r in valid),
            avg_orchestration_ms=statistics.mean(r.agent_orchestration_ms for r in valid),
            avg_ttft_ms=statistics.mean(r.time_to_first_token_ms for r in valid),
        )

        logger.info(
            f"{pipeline_type} latency: p50={report.p50_ms:.0f}ms "
            f"p95={report.p95_ms:.0f}ms p99={report.p99_ms:.0f}ms"
        )
        return report

    @staticmethod
    def _percentile(sorted_values: list[float], pct: int) -> float:
        """Compute percentile from pre-sorted values."""
        if not sorted_values:
            return 0.0
        k = (len(sorted_values) - 1) * pct / 100
        f = int(k)
        c = f + 1
        if c >= len(sorted_values):
            return sorted_values[-1]
        d = k - f
        return sorted_values[f] + d * (sorted_values[c] - sorted_values[f])
