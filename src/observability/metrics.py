"""
Agent-level metrics for monitoring the multi-agent research pipeline.

Tracks latency, retry counts, tool call counts, and agent-level
performance for operational monitoring and evaluation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class AgentMetrics:
    """Metrics for a single agent invocation."""

    agent_name: str
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float | None = None
    tool_calls: int = 0
    llm_calls: int = 0
    results_count: int = 0
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return (time.perf_counter() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    def complete(self, results_count: int = 0) -> None:
        self.end_time = time.perf_counter()
        self.results_count = results_count

    def fail(self, error: str) -> None:
        self.end_time = time.perf_counter()
        self.error = error

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "duration_ms": round(self.duration_ms, 2),
            "tool_calls": self.tool_calls,
            "llm_calls": self.llm_calls,
            "results_count": self.results_count,
            "error": self.error,
        }


@dataclass
class PipelineMetrics:
    """Aggregate metrics for a full research pipeline run."""

    trace_id: str
    query: str
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float | None = None
    agent_metrics: list[AgentMetrics] = field(default_factory=list)
    retries: int = 0
    final_confidence: float = 0.0

    @property
    def total_duration_ms(self) -> float:
        if self.end_time is None:
            return (time.perf_counter() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    def start_agent(self, agent_name: str) -> AgentMetrics:
        metrics = AgentMetrics(agent_name=agent_name)
        self.agent_metrics.append(metrics)
        return metrics

    def complete(self, confidence: float = 0.0) -> None:
        self.end_time = time.perf_counter()
        self.final_confidence = confidence

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "query": self.query[:100],
            "total_duration_ms": round(self.total_duration_ms, 2),
            "retries": self.retries,
            "final_confidence": self.final_confidence,
            "agents": [m.to_dict() for m in self.agent_metrics],
            "total_tool_calls": sum(m.tool_calls for m in self.agent_metrics),
            "total_llm_calls": sum(m.llm_calls for m in self.agent_metrics),
        }

    def log_summary(self) -> None:
        """Log a structured summary of the pipeline run."""
        logger.info(
            f"Pipeline complete | trace={self.trace_id} | "
            f"duration={self.total_duration_ms:.0f}ms | "
            f"agents={len(self.agent_metrics)} | "
            f"retries={self.retries} | "
            f"confidence={self.final_confidence:.2f}"
        )
