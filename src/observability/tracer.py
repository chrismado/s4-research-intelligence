"""
Decision tracing for the multi-agent research pipeline.

Records every agent decision, tool call, and timing for full
observability. Supports both Langfuse integration (when available)
and structured JSON logging as fallback.
"""

from __future__ import annotations

import contextlib
import time
import uuid
from datetime import datetime, timezone

from loguru import logger

from src.agents.state import TraceEntry

# Try Langfuse, fall back to structured logging
try:
    from langfuse import Langfuse

    _langfuse: Langfuse | None = Langfuse()
    _LANGFUSE_AVAILABLE = True
    logger.info("Langfuse tracing enabled")
except Exception:
    _langfuse = None
    _LANGFUSE_AVAILABLE = False
    logger.info("Langfuse not configured — using structured JSON logging for traces")


class ResearchTracer:
    """
    Traces agent decisions and tool calls through the research pipeline.

    Each research query gets a unique trace_id. All agent operations
    within that query are recorded as spans on that trace.
    """

    def __init__(self, trace_id: str | None = None):
        self.trace_id = trace_id or f"trace_{uuid.uuid4().hex[:12]}"
        self.entries: list[TraceEntry] = []
        self._span_stack: list[dict] = []
        self._langfuse_trace = None

        if _LANGFUSE_AVAILABLE and _langfuse:
            try:
                self._langfuse_trace = _langfuse.trace(
                    id=self.trace_id,
                    name="research_query",
                )
            except Exception as e:
                logger.debug(f"Langfuse trace creation failed: {e}")

    def start_span(self, node: str, action: str, inputs: dict | None = None) -> dict:
        """Start a timed span for an agent operation."""
        span = {
            "node": node,
            "action": action,
            "start_time": time.perf_counter(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inputs": inputs or {},
        }
        self._span_stack.append(span)

        if self._langfuse_trace:
            with contextlib.suppress(Exception):
                span["_langfuse_span"] = self._langfuse_trace.span(
                    name=f"{node}.{action}",
                    input=inputs,
                )

        logger.debug(f"[{self.trace_id}] START {node}.{action}")
        return span

    def end_span(
        self,
        span: dict,
        outputs: dict | None = None,
        error: str | None = None,
    ) -> TraceEntry:
        """End a timed span and record the trace entry."""
        duration_ms = (time.perf_counter() - span["start_time"]) * 1000

        entry: TraceEntry = {
            "node": span["node"],
            "action": span["action"],
            "timestamp": span["timestamp"],
            "duration_ms": round(duration_ms, 2),
            "inputs": span["inputs"],
            "outputs": outputs or {},
            "error": error,
        }
        self.entries.append(entry)

        # Remove from stack
        if span in self._span_stack:
            self._span_stack.remove(span)

        # Langfuse span end
        langfuse_span = span.get("_langfuse_span")
        if langfuse_span:
            with contextlib.suppress(Exception):
                langfuse_span.end(output=outputs, level="ERROR" if error else "DEFAULT")

        level = "WARNING" if error else "DEBUG"
        logger.log(
            level,
            f"[{self.trace_id}] END {span['node']}.{span['action']} "
            f"({duration_ms:.0f}ms){f' ERROR: {error}' if error else ''}",
        )

        return entry

    def log_decision(self, node: str, decision: str, reasoning: str = "") -> None:
        """Log a decision point in the trace without timing."""
        entry: TraceEntry = {
            "node": node,
            "action": "decision",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": 0.0,
            "inputs": {"reasoning": reasoning},
            "outputs": {"decision": decision},
            "error": None,
        }
        self.entries.append(entry)
        logger.info(f"[{self.trace_id}] DECISION {node}: {decision}")

    def get_trace(self) -> list[TraceEntry]:
        """Get the full trace as a list of entries."""
        return list(self.entries)

    def get_summary(self) -> dict:
        """Get a summary of the trace for the API response."""
        total_ms = sum(e.get("duration_ms", 0) for e in self.entries)
        agents_called = list({e["node"] for e in self.entries if e["action"] != "decision"})
        decisions = [e for e in self.entries if e["action"] == "decision"]
        errors = [e for e in self.entries if e.get("error")]

        return {
            "trace_id": self.trace_id,
            "total_duration_ms": round(total_ms, 2),
            "agents_called": agents_called,
            "num_decisions": len(decisions),
            "num_errors": len(errors),
            "entries": self.entries,
        }

    def flush(self) -> None:
        """Flush any pending Langfuse data."""
        if _langfuse:
            with contextlib.suppress(Exception):
                _langfuse.flush()
