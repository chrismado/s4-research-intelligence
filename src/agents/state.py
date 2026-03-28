"""
Research state schema for the multi-agent orchestration layer.

Defines the shared state that flows through the LangGraph StateGraph.
Each agent reads from and writes to specific fields in this state.
"""

from __future__ import annotations

from typing import TypedDict


class AgentCall(TypedDict):
    """A planned agent invocation in the research plan."""

    agent: str  # "corpus_search" | "cross_reference" | "timeline" | "fact_check"
    reason: str  # why this agent is being called
    depends_on: list[str]  # agents that must complete first
    priority: int  # execution order (lower = first)


class CorpusResult(TypedDict, total=False):
    """Result from the corpus search agent."""

    content: str
    source_file: str
    source_type: str
    relevance_score: float
    reliability_score: float
    combined_score: float
    excerpt: str


class CrossRefReport(TypedDict, total=False):
    """Cross-reference corroboration report."""

    claim: str
    corroborating: list[dict]  # sources that agree
    contradicting: list[dict]  # sources that disagree
    unresolved: list[dict]  # ambiguous or insufficient evidence
    summary: str


class TimelineResult(TypedDict, total=False):
    """A chronological event extracted by the timeline agent."""

    date: str | None
    description: str
    source: str
    confidence: float
    category: str  # personal | professional | government | scientific | media


class FactCheckResult(TypedDict, total=False):
    """Verdict from the fact-check agent."""

    claim: str
    verdict: str  # VERIFIED | DISPUTED | UNVERIFIABLE | CONTRADICTED
    confidence: float
    supporting_sources: list[dict]
    contradicting_sources: list[dict]
    reasoning: str


class TraceEntry(TypedDict, total=False):
    """A single entry in the decision trace for observability."""

    node: str
    action: str
    timestamp: str
    duration_ms: float
    inputs: dict
    outputs: dict
    error: str | None


class ResearchState(TypedDict, total=False):
    """
    Shared state for the multi-agent research pipeline.

    Flows through the LangGraph StateGraph. Each node reads relevant
    fields and writes its results back into the state.
    """

    # Input
    query: str
    query_type: str  # "factual" | "timeline" | "verification" | "exploration"

    # Planning
    research_plan: list[AgentCall]

    # Agent results
    corpus_results: list[CorpusResult]
    cross_ref_results: CrossRefReport
    timeline_results: list[TimelineResult]
    fact_check_results: list[FactCheckResult]

    # Synthesis
    synthesis: str
    confidence: float
    sources_cited: list[dict]

    # Control flow
    retries: int
    max_retries: int

    # Observability
    trace: list[TraceEntry]
    trace_id: str
