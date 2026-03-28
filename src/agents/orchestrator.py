"""
Research Orchestrator — LangGraph StateGraph for multi-agent coordination.

Implements the Plan → Dispatch → Execute → Synthesize → Evaluate loop.
This is the main entry point for agentic research queries.

Architecture:
    analyze_query -> dispatch -> [agents] -> synthesize -> evaluate
    (retry if low quality, max 2 retries)
"""

from __future__ import annotations

import json
import uuid

from langgraph.graph import END, StateGraph
from loguru import logger

from src.agents.corpus_search import corpus_search_node
from src.agents.cross_reference import cross_reference_node
from src.agents.fact_checker import fact_check_node
from src.agents.state import ResearchState
from src.agents.timeline_agent import timeline_node
from src.agents.tools import llm_call, make_trace_entry
from src.observability.tracer import ResearchTracer
from src.prompts.agent_prompts import (
    ORCHESTRATOR_PLAN_PROMPT,
    ORCHESTRATOR_SYSTEM_PROMPT,
    SELF_EVAL_PROMPT,
    SELF_EVAL_SYSTEM_PROMPT,
    SYNTHESIS_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
)

# --- Trace store for retrieving traces by ID ---
_trace_store: dict[str, dict] = {}


def get_trace(trace_id: str) -> dict | None:
    """Retrieve a stored trace by ID."""
    return _trace_store.get(trace_id)


# --- Orchestrator nodes ---


def analyze_query_node(state: ResearchState) -> dict:
    """
    Classify the query and create a research plan.

    Determines which agents to invoke based on query type.
    """
    query = state["query"]
    trace_id = state.get("trace_id", f"trace_{uuid.uuid4().hex[:12]}")
    logger.info(f"[{trace_id}] Analyzing query: {query[:80]}...")

    plan_prompt = ORCHESTRATOR_PLAN_PROMPT.format(query=query)
    llm_output = llm_call(ORCHESTRATOR_SYSTEM_PROMPT, plan_prompt)

    query_type = llm_output.get("query_type", "exploration")
    agents = llm_output.get("agents", [])
    reasoning = llm_output.get("reasoning", "")

    # Ensure corpus_search is always included (it's the foundation)
    agent_names = {a.get("agent") for a in agents}
    if "corpus_search" not in agent_names:
        agents.insert(
            0,
            {
                "agent": "corpus_search",
                "reason": "Foundation retrieval — always needed",
                "depends_on": [],
                "priority": 0,
            },
        )

    # Validate agent names
    valid_agents = {"corpus_search", "cross_reference", "timeline", "fact_check"}
    agents = [a for a in agents if a.get("agent") in valid_agents]

    logger.info(f"[{trace_id}] Plan: type={query_type}, " f"agents={[a['agent'] for a in agents]}")

    return {
        "query_type": query_type,
        "research_plan": agents,
        "retries": state.get("retries", 0),
        "max_retries": state.get("max_retries", 2),
        "trace_id": trace_id,
        "trace": state.get("trace", [])
        + [
            make_trace_entry(
                node="orchestrator",
                action="plan",
                inputs={"query": query},
                outputs={
                    "query_type": query_type,
                    "agents_planned": [a["agent"] for a in agents],
                    "reasoning": reasoning,
                },
            )
        ],
    }


def dispatch_node(state: ResearchState) -> dict:
    """
    Route to appropriate sub-agents based on the research plan.

    This node just marks which agents should be called.
    The actual routing is done via conditional edges.
    """
    plan = state.get("research_plan", [])
    agents_to_call = [a["agent"] for a in plan]

    logger.info(f"[{state.get('trace_id', '?')}] Dispatching to: {agents_to_call}")

    return {
        "trace": state.get("trace", [])
        + [
            make_trace_entry(
                node="orchestrator",
                action="dispatch",
                inputs={"plan": plan},
                outputs={"dispatched": agents_to_call},
            )
        ],
    }


def synthesize_node(state: ResearchState) -> dict:
    """
    Merge results from all agents into a final structured response.
    """
    query = state["query"]
    trace_id = state.get("trace_id", "?")
    logger.info(f"[{trace_id}] Synthesizing results...")

    # Format agent results for the synthesis prompt
    corpus_findings = "No corpus search was performed."
    if state.get("corpus_results"):
        findings = []
        for r in state["corpus_results"]:
            findings.append(
                f"- [{r.get('source_file', 'unknown')}] "
                f"(reliability: {r.get('reliability_score', 0.5):.2f}): "
                f"{r.get('excerpt', '')[:200]}"
            )
        corpus_findings = "\n".join(findings)

    cross_ref_report = "No cross-referencing was performed."
    if state.get("cross_ref_results"):
        cr = state["cross_ref_results"]
        cross_ref_report = json.dumps(cr, indent=2, default=str)

    timeline_analysis = "No timeline analysis was performed."
    if state.get("timeline_results"):
        events = []
        for e in state["timeline_results"]:
            date = e.get("date", "?")
            desc = e.get("description", "")
            src = e.get("source", "")
            events.append(f"- [{date}] {desc} ({src})")
        timeline_analysis = "\n".join(events)

    fact_check_results = "No fact-checking was performed."
    if state.get("fact_check_results"):
        checks = []
        for fc in state["fact_check_results"]:
            checks.append(
                f"- Claim: {fc.get('claim', '?')}\n"
                f"  Verdict: {fc.get('verdict', '?')} "
                f"(confidence: {fc.get('confidence', 0):.2f})\n"
                f"  Reasoning: {fc.get('reasoning', '')}"
            )
        fact_check_results = "\n".join(checks)

    synthesis_prompt = SYNTHESIS_PROMPT.format(
        query=query,
        corpus_findings=corpus_findings,
        cross_ref_report=cross_ref_report,
        timeline_analysis=timeline_analysis,
        fact_check_results=fact_check_results,
    )

    llm_output = llm_call(SYNTHESIS_SYSTEM_PROMPT, synthesis_prompt)

    synthesis = llm_output.get("answer", "Unable to synthesize results.")
    confidence = llm_output.get("confidence", 0.5)
    sources_cited = llm_output.get("sources_cited", [])

    return {
        "synthesis": synthesis,
        "confidence": confidence,
        "sources_cited": (
            [{"source_file": s} for s in sources_cited] if isinstance(sources_cited, list) else []
        ),
        "trace": state.get("trace", [])
        + [
            make_trace_entry(
                node="orchestrator",
                action="synthesize",
                inputs={
                    "corpus_count": len(state.get("corpus_results", [])),
                    "has_cross_ref": bool(state.get("cross_ref_results")),
                    "timeline_count": len(state.get("timeline_results", [])),
                    "fact_check_count": len(state.get("fact_check_results", [])),
                },
                outputs={
                    "confidence": confidence,
                    "sources_count": (len(sources_cited) if isinstance(sources_cited, list) else 0),
                },
            )
        ],
    }


def evaluate_node(state: ResearchState) -> dict:
    """
    Self-evaluate the synthesized answer quality.

    If quality is too low, triggers a retry (up to max_retries).
    """
    query = state["query"]
    synthesis = state.get("synthesis", "")
    confidence = state.get("confidence", 0.0)
    sources = state.get("sources_cited", [])
    retries = state.get("retries", 0)
    trace_id = state.get("trace_id", "?")

    logger.info(f"[{trace_id}] Self-evaluating (retry {retries})...")

    eval_prompt = SELF_EVAL_PROMPT.format(
        query=query,
        answer=synthesis,
        sources=json.dumps(sources, default=str),
    )

    llm_output = llm_call(SELF_EVAL_SYSTEM_PROMPT, eval_prompt)

    overall_score = llm_output.get("overall_score", confidence)
    should_retry = llm_output.get("should_retry", False)
    issues = llm_output.get("issues", [])

    # Only retry if score is low AND we haven't exceeded max retries
    max_retries = state.get("max_retries", 2)
    will_retry = should_retry and retries < max_retries and overall_score < 0.4

    if will_retry:
        logger.info(
            f"[{trace_id}] Evaluation score {overall_score:.2f} — "
            f"retrying ({retries + 1}/{max_retries})"
        )

    return {
        "confidence": overall_score,
        "retries": retries + (1 if will_retry else 0),
        "trace": state.get("trace", [])
        + [
            make_trace_entry(
                node="orchestrator",
                action="evaluate",
                inputs={
                    "synthesis_preview": synthesis[:200],
                    "retry_count": retries,
                },
                outputs={
                    "overall_score": overall_score,
                    "should_retry": will_retry,
                    "issues": issues,
                    "eval_details": {
                        k: llm_output.get(k, 0)
                        for k in [
                            "completeness",
                            "citation_quality",
                            "balance",
                            "confidence_calibration",
                        ]
                    },
                },
            )
        ],
    }


# --- Routing functions ---


def should_cross_reference(state: ResearchState) -> str:
    """Decide whether to run cross-referencing after corpus search."""
    plan = state.get("research_plan", [])
    agent_names = {a["agent"] for a in plan}

    if "cross_reference" in agent_names:
        return "cross_reference"
    if "timeline" in agent_names:
        return "timeline"
    if "fact_check" in agent_names:
        return "fact_check"
    return "synthesize"


def should_timeline(state: ResearchState) -> str:
    """Decide whether to run timeline after cross-reference."""
    plan = state.get("research_plan", [])
    agent_names = {a["agent"] for a in plan}

    if "timeline" in agent_names:
        return "timeline"
    if "fact_check" in agent_names:
        return "fact_check"
    return "synthesize"


def should_fact_check(state: ResearchState) -> str:
    """Decide whether to run fact-check after timeline."""
    plan = state.get("research_plan", [])
    agent_names = {a["agent"] for a in plan}

    if "fact_check" in agent_names:
        return "fact_check"
    return "synthesize"


def should_retry(state: ResearchState) -> str:
    """Decide whether to retry after evaluation."""
    retries = state.get("retries", 0)
    max_retries = state.get("max_retries", 2)
    confidence = state.get("confidence", 1.0)

    # Retry if confidence is very low and we haven't exceeded max retries
    if confidence < 0.4 and retries < max_retries:
        return "corpus_search"  # restart from corpus search
    return END


# --- Graph construction ---


def build_research_graph() -> StateGraph:
    """
    Build the LangGraph StateGraph for multi-agent research.

    Flow:
        analyze_query -> dispatch -> corpus_search ->
        [cross_reference] -> [timeline] -> [fact_check] ->
        synthesize -> evaluate -> [retry or END]
    """
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("analyze_query", analyze_query_node)
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("corpus_search", corpus_search_node)
    graph.add_node("cross_reference", cross_reference_node)
    graph.add_node("timeline", timeline_node)
    graph.add_node("fact_check", fact_check_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("evaluate", evaluate_node)

    # Set entry point
    graph.set_entry_point("analyze_query")

    # Fixed edges
    graph.add_edge("analyze_query", "dispatch")
    graph.add_edge("dispatch", "corpus_search")

    # Conditional edges based on research plan
    graph.add_conditional_edges(
        "corpus_search",
        should_cross_reference,
        {
            "cross_reference": "cross_reference",
            "timeline": "timeline",
            "fact_check": "fact_check",
            "synthesize": "synthesize",
        },
    )

    graph.add_conditional_edges(
        "cross_reference",
        should_timeline,
        {
            "timeline": "timeline",
            "fact_check": "fact_check",
            "synthesize": "synthesize",
        },
    )

    graph.add_conditional_edges(
        "timeline",
        should_fact_check,
        {
            "fact_check": "fact_check",
            "synthesize": "synthesize",
        },
    )

    graph.add_edge("fact_check", "synthesize")
    graph.add_edge("synthesize", "evaluate")

    # Retry logic
    graph.add_conditional_edges(
        "evaluate",
        should_retry,
        {
            "corpus_search": "corpus_search",
            END: END,
        },
    )

    return graph


# Compile the graph once at module level
_compiled_graph = None


def _get_graph():
    """Get or compile the research graph."""
    global _compiled_graph
    if _compiled_graph is None:
        graph = build_research_graph()
        _compiled_graph = graph.compile()
    return _compiled_graph


def run_agent_query(query: str, trace_id: str | None = None) -> dict:
    """
    Execute a multi-agent research query.

    This is the main entry point for agentic research.

    Args:
        query: The research question
        trace_id: Optional trace ID for observability

    Returns:
        dict with: synthesis, confidence, sources_cited, trace, trace_id,
                   corpus_results, cross_ref_results, timeline_results, fact_check_results
    """
    tid = trace_id or f"trace_{uuid.uuid4().hex[:12]}"
    logger.info(f"[{tid}] Starting agentic research: {query[:80]}...")

    # Initialize observability tracer
    tracer = ResearchTracer(trace_id=tid)
    pipeline_span = tracer.start_span(
        "orchestrator",
        "pipeline",
        inputs={"query": query},
    )

    compiled = _get_graph()

    initial_state: ResearchState = {
        "query": query,
        "query_type": "",
        "research_plan": [],
        "corpus_results": [],
        "cross_ref_results": {},
        "timeline_results": [],
        "fact_check_results": [],
        "synthesis": "",
        "confidence": 0.0,
        "sources_cited": [],
        "retries": 0,
        "max_retries": 2,
        "trace": [],
        "trace_id": tid,
    }

    # Execute the graph
    final_state = compiled.invoke(initial_state)

    # Record each graph trace entry into the observability tracer
    for entry in final_state.get("trace", []):
        tracer.log_decision(
            node=entry.get("node", "unknown"),
            decision=entry.get("action", "unknown"),
            reasoning=json.dumps(
                entry.get("outputs", {}),
                default=str,
            )[:300],
        )

    # End pipeline span
    tracer.end_span(
        pipeline_span,
        outputs={
            "confidence": final_state.get("confidence", 0.0),
            "retries": final_state.get("retries", 0),
            "query_type": final_state.get("query_type", ""),
        },
    )
    tracer.flush()

    # Store trace for later retrieval (includes both graph trace
    # and observability summary)
    trace_summary = {
        "trace_id": tid,
        "query": query,
        "query_type": final_state.get("query_type", ""),
        "research_plan": final_state.get("research_plan", []),
        "trace": final_state.get("trace", []),
        "retries": final_state.get("retries", 0),
        "confidence": final_state.get("confidence", 0.0),
        "observability": tracer.get_summary(),
    }
    _trace_store[tid] = trace_summary

    logger.info(
        f"[{tid}] Research complete | "
        f"confidence={final_state.get('confidence', 0):.2f} | "
        f"retries={final_state.get('retries', 0)}"
    )

    return {
        "synthesis": final_state.get("synthesis", ""),
        "confidence": final_state.get("confidence", 0.0),
        "sources_cited": final_state.get("sources_cited", []),
        "corpus_results": final_state.get("corpus_results", []),
        "cross_ref_results": final_state.get("cross_ref_results", {}),
        "timeline_results": final_state.get("timeline_results", []),
        "fact_check_results": final_state.get("fact_check_results", []),
        "query_type": final_state.get("query_type", ""),
        "research_plan": final_state.get("research_plan", []),
        "retries": final_state.get("retries", 0),
        "trace_id": tid,
        "trace": final_state.get("trace", []),
    }
