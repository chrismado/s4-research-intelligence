"""
Timeline Agent — extracts and orders chronological events from sources.

Validates date consistency across sources, detects temporal conflicts,
and builds a structured timeline with confidence scores per event.
"""

from __future__ import annotations

from loguru import logger

from src.agents.state import ResearchState
from src.agents.tools import (
    assemble_context_tool,
    corpus_search_tool,
    llm_call,
    make_trace_entry,
)
from src.prompts.agent_prompts import TIMELINE_SYSTEM_PROMPT


def timeline_node(state: ResearchState) -> dict:
    """
    LangGraph node: extract and validate a chronological timeline.

    Uses corpus results if available, otherwise performs its own search.
    Focuses on extracting explicit date mentions and ordering events.
    """
    query = state["query"]
    corpus_results = state.get("corpus_results", [])
    logger.info(f"Timeline agent executing for: {query[:80]}...")

    # Step 1: Use existing corpus results or search
    evidence = corpus_results or corpus_search_tool(query=query, top_k=10)

    if not evidence:
        return {
            "timeline_results": [],
            "trace": state.get("trace", [])
            + [
                make_trace_entry(
                    node="timeline",
                    action="skip",
                    inputs={"query": query},
                    outputs={"reason": "no_evidence"},
                )
            ],
        }

    # Step 2: Build context and extract timeline
    context = assemble_context_tool(evidence)

    analysis_prompt = (
        "Extract a chronological timeline from the "
        "following source material.\n\n"
        f"## Research Focus\n{query}\n\n"
        f"## Source Material\n{context}\n\n"
        "Extract only events with explicit dates. "
        "Validate date consistency across sources."
    )
    llm_output = llm_call(TIMELINE_SYSTEM_PROMPT, analysis_prompt)

    # Step 3: Parse timeline events
    events = llm_output.get("events", [])
    conflicts = llm_output.get("conflicts", [])
    span = llm_output.get("span", "")

    timeline_results = []
    for event in events:
        timeline_results.append(
            {
                "date": event.get("date"),
                "description": event.get("description", ""),
                "source": event.get("source", "unknown"),
                "confidence": event.get("confidence", 0.5),
                "category": event.get("category", ""),
            }
        )

    # Sort by date where possible
    def sort_key(e: dict) -> str:
        return e.get("date") or "9999"

    timeline_results.sort(key=sort_key)

    return {
        "timeline_results": timeline_results,
        "trace": state.get("trace", [])
        + [
            make_trace_entry(
                node="timeline",
                action="extract",
                inputs={
                    "query": query,
                    "evidence_count": len(evidence),
                },
                outputs={
                    "events_count": len(timeline_results),
                    "conflicts_count": len(conflicts),
                    "span": span,
                },
            )
        ],
    }
