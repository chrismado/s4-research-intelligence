"""
Cross-Reference Agent — finds corroborating or contradicting evidence.

Takes claims from corpus search results and validates them by
searching for evidence across different source types. Produces
a corroboration report showing agreement and disagreement.
"""

from __future__ import annotations

from loguru import logger

from src.agents.state import ResearchState
from src.agents.tools import (
    assemble_context_tool,
    filtered_search_tool,
    llm_call,
    make_trace_entry,
)
from src.prompts.agent_prompts import CROSS_REFERENCE_SYSTEM_PROMPT

# Source types to cross-reference against
_SOURCE_TYPES = [
    "government_document",
    "interview_transcript",
    "scientific_paper",
    "news_article",
    "eyewitness_account",
    "archival_reference",
]


def cross_reference_node(state: ResearchState) -> dict:
    """
    LangGraph node: cross-reference claims across source types.

    Searches for evidence from multiple source types and compares
    what different categories of sources say about the same claims.
    """
    query = state["query"]
    corpus_results = state.get("corpus_results", [])
    logger.info(f"Cross-reference agent executing for: {query[:80]}...")

    if not corpus_results:
        return {
            "cross_ref_results": {
                "claim": query,
                "corroborating": [],
                "contradicting": [],
                "unresolved": [],
                "summary": ("No corpus results available " "for cross-referencing."),
            },
            "trace": state.get("trace", [])
            + [
                make_trace_entry(
                    node="cross_reference",
                    action="skip",
                    inputs={"query": query},
                    outputs={"reason": "no_corpus_results"},
                )
            ],
        }

    # Step 1: Gather evidence from each source type
    all_evidence = []
    for source_type in _SOURCE_TYPES:
        hits = filtered_search_tool(
            query=query,
            source_type=source_type,
            top_k=3,
        )
        all_evidence.extend(hits)

    if not all_evidence:
        return {
            "cross_ref_results": {
                "claim": query,
                "corroborating": [],
                "contradicting": [],
                "unresolved": [],
                "summary": ("No evidence found across source types " "for cross-referencing."),
            },
            "trace": state.get("trace", [])
            + [
                make_trace_entry(
                    node="cross_reference",
                    action="search",
                    inputs={
                        "query": query,
                        "source_types_searched": _SOURCE_TYPES,
                    },
                    outputs={"evidence_count": 0},
                )
            ],
        }

    # Step 2: Build context and ask LLM to analyze cross-references
    context = assemble_context_tool(all_evidence)

    analysis_prompt = (
        "Cross-reference the following evidence "
        "from different source types.\n\n"
        f"## Research Question / Claim\n{query}\n\n"
        f"## Evidence from Multiple Source Types\n{context}\n\n"
        "Analyze whether these sources corroborate "
        "or contradict each other."
    )
    llm_output = llm_call(CROSS_REFERENCE_SYSTEM_PROMPT, analysis_prompt)

    cross_ref_results = {
        "claim": llm_output.get("claim", query),
        "corroborating": llm_output.get("corroborating", []),
        "contradicting": llm_output.get("contradicting", []),
        "unresolved": llm_output.get("unresolved", []),
        "summary": llm_output.get("summary", ""),
    }

    return {
        "cross_ref_results": cross_ref_results,
        "trace": state.get("trace", [])
        + [
            make_trace_entry(
                node="cross_reference",
                action="analyze",
                inputs={
                    "query": query,
                    "evidence_count": len(all_evidence),
                },
                outputs={
                    "corroborating_count": len(cross_ref_results["corroborating"]),
                    "contradicting_count": len(cross_ref_results["contradicting"]),
                    "unresolved_count": len(cross_ref_results["unresolved"]),
                },
            )
        ],
    }
