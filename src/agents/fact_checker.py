"""
Fact-Check Agent — verifies or refutes specific claims with evidence.

Searches the corpus for direct evidence, compares source types,
and assigns a verdict: VERIFIED / DISPUTED / UNVERIFIABLE / CONTRADICTED.
"""

from __future__ import annotations

from loguru import logger

from src.agents.state import ResearchState
from src.agents.tools import (
    assemble_context_tool,
    corpus_search_tool,
    filtered_search_tool,
    llm_call,
    make_trace_entry,
)
from src.prompts.agent_prompts import FACT_CHECK_SYSTEM_PROMPT


def fact_check_node(state: ResearchState) -> dict:
    """
    LangGraph node: verify or refute claims found in the research.

    Searches for evidence from multiple source types and asks the LLM
    to assign a verdict with supporting/contradicting citations.
    """
    query = state["query"]
    logger.info(f"Fact-check agent executing for: {query[:80]}...")

    # Step 1: Gather evidence from multiple angles
    direct_hits = corpus_search_tool(query=query, top_k=8)

    gov_hits = filtered_search_tool(
        query=query,
        source_type="government_document",
        top_k=3,
    )

    interview_hits = filtered_search_tool(
        query=query,
        source_type="interview_transcript",
        top_k=3,
    )

    # Combine all evidence (deduplicate by source_file)
    seen_files: set[str] = set()
    all_evidence = []
    for hit in direct_hits + gov_hits + interview_hits:
        source_file = hit.get("source_file", "")
        if source_file not in seen_files:
            seen_files.add(source_file)
            all_evidence.append(hit)

    if not all_evidence:
        return {
            "fact_check_results": [
                {
                    "claim": query,
                    "verdict": "UNVERIFIABLE",
                    "confidence": 0.1,
                    "supporting_sources": [],
                    "contradicting_sources": [],
                    "reasoning": ("No relevant evidence found in the corpus."),
                }
            ],
            "trace": state.get("trace", [])
            + [
                make_trace_entry(
                    node="fact_check",
                    action="verdict",
                    inputs={"query": query},
                    outputs={
                        "verdict": "UNVERIFIABLE",
                        "reason": "no_evidence",
                    },
                )
            ],
        }

    # Step 2: Build context and ask LLM for verdict
    context = assemble_context_tool(all_evidence)

    analysis_prompt = (
        "Fact-check the following claim using the "
        "provided evidence.\n\n"
        f"## Claim to Verify\n{query}\n\n"
        f"## Evidence\n{context}\n\n"
        "Assess the evidence for and against this claim. "
        "Assign a verdict."
    )
    llm_output = llm_call(FACT_CHECK_SYSTEM_PROMPT, analysis_prompt)

    fact_check_result = {
        "claim": llm_output.get("claim", query),
        "verdict": llm_output.get("verdict", "UNVERIFIABLE"),
        "confidence": llm_output.get("confidence", 0.5),
        "supporting_sources": llm_output.get("supporting_sources", []),
        "contradicting_sources": llm_output.get("contradicting_sources", []),
        "reasoning": llm_output.get("reasoning", ""),
    }

    return {
        "fact_check_results": [fact_check_result],
        "trace": state.get("trace", [])
        + [
            make_trace_entry(
                node="fact_check",
                action="verdict",
                inputs={
                    "query": query,
                    "evidence_count": len(all_evidence),
                },
                outputs={
                    "verdict": fact_check_result["verdict"],
                    "confidence": fact_check_result["confidence"],
                    "supporting_count": len(fact_check_result["supporting_sources"]),
                    "contradicting_count": len(fact_check_result["contradicting_sources"]),
                },
            )
        ],
    }
