"""
Corpus Search Agent — wraps the existing RAG pipeline as a LangGraph node.

This agent performs source-weighted semantic search with reranking
using the existing retrieval pipeline. It does NOT duplicate any
retrieval logic — it calls the existing tools.
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
from src.prompts.agent_prompts import CORPUS_SEARCH_SYSTEM_PROMPT


def corpus_search_node(state: ResearchState) -> dict:
    """
    LangGraph node: search the corpus for relevant information.

    Uses the existing RAG pipeline's hybrid search and source-weighted
    reranking. Adds LLM analysis of the retrieved results.
    """
    query = state["query"]
    logger.info(f"Corpus search agent executing for: {query[:80]}...")

    # Step 1: Retrieve using existing pipeline tools
    hits = corpus_search_tool(query=query, top_k=8)

    if not hits:
        return {
            "corpus_results": [],
            "trace": state.get("trace", [])
            + [
                make_trace_entry(
                    node="corpus_search",
                    action="search",
                    inputs={"query": query},
                    outputs={"result_count": 0},
                    error="No results found",
                )
            ],
        }

    # Step 2: Build context from hits
    context = assemble_context_tool(hits)

    # Step 3: LLM analysis of retrieved results
    analysis_prompt = (
        f"Analyze the following retrieved sources "
        f"to answer the research question.\n\n"
        f"## Research Question\n{query}\n\n"
        f"## Retrieved Sources\n{context}"
    )
    llm_output = llm_call(CORPUS_SEARCH_SYSTEM_PROMPT, analysis_prompt)

    # Step 4: Merge LLM analysis with raw hits for downstream agents
    corpus_results = []
    for hit in hits:
        corpus_results.append(
            {
                "content": hit["content"],
                "source_file": hit["source_file"],
                "source_type": hit["source_type"],
                "relevance_score": hit["relevance_score"],
                "reliability_score": hit["reliability_score"],
                "combined_score": hit["combined_score"],
                "excerpt": hit["excerpt"],
            }
        )

    # Attach LLM findings to state for synthesis
    findings = llm_output.get("findings", "")
    key_claims = llm_output.get("key_claims", [])
    gaps = llm_output.get("gaps", [])

    return {
        "corpus_results": corpus_results,
        "trace": state.get("trace", [])
        + [
            make_trace_entry(
                node="corpus_search",
                action="search_and_analyze",
                inputs={"query": query},
                outputs={
                    "result_count": len(corpus_results),
                    "findings_preview": findings[:200] if findings else "",
                    "key_claims_count": len(key_claims),
                    "gaps": gaps,
                },
            )
        ],
    }
