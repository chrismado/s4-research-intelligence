"""
Tool wrappers that expose existing pipeline functionality to agents.

These tools bridge the existing RAG pipeline and vector store with
the new agent orchestration layer. Each tool wraps existing code —
no retrieval logic is duplicated here.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from loguru import logger
from ollama import Client as OllamaClient

from config.settings import settings
from src.ingestion.vectorstore import VectorStore
from src.models.queries import ResearchQuery
from src.retrieval.pipeline import ResearchPipeline


def make_trace_entry(
    node: str,
    action: str,
    inputs: dict | None = None,
    outputs: dict | None = None,
    error: str | None = None,
    duration_ms: float = 0.0,
) -> dict:
    """Create a trace entry with a real UTC timestamp."""
    return {
        "node": node,
        "action": action,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": duration_ms,
        "inputs": inputs or {},
        "outputs": outputs or {},
        "error": error,
    }


# Singleton instances shared across agents
_vector_store: VectorStore | None = None
_pipeline: ResearchPipeline | None = None
_llm_client: OllamaClient | None = None


def _get_vector_store() -> VectorStore:
    """Get or create the shared vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def _get_pipeline() -> ResearchPipeline:
    """Get or create the shared research pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = ResearchPipeline(vector_store=_get_vector_store())
    return _pipeline


def _get_llm_client() -> OllamaClient:
    """Get or create the shared Ollama client."""
    global _llm_client
    if _llm_client is None:
        _llm_client = OllamaClient(host=settings.llm_base_url)
    return _llm_client


def llm_call(system_prompt: str, user_prompt: str) -> dict:
    """
    Call the LLM with a system prompt and user prompt, returning parsed JSON.

    Falls back to wrapping raw text if JSON parsing fails.
    """
    client = _get_llm_client()

    response = client.chat(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": settings.llm_temperature,
            "num_predict": settings.llm_max_tokens,
            **({"num_gpu": settings.llm_num_gpu} if settings.llm_num_gpu >= 0 else {}),
        },
        format="json",
    )

    raw = response["message"]["content"]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON response, wrapping as raw text")
        return {"raw_response": raw, "parse_error": True}


def corpus_search_tool(
    query: str,
    source_types: list[str] | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """
    Search the corpus using the existing RAG pipeline's retrieval and reranking.

    Wraps VectorStore.hybrid_search + ResearchPipeline._source_weighted_rerank.
    Returns ranked results with content, metadata, and scores.
    """
    store = _get_vector_store()
    pipeline = _get_pipeline()

    # Build metadata filter
    where = None
    if source_types:
        if len(source_types) == 1:
            where = {"source_type": {"$eq": source_types[0]}}
        else:
            where = {"source_type": {"$in": source_types}}

    k = top_k or settings.retrieval_top_k

    # Use hybrid search if enabled (same as existing pipeline)
    if settings.hybrid_search_enabled:
        hits = store.hybrid_search(
            query=query,
            top_k=k * 2,
            where=where,
            semantic_weight=settings.hybrid_semantic_weight,
            keyword_weight=settings.hybrid_keyword_weight,
        )
    else:
        hits = store.search(query=query, top_k=k * 2, where=where)

    if not hits:
        return []

    # Apply source-weighted reranking (the core differentiator)
    rerank_n = top_k or settings.rerank_top_n
    ranked = pipeline._source_weighted_rerank(hits, top_n=rerank_n)

    # Return structured results
    results = []
    for hit in ranked:
        meta = hit["metadata"]
        results.append(
            {
                "content": hit["content"],
                "source_file": meta["source_file"],
                "source_type": meta["source_type"],
                "title": meta.get("title", "Unknown"),
                "author": meta.get("author", "Unknown"),
                "date_created": meta.get("date_created", "Unknown"),
                "relevance_score": round(hit["relevance_score"], 4),
                "reliability_score": meta.get("reliability_score", 0.5),
                "combined_score": round(hit["combined_score"], 4),
                "excerpt": hit["content"][:300],
            }
        )

    logger.debug(f"Corpus search returned {len(results)} results for: {query[:80]}")
    return results


def full_rag_query_tool(question: str) -> dict:
    """
    Run the full existing RAG pipeline (retrieval + LLM generation).

    This is the existing pipeline used by the /api/v1/research endpoint.
    Returns the complete ResearchResponse as a dict.
    """
    pipeline = _get_pipeline()
    rq = ResearchQuery(question=question)
    response = pipeline.query(rq)
    return response.model_dump(mode="json")


def filtered_search_tool(
    query: str,
    source_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    top_k: int = 10,
) -> list[dict]:
    """
    Search the corpus with specific metadata filters.

    Used by cross-reference and fact-check agents to retrieve
    evidence from specific source types or date ranges.
    """
    store = _get_vector_store()
    pipeline = _get_pipeline()

    conditions = []
    if source_type:
        conditions.append({"source_type": {"$eq": source_type}})
    if date_from:
        conditions.append({"date_created": {"$gte": date_from}})
    if date_to:
        conditions.append({"date_created": {"$lte": date_to}})

    where = None
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    hits = store.search(query=query, top_k=top_k, where=where)

    if not hits:
        return []

    ranked = pipeline._source_weighted_rerank(hits, top_n=top_k)

    results = []
    for hit in ranked:
        meta = hit["metadata"]
        results.append(
            {
                "content": hit["content"],
                "source_file": meta["source_file"],
                "source_type": meta["source_type"],
                "title": meta.get("title", "Unknown"),
                "relevance_score": round(hit["relevance_score"], 4),
                "reliability_score": meta.get("reliability_score", 0.5),
                "combined_score": round(hit["combined_score"], 4),
            }
        )

    return results


def assemble_context_tool(hits: list[dict]) -> str:
    """
    Assemble retrieval hits into a formatted context block.

    Uses the existing pipeline's context assembly logic.
    """
    pipeline = _get_pipeline()
    # Transform tool results back into the format pipeline._assemble_context expects
    formatted_hits = []
    for hit in hits:
        formatted_hits.append(
            {
                "content": hit.get("content", ""),
                "metadata": {
                    "source_file": hit.get("source_file", "unknown"),
                    "source_type": hit.get("source_type", "unknown"),
                    "title": hit.get("title", "Unknown"),
                    "author": hit.get("author", "Unknown"),
                    "date_created": hit.get("date_created", "Unknown"),
                    "reliability_score": hit.get("reliability_score", 0.5),
                },
            }
        )
    return pipeline._assemble_context(formatted_hits)
