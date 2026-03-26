"""
Retrieval pipeline — source-weighted semantic search with reranking.

This is the core differentiator: retrieval results are scored by a weighted
combination of semantic similarity and source reliability. Government documents
and eyewitness accounts rank higher than news articles for the same similarity
score. This produces research-grade results, not generic RAG output.
"""

import json
import time
from typing import Optional

import httpx
from loguru import logger
from ollama import Client as OllamaClient

from config.settings import settings
from src.ingestion.vectorstore import VectorStore
from src.models.documents import SourceType
from src.models.queries import (
    Contradiction,
    ResearchQuery,
    ResearchResponse,
    SourceReference,
    TimelineEvent,
)
from src.prompts.templates import RESEARCH_QUERY_PROMPT, SYSTEM_PROMPT
from src.retrieval.memory import ConversationMemory


class ResearchPipeline:
    """
    End-to-end retrieval-augmented generation pipeline.

    Flow:
    1. Query → vector search (with optional metadata filters)
    2. Source-weighted reranking
    3. Context assembly with provenance headers
    4. LLM generation with structured output
    5. Response parsing with citation validation
    """

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()
        self.llm_client = OllamaClient(host=settings.llm_base_url)
        self.memory = ConversationMemory()

    def _build_metadata_filter(self, query: ResearchQuery) -> Optional[dict]:
        """Convert query filters to ChromaDB where clause."""
        conditions = []

        if query.source_types:
            type_values = [st.value for st in query.source_types]
            if len(type_values) == 1:
                conditions.append({"source_type": {"$eq": type_values[0]}})
            else:
                conditions.append({"source_type": {"$in": type_values}})

        if query.date_range_start:
            conditions.append({"date_created": {"$gte": str(query.date_range_start)}})
        if query.date_range_end:
            conditions.append({"date_created": {"$lte": str(query.date_range_end)}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _source_weighted_rerank(self, hits: list[dict], top_n: int) -> list[dict]:
        """
        Rerank retrieval hits using combined relevance + reliability scoring.

        combined_score = (relevance_score * 0.65) + (reliability_score * 0.35)

        This means a government FOIA document at 0.7 similarity outranks a
        news article at 0.8 similarity — which is correct for documentary research.
        """
        for hit in hits:
            relevance = hit["relevance_score"]
            reliability = hit["metadata"].get("reliability_score", 0.5)
            hit["combined_score"] = (relevance * 0.65) + (reliability * 0.35)

        hits.sort(key=lambda h: h["combined_score"], reverse=True)

        if settings.rerank_enabled:
            return hits[:top_n]
        return hits

    def _assemble_context(self, hits: list[dict]) -> str:
        """
        Assemble retrieved chunks into a structured context block.

        Each chunk gets a provenance header so the LLM can cite properly.
        """
        context_parts = []
        for i, hit in enumerate(hits, 1):
            meta = hit["metadata"]
            header = (
                f"--- Source {i}: {meta['source_file']} ---\n"
                f"Type: {meta['source_type']} | "
                f"Title: {meta['title']} | "
                f"Author: {meta.get('author', 'Unknown')} | "
                f"Date: {meta.get('date_created', 'Unknown')} | "
                f"Reliability: {meta.get('reliability_score', 0.5):.2f}"
            )
            context_parts.append(f"{header}\n{hit['content']}")

        return "\n\n".join(context_parts)

    def _generate(self, context: str, question: str) -> dict:
        """Call LLM with assembled context and parse structured response."""
        prompt = RESEARCH_QUERY_PROMPT.format(context=context, question=question)

        response = self.llm_client.chat(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
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
            logger.warning("LLM returned non-JSON response, wrapping as plain answer")
            return {
                "answer": raw,
                "sources_used": [],
                "contradictions": [],
                "timeline_events": [],
                "confidence": 0.3,
                "reasoning": "Response was not structured JSON — confidence reduced.",
            }

    def query(self, research_query: ResearchQuery) -> ResearchResponse:
        """
        Execute the full RAG pipeline for a research question.

        Returns a structured response with citations, contradictions,
        timeline events, and confidence scoring.
        """
        logger.info(f"Research query: {research_query.question[:100]}...")

        # 1. Vector search with metadata filters
        where_filter = self._build_metadata_filter(research_query)
        top_k = research_query.top_k or settings.retrieval_top_k

        if settings.hybrid_search_enabled:
            hits = self.vector_store.hybrid_search(
                query=research_query.question,
                top_k=top_k * 2,
                where=where_filter,
                semantic_weight=settings.hybrid_semantic_weight,
                keyword_weight=settings.hybrid_keyword_weight,
            )
        else:
            hits = self.vector_store.search(
                query=research_query.question,
                top_k=top_k * 2,
                where=where_filter,
            )

        if not hits:
            return ResearchResponse(
                answer="No relevant sources found in the research corpus for this query.",
                sources=[],
                confidence=0.0,
                reasoning="Vector search returned zero results. The corpus may not contain information about this topic.",
            )

        # 2. Source-weighted reranking
        rerank_n = research_query.top_k or settings.rerank_top_n
        ranked_hits = self._source_weighted_rerank(hits, top_n=rerank_n)

        # 3. Assemble context
        context = self._assemble_context(ranked_hits)

        # 4. Generate response
        llm_output = self._generate(context, research_query.question)

        # 5. Build structured response
        sources = []
        for hit in ranked_hits:
            meta = hit["metadata"]
            sources.append(
                SourceReference(
                    source_file=meta["source_file"],
                    source_type=SourceType(meta["source_type"]),
                    title=meta["title"],
                    author=meta.get("author") or None,
                    date_created=meta.get("date_created") or None,
                    relevance_score=round(hit["relevance_score"], 4),
                    reliability_score=meta.get("reliability_score", 0.5),
                    combined_score=round(hit["combined_score"], 4),
                    excerpt=hit["content"][:300],
                )
            )

        contradictions = [
            Contradiction(**c) for c in llm_output.get("contradictions", [])
        ]

        timeline = [
            TimelineEvent(**e) for e in llm_output.get("timeline_events", [])
        ]

        return ResearchResponse(
            answer=llm_output.get("answer", "No answer generated."),
            sources=sources,
            contradictions=contradictions,
            timeline=timeline,
            confidence=llm_output.get("confidence", 0.5),
            reasoning=llm_output.get("reasoning", ""),
        )

    def query_with_memory(self, research_query: ResearchQuery) -> ResearchResponse:
        """
        Execute RAG pipeline with conversation memory.

        Enriches the question with prior conversation context,
        then stores the result for future turns.
        """
        original_question = research_query.question
        enriched = self.memory.enrich_question(original_question)
        research_query = research_query.model_copy(update={"question": enriched})

        response = self.query(research_query)

        # Store this turn in memory
        sources_summary = ", ".join(s.source_file for s in response.sources[:3])
        self.memory.add_turn(
            question=original_question,
            answer=response.answer,
            sources_summary=sources_summary,
        )

        return response

    # --- Async support for production FastAPI deployment ---

    async def _async_generate(self, context: str, question: str) -> dict:
        """Async LLM call using httpx for non-blocking FastAPI routes."""
        prompt = RESEARCH_QUERY_PROMPT.format(context=context, question=question)

        async with httpx.AsyncClient(timeout=float(settings.llm_timeout)) as client:
            resp = await client.post(
                f"{settings.llm_base_url}/api/chat",
                json={
                    "model": settings.llm_model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "options": {
                        "temperature": settings.llm_temperature,
                        "num_predict": settings.llm_max_tokens,
                        **({"num_gpu": settings.llm_num_gpu} if settings.llm_num_gpu >= 0 else {}),
                    },
                    "format": "json",
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        raw = data["message"]["content"]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON response, wrapping as plain answer")
            return {
                "answer": raw,
                "sources_used": [],
                "contradictions": [],
                "timeline_events": [],
                "confidence": 0.3,
                "reasoning": "Response was not structured JSON -- confidence reduced.",
            }

    async def async_query(self, research_query: ResearchQuery) -> ResearchResponse:
        """
        Async version of query() for non-blocking FastAPI routes.

        Uses httpx for the LLM call; retrieval is still synchronous
        (ChromaDB is fast enough that async isn't needed there).
        """
        start = time.perf_counter()
        logger.info(f"Async research query: {research_query.question[:100]}...")

        # 1. Vector search (sync — sub-millisecond)
        where_filter = self._build_metadata_filter(research_query)
        top_k = research_query.top_k or settings.retrieval_top_k

        if settings.hybrid_search_enabled:
            hits = self.vector_store.hybrid_search(
                query=research_query.question,
                top_k=top_k * 2,
                where=where_filter,
                semantic_weight=settings.hybrid_semantic_weight,
                keyword_weight=settings.hybrid_keyword_weight,
            )
        else:
            hits = self.vector_store.search(
                query=research_query.question,
                top_k=top_k * 2,
                where=where_filter,
            )

        if not hits:
            return ResearchResponse(
                answer="No relevant sources found in the research corpus for this query.",
                sources=[],
                confidence=0.0,
                reasoning="Vector search returned zero results.",
            )

        # 2. Rerank
        rerank_n = research_query.top_k or settings.rerank_top_n
        ranked_hits = self._source_weighted_rerank(hits, top_n=rerank_n)

        # 3. Context
        context = self._assemble_context(ranked_hits)

        # 4. Async LLM generation
        llm_output = await self._async_generate(context, research_query.question)

        # 5. Build response (same as sync)
        sources = []
        for hit in ranked_hits:
            meta = hit["metadata"]
            sources.append(
                SourceReference(
                    source_file=meta["source_file"],
                    source_type=SourceType(meta["source_type"]),
                    title=meta["title"],
                    author=meta.get("author") or None,
                    date_created=meta.get("date_created") or None,
                    relevance_score=round(hit["relevance_score"], 4),
                    reliability_score=meta.get("reliability_score", 0.5),
                    combined_score=round(hit["combined_score"], 4),
                    excerpt=hit["content"][:300],
                )
            )

        contradictions = [
            Contradiction(**c) for c in llm_output.get("contradictions", [])
        ]
        timeline = [
            TimelineEvent(**e) for e in llm_output.get("timeline_events", [])
        ]

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"Async query complete | latency={elapsed_ms:.0f}ms | "
            f"sources={len(sources)} | confidence={llm_output.get('confidence', 0.5):.2f}"
        )

        return ResearchResponse(
            answer=llm_output.get("answer", "No answer generated."),
            sources=sources,
            contradictions=contradictions,
            timeline=timeline,
            confidence=llm_output.get("confidence", 0.5),
            reasoning=llm_output.get("reasoning", ""),
        )
