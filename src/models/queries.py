"""
Query and response models for the research assistant API.
"""

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field

from .documents import SourceType


class ResearchQuery(BaseModel):
    """A research question submitted to the assistant."""

    question: str = Field(description="Natural language research question")
    source_types: Optional[list[SourceType]] = Field(
        default=None, description="Filter to specific source types"
    )
    date_range_start: Optional[date] = Field(default=None, description="Only sources from after this date")
    date_range_end: Optional[date] = Field(default=None, description="Only sources before this date")
    subjects: Optional[list[str]] = Field(
        default=None, description="Filter to sources mentioning these subjects"
    )
    top_k: Optional[int] = Field(default=None, description="Override default retrieval count")
    include_contradictions: bool = Field(
        default=True, description="Flag contradictions across sources"
    )


class SourceReference(BaseModel):
    """A cited source in the response with provenance."""

    source_file: str
    source_type: SourceType
    title: str
    author: Optional[str] = None
    date_created: Optional[date] = None
    relevance_score: float = Field(description="Semantic similarity score")
    reliability_score: float = Field(description="Source reliability weight")
    combined_score: float = Field(description="Weighted final score")
    excerpt: str = Field(description="Relevant excerpt from this source")


class Contradiction(BaseModel):
    """A detected contradiction between two sources."""

    claim_a: str
    source_a: str
    claim_b: str
    source_b: str
    explanation: str


class TimelineEvent(BaseModel):
    """An extracted event for timeline reconstruction."""

    date: Optional[str] = None
    description: str
    source: str
    confidence: float = Field(ge=0.0, le=1.0)


class ResearchResponse(BaseModel):
    """Full response from the research assistant."""

    answer: str = Field(description="Synthesized answer with inline citations")
    sources: list[SourceReference] = Field(description="Cited sources with provenance")
    contradictions: list[Contradiction] = Field(
        default_factory=list, description="Detected contradictions across sources"
    )
    timeline: list[TimelineEvent] = Field(
        default_factory=list, description="Extracted timeline events relevant to query"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Overall answer confidence based on source quality and agreement"
    )
    reasoning: str = Field(
        default="", description="Chain-of-thought reasoning trace for transparency"
    )
