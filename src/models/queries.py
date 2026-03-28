"""
Query and response models for the research assistant API.
"""

from datetime import date

from pydantic import BaseModel, Field

from .documents import SourceType


class ResearchQuery(BaseModel):
    """A research question submitted to the assistant."""

    question: str = Field(
        min_length=1,
        max_length=10000,
        description="Natural language research question",
    )
    source_types: list[SourceType] | None = Field(
        default=None, description="Filter to specific source types"
    )
    date_range_start: date | None = Field(
        default=None, description="Only sources from after this date"
    )
    date_range_end: date | None = Field(
        default=None, description="Only sources before this date"
    )
    subjects: list[str] | None = Field(
        default=None, description="Filter to sources mentioning these subjects"
    )
    top_k: int | None = Field(default=None, description="Override default retrieval count")
    include_contradictions: bool = Field(
        default=True, description="Flag contradictions across sources"
    )


class SourceReference(BaseModel):
    """A cited source in the response with provenance."""

    source_file: str
    source_type: SourceType
    title: str
    author: str | None = None
    date_created: date | None = None
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

    date: str | None = None
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
        ge=0.0,
        le=1.0,
        description="Overall answer confidence based on source quality and agreement",
    )
    reasoning: str = Field(
        default="", description="Chain-of-thought reasoning trace for transparency"
    )
