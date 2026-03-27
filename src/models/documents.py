"""
Document models — schema for ingested sources with rich metadata.

Every document in the S4 corpus carries provenance metadata that feeds into
source-weighted retrieval. This is what separates this from a tutorial RAG:
we score retrieval results not just by semantic similarity but by source
reliability, temporal relevance, and cross-reference density.
"""

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    INTERVIEW_TRANSCRIPT = "interview_transcript"
    GOVERNMENT_DOCUMENT = "government_document"
    ARCHIVAL_REFERENCE = "archival_reference"
    NEWS_ARTICLE = "news_article"
    PRODUCTION_NOTE = "production_note"
    EYEWITNESS_ACCOUNT = "eyewitness_account"
    SCIENTIFIC_PAPER = "scientific_paper"
    BOOK_EXCERPT = "book_excerpt"


class DocumentMetadata(BaseModel):
    """Rich metadata attached to every chunk in the vector store."""

    source_file: str = Field(description="Original filename")
    source_type: SourceType
    title: str = Field(description="Document or section title")
    author: str | None = None
    date_created: date | None = None
    date_range_start: date | None = None
    date_range_end: date | None = None
    subjects: list[str] = Field(
        default_factory=list, description="People, places, programs mentioned"
    )
    classification: str | None = Field(
        default=None, description="e.g. 'FOIA release', 'public testimony', 'classified (leaked)'"
    )
    language: str = "en"
    reliability_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Source reliability weight for retrieval scoring"
    )
    chunk_index: int = Field(
        default=0, description="Position of this chunk within the source document"
    )
    total_chunks: int = Field(default=1, description="Total chunks from this source document")


class IngestedDocument(BaseModel):
    """A document after ingestion, before chunking."""

    id: str
    content: str
    metadata: DocumentMetadata
    token_count: int = 0


class DocumentChunk(BaseModel):
    """A chunked fragment ready for embedding and vector storage."""

    id: str
    content: str
    metadata: DocumentMetadata
    embedding: list[float] | None = None
