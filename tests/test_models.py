"""Tests for Pydantic models — schema validation and edge cases."""

from datetime import date

import pytest

from src.models.documents import DocumentMetadata, SourceType
from src.models.queries import ResearchQuery, ResearchResponse


class TestDocumentMetadata:
    def test_create_with_defaults(self):
        meta = DocumentMetadata(
            source_file="test.txt",
            source_type=SourceType.PRODUCTION_NOTE,
            title="Test Document",
        )
        assert meta.reliability_score == 0.5
        assert meta.chunk_index == 0
        assert meta.language == "en"

    def test_reliability_bounds(self):
        with pytest.raises(Exception):  # noqa: B017
            DocumentMetadata(
                source_file="test.txt",
                source_type=SourceType.PRODUCTION_NOTE,
                title="Test",
                reliability_score=1.5,
            )

    def test_subjects_default_empty(self):
        meta = DocumentMetadata(
            source_file="test.txt",
            source_type=SourceType.INTERVIEW_TRANSCRIPT,
            title="Interview",
        )
        assert meta.subjects == []

    def test_all_source_types_valid(self):
        for st in SourceType:
            meta = DocumentMetadata(
                source_file="test.txt",
                source_type=st,
                title="Test",
            )
            assert meta.source_type == st


class TestResearchQuery:
    def test_minimal_query(self):
        q = ResearchQuery(question="What is S4?")
        assert q.source_types is None
        assert q.include_contradictions is True

    def test_filtered_query(self):
        q = ResearchQuery(
            question="Government docs about Area 51",
            source_types=[SourceType.GOVERNMENT_DOCUMENT],
            date_range_start=date(1990, 1, 1),
            date_range_end=date(2000, 12, 31),
        )
        assert len(q.source_types) == 1
        assert q.date_range_start.year == 1990


class TestResearchResponse:
    def test_empty_response(self):
        r = ResearchResponse(
            answer="No sources found.",
            sources=[],
            confidence=0.0,
        )
        assert r.contradictions == []
        assert r.timeline == []

    def test_confidence_bounds(self):
        with pytest.raises(Exception):  # noqa: B017
            ResearchResponse(answer="test", sources=[], confidence=1.5)
