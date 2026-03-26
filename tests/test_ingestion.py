"""Tests for document ingestion pipeline."""

import pytest
from pathlib import Path

from src.ingestion.loader import _detect_source_type, _generate_doc_id
from src.models.documents import SourceType


class TestSourceTypeDetection:
    def test_interview_transcript(self):
        assert _detect_source_type(Path("lazar_interview_1989.txt")) == SourceType.INTERVIEW_TRANSCRIPT

    def test_government_document(self):
        assert _detect_source_type(Path("foia_release_doe.pdf")) == SourceType.GOVERNMENT_DOCUMENT

    def test_news_article(self):
        assert _detect_source_type(Path("klas_news_report.txt")) == SourceType.NEWS_ARTICLE

    def test_scientific_paper(self):
        assert _detect_source_type(Path("element115_paper.pdf")) == SourceType.SCIENTIFIC_PAPER

    def test_default_production_note(self):
        assert _detect_source_type(Path("random_file.txt")) == SourceType.PRODUCTION_NOTE

    def test_manifest_override(self):
        assert _detect_source_type(
            Path("anything.txt"), manifest_type="eyewitness_account"
        ) == SourceType.EYEWITNESS_ACCOUNT


class TestDocumentId:
    def test_deterministic(self):
        id1 = _generate_doc_id("file.txt", "content here")
        id2 = _generate_doc_id("file.txt", "content here")
        assert id1 == id2

    def test_different_content_different_id(self):
        id1 = _generate_doc_id("file.txt", "content A")
        id2 = _generate_doc_id("file.txt", "content B")
        assert id1 != id2
