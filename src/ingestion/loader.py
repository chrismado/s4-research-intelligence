"""
Document loader — ingests raw files with metadata extraction.

Handles: PDF, DOCX, Markdown, plain text, and structured JSON manifests.
Each document gets rich metadata (source type, dates, subjects) that feeds
into the source-weighted retrieval pipeline.
"""

import hashlib
import json
from pathlib import Path

from loguru import logger

from config.settings import settings
from src.models.documents import DocumentMetadata, IngestedDocument, SourceType


def _generate_doc_id(filepath: str, content: str) -> str:
    """Deterministic document ID from path + content hash."""
    return hashlib.sha256(f"{filepath}:{content[:500]}".encode()).hexdigest()[:16]


def _detect_source_type(filepath: Path, manifest_type: str | None = None) -> SourceType:
    """Infer source type from filename conventions or manifest override."""
    if manifest_type:
        return SourceType(manifest_type)

    name = filepath.stem.lower()
    type_hints = {
        "transcript": SourceType.INTERVIEW_TRANSCRIPT,
        "interview": SourceType.INTERVIEW_TRANSCRIPT,
        "testimony": SourceType.EYEWITNESS_ACCOUNT,
        "foia": SourceType.GOVERNMENT_DOCUMENT,
        "gov": SourceType.GOVERNMENT_DOCUMENT,
        "classified": SourceType.GOVERNMENT_DOCUMENT,
        "archive": SourceType.ARCHIVAL_REFERENCE,
        "museum": SourceType.ARCHIVAL_REFERENCE,
        "news": SourceType.NEWS_ARTICLE,
        "article": SourceType.NEWS_ARTICLE,
        "paper": SourceType.SCIENTIFIC_PAPER,
        "journal": SourceType.SCIENTIFIC_PAPER,
        "book": SourceType.BOOK_EXCERPT,
        "note": SourceType.PRODUCTION_NOTE,
        "production": SourceType.PRODUCTION_NOTE,
    }
    for hint, stype in type_hints.items():
        if hint in name:
            return stype

    return SourceType.PRODUCTION_NOTE


def _load_text_file(filepath: Path) -> str:
    """Load plain text or markdown file."""
    return filepath.read_text(encoding="utf-8")


def _load_pdf(filepath: Path) -> str:
    """Load PDF using unstructured."""
    from unstructured.partition.pdf import partition_pdf

    elements = partition_pdf(str(filepath))
    return "\n\n".join(str(el) for el in elements)


def _load_docx(filepath: Path) -> str:
    """Load DOCX using unstructured."""
    from unstructured.partition.docx import partition_docx

    elements = partition_docx(str(filepath))
    return "\n\n".join(str(el) for el in elements)


def load_document(
    filepath: Path,
    metadata_override: dict | None = None,
) -> IngestedDocument:
    """
    Load a single document with metadata extraction.

    Args:
        filepath: Path to the raw document.
        metadata_override: Optional dict to override/supplement auto-detected metadata.

    Returns:
        IngestedDocument with content and rich metadata.
    """
    logger.info(f"Loading: {filepath.name}")

    suffix = filepath.suffix.lower()
    loaders = {
        ".txt": _load_text_file,
        ".md": _load_text_file,
        ".pdf": _load_pdf,
        ".docx": _load_docx,
    }

    loader = loaders.get(suffix)
    if not loader:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {list(loaders.keys())}")

    content = loader(filepath)

    # Build metadata
    override = metadata_override or {}
    source_type = _detect_source_type(filepath, override.get("source_type"))
    reliability = settings.source_reliability_weights.get(source_type.value, 0.5)

    metadata = DocumentMetadata(
        source_file=filepath.name,
        source_type=source_type,
        title=override.get("title", filepath.stem.replace("_", " ").title()),
        author=override.get("author"),
        date_created=override.get("date_created"),
        date_range_start=override.get("date_range_start"),
        date_range_end=override.get("date_range_end"),
        subjects=override.get("subjects", []),
        classification=override.get("classification"),
        language=override.get("language", "en"),
        reliability_score=reliability,
    )

    doc_id = _generate_doc_id(str(filepath), content)

    return IngestedDocument(
        id=doc_id,
        content=content,
        metadata=metadata,
    )


def load_from_manifest(manifest_path: Path) -> list[IngestedDocument]:
    """
    Batch-load documents from a JSON manifest file.

    The manifest is a JSON array where each entry specifies:
    {
        "file": "relative/path/to/doc.pdf",
        "source_type": "government_document",
        "title": "FOIA Release — S4 Facility Reference",
        "author": "US DOE",
        "date_created": "1989-11-01",
        "subjects": ["Bob Lazar", "S4", "Area 51"],
        "classification": "FOIA release"
    }
    """
    logger.info(f"Loading manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    base_dir = manifest_path.parent

    documents = []
    for entry in manifest:
        filepath = base_dir / entry.pop("file")
        if not filepath.exists():
            logger.warning(f"File not found, skipping: {filepath}")
            continue
        doc = load_document(filepath, metadata_override=entry)
        documents.append(doc)

    logger.info(f"Loaded {len(documents)} documents from manifest")
    return documents
