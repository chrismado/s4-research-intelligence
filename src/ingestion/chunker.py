"""
Intelligent document chunking with metadata propagation.

Uses recursive character splitting with overlap, but preserves document
structure (headings, paragraphs) and propagates source metadata to every
chunk. This ensures retrieval results always carry full provenance.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from config.settings import settings
from src.models.documents import DocumentChunk, IngestedDocument


def chunk_document(document: IngestedDocument) -> list[DocumentChunk]:
    """
    Split a document into overlapping chunks with metadata propagation.

    Each chunk inherits the parent document's metadata, plus gets a
    chunk_index and total_chunks field for reassembly context.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    text_chunks = splitter.split_text(document.content)
    total = len(text_chunks)

    chunks = []
    for i, text in enumerate(text_chunks):
        metadata = document.metadata.model_copy(
            update={"chunk_index": i, "total_chunks": total}
        )

        chunk = DocumentChunk(
            id=f"{document.id}_chunk_{i:04d}",
            content=text,
            metadata=metadata,
        )
        chunks.append(chunk)

    logger.debug(f"Chunked '{document.metadata.title}' → {total} chunks")
    return chunks


def chunk_documents(documents: list[IngestedDocument]) -> list[DocumentChunk]:
    """Chunk a batch of documents."""
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))

    logger.info(
        f"Total chunks: {len(all_chunks)} from {len(documents)} documents"
    )
    return all_chunks
