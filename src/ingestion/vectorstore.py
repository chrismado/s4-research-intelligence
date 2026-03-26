"""
Vector store management — ChromaDB with metadata-rich storage + BM25 hybrid search.

Handles embedding, upserting, and persistence of document chunks.
Metadata is stored alongside vectors for filtered retrieval.
Hybrid search combines semantic (vector) and keyword (BM25) results
for improved retrieval of specific names, dates, and identifiers.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger
from rank_bm25 import BM25Okapi

from config.settings import settings
from src.models.documents import DocumentChunk


class VectorStore:
    """ChromaDB-backed vector store with HuggingFace embeddings."""

    def __init__(self):
        self._embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": settings.embedding_device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
        )

        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

        # BM25 index for hybrid search (lazy-built on first hybrid query)
        self._bm25_index: BM25Okapi | None = None
        self._bm25_corpus: list[dict] | None = None

        logger.info(
            f"VectorStore ready: {self._collection.count()} chunks in '{settings.chroma_collection}'"
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self._embeddings.embed_query(text)

    def add_chunks(self, chunks: list[DocumentChunk], batch_size: int = 100) -> int:
        """
        Embed and upsert chunks into the vector store.

        Returns the number of chunks added.
        """
        added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            ids = [c.id for c in batch]
            texts = [c.content for c in batch]
            embeddings = self._embeddings.embed_documents(texts)

            # Flatten metadata for ChromaDB (no nested objects)
            metadatas = []
            for c in batch:
                meta = {
                    "source_file": c.metadata.source_file,
                    "source_type": c.metadata.source_type.value,
                    "title": c.metadata.title,
                    "author": c.metadata.author or "",
                    "date_created": str(c.metadata.date_created) if c.metadata.date_created else "",
                    "subjects": ",".join(c.metadata.subjects),
                    "classification": c.metadata.classification or "",
                    "reliability_score": c.metadata.reliability_score,
                    "chunk_index": c.metadata.chunk_index,
                    "total_chunks": c.metadata.total_chunks,
                }
                metadatas.append(meta)

            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            added += len(batch)
            logger.debug(f"Upserted batch {i // batch_size + 1}: {len(batch)} chunks")

        logger.info(f"Added {added} chunks to vector store (total: {self.count})")
        return added

    def search(
        self,
        query: str,
        top_k: int | None = None,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Semantic search with optional metadata filtering.

        Returns list of dicts with: id, content, metadata, distance.
        """
        k = top_k or settings.retrieval_top_k
        query_embedding = self.embed_text(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "relevance_score": 1.0 - results["distances"][0][i],  # cosine distance → similarity
            })

        return hits

    def _build_bm25_index(self):
        """Build BM25 index from all documents in the collection."""
        total = self._collection.count()
        if total == 0:
            self._bm25_index = None
            self._bm25_corpus = []
            return

        # Fetch all documents from ChromaDB
        all_docs = self._collection.get(
            include=["documents", "metadatas"],
            limit=total,
        )

        self._bm25_corpus = []
        tokenized = []
        for i in range(len(all_docs["ids"])):
            entry = {
                "id": all_docs["ids"][i],
                "content": all_docs["documents"][i],
                "metadata": all_docs["metadatas"][i],
            }
            self._bm25_corpus.append(entry)
            tokenized.append(all_docs["documents"][i].lower().split())

        self._bm25_index = BM25Okapi(tokenized)
        logger.debug(f"BM25 index built with {len(tokenized)} documents")

    def hybrid_search(
        self,
        query: str,
        top_k: int | None = None,
        where: dict | None = None,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> list[dict]:
        """
        Hybrid search combining semantic (vector) and keyword (BM25) results.

        BM25 excels at matching specific names, dates, and identifiers
        (e.g., "Element 115", "Bob Lazar", "1989") while semantic search
        captures meaning. The weighted combination improves retrieval for
        documentary research where both matter.

        Args:
            query: Search query
            top_k: Number of results to return
            where: ChromaDB metadata filter
            semantic_weight: Weight for vector search scores (default 0.7)
            keyword_weight: Weight for BM25 scores (default 0.3)
        """
        k = top_k or settings.retrieval_top_k

        # Semantic search
        semantic_hits = self.search(query, top_k=k * 2, where=where)

        # BM25 keyword search
        if self._bm25_index is None:
            self._build_bm25_index()

        if not self._bm25_corpus:
            return semantic_hits[:k]

        query_tokens = query.lower().split()
        bm25_scores = self._bm25_index.get_scores(query_tokens)

        # Normalize BM25 scores to [0, 1]
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        bm25_by_id = {}
        for i, score in enumerate(bm25_scores):
            doc_id = self._bm25_corpus[i]["id"]
            bm25_by_id[doc_id] = score / max_bm25

        # Merge scores
        merged = {}
        for hit in semantic_hits:
            doc_id = hit["id"]
            semantic_score = hit["relevance_score"]
            bm25_score = bm25_by_id.get(doc_id, 0.0)
            hit["bm25_score"] = bm25_score
            hit["hybrid_score"] = (semantic_score * semantic_weight) + (bm25_score * keyword_weight)
            merged[doc_id] = hit

        # Add BM25-only hits not in semantic results
        for i, score in enumerate(bm25_scores):
            doc_id = self._bm25_corpus[i]["id"]
            if doc_id not in merged and score > 0:
                entry = self._bm25_corpus[i]
                merged[doc_id] = {
                    "id": doc_id,
                    "content": entry["content"],
                    "metadata": entry["metadata"],
                    "distance": 1.0,
                    "relevance_score": 0.0,
                    "bm25_score": score / max_bm25,
                    "hybrid_score": (score / max_bm25) * keyword_weight,
                }

        # Sort by hybrid score and return top_k
        results = sorted(merged.values(), key=lambda h: h["hybrid_score"], reverse=True)
        return results[:k]

    def invalidate_bm25(self):
        """Force BM25 index rebuild on next hybrid search."""
        self._bm25_index = None
        self._bm25_corpus = None

    def clear(self):
        """Delete all chunks from the collection."""
        self._client.delete_collection(settings.chroma_collection)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        self.invalidate_bm25()
        logger.warning("Vector store cleared")
