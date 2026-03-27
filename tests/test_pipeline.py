"""Integration tests for the retrieval pipeline."""

from pathlib import Path

import pytest

from src.ingestion.chunker import chunk_document
from src.ingestion.loader import load_from_manifest
from src.ingestion.vectorstore import VectorStore
from src.models.documents import DocumentMetadata, IngestedDocument, SourceType
from src.retrieval.memory import ConversationMemory


class TestVectorStoreSearch:
    """Test vector store search with real embeddings (no LLM needed)."""

    @pytest.fixture(autouse=True)
    def setup_store(self, tmp_path):
        """Create a temp vector store with sample data."""
        from config.settings import settings

        # Override vector dir for test isolation
        original_dir = settings.chroma_persist_dir
        settings.chroma_persist_dir = str(tmp_path / "vectors")

        self.store = VectorStore()

        # Create and ingest a small test doc
        doc = IngestedDocument(
            id="test_doc_1",
            content=(
                "Bob Lazar claimed he worked at S4 near Area 51"
                " in 1988. He described reverse-engineering alien"
                " propulsion systems that used Element 115 as fuel."
            ),
            metadata=DocumentMetadata(
                source_file="test_interview.txt",
                source_type=SourceType.INTERVIEW_TRANSCRIPT,
                title="Test Interview",
                reliability_score=0.7,
            ),
        )
        chunks = chunk_document(doc)
        self.store.add_chunks(chunks)

        yield

        settings.chroma_persist_dir = original_dir

    def test_search_returns_results(self):
        results = self.store.search("Element 115", top_k=3)
        assert len(results) > 0

    def test_search_relevance_score(self):
        results = self.store.search("Bob Lazar S4", top_k=3)
        assert all(r["relevance_score"] >= 0 for r in results)

    def test_search_metadata_preserved(self):
        results = self.store.search("alien propulsion", top_k=1)
        assert results[0]["metadata"]["source_type"] == "interview_transcript"
        assert results[0]["metadata"]["source_file"] == "test_interview.txt"

    def test_search_with_filter(self):
        results = self.store.search(
            "Element 115",
            where={"source_type": {"$eq": "interview_transcript"}},
        )
        assert len(results) > 0
        assert all(r["metadata"]["source_type"] == "interview_transcript" for r in results)


class TestConversationMemory:
    def test_empty_memory_returns_none(self):
        mem = ConversationMemory()
        assert mem.get_context_prompt() is None

    def test_enrich_question_no_history(self):
        mem = ConversationMemory()
        assert mem.enrich_question("What is S4?") == "What is S4?"

    def test_enrich_question_with_history(self):
        mem = ConversationMemory()
        mem.add_turn("Who is Bob Lazar?", "Bob Lazar is a physicist who claimed...")
        enriched = mem.enrich_question("What about his education?")
        assert "CONVERSATION HISTORY" in enriched
        assert "Bob Lazar" in enriched
        assert "What about his education?" in enriched

    def test_max_turns_limit(self):
        mem = ConversationMemory(max_turns=2)
        mem.add_turn("Q1", "A1")
        mem.add_turn("Q2", "A2")
        mem.add_turn("Q3", "A3")
        assert mem.turn_count == 2
        assert mem.turns[0].question == "Q2"

    def test_clear(self):
        mem = ConversationMemory()
        mem.add_turn("Q1", "A1")
        mem.clear()
        assert mem.turn_count == 0


class TestHybridSearch:
    """Test BM25 + semantic hybrid search."""

    @pytest.fixture(autouse=True)
    def setup_store(self, tmp_path):
        from config.settings import settings

        original_dir = settings.chroma_persist_dir
        settings.chroma_persist_dir = str(tmp_path / "vectors")

        self.store = VectorStore()

        # Two docs — one with a specific keyword, one semantically related
        doc1 = IngestedDocument(
            id="keyword_doc",
            content=(
                "Element 115 was synthesized at JINR Dubna in 2003"
                " by Oganessian. The isotope had a half-life of"
                " 87 milliseconds."
            ),
            metadata=DocumentMetadata(
                source_file="element115_paper.txt",
                source_type=SourceType.SCIENTIFIC_PAPER,
                title="Element 115 Synthesis",
                reliability_score=0.9,
            ),
        )
        doc2 = IngestedDocument(
            id="semantic_doc",
            content=(
                "The propulsion system used a super-heavy fuel source"
                " that generated its own gravitational field when"
                " bombarded with protons."
            ),
            metadata=DocumentMetadata(
                source_file="lazar_interview.txt",
                source_type=SourceType.INTERVIEW_TRANSCRIPT,
                title="Lazar Interview",
                reliability_score=0.7,
            ),
        )
        for doc in [doc1, doc2]:
            chunks = chunk_document(doc)
            self.store.add_chunks(chunks)

        yield
        settings.chroma_persist_dir = original_dir

    def test_hybrid_search_returns_results(self):
        results = self.store.hybrid_search("Element 115", top_k=5)
        assert len(results) > 0

    def test_hybrid_search_has_scores(self):
        results = self.store.hybrid_search("Element 115 Oganessian", top_k=5)
        for r in results:
            assert "hybrid_score" in r
            assert "bm25_score" in r

    def test_keyword_match_boosted(self):
        """BM25 should boost exact keyword matches like 'Oganessian'."""
        results = self.store.hybrid_search("Oganessian JINR Dubna", top_k=5)
        # The paper doc should rank first because it has exact keyword matches
        assert results[0]["metadata"]["source_file"] == "element115_paper.txt"


class TestManifestIngestion:
    def test_load_manifest(self):
        manifest_path = Path("data/raw/manifest.json")
        if manifest_path.exists():
            docs = load_from_manifest(manifest_path)
            assert len(docs) == 6
            assert all(doc.content for doc in docs)
            assert all(doc.metadata.source_file for doc in docs)

    def test_chunk_preserves_metadata(self):
        doc = IngestedDocument(
            id="meta_test",
            content="Test content about S4 and Area 51. " * 50,
            metadata=DocumentMetadata(
                source_file="test.txt",
                source_type=SourceType.GOVERNMENT_DOCUMENT,
                title="Test Doc",
                author="DOE",
                reliability_score=0.95,
            ),
        )
        chunks = chunk_document(doc)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata.source_type == SourceType.GOVERNMENT_DOCUMENT
            assert chunk.metadata.reliability_score == 0.95
            assert chunk.metadata.author == "DOE"
