"""
RAG evaluation pipeline using RAGAS metrics.

This demonstrates production-grade quality assurance:
- Faithfulness: Does the answer stick to the provided context?
- Answer relevancy: Does the answer address the question?
- Context precision: Are the retrieved chunks actually relevant?
- Context recall: Did we retrieve all necessary chunks?

Run with: s4ri evaluate --test-set data/evaluation/test_queries.json
"""

import json
from pathlib import Path

from loguru import logger

from src.models.queries import ResearchQuery
from src.retrieval.pipeline import ResearchPipeline


class RAGEvaluator:
    """Evaluate RAG pipeline quality using test queries with ground truth."""

    def __init__(self, pipeline: ResearchPipeline | None = None):
        self.pipeline = pipeline or ResearchPipeline()

    def load_test_set(self, path: Path) -> list[dict]:
        """
        Load evaluation test set. Each entry:
        {
            "question": "What year did Lazar claim to start at S4?",
            "ground_truth": "Lazar claimed he began working at S4 in late 1988.",
            "expected_sources": ["lazar_testimony_1989.txt"],
            "expected_date_mentions": ["1988", "1989"]
        }
        """
        return json.loads(path.read_text())

    def evaluate_single(self, test_case: dict) -> dict:
        """Run a single evaluation query and score against ground truth."""
        query = ResearchQuery(question=test_case["question"])
        response = self.pipeline.query(query)

        # Source coverage: did we retrieve the expected sources?
        retrieved_files = {s.source_file for s in response.sources}
        expected_files = set(test_case.get("expected_sources", []))
        source_recall = (
            len(retrieved_files & expected_files) / len(expected_files)
            if expected_files
            else 1.0
        )

        # Date mention coverage
        expected_dates = test_case.get("expected_date_mentions", [])
        date_hits = sum(1 for d in expected_dates if d in response.answer)
        date_coverage = date_hits / len(expected_dates) if expected_dates else 1.0

        # Citation check: does the answer contain [Source: ...] references?
        has_citations = "[Source:" in response.answer or "source" in response.answer.lower()

        # Contradiction flag check
        expected_contradictions = test_case.get("expect_contradictions", False)
        contradiction_correct = (len(response.contradictions) > 0) == expected_contradictions

        return {
            "question": test_case["question"],
            "answer": response.answer[:500],
            "confidence": response.confidence,
            "source_recall": source_recall,
            "date_coverage": date_coverage,
            "has_citations": has_citations,
            "contradiction_detection_correct": contradiction_correct,
            "num_sources": len(response.sources),
            "num_contradictions": len(response.contradictions),
            "num_timeline_events": len(response.timeline),
        }

    def evaluate_batch(self, test_set_path: Path) -> dict:
        """Run full evaluation suite and return aggregate metrics."""
        test_cases = self.load_test_set(test_set_path)
        results = []

        for case in test_cases:
            logger.info(f"Evaluating: {case['question'][:80]}...")
            result = self.evaluate_single(case)
            results.append(result)

        # Aggregate
        n = len(results)
        aggregate = {
            "total_queries": n,
            "avg_confidence": sum(r["confidence"] for r in results) / n,
            "avg_source_recall": sum(r["source_recall"] for r in results) / n,
            "avg_date_coverage": sum(r["date_coverage"] for r in results) / n,
            "citation_rate": sum(r["has_citations"] for r in results) / n,
            "contradiction_accuracy": sum(
                r["contradiction_detection_correct"] for r in results
            ) / n,
            "individual_results": results,
        }

        logger.info(
            f"Evaluation complete: {n} queries | "
            f"avg_confidence={aggregate['avg_confidence']:.2f} | "
            f"source_recall={aggregate['avg_source_recall']:.2f} | "
            f"citation_rate={aggregate['citation_rate']:.0%}"
        )

        return aggregate
