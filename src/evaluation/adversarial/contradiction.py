"""Contradiction injection — test if the system detects planted contradictions."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from src.models.queries import ResearchQuery
from src.retrieval.pipeline import ResearchPipeline


@dataclass
class ContradictionResult:
    """Result of a contradiction injection test."""

    test_id: str
    query: str
    injected_claim: str
    actual_claim: str
    detected: bool
    response_excerpt: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "query": self.query,
            "injected_claim": self.injected_claim,
            "actual_claim": self.actual_claim,
            "detected": self.detected,
            "response_excerpt": self.response_excerpt[:300],
            "confidence": self.confidence,
        }


class ContradictionInjector:
    """Tests whether the system detects contradictions injected into queries.

    The injector embeds a false premise into a query and checks whether
    the system corrects or flags the contradiction rather than accepting it.
    """

    def __init__(self, pipeline: ResearchPipeline | None = None):
        self.pipeline = pipeline or ResearchPipeline()

    def test(self, test_case: dict) -> ContradictionResult:
        """Run a single contradiction injection test.

        Args:
            test_case: Dict with keys: id, query, injected_claim, actual_claim

        Returns:
            ContradictionResult indicating whether the system detected the false premise.
        """
        query_text = test_case["query"]
        injected = test_case.get("injected_claim", "")
        actual = test_case.get("actual_claim", "")

        try:
            query = ResearchQuery(question=query_text)
            response = self.pipeline.query(query)
            answer = response.answer.lower()

            # Detection heuristics: did the system push back on the false premise?
            detected = self._check_detection(answer, injected, actual)

            return ContradictionResult(
                test_id=test_case.get("id", "unknown"),
                query=query_text,
                injected_claim=injected,
                actual_claim=actual,
                detected=detected,
                response_excerpt=response.answer[:300],
                confidence=response.confidence,
            )
        except Exception as e:
            logger.error(f"Contradiction test failed for {test_case.get('id')}: {e}")
            return ContradictionResult(
                test_id=test_case.get("id", "unknown"),
                query=query_text,
                injected_claim=injected,
                actual_claim=actual,
                detected=False,
                response_excerpt=f"ERROR: {e}",
                confidence=0.0,
            )

    def _check_detection(self, answer: str, injected: str, actual: str) -> bool:
        """Heuristically check if the system detected the contradiction."""
        correction_signals = [
            "however",
            "actually",
            "incorrect",
            "not accurate",
            "contrary to",
            "the claim that",
            "no evidence",
            "this is not supported",
            "contradicts",
            "differs from",
            "in fact",
            "correction",
            "inaccurate",
            "mistaken",
        ]

        # Check if the answer pushes back on the injected claim
        has_correction = any(signal in answer for signal in correction_signals)

        # Check if the answer includes the actual/correct information
        actual_words = actual.lower().split()
        key_words = [w for w in actual_words if len(w) > 4]
        has_actual_info = (
            key_words and sum(1 for w in key_words if w in answer) >= len(key_words) * 0.3
        )

        return has_correction or has_actual_info
