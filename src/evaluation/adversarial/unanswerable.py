"""Unanswerable query testing — verify the system abstains when no answer exists."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from src.models.queries import ResearchQuery
from src.retrieval.pipeline import ResearchPipeline


@dataclass
class UnanswerableResult:
    """Result of an unanswerable query test."""

    test_id: str
    query: str
    reason: str
    abstained: bool
    response_excerpt: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "query": self.query,
            "reason": self.reason,
            "abstained": self.abstained,
            "response_excerpt": self.response_excerpt[:300],
            "confidence": self.confidence,
        }


class UnanswerableTester:
    """Tests whether the system abstains on queries with no valid answer in the corpus.

    A well-calibrated system should express uncertainty or explicitly state
    that the information is not available rather than fabricating an answer.
    """

    def __init__(
        self,
        pipeline: ResearchPipeline | None = None,
        confidence_threshold: float = 0.4,
    ):
        self.pipeline = pipeline or ResearchPipeline()
        self.confidence_threshold = confidence_threshold

    def test(self, test_case: dict) -> UnanswerableResult:
        """Run a single unanswerable query test.

        Args:
            test_case: Dict with keys: id, query, reason

        Returns:
            UnanswerableResult indicating whether the system correctly abstained.
        """
        query_text = test_case["query"]
        reason = test_case.get("reason", "")

        try:
            query = ResearchQuery(question=query_text)
            response = self.pipeline.query(query)
            answer = response.answer.lower()

            abstained = self._check_abstention(answer, response.confidence)

            return UnanswerableResult(
                test_id=test_case.get("id", "unknown"),
                query=query_text,
                reason=reason,
                abstained=abstained,
                response_excerpt=response.answer[:300],
                confidence=response.confidence,
            )
        except Exception as e:
            logger.error(f"Unanswerable test failed for {test_case.get('id')}: {e}")
            return UnanswerableResult(
                test_id=test_case.get("id", "unknown"),
                query=query_text,
                reason=reason,
                abstained=False,
                response_excerpt=f"ERROR: {e}",
                confidence=0.0,
            )

    def _check_abstention(self, answer: str, confidence: float) -> bool:
        """Check if the system appropriately abstained from answering."""
        abstention_signals = [
            "no information",
            "not found",
            "no evidence",
            "cannot determine",
            "not available",
            "no data",
            "not mentioned",
            "no reference",
            "unclear",
            "insufficient",
            "not discuss",
            "no record",
            "unable to find",
            "does not appear",
            "no mention",
            "i don't have",
            "cannot find",
            "not in the",
        ]

        has_abstention_language = any(signal in answer for signal in abstention_signals)

        # Low confidence is also a sign of appropriate calibration
        low_confidence = confidence < self.confidence_threshold

        return has_abstention_language or low_confidence
