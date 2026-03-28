"""Tests for hallucination detection and scoring."""

from unittest.mock import MagicMock

from src.evaluation.hallucination.claim_extractor import Claim, ClaimStatus
from src.evaluation.hallucination.detector import HallucinationDetector, HallucinationReport
from src.evaluation.hallucination.scorer import AggregateHallucinationScore, HallucinationScorer


class TestHallucinationDetector:
    def test_analyze_empty_response(self):
        detector = HallucinationDetector.__new__(HallucinationDetector)
        detector.extractor = MagicMock()
        detector.extractor.extract.return_value = []
        detector.support_threshold = 0.75
        detector.contradiction_threshold = 0.60

        report = detector.analyze("test query", "")
        assert report.grounding_score == 1.0
        assert len(report.claims) == 0

    def test_report_to_dict(self):
        report = HallucinationReport(
            query="test",
            response="test response",
            claims=[],
            hallucination_rate=0.25,
            fabrication_rate=0.05,
            grounding_score=0.70,
        )
        d = report.to_dict()
        assert d["hallucination_rate"] == 0.25
        assert d["fabrication_rate"] == 0.05
        assert d["grounding_score"] == 0.70
        assert d["total_claims"] == 0

    def test_is_negation_detection(self):
        detector = HallucinationDetector.__new__(HallucinationDetector)

        # Claim positive, source negative = contradiction
        assert detector._is_negation(
            "Lazar worked at S4", "There is no evidence that anyone worked at S4"
        )

        # Both positive = not contradiction
        assert not detector._is_negation(
            "Lazar worked at S4", "Lazar was employed at the S4 facility"
        )


class TestHallucinationScorer:
    def test_score_empty_batch(self):
        scorer = HallucinationScorer.__new__(HallucinationScorer)
        scorer.history_dir = MagicMock()
        result = scorer.score_batch([])
        assert result.total_queries == 0

    def test_score_batch(self):
        reports = [
            HallucinationReport(
                query="q1",
                response="r1",
                claims=[Claim(text="c1", source_sentence="s1", status=ClaimStatus.SUPPORTED)],
                supported_claims=[Claim(text="c1", source_sentence="s1")],
                unsupported_claims=[],
                contradicted_claims=[],
                hallucination_rate=0.0,
                fabrication_rate=0.0,
                grounding_score=1.0,
            ),
            HallucinationReport(
                query="q2",
                response="r2",
                claims=[
                    Claim(text="c2", source_sentence="s2", status=ClaimStatus.SUPPORTED),
                    Claim(text="c3", source_sentence="s3", status=ClaimStatus.UNSUPPORTED),
                ],
                supported_claims=[Claim(text="c2", source_sentence="s2")],
                unsupported_claims=[Claim(text="c3", source_sentence="s3")],
                contradicted_claims=[],
                hallucination_rate=0.5,
                fabrication_rate=0.0,
                grounding_score=0.5,
            ),
        ]

        scorer = HallucinationScorer.__new__(HallucinationScorer)
        scorer.history_dir = MagicMock()
        result = scorer.score_batch(reports)

        assert result.total_queries == 2
        assert result.total_claims == 3
        assert result.total_supported == 2
        assert result.total_unsupported == 1
        assert result.avg_grounding_score == 0.75
        assert result.avg_hallucination_rate == 0.25

    def test_aggregate_to_dict(self):
        score = AggregateHallucinationScore(
            total_queries=10,
            total_claims=50,
            avg_grounding_score=0.82,
        )
        d = score.to_dict()
        assert d["total_queries"] == 10
        assert d["avg_grounding_score"] == 0.82
