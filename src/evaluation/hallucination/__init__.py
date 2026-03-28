"""Hallucination detection — claim extraction, source matching, and scoring."""

from src.evaluation.hallucination.claim_extractor import Claim, ClaimExtractor
from src.evaluation.hallucination.detector import HallucinationDetector, HallucinationReport
from src.evaluation.hallucination.scorer import HallucinationScorer

__all__ = [
    "Claim",
    "ClaimExtractor",
    "HallucinationDetector",
    "HallucinationReport",
    "HallucinationScorer",
]
