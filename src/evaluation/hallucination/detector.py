"""Hallucination detection — match extracted claims against source material."""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from src.evaluation.hallucination.claim_extractor import Claim, ClaimExtractor, ClaimStatus


@dataclass
class HallucinationReport:
    """Full hallucination analysis for a single query-response pair."""

    query: str
    response: str
    claims: list[Claim] = field(default_factory=list)
    supported_claims: list[Claim] = field(default_factory=list)
    unsupported_claims: list[Claim] = field(default_factory=list)
    contradicted_claims: list[Claim] = field(default_factory=list)
    hallucination_rate: float = 0.0
    fabrication_rate: float = 0.0
    grounding_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "response": self.response[:500],
            "total_claims": len(self.claims),
            "supported": len(self.supported_claims),
            "unsupported": len(self.unsupported_claims),
            "contradicted": len(self.contradicted_claims),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "fabrication_rate": round(self.fabrication_rate, 4),
            "grounding_score": round(self.grounding_score, 4),
            "claims": [c.to_dict() for c in self.claims],
        }


class HallucinationDetector:
    """Detect hallucinations by matching claims against the vector store.

    Three-phase pipeline:
    1. Extract claims from LLM response (ClaimExtractor)
    2. Search vector store for supporting evidence per claim
    3. Score claims as supported / unsupported / contradicted
    """

    def __init__(
        self,
        vector_store=None,
        support_threshold: float = 0.75,
        contradiction_threshold: float = 0.60,
    ):
        """Initialize the detector.

        Args:
            vector_store: VectorStore instance for source matching.
                If None, will be initialized lazily.
            support_threshold: Minimum cosine similarity to consider
                a claim supported by a source.
            contradiction_threshold: Minimum similarity to a negating
                source to flag as contradicted.
        """
        self._vector_store = vector_store
        self.support_threshold = support_threshold
        self.contradiction_threshold = contradiction_threshold
        self.extractor = ClaimExtractor()

    @property
    def vector_store(self):
        if self._vector_store is None:
            from src.ingestion.vectorstore import VectorStore

            self._vector_store = VectorStore()
        return self._vector_store

    def analyze(self, query: str, response: str) -> HallucinationReport:
        """Run full hallucination analysis on a query-response pair.

        Returns a HallucinationReport with per-claim status and aggregate scores.
        """
        claims = self.extractor.extract(response)
        if not claims:
            return HallucinationReport(
                query=query,
                response=response,
                grounding_score=1.0,
            )

        supported = []
        unsupported = []
        contradicted = []

        for claim in claims:
            self._match_claim(claim)

            if claim.status == ClaimStatus.SUPPORTED:
                supported.append(claim)
            elif claim.status == ClaimStatus.CONTRADICTED:
                contradicted.append(claim)
            else:
                unsupported.append(claim)

        total = len(claims)
        report = HallucinationReport(
            query=query,
            response=response,
            claims=claims,
            supported_claims=supported,
            unsupported_claims=unsupported,
            contradicted_claims=contradicted,
            hallucination_rate=len(unsupported) / total if total else 0.0,
            fabrication_rate=len(contradicted) / total if total else 0.0,
            grounding_score=len(supported) / total if total else 0.0,
        )

        logger.debug(
            f"Hallucination analysis: {total} claims, "
            f"{len(supported)} supported, {len(unsupported)} unsupported, "
            f"{len(contradicted)} contradicted"
        )
        return report

    def _match_claim(self, claim: Claim) -> None:
        """Search vector store for evidence supporting or contradicting a claim."""
        try:
            results = self.vector_store.search(claim.text, top_k=3)
        except Exception as e:
            logger.warning(f"Vector store search failed for claim: {e}")
            claim.status = ClaimStatus.UNSUPPORTED
            return

        if not results:
            claim.status = ClaimStatus.UNSUPPORTED
            return

        best = results[0]
        score = best.get("relevance_score", 0.0)
        content = best.get("content", "")
        source_file = best.get("metadata", {}).get("source_file", "unknown")

        claim.similarity_score = score
        claim.matching_source = source_file
        claim.matching_excerpt = content[:300]

        if score >= self.support_threshold:
            # Check for negation patterns that would indicate contradiction
            if self._is_negation(claim.text, content):
                claim.status = ClaimStatus.CONTRADICTED
            else:
                claim.status = ClaimStatus.SUPPORTED
        elif score >= self.contradiction_threshold:
            # Moderate similarity but check for semantic opposition
            if self._is_negation(claim.text, content):
                claim.status = ClaimStatus.CONTRADICTED
            else:
                claim.status = ClaimStatus.UNSUPPORTED
        else:
            claim.status = ClaimStatus.UNSUPPORTED

    def _is_negation(self, claim: str, source: str) -> bool:
        """Heuristic check for negation between claim and source content.

        Looks for negation patterns that suggest the source contradicts
        rather than supports the claim.
        """
        negation_markers = [
            "not ",
            "never ",
            "no ",
            "denied",
            "false",
            "incorrect",
            "disproven",
            "debunked",
            "no evidence",
            "no record",
            "contradicts",
            "refuted",
            "unsubstantiated",
        ]
        claim_lower = claim.lower()
        source_lower = source.lower()

        # If the claim itself contains negation, source agreement isn't contradictory
        claim_has_negation = any(m in claim_lower for m in negation_markers)
        source_has_negation = any(m in source_lower for m in negation_markers)

        # Contradiction = one has negation but not the other
        return claim_has_negation != source_has_negation
