"""Tests for claim extraction from LLM responses."""

from src.evaluation.hallucination.claim_extractor import Claim, ClaimExtractor, ClaimStatus


class TestClaimExtractor:
    def setup_method(self):
        self.extractor = ClaimExtractor()

    def test_extract_single_sentence(self):
        response = "Bob Lazar claimed he worked at a facility called S4 near Area 51."
        claims = self.extractor.extract(response)
        assert len(claims) >= 1
        assert claims[0].status == ClaimStatus.PENDING

    def test_extract_multiple_sentences(self):
        response = (
            "Lazar described working with Element 115 as a fuel source. "
            "He claimed the element powered gravity amplifiers on alien craft. "
            "The facility was located near Papoose Lake."
        )
        claims = self.extractor.extract(response)
        assert len(claims) >= 3

    def test_skip_hedging_language(self):
        response = (
            "However, this remains unconfirmed. "
            "I think it's worth noting the historical context. "
            "Lazar stated he saw nine craft at S4."
        )
        claims = self.extractor.extract(response)
        # Should skip hedging and meta-commentary, keep the factual claim
        factual = [c for c in claims if "nine craft" in c.text.lower()]
        assert len(factual) >= 1

    def test_empty_response(self):
        claims = self.extractor.extract("")
        assert claims == []

    def test_none_response(self):
        claims = self.extractor.extract(None)
        assert claims == []

    def test_short_sentences_filtered(self):
        response = "Yes. No. Lazar claimed to work at S4 in late 1988."
        claims = self.extractor.extract(response)
        # "Yes" and "No" should be filtered out
        assert all("yes" not in c.text.lower() for c in claims)

    def test_compound_sentence_splitting(self):
        response = "Lazar worked at S4 in 1988 and he described seeing nine different craft."
        claims = self.extractor.extract(response)
        # May be split into two claims
        assert len(claims) >= 1

    def test_claim_has_source_sentence(self):
        response = "Element 115 was described as a stable isotope used for gravity propulsion."
        claims = self.extractor.extract(response)
        assert len(claims) >= 1
        assert claims[0].source_sentence == response


class TestClaim:
    def test_to_dict(self):
        claim = Claim(
            text="Lazar worked at S4",
            source_sentence="Lazar worked at S4 in 1988.",
            status=ClaimStatus.SUPPORTED,
            similarity_score=0.85,
            matching_source="interview.txt",
            matching_excerpt="He began working there...",
        )
        d = claim.to_dict()
        assert d["text"] == "Lazar worked at S4"
        assert d["status"] == "supported"
        assert d["similarity_score"] == 0.85
        assert d["matching_source"] == "interview.txt"
