"""Extract discrete factual claims from LLM responses for verification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import nltk

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import sent_tokenize


class ClaimStatus(str, Enum):
    PENDING = "pending"
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


@dataclass
class Claim:
    """A single testable factual assertion extracted from an LLM response."""

    text: str
    source_sentence: str
    status: ClaimStatus = ClaimStatus.PENDING
    similarity_score: float = 0.0
    matching_source: str | None = None
    matching_excerpt: str | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source_sentence": self.source_sentence,
            "status": self.status.value,
            "similarity_score": self.similarity_score,
            "matching_source": self.matching_source,
            "matching_excerpt": self.matching_excerpt,
        }


# Patterns that indicate non-factual content (hedging, meta-commentary, etc.)
_SKIP_PATTERNS = [
    r"^(however|therefore|in summary|in conclusion|overall|note that"
    r"|it\s*'?s? (important|worth|notable))",
    r"^(I |my |we |our |let me|as an? )",
    r"^(based on|according to) (the|my|our) (analysis|review|findings)",
    r"^(yes|no|sure|okay)[,.]?\s*$",
]

_SKIP_RE = re.compile("|".join(_SKIP_PATTERNS), re.IGNORECASE)


class ClaimExtractor:
    """Splits an LLM response into discrete, verifiable factual claims."""

    def __init__(self, min_claim_words: int = 4, max_claim_words: int = 60):
        self.min_claim_words = min_claim_words
        self.max_claim_words = max_claim_words

    def extract(self, response: str | None) -> list[Claim]:
        """Extract factual claims from a response string.

        Pipeline:
        1. Sentence-tokenize the response
        2. Filter out non-factual sentences (hedging, meta-commentary)
        3. Split compound sentences on conjunctions where appropriate
        4. Return a list of Claim objects ready for source matching
        """
        if not response or not response.strip():
            return []

        sentences = sent_tokenize(response)
        claims: list[Claim] = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Skip non-factual content
            if _SKIP_RE.match(sentence):
                continue

            # Split compound claims on " and " when both halves are substantial
            sub_claims = self._split_compound(sentence)

            for claim_text in sub_claims:
                claim_text = claim_text.strip().rstrip(".")
                words = claim_text.split()

                if len(words) < self.min_claim_words:
                    continue
                if len(words) > self.max_claim_words:
                    # Truncate extremely long claims
                    claim_text = " ".join(words[: self.max_claim_words])

                claims.append(Claim(text=claim_text, source_sentence=sentence))

        return claims

    def _split_compound(self, sentence: str) -> list[str]:
        """Split compound sentences into individual claims.

        Only splits on ' and ' or '; ' when both resulting parts
        contain a subject-verb structure (heuristic: both > 4 words).
        """
        # Try splitting on semicolons first
        if "; " in sentence:
            parts = sentence.split("; ")
            if all(len(p.split()) >= self.min_claim_words for p in parts):
                return parts

        # Try splitting on ' and ' for compound claims
        if " and " in sentence:
            parts = sentence.split(" and ", 1)
            if all(len(p.split()) >= self.min_claim_words for p in parts):
                return parts

        return [sentence]
