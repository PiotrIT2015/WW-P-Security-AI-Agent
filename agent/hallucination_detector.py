"""
agent/hallucination_detector.py

Hallucination assessment component for WW-P Security AI Agent.

Responsibilities:
- Split generated answers into individual claims.
- Compare claims against authorized RAG context.
- Identify claims without sufficient lexical evidence.
- Detect insufficient reference context.
- Produce structured evaluation reports.

Important:
This component uses deterministic lexical matching.
It is not a semantic entailment model and cannot guarantee
detection of every hallucination or contradiction.

Compatible with Python 3.10+.
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Any


# ============================================================
# Constants
# ============================================================

DEFAULT_MINIMUM_OVERLAP = 0.35
DEFAULT_MINIMUM_MATCHED_TERMS = 2

TOKEN_PATTERN = re.compile(
    r"[a-z0-9]+(?:[._-][a-z0-9]+)*",
    re.IGNORECASE,
)

SENTENCE_PATTERN = re.compile(
    r"(?<=[.!?])\s+|\n+"
)

STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at",
    "be", "been", "being", "but", "by",
    "can", "could",
    "did", "do", "does",
    "for", "from",
    "had", "has", "have", "he", "her",
    "here", "hers", "him", "his", "how",
    "i", "if", "in", "into", "is", "it",
    "its",
    "may", "might", "more", "most", "must",
    "no", "not",
    "of", "on", "or", "our", "ours",
    "she", "should", "so", "some",
    "such",
    "than", "that", "the", "their", "theirs",
    "them", "then", "there", "these", "they",
    "this", "those", "through", "to",
    "up",
    "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "will",
    "with", "would",
    "you", "your", "yours",
})


# ============================================================
# Data Models
# ============================================================

@dataclass(frozen=True)
class ClaimAssessment:
    """
    Assessment of one generated claim.

    status:
        supported
        unsupported
        insufficient_context
    """

    claim: str
    status: str
    overlap_ratio: float
    matched_terms: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:

        return {
            "claim": self.claim,
            "status": self.status,
            "overlap_ratio": self.overlap_ratio,
            "matched_terms": list(self.matched_terms),
        }


@dataclass(frozen=True)
class HallucinationReport:
    """
    Structured response-grounding evaluation report.

    status:
        grounded
        contains_unsupported_claims
        insufficient_context
        no_claims
    """

    status: str
    claims: tuple[ClaimAssessment, ...]
    supported_count: int
    unsupported_count: int
    unverified_count: int
    coverage: float

    @property
    def is_grounded(self) -> bool:

        return (
            self.status == "grounded"
            and self.unsupported_count == 0
            and self.unverified_count == 0
        )

    def to_dict(self) -> dict[str, Any]:

        return {
            "status": self.status,
            "is_grounded": self.is_grounded,
            "supported_count": self.supported_count,
            "unsupported_count": self.unsupported_count,
            "unverified_count": self.unverified_count,
            "coverage": self.coverage,
            "claims": [
                claim.to_dict()
                for claim in self.claims
            ],
        }


# ============================================================
# Hallucination Detector
# ============================================================

class HallucinationDetector:
    """
    Evaluates whether generated claims have lexical support
    in the supplied reference context.

    The detector does not access FAISS, the vector store,
    or unauthorized documents.

    It only evaluates the context explicitly provided
    by the caller.
    """

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    INSUFFICIENT_CONTEXT = "insufficient_context"

    GROUNDED = "grounded"
    CONTAINS_UNSUPPORTED = "contains_unsupported_claims"
    NO_CONTEXT = "insufficient_context"
    NO_CLAIMS = "no_claims"

    def __init__(
        self,
        minimum_overlap: float = DEFAULT_MINIMUM_OVERLAP,
        minimum_matched_terms: int = DEFAULT_MINIMUM_MATCHED_TERMS,
    ) -> None:

        if (
            isinstance(minimum_overlap, bool)
            or not isinstance(minimum_overlap, (int, float))
            or not 0.0 <= minimum_overlap <= 1.0
        ):
            raise ValueError(
                "minimum_overlap must be between 0 and 1."
            )

        if (
            isinstance(minimum_matched_terms, bool)
            or not isinstance(minimum_matched_terms, int)
            or minimum_matched_terms < 1
        ):
            raise ValueError(
                "minimum_matched_terms must be a positive integer."
            )

        self.minimum_overlap = float(minimum_overlap)

        self.minimum_matched_terms = minimum_matched_terms

    # ========================================================
    # Tokenization
    # ========================================================

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """
        Extract meaningful normalized terms.

        Stop words are removed.
        """

        tokens = {
            token.lower()
            for token in TOKEN_PATTERN.findall(text)
        }

        return {
            token
            for token in tokens
            if token not in STOP_WORDS
        }

    # ========================================================
    # Claim Extraction
    # ========================================================

    @staticmethod
    def extract_claims(
        response: str,
    ) -> tuple[str, ...]:
        """
        Split a response into candidate claims.

        This is sentence extraction, not advanced NLP
        claim recognition.
        """

        if not isinstance(response, str):
            raise ValueError(
                "response must be a string."
            )

        if not response.strip():
            return ()

        sentences = SENTENCE_PATTERN.split(
            response.strip()
        )

        claims = []

        for sentence in sentences:

            cleaned = sentence.strip()

            if not cleaned:
                continue

            # Ignore Markdown headings.
            cleaned = re.sub(
                r"^#{1,6}\s*",
                "",
                cleaned,
            )

            # Ignore empty list markers.
            cleaned = re.sub(
                r"^[-*+]\s+",
                "",
                cleaned,
            )

            if not cleaned:
                continue

            if not HallucinationDetector._tokenize(cleaned):
                continue

            claims.append(cleaned)

        return tuple(claims)

    # ========================================================
    # Single Claim Evaluation
    # ========================================================

    def evaluate_claim(
        self,
        claim: str,
        context: str,
    ) -> ClaimAssessment:
        """
        Evaluate one claim against the supplied context.
        """

        if not isinstance(claim, str) or not claim.strip():
            raise ValueError(
                "claim must be a non-empty string."
            )

        if not isinstance(context, str):
            raise ValueError(
                "context must be a string."
            )

        if not context.strip():

            return ClaimAssessment(
                claim=claim.strip(),
                status=self.INSUFFICIENT_CONTEXT,
                overlap_ratio=0.0,
                matched_terms=(),
            )

        claim_terms = self._tokenize(claim)

        context_terms = self._tokenize(context)

        if not claim_terms:

            return ClaimAssessment(
                claim=claim.strip(),
                status=self.UNSUPPORTED,
                overlap_ratio=0.0,
                matched_terms=(),
            )

        matched_terms = claim_terms.intersection(
            context_terms
        )

        overlap_ratio = (
            len(matched_terms) / len(claim_terms)
        )

        supported = (
            len(matched_terms) >= self.minimum_matched_terms
            and overlap_ratio >= self.minimum_overlap
        )

        return ClaimAssessment(
            claim=claim.strip(),
            status=(
                self.SUPPORTED
                if supported
                else self.UNSUPPORTED
            ),
            overlap_ratio=round(overlap_ratio, 4),
            matched_terms=tuple(sorted(matched_terms)),
        )

    # ========================================================
    # Full Response Evaluation
    # ========================================================

    def evaluate(
        self,
        response: str,
        context: str,
    ) -> HallucinationReport:
        """
        Evaluate all candidate claims in a response.

        The evaluation uses only the supplied context.
        """

        if not isinstance(response, str):
            raise ValueError(
                "response must be a string."
            )

        if not isinstance(context, str):
            raise ValueError(
                "context must be a string."
            )

        claims = self.extract_claims(response)

        if not claims:

            return HallucinationReport(
                status=self.NO_CLAIMS,
                claims=(),
                supported_count=0,
                unsupported_count=0,
                unverified_count=0,
                coverage=0.0,
            )

        # ----------------------------------------------------
        # Insufficient context
        # ----------------------------------------------------

        if not context.strip():

            assessments = tuple(
                ClaimAssessment(
                    claim=claim,
                    status=self.INSUFFICIENT_CONTEXT,
                    overlap_ratio=0.0,
                    matched_terms=(),
                )
                for claim in claims
            )

            return HallucinationReport(
                status=self.NO_CONTEXT,
                claims=assessments,
                supported_count=0,
                unsupported_count=0,
                unverified_count=len(assessments),
                coverage=0.0,
            )

        # ----------------------------------------------------
        # Evaluate claims
        # ----------------------------------------------------

        assessments = tuple(
            self.evaluate_claim(
                claim=claim,
                context=context,
            )
            for claim in claims
        )

        supported_count = sum(
            claim.status == self.SUPPORTED
            for claim in assessments
        )

        unsupported_count = sum(
            claim.status == self.UNSUPPORTED
            for claim in assessments
        )

        unverified_count = sum(
            claim.status == self.INSUFFICIENT_CONTEXT
            for claim in assessments
        )

        coverage = (
            supported_count / len(assessments)
        )

        status = (
            self.GROUNDED
            if unsupported_count == 0
            and unverified_count == 0
            else self.CONTAINS_UNSUPPORTED
        )

        return HallucinationReport(
            status=status,
            claims=assessments,
            supported_count=supported_count,
            unsupported_count=unsupported_count,
            unverified_count=unverified_count,
            coverage=round(coverage, 4),
        )