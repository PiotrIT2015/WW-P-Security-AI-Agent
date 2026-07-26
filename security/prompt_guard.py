"""
security/prompt_guard.py

Secure RAG
Prompt Security Engine

Stage 2.4A

Python 3.12+
"""

from __future__ import annotations

import re
import unicodedata

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


# =============================================================================
# Decision
# =============================================================================


class Decision(Enum):
    """
    Final security decision.
    """

    ALLOW = "allow"

    REVIEW = "review"

    BLOCK = "block"


# =============================================================================
# Report
# =============================================================================


@dataclass(slots=True)
class SecurityReport:
    """
    Analysis result.
    """

    decision: Decision

    risk_score: int

    reasons: list[str] = field(default_factory=list)

    matched_rules: list[str] = field(default_factory=list)

    normalized_text: str = ""


# =============================================================================
# Prompt Guard
# =============================================================================


class PromptGuard:

    """
    Prompt security engine.

    This class performs:

    - normalization

    - rule matching

    - risk scoring

    - security decision
    """

    # ------------------------------------------------------

    BLOCK_PATTERNS = {

        "ignore previous instructions": 60,

        "ignore all previous": 60,

        "forget previous instructions": 60,

        "developer message": 50,

        "system prompt": 60,

        "reveal your prompt": 60,

        "show hidden instructions": 60,

        "act as root": 50,

        "bypass restrictions": 60,

        "disable safety": 70,

    }

    # ------------------------------------------------------

    REVIEW_PATTERNS = {

        "simulate": 15,

        "pretend": 10,

        "override": 20,

        "admin": 15,

        "sudo": 15,

        "confidential": 10,

    }

    # ------------------------------------------------------

    def normalize(
        self,
        text: str,
    ) -> str:
        """
        Normalize text before analysis.
        """

        text = unicodedata.normalize(
            "NFKC",
            text,
        )

        text = text.lower()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # ------------------------------------------------------

    def _match(
        self,
        text: str,
        rules: dict[str, int],
    ) -> tuple[int, list[str]]:

        score = 0

        matched = []

        for pattern, value in rules.items():

            if pattern in text:

                score += value

                matched.append(pattern)

        return score, matched

    # ------------------------------------------------------

    def analyze(
        self,
        text: str,
    ) -> SecurityReport:

        normalized = self.normalize(text)

        score = 0

        reasons: list[str] = []

        matched: list[str] = []

        block_score, block_rules = self._match(
            normalized,
            self.BLOCK_PATTERNS,
        )

        review_score, review_rules = self._match(
            normalized,
            self.REVIEW_PATTERNS,
        )

        score += block_score
        score += review_score

        matched.extend(block_rules)
        matched.extend(review_rules)

        if block_rules:
            reasons.append(
                "Prompt injection indicators detected."
            )

        if review_rules:
            reasons.append(
                "Potentially suspicious instructions detected."
            )

        score = min(score, 100)

        if score >= 70:

            decision = Decision.BLOCK

        elif score >= 30:

            decision = Decision.REVIEW

        else:

            decision = Decision.ALLOW

        return SecurityReport(

            decision=decision,

            risk_score=score,

            reasons=reasons,

            matched_rules=matched,

            normalized_text=normalized,

        )

    # ------------------------------------------------------

    def is_allowed(
        self,
        text: str,
    ) -> bool:

        report = self.analyze(text)

        return report.decision is Decision.ALLOW


# =============================================================================
# Global instance
# =============================================================================

prompt_guard = PromptGuard()