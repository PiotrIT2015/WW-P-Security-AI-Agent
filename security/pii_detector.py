"""
security/pii_detector.py

Personally Identifiable Information (PII) detector
for Secure RAG.

Python 3.12+
"""

from __future__ import annotations

import hashlib
import re

from dataclasses import dataclass
from enum import Enum


# =============================================================================
# Types
# =============================================================================

class PIIType(Enum):

    EMAIL = "email"

    PHONE = "phone"

    PESEL = "pesel"

    IBAN = "iban"

    IP = "ip"

    CREDIT_CARD = "credit_card"


# =============================================================================
# Entity
# =============================================================================

@dataclass(slots=True)
class PIIEntity:

    pii_type: PIIType

    value: str

    start: int

    end: int


# =============================================================================
# Report
# =============================================================================

@dataclass(slots=True)
class PIIReport:

    entities: list[PIIEntity]

    redacted_text: str

    count: int


# =============================================================================
# Detector
# =============================================================================

class PIIDetector:

    """
    Detects and anonymizes PII.
    """

    PATTERNS = {

        PIIType.EMAIL:
            re.compile(
                r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
            ),

        PIIType.PHONE:
            re.compile(
                r"(?:\+48\s?)?(?:\d{3}[\s-]?){2}\d{3}"
            ),

        PIIType.PESEL:
            re.compile(
                r"\b\d{11}\b"
            ),

        PIIType.IP:
            re.compile(
                r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
            ),

        PIIType.IBAN:
            re.compile(
                r"\bPL\d{26}\b"
            ),

        PIIType.CREDIT_CARD:
            re.compile(
                r"\b(?:\d{4}[- ]?){3}\d{4}\b"
            ),
    }

    # ---------------------------------------------------------

    def detect(
        self,
        text: str,
    ) -> list[PIIEntity]:

        entities = []

        for pii_type, pattern in self.PATTERNS.items():

            for match in pattern.finditer(text):

                entities.append(

                    PIIEntity(

                        pii_type=pii_type,

                        value=match.group(),

                        start=match.start(),

                        end=match.end(),

                    )

                )

        entities.sort(
            key=lambda e: e.start
        )

        return entities

    # ---------------------------------------------------------

    def redact(
        self,
        text: str,
    ) -> PIIReport:

        entities = self.detect(text)

        redacted = text

        for entity in reversed(entities):

            replacement = f"<{entity.pii_type.value.upper()}>"

            redacted = (

                redacted[:entity.start]

                + replacement

                + redacted[entity.end:]

            )

        return PIIReport(

            entities=entities,

            redacted_text=redacted,

            count=len(entities),

        )

    # ---------------------------------------------------------

    def hash_entity(
        self,
        value: str,
    ) -> str:
        """
        SHA256 hash for audit logs.
        """

        return hashlib.sha256(

            value.encode()

        ).hexdigest()


# =============================================================================
# Global instance
# =============================================================================

pii_detector = PIIDetector()