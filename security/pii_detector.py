"""
security/pii_detector.py

WW-P Secure RAG

PII Detection and Masking Module

Stage:
5.3

Python 3.12+
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Pattern
import logging
import re


logger = logging.getLogger(
    "secure_rag.security.pii_detector"
)


# ============================================================
# Exceptions
# ============================================================


class PIIDetectorError(Exception):
    """
    Base PII detector exception.
    """



# ============================================================
# PII Types
# ============================================================


class PIIType(Enum):
    """
    Type of detected sensitive information.
    """

    EMAIL = "email"

    PHONE = "phone"

    PESEL = "pesel"

    IP_ADDRESS = "ip_address"

    CREDIT_CARD = "credit_card"

    API_KEY = "api_key"

    PRIVATE_KEY = "private_key"



# ============================================================
# Rule
# ============================================================


@dataclass(slots=True)
class PIIRule:
    """
    PII detection rule.
    """

    name: str

    pii_type: PIIType

    pattern: str

    replacement: str

    enabled: bool = True

    compiled_pattern: Pattern[str] | None = None



    def compile(
        self
    ) -> None:

        self.compiled_pattern = re.compile(
            self.pattern,
            re.IGNORECASE
        )



# ============================================================
# Result
# ============================================================


@dataclass(slots=True)
class PIIResult:
    """
    Result of PII scan.
    """

    text: str

    detected: bool = False

    matches: list[str] = field(
        default_factory=list
    )

    types: list[PIIType] = field(
        default_factory=list
    )

    sanitized_text: str = ""



# ============================================================
# Default Rules
# ============================================================


def default_pii_rules() -> list[PIIRule]:
    """
    Default PII patterns.
    """

    return [

        PIIRule(
            name="email",
            pii_type=PIIType.EMAIL,
            pattern=
            r"\b[A-Za-z0-9._%+-]+@"
            r"[A-Za-z0-9.-]+\."
            r"[A-Za-z]{2,}\b",
            replacement="[EMAIL_REMOVED]",
        ),


        PIIRule(
            name="phone",
            pii_type=PIIType.PHONE,
            pattern=
            r"\b(?:\+?\d{1,3})?"
            r"[\s-]?"
            r"\d{3}[\s-]?"
            r"\d{3}"
            r"[\s-]?"
            r"\d{3}\b",
            replacement="[PHONE_REMOVED]",
        ),


        PIIRule(
            name="pesel",
            pii_type=PIIType.PESEL,
            pattern=
            r"\b\d{11}\b",
            replacement="[PESEL_REMOVED]",
        ),


        PIIRule(
            name="ip_address",
            pii_type=PIIType.IP_ADDRESS,
            pattern=
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            replacement="[IP_REMOVED]",
        ),


        PIIRule(
            name="credit_card",
            pii_type=PIIType.CREDIT_CARD,
            pattern=
            r"\b(?:\d[ -]*?){13,19}\b",
            replacement="[CARD_REMOVED]",
        ),


        PIIRule(
            name="api_key",
            pii_type=PIIType.API_KEY,
            pattern=
            r"\b(sk-[A-Za-z0-9]{20,})\b",
            replacement="[API_KEY_REMOVED]",
        ),


        PIIRule(
            name="private_key",
            pii_type=PIIType.PRIVATE_KEY,
            pattern=
            r"-----BEGIN.*PRIVATE KEY-----",
            replacement="[PRIVATE_KEY_REMOVED]",
        ),

    ]



# ============================================================
# PII Detector
# ============================================================


class PIIDetector:
    """
    Detects and removes sensitive information.
    """



    def __init__(
        self,
    ) -> None:


        self.rules: list[PIIRule] = []


        self._load_rules()


        logger.info(
            "PIIDetector initialized."
        )



    # --------------------------------------------------------

    def _load_rules(
        self,
    ) -> None:

        for rule in default_pii_rules():

            rule.compile()

            self.rules.append(
                rule
            )


        logger.info(
            "Loaded %d PII rules.",
            len(self.rules),
        )



    # --------------------------------------------------------

    def scan(
        self,
        text: str,
    ) -> PIIResult:
        """
        Detect PII.
        """

        result = PIIResult(
            text=text
        )


        sanitized = text


        for rule in self.rules:

            if not rule.enabled:

                continue


            matches = (
                rule.compiled_pattern.findall(
                    sanitized
                )
            )


            if matches:

                result.detected = True

                result.matches.extend(
                    matches
                    if isinstance(matches, list)
                    else [matches]
                )

                result.types.append(
                    rule.pii_type
                )


                sanitized = (
                    rule.compiled_pattern.sub(
                        rule.replacement,
                        sanitized
                    )
                )


        result.sanitized_text = sanitized


        logger.info(
            "PII detected=%s types=%s",
            result.detected,
            result.types,
        )


        return result



    # --------------------------------------------------------

    def remove_sensitive_data(
        self,
        text: str,
    ) -> str:
        """
        Return sanitized text.
        """

        return (
            self.scan(text)
            .sanitized_text
        )