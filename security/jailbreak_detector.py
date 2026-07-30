"""
security/jailbreak_detector.py

WW-P Secure RAG

Jailbreak Detection Module

Stage:
5.2

Python 3.12+
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Pattern
import logging
import re


logger = logging.getLogger(
    "secure_rag.security.jailbreak_detector"
)


# ============================================================
# Exceptions
# ============================================================


class JailbreakDetectorError(Exception):
    """
    Base jailbreak detector exception.
    """



class JailbreakDetected(JailbreakDetectorError):
    """
    Raised when jailbreak attempt is detected.
    """

    def __init__(
        self,
        message: str,
        result: "JailbreakResult | None" = None,
    ) -> None:

        super().__init__(
            message
        )

        self.result = result



# ============================================================
# Severity
# ============================================================


class JailbreakSeverity(Enum):
    """
    Jailbreak risk level.
    """

    LOW = "low"

    MEDIUM = "medium"

    HIGH = "high"

    CRITICAL = "critical"



# ============================================================
# Detection Rule
# ============================================================


@dataclass(slots=True)
class JailbreakRule:
    """
    Single jailbreak detection rule.
    """

    name: str

    pattern: str

    score: int

    severity: JailbreakSeverity

    enabled: bool = True

    compiled_pattern: Pattern[str] | None = None


    def compile(
        self,
        ignore_case: bool = True,
    ) -> None:

        flags = 0

        if ignore_case:

            flags |= re.IGNORECASE


        self.compiled_pattern = re.compile(
            self.pattern,
            flags
        )



# ============================================================
# Result
# ============================================================


@dataclass(slots=True)
class JailbreakResult:
    """
    Jailbreak analysis result.
    """

    prompt: str

    score: int = 0

    matched_rules: list[str] = field(
        default_factory=list
    )

    severity: JailbreakSeverity = (
        JailbreakSeverity.LOW
    )

    allowed: bool = True

    reason: str = ""



# ============================================================
# Configuration
# ============================================================


@dataclass(slots=True)
class JailbreakConfig:
    """
    Detector configuration.
    """

    block_score: int = 70

    warning_score: int = 30

    ignore_case: bool = True

    enable_logging: bool = True



# ============================================================
# Default Rules
# ============================================================


def default_jailbreak_rules() -> list[JailbreakRule]:
    """
    Default jailbreak patterns.
    """

    return [

        JailbreakRule(
            name="dan_mode",
            pattern=
            r"dan\s+mode",
            score=40,
            severity=
            JailbreakSeverity.HIGH,
        ),


        JailbreakRule(
            name="developer_mode",
            pattern=
            r"developer\s+mode",
            score=35,
            severity=
            JailbreakSeverity.HIGH,
        ),


        JailbreakRule(
            name="unrestricted_ai",
            pattern=
            r"unrestricted\s+ai",
            score=40,
            severity=
            JailbreakSeverity.HIGH,
        ),


        JailbreakRule(
            name="no_rules",
            pattern=
            r"without\s+rules",
            score=30,
            severity=
            JailbreakSeverity.MEDIUM,
        ),


        JailbreakRule(
            name="ignore_policy",
            pattern=
            r"ignore\s+(safety|policy|rules)",
            score=45,
            severity=
            JailbreakSeverity.CRITICAL,
        ),


        JailbreakRule(
            name="roleplay_bypass",
            pattern=
            r"you\s+are\s+now",
            score=20,
            severity=
            JailbreakSeverity.MEDIUM,
        ),


        JailbreakRule(
            name="evil_assistant",
            pattern=
            r"evil\s+assistant",
            score=40,
            severity=
            JailbreakSeverity.HIGH,
        ),

    ]



# ============================================================
# Jailbreak Detector Engine
# ============================================================


class JailbreakDetector:
    """
    Jailbreak detection engine.

    Detects:
    - policy bypass attempts
    - unrestricted mode requests
    - malicious roleplay
    - safety disabling
    """



    def __init__(
        self,
        config: JailbreakConfig | None = None,
    ) -> None:


        self.config = (
            config
            or JailbreakConfig()
        )


        self.rules: list[JailbreakRule] = []


        self._load_rules()


        logger.info(
            "JailbreakDetector initialized."
        )



    # --------------------------------------------------------

    def _load_rules(
        self,
    ) -> None:

        for rule in default_jailbreak_rules():

            rule.compile(
                self.config.ignore_case
            )

            self.rules.append(
                rule
            )


        logger.info(
            "Loaded %d jailbreak rules.",
            len(self.rules),
        )



    # --------------------------------------------------------

    def _check_rule(
        self,
        rule: JailbreakRule,
        text: str,
    ) -> bool:

        if not rule.enabled:

            return False


        return bool(
            rule.compiled_pattern.search(
                text
            )
        )



    # --------------------------------------------------------

    def _severity(
        self,
        score: int,
    ) -> JailbreakSeverity:


        if score >= 80:

            return JailbreakSeverity.CRITICAL


        if score >= 40:

            return JailbreakSeverity.HIGH


        if score >= 20:

            return JailbreakSeverity.MEDIUM


        return JailbreakSeverity.LOW



    # --------------------------------------------------------

    def scan(
        self,
        prompt: str,
    ) -> JailbreakResult:
        """
        Analyze prompt.
        """

        result = JailbreakResult(
            prompt=prompt
        )


        for rule in self.rules:

            if self._check_rule(
                rule,
                prompt
            ):

                result.score += (
                    rule.score
                )

                result.matched_rules.append(
                    rule.name
                )


        result.severity = (
            self._severity(
                result.score
            )
        )


        if result.score >= (
            self.config.warning_score
        ):

            result.reason = (
                "Possible jailbreak attempt detected."
            )


        logger.info(
            "Jailbreak scan score=%s rules=%s",
            result.score,
            result.matched_rules,
        )


        return result



    # --------------------------------------------------------

    def validate(
        self,
        prompt: str,
    ) -> JailbreakResult:
        """
        Enforce security policy.
        """

        result = self.scan(
            prompt
        )


        if result.score >= (
            self.config.block_score
        ):


            logger.warning(
                "Jailbreak blocked."
            )


            raise JailbreakDetected(
                message=
                "Jailbreak attempt blocked.",
                result=result,
            )


        return result