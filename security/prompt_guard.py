"""
security/prompt_guard.py

WW-P Secure RAG

Prompt Injection Protection Module

Version:
5.1

Features:
- Regex based prompt injection detection
- Risk scoring
- Severity classification
- Security policy enforcement
- Blocking dangerous prompts

Python 3.12+
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Pattern
import logging
import re


logger = logging.getLogger(
    "secure_rag.security.prompt_guard"
)


# ============================================================
# Exceptions
# ============================================================


class PromptGuardError(Exception):
    """
    Base Prompt Guard exception.
    """



class PromptInjectionDetected(PromptGuardError):
    """
    Raised when malicious prompt is blocked.
    """


    def __init__(
        self,
        message: str,
        result: "ScanResult | None" = None,
    ) -> None:

        super().__init__(
            message
        )

        self.result = result



# ============================================================
# Severity
# ============================================================


class Severity(Enum):
    """
    Risk severity level.
    """

    LOW = "low"

    MEDIUM = "medium"

    HIGH = "high"

    CRITICAL = "critical"



# ============================================================
# Detection Rule
# ============================================================


@dataclass(slots=True)
class DetectionRule:
    """
    Single prompt injection rule.
    """


    name: str

    pattern: str

    score: int

    severity: Severity

    enabled: bool = True

    compiled_pattern: Pattern[str] | None = None



    def compile(
        self,
        ignore_case: bool = True,
    ) -> None:
        """
        Compile regex.
        """

        flags = 0

        if ignore_case:

            flags |= re.IGNORECASE


        self.compiled_pattern = re.compile(
            self.pattern,
            flags
        )



# ============================================================
# Scan Result
# ============================================================


@dataclass(slots=True)
class ScanResult:
    """
    Prompt analysis result.
    """


    prompt: str

    score: int = 0

    matched_rules: list[str] = field(
        default_factory=list
    )

    severity: Severity = Severity.LOW

    allowed: bool = True

    reason: str = ""



# ============================================================
# Configuration
# ============================================================


@dataclass(slots=True)
class PromptGuardConfig:
    """
    Prompt Guard configuration.
    """


    block_score: int = 70

    warning_score: int = 30

    ignore_case: bool = True

    enable_logging: bool = True



# ============================================================
# Default Detection Rules
# ============================================================


def default_rules() -> list[DetectionRule]:
    """
    Built-in prompt injection rules.
    """


    return [

        DetectionRule(
            name="ignore_previous",
            pattern=
            r"ignore\s+(all\s+)?previous\s+instructions?",
            score=35,
            severity=Severity.HIGH,
        ),


        DetectionRule(
            name="system_prompt",
            pattern=
            r"system\s+prompt",
            score=30,
            severity=Severity.HIGH,
        ),


        DetectionRule(
            name="reveal_prompt",
            pattern=
            r"reveal.*prompt",
            score=40,
            severity=Severity.CRITICAL,
        ),


        DetectionRule(
            name="developer_message",
            pattern=
            r"developer\s+message",
            score=35,
            severity=Severity.HIGH,
        ),


        DetectionRule(
            name="forget_instructions",
            pattern=
            r"forget\s+all\s+instructions",
            score=45,
            severity=Severity.CRITICAL,
        ),


        DetectionRule(
            name="override",
            pattern=
            r"override",
            score=15,
            severity=Severity.MEDIUM,
        ),


        DetectionRule(
            name="bypass_security",
            pattern=
            r"bypass",
            score=25,
            severity=Severity.HIGH,
        ),


        DetectionRule(
            name="disable_security",
            pattern=
            r"disable\s+security",
            score=45,
            severity=Severity.CRITICAL,
        ),


        DetectionRule(
            name="act_as",
            pattern=
            r"act\s+as",
            score=10,
            severity=Severity.LOW,
        ),


        DetectionRule(
            name="pretend",
            pattern=
            r"pretend\s+to\s+be",
            score=10,
            severity=Severity.LOW,
        ),
    ]



# ============================================================
# Prompt Guard Engine
# ============================================================


class PromptGuard:
    """
    Prompt Injection Detection Engine.
    """



    def __init__(
        self,
        config: PromptGuardConfig | None = None,
    ) -> None:


        self.config = (
            config
            or PromptGuardConfig()
        )


        self.rules: list[DetectionRule] = []


        self._load_default_rules()


        logger.info(
            "PromptGuard initialized."
        )



    # --------------------------------------------------------

    def _load_default_rules(
        self,
    ) -> None:
        """
        Load default security rules.
        """


        for rule in default_rules():

            rule.compile(
                self.config.ignore_case
            )

            self.rules.append(
                rule
            )


        logger.info(
            "Loaded %d rules.",
            len(self.rules)
        )



    # --------------------------------------------------------

    def register_rule(
        self,
        rule: DetectionRule,
    ) -> None:
        """
        Add custom rule.
        """


        rule.compile(
            self.config.ignore_case
        )


        self.rules.append(
            rule
        )



    # --------------------------------------------------------

    def get_rules(
        self,
    ) -> list[DetectionRule]:

        return list(
            self.rules
        )



    # --------------------------------------------------------

    def _check_rule(
        self,
        rule: DetectionRule,
        prompt: str,
    ) -> bool:
        """
        Check single rule.
        """


        if not rule.enabled:

            return False


        if rule.compiled_pattern is None:

            rule.compile(
                self.config.ignore_case
            )


        return bool(
            rule.compiled_pattern.search(
                prompt
            )
        )



    # --------------------------------------------------------

    def _calculate_severity(
        self,
        score: int,
    ) -> Severity:
        """
        Convert score to severity.
        """


        if score >= 80:

            return Severity.CRITICAL


        if score >= 40:

            return Severity.HIGH


        if score >= 20:

            return Severity.MEDIUM


        return Severity.LOW



    # --------------------------------------------------------

    def scan(
        self,
        prompt: str,
    ) -> ScanResult:
        """
        Analyze prompt.
        """


        if not isinstance(
            prompt,
            str
        ):

            raise TypeError(
                "Prompt must be string."
            )


        result = ScanResult(
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
            self._calculate_severity(
                result.score
            )
        )



        if result.score >= (
            self.config.warning_score
        ):

            result.reason = (
                "Suspicious prompt detected."
            )



        if self.config.enable_logging:

            logger.info(
                "Prompt scan: score=%s severity=%s rules=%s",
                result.score,
                result.severity.value,
                result.matched_rules,
            )


        return result



    # --------------------------------------------------------

    def validate(
        self,
        prompt: str,
    ) -> ScanResult:
        """
        Validate prompt against security policy.
        """


        result = self.scan(
            prompt
        )


        if result.score >= (
            self.config.block_score
        ):


            logger.warning(
                "Prompt blocked: %s",
                result.matched_rules
            )


            raise PromptInjectionDetected(
                message=
                "Prompt blocked by security policy.",
                result=result,
            )


        return result



    # --------------------------------------------------------

    def is_suspicious(
        self,
        prompt: str,
    ) -> bool:
        """
        Check if prompt requires review.
        """


        result = self.scan(
            prompt
        )


        return (
            result.score
            >= self.config.warning_score
        )