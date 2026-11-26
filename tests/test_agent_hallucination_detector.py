"""
tests/test_agent_hallucination_detector.py

Tests for HallucinationDetector.

Stage 3.6.2:
- Claim extraction.
- Context grounding.
- Unsupported claims.
- Insufficient context.
- Evaluation statistics.
- Input validation.

Compatible with Python 3.10+.
"""

import pytest

from agent.hallucination_detector import (
    HallucinationDetector,
    ClaimAssessment,
    HallucinationReport,
)


@pytest.fixture
def detector():
    return HallucinationDetector()


@pytest.fixture
def context():
    return (
        "DNS translates domain names into IP addresses. "
        "DNS allows users to access network services "
        "using readable domain names."
    )


# ============================================================
# INITIALIZATION
# ============================================================

def test_detector_initialization():

    detector = HallucinationDetector()

    assert detector.minimum_overlap == 0.35

    assert detector.minimum_matched_terms == 2


def test_detector_rejects_invalid_overlap():

    with pytest.raises(ValueError):

        HallucinationDetector(
            minimum_overlap=1.5
        )


def test_detector_rejects_invalid_minimum_terms():

    with pytest.raises(ValueError):

        HallucinationDetector(
            minimum_matched_terms=0
        )


# ============================================================
# CLAIM EXTRACTION
# ============================================================

def test_extracts_single_claim(detector):

    claims = detector.extract_claims(
        "DNS translates domain names into IP addresses."
    )

    assert len(claims) == 1

    assert "DNS translates" in claims[0]


def test_extracts_multiple_claims(detector):

    claims = detector.extract_claims(
        "DNS translates domain names. "
        "It uses readable names."
    )

    assert len(claims) == 2


def test_extracts_claims_from_multiline_response(detector):

    claims = detector.extract_claims(
        "DNS translates domain names.\n"
        "IP addresses identify network endpoints."
    )

    assert len(claims) == 2


def test_ignores_empty_response(detector):

    assert detector.extract_claims("") == ()


def test_rejects_invalid_response_type(detector):

    with pytest.raises(ValueError):

        detector.extract_claims(None)


# ============================================================
# CLAIM EVALUATION
# ============================================================

def test_accepts_supported_claim(detector, context):

    result = detector.evaluate_claim(
        claim="DNS translates domain names into IP addresses.",
        context=context,
    )

    assert isinstance(result, ClaimAssessment)

    assert result.status == "supported"

    assert result.overlap_ratio > 0


def test_rejects_unsupported_claim(detector, context):

    result = detector.evaluate_claim(
        claim="The server uses quantum encryption.",
        context=context,
    )

    assert result.status == "unsupported"

    assert result.overlap_ratio == 0.0


def test_detects_insufficient_context(detector):

    result = detector.evaluate_claim(
        claim="DNS translates domain names.",
        context="",
    )

    assert result.status == "insufficient_context"


def test_rejects_empty_claim(detector, context):

    with pytest.raises(ValueError):

        detector.evaluate_claim(
            claim=" ",
            context=context,
        )


# ============================================================
# FULL RESPONSE EVALUATION
# ============================================================

def test_evaluates_grounded_response(detector, context):

    report = detector.evaluate(
        response=(
            "DNS translates domain names into IP addresses."
        ),
        context=context,
    )

    assert isinstance(report, HallucinationReport)

    assert report.status == "grounded"

    assert report.supported_count == 1

    assert report.unsupported_count == 0

    assert report.coverage == 1.0


def test_detects_unsupported_claim(detector, context):

    report = detector.evaluate(
        response=(
            "DNS translates domain names into IP addresses. "
            "DNS uses quantum encryption."
        ),
        context=context,
    )

    assert report.status == "contains_unsupported_claims"

    assert report.supported_count == 1

    assert report.unsupported_count == 1

    assert report.coverage == 0.5


def test_detects_empty_context(detector):

    report = detector.evaluate(
        response="DNS translates domain names.",
        context="",
    )

    assert report.status == "insufficient_context"

    assert report.unverified_count == 1

    assert report.supported_count == 0


def test_handles_empty_response(detector, context):

    report = detector.evaluate(
        response="",
        context=context,
    )

    assert report.status == "no_claims"

    assert report.claims == ()


def test_rejects_invalid_context_type(detector):

    with pytest.raises(ValueError):

        detector.evaluate(
            response="DNS translates domain names.",
            context=None,
        )


# ============================================================
# REPORT
# ============================================================

def test_report_to_dict(detector, context):

    report = detector.evaluate(
        response="DNS translates domain names into IP addresses.",
        context=context,
    )

    result = report.to_dict()

    assert result["status"] == "grounded"

    assert result["is_grounded"] is True

    assert result["supported_count"] == 1

    assert isinstance(result["claims"], list)


def test_unsupported_report_is_not_grounded(detector, context):

    report = detector.evaluate(
        response="The server uses quantum encryption.",
        context=context,
    )

    assert report.is_grounded is False


def test_report_contains_matched_terms(detector, context):

    report = detector.evaluate(
        response="DNS translates domain names into IP addresses.",
        context=context,
    )

    assert report.claims[0].matched_terms

    assert "dns" in report.claims[0].matched_terms