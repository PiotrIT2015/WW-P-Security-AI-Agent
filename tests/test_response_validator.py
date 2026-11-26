import pytest

from agent.response_validator import (
    ResponseValidator,
    ResponseValidationError,
)


@pytest.fixture
def validator():
    return ResponseValidator()


# ============================================================
# BASIC RESPONSE VALIDATION
# ============================================================

def test_accepts_valid_response(validator):

    result = validator.validate(
        "This is a valid response."
    )

    assert result == "This is a valid response."


def test_strips_response_whitespace(validator):

    result = validator.validate(
        "  Valid response.  "
    )

    assert result == "Valid response."


def test_rejects_empty_response(validator):

    with pytest.raises(ResponseValidationError):

        validator.validate("")


def test_rejects_whitespace_response(validator):

    with pytest.raises(ResponseValidationError):

        validator.validate("   ")


def test_rejects_non_string_response(validator):

    with pytest.raises(ResponseValidationError):

        validator.validate(
            {"content": "response"}
        )


def test_rejects_null_character(validator):

    with pytest.raises(ResponseValidationError):

        validator.validate(
            "Invalid\x00response"
        )


def test_rejects_oversized_response():

    validator = ResponseValidator(
        max_response_chars=10
    )

    with pytest.raises(ResponseValidationError):

        validator.validate(
            "This response is too long."
        )


# ============================================================
# SOURCE IDENTIFIER VALIDATION
# ============================================================

def test_accepts_authorized_citations(validator):

    result = validator.validate(
        "DNS translates names into IP addresses [1].",
        authorized_source_ids=[1, 2],
    )

    assert "[1]" in result


def test_accepts_multiple_authorized_citations(validator):

    result = validator.validate(
        "DNS uses records [1] and [2].",
        authorized_source_ids=[1, 2],
    )

    assert "[1]" in result
    assert "[2]" in result


def test_rejects_unauthorized_citation(validator):

    with pytest.raises(ResponseValidationError):

        validator.validate(
            "DNS uses records [3].",
            authorized_source_ids=[1, 2],
        )


def test_rejects_citation_without_sources(validator):

    with pytest.raises(ResponseValidationError):

        validator.validate(
            "DNS uses records [1].",
            authorized_source_ids=[],
        )


def test_accepts_response_without_citations(validator):

    result = validator.validate(
        "DNS translates domain names into IP addresses.",
        authorized_source_ids=[1, 2],
    )

    assert "DNS translates" in result


def test_rejects_malformed_citation(validator):

    with pytest.raises(ResponseValidationError):

        validator.validate(
            "DNS uses record [ 1 ].",
            authorized_source_ids=[1],
        )


def test_rejects_multiple_ids_in_single_citation(validator):

    with pytest.raises(ResponseValidationError):

        validator.validate(
            "DNS uses records [1, 2].",
            authorized_source_ids=[1, 2],
        )


def test_rejects_zero_citation(validator):

    with pytest.raises(ResponseValidationError):

        validator.validate(
            "DNS uses record [0].",
            authorized_source_ids=[1],
        )


# ============================================================
# SOURCE HEADER PROTECTION
# ============================================================

def test_rejects_model_generated_source_header(validator):

    with pytest.raises(ResponseValidationError):

        validator.validate(
            "DNS translates names.\n\nRetrieved sources:\n[1]",
            authorized_source_ids=[1],
        )


def test_rejects_case_insensitive_source_header(validator):

    with pytest.raises(ResponseValidationError):

        validator.validate(
            "retrieved SOURCES:\n[1]",
            authorized_source_ids=[1],
        )


# ============================================================
# AUTHORIZED SOURCE ID VALIDATION
# ============================================================

def test_rejects_invalid_source_id(validator):

    with pytest.raises(ResponseValidationError):

        validator.validate(
            "Valid answer.",
            authorized_source_ids=[0],
        )


def test_rejects_boolean_source_id(validator):

    with pytest.raises(ResponseValidationError):

        validator.validate(
            "Valid answer.",
            authorized_source_ids=[True],
        )


def test_rejects_duplicate_source_ids(validator):

    with pytest.raises(ResponseValidationError):

        validator.validate(
            "Valid answer.",
            authorized_source_ids=[1, 1],
        )


# ============================================================
# CONFIGURATION
# ============================================================

def test_rejects_invalid_max_response_chars():

    with pytest.raises(ValueError):

        ResponseValidator(
            max_response_chars=0
        )


def test_rejects_boolean_max_response_chars():

    with pytest.raises(ValueError):

        ResponseValidator(
            max_response_chars=True
        )