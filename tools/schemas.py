"""
tools/schemas.py

Input validation for WW-P Security AI Agent tools.

Python 3.12+
"""

from __future__ import annotations

from typing import Any


class ToolValidationError(Exception):
    """Raised when tool arguments are invalid."""

    pass


# =============================================================================
# Type validation
# =============================================================================

def _validate_type(
    value: Any,
    expected_type: str,
) -> bool:

    if expected_type == "string":
        return isinstance(value, str)

    if expected_type == "integer":
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
        )

    if expected_type == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        )

    if expected_type == "boolean":
        return isinstance(value, bool)

    if expected_type == "array":
        return isinstance(value, list)

    if expected_type == "object":
        return isinstance(value, dict)

    if expected_type == "null":
        return value is None

    raise ToolValidationError(
        f"Unsupported schema type: {expected_type}"
    )


# =============================================================================
# Argument validation
# =============================================================================

def validate_arguments(
    arguments: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate arguments against a JSON-schema-like definition.

    Supported:
        string
        integer
        number
        boolean
        array
        object
        null

    Also supports:
        required
        enum
        minimum
        maximum
        minLength
        maxLength
        additionalProperties
    """

    if not isinstance(arguments, dict):
        raise ToolValidationError(
            "Tool arguments must be an object."
        )

    if not isinstance(schema, dict):
        raise ToolValidationError(
            "Invalid tool schema."
        )

    if schema.get("type") != "object":
        raise ToolValidationError(
            "Root schema must have type='object'."
        )

    properties = schema.get(
        "properties",
        {},
    )

    required = schema.get(
        "required",
        [],
    )

    if not isinstance(properties, dict):
        raise ToolValidationError(
            "Schema properties must be an object."
        )

    if not isinstance(required, list):
        raise ToolValidationError(
            "Schema required must be a list."
        )

    # Check required parameters.

    for field in required:

        if field not in arguments:
            raise ToolValidationError(
                f"Missing required parameter: {field}"
            )

    # Reject unexpected parameters.

    if schema.get("additionalProperties", False) is False:

        unexpected = (
            set(arguments.keys())
            - set(properties.keys())
        )

        if unexpected:
            raise ToolValidationError(
                "Unexpected parameters: "
                + ", ".join(sorted(unexpected))
            )

    # Validate supplied parameters.

    for field, value in arguments.items():

        if field not in properties:
            continue

        definition = properties[field]

        if not isinstance(definition, dict):
            raise ToolValidationError(
                f"Invalid schema for parameter: {field}"
            )

        expected_type = definition.get("type")

        if not expected_type:
            raise ToolValidationError(
                f"Missing type for parameter: {field}"
            )

        if not _validate_type(
            value,
            expected_type,
        ):
            raise ToolValidationError(
                f"Invalid type for parameter: {field}"
            )

        # Enum validation.

        if "enum" in definition:

            if value not in definition["enum"]:
                raise ToolValidationError(
                    f"Invalid value for parameter: {field}"
                )

        # String validation.

        if isinstance(value, str):

            min_length = definition.get(
                "minLength"
            )

            max_length = definition.get(
                "maxLength"
            )

            if min_length is not None and len(value) < min_length:
                raise ToolValidationError(
                    f"Parameter '{field}' is too short."
                )

            if max_length is not None and len(value) > max_length:
                raise ToolValidationError(
                    f"Parameter '{field}' is too long."
                )

        # Numeric validation.

        if isinstance(value, (int, float)) and not isinstance(value, bool):

            minimum = definition.get("minimum")

            maximum = definition.get("maximum")

            if minimum is not None and value < minimum:
                raise ToolValidationError(
                    f"Parameter '{field}' is below minimum."
                )

            if maximum is not None and value > maximum:
                raise ToolValidationError(
                    f"Parameter '{field}' exceeds maximum."
                )

    return arguments