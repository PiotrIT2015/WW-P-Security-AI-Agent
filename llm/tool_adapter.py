"""
llm/tool_adapter.py

Converts ToolRegistry definitions into Ollama-compatible
tool definitions.

Responsibilities:
- Read tool definitions from ToolRegistry.
- Validate tool metadata.
- Validate input schemas.
- Convert definitions into Ollama Chat API format.
- Prevent internal permission metadata from being exposed.

Security:
- Does not execute tools.
- Does not authorize users.
- Does not bypass ToolExecutor.
- Does not expose required_roles to the LLM.

Compatible with Python 3.10+.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any, Protocol


logger = logging.getLogger("secure_rag.llm.tool_adapter")


class ToolDefinitionAdapterError(ValueError):
    """Raised when a tool definition cannot be adapted."""


class ToolRegistryProtocol(Protocol):
    """Minimal interface required from ToolRegistry."""

    def list_tools(self) -> list[dict[str, Any]]:
        ...


class ToolDefinitionAdapter:
    """
    Converts registered tools into Ollama-compatible definitions.
    """

    def __init__(self, registry: ToolRegistryProtocol) -> None:

        if registry is None:
            raise ValueError("Tool registry cannot be None.")

        if not callable(getattr(registry, "list_tools", None)):
            raise TypeError(
                "Registry must provide a callable list_tools() method."
            )

        self.registry = registry

    # =========================================================
    # Tool name validation
    # =========================================================

    @staticmethod
    def _validate_name(name: Any) -> str:

        if not isinstance(name, str):
            raise ToolDefinitionAdapterError(
                "Tool name must be a string."
            )

        if not name:
            raise ToolDefinitionAdapterError(
                "Tool name cannot be empty."
            )

        if name != name.strip():
            raise ToolDefinitionAdapterError(
                "Tool name cannot contain surrounding whitespace."
            )

        if not name.replace("_", "").isalnum():
            raise ToolDefinitionAdapterError(
                f"Invalid tool name: {name}"
            )

        return name

    # =========================================================
    # Description validation
    # =========================================================

    @staticmethod
    def _validate_description(description: Any) -> str:

        if not isinstance(description, str):
            raise ToolDefinitionAdapterError(
                "Tool description must be a string."
            )

        if not description.strip():
            raise ToolDefinitionAdapterError(
                "Tool description cannot be empty."
            )

        return description.strip()

    # =========================================================
    # JSON Schema validation
    # =========================================================

    @staticmethod
    def _validate_schema(schema: Any) -> dict[str, Any]:
        """
        Validate the basic structure of a tool input schema.

        The schema must describe a JSON object.
        """

        if not isinstance(schema, dict):
            raise ToolDefinitionAdapterError(
                "Tool input schema must be a dictionary."
            )

        if schema.get("type") != "object":
            raise ToolDefinitionAdapterError(
                "Tool input schema must have type='object'."
            )

        properties = schema.get("properties")

        if not isinstance(properties, dict):
            raise ToolDefinitionAdapterError(
                "Tool schema properties must be a dictionary."
            )

        required = schema.get("required", [])

        if not isinstance(required, list):
            raise ToolDefinitionAdapterError(
                "Tool schema required must be a list."
            )

        if not all(
            isinstance(item, str) and item
            for item in required
        ):
            raise ToolDefinitionAdapterError(
                "Required schema fields must be non-empty strings."
            )

        if len(required) != len(set(required)):
            raise ToolDefinitionAdapterError(
                "Tool schema contains duplicate required fields."
            )

        unknown_required = set(required) - set(properties)

        if unknown_required:
            raise ToolDefinitionAdapterError(
                "Required fields are missing from properties: "
                + ", ".join(sorted(unknown_required))
            )

        additional_properties = schema.get(
            "additionalProperties",
            False,
        )

        if not isinstance(additional_properties, bool):
            raise ToolDefinitionAdapterError(
                "additionalProperties must be a boolean."
            )

        # Copy the schema to prevent modifications to the registry.
        validated_schema = deepcopy(schema)

        # Ollama receives a complete JSON Schema object.
        validated_schema.setdefault(
            "required",
            [],
        )

        validated_schema.setdefault(
            "additionalProperties",
            False,
        )

        # Ensure that the schema can be safely serialized as JSON.
        try:
            json.dumps(
                validated_schema,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ToolDefinitionAdapterError(
                "Tool schema must contain valid JSON-compatible values."
            ) from exc

        return validated_schema

    # =========================================================
    # Single tool conversion
    # =========================================================

    def adapt_tool_definition(
        self,
        definition: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Convert one registry definition into Ollama format.

        Expected registry definition:

            {
                "name": "system_status",
                "description": "Returns system status.",
                "required_roles": ["admin"],
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False
                }
            }
        """

        if not isinstance(definition, dict):
            raise ToolDefinitionAdapterError(
                "Tool definition must be a dictionary."
            )

        name = self._validate_name(
            definition.get("name")
        )

        description = self._validate_description(
            definition.get("description")
        )

        schema = self._validate_schema(
            definition.get("input_schema")
        )

        ollama_definition = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": schema,
            },
        }

        return ollama_definition

    # =========================================================
    # Registry conversion
    # =========================================================

    def adapt_registry(self) -> list[dict[str, Any]]:
        """
        Convert all registered tools into Ollama format.

        Returns:
            List of Ollama-compatible tool definitions.

        Raises:
            ToolDefinitionAdapterError:
                If registry data or any tool definition is invalid.
        """

        try:
            definitions = self.registry.list_tools()

        except Exception as exc:
            logger.exception(
                "Unable to retrieve tool definitions from registry."
            )

            raise ToolDefinitionAdapterError(
                "Unable to retrieve tool definitions from registry."
            ) from exc

        if not isinstance(definitions, list):
            raise ToolDefinitionAdapterError(
                "ToolRegistry.list_tools() must return a list."
            )

        adapted_tools: list[dict[str, Any]] = []

        registered_names: set[str] = set()

        for index, definition in enumerate(definitions):

            try:
                adapted = self.adapt_tool_definition(
                    definition
                )

            except ToolDefinitionAdapterError as exc:
                raise ToolDefinitionAdapterError(
                    f"Invalid tool definition at index {index}: {exc}"
                ) from exc

            tool_name = adapted["function"]["name"]

            if tool_name in registered_names:
                raise ToolDefinitionAdapterError(
                    f"Duplicate tool definition: {tool_name}"
                )

            registered_names.add(tool_name)

            adapted_tools.append(adapted)

        logger.info(
            "Adapted %d tool definitions for Ollama.",
            len(adapted_tools),
        )

        return adapted_tools

    # =========================================================
    # Utility
    # =========================================================

    def count(self) -> int:
        """
        Return the number of definitions available for Ollama.
        """

        return len(self.adapt_registry())