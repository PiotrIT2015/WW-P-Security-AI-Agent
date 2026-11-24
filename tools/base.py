"""
tools/base.py

Base interface for WW-P Security AI Agent tools.

Python 3.12+
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from authentication.auth import UserContext


# =============================================================================
# Tool result
# =============================================================================

@dataclass(slots=True)
class ToolResult:
    """
    Standardized tool execution result.
    """

    tool_name: str

    success: bool

    output: dict[str, Any] | list[Any] | str | None = None

    error: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =============================================================================
# Base tool
# =============================================================================

class BaseTool(ABC):
    """
    Abstract base class for all agent tools.

    Every tool must define:

    - name
    - description
    - required_roles
    - input_schema
    - execute()
    """

    name: str = ""

    description: str = ""

    required_roles: set[str] = set()

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    @abstractmethod
    def execute(
        self,
        arguments: dict[str, Any],
        user: UserContext,
    ) -> dict[str, Any] | list[Any] | str:
        """
        Execute the tool.

        Args:
            arguments: Validated tool arguments.
            user: Authenticated user context.

        Returns:
            Tool output.
        """
        raise NotImplementedError

    def get_definition(self) -> dict[str, Any]:
        """
        Return tool metadata.

        This definition can later be adapted
        to the Ollama tool-calling interface.
        """

        return {
            "name": self.name,
            "description": self.description,
            "required_roles": sorted(
                self.required_roles
            ),
            "input_schema": self.input_schema,
        }

    def validate_definition(self) -> None:
        """
        Validate basic tool configuration.
        """

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(
                "Tool name cannot be empty."
            )

        if not self.name.replace("_", "").isalnum():
            raise ValueError(
                f"Invalid tool name: {self.name}"
            )

        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError(
                f"Tool '{self.name}' requires a description."
            )

        if not isinstance(self.required_roles, set):
            raise TypeError(
                "required_roles must be a set."
            )

        if not all(
            isinstance(role, str) and role.strip()
            for role in self.required_roles
        ):
            raise ValueError(
                "Tool roles must contain non-empty strings."
            )

        if not isinstance(self.input_schema, dict):
            raise TypeError(
                "input_schema must be a dictionary."
            )

        if self.input_schema.get("type") != "object":
            raise ValueError(
                "Tool input schema must have type='object'."
            )