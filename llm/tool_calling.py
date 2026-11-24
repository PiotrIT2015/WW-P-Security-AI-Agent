"""
llm/tool_calling.py

Parser for tool calls returned by Ollama Chat API.

Responsibilities:
- Parse assistant messages containing tool_calls.
- Validate tool call structure.
- Validate tool names and arguments.
- Normalize calls into ToolCall objects.

Security:
- Does not execute tools.
- Does not authorize tool calls.
- Does not modify the ToolRegistry.
- ToolExecutor remains the only execution path.

Compatible with Python 3.10+.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ToolCallParseError(ValueError):
    """Raised when an Ollama tool call has an invalid structure."""


@dataclass(frozen=True, slots=True)
class ToolCall:
    """
    Normalized tool call.

    Attributes:
        name: Registered tool name.
        arguments: Arguments supplied by the LLM.
    """

    name: str
    arguments: dict[str, Any]


def _validate_tool_name(name: Any) -> str:
    """
    Validate a tool name.

    The validation follows the naming convention used by BaseTool.
    """

    if not isinstance(name, str):
        raise ToolCallParseError(
            "Tool name must be a string."
        )

    if not name:
        raise ToolCallParseError(
            "Tool name cannot be empty."
        )

    if name != name.strip():
        raise ToolCallParseError(
            "Tool name cannot contain leading or trailing whitespace."
        )

    if not name.replace("_", "").isalnum():
        raise ToolCallParseError(
            f"Invalid tool name: {name}"
        )

    return name


def _validate_arguments(arguments: Any) -> dict[str, Any]:
    """
    Validate tool arguments.

    Ollama tool calls are expected to provide arguments
    as a JSON object, represented by a Python dictionary.
    """

    if not isinstance(arguments, dict):
        raise ToolCallParseError(
            "Tool arguments must be an object."
        )

    return dict(arguments)


def parse_tool_calls(
    message: dict[str, Any],
) -> list[ToolCall]:
    """
    Parse tool calls from an Ollama assistant message.

    Expected input:

        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "system_status",
                        "arguments": {}
                    }
                }
            ]
        }

    Returns:
        List of normalized ToolCall objects.

    Returns an empty list when the assistant message
    does not contain any tool calls.

    Raises:
        ToolCallParseError:
            If the message or any tool call is malformed.
    """

    if not isinstance(message, dict):
        raise ToolCallParseError(
            "Assistant message must be a dictionary."
        )

    role = message.get("role")

    if role != "assistant":
        raise ToolCallParseError(
            "Tool calls can only be parsed from assistant messages."
        )

    if "tool_calls" not in message:
        return []

    raw_tool_calls = message["tool_calls"]

    if not isinstance(raw_tool_calls, list):
        raise ToolCallParseError(
            "tool_calls must be a list."
        )

    parsed_calls: list[ToolCall] = []

    for index, raw_call in enumerate(raw_tool_calls):

        if not isinstance(raw_call, dict):
            raise ToolCallParseError(
                f"Tool call at index {index} must be a dictionary."
            )

        function = raw_call.get("function")

        if not isinstance(function, dict):
            raise ToolCallParseError(
                f"Tool call at index {index} must contain a function object."
            )

        if "name" not in function:
            raise ToolCallParseError(
                f"Tool call at index {index} is missing function name."
            )

        if "arguments" not in function:
            raise ToolCallParseError(
                f"Tool call at index {index} is missing arguments."
            )

        name = _validate_tool_name(
            function["name"]
        )

        arguments = _validate_arguments(
            function["arguments"]
        )

        parsed_calls.append(
            ToolCall(
                name=name,
                arguments=arguments,
            )
        )

    return parsed_calls