"""
llm/tool_result_handler.py

Converts ToolResult objects into Ollama-compatible tool messages.

Responsibilities:
- Convert a single ToolResult into an Ollama tool message.
- Convert multiple ToolResult objects.
- Preserve tool names.
- Serialize execution results into JSON.
- Handle successful and failed tool executions.
- Validate input before processing.

Security:
- Does not execute tools.
- Does not bypass ToolExecutor.
- Does not modify permissions.
- Does not expose internal execution metadata.
- Does not trust arbitrary result objects.

Compatible with Python 3.10+.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from tools.base import ToolResult


logger = logging.getLogger("secure_rag.llm.tool_result_handler")


class ToolResultHandlerError(ValueError):
    """Raised when tool results cannot be processed."""


class ToolResultHandler:

    def __init__(self) -> None:
        logger.info("ToolResultHandler initialized.")

    @staticmethod
    def _validate_result(result: Any) -> ToolResult:

        if not isinstance(result, ToolResult):
            raise ToolResultHandlerError(
                "Expected a ToolResult instance."
            )

        if not isinstance(result.tool_name, str):
            raise ToolResultHandlerError(
                "Tool name must be a string."
            )

        if not result.tool_name.strip():
            raise ToolResultHandlerError(
                "Tool name cannot be empty."
            )

        if not isinstance(result.success, bool):
            raise ToolResultHandlerError(
                "Tool success must be a boolean."
            )

        if result.error is not None and not isinstance(
            result.error, str
        ):
            raise ToolResultHandlerError(
                "Tool error must be a string or None."
            )

        return result

    @staticmethod
    def _build_payload(result: ToolResult) -> dict[str, Any]:

        return {
            "success": result.success,
            "output": result.output,
            "error": result.error,
        }

    @staticmethod
    def _serialize_payload(payload: dict[str, Any]) -> str:

        try:
            return json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )

        except (TypeError, ValueError) as exc:

            raise ToolResultHandlerError(
                "Tool result is not JSON serializable."
            ) from exc

    def to_tool_message(
        self,
        result: ToolResult,
    ) -> dict[str, str]:
        """
        Convert one ToolResult into an Ollama tool message.
        """

        validated_result = self._validate_result(result)

        payload = self._build_payload(validated_result)

        content = self._serialize_payload(payload)

        message = {
            "role": "tool",
            "tool_name": validated_result.tool_name,
            "content": content,
        }

        logger.info(
            "Tool result converted: tool=%s success=%s",
            validated_result.tool_name,
            validated_result.success,
        )

        return message

    def to_tool_messages(
        self,
        results: list[ToolResult],
    ) -> list[dict[str, str]]:
        """
        Convert multiple ToolResult objects.

        Input validation is completed before conversion.
        The original execution order is preserved.
        """

        if not isinstance(results, list):
            raise ToolResultHandlerError(
                "Results must be provided as a list."
            )

        validated_results = []

        for index, result in enumerate(results):

            try:
                validated_results.append(
                    self._validate_result(result)
                )

            except ToolResultHandlerError as exc:

                raise ToolResultHandlerError(
                    f"Invalid ToolResult at index {index}: {exc}"
                ) from exc

        messages = []

        for result in validated_results:

            messages.append(
                self.to_tool_message(result)
            )

        logger.info(
            "Tool results converted: count=%d",
            len(messages),
        )

        return messages

    def handle_results(
        self,
        results: list[ToolResult],
    ) -> list[dict[str, str]]:
        """
        Public interface for converting execution results.
        """

        return self.to_tool_messages(results)