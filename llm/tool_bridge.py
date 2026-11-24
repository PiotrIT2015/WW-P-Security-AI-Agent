"""
llm/tool_bridge.py

Connects Ollama tool calls with the existing ToolExecutor.

Responsibilities:
- Accept parsed ToolCall objects.
- Forward calls to ToolExecutor.
- Preserve execution order.
- Support multiple tool calls.
- Return standardized ToolResult objects.
- Handle unexpected executor errors.

Security:
- Never executes tools directly.
- Never bypasses ToolExecutor.
- Does not implement its own permission system.
- Does not bypass ToolSecurityManager.
- Does not trust LLM-generated arguments.

Compatible with Python 3.10+.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from authentication.auth import UserContext
from tools.base import ToolResult
from .tool_calling import (
    ToolCall,
    ToolCallParseError,
    parse_tool_calls,
)


logger = logging.getLogger("secure_rag.llm.tool_bridge")


class ToolExecutionBridgeError(ValueError):
    """Raised when the bridge receives invalid input."""


class ToolExecutorProtocol(Protocol):
    """Minimal interface required from ToolExecutor."""

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        user: UserContext,
    ) -> ToolResult:
        ...


class ToolExecutionBridge:
    """
    Bridge between LLM-generated tool calls and ToolExecutor.

    All actual tool execution is delegated to ToolExecutor.
    """

    def __init__(
        self,
        executor: ToolExecutorProtocol,
    ) -> None:

        if executor is None:
            raise ValueError(
                "ToolExecutor cannot be None."
            )

        if not callable(
            getattr(executor, "execute", None)
        ):
            raise TypeError(
                "Executor must provide a callable execute() method."
            )

        self.executor = executor

    # =========================================================
    # Input validation
    # =========================================================

    @staticmethod
    def _validate_tool_calls(
        tool_calls: Any,
    ) -> list[ToolCall]:
        """
        Validate a collection of parsed tool calls.

        Only ToolCall instances are accepted.
        """

        if not isinstance(tool_calls, list):
            raise ToolExecutionBridgeError(
                "tool_calls must be a list."
            )

        for index, call in enumerate(tool_calls):

            if not isinstance(call, ToolCall):
                raise ToolExecutionBridgeError(
                    f"Invalid ToolCall at index {index}."
                )

            if not isinstance(call.name, str) or not call.name:
                raise ToolExecutionBridgeError(
                    f"Invalid tool name at index {index}."
                )

            if not isinstance(call.arguments, dict):
                raise ToolExecutionBridgeError(
                    f"Invalid arguments at index {index}."
                )

        return tool_calls

    # =========================================================
    # Single tool execution
    # =========================================================

    def execute_tool_call(
        self,
        tool_call: ToolCall,
        user: UserContext,
    ) -> ToolResult:
        """
        Execute one parsed tool call through ToolExecutor.

        Unexpected exceptions are converted into a generic
        failed ToolResult. Internal exception details are
        logged, not exposed to the caller.
        """

        if not isinstance(tool_call, ToolCall):
            raise ToolExecutionBridgeError(
                "Expected a ToolCall instance."
            )

        if not isinstance(tool_call.arguments, dict):
            raise ToolExecutionBridgeError(
                "Tool arguments must be an object."
            )

        try:
            result = self.executor.execute(
                tool_name=tool_call.name,
                arguments=dict(tool_call.arguments),
                user=user,
            )

        except Exception as exc:

            logger.exception(
                "Unexpected ToolExecutor error: tool=%s",
                tool_call.name,
            )

            return ToolResult(
                tool_name=tool_call.name,
                success=False,
                error="Tool execution failed.",
                metadata={
                    "bridge_error": type(exc).__name__,
                },
            )

        if not isinstance(result, ToolResult):

            logger.error(
                "ToolExecutor returned an invalid result: tool=%s",
                tool_call.name,
            )

            return ToolResult(
                tool_name=tool_call.name,
                success=False,
                error="Invalid tool execution result.",
                metadata={
                    "bridge_error": "InvalidToolResult",
                },
            )

        logger.info(
            "Tool call completed: tool=%s success=%s",
            tool_call.name,
            result.success,
        )

        return result

    # =========================================================
    # Multiple tool calls
    # =========================================================

    def execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        user: UserContext,
    ) -> list[ToolResult]:
        """
        Execute multiple tool calls sequentially.

        Execution order is preserved.

        A failed tool call does not prevent subsequent
        calls from being processed.

        Permission and security failures are returned
        by ToolExecutor as ToolResult objects.
        """

        validated_calls = self._validate_tool_calls(
            tool_calls
        )

        results: list[ToolResult] = []

        for tool_call in validated_calls:

            result = self.execute_tool_call(
                tool_call=tool_call,
                user=user,
            )

            results.append(result)

        logger.info(
            "Tool execution batch completed: calls=%d successful=%d failed=%d",
            len(results),
            sum(1 for item in results if item.success),
            sum(1 for item in results if not item.success),
        )

        return results

    # =========================================================
    # Full assistant message processing
    # =========================================================

    def execute_message(
        self,
        message: dict[str, Any],
        user: UserContext,
    ) -> list[ToolResult]:
        """
        Parse an Ollama assistant message and execute
        its tool calls through ToolExecutor.

        A message without tool calls returns an empty list.

        Malformed assistant messages raise ToolCallParseError.
        """

        try:
            tool_calls = parse_tool_calls(message)

        except ToolCallParseError:
            logger.warning(
                "Unable to parse assistant tool calls."
            )
            raise

        return self.execute_tool_calls(
            tool_calls=tool_calls,
            user=user,
        )