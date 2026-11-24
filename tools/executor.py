"""
tools/executor.py

Secure tool execution engine.

WW-P Security AI Agent

Python 3.10+
"""

from __future__ import annotations

import json
import logging
import time

from authentication.auth import UserContext

from tools.base import ToolResult
from tools.registry import (
    ToolRegistry,
    ToolNotFoundError,
)
from tools.schemas import (
    validate_arguments,
    ToolValidationError,
)


logger = logging.getLogger("secure_rag.tools.executor")


class ToolExecutor:
    """
    Executes registered tools under authenticated user context.

    Security controls:

    - authenticated user required
    - user context validation
    - role-based access control
    - argument validation
    - registered tools only
    - JSON-compatible output validation
    - controlled exception handling
    - execution logging
    """

    def __init__(
        self,
        registry: ToolRegistry,
    ) -> None:

        if not isinstance(registry, ToolRegistry):
            raise TypeError(
                "registry must be an instance of ToolRegistry."
            )

        self.registry = registry

    # =========================================================================
    # Authentication
    # =========================================================================

    @staticmethod
    def _validate_user(
        user: UserContext,
    ) -> bool:
        """
        Validate authenticated user context.
        """

        if not isinstance(user, UserContext):
            return False

        if not isinstance(user.user_id, str):
            return False

        if not user.user_id.strip():
            return False

        if not isinstance(user.username, str):
            return False

        if not user.username.strip():
            return False

        if not isinstance(user.session_id, str):
            return False

        if not user.session_id.strip():
            return False

        return True

    # =========================================================================
    # Authorization
    # =========================================================================

    @staticmethod
    def _authorize(
        user: UserContext,
        required_roles: set[str],
    ) -> bool:
        """
        Check whether user can execute a tool.

        Security policy:

        - Invalid users are denied.
        - Admin has access to all registered tools.
        - Empty required_roles means any authenticated user.
        - Otherwise, at least one required role must match.
        """

        if not ToolExecutor._validate_user(user):
            return False

        if not isinstance(required_roles, set):
            return False

        user_roles = {
            role.casefold()
            for role in user.roles
            if isinstance(role, str)
        }

        required = {
            role.casefold()
            for role in required_roles
            if isinstance(role, str)
        }

        # Explicit admin override.
        if "admin" in user_roles:
            return True

        # Empty role requirements mean authenticated access.
        if not required:
            return True

        return bool(
            user_roles.intersection(required)
        )

    # =========================================================================
    # Output validation
    # =========================================================================

    @staticmethod
    def _validate_output(output) -> bool:
        """
        Check whether tool output can safely be serialized to JSON.

        Supported top-level types:
        - dict
        - list
        - str

        NaN and Infinity are rejected.
        """

        if not isinstance(output, (dict, list, str)):
            return False

        try:
            json.dumps(
                output,
                ensure_ascii=False,
                allow_nan=False,
            )

        except (TypeError, ValueError, OverflowError):
            return False

        return True

    # =========================================================================
    # Execution
    # =========================================================================

    def execute(
        self,
        tool_name: str,
        arguments: dict,
        user: UserContext,
    ) -> ToolResult:
        """
        Execute a registered tool.

        Execution sequence:

        1. Validate user.
        2. Validate tool name.
        3. Retrieve tool.
        4. Check permissions.
        5. Validate arguments.
        6. Execute tool.
        7. Validate output.
        8. Return ToolResult.
        """

        # 1. Authentication.

        if not self._validate_user(user):

            logger.warning(
                "Tool execution denied: invalid user context."
            )

            return ToolResult(
                tool_name=(
                    tool_name
                    if isinstance(tool_name, str)
                    else "unknown"
                ),
                success=False,
                error="Authentication required.",
            )

        # 2. Tool name validation.

        if (
            not isinstance(tool_name, str)
            or not tool_name.strip()
        ):

            return ToolResult(
                tool_name="unknown",
                success=False,
                error="Invalid tool name.",
            )

        # 3. Registry lookup.

        try:

            tool = self.registry.get(tool_name)

        except ToolNotFoundError:

            logger.warning(
                "Unknown tool requested: user=%s tool=%s",
                user.user_id,
                tool_name,
            )

            return ToolResult(
                tool_name=tool_name,
                success=False,
                error="Tool not found.",
            )

        # 4. Authorization.

        if not self._authorize(
            user,
            tool.required_roles,
        ):

            logger.warning(
                "Tool access denied: user=%s tool=%s",
                user.user_id,
                tool_name,
            )

            return ToolResult(
                tool_name=tool_name,
                success=False,
                error="Permission denied.",
            )

        # 5. Argument validation.

        if not isinstance(arguments, dict):

            return ToolResult(
                tool_name=tool_name,
                success=False,
                error="Tool arguments must be an object.",
            )

        try:

            validated_arguments = validate_arguments(
                arguments,
                tool.input_schema,
            )

        except ToolValidationError as error:

            logger.warning(
                "Tool argument validation failed: "
                "tool=%s error=%s",
                tool_name,
                str(error),
            )

            return ToolResult(
                tool_name=tool_name,
                success=False,
                error="Invalid tool arguments.",
            )

        # 6. Execute tool.

        start_time = time.perf_counter()

        try:

            output = tool.execute(
                validated_arguments,
                user,
            )

            execution_time = (
                time.perf_counter() - start_time
            )

            # 7. Validate output.

            if not self._validate_output(output):

                logger.error(
                    "Tool returned invalid output: "
                    "tool=%s",
                    tool_name,
                )

                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    error="Tool returned an invalid result.",
                    metadata={
                        "execution_time_ms": round(
                            execution_time * 1000,
                            2,
                        )
                    },
                )

            logger.info(
                "Tool executed successfully: "
                "user=%s tool=%s duration_ms=%.2f",
                user.user_id,
                tool_name,
                execution_time * 1000,
            )

            return ToolResult(
                tool_name=tool_name,
                success=True,
                output=output,
                metadata={
                    "execution_time_ms": round(
                        execution_time * 1000,
                        2,
                    )
                },
            )

        except Exception as error:

            execution_time = (
                time.perf_counter() - start_time
            )

            logger.error(
                "Tool execution failed: "
                "tool=%s error_type=%s",
                tool_name,
                type(error).__name__,
            )

            return ToolResult(
                tool_name=tool_name,
                success=False,
                error="Tool execution failed.",
                metadata={
                    "execution_time_ms": round(
                        execution_time * 1000,
                        2,
                    )
                },
            )