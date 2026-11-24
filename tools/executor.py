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

from tools.security import (
    ToolSecurityManager,
)

from tools.schemas import (
    validate_arguments,
    ToolValidationError,
)

from tools.permissions import (
    ToolPermissionManager,
    ToolPermissionPolicy,
)


logger = logging.getLogger("secure_rag.tools.executor")


class ToolExecutor:
    """
    Secure execution engine for registered agent tools.

    Security controls:

    - authenticated user required
    - centralized permission management
    - RBAC
    - user-based access
    - group-based access
    - explicit deny rules
    - default deny
    - argument validation
    - registered tools only
    - JSON-compatible output validation
    - controlled exception handling
    - execution logging
    """

    def __init__(
        self,
        registry: ToolRegistry,
        permission_manager: ToolPermissionManager | None = None,
        security_manager: ToolSecurityManager | None = None,
    ) -> None:

        if not isinstance(registry, ToolRegistry):
            raise TypeError(
                "registry must be an instance of ToolRegistry."
            )

        self.registry = registry

        self.permission_manager = (
            permission_manager
            if permission_manager is not None
            else ToolPermissionManager()
        )

        self.security_manager = (
            security_manager
            if security_manager is not None
            else ToolSecurityManager()
        )

        if not isinstance(
            self.permission_manager,
            ToolPermissionManager,
        ):
            raise TypeError(
                "permission_manager must be ToolPermissionManager."
            )

        if not isinstance(
            self.security_manager,
            ToolSecurityManager,
        ):
            raise TypeError(
                "security_manager must be ToolSecurityManager."
            )

        self._initialize_default_policies()

    # =========================================================================
    # Default policies
    # =========================================================================

    def _initialize_default_policies(self) -> None:
        """
        Create initial policies from registered tool definitions.

        Tools with required_roles:
            Access requires a matching role.

        Tools without required_roles:
            Any authenticated user can access.

        Explicit policies can later replace these defaults.
        """

        for tool_name in self.registry.list_names():

            tool = self.registry.get(tool_name)

            try:

                self.permission_manager.get_policy(
                    tool_name
                )

                continue

            except Exception:
                pass

            required_roles = frozenset(
                role.casefold()
                for role in tool.required_roles
            )

            policy = ToolPermissionPolicy(
                allowed_roles=required_roles,
                allow_authenticated=not bool(required_roles),
            )

            self.permission_manager.add_policy(
                tool_name,
                policy,
            )

    # =========================================================================
    # Authentication
    # =========================================================================

    @staticmethod
    def _validate_user(
        user: UserContext,
    ) -> bool:

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

    def _authorize(
        self,
        user: UserContext,
        tool_name: str,
    ) -> bool:

        result = self.permission_manager.authorize(
            tool_name,
            user,
        )

        return result.allowed

    # =========================================================================
    # Output validation
    # =========================================================================

    @staticmethod
    def _validate_output(output) -> bool:

        if not isinstance(output, (dict, list, str)):
            return False

        try:

            json.dumps(
                output,
                ensure_ascii=False,
                allow_nan=False,
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):

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

        1. Validate authentication.
        2. Validate tool name.
        3. Retrieve tool.
        4. Evaluate permission policy.
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

        # 4. Permission evaluation.

        permission = self.permission_manager.authorize(
            tool_name,
            user,
        )

        if not permission.allowed:

            logger.warning(
                "Tool access denied: "
                "user=%s tool=%s reason=%s",
                user.user_id,
                tool_name,
                permission.reason,
            )

            return ToolResult(
                tool_name=tool_name,
                success=False,
                error="Permission denied.",
                metadata={
                    "permission_reason": permission.reason,
                },
            )
        
        # 5. Basic argument type validation.

        if not isinstance(arguments, dict):

            return ToolResult(
                tool_name=tool_name,
                success=False,
                error="Tool arguments must be an object.",
            )
        
        # 5. Tool security checks.

        security = self.security_manager.evaluate(
            tool_name=tool_name,
            user_id=user.user_id,
            arguments=arguments,
        )

        if not security.allowed:

            logger.warning(
                "Tool security check denied: "
                "user=%s tool=%s reason=%s",
                user.user_id,
                tool_name,
                security.reason,
            )

            return ToolResult(
                tool_name=tool_name,
                success=False,
                error="Tool security validation failed.",
                metadata={
                    "security_reason": security.reason,
                },
            )

        # 5. Argument validation.

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
                    "Tool returned invalid output: tool=%s",
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
                    ),
                    "permission_reason": permission.reason,
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