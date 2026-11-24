"""
tools/security/manager.py

Security management for WW-P Security AI Agent tools.

Python 3.10+
"""

from __future__ import annotations

import json
import logging
import threading
import time

from collections import defaultdict, deque
from dataclasses import dataclass


logger = logging.getLogger("secure_rag.tools.security")


# =============================================================================
# Security decision
# =============================================================================

@dataclass(frozen=True, slots=True)
class ToolSecurityDecision:
    allowed: bool
    reason: str
    tool_name: str
    user_id: str


# =============================================================================
# Security configuration
# =============================================================================

@dataclass(frozen=True, slots=True)
class ToolSecurityConfig:

    max_calls: int = 60

    window_seconds: int = 60

    max_argument_bytes: int = 65536

    max_nesting_depth: int = 8

    def __post_init__(self):

        if (
            not isinstance(self.max_calls, int)
            or isinstance(self.max_calls, bool)
            or self.max_calls < 1
        ):
            raise ValueError(
                "max_calls must be a positive integer."
            )

        if (
            not isinstance(self.window_seconds, int)
            or isinstance(self.window_seconds, bool)
            or self.window_seconds < 1
        ):
            raise ValueError(
                "window_seconds must be a positive integer."
            )

        if (
            not isinstance(self.max_argument_bytes, int)
            or isinstance(self.max_argument_bytes, bool)
            or self.max_argument_bytes < 1
        ):
            raise ValueError(
                "max_argument_bytes must be a positive integer."
            )

        if (
            not isinstance(self.max_nesting_depth, int)
            or isinstance(self.max_nesting_depth, bool)
            or self.max_nesting_depth < 1
        ):
            raise ValueError(
                "max_nesting_depth must be a positive integer."
            )


# =============================================================================
# Security manager
# =============================================================================

class ToolSecurityManager:
    """
    Security layer for tool execution.

    Controls:

    - tool blocking
    - rate limiting
    - argument size validation
    - argument nesting validation
    - JSON serialization validation
    """

    def __init__(
        self,
        config: ToolSecurityConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else ToolSecurityConfig()
        )

        if not isinstance(self.config, ToolSecurityConfig):
            raise TypeError(
                "config must be ToolSecurityConfig."
            )

        self._blocked_tools: set[str] = set()

        self._calls = defaultdict(deque)

        self._lock = threading.Lock()

    # =========================================================================
    # Tool management
    # =========================================================================

    @staticmethod
    def _normalize_tool_name(tool_name: str) -> str:

        if (
            not isinstance(tool_name, str)
            or not tool_name.strip()
        ):
            raise ValueError(
                "Tool name cannot be empty."
            )

        return tool_name.strip().casefold()

    def block_tool(self, tool_name: str) -> None:

        name = self._normalize_tool_name(tool_name)

        with self._lock:
            self._blocked_tools.add(name)

        logger.warning(
            "Tool blocked by security manager: tool=%s",
            name,
        )

    def unblock_tool(self, tool_name: str) -> None:

        name = self._normalize_tool_name(tool_name)

        with self._lock:
            self._blocked_tools.discard(name)

        logger.info(
            "Tool unblocked: tool=%s",
            name,
        )

    def is_blocked(self, tool_name: str) -> bool:

        name = self._normalize_tool_name(tool_name)

        with self._lock:
            return name in self._blocked_tools

    # =========================================================================
    # Argument validation
    # =========================================================================

    def _validate_nesting(
        self,
        value,
        depth: int = 0,
    ) -> bool:
        """
        Reject excessively nested input structures.
        """

        if depth > self.config.max_nesting_depth:
            return False

        if isinstance(value, dict):

            for key, item in value.items():

                if not isinstance(key, str):
                    return False

                if not self._validate_nesting(
                    item,
                    depth + 1,
                ):
                    return False

        elif isinstance(value, list):

            for item in value:

                if not self._validate_nesting(
                    item,
                    depth + 1,
                ):
                    return False

        return True

    def validate_arguments(
        self,
        arguments: dict,
    ) -> tuple[bool, str]:
        """
        Validate arguments before tool execution.
        """

        if not isinstance(arguments, dict):
            return False, "arguments_not_object"

        if not self._validate_nesting(arguments):
            return False, "maximum_nesting_depth_exceeded"

        try:

            serialized = json.dumps(
                arguments,
                ensure_ascii=False,
                allow_nan=False,
            )

            argument_size = len(
                serialized.encode("utf-8")
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):

            return False, "invalid_json_arguments"

        if argument_size > self.config.max_argument_bytes:

            return False, "argument_size_exceeded"

        return True, "arguments_valid"

    # =========================================================================
    # Rate limiting
    # =========================================================================

    def _check_rate_limit(
        self,
        tool_name: str,
        user_id: str,
    ) -> bool:
        """
        Apply a sliding-window rate limit.

        Limits are applied independently per user and tool.
        """

        now = time.monotonic()

        key = (
            user_id,
            tool_name,
        )

        with self._lock:

            calls = self._calls[key]

            cutoff = (
                now - self.config.window_seconds
            )

            while calls and calls[0] <= cutoff:
                calls.popleft()

            if len(calls) >= self.config.max_calls:
                return False

            calls.append(now)

            return True

    # =========================================================================
    # Security evaluation
    # =========================================================================

    def evaluate(
        self,
        tool_name: str,
        user_id: str,
        arguments: dict,
    ) -> ToolSecurityDecision:
        """
        Evaluate whether a tool invocation is safe to execute.
        """

        safe_user_id = (
            user_id
            if isinstance(user_id, str)
            else "unknown"
        )

        try:
            name = self._normalize_tool_name(tool_name)

        except ValueError:

            return ToolSecurityDecision(
                False,
                "invalid_tool_name",
                "unknown",
                safe_user_id,
            )

        def decision(
            allowed: bool,
            reason: str,
        ) -> ToolSecurityDecision:

            result = ToolSecurityDecision(
                allowed=allowed,
                reason=reason,
                tool_name=name,
                user_id=safe_user_id,
            )

            logger.info(
                "Tool security decision: "
                "user=%s tool=%s allowed=%s reason=%s",
                safe_user_id,
                name,
                allowed,
                reason,
            )

            return result

        if (
            not isinstance(user_id, str)
            or not user_id.strip()
        ):

            return decision(
                False,
                "invalid_user_id",
            )

        # 1. Blocked tool check.

        if self.is_blocked(name):

            return decision(
                False,
                "tool_blocked",
            )

        # 2. Argument validation.

        valid, reason = self.validate_arguments(
            arguments
        )

        if not valid:

            return decision(
                False,
                reason,
            )

        # 3. Rate limiting.

        if not self._check_rate_limit(
            name,
            user_id,
        ):

            return decision(
                False,
                "rate_limit_exceeded",
            )

        return decision(
            True,
            "security_checks_passed",
        )