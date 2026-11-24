"""
tools/permissions.py

Tool-level permission management.

WW-P Security AI Agent

Python 3.10+
"""

from __future__ import annotations

import logging

from dataclasses import dataclass, field

from authentication.auth import UserContext


logger = logging.getLogger("secure_rag.tools.permissions")


# =============================================================================
# Exceptions
# =============================================================================

class ToolPermissionError(Exception):
    """Base exception for tool permission errors."""


class ToolPolicyAlreadyExistsError(ToolPermissionError):
    """Raised when a policy already exists."""


class ToolPolicyNotFoundError(ToolPermissionError):
    """Raised when a policy does not exist."""


# =============================================================================
# Permission decision
# =============================================================================

@dataclass(frozen=True, slots=True)
class PermissionDecision:
    """
    Result of a tool authorization decision.
    """

    allowed: bool

    reason: str

    tool_name: str

    user_id: str


# =============================================================================
# Tool access policy
# =============================================================================

@dataclass(frozen=True, slots=True)
class ToolPermissionPolicy:
    """
    Access policy for a single tool.

    Explicit deny rules always take precedence.
    """

    allowed_roles: frozenset[str] = field(
        default_factory=frozenset
    )

    allowed_users: frozenset[str] = field(
        default_factory=frozenset
    )

    allowed_groups: frozenset[str] = field(
        default_factory=frozenset
    )

    denied_roles: frozenset[str] = field(
        default_factory=frozenset
    )

    denied_users: frozenset[str] = field(
        default_factory=frozenset
    )

    denied_groups: frozenset[str] = field(
        default_factory=frozenset
    )

    allow_authenticated: bool = False

    def __post_init__(self) -> None:
        """
        Validate and normalize policy configuration.
        """

        fields = (
            "allowed_roles",
            "allowed_users",
            "allowed_groups",
            "denied_roles",
            "denied_users",
            "denied_groups",
        )

        for field_name in fields:

            values = getattr(self, field_name)

            if not isinstance(values, (set, frozenset)):
                raise TypeError(
                    f"{field_name} must be a set or frozenset."
                )

            normalized = frozenset(
                value.strip().casefold()
                for value in values
                if isinstance(value, str) and value.strip()
            )

            if len(normalized) != len(values):
                raise ValueError(
                    f"{field_name} contains invalid values."
                )

            object.__setattr__(
                self,
                field_name,
                normalized,
            )

        if not isinstance(self.allow_authenticated, bool):
            raise TypeError(
                "allow_authenticated must be a boolean."
            )


# =============================================================================
# Permission manager
# =============================================================================

class ToolPermissionManager:
    """
    Central permission manager for agent tools.

    Security rules:

    1. Invalid users are denied.
    2. Missing policies are denied.
    3. Explicit deny rules override allow rules.
    4. Admin does not bypass explicit deny rules.
    5. Access requires a matching allow rule.
    """

    def __init__(self) -> None:

        self._policies: dict[
            str,
            ToolPermissionPolicy
        ] = {}

    # =========================================================================
    # Validation
    # =========================================================================

    @staticmethod
    def _validate_tool_name(tool_name: str) -> None:

        if (
            not isinstance(tool_name, str)
            or not tool_name.strip()
        ):
            raise ValueError(
                "Tool name cannot be empty."
            )

    @staticmethod
    def _validate_user(user: UserContext) -> bool:

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
    # Policy management
    # =========================================================================

    def add_policy(
        self,
        tool_name: str,
        policy: ToolPermissionPolicy,
        replace: bool = False,
    ) -> None:
        """
        Register a permission policy for a tool.
        """

        self._validate_tool_name(tool_name)

        if not isinstance(policy, ToolPermissionPolicy):
            raise TypeError(
                "policy must be ToolPermissionPolicy."
            )

        normalized_name = tool_name.strip().casefold()

        if (
            normalized_name in self._policies
            and not replace
        ):
            raise ToolPolicyAlreadyExistsError(
                f"Policy already exists: {tool_name}"
            )

        self._policies[normalized_name] = policy

        logger.info(
            "Tool permission policy registered: tool=%s",
            normalized_name,
        )

    def get_policy(
        self,
        tool_name: str,
    ) -> ToolPermissionPolicy:

        self._validate_tool_name(tool_name)

        normalized_name = tool_name.strip().casefold()

        policy = self._policies.get(normalized_name)

        if policy is None:
            raise ToolPolicyNotFoundError(
                f"Policy not found: {tool_name}"
            )

        return policy

    def remove_policy(
        self,
        tool_name: str,
    ) -> bool:

        self._validate_tool_name(tool_name)

        normalized_name = tool_name.strip().casefold()

        if normalized_name not in self._policies:
            raise ToolPolicyNotFoundError(
                f"Policy not found: {tool_name}"
            )

        del self._policies[normalized_name]

        logger.info(
            "Tool permission policy removed: tool=%s",
            normalized_name,
        )

        return True

    def list_policies(self) -> list[str]:

        return sorted(self._policies.keys())

    # =========================================================================
    # Authorization
    # =========================================================================

    def authorize(
        self,
        tool_name: str,
        user: UserContext,
    ) -> PermissionDecision:
        """
        Evaluate tool access policy.

        Deny rules are evaluated before allow rules.
        """

        safe_user_id = (
            user.user_id
            if isinstance(user, UserContext)
            and isinstance(user.user_id, str)
            else "unknown"
        )

        def decision(
            allowed: bool,
            reason: str,
        ) -> PermissionDecision:

            result = PermissionDecision(
                allowed=allowed,
                reason=reason,
                tool_name=tool_name,
                user_id=safe_user_id,
            )

            logger.info(
                "Tool permission decision: "
                "user=%s tool=%s allowed=%s reason=%s",
                safe_user_id,
                tool_name,
                allowed,
                reason,
            )

            return result

        # Authentication validation.

        if not self._validate_user(user):

            return decision(
                False,
                "invalid_user",
            )

        # Policy lookup.

        try:

            policy = self.get_policy(tool_name)

        except ToolPolicyNotFoundError:

            return decision(
                False,
                "policy_not_found",
            )

        user_id = user.user_id.casefold()

        roles = {
            role.casefold()
            for role in user.roles
            if isinstance(role, str)
        }

        groups = {
            group.casefold()
            for group in user.groups
            if isinstance(group, str)
        }

        # -------------------------------------------------------------
        # 1. Explicit deny rules
        # -------------------------------------------------------------

        if user_id in policy.denied_users:

            return decision(
                False,
                "user_explicitly_denied",
            )

        if roles.intersection(policy.denied_roles):

            return decision(
                False,
                "role_explicitly_denied",
            )

        if groups.intersection(policy.denied_groups):

            return decision(
                False,
                "group_explicitly_denied",
            )

        # -------------------------------------------------------------
        # 2. Explicit allow rules
        # -------------------------------------------------------------

        if user_id in policy.allowed_users:

            return decision(
                True,
                "user_allowed",
            )

        if groups.intersection(policy.allowed_groups):

            return decision(
                True,
                "group_allowed",
            )

        if roles.intersection(policy.allowed_roles):

            return decision(
                True,
                "role_allowed",
            )

        # -------------------------------------------------------------
        # 3. Administrator access
        # -------------------------------------------------------------

        if "admin" in roles:

            return decision(
                True,
                "admin_access",
            )

        # -------------------------------------------------------------
        # 4. Authenticated access
        # -------------------------------------------------------------

        if policy.allow_authenticated:

            return decision(
                True,
                "authenticated_access",
            )

        # -------------------------------------------------------------
        # 5. Default deny
        # -------------------------------------------------------------

        return decision(
            False,
            "no_matching_permission",
        )