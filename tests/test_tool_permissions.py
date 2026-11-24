"""
tests/test_tool_permissions.py

Tests for tool permission management.
"""

import pytest

from authentication.auth import UserContext

from tools.permissions import (
    ToolPermissionManager,
    ToolPermissionPolicy,
    ToolPolicyAlreadyExistsError,
    ToolPolicyNotFoundError,
)


@pytest.fixture
def manager():

    return ToolPermissionManager()


@pytest.fixture
def analyst():

    return UserContext(
        user_id="user-1",
        username="analyst",
        roles=frozenset({"analyst"}),
        groups=frozenset({"security"}),
        authenticated_at=1000.0,
        session_id="session-1",
    )


@pytest.fixture
def viewer():

    return UserContext(
        user_id="user-2",
        username="viewer",
        roles=frozenset({"viewer"}),
        groups=frozenset({"general"}),
        authenticated_at=1000.0,
        session_id="session-2",
    )


@pytest.fixture
def admin():

    return UserContext(
        user_id="admin-1",
        username="admin",
        roles=frozenset({"admin"}),
        groups=frozenset({"administrators"}),
        authenticated_at=1000.0,
        session_id="session-admin",
    )


# =============================================================================
# RBAC
# =============================================================================

def test_allowed_role(manager, analyst):

    manager.add_policy(
        "security_tool",
        ToolPermissionPolicy(
            allowed_roles=frozenset({"analyst"})
        ),
    )

    result = manager.authorize(
        "security_tool",
        analyst,
    )

    assert result.allowed is True
    assert result.reason == "role_allowed"


def test_denied_role(manager, viewer):

    manager.add_policy(
        "security_tool",
        ToolPermissionPolicy(
            allowed_roles=frozenset({"analyst"})
        ),
    )

    result = manager.authorize(
        "security_tool",
        viewer,
    )

    assert result.allowed is False


# =============================================================================
# Group-based access
# =============================================================================

def test_allowed_group(manager, analyst):

    manager.add_policy(
        "security_tool",
        ToolPermissionPolicy(
            allowed_groups=frozenset({"security"})
        ),
    )

    result = manager.authorize(
        "security_tool",
        analyst,
    )

    assert result.allowed is True
    assert result.reason == "group_allowed"


# =============================================================================
# User-based access
# =============================================================================

def test_allowed_user(manager, viewer):

    manager.add_policy(
        "private_tool",
        ToolPermissionPolicy(
            allowed_users=frozenset({"user-2"})
        ),
    )

    result = manager.authorize(
        "private_tool",
        viewer,
    )

    assert result.allowed is True
    assert result.reason == "user_allowed"


# =============================================================================
# Explicit deny
# =============================================================================

def test_denied_role_overrides_allowed_role(
    manager,
    analyst,
):

    manager.add_policy(
        "security_tool",
        ToolPermissionPolicy(
            allowed_roles=frozenset({"analyst"}),
            denied_roles=frozenset({"analyst"}),
        ),
    )

    result = manager.authorize(
        "security_tool",
        analyst,
    )

    assert result.allowed is False
    assert result.reason == "role_explicitly_denied"


def test_denied_group_overrides_allowed_group(
    manager,
    analyst,
):

    manager.add_policy(
        "security_tool",
        ToolPermissionPolicy(
            allowed_groups=frozenset({"security"}),
            denied_groups=frozenset({"security"}),
        ),
    )

    result = manager.authorize(
        "security_tool",
        analyst,
    )

    assert result.allowed is False
    assert result.reason == "group_explicitly_denied"


def test_denied_user_overrides_admin(
    manager,
    admin,
):

    manager.add_policy(
        "admin_tool",
        ToolPermissionPolicy(
            denied_users=frozenset({"admin-1"})
        ),
    )

    result = manager.authorize(
        "admin_tool",
        admin,
    )

    assert result.allowed is False
    assert result.reason == "user_explicitly_denied"


# =============================================================================
# Default deny
# =============================================================================

def test_missing_policy_denies_access(
    manager,
    analyst,
):

    result = manager.authorize(
        "unknown_tool",
        analyst,
    )

    assert result.allowed is False
    assert result.reason == "policy_not_found"


def test_no_matching_permission_denies_access(
    manager,
    viewer,
):

    manager.add_policy(
        "security_tool",
        ToolPermissionPolicy(
            allowed_roles=frozenset({"analyst"})
        ),
    )

    result = manager.authorize(
        "security_tool",
        viewer,
    )

    assert result.allowed is False
    assert result.reason == "no_matching_permission"


# =============================================================================
# Admin
# =============================================================================

def test_admin_allowed_when_not_explicitly_denied(
    manager,
    admin,
):

    manager.add_policy(
        "admin_tool",
        ToolPermissionPolicy(
            allowed_roles=frozenset({"analyst"})
        ),
    )

    result = manager.authorize(
        "admin_tool",
        admin,
    )

    assert result.allowed is True
    assert result.reason == "admin_access"


# =============================================================================
# Authenticated access
# =============================================================================

def test_authenticated_access(manager, viewer):

    manager.add_policy(
        "public_tool",
        ToolPermissionPolicy(
            allow_authenticated=True
        ),
    )

    result = manager.authorize(
        "public_tool",
        viewer,
    )

    assert result.allowed is True
    assert result.reason == "authenticated_access"


# =============================================================================
# Validation
# =============================================================================

def test_duplicate_policy_rejected(manager):

    policy = ToolPermissionPolicy(
        allowed_roles=frozenset({"analyst"})
    )

    manager.add_policy(
        "security_tool",
        policy,
    )

    with pytest.raises(ToolPolicyAlreadyExistsError):

        manager.add_policy(
            "security_tool",
            policy,
        )


def test_policy_can_be_replaced(manager):

    manager.add_policy(
        "security_tool",
        ToolPermissionPolicy(
            allowed_roles=frozenset({"analyst"})
        ),
    )

    manager.add_policy(
        "security_tool",
        ToolPermissionPolicy(
            allowed_roles=frozenset({"viewer"})
        ),
        replace=True,
    )

    assert "security_tool" in manager.list_policies()


def test_policy_not_found(manager):

    with pytest.raises(ToolPolicyNotFoundError):

        manager.get_policy(
            "unknown_tool"
        )


def test_case_insensitive_roles(manager, analyst):

    manager.add_policy(
        "security_tool",
        ToolPermissionPolicy(
            allowed_roles=frozenset({"ANALYST"})
        ),
    )

    result = manager.authorize(
        "security_tool",
        analyst,
    )

    assert result.allowed is True


def test_policy_rejects_invalid_role():

    with pytest.raises(ValueError):

        ToolPermissionPolicy(
            allowed_roles=frozenset({"analyst", ""})
        )