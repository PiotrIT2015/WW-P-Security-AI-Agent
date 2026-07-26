"""
Authentication and authorization package.
"""

from .auth import (
    AccountStatus,
    AuthenticationError,
    AuthenticationManager,
    InvalidCredentialsError,
    AccountDisabledError,
    SessionError,
    Session,
    User,
    UserContext,
    auth_manager,
)

from .permissions import (
    AccessDecision,
    AuthorizationResult,
    DocumentACL,
    InvalidPolicyError,
    PermissionManager,
    Role,
    RoleHierarchy,
    permission_manager,
)


__all__ = [

    # Authentication
    "AccountStatus",
    "AuthenticationError",
    "AuthenticationManager",
    "InvalidCredentialsError",
    "AccountDisabledError",
    "SessionError",
    "Session",
    "User",
    "UserContext",
    "auth_manager",

    # Authorization
    "AccessDecision",
    "AuthorizationResult",
    "DocumentACL",
    "InvalidPolicyError",
    "PermissionManager",
    "Role",
    "RoleHierarchy",
    "permission_manager",
]