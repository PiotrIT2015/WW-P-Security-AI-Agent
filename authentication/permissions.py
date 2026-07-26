"""
authentication/permissions.py

Authorization layer for Secure RAG.

Python 3.12+

Provides:

- RBAC
- document ACL
- group-based access
- owner-based access
- classification-based access
- explicit deny
- permission evaluation

Authentication is handled separately by auth.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from authentication.auth import UserContext


# =============================================================================
# Exceptions
# =============================================================================


class PermissionError(Exception):
    """Base authorization exception."""

    pass


class InvalidPolicyError(PermissionError):
    """Raised when an ACL policy is invalid."""

    pass


# =============================================================================
# Access decision
# =============================================================================


class AccessDecision(str, Enum):
    """
    Authorization decision.
    """

    ALLOW = "allow"
    DENY = "deny"


# =============================================================================
# Roles
# =============================================================================


class Role:
    """
    Standard Secure RAG roles.
    """

    ADMIN = "admin"

    ANALYST = "analyst"

    USER = "user"

    READER = "reader"

    SERVICE = "service"


# =============================================================================
# ACL policy
# =============================================================================


@dataclass(slots=True)
class DocumentACL:
    """
    Access control list attached to a document.

    Empty allow_* collections mean that the corresponding
    restriction is not explicitly required.

    deny_* always takes precedence.
    """

    document_id: str

    allowed_users: set[str] = field(
        default_factory=set
    )

    allowed_groups: set[str] = field(
        default_factory=set
    )

    allowed_roles: set[str] = field(
        default_factory=set
    )

    denied_users: set[str] = field(
        default_factory=set
    )

    denied_groups: set[str] = field(
        default_factory=set
    )

    denied_roles: set[str] = field(
        default_factory=set
    )

    owner_id: str | None = None

    classification: str = "internal"

    min_role: str | None = None


# =============================================================================
# Authorization result
# =============================================================================


@dataclass(slots=True, frozen=True)
class AuthorizationResult:
    """
    Result of authorization evaluation.

    reason is useful for auditing and debugging.
    It should not normally be exposed directly to an
    untrusted client.
    """

    decision: AccessDecision

    reason: str

    document_id: str

    user_id: str

    @property
    def allowed(self) -> bool:
        return self.decision == AccessDecision.ALLOW


# =============================================================================
# Role hierarchy
# =============================================================================


class RoleHierarchy:
    """
    Defines role privilege levels.

    Higher number = greater privilege.
    """

    DEFAULT_LEVELS = {
        Role.USER: 10,
        Role.READER: 20,
        Role.ANALYST: 30,
        Role.SERVICE: 40,
        Role.ADMIN: 100,
    }

    def __init__(
        self,
        levels: Mapping[str, int] | None = None,
    ) -> None:

        self._levels = dict(
            levels
            if levels is not None
            else self.DEFAULT_LEVELS
        )

    def level(
        self,
        role: str,
    ) -> int:

        return self._levels.get(
            role,
            0,
        )

    def satisfies(
        self,
        roles: set[str] | frozenset[str],
        minimum_role: str,
    ) -> bool:

        required_level = self.level(
            minimum_role
        )

        if required_level == 0:

            return False

        return any(
            self.level(role) >= required_level
            for role in roles
        )


# =============================================================================
# Permission manager
# =============================================================================


class PermissionManager:
    """
    Central authorization engine for Secure RAG.

    Authorization is performed AFTER vector retrieval.

    This prevents FAISS from becoming an authorization boundary.
    """

    PUBLIC_CLASSIFICATION = "public"

    CLASSIFICATION_LEVELS = {
        "public": 0,
        "internal": 10,
        "confidential": 20,
        "restricted": 30,
        "secret": 40,
    }

    def __init__(
        self,
        role_hierarchy: RoleHierarchy | None = None,
    ) -> None:

        self.role_hierarchy = (
            role_hierarchy
            or RoleHierarchy()
        )

        self._policies: dict[
            str,
            DocumentACL,
        ] = {}

    # =========================================================================
    # Policy registration
    # =========================================================================

    def add_policy(
        self,
        policy: DocumentACL,
    ) -> None:

        self._validate_policy(
            policy
        )

        self._policies[
            policy.document_id
        ] = policy

    # =========================================================================
    # Policy lookup
    # =========================================================================

    def get_policy(
        self,
        document_id: str,
    ) -> DocumentACL | None:

        return self._policies.get(
            document_id
        )

    # =========================================================================
    # Authorization
    # =========================================================================

    def authorize(
        self,
        user: UserContext,
        document_id: str,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> AuthorizationResult:
        """
        Determine whether a user can access a document.

        Order:

        1. validate identity
        2. explicit deny
        3. admin override
        4. owner
        5. policy
        6. classification
        7. default deny
        """

        if not user.user_id:

            return self._deny(
                user,
                document_id,
                "missing user identity",
            )

        policy = self._policies.get(
            document_id
        )

        metadata = metadata or {}

        # ---------------------------------------------------------
        # Explicit deny
        # ---------------------------------------------------------

        if policy is not None:

            if user.user_id in policy.denied_users:

                return self._deny(
                    user,
                    document_id,
                    "user explicitly denied",
                )

            if (
                user.groups
                & policy.denied_groups
            ):

                return self._deny(
                    user,
                    document_id,
                    "user group explicitly denied",
                )

            if (
                user.roles
                & policy.denied_roles
            ):

                return self._deny(
                    user,
                    document_id,
                    "user role explicitly denied",
                )

        # ---------------------------------------------------------
        # Administrator
        # ---------------------------------------------------------

        if Role.ADMIN in user.roles:

            return self._allow(
                user,
                document_id,
                "administrator role",
            )

        # ---------------------------------------------------------
        # Owner
        # ---------------------------------------------------------

        owner_id = (
            policy.owner_id
            if policy is not None
            else metadata.get("owner_id")
        )

        if (
            owner_id is not None
            and str(owner_id) == user.user_id
        ):

            return self._allow(
                user,
                document_id,
                "document owner",
            )

        # ---------------------------------------------------------
        # Classification
        # ---------------------------------------------------------

        classification = str(
            (
                policy.classification
                if policy is not None
                else metadata.get(
                    "classification",
                    "internal",
                )
            )
        ).casefold()

        classification_result = (
            self._check_classification(
                user,
                classification,
            )
        )

        if not classification_result:

            return self._deny(
                user,
                document_id,
                "classification restriction",
            )

        # ---------------------------------------------------------
        # No policy
        # ---------------------------------------------------------

        if policy is None:

            # Public documents can be accessed
            # by authenticated users.

            if classification == self.PUBLIC_CLASSIFICATION:

                return self._allow(
                    user,
                    document_id,
                    "public document",
                )

            return self._deny(
                user,
                document_id,
                "no access policy",
            )

        # ---------------------------------------------------------
        # Minimum role
        # ---------------------------------------------------------

        if policy.min_role is not None:

            if not self.role_hierarchy.satisfies(
                user.roles,
                policy.min_role,
            ):

                return self._deny(
                    user,
                    document_id,
                    "minimum role not satisfied",
                )

        # ---------------------------------------------------------
        # Explicit user allow
        # ---------------------------------------------------------

        if user.user_id in policy.allowed_users:

            return self._allow(
                user,
                document_id,
                "explicit user permission",
            )

        # ---------------------------------------------------------
        # Group allow
        # ---------------------------------------------------------

        if user.groups & policy.allowed_groups:

            return self._allow(
                user,
                document_id,
                "group permission",
            )

        # ---------------------------------------------------------
        # Role allow
        # ---------------------------------------------------------

        if user.roles & policy.allowed_roles:

            return self._allow(
                user,
                document_id,
                "role permission",
            )

        # ---------------------------------------------------------
        # Default deny
        # ---------------------------------------------------------

        return self._deny(
            user,
            document_id,
            "default deny",
        )

    # =========================================================================
    # Batch authorization
    # =========================================================================

    def filter_documents(
        self,
        user: UserContext,
        document_ids: list[str],
        *,
        metadata: Mapping[str, Mapping[str, object]]
        | None = None,
    ) -> list[str]:
        """
        Filter a list of document IDs according to ACL.

        Only authorized documents are returned.
        """

        result: list[str] = []

        metadata = metadata or {}

        for document_id in document_ids:

            decision = self.authorize(

                user,

                document_id,

                metadata=metadata.get(
                    document_id
                ),

            )

            if decision.allowed:

                result.append(
                    document_id
                )

        return result

    # =========================================================================
    # Batch chunk filtering
    # =========================================================================

    def filter_chunks(
        self,
        user: UserContext,
        chunks: list[Mapping[str, object]],
    ) -> list[Mapping[str, object]]:
        """
        Filter retrieved chunks.

        Expected chunk metadata:

            {
                "chunk_id": "...",
                "document_id": "...",
                "classification": "...",
                "owner_id": "..."
            }
        """

        allowed: list[
            Mapping[str, object]
        ] = []

        for chunk in chunks:

            document_id = str(
                chunk.get(
                    "document_id",
                    "",
                )
            )

            if not document_id:

                continue

            decision = self.authorize(

                user,

                document_id,

                metadata=chunk,

            )

            if decision.allowed:

                allowed.append(
                    chunk
                )

        return allowed

    # =========================================================================
    # Classification
    # =========================================================================

    def _check_classification(
        self,
        user: UserContext,
        classification: str,
    ) -> bool:

        classification = (
            classification.casefold()
        )

        required = self.CLASSIFICATION_LEVELS.get(
            classification
        )

        if required is None:

            # Unknown classification must not
            # accidentally become accessible.

            return False

        # Admin is handled separately.

        if Role.ADMIN in user.roles:

            return True

        # Public is available to any authenticated
        # user.

        if required == 0:

            return True

        # Analysts can access confidential data.
        if (
            Role.ANALYST in user.roles
            and required <= 20
        ):

            return True

        # Readers/users are limited to internal.
        if required <= 10:

            return bool(
                user.roles
                & {
                    Role.USER,
                    Role.READER,
                    Role.ANALYST,
                    Role.SERVICE,
                }
            )

        # Service role can access restricted
        # data only when explicitly permitted
        # by policy.

        return False

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _allow(
        user: UserContext,
        document_id: str,
        reason: str,
    ) -> AuthorizationResult:

        return AuthorizationResult(

            decision=AccessDecision.ALLOW,

            reason=reason,

            document_id=document_id,

            user_id=user.user_id,

        )

    @staticmethod
    def _deny(
        user: UserContext,
        document_id: str,
        reason: str,
    ) -> AuthorizationResult:

        return AuthorizationResult(

            decision=AccessDecision.DENY,

            reason=reason,

            document_id=document_id,

            user_id=user.user_id,

        )

    # =========================================================================
    # Policy validation
    # =========================================================================

    @staticmethod
    def _validate_policy(
        policy: DocumentACL,
    ) -> None:

        if not policy.document_id:

            raise InvalidPolicyError(
                "document_id cannot be empty."
            )

        valid_classifications = {
            "public",
            "internal",
            "confidential",
            "restricted",
            "secret",
        }

        if (
            policy.classification.casefold()
            not in valid_classifications
        ):

            raise InvalidPolicyError(
                "Unknown document classification."
            )


# =============================================================================
# Default manager
# =============================================================================


permission_manager = PermissionManager()