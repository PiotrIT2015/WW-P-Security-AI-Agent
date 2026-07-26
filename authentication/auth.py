"""
authentication/auth.py

Authentication layer for Secure RAG.

Python 3.12+

Responsibilities:

- user identity
- password hashing
- password verification
- account status
- authentication context
- session token generation

Authorization is intentionally implemented separately
in authentication/permissions.py.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time

from dataclasses import dataclass, field
from enum import Enum


try:
    from argon2 import PasswordHasher
    from argon2.exceptions import (
        InvalidHash,
        VerificationError,
        VerifyMismatchError,
    )
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "argon2-cffi is required. "
        "Install it with: pip install argon2-cffi"
    ) from error


# =============================================================================
# Exceptions
# =============================================================================


class AuthenticationError(Exception):
    """Base authentication exception."""

    pass


class InvalidCredentialsError(AuthenticationError):
    """Raised when authentication fails."""

    pass


class AccountDisabledError(AuthenticationError):
    """Raised when an account is disabled."""

    pass


class SessionError(AuthenticationError):
    """Raised when session validation fails."""

    pass


# =============================================================================
# Enums
# =============================================================================


class AccountStatus(str, Enum):
    """
    User account state.
    """

    ACTIVE = "active"

    DISABLED = "disabled"

    LOCKED = "locked"


# =============================================================================
# User
# =============================================================================


@dataclass(slots=True)
class User:
    """
    Authenticated Secure RAG user.

    password_hash is the Argon2id hash.
    The plaintext password is never stored.
    """

    user_id: str

    username: str

    password_hash: str

    roles: set[str] = field(
        default_factory=set
    )

    groups: set[str] = field(
        default_factory=set
    )

    status: AccountStatus = (
        AccountStatus.ACTIVE
    )


# =============================================================================
# Authentication context
# =============================================================================


@dataclass(slots=True, frozen=True)
class UserContext:
    """
    Identity context passed through the Secure RAG pipeline.
    """

    user_id: str

    username: str

    roles: frozenset[str]

    groups: frozenset[str]

    authenticated_at: float

    session_id: str


# =============================================================================
# Session
# =============================================================================


@dataclass(slots=True)
class Session:
    """
    Server-side authentication session.

    The session stores only a hash of the session token.
    """

    session_id: str

    token_hash: bytes

    user_id: str

    created_at: float

    expires_at: float

    revoked: bool = False


# =============================================================================
# Authentication manager
# =============================================================================


class AuthenticationManager:
    """
    Authentication service.

    Responsibilities:

    - create users
    - verify passwords
    - create sessions
    - validate sessions
    - revoke sessions

    It does NOT decide whether a user can access a document.
    """

    def __init__(
        self,
        session_ttl: int = 3600,
    ) -> None:

        if session_ttl <= 0:

            raise ValueError(
                "session_ttl must be greater than zero."
            )

        self.session_ttl = session_ttl

        self._users: dict[
            str,
            User,
        ] = {}

        self._sessions: dict[
            str,
            Session,
        ] = {}

        self._password_hasher = (
            PasswordHasher()
        )

    # =========================================================================
    # Password hashing
    # =========================================================================

    def hash_password(
        self,
        password: str,
    ) -> str:
        """
        Generate an Argon2id password hash.
        """

        self._validate_password(
            password
        )

        return self._password_hasher.hash(
            password
        )

    # =========================================================================
    # Password verification
    # =========================================================================

    def verify_password(
        self,
        password: str,
        password_hash: str,
    ) -> bool:
        """
        Verify plaintext password against Argon2id hash.
        """

        if not password:
            return False

        if not password_hash:
            return False

        try:

            return self._password_hasher.verify(
                password_hash,
                password,
            )

        except (
            VerifyMismatchError,
            VerificationError,
            InvalidHash,
        ):

            return False

    # =========================================================================
    # User creation
    # =========================================================================

    def create_user(
        self,
        username: str,
        password: str,
        *,
        user_id: str | None = None,
        roles: set[str] | None = None,
        groups: set[str] | None = None,
    ) -> User:
        """
        Create a new user.
        """

        username = username.strip()

        if not username:

            raise AuthenticationError(
                "Username cannot be empty."
            )

        normalized_username = (
            username.casefold()
        )

        if normalized_username in self._users:

            raise AuthenticationError(
                "User already exists."
            )

        if user_id is None:

            user_id = secrets.token_urlsafe(
                16
            )

        password_hash = (
            self.hash_password(
                password
            )
        )

        user = User(

            user_id=user_id,

            username=username,

            password_hash=password_hash,

            roles=set(
                roles or set()
            ),

            groups=set(
                groups or set()
            ),

            status=AccountStatus.ACTIVE,

        )

        self._users[
            normalized_username
        ] = user

        return user

    # =========================================================================
    # Authentication
    # =========================================================================

    def authenticate(
        self,
        username: str,
        password: str,
    ) -> tuple[UserContext, str]:
        """
        Authenticate user and create session.

        Returns:

            (UserContext, session_token)
        """

        normalized_username = (
            username.strip().casefold()
        )

        user = self._users.get(
            normalized_username
        )

        if user is None:

            raise InvalidCredentialsError(
                "Invalid username or password."
            )

        if user.status == AccountStatus.DISABLED:

            raise AccountDisabledError(
                "Account is disabled."
            )

        if user.status == AccountStatus.LOCKED:

            raise AccountDisabledError(
                "Account is locked."
            )

        if not self.verify_password(
            password,
            user.password_hash,
        ):

            raise InvalidCredentialsError(
                "Invalid username or password."
            )

        return self._create_session(
            user
        )

    # =========================================================================
    # Session creation
    # =========================================================================

    def _create_session(
        self,
        user: User,
    ) -> tuple[UserContext, str]:

        session_id = secrets.token_urlsafe(
            24
        )

        session_token = secrets.token_urlsafe(
            48
        )

        now = time.time()

        session = Session(

            session_id=session_id,

            token_hash=self._hash_session_token(
                session_token
            ),

            user_id=user.user_id,

            created_at=now,

            expires_at=(
                now + self.session_ttl
            ),

        )

        self._sessions[
            session_id
        ] = session

        context = UserContext(

            user_id=user.user_id,

            username=user.username,

            roles=frozenset(
                user.roles
            ),

            groups=frozenset(
                user.groups
            ),

            authenticated_at=now,

            session_id=session_id,

        )

        return (
            context,
            session_token,
        )

    # =========================================================================
    # Session validation
    # =========================================================================

    def validate_session(
        self,
        session_id: str,
        session_token: str,
    ) -> UserContext:
        """
        Validate an active authentication session.
        """

        session = self._sessions.get(
            session_id
        )

        if session is None:

            raise SessionError(
                "Invalid session."
            )

        if session.revoked:

            raise SessionError(
                "Session has been revoked."
            )

        now = time.time()

        if now >= session.expires_at:

            session.revoked = True

            raise SessionError(
                "Session has expired."
            )

        supplied_hash = (
            self._hash_session_token(
                session_token
            )
        )

        if not hmac.compare_digest(
            supplied_hash,
            session.token_hash,
        ):

            raise SessionError(
                "Invalid session token."
            )

        user = self._find_user_by_id(
            session.user_id
        )

        if user is None:

            raise SessionError(
                "Session user no longer exists."
            )

        if user.status != AccountStatus.ACTIVE:

            raise SessionError(
                "User account is not active."
            )

        return UserContext(

            user_id=user.user_id,

            username=user.username,

            roles=frozenset(
                user.roles
            ),

            groups=frozenset(
                user.groups
            ),

            authenticated_at=session.created_at,

            session_id=session.session_id,

        )

    # =========================================================================
    # Session revocation
    # =========================================================================

    def revoke_session(
        self,
        session_id: str,
    ) -> None:
        """
        Revoke an existing session.
        """

        session = self._sessions.get(
            session_id
        )

        if session is None:

            return

        session.revoked = True

    # =========================================================================
    # User lookup
    # =========================================================================

    def get_user(
        self,
        username: str,
    ) -> User | None:
        """
        Return a user by username.
        """

        return self._users.get(
            username.strip().casefold()
        )

    # =========================================================================
    # Internal lookup
    # =========================================================================

    def _find_user_by_id(
        self,
        user_id: str,
    ) -> User | None:

        for user in self._users.values():

            if user.user_id == user_id:

                return user

        return None

    # =========================================================================
    # Security helpers
    # =========================================================================

    @staticmethod
    def _hash_session_token(
        token: str,
    ) -> bytes:
        """
        Hash session token before storing it.

        SHA-256 is appropriate here because the token itself
        is generated using a cryptographically secure random
        generator and therefore has high entropy.
        """

        return hashlib.sha256(
            token.encode(
                "utf-8"
            )
        ).digest()

    # =========================================================================
    # Password validation
    # =========================================================================

    @staticmethod
    def _validate_password(
        password: str,
    ) -> None:

        if not isinstance(
            password,
            str,
        ):

            raise TypeError(
                "Password must be a string."
            )

        if len(password) < 12:

            raise AuthenticationError(
                "Password must contain at least "
                "12 characters."
            )


# =============================================================================
# Default manager
# =============================================================================


auth_manager = AuthenticationManager()