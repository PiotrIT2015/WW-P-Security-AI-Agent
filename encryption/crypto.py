"""
encryption/crypto.py

Cryptographic utilities for Secure RAG.

Python 3.12+

Security properties:
    - AES-256-GCM
    - authenticated encryption
    - cryptographically secure nonces
    - key validation
    - no hard-coded secrets
"""

from __future__ import annotations

import base64
import os

from dataclasses import dataclass

from cryptography.exceptions import InvalidTag

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# =============================================================================
# Constants
# =============================================================================

AES_KEY_SIZE = 32
NONCE_SIZE = 12


# =============================================================================
# Exceptions
# =============================================================================


class CryptoError(Exception):
    """Base cryptographic exception."""

    pass


class CryptoIntegrityError(CryptoError):
    """
    Raised when encrypted data fails authentication.
    """

    pass


# =============================================================================
# Encrypted payload
# =============================================================================


@dataclass(slots=True, frozen=True)
class EncryptedPayload:
    """
    Represents encrypted application data.
    """

    nonce: bytes
    ciphertext: bytes


# =============================================================================
# Crypto Manager
# =============================================================================


class CryptoManager:
    """
    Central cryptographic service.

    AES-256-GCM is used for authenticated encryption.
    """

    KEY_SIZE = AES_KEY_SIZE
    NONCE_SIZE = NONCE_SIZE

    # =========================================================================
    # Key generation
    # =========================================================================

    @classmethod
    def generate_key(cls) -> bytes:
        """
        Generate a cryptographically secure AES-256 key.
        """

        return AESGCM.generate_key(
            bit_length=256
        )

    # =========================================================================
    # Key validation
    # =========================================================================

    @classmethod
    def validate_key(
        cls,
        key: bytes,
    ) -> bytes:
        """
        Validate AES-256 key.
        """

        if not isinstance(
            key,
            bytes,
        ):

            raise CryptoError(
                "Encryption key must be bytes."
            )

        if len(key) != cls.KEY_SIZE:

            raise CryptoError(
                "AES-256 key must contain exactly "
                "32 bytes."
            )

        return key

    # =========================================================================
    # Base64
    # =========================================================================

    @classmethod
    def encode_key(
        cls,
        key: bytes,
    ) -> str:
        """
        Encode binary key for storage in an environment variable.
        """

        cls.validate_key(
            key
        )

        return base64.b64encode(
            key
        ).decode(
            "ascii"
        )

    # -------------------------------------------------------------------------

    @classmethod
    def decode_key(
        cls,
        value: str,
    ) -> bytes:
        """
        Decode Base64 encoded AES-256 key.
        """

        if not value:

            raise CryptoError(
                "Encryption key is empty."
            )

        try:

            key = base64.b64decode(
                value,
                validate=True,
            )

        except Exception as error:

            raise CryptoError(
                "Invalid Base64 encryption key."
            ) from error

        return cls.validate_key(
            key
        )

    # =========================================================================
    # Encryption
    # =========================================================================

    @classmethod
    def encrypt(
        cls,
        plaintext: bytes,
        key: bytes,
        *,
        associated_data: bytes | None = None,
    ) -> EncryptedPayload:
        """
        Encrypt data using AES-256-GCM.
        """

        cls.validate_key(
            key
        )

        if not isinstance(
            plaintext,
            bytes,
        ):

            raise TypeError(
                "Plaintext must be bytes."
            )

        nonce = os.urandom(
            cls.NONCE_SIZE
        )

        aes = AESGCM(
            key
        )

        ciphertext = aes.encrypt(
            nonce,
            plaintext,
            associated_data,
        )

        return EncryptedPayload(

            nonce=nonce,

            ciphertext=ciphertext,

        )

    # =========================================================================
    # Decryption
    # =========================================================================

    @classmethod
    def decrypt(
        cls,
        payload: EncryptedPayload,
        key: bytes,
        *,
        associated_data: bytes | None = None,
    ) -> bytes:
        """
        Decrypt and authenticate AES-256-GCM payload.
        """

        cls.validate_key(
            key
        )

        if not isinstance(
            payload,
            EncryptedPayload,
        ):

            raise TypeError(
                "payload must be EncryptedPayload."
            )

        if len(
            payload.nonce
        ) != cls.NONCE_SIZE:

            raise CryptoError(
                "Invalid AES-GCM nonce."
            )

        try:

            return AESGCM(
                key
            ).decrypt(

                payload.nonce,

                payload.ciphertext,

                associated_data,

            )

        except InvalidTag as error:

            raise CryptoIntegrityError(
                "Encrypted data failed "
                "authentication."
            ) from error

    # =========================================================================
    # Text helpers
    # =========================================================================

    @classmethod
    def encrypt_text(
        cls,
        text: str,
        key: bytes,
        *,
        associated_data: bytes | None = None,
    ) -> EncryptedPayload:
        """
        Encrypt UTF-8 text.
        """

        return cls.encrypt(

            text.encode(
                "utf-8"
            ),

            key,

            associated_data=associated_data,

        )

    # -------------------------------------------------------------------------

    @classmethod
    def decrypt_text(
        cls,
        payload: EncryptedPayload,
        key: bytes,
        *,
        associated_data: bytes | None = None,
    ) -> str:
        """
        Decrypt UTF-8 text.
        """

        plaintext = cls.decrypt(

            payload,

            key,

            associated_data=associated_data,

        )

        try:

            return plaintext.decode(
                "utf-8"
            )

        except UnicodeDecodeError as error:

            raise CryptoError(
                "Decrypted payload is not valid UTF-8."
            ) from error


# =============================================================================
# Default manager
# =============================================================================

crypto = CryptoManager()