"""
Vector store package.
"""

from .encrypted_faiss import (
    EncryptedFAISS,
    VectorSearchResult,
    VectorStoreError,
    VectorStoreSecurityError,
)

__all__ = [
    "EncryptedFAISS",
    "VectorSearchResult",
    "VectorStoreError",
    "VectorStoreSecurityError",
]