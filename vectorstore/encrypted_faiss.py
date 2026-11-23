
"""
vectorstore/encrypted_faiss.py

Encrypted FAISS vector store for Secure RAG.

Python 3.12+

Security properties:

- FAISS index encrypted at rest.
- AES-256-GCM confidentiality and integrity.
- Chunk IDs mapped to FAISS vector IDs.
- Document text and metadata encrypted together.
- Atomic index writes.
- No pickle deserialization.
- Input and payload validation.
- Compatible with SecureRetriever.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import tempfile

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from documents.chunker import DocumentChunk


# =============================================================================
# Exceptions
# =============================================================================


class VectorStoreError(Exception):
    """Base exception for vector store errors."""

    pass


class VectorStoreSecurityError(VectorStoreError):
    """Raised when encrypted vector data fails validation."""

    pass


# =============================================================================
# Search result
# =============================================================================


@dataclass(slots=True, frozen=True)
class VectorSearchResult:
    """
    Result returned from vector similarity search.

    Compatible with SecureRetriever._from_object().
    """

    vector_id: int

    chunk_id: str

    document_id: str

    text: str

    score: float

    metadata: dict[str, Any]


# =============================================================================
# Encrypted FAISS
# =============================================================================


class EncryptedFAISS:
    """
    Secure FAISS vector store.

    Architecture:

        FAISS vector ID
                |
                v
             chunk_id
                |
                v
           document_id
                |
                v
          encrypted metadata

    The FAISS index and metadata are serialized together
    and encrypted using AES-256-GCM.
    """

    VERSION = 1

    NONCE_SIZE = 12

    KEY_SIZE = 32

    MAGIC = b"WWP-SRAG-FAISS"

    def __init__(
        self,
        storage_path: str | Path,
        key: bytes,
        dimension: int | None = None,
    ) -> None:

        self.storage_path = Path(storage_path)

        self.key = self._validate_key(key)

        if dimension is not None:

            if not isinstance(dimension, int) or dimension <= 0:

                raise VectorStoreError(
                    "Dimension must be a positive integer."
                )

        self.dimension = dimension

        self._index: faiss.Index | None = None

        self._chunk_ids: list[str] = []

        self._metadata: dict[str, dict[str, Any]] = {}

    # =========================================================================
    # Key management
    # =========================================================================

    @classmethod
    def generate_key(cls) -> bytes:
        """
        Generate a cryptographically secure AES-256 key.
        """

        return AESGCM.generate_key(
            bit_length=256
        )

    @staticmethod
    def key_to_base64(
        key: bytes,
    ) -> str:
        """
        Convert binary encryption key to Base64.
        """

        if not isinstance(key, bytes):

            raise TypeError(
                "Encryption key must be bytes."
            )

        return base64.b64encode(
            key
        ).decode("ascii")

    @classmethod
    def key_from_base64(
        cls,
        value: str,
    ) -> bytes:
        """
        Decode Base64 AES-256 key.
        """

        if not isinstance(value, str):

            raise VectorStoreSecurityError(
                "Encryption key must be a string."
            )

        try:

            key = base64.b64decode(
                value,
                validate=True,
            )

        except Exception as error:

            raise VectorStoreSecurityError(
                "Invalid Base64 encryption key."
            ) from error

        return cls._validate_key(key)

    @classmethod
    def _validate_key(
        cls,
        key: bytes,
    ) -> bytes:
        """
        Validate AES-256 key.
        """

        if not isinstance(key, bytes):

            raise TypeError(
                "Encryption key must be bytes."
            )

        if len(key) != cls.KEY_SIZE:

            raise VectorStoreSecurityError(
                "AES-256 requires exactly 32 bytes."
            )

        return key

    # =========================================================================
    # Index management
    # =========================================================================

    def _create_index(
        self,
        dimension: int,
    ) -> faiss.Index:
        """
        Create FAISS inner-product index.

        For normalized vectors, inner product corresponds
        to cosine similarity.
        """

        if dimension <= 0:

            raise VectorStoreError(
                "FAISS dimension must be positive."
            )

        return faiss.IndexFlatIP(
            dimension
        )

    # =========================================================================
    # Add chunks
    # =========================================================================

    def add_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        vectors: np.ndarray,
    ) -> None:
        """
        Add document chunks and their embeddings.

        Each vector must correspond positionally to its chunk.
        """

        if len(chunks) != len(vectors):

            raise VectorStoreError(
                "Number of chunks does not match "
                "number of vectors."
            )

        if not chunks:

            return

        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        if vectors.ndim != 2:

            raise VectorStoreError(
                "Vectors must be a two-dimensional matrix."
            )

        if vectors.shape[0] != len(chunks):

            raise VectorStoreError(
                "Vector count mismatch."
            )

        dimension = vectors.shape[1]

        if dimension <= 0:

            raise VectorStoreError(
                "Invalid embedding dimension."
            )

        if not np.all(np.isfinite(vectors)):

            raise VectorStoreError(
                "Embeddings contain invalid numerical values."
            )

        if self.dimension is not None:

            if dimension != self.dimension:

                raise VectorStoreError(
                    "Embedding dimension does not match "
                    "configured dimension."
                )

        if self._index is not None:

            if dimension != self.dimension:

                raise VectorStoreError(
                    "Embedding dimension does not match "
                    "existing FAISS index."
                )

        # ---------------------------------------------------------
        # Validate chunks before modifying the index.
        # ---------------------------------------------------------

        new_chunk_ids = []

        new_metadata = {}

        for chunk in chunks:

            if not isinstance(chunk, DocumentChunk):

                raise VectorStoreError(
                    "Invalid document chunk."
                )

            if not chunk.chunk_id or not chunk.document_id:

                raise VectorStoreError(
                    "Chunk ID and document ID are required."
                )

            if not isinstance(chunk.text, str) or not chunk.text.strip():

                raise VectorStoreError(
                    "Document chunk text cannot be empty."
                )

            if chunk.chunk_id in self._metadata:

                raise VectorStoreError(
                    f"Duplicate chunk ID: {chunk.chunk_id}"
                )

            if chunk.chunk_id in new_chunk_ids:

                raise VectorStoreError(
                    f"Duplicate chunk ID in batch: {chunk.chunk_id}"
                )

            metadata = dict(
                chunk.metadata
            )

            # Security-critical fields override user metadata.

            metadata.update({

                "chunk_id": chunk.chunk_id,

                "document_id": chunk.document_id,

                "text": chunk.text,

                "source": chunk.source,

                "position": chunk.position,

                "page": chunk.page,

            })

            new_chunk_ids.append(
                chunk.chunk_id
            )

            new_metadata[chunk.chunk_id] = metadata

        # ---------------------------------------------------------
        # Validate JSON serializability.
        # ---------------------------------------------------------

        try:

            json.dumps(
                new_metadata,
                ensure_ascii=False,
            )

        except (TypeError, ValueError) as error:

            raise VectorStoreError(
                "Chunk metadata is not JSON serializable."
            ) from error

        # ---------------------------------------------------------
        # Prepare FAISS index.
        # ---------------------------------------------------------

        vectors = np.ascontiguousarray(
            vectors,
            dtype=np.float32,
        )

        if self._index is None:

            self._index = self._create_index(
                dimension
            )

            self.dimension = dimension

        # ---------------------------------------------------------
        # Add vectors.
        # ---------------------------------------------------------

        try:

            self._index.add(
                vectors
            )

        except Exception as error:

            raise VectorStoreError(
                "Unable to add vectors to FAISS index."
            ) from error

        # ---------------------------------------------------------
        # Update mappings.
        # ---------------------------------------------------------

        self._chunk_ids.extend(
            new_chunk_ids
        )

        self._metadata.update(
            new_metadata
        )

    # =========================================================================
    # Search
    # =========================================================================

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        """
        Search nearest vectors.

        Returns document text and metadata required by
        SecureRetriever.

        FAISS search does NOT authorize document access.
        Authorization must be performed by PermissionManager.
        """

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than zero."
            )

        if self._index is None:

            return []

        vector = np.asarray(
            query_vector,
            dtype=np.float32,
        )

        if vector.ndim == 1:

            vector = vector.reshape(
                1,
                -1,
            )

        if vector.ndim != 2 or vector.shape[0] != 1:

            raise VectorStoreError(
                "Query vector must have shape "
                "(dimension,) or (1, dimension)."
            )

        if vector.shape[1] != self.dimension:

            raise VectorStoreError(
                "Query vector dimension does not "
                "match FAISS index."
            )

        if not np.all(np.isfinite(vector)):

            raise VectorStoreError(
                "Query vector contains invalid numerical values."
            )

        vector = np.ascontiguousarray(
            vector,
            dtype=np.float32,
        )

        search_count = min(
            top_k,
            self._index.ntotal,
        )

        try:

            scores, ids = self._index.search(
                vector,
                search_count,
            )

        except Exception as error:

            raise VectorStoreError(
                "FAISS similarity search failed."
            ) from error

        results: list[VectorSearchResult] = []

        for score, vector_id in zip(
            scores[0],
            ids[0],
        ):

            if vector_id < 0:

                continue

            vector_id = int(
                vector_id
            )

            if vector_id >= len(self._chunk_ids):

                raise VectorStoreSecurityError(
                    "Invalid FAISS vector mapping."
                )

            chunk_id = self._chunk_ids[
                vector_id
            ]

            metadata = self._metadata.get(
                chunk_id
            )

            if metadata is None:

                raise VectorStoreSecurityError(
                    "Missing chunk metadata."
                )

            document_id = metadata.get(
                "document_id"
            )

            text = metadata.get(
                "text"
            )

            if not document_id or not text:

                raise VectorStoreSecurityError(
                    "Incomplete document metadata."
                )

            # Text is returned separately.
            # Remaining metadata is passed to RBAC/ACL.

            result_metadata = {

                key: value

                for key, value in metadata.items()

                if key != "text"

            }

            results.append(
                VectorSearchResult(

                    vector_id=vector_id,

                    chunk_id=chunk_id,

                    document_id=document_id,

                    text=text,

                    score=float(score),

                    metadata=result_metadata,

                )
            )

        return results

    # =========================================================================
    # Metadata
    # =========================================================================

    def get_metadata(
        self,
        chunk_id: str,
    ) -> dict[str, Any] | None:
        """
        Return metadata associated with a chunk.
        """

        metadata = self._metadata.get(
            chunk_id
        )

        if metadata is None:

            return None

        return dict(
            metadata
        )

    # =========================================================================
    # Save
    # =========================================================================

    def save(self) -> None:
        """
        Serialize FAISS index and metadata.

        Encrypt the complete payload using AES-256-GCM.
        """

        if self._index is None:

            raise VectorStoreError(
                "Cannot save an empty vector store."
            )

        if self._index.ntotal != len(self._chunk_ids):

            raise VectorStoreSecurityError(
                "FAISS vector mapping is inconsistent."
            )

        self.storage_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:

            index_bytes = faiss.serialize_index(
                self._index
            ).tobytes()

        except Exception as error:

            raise VectorStoreError(
                "Unable to serialize FAISS index."
            ) from error

        payload = {

            "version": self.VERSION,

            "dimension": self.dimension,

            "count": len(self._chunk_ids),

            "chunk_ids": self._chunk_ids,

            "metadata": self._metadata,

            "index_sha256": hashlib.sha256(
                index_bytes
            ).hexdigest(),

            "index": base64.b64encode(
                index_bytes
            ).decode("ascii"),

        }

        try:

            plaintext = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")

        except (TypeError, ValueError) as error:

            raise VectorStoreError(
                "Unable to serialize vector store metadata."
            ) from error

        nonce = os.urandom(
            self.NONCE_SIZE
        )

        try:

            ciphertext = AESGCM(
                self.key
            ).encrypt(
                nonce,
                plaintext,
                self._aad(),
            )

        except Exception as error:

            raise VectorStoreSecurityError(
                "Unable to encrypt FAISS store."
            ) from error

        container = (
            self.MAGIC
            + nonce
            + ciphertext
        )

        self._atomic_write(
            container
        )

    # =========================================================================
    # Load
    # =========================================================================

    def load(self) -> None:
        """
        Load and decrypt an encrypted FAISS store.

        The index is not modified until validation succeeds.
        """

        if not self.storage_path.exists():

            raise VectorStoreError(
                f"Vector store not found: {self.storage_path}"
            )

        try:

            container = self.storage_path.read_bytes()

        except OSError as error:

            raise VectorStoreError(
                "Unable to read encrypted vector store."
            ) from error

        minimum_size = (
            len(self.MAGIC)
            + self.NONCE_SIZE
            + 16
        )

        if len(container) < minimum_size:

            raise VectorStoreSecurityError(
                "Encrypted vector store is truncated."
            )

        if not container.startswith(self.MAGIC):

            raise VectorStoreSecurityError(
                "Invalid vector store format."
            )

        offset = len(self.MAGIC)

        nonce = container[
            offset:offset + self.NONCE_SIZE
        ]

        ciphertext = container[
            offset + self.NONCE_SIZE:
        ]

        try:

            plaintext = AESGCM(
                self.key
            ).decrypt(
                nonce,
                ciphertext,
                self._aad(),
            )

        except InvalidTag as error:

            raise VectorStoreSecurityError(
                "Authentication failed. "
                "The key may be wrong or the "
                "vector store may have been modified."
            ) from error

        except Exception as error:

            raise VectorStoreSecurityError(
                "Unable to decrypt vector store."
            ) from error

        try:

            payload = json.loads(
                plaintext.decode("utf-8")
            )

        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as error:

            raise VectorStoreSecurityError(
                "Decrypted vector store contains invalid data."
            ) from error

        self._validate_payload(
            payload
        )

        try:

            index_bytes = base64.b64decode(
                payload["index"],
                validate=True,
            )

        except Exception as error:

            raise VectorStoreSecurityError(
                "Invalid encoded FAISS index."
            ) from error

        expected_hash = payload[
            "index_sha256"
        ]

        actual_hash = hashlib.sha256(
            index_bytes
        ).hexdigest()

        if not _constant_time_equal(
            actual_hash,
            expected_hash,
        ):

            raise VectorStoreSecurityError(
                "FAISS index integrity check failed."
            )

        try:

            array = np.frombuffer(
                index_bytes,
                dtype=np.uint8,
            )

            index = faiss.deserialize_index(
                array
            )

        except Exception as error:

            raise VectorStoreSecurityError(
                "Unable to deserialize FAISS index."
            ) from error

        dimension = payload["dimension"]

        if index.d != dimension:

            raise VectorStoreSecurityError(
                "FAISS dimension mismatch."
            )

        if index.ntotal != payload["count"]:

            raise VectorStoreSecurityError(
                "FAISS vector count does not match metadata."
            )

        if (
            self.dimension is not None
            and self.dimension != dimension
        ):

            raise VectorStoreSecurityError(
                "Loaded index dimension does not match "
                "configured dimension."
            )

        # Commit validated state.

        self._index = index

        self.dimension = dimension

        self._chunk_ids = list(
            payload["chunk_ids"]
        )

        self._metadata = {
            key: dict(value)
            for key, value in payload["metadata"].items()
        }

    # =========================================================================
    # Payload validation
    # =========================================================================

    def _validate_payload(
        self,
        payload: dict,
    ) -> None:
        """
        Validate decrypted payload before FAISS deserialization.
        """

        if not isinstance(payload, dict):

            raise VectorStoreSecurityError(
                "Invalid vector store payload."
            )

        if payload.get("version") != self.VERSION:

            raise VectorStoreSecurityError(
                "Unsupported vector store version."
            )

        required = {
            "dimension",
            "count",
            "chunk_ids",
            "metadata",
            "index_sha256",
            "index",
        }

        if not required.issubset(payload):

            raise VectorStoreSecurityError(
                "Vector store payload is incomplete."
            )

        dimension = payload["dimension"]

        count = payload["count"]

        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
        ):

            raise VectorStoreSecurityError(
                "Invalid vector dimension."
            )

        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
        ):

            raise VectorStoreSecurityError(
                "Invalid vector count."
            )

        chunk_ids = payload["chunk_ids"]

        if not isinstance(chunk_ids, list):

            raise VectorStoreSecurityError(
                "Invalid chunk ID mapping."
            )

        if len(chunk_ids) != count:

            raise VectorStoreSecurityError(
                "Chunk ID count mismatch."
            )

        if any(
            not isinstance(item, str) or not item
            for item in chunk_ids
        ):

            raise VectorStoreSecurityError(
                "Invalid chunk ID."
            )

        if len(set(chunk_ids)) != len(chunk_ids):

            raise VectorStoreSecurityError(
                "Duplicate chunk IDs in vector store."
            )

        metadata = payload["metadata"]

        if not isinstance(metadata, dict):

            raise VectorStoreSecurityError(
                "Invalid metadata mapping."
            )

        if set(metadata.keys()) != set(chunk_ids):

            raise VectorStoreSecurityError(
                "Metadata and chunk ID mappings do not match."
            )

        for chunk_id, item in metadata.items():

            if not isinstance(item, dict):

                raise VectorStoreSecurityError(
                    "Invalid chunk metadata."
                )

            if item.get("chunk_id") != chunk_id:

                raise VectorStoreSecurityError(
                    "Chunk metadata ID mismatch."
                )

            if not isinstance(
                item.get("document_id"),
                str,
            ) or not item["document_id"]:

                raise VectorStoreSecurityError(
                    "Invalid document ID."
                )

            if not isinstance(
                item.get("text"),
                str,
            ) or not item["text"]:

                raise VectorStoreSecurityError(
                    "Invalid document text."
                )

        if not isinstance(
            payload["index_sha256"],
            str,
        ):

            raise VectorStoreSecurityError(
                "Invalid index hash."
            )

        if len(payload["index_sha256"]) != 64:

            raise VectorStoreSecurityError(
                "Invalid index hash length."
            )

        if not isinstance(
            payload["index"],
            str,
        ):

            raise VectorStoreSecurityError(
                "Invalid encoded index."
            )

    # =========================================================================
    # Additional authenticated data
    # =========================================================================

    def _aad(self) -> bytes:
        """
        Additional authenticated data.

        AAD is authenticated but not encrypted.
        """

        return (
            self.MAGIC
            + b":v"
            + str(self.VERSION).encode("ascii")
        )

    # =========================================================================
    # Atomic write
    # =========================================================================

    def _atomic_write(
        self,
        data: bytes,
    ) -> None:
        """
        Write encrypted data atomically.

        Prevents partially written files from replacing
        a valid encrypted store.
        """

        self.storage_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fd, temp_name = tempfile.mkstemp(
            prefix=".faiss-",
            suffix=".tmp",
            dir=self.storage_path.parent,
        )

        try:

            with os.fdopen(
                fd,
                "wb",
            ) as file:

                file.write(data)

                file.flush()

                os.fsync(
                    file.fileno()
                )

            os.replace(
                temp_name,
                self.storage_path,
            )

        except Exception:

            try:

                os.unlink(
                    temp_name
                )

            except OSError:

                pass

            raise

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def count(self) -> int:
        """
        Number of vectors currently stored.
        """

        if self._index is None:

            return 0

        return int(
            self._index.ntotal
        )


# =============================================================================
# Utility
# =============================================================================


def _constant_time_equal(
    first: str,
    second: str,
) -> bool:
    """
    Compare strings using constant-time comparison.
    """

    return hmac.compare_digest(
        first,
        second,
    )
