"""
vectorstore/encrypted_faiss.py

Encrypted FAISS vector store for Secure RAG.

Python 3.12+

Security properties:

- FAISS index is encrypted at rest.
- AES-256-GCM provides confidentiality and integrity.
- Chunk IDs are mapped to FAISS vector IDs.
- Metadata is kept separately from the vector index.
- Index writes are atomic.
- Decryption does not use pickle.
- Loading an index never executes arbitrary Python objects.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

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
    """

    vector_id: int

    chunk_id: str

    score: float


# =============================================================================
# Encrypted FAISS
# =============================================================================


class EncryptedFAISS:
    """
    Secure FAISS vector store.

    FAISS stores vectors.

    A separate mapping connects:

        FAISS vector ID
                ↓
            chunk_id

    The chunk itself is not stored inside FAISS.

    This separation is important because authorization should
    happen outside the vector index.
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

        self.storage_path = Path(
            storage_path
        )

        self.key = self._validate_key(
            key
        )

        self.dimension = dimension

        self._index: faiss.Index | None = None

        self._chunk_ids: list[str] = []

        self._metadata: dict[str, dict] = {}

    # =========================================================================
    # Key
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
        Convert binary key to Base64 representation.
        """

        return base64.b64encode(
            key
        ).decode("ascii")

    @classmethod
    def key_from_base64(
        cls,
        value: str,
    ) -> bytes:
        """
        Decode a Base64 AES-256 key.
        """

        try:

            key = base64.b64decode(
                value,
                validate=True,
            )

        except Exception as error:

            raise VectorStoreSecurityError(
                "Invalid Base64 encryption key."
            ) from error

        return cls._validate_key(
            key
        )

    @classmethod
    def _validate_key(
        cls,
        key: bytes,
    ) -> bytes:

        if not isinstance(
            key,
            bytes,
        ):

            raise TypeError(
                "Encryption key must be bytes."
            )

        if len(key) != cls.KEY_SIZE:

            raise VectorStoreSecurityError(
                "AES-256 requires exactly 32 bytes."
            )

        return key

    # =========================================================================
    # Index
    # =========================================================================

    def _create_index(
        self,
        dimension: int,
    ) -> faiss.Index:

        if dimension <= 0:

            raise VectorStoreError(
                "FAISS dimension must be positive."
            )

        # Inner product is appropriate for normalized
        # embeddings and therefore corresponds to cosine
        # similarity.

        index = faiss.IndexFlatIP(
            dimension
        )

        return index

    # =========================================================================
    # Add
    # =========================================================================

    def add_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        vectors: np.ndarray,
    ) -> None:
        """
        Add document chunks and their embeddings.

        Vectors must correspond positionally to chunks.
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

        dimension = vectors.shape[1]

        if self._index is None:

            self._index = self._create_index(
                dimension
            )

            self.dimension = dimension

        if self.dimension != dimension:

            raise VectorStoreError(
                "Embedding dimension does not match "
                "existing FAISS index."
            )

        # Defensive copy.

        vectors = np.ascontiguousarray(
            vectors,
            dtype=np.float32,
        )

        start_id = len(
            self._chunk_ids
        )

        self._index.add(
            vectors
        )

        for offset, chunk in enumerate(chunks):

            vector_id = start_id + offset

            self._chunk_ids.append(
                chunk.chunk_id
            )

            self._metadata[
                chunk.chunk_id
            ] = {

                "chunk_id":
                    chunk.chunk_id,

                "document_id":
                    chunk.document_id,

                "source":
                    chunk.source,

                "position":
                    chunk.position,

                **chunk.metadata,

            }

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

        Returns only valid mapped vector IDs.
        """

        if self._index is None:

            return []

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than zero."
            )

        vector = np.asarray(
            query_vector,
            dtype=np.float32,
        )

        if vector.ndim == 1:

            vector = vector.reshape(
                1,
                -1,
            )

        if vector.ndim != 2:

            raise VectorStoreError(
                "Query vector must have shape "
                "(dimension,) or (1, dimension)."
            )

        if vector.shape[1] != self.dimension:

            raise VectorStoreError(
                "Query vector dimension does not "
                "match FAISS index."
            )

        vector = np.ascontiguousarray(
            vector,
            dtype=np.float32,
        )

        scores, ids = self._index.search(
            vector,
            min(
                top_k,
                self._index.ntotal,
            ),
        )

        results: list[VectorSearchResult] = []

        for score, vector_id in zip(
            scores[0],
            ids[0],
        ):

            if vector_id < 0:

                continue

            if vector_id >= len(
                self._chunk_ids
            ):

                continue

            results.append(

                VectorSearchResult(

                    vector_id=int(
                        vector_id
                    ),

                    chunk_id=self._chunk_ids[
                        vector_id
                    ],

                    score=float(
                        score
                    ),
                )
            )

        return results

    # =========================================================================
    # Metadata
    # =========================================================================

    def get_metadata(
        self,
        chunk_id: str,
    ) -> dict | None:
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
        Serialize FAISS index and metadata and encrypt them.

        The entire payload is authenticated using AES-GCM.
        """

        if self._index is None:

            raise VectorStoreError(
                "Cannot save an empty vector store."
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

            "version":
                self.VERSION,

            "dimension":
                self.dimension,

            "count":
                len(self._chunk_ids),

            "chunk_ids":
                self._chunk_ids,

            "metadata":
                self._metadata,

            "index_sha256":
                hashlib.sha256(
                    index_bytes
                ).hexdigest(),

            "index":
                base64.b64encode(
                    index_bytes
                ).decode("ascii"),

        }

        plaintext = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
        ).encode(
            "utf-8"
        )

        nonce = os.urandom(
            self.NONCE_SIZE
        )

        aad = self._aad()

        try:

            ciphertext = AESGCM(
                self.key
            ).encrypt(
                nonce,
                plaintext,
                aad,
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
        """

        if not self.storage_path.exists():

            raise VectorStoreError(
                f"Vector store not found: "
                f"{self.storage_path}"
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

        if not container.startswith(
            self.MAGIC
        ):

            raise VectorStoreSecurityError(
                "Invalid vector store format."
            )

        offset = len(
            self.MAGIC
        )

        nonce = container[
            offset:
            offset + self.NONCE_SIZE
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
                plaintext.decode(
                    "utf-8"
                )
            )

        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as error:

            raise VectorStoreSecurityError(
                "Decrypted vector store contains "
                "invalid data."
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

        dimension = payload[
            "dimension"
        ]

        if index.d != dimension:

            raise VectorStoreSecurityError(
                "FAISS dimension mismatch."
            )

        if index.ntotal != payload[
            "count"
        ]:

            raise VectorStoreSecurityError(
                "FAISS vector count does not "
                "match metadata."
            )

        self._index = index

        self.dimension = int(
            dimension
        )

        self._chunk_ids = list(
            payload["chunk_ids"]
        )

        self._metadata = dict(
            payload["metadata"]
        )

    # =========================================================================
    # Validation
    # =========================================================================

    def _validate_payload(
        self,
        payload: dict,
    ) -> None:

        if not isinstance(
            payload,
            dict,
        ):

            raise VectorStoreSecurityError(
                "Invalid vector store payload."
            )

        if payload.get(
            "version"
        ) != self.VERSION:

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

        if not required.issubset(
            payload
        ):

            raise VectorStoreSecurityError(
                "Vector store payload is incomplete."
            )

        if not isinstance(
            payload["dimension"],
            int,
        ):

            raise VectorStoreSecurityError(
                "Invalid vector dimension."
            )

        if not isinstance(
            payload["count"],
            int,
        ):

            raise VectorStoreSecurityError(
                "Invalid vector count."
            )

        if not isinstance(
            payload["chunk_ids"],
            list,
        ):

            raise VectorStoreSecurityError(
                "Invalid chunk ID mapping."
            )

        if len(
            payload["chunk_ids"]
        ) != payload["count"]:

            raise VectorStoreSecurityError(
                "Chunk ID count mismatch."
            )

        if not isinstance(
            payload["metadata"],
            dict,
        ):

            raise VectorStoreSecurityError(
                "Invalid metadata mapping."
            )

    # =========================================================================
    # AAD
    # =========================================================================

    def _aad(self) -> bytes:
        """
        Additional authenticated data.

        It is authenticated but not encrypted.
        """

        return (
            self.MAGIC
            + b":v"
            + str(
                self.VERSION
            ).encode(
                "ascii"
            )
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

        Prevents partially written index files from replacing
        a valid store.
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

                file.write(
                    data
                )

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

    return hashlib.sha256(
        first.encode()
    ).digest() == hashlib.sha256(
        second.encode()
    ).digest()