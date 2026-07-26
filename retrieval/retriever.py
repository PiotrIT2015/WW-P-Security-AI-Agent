"""
retrieval/retriever.py

Secure retrieval pipeline for Secure RAG.

Python 3.12+

Security model:

    Query
      |
      v
    Embedding
      |
      v
    Vector search
      |
      v
    Candidate chunks
      |
      v
    ACL / RBAC filtering
      |
      v
    Authorized chunks
      |
      v
    RAG context

IMPORTANT:

Vector similarity is NOT an authorization mechanism.

FAISS returns candidates only.
PermissionManager decides whether the candidate
may be exposed to the user/LLM.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from authentication.auth import UserContext
from authentication.permissions import (
    PermissionManager,
)


logger = logging.getLogger(
    __name__
)


# =============================================================================
# Exceptions
# =============================================================================


class RetrievalError(Exception):
    """Base retrieval exception."""

    pass


class InvalidQueryError(RetrievalError):
    """Raised when a query is invalid."""

    pass


class VectorStoreError(RetrievalError):
    """Raised when vector store access fails."""

    pass


# =============================================================================
# Retrieved chunk
# =============================================================================


@dataclass(slots=True, frozen=True)
class RetrievedChunk:
    """
    Normalized result returned by the vector store.

    The text is kept separate from metadata.

    document_id is mandatory because authorization must
    be performed against the document that owns the chunk.
    """

    chunk_id: str

    document_id: str

    text: str

    score: float

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


# =============================================================================
# Retrieval result
# =============================================================================


@dataclass(slots=True, frozen=True)
class RetrievalResult:
    """
    Secure retrieval response.
    """

    query: str

    chunks: tuple[RetrievedChunk, ...]

    candidates_count: int

    authorized_count: int

    denied_count: int

    user_id: str

    @property
    def empty(self) -> bool:
        return not bool(
            self.chunks
        )

    @property
    def context(self) -> str:
        """
        Build plain text context for the LLM.

        Only authorized chunks are included because
        this object is created AFTER ACL filtering.
        """

        parts: list[str] = []

        for index, chunk in enumerate(
            self.chunks,
            start=1,
        ):

            parts.append(
                (
                    f"[Context {index}]\n"
                    f"Document: {chunk.document_id}\n"
                    f"Chunk: {chunk.chunk_id}\n"
                    f"{chunk.text}"
                )
            )

        return "\n\n".join(
            parts
        )


# =============================================================================
# Vector store protocol
# =============================================================================


class VectorStoreProtocol(Protocol):
    """
    Interface expected from EncryptedFAISS.

    Your existing EncryptedFAISS implementation can be
    adapted to this interface without coupling the
    retriever to FAISS internals.
    """

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
    ) -> Sequence[Any]:
        ...


# =============================================================================
# Embedder protocol
# =============================================================================


class EmbedderProtocol(Protocol):
    """
    Interface expected from embeddings/embedder.py.
    """

    def embed_query(
        self,
        text: str,
    ) -> Sequence[float]:
        ...


# =============================================================================
# Secure Retriever
# =============================================================================


class SecureRetriever:
    """
    Authorization-aware RAG retriever.

    Pipeline:

        query
          |
          v
        embedding
          |
          v
        vector store
          |
          v
        candidate normalization
          |
          v
        ACL/RBAC
          |
          v
        authorized chunks
    """

    def __init__(
        self,
        embedder: EmbedderProtocol,
        vector_store: VectorStoreProtocol,
        permission_manager: PermissionManager,
        *,
        default_top_k: int = 5,
        authorization_multiplier: int = 4,
    ) -> None:

        if default_top_k <= 0:

            raise ValueError(
                "default_top_k must be greater than zero."
            )

        if authorization_multiplier <= 0:

            raise ValueError(
                "authorization_multiplier must "
                "be greater than zero."
            )

        self.embedder = embedder

        self.vector_store = vector_store

        self.permission_manager = (
            permission_manager
        )

        self.default_top_k = (
            default_top_k
        )

        self.authorization_multiplier = (
            authorization_multiplier
        )

    # =========================================================================
    # Public retrieval API
    # =========================================================================

    def retrieve(
        self,
        query: str,
        user: UserContext,
        *,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """
        Perform secure retrieval.

        We retrieve more candidates than the requested final
        number because ACL filtering can remove documents.

        Example:

            requested top_k = 5
            multiplier = 4

            vector search = 20 candidates

            ACL filtering may reduce them to 5 or fewer.
        """

        query = self._validate_query(
            query
        )

        if not isinstance(
            user,
            UserContext,
        ):

            raise InvalidQueryError(
                "A valid authenticated UserContext "
                "is required."
            )

        requested_k = (
            top_k
            if top_k is not None
            else self.default_top_k
        )

        if requested_k <= 0:

            raise InvalidQueryError(
                "top_k must be greater than zero."
            )

        search_k = (
            requested_k
            * self.authorization_multiplier
        )

        # ---------------------------------------------------------
        # Step 1 — query embedding
        # ---------------------------------------------------------

        try:

            query_vector = (
                self.embedder.embed_query(
                    query
                )
            )

        except Exception as error:

            logger.exception(
                "Query embedding failed."
            )

            raise RetrievalError(
                "Unable to generate query embedding."
            ) from error

        # ---------------------------------------------------------
        # Step 2 — vector search
        # ---------------------------------------------------------

        try:

            raw_candidates = (
                self.vector_store.search(

                    query_vector,

                    search_k,

                )
            )

        except Exception as error:

            logger.exception(
                "Vector search failed."
            )

            raise VectorStoreError(
                "Vector store search failed."
            ) from error

        # ---------------------------------------------------------
        # Step 3 — normalize candidates
        # ---------------------------------------------------------

        candidates = (
            self._normalize_results(
                raw_candidates
            )
        )

        # ---------------------------------------------------------
        # Step 4 — authorization
        # ---------------------------------------------------------

        authorized: list[
            RetrievedChunk
        ] = []

        denied_count = 0

        for candidate in candidates:

            result = (
                self.permission_manager.authorize(

                    user,

                    candidate.document_id,

                    metadata=candidate.metadata,

                )
            )

            if result.allowed:

                authorized.append(
                    candidate
                )

            else:

                denied_count += 1

                logger.info(

                    "RAG document access denied: "
                    "user=%s document=%s",

                    user.user_id,

                    candidate.document_id,

                )

            if len(authorized) >= requested_k:

                break

        # ---------------------------------------------------------
        # Step 5 — final result
        # ---------------------------------------------------------

        final_chunks = tuple(
            authorized[
                :requested_k
            ]
        )

        logger.info(

            "Secure retrieval completed: "
            "user=%s candidates=%d authorized=%d denied=%d",

            user.user_id,

            len(candidates),

            len(final_chunks),

            denied_count,

        )

        return RetrievalResult(

            query=query,

            chunks=final_chunks,

            candidates_count=len(
                candidates
            ),

            authorized_count=len(
                final_chunks
            ),

            denied_count=denied_count,

            user_id=user.user_id,

        )

    # =========================================================================
    # Candidate normalization
    # =========================================================================

    @staticmethod
    def _normalize_results(
        results: Sequence[Any],
    ) -> list[RetrievedChunk]:
        """
        Convert vector-store-specific results into
        RetrievedChunk objects.

        Supported formats:

        1. RetrievedChunk

        2. dict:

            {
                "chunk_id": "...",
                "document_id": "...",
                "text": "...",
                "score": 0.91,
                "metadata": {...}
            }

        3. object with corresponding attributes.
        """

        normalized: list[
            RetrievedChunk
        ] = []

        for index, item in enumerate(
            results
        ):

            if isinstance(
                item,
                RetrievedChunk,
            ):

                normalized.append(
                    item
                )

                continue

            if isinstance(
                item,
                Mapping,
            ):

                chunk = (
                    SecureRetriever
                    ._from_mapping(
                        item,
                        index,
                    )
                )

            else:

                chunk = (
                    SecureRetriever
                    ._from_object(
                        item,
                        index,
                    )
                )

            if chunk is not None:

                normalized.append(
                    chunk
                )

        return normalized

    # =========================================================================
    # Mapping normalization
    # =========================================================================

    @staticmethod
    def _from_mapping(
        item: Mapping[str, Any],
        index: int,
    ) -> RetrievedChunk | None:

        document_id = str(
            item.get(
                "document_id",
                "",
            )
        )

        text = str(
            item.get(
                "text",
                "",
            )
        )

        if not document_id or not text:

            logger.warning(
                "Ignoring malformed vector result."
            )

            return None

        chunk_id = str(
            item.get(
                "chunk_id",
                f"unknown-{index}",
            )
        )

        score = float(
            item.get(
                "score",
                0.0,
            )
        )

        metadata = item.get(
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            Mapping,
        ):

            metadata = {}

        # Security-critical fields must be copied
        # into metadata even if the vector store
        # omitted them from nested metadata.

        merged_metadata = dict(
            metadata
        )

        merged_metadata.setdefault(
            "document_id",
            document_id,
        )

        if "classification" in item:

            merged_metadata.setdefault(
                "classification",
                item["classification"],
            )

        if "owner_id" in item:

            merged_metadata.setdefault(
                "owner_id",
                item["owner_id"],
            )

        return RetrievedChunk(

            chunk_id=chunk_id,

            document_id=document_id,

            text=text,

            score=score,

            metadata=merged_metadata,

        )

    # =========================================================================
    # Object normalization
    # =========================================================================

    @staticmethod
    def _from_object(
        item: Any,
        index: int,
    ) -> RetrievedChunk | None:

        document_id = getattr(
            item,
            "document_id",
            None,
        )

        text = getattr(
            item,
            "text",
            None,
        )

        if not document_id or not text:

            logger.warning(
                "Ignoring malformed vector result object."
            )

            return None

        chunk_id = getattr(
            item,
            "chunk_id",
            f"unknown-{index}",
        )

        score = getattr(
            item,
            "score",
            0.0,
        )

        metadata = getattr(
            item,
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            Mapping,
        ):

            metadata = {}

        merged_metadata = dict(
            metadata
        )

        merged_metadata.setdefault(
            "document_id",
            str(document_id),
        )

        owner_id = getattr(
            item,
            "owner_id",
            None,
        )

        if owner_id is not None:

            merged_metadata.setdefault(
                "owner_id",
                owner_id,
            )

        classification = getattr(
            item,
            "classification",
            None,
        )

        if classification is not None:

            merged_metadata.setdefault(
                "classification",
                classification,
            )

        return RetrievedChunk(

            chunk_id=str(
                chunk_id
            ),

            document_id=str(
                document_id
            ),

            text=str(
                text
            ),

            score=float(
                score
            ),

            metadata=merged_metadata,

        )

    # =========================================================================
    # Query validation
    # =========================================================================

    @staticmethod
    def _validate_query(
        query: str,
    ) -> str:

        if not isinstance(
            query,
            str,
        ):

            raise InvalidQueryError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:

            raise InvalidQueryError(
                "Query cannot be empty."
            )

        # Avoid uncontrolled memory usage and
        # embedding abuse.

        if len(query) > 10_000:

            raise InvalidQueryError(
                "Query is too long."
            )

        return query


# =============================================================================
# Factory
# =============================================================================


def create_secure_retriever(
    embedder: EmbedderProtocol,
    vector_store: VectorStoreProtocol,
    permission_manager: PermissionManager,
    *,
    top_k: int = 5,
) -> SecureRetriever:
    """
    Convenience factory.
    """

    return SecureRetriever(

        embedder=embedder,

        vector_store=vector_store,

        permission_manager=permission_manager,

        default_top_k=top_k,

    )