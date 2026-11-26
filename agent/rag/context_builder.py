"""
agent/rag/context_builder.py

Validates and builds a trusted RAG context for WW-P Security AI Agent.

Responsibilities:
- Validate retrieved RAG context.
- Verify user ownership.
- Validate retrieval statistics.
- Validate source attribution metadata.
- Enforce context size limits.
- Produce an immutable BuiltRAGContext.

Python 3.10+
"""

from dataclasses import dataclass, field
from typing import Any

from authentication.auth import UserContext
from agent.rag.adapter import AgentRAGContext
from agent.rag.sources import RAGSource


# ============================================================
# Exceptions
# ============================================================

class ContextBuilderError(Exception):
    """Base exception for RAG context builder errors."""


class ContextValidationError(ContextBuilderError):
    """Raised when retrieved context is invalid."""


class ContextSecurityError(ContextBuilderError):
    """Raised when a security or ownership validation fails."""


class ContextSizeError(ContextBuilderError):
    """Raised when the retrieved context exceeds configured limits."""


# ============================================================
# Built RAG Context
# ============================================================

@dataclass(frozen=True, slots=True)
class BuiltRAGContext:
    """
    Immutable, validated RAG context ready for AgentCore.

    The context contains only information returned by the
    authorized retrieval pipeline.

    Sources are preserved as structured metadata rather than
    reconstructed from the context text.
    """

    query: str
    context: str
    user_id: str

    candidates_count: int
    authorized_count: int
    denied_count: int

    sources: tuple[RAGSource, ...] = field(default_factory=tuple)

    @property
    def empty(self) -> bool:
        """Return True when no authorized context is available."""
        return not bool(self.context.strip())

    @property
    def context_chars(self) -> int:
        """Return the number of characters in the context."""
        return len(self.context)

    @property
    def source_count(self) -> int:
        """Return the number of attributed sources."""
        return len(self.sources)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the built context into a serializable dictionary.

        Source metadata is explicitly included.
        """

        return {
            "query": self.query,
            "context": self.context,
            "user_id": self.user_id,
            "candidates_count": self.candidates_count,
            "authorized_count": self.authorized_count,
            "denied_count": self.denied_count,
            "empty": self.empty,
            "context_chars": self.context_chars,
            "source_count": self.source_count,
            "sources": [
                {
                    "citation_id": source.citation_id,
                    "document_id": source.document_id,
                    "chunk_id": source.chunk_id,
                    "score": source.score,
                }
                for source in self.sources
            ],
        }


# ============================================================
# Context Builder
# ============================================================

class RAGContextBuilder:
    """
    Builds and validates an AgentRAGContext.

    Security principles:
    - User identity must match the retrieval result.
    - Only authorized retrieval context is accepted.
    - Retrieval statistics must be consistent.
    - Source metadata must be structurally valid.
    - Context size is limited.
    - Empty context can be explicitly allowed or rejected.
    """

    def __init__(
        self,
        max_context_chars: int = 24000,
        allow_empty_context: bool = True,
    ) -> None:

        if (
            not isinstance(max_context_chars, int)
            or isinstance(max_context_chars, bool)
            or max_context_chars <= 0
        ):
            raise ContextValidationError(
                "max_context_chars must be a positive integer."
            )

        if not isinstance(allow_empty_context, bool):
            raise ContextValidationError(
                "allow_empty_context must be a boolean."
            )

        self.max_context_chars = max_context_chars
        self.allow_empty_context = allow_empty_context

    # ========================================================
    # Public API
    # ========================================================

    def build(
        self,
        retrieved_context: AgentRAGContext,
        user: UserContext,
    ) -> BuiltRAGContext:
        """
        Validate and build an immutable RAG context.

        Args:
            retrieved_context:
                Context returned by RAGAdapter.retrieve().

            user:
                Authenticated user requesting the context.

        Returns:
            BuiltRAGContext

        Raises:
            ContextValidationError
            ContextSecurityError
            ContextSizeError
        """

        self._validate_user(user)

        self._validate_retrieved_context(retrieved_context)

        self._validate_ownership(
            retrieved_context=retrieved_context,
            user=user,
        )

        self._validate_statistics(retrieved_context)

        context = self._normalize_context(
            retrieved_context.context
        )

        self._validate_context_size(context)

        sources = self._validate_sources(
            retrieved_context.sources,
            retrieved_context.authorized_count,
        )

        self._validate_empty_context(
            context=context,
            authorized_count=retrieved_context.authorized_count,
            sources=sources,
        )

        return BuiltRAGContext(
            query=retrieved_context.query.strip(),
            context=context,
            user_id=user.user_id,
            candidates_count=retrieved_context.candidates_count,
            authorized_count=retrieved_context.authorized_count,
            denied_count=retrieved_context.denied_count,
            sources=sources,
        )

    # ========================================================
    # User Validation
    # ========================================================

    @staticmethod
    def _validate_user(user: UserContext) -> None:
        """Validate authenticated user context."""

        if not isinstance(user, UserContext):
            raise ContextSecurityError(
                "A valid UserContext is required."
            )

        if not isinstance(user.user_id, str) or not user.user_id.strip():
            raise ContextSecurityError(
                "Invalid user_id."
            )

        if not isinstance(user.username, str) or not user.username.strip():
            raise ContextSecurityError(
                "Invalid username."
            )

        if not isinstance(user.session_id, str) or not user.session_id.strip():
            raise ContextSecurityError(
                "Invalid session_id."
            )

    # ========================================================
    # Retrieved Context Validation
    # ========================================================

    @staticmethod
    def _validate_retrieved_context(
        retrieved_context: AgentRAGContext,
    ) -> None:
        """Validate the object returned by RAGAdapter."""

        if not isinstance(retrieved_context, AgentRAGContext):
            raise ContextValidationError(
                "Expected an AgentRAGContext instance."
            )

        if (
            not isinstance(retrieved_context.query, str)
            or not retrieved_context.query.strip()
        ):
            raise ContextValidationError(
                "Retrieved query must be a non-empty string."
            )

        if not isinstance(retrieved_context.context, str):
            raise ContextValidationError(
                "Retrieved context must be a string."
            )

        if (
            not isinstance(retrieved_context.user_id, str)
            or not retrieved_context.user_id.strip()
        ):
            raise ContextValidationError(
                "Retrieved context contains an invalid user_id."
            )

    # ========================================================
    # Ownership Validation
    # ========================================================

    @staticmethod
    def _validate_ownership(
        retrieved_context: AgentRAGContext,
        user: UserContext,
    ) -> None:
        """
        Verify that the retrieved context belongs to the
        currently authenticated user.
        """

        if retrieved_context.user_id != user.user_id:
            raise ContextSecurityError(
                "RAG context ownership mismatch."
            )

    # ========================================================
    # Retrieval Statistics
    # ========================================================

    @staticmethod
    def _validate_statistics(
        retrieved_context: AgentRAGContext,
    ) -> None:
        """
        Validate retrieval statistics.

        All statistics must be non-negative integers.
        Boolean values are explicitly rejected because bool
        is a subclass of int in Python.
        """

        statistics = {
            "candidates_count": retrieved_context.candidates_count,
            "authorized_count": retrieved_context.authorized_count,
            "denied_count": retrieved_context.denied_count,
        }

        for name, value in statistics.items():

            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ContextValidationError(
                    f"Invalid retrieval statistic: {name}."
                )

        candidates = retrieved_context.candidates_count
        authorized = retrieved_context.authorized_count
        denied = retrieved_context.denied_count

        if authorized > candidates:
            raise ContextValidationError(
                "authorized_count cannot exceed candidates_count."
            )

        if denied > candidates:
            raise ContextValidationError(
                "denied_count cannot exceed candidates_count."
            )

        if authorized + denied > candidates:
            raise ContextValidationError(
                "Authorized and denied counts exceed candidates_count."
            )

    # ========================================================
    # Context Normalization
    # ========================================================

    @staticmethod
    def _normalize_context(context: str) -> str:
        """
        Normalize context without changing its substantive content.

        Leading and trailing whitespace is removed.
        """

        if not isinstance(context, str):
            raise ContextValidationError(
                "Context must be a string."
            )

        return context.strip()

    # ========================================================
    # Context Size Validation
    # ========================================================

    def _validate_context_size(self, context: str) -> None:
        """Enforce the configured context size limit."""

        if len(context) > self.max_context_chars:
            raise ContextSizeError(
                "Retrieved context exceeds the configured limit: "
                f"{len(context)} > {self.max_context_chars} characters."
            )

    # ========================================================
    # Source Validation
    # ========================================================

    @staticmethod
    def _validate_sources(
        sources: tuple[RAGSource, ...],
        authorized_count: int,
    ) -> tuple[RAGSource, ...]:
        """
        Validate structured source attribution metadata.

        Sources must:
        - Be provided as a tuple.
        - Contain only RAGSource instances.
        - Have unique citation IDs.
        - Have sequential citation IDs starting from 1.
        - Match the number of authorized chunks.

        The source metadata is not reconstructed from plain text.
        """

        if not isinstance(sources, tuple):
            raise ContextValidationError(
                "Sources must be provided as a tuple."
            )

        if len(sources) != authorized_count:
            raise ContextValidationError(
                "Source count does not match authorized_count."
            )

        citation_ids: list[int] = []

        for source in sources:

            if not isinstance(source, RAGSource):
                raise ContextValidationError(
                    "Invalid source object. Expected RAGSource."
                )

            if (
                not isinstance(source.citation_id, int)
                or isinstance(source.citation_id, bool)
                or source.citation_id <= 0
            ):
                raise ContextValidationError(
                    "Invalid source citation_id."
                )

            if (
                not isinstance(source.document_id, str)
                or not source.document_id.strip()
            ):
                raise ContextValidationError(
                    "Invalid source document_id."
                )

            if (
                not isinstance(source.chunk_id, str)
                or not source.chunk_id.strip()
            ):
                raise ContextValidationError(
                    "Invalid source chunk_id."
                )

            citation_ids.append(source.citation_id)

        expected_ids = list(range(1, len(sources) + 1))

        if citation_ids != expected_ids:
            raise ContextValidationError(
                "Source citation IDs must be sequential, starting from 1."
            )

        return sources

    # ========================================================
    # Empty Context Validation
    # ========================================================

    def _validate_empty_context(
        self,
        context: str,
        authorized_count: int,
        sources: tuple[RAGSource, ...],
    ) -> None:
        """
        Validate consistency between context, authorized chunks
        and source metadata.
        """

        is_empty = not bool(context)

        if is_empty and authorized_count > 0:
            raise ContextValidationError(
                "Context is empty despite authorized retrieval results."
            )

        if not is_empty and authorized_count == 0:
            raise ContextValidationError(
                "Context contains data but authorized_count is zero."
            )

        if is_empty and sources:
            raise ContextValidationError(
                "Empty context cannot contain attributed sources."
            )

        if not is_empty and not sources:
            raise ContextValidationError(
                "Non-empty context must contain source attribution metadata."
            )

        if is_empty and not self.allow_empty_context:
            raise ContextValidationError(
                "Empty RAG context is not allowed."
            )