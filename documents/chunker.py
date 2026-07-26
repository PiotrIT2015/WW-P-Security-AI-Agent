"""
documents/chunker.py

Document chunking engine for Secure RAG.

Python 3.12+
"""

from __future__ import annotations

import hashlib
import re

from dataclasses import dataclass, field
from pathlib import Path

from documents.extractors import ExtractedDocument


# =============================================================================
# Exceptions
# =============================================================================


class ChunkerError(Exception):
    """Raised when document chunking fails."""

    pass


# =============================================================================
# Models
# =============================================================================


@dataclass(slots=True)
class DocumentChunk:
    """
    A single chunk prepared for embedding and vector storage.
    """

    chunk_id: str

    document_id: str

    text: str

    position: int

    source: str

    page: int | None = None

    metadata: dict[str, str] = field(
        default_factory=dict
    )


# =============================================================================
# Chunker
# =============================================================================


class DocumentChunker:
    """
    Splits extracted documents into embedding-ready chunks.

    Parameters
    ----------
    chunk_size:
        Maximum approximate size of a chunk in characters.

    overlap:
        Number of characters from the previous chunk
        retained as context.

    min_chunk_size:
        Chunks shorter than this value are normally merged
        with the next chunk.
    """

    def __init__(
        self,
        chunk_size: int = 1200,
        overlap: int = 200,
        min_chunk_size: int = 100,
    ) -> None:

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero"
            )

        if overlap < 0:
            raise ValueError(
                "overlap cannot be negative"
            )

        if overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size"
            )

        if min_chunk_size <= 0:
            raise ValueError(
                "min_chunk_size must be greater than zero"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    # =========================================================================
    # Public API
    # =========================================================================

    def chunk(
        self,
        document: ExtractedDocument,
        metadata: dict[str, str] | None = None,
    ) -> list[DocumentChunk]:
        """
        Split ExtractedDocument into DocumentChunk objects.
        """

        if not document.text.strip():
            return []

        document_id = self._document_id(
            document
        )

        normalized = self._normalize(
            document.text
        )

        paragraphs = self._paragraphs(
            normalized
        )

        sections = self._build_sections(
            paragraphs
        )

        chunks = self._build_chunks(
            sections
        )

        result: list[DocumentChunk] = []

        for index, text in enumerate(chunks):

            chunk_id = self._chunk_id(
                document_id=document_id,
                position=index,
                text=text,
            )

            chunk_metadata = dict(
                metadata or {}
            )

            chunk_metadata.update(
                {
                    "document_id": document_id,
                    "position": str(index),
                    "source": str(document.source),
                }
            )

            result.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    text=text,
                    position=index,
                    source=str(document.source),
                    metadata=chunk_metadata,
                )
            )

        return result

    # =========================================================================
    # Normalization
    # =========================================================================

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:
        """
        Normalize excessive whitespace without destroying
        paragraph boundaries.
        """

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    # =========================================================================
    # Paragraph processing
    # =========================================================================

    @staticmethod
    def _paragraphs(
        text: str,
    ) -> list[str]:
        """
        Split text into paragraphs.
        """

        paragraphs = re.split(
            r"\n\s*\n",
            text,
        )

        return [
            paragraph.strip()
            for paragraph in paragraphs
            if paragraph.strip()
        ]

    # =========================================================================
    # Section processing
    # =========================================================================

    def _build_sections(
        self,
        paragraphs: list[str],
    ) -> list[str]:
        """
        Build semantic-ish sections from paragraphs.

        A section attempts to preserve paragraph boundaries.
        """

        sections: list[str] = []

        current: list[str] = []
        current_size = 0

        for paragraph in paragraphs:

            paragraph_size = len(
                paragraph
            )

            if (
                current
                and current_size + paragraph_size + 2
                > self.chunk_size
            ):

                sections.append(
                    "\n\n".join(current)
                )

                current = []
                current_size = 0

            current.append(
                paragraph
            )

            current_size += (
                paragraph_size + 2
            )

        if current:

            sections.append(
                "\n\n".join(current)
            )

        return sections

    # =========================================================================
    # Chunk construction
    # =========================================================================

    def _build_chunks(
        self,
        sections: list[str],
    ) -> list[str]:
        """
        Convert sections into final chunks and apply overlap.
        """

        if not sections:
            return []

        chunks: list[str] = []

        for section in sections:

            if len(section) <= self.chunk_size:

                chunks.append(
                    section
                )

                continue

            chunks.extend(
                self._split_large_section(
                    section
                )
            )

        return self._merge_small_chunks(
            chunks
        )

    # =========================================================================
    # Large section splitting
    # =========================================================================

    def _split_large_section(
        self,
        text: str,
    ) -> list[str]:
        """
        Split an oversized section while trying to preserve
        sentence boundaries.
        """

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        chunks: list[str] = []

        current = ""

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            candidate = (
                f"{current} {sentence}"
                if current
                else sentence
            )

            if (
                current
                and len(candidate) > self.chunk_size
            ):

                chunks.append(
                    current.strip()
                )

                overlap_text = self._overlap_text(
                    current
                )

                current = (
                    f"{overlap_text} {sentence}"
                ).strip()

            else:

                current = candidate

        if current:

            chunks.append(
                current.strip()
            )

        return chunks

    # =========================================================================
    # Overlap
    # =========================================================================

    def _overlap_text(
        self,
        text: str,
    ) -> str:
        """
        Return the trailing overlap portion of a chunk.
        """

        if self.overlap == 0:
            return ""

        return text[
            max(
                0,
                len(text) - self.overlap,
            ):
        ].strip()

    # =========================================================================
    # Small chunks
    # =========================================================================

    def _merge_small_chunks(
        self,
        chunks: list[str],
    ) -> list[str]:
        """
        Merge very small chunks with their successor where possible.
        """

        if len(chunks) <= 1:
            return chunks

        result: list[str] = []

        buffer = ""

        for chunk in chunks:

            if not buffer:

                buffer = chunk

                continue

            if (
                len(buffer) < self.min_chunk_size
                and len(buffer) + len(chunk) + 2
                <= self.chunk_size
            ):

                buffer = (
                    f"{buffer}\n\n{chunk}"
                )

            else:

                result.append(
                    buffer
                )

                buffer = chunk

        if buffer:

            result.append(
                buffer
            )

        return result

    # =========================================================================
    # IDs
    # =========================================================================

    @staticmethod
    def _document_id(
        document: ExtractedDocument,
    ) -> str:
        """
        Generate stable document identifier.

        Source path alone is intentionally not used because
        the same document may be moved.
        """

        digest = hashlib.sha256(
            document.text.encode(
                "utf-8"
            )
        ).hexdigest()

        return digest

    # -------------------------------------------------------------------------

    @staticmethod
    def _chunk_id(
        document_id: str,
        position: int,
        text: str,
    ) -> str:
        """
        Generate stable chunk identifier.
        """

        payload = (
            f"{document_id}:"
            f"{position}:"
            f"{text}"
        )

        return hashlib.sha256(
            payload.encode(
                "utf-8"
            )
        ).hexdigest()


# =============================================================================
# Global instance
# =============================================================================


document_chunker = DocumentChunker()