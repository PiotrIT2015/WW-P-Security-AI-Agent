"""
documents/metadata.py

Document metadata management
for Secure RAG.

Python 3.12+
"""

from __future__ import annotations


import hashlib
import mimetypes

from dataclasses import dataclass, field

from datetime import datetime, timezone

from pathlib import Path



# =============================================================================
# Model
# =============================================================================


@dataclass(slots=True)
class DocumentMetadata:
    """
    Metadata attached to document.
    """

    filename: str

    path: str

    sha256: str

    mime_type: str

    extension: str

    size: int

    created_at: str

    indexed_at: str

    pages: int

    language: str

    classification: str

    owner: str

    tags: list[str] = field(
        default_factory=list
    )



# =============================================================================
# Metadata Engine
# =============================================================================


class MetadataEngine:
    """
    Generates document metadata.

    Used before:
        - chunking
        - embeddings
        - vector storage
    """

    # ---------------------------------------------------------

    def generate(
        self,
        file_path: str | Path,
        pages: int = 0,
        language: str = "unknown",
        owner: str = "system",
    ) -> DocumentMetadata:


        path = Path(
            file_path
        )


        return DocumentMetadata(

            filename=path.name,

            path=str(
                path.absolute()
            ),

            sha256=self._hash(
                path
            ),

            mime_type=self._mime(
                path
            ),

            extension=path.suffix.lower(),

            size=path.stat().st_size,

            created_at=self._timestamp(
                path.stat().st_ctime
            ),

            indexed_at=self._timestamp(),

            pages=pages,

            language=language,

            classification="internal",

            owner=owner,

        )


    # ---------------------------------------------------------

    def _hash(
        self,
        path: Path,
    ) -> str:

        sha = hashlib.sha256()


        with path.open(
            "rb"
        ) as file:

            while chunk := file.read(
                8192
            ):

                sha.update(
                    chunk
                )


        return sha.hexdigest()


    # ---------------------------------------------------------

    def _mime(
        self,
        path: Path,
    ) -> str:

        mime, _ = mimetypes.guess_type(
            str(path)
        )


        return mime or (
            "application/octet-stream"
        )


    # ---------------------------------------------------------

    def _timestamp(
        self,
        value: float | None = None,
    ) -> str:


        if value:

            date = datetime.fromtimestamp(
                value,
                tz=timezone.utc,
            )

        else:

            date = datetime.now(
                timezone.utc
            )


        return date.isoformat()



# =============================================================================
# Global instance
# =============================================================================


metadata_engine = MetadataEngine()