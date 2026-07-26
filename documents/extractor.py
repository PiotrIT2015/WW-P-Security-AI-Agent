
"""
documents/extractor.py

Document extraction orchestrator.

Python 3.12+
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from documents.extractors import (
    BaseExtractor,
    PDFExtractor,
    DOCXExtractor,
    TXTExtractor,
    HTMLExtractor,
    MarkdownExtractor,
)


class ExtractorError(Exception):
    """Raised when document extraction cannot be performed."""
    pass


class DocumentExtractor:
    """
    Coordinates document extraction by delegating
    to registered extractors.

    Extractors are checked in registration order.
    The first extractor that supports the given file
    is used.
    """

    def __init__(self) -> None:
        self._extractors: list[BaseExtractor] = []

    # ------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------

    def register(self, extractor: BaseExtractor) -> None:
        """
        Register a document extractor.

        Args:
            extractor: Extractor implementing BaseExtractor.
        """
        if not isinstance(extractor, BaseExtractor):
            raise TypeError(
                "extractor must be an instance of BaseExtractor"
            )

        if extractor not in self._extractors:
            self._extractors.append(extractor)

    # ------------------------------------------------------------
    # Extractor selection
    # ------------------------------------------------------------

    def get_extractor(
        self,
        path: str | Path,
    ) -> BaseExtractor:
        """
        Find an extractor capable of processing the given file.

        Args:
            path: Path to the document.

        Returns:
            A suitable registered extractor.

        Raises:
            FileNotFoundError:
                If the file does not exist.
            ExtractorError:
                If no suitable extractor is registered.
        """

        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Document does not exist: {file_path}"
            )

        if not file_path.is_file():
            raise ExtractorError(
                f"Path is not a file: {file_path}"
            )

        for extractor in self._extractors:
            try:
                if extractor.supports(file_path):
                    return extractor
            except Exception as exc:
                raise ExtractorError(
                    f"Extractor {extractor.__class__.__name__} "
                    f"failed while checking {file_path}: {exc}"
                ) from exc

        raise ExtractorError(
            f"No suitable extractor found for: {file_path}"
        )

    # ------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------

    def extract(
        self,
        path: str | Path,
        **kwargs: Any,
    ) -> str:
        """
        Extract text from a document.

        Args:
            path: Path to the document.
            **kwargs: Additional arguments passed to the extractor.

        Returns:
            Extracted document text.

        Raises:
            ExtractorError:
                If extraction fails.
        """

        file_path = Path(path)
        extractor = self.get_extractor(file_path)

        try:
            result = extractor.extract(
                file_path,
                **kwargs,
            )
        except Exception as exc:
            raise ExtractorError(
                f"Failed to extract document "
                f"'{file_path}' using "
                f"{extractor.__class__.__name__}: {exc}"
            ) from exc

        if result is None:
            return ""

        return str(result)

    # ------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------

    def registered_extractors(self) -> list[str]:
        """
        Return names of currently registered extractors.
        """
        return [
            extractor.__class__.__name__
            for extractor in self._extractors
        ]


# =============================================================================
# Global document extractor
# =============================================================================

document_extractor = DocumentExtractor()

document_extractor.register(
    PDFExtractor()
)

document_extractor.register(
    DOCXExtractor()
)

document_extractor.register(
    TXTExtractor()
)

document_extractor.register(
    HTMLExtractor()
)

document_extractor.register(
    MarkdownExtractor()
)

