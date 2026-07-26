"""
documents/extractors/txt.py

Plain text extractor.

Python 3.12+
"""

from __future__ import annotations

from pathlib import Path

from .base import (
    BaseExtractor,
    ExtractedDocument,
)


class TXTExtractor(BaseExtractor):

    SUPPORTED = {
        ".txt",
    }

    def supports(
        self,
        suffix: str,
    ) -> bool:

        return suffix.lower() in self.SUPPORTED

    def extract(
        self,
        file_path: Path,
    ) -> ExtractedDocument:

        try:

            text = file_path.read_text(
                encoding="utf-8"
            )

        except UnicodeDecodeError:

            text = file_path.read_text(
                encoding="latin-1"
            )

        return ExtractedDocument(

            text=text,

            pages=0,

            source=file_path,

            metadata={
                "type": ".txt",
                "extractor": "plain-text",
            },
        )