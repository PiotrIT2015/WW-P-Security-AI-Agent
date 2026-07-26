"""
documents/extractors/pdf.py

PDF document extractor.

Uses pypdf.

Python 3.12+
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from .base import (
    BaseExtractor,
    ExtractedDocument,
)


class PDFExtractor(BaseExtractor):
    """
    Extract text from PDF files.
    """

    SUPPORTED = {
        ".pdf"
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


        reader = PdfReader(
            str(file_path)
        )


        pages = []

        for page in reader.pages:

            text = page.extract_text()

            if text:

                pages.append(
                    text
                )


        content = "\n".join(
            pages
        )


        return ExtractedDocument(

            text=content,

            pages=len(
                reader.pages
            ),

            source=file_path,

            metadata={

                "type":
                    "pdf",

                "extractor":
                    "pypdf",

            },

        )