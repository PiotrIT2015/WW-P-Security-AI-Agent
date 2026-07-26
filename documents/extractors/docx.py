"""
documents/extractors/docx.py

Microsoft Word extractor.

Uses python-docx.

Python 3.12+
"""

from __future__ import annotations

from pathlib import Path

from docx import Document

from .base import (
    BaseExtractor,
    ExtractedDocument,
)


class DOCXExtractor(BaseExtractor):

    """
    Extract text from DOCX files.
    """


    SUPPORTED = {
        ".docx"
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


        document = Document(
            str(file_path)
        )


        paragraphs = []


        for paragraph in document.paragraphs:

            if paragraph.text.strip():

                paragraphs.append(
                    paragraph.text
                )


        text = "\n".join(
            paragraphs
        )


        return ExtractedDocument(

            text=text,

            pages=0,

            source=file_path,

            metadata={

                "type":
                    "docx",

                "extractor":
                    "python-docx",

            },

        )