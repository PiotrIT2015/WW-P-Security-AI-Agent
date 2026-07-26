"""
documents/extractors/markdown.py

Markdown extractor.

Python 3.12+
"""

from __future__ import annotations

from pathlib import Path

import re

from .base import (
    BaseExtractor,
    ExtractedDocument,
)


class MarkdownExtractor(BaseExtractor):
    """
    Extract text from Markdown files.
    """


    SUPPORTED = {
        ".md",
        ".markdown",
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


        content = file_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )


        # Remove markdown syntax

        text = re.sub(
            r"#{1,6}\s",
            "",
            content,
        )


        text = re.sub(
            r"\*\*(.*?)\*\*",
            r"\1",
            text,
        )


        text = re.sub(
            r"\[(.*?)\]\(.*?\)",
            r"\1",
            text,
        )


        return ExtractedDocument(

            text=text,

            pages=0,

            source=file_path,

            metadata={

                "type":
                    "markdown",

                "extractor":
                    "markdown-parser",

            },

        )