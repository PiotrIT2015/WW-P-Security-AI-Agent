"""
documents/extractors/html.py

HTML document extractor.

Uses BeautifulSoup.

Python 3.12+
"""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from .base import (
    BaseExtractor,
    ExtractedDocument,
)


class HTMLExtractor(BaseExtractor):
    """
    Extract text from HTML documents.
    """

    SUPPORTED = {
        ".html",
        ".htm",
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

        html = file_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )


        soup = BeautifulSoup(
            html,
            "lxml",
        )


        # Remove unnecessary elements

        for element in soup(
            [
                "script",
                "style",
                "noscript",
            ]
        ):

            element.decompose()


        text = soup.get_text(
            separator="\n"
        )


        text = "\n".join(

            line.strip()

            for line in text.splitlines()

            if line.strip()

        )


        return ExtractedDocument(

            text=text,

            pages=0,

            source=file_path,

            metadata={

                "type":
                    "html",

                "extractor":
                    "beautifulsoup",

            },

        )