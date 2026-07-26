"""
Extractor package.
"""

from .base import (
    BaseExtractor,
    ExtractedDocument,
)

from .pdf import PDFExtractor
from .docx import DOCXExtractor
from .txt import TXTExtractor
from .html import HTMLExtractor
from .markdown import MarkdownExtractor


__all__ = [

    "BaseExtractor",

    "ExtractedDocument",

    "PDFExtractor",

    "DOCXExtractor",

    "TXTExtractor",

    "HTMLExtractor",

    "MarkdownExtractor",

]