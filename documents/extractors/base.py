"""
documents/extractors/base.py

Base classes for document extractors.

Python 3.12+
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


# =============================================================================
# Models
# =============================================================================

@dataclass(slots=True)
class ExtractedDocument:
    """
    Standard output produced by every extractor.
    """

    text: str

    pages: int

    source: Path

    metadata: dict[str, str] = field(default_factory=dict)


# =============================================================================
# Base Extractor
# =============================================================================

class BaseExtractor(ABC):
    """
    Base class for every extractor.
    """

    @abstractmethod
    def supports(
        self,
        suffix: str,
    ) -> bool:
        """
        Returns True if extractor supports file extension.
        """
        raise NotImplementedError

    @abstractmethod
    def extract(
        self,
        file_path: Path,
    ) -> ExtractedDocument:
        """
        Extract text from document.
        """
        raise NotImplementedError