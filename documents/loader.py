"""
documents/loader.py

Secure document loader.

Python 3.12+
"""

from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass
from pathlib import Path

from config import settings
from security.antivirus import antivirus
from security.yara_scan import yara_scanner


# =============================================================================
# Exceptions
# =============================================================================

class DocumentLoaderError(Exception):
    """Raised when a document cannot be loaded."""
    pass


# =============================================================================
# Descriptor
# =============================================================================

@dataclass(slots=True)
class DocumentDescriptor:
    """
    Metadata describing an uploaded document.
    """

    path: Path
    filename: str
    extension: str
    mime_type: str
    size: int
    sha256: str

    antivirus: dict
    yara: dict


# =============================================================================
# Loader
# =============================================================================

class DocumentLoader:
    """
    Secure document loader.

    Responsibilities:
        - file validation
        - extension validation
        - size validation
        - SHA256 calculation
        - ClamAV scan
        - YARA scan
    """

    def __init__(self):

        self.allowed = {
            ext.lower()
            for ext in settings.ALLOWED_EXTENSIONS
        }

        self.max_size = (
            settings.MAX_UPLOAD_SIZE_MB
            * 1024
            * 1024
        )

    # ------------------------------------------------------------------

    def load(
        self,
        file_path: str | Path,
    ) -> DocumentDescriptor:

        path = Path(file_path)

        self._validate_exists(path)
        self._validate_extension(path)
        self._validate_size(path)

        sha256 = self._sha256(path)

        av_report = antivirus.scan(path)

        if av_report.get("infected"):

            raise DocumentLoaderError(
                "Malware detected."
            )

        yara_report = yara_scanner.scan_file(path)

        mime, _ = mimetypes.guess_type(path)

        return DocumentDescriptor(

            path=path,

            filename=path.name,

            extension=path.suffix.lower(),

            mime_type=mime or "application/octet-stream",

            size=path.stat().st_size,

            sha256=sha256,

            antivirus=av_report,

            yara=yara_report,

        )

    # ------------------------------------------------------------------

    def _validate_exists(
        self,
        path: Path,
    ) -> None:

        if not path.exists():

            raise DocumentLoaderError(
                f"File not found: {path}"
            )

        if not path.is_file():

            raise DocumentLoaderError(
                "Path is not a file."
            )

    # ------------------------------------------------------------------

    def _validate_extension(
        self,
        path: Path,
    ) -> None:

        if path.suffix.lower() not in self.allowed:

            raise DocumentLoaderError(
                f"Unsupported file type: {path.suffix}"
            )

    # ------------------------------------------------------------------

    def _validate_size(
        self,
        path: Path,
    ) -> None:

        size = path.stat().st_size

        if size > self.max_size:

            raise DocumentLoaderError(
                "File exceeds maximum allowed size."
            )

    # ------------------------------------------------------------------

    @staticmethod
    def _sha256(
        path: Path,
    ) -> str:

        digest = hashlib.sha256()

        with path.open("rb") as file:

            while chunk := file.read(8192):

                digest.update(chunk)

        return digest.hexdigest()


# =============================================================================
# Global loader
# =============================================================================

document_loader = DocumentLoader()