"""
security/antivirus.py

Antivirus scanner module.

Uses ClamAV when available.
"""

from __future__ import annotations


from pathlib import Path


import hashlib


from config import settings



class AntivirusError(Exception):
    pass



class AntivirusScanner:
    """
    File security scanner.

    Checks:
    - SHA256 hash
    - ClamAV malware scan
    """


    def __init__(self):

        self.enabled = settings.ENABLE_CLAMAV



    # --------------------------------------------------

    def calculate_hash(
        self,
        file_path: str | Path,
    ) -> str:
        """
        Calculate SHA256 hash.
        """

        path = Path(file_path)


        sha256 = hashlib.sha256()


        with open(
            path,
            "rb"
        ) as file:

            for block in iter(
                lambda: file.read(4096),
                b""
            ):

                sha256.update(
                    block
                )


        return sha256.hexdigest()



    # --------------------------------------------------

    def scan(
        self,
        file_path: str | Path,
    ) -> dict:
        """
        Scan file.

        Returns security report.
        """

        path = Path(file_path)


        if not path.exists():

            raise AntivirusError(
                "File does not exist"
            )


        result = {

            "file":
                path.name,

            "sha256":
                self.calculate_hash(path),

            "infected":
                False,

            "scanner":
                "SHA256",

        }


        if self.enabled:

            result.update(
                self._clamav_scan(path)
            )


        return result



    # --------------------------------------------------

    def _clamav_scan(
        self,
        path: Path,
    ) -> dict:
        """
        ClamAV integration.

        If ClamAV is unavailable,
        system continues in safe mode.
        """

        try:

            import pyclamd


            cd = pyclamd.ClamdUnixSocket()


            scan = cd.scan_file(
                str(path)
            )


            if scan:

                return {

                    "infected":
                        True,

                    "threat":
                        scan,

                }


            return {

                "infected":
                    False

            }


        except Exception:


            return {

                "clamav":
                    "unavailable",

                "infected":
                    False,

            }



# Global instance

antivirus = AntivirusScanner()