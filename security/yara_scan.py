"""
security/yara_scan.py

YARA malware and threat pattern scanner.

Secure RAG security layer.

Python 3.12+
"""

from __future__ import annotations


from pathlib import Path


import tempfile


from config import settings



class YaraScannerError(Exception):
    """
    YARA scanner exception.
    """
    pass



class YaraScanner:
    """
    YARA based document scanner.

    Detects:
    - malicious patterns
    - scripts
    - suspicious instructions
    - hidden payloads
    """

    def __init__(
        self,
        rules_path: str | Path =
        "security/rules/secure_rag_rules.yar",
    ):

        self.enabled = settings.ENABLE_YARA

        self.rules_path = Path(
            rules_path
        )

        self.rules = None


        if self.enabled:

            self._load_rules()



    # ==================================================
    # Rules loading
    # ==================================================

    def _load_rules(self):
        """
        Load YARA rules.
        """

        try:

            import yara


            if not self.rules_path.exists():

                raise YaraScannerError(
                    "YARA rules file missing"
                )


            self.rules = yara.compile(
                filepath=str(
                    self.rules_path
                )
            )


        except ImportError:

            self.rules = None



        except Exception as error:

            raise YaraScannerError(
                f"YARA initialization failed: {error}"
            )



    # ==================================================
    # Scan file
    # ==================================================

    def scan_file(
        self,
        file_path: str | Path,
    ) -> dict:
        """
        Scan file with YARA.
        """

        path = Path(
            file_path
        )


        if not path.exists():

            raise YaraScannerError(
                "File not found"
            )


        result = {

            "file":
                path.name,

            "matched":
                False,

            "rules":
                [],

        }


        if not self.enabled:

            result["status"] = (
                "disabled"
            )

            return result



        if self.rules is None:

            result["status"] = (
                "yara unavailable"
            )

            return result



        try:

            matches = self.rules.match(
                str(path)
            )


            if matches:

                result["matched"] = True


                result["rules"] = [

                    match.rule

                    for match in matches

                ]


            result["status"] = (
                "completed"
            )


            return result



        except Exception as error:


            raise YaraScannerError(
                f"Scan failed: {error}"
            )



    # ==================================================
    # Scan text
    # ==================================================

    def scan_text(
        self,
        text: str,
    ) -> dict:
        """
        Scan extracted document text.

        Useful for:
        - prompt injection detection
        - hidden instructions
        """

        temporary = None


        try:

            with tempfile.NamedTemporaryFile(
                mode="w",
                delete=False,
                encoding="utf-8",
            ) as file:


                file.write(
                    text
                )

                temporary = file.name



            return self.scan_file(
                temporary
            )


        finally:

            if temporary:

                Path(
                    temporary
                ).unlink(
                    missing_ok=True
                )



# Global instance

yara_scanner = YaraScanner()