"""
security/sanitizer.py

WW-P Secure RAG

Input / Output Sanitization Module

Stage:
5.4

Python 3.12+
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import unicodedata


logger = logging.getLogger(
    "secure_rag.security.sanitizer"
)


# ============================================================
# Configuration
# ============================================================


@dataclass(slots=True)
class SanitizerConfig:
    """
    Sanitizer configuration.
    """

    max_length: int = 8000

    normalize_unicode: bool = True

    remove_control_chars: bool = True

    remove_scripts: bool = True

    collapse_spaces: bool = True

    enable_logging: bool = True



# ============================================================
# Sanitizer
# ============================================================


class Sanitizer:
    """
    Text sanitization engine.

    Responsibilities:

    - normalize text
    - remove dangerous characters
    - clean LLM output
    - enforce limits
    """



    def __init__(
        self,
        config: SanitizerConfig | None = None,
    ) -> None:


        self.config = (
            config
            or SanitizerConfig()
        )


        logger.info(
            "Sanitizer initialized."
        )



    # --------------------------------------------------------

    def normalize(
        self,
        text: str,
    ) -> str:
        """
        Unicode normalization.
        """

        if not self.config.normalize_unicode:

            return text


        return unicodedata.normalize(
            "NFKC",
            text
        )



    # --------------------------------------------------------

    def remove_control_characters(
        self,
        text: str,
    ) -> str:
        """
        Remove invisible control chars.
        """

        if not self.config.remove_control_chars:

            return text


        return "".join(
            char
            for char in text
            if not unicodedata.category(
                char
            ).startswith("C")
        )



    # --------------------------------------------------------

    def remove_dangerous_patterns(
        self,
        text: str,
    ) -> str:
        """
        Remove suspicious script fragments.
        """


        if not self.config.remove_scripts:

            return text


        patterns = [

            r"<script.*?>.*?</script>",

            r"javascript:",

            r"data:text/html",

            r"eval\(",

        ]


        cleaned = text


        for pattern in patterns:

            cleaned = re.sub(
                pattern,
                "",
                cleaned,
                flags=re.IGNORECASE |
                re.DOTALL,
            )


        return cleaned



    # --------------------------------------------------------

    def collapse_whitespace(
        self,
        text: str,
    ) -> str:
        """
        Normalize spaces.
        """

        if not self.config.collapse_spaces:

            return text


        return re.sub(
            r"\s+",
            " ",
            text
        ).strip()



    # --------------------------------------------------------

    def limit_length(
        self,
        text: str,
    ) -> str:
        """
        Limit output size.
        """

        return (
            text[:self.config.max_length]
        )



    # --------------------------------------------------------

    def clean(
        self,
        text: str,
    ) -> str:
        """
        Full sanitization pipeline.
        """


        if not isinstance(
            text,
            str
        ):

            raise TypeError(
                "Text must be string."
            )


        original_length = len(
            text
        )


        text = self.normalize(
            text
        )


        text = (
            self.remove_control_characters(
                text
            )
        )


        text = (
            self.remove_dangerous_patterns(
                text
            )
        )


        text = (
            self.collapse_whitespace(
                text
            )
        )


        text = (
            self.limit_length(
                text
            )
        )


        if self.config.enable_logging:

            logger.info(
                "Sanitized text: %d -> %d chars",
                original_length,
                len(text),
            )


        return text



    # --------------------------------------------------------

    def sanitize_input(
        self,
        text: str,
    ) -> str:
        """
        Alias for input processing.
        """

        return self.clean(
            text
        )



    # --------------------------------------------------------

    def sanitize_output(
        self,
        text: str,
    ) -> str:
        """
        Alias for LLM response cleaning.
        """

        return self.clean(
            text
        )