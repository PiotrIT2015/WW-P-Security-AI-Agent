"""LLM generation adapter used by the Secure RAG application."""
from __future__ import annotations

import logging
from typing import Protocol

from .ollama_client import OllamaClient, OllamaClientError

logger = logging.getLogger("secure_rag.llm.generator")


class TextGenerationClient(Protocol):
    def generate(self, prompt: str) -> str: ...


class LLMGenerator:
    """Small adapter exposing the ``generate(prompt) -> str`` app interface."""

    def __init__(self, client: TextGenerationClient | None = None) -> None:
        self.client = client or OllamaClient()

    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        try:
            result = self.client.generate(prompt)
        except OllamaClientError:
            logger.exception("Ollama generation request failed.")
            raise
        if not isinstance(result, str) or not result.strip():
            raise OllamaClientError("LLM provider returned an empty response.")
        return result.strip()
