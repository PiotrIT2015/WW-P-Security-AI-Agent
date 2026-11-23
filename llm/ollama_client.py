"""Minimal, dependency-free client for the local Ollama HTTP API."""
from __future__ import annotations

import json
import logging
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger("secure_rag.llm.ollama")


class OllamaClientError(RuntimeError):
    """Raised when Ollama cannot process a generation request."""


class OllamaClient:
    """HTTP client for Ollama's non-streaming ``/api/generate`` endpoint."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "phi3:latest",
        temperature: float = 0.2,
        timeout: float = 180.0,
    ) -> None:
        base_url = base_url.strip().rstrip("/")
        model = model.strip()
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("Ollama base_url must use http:// or https://")
        if not model:
            raise ValueError("Ollama model cannot be empty.")
        if not 0.0 <= float(temperature) <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if float(timeout) <= 0:
            raise ValueError("timeout must be greater than zero")

        self.base_url = base_url
        self.model = model
        self.temperature = float(temperature)
        self.timeout = float(timeout)

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt and return Ollama's response text."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except HTTPError as error:
            # Do not include request/prompt data in exceptions or logs.
            detail = error.read(2048).decode("utf-8", errors="replace")
            raise OllamaClientError(
                f"Ollama HTTP error {error.code}: {detail[:500]}"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise OllamaClientError(
                "Cannot connect to Ollama. Check that the Ollama service "
                "is running and the configured URL is correct."
            ) from error

        try:
            data: Any = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OllamaClientError("Ollama returned invalid JSON.") from error

        if not isinstance(data, dict):
            raise OllamaClientError("Ollama returned an unexpected response format.")
        if data.get("error"):
            raise OllamaClientError(f"Ollama error: {str(data['error'])[:500]}")

        generated = data.get("response")
        if not isinstance(generated, str) or not generated.strip():
            raise OllamaClientError("Ollama returned an empty generation.")
        return generated.strip()

    def health_check(self) -> bool:
        """Return True if Ollama responds to ``/api/tags``."""
        request = Request(
            f"{self.base_url}/api/tags",
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with urlopen(request, timeout=min(self.timeout, 10.0)) as response:
                data = json.loads(response.read().decode("utf-8"))
            return isinstance(data, dict) and isinstance(data.get("models", []), list)
        except (HTTPError, URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
