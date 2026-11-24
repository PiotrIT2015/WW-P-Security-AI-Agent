"""
llm/ollama_client.py

HTTP client for local Ollama API.

Supports:
- /api/generate
- /api/chat
- Native tool calling
- Health checks

WW-P Security AI Agent

Python 3.10+
"""

from __future__ import annotations

import json
import logging

from typing import Any

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


logger = logging.getLogger("secure_rag.llm.ollama")


# =============================================================================
# Exceptions
# =============================================================================

class OllamaClientError(RuntimeError):
    """Raised when Ollama cannot process a request."""


# =============================================================================
# Ollama Client
# =============================================================================

class OllamaClient:
    """
    HTTP client for the local Ollama API.

    Supports standard text generation and
    native function/tool calling.
    """

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
            raise ValueError(
                "Ollama base_url must use http:// or https://"
            )

        if not model:
            raise ValueError(
                "Ollama model cannot be empty."
            )

        if not 0.0 <= float(temperature) <= 2.0:
            raise ValueError(
                "temperature must be between 0.0 and 2.0"
            )

        if float(timeout) <= 0:
            raise ValueError(
                "timeout must be greater than zero"
            )

        self.base_url = base_url
        self.model = model
        self.temperature = float(temperature)
        self.timeout = float(timeout)

    # =========================================================================
    # HTTP helper
    # =========================================================================

    def _post_json(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Send a JSON POST request to Ollama.

        Sensitive prompt content is not logged.
        """

        try:
            body = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")

        except (TypeError, ValueError) as error:
            raise OllamaClientError(
                "Request payload is not valid JSON."
            ) from error

        request = Request(
            f"{self.base_url}{endpoint}",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(
                request,
                timeout=self.timeout,
            ) as response:

                raw = response.read()

        except HTTPError as error:

            detail = error.read(2048).decode(
                "utf-8",
                errors="replace",
            )

            raise OllamaClientError(
                f"Ollama HTTP error {error.code}: {detail[:500]}"
            ) from error

        except (URLError, TimeoutError, OSError) as error:

            raise OllamaClientError(
                "Cannot connect to Ollama. Check that the "
                "Ollama service is running and the URL is correct."
            ) from error

        try:
            data = json.loads(
                raw.decode("utf-8")
            )

        except (UnicodeDecodeError, json.JSONDecodeError) as error:

            raise OllamaClientError(
                "Ollama returned invalid JSON."
            ) from error

        if not isinstance(data, dict):

            raise OllamaClientError(
                "Ollama returned an unexpected response format."
            )

        if data.get("error"):

            raise OllamaClientError(
                f"Ollama error: {str(data['error'])[:500]}"
            )

        return data

    # =========================================================================
    # Standard text generation
    # =========================================================================

    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Generate text using /api/generate.

        Preserves compatibility with the existing Secure RAG.
        """

        if not isinstance(prompt, str) or not prompt.strip():

            raise ValueError(
                "prompt must be a non-empty string"
            )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        data = self._post_json(
            "/api/generate",
            payload,
        )

        generated = data.get("response")

        if not isinstance(generated, str) or not generated.strip():

            raise OllamaClientError(
                "Ollama returned an empty generation."
            )

        return generated.strip()

    # =========================================================================
    # Chat API
    # =========================================================================

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Send a chat request using /api/chat.

        Supports native Ollama tool calling.

        Args:
            messages:
                Conversation history.

            tools:
                Optional list of tool definitions.

        Returns:
            Assistant message dictionary.

        Example response with tool call:

        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "example_tool",
                        "arguments": {}
                    }
                }
            ]
        }
        """

        if not isinstance(messages, list) or not messages:

            raise ValueError(
                "messages must be a non-empty list"
            )

        for message in messages:

            if not isinstance(message, dict):

                raise ValueError(
                    "Every message must be a dictionary."
                )

            role = message.get("role")

            if not isinstance(role, str) or not role.strip():

                raise ValueError(
                    "Every message must contain a valid role."
                )

        if tools is not None:

            if not isinstance(tools, list):

                raise ValueError(
                    "tools must be a list."
                )

            if not all(
                isinstance(tool, dict)
                for tool in tools
            ):

                raise ValueError(
                    "Every tool definition must be a dictionary."
                )

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        if tools:
            payload["tools"] = tools

        data = self._post_json(
            "/api/chat",
            payload,
        )

        assistant_message = data.get("message")

        if not isinstance(assistant_message, dict):

            raise OllamaClientError(
                "Ollama returned an invalid chat message."
            )

        if assistant_message.get("role") != "assistant":

            raise OllamaClientError(
                "Ollama returned an unexpected message role."
            )

        content = assistant_message.get("content", "")

        if not isinstance(content, str):

            raise OllamaClientError(
                "Ollama returned invalid message content."
            )

        tool_calls = assistant_message.get("tool_calls", [])

        if not isinstance(tool_calls, list):

            raise OllamaClientError(
                "Ollama returned invalid tool_calls."
            )

        return assistant_message

    # =========================================================================
    # Health check
    # =========================================================================

    def health_check(self) -> bool:
        """
        Check whether Ollama is available.
        """

        request = Request(
            f"{self.base_url}/api/tags",
            headers={
                "Accept": "application/json",
            },
            method="GET",
        )

        try:

            with urlopen(
                request,
                timeout=min(self.timeout, 10.0),
            ) as response:

                data = json.loads(
                    response.read().decode("utf-8")
                )

            return (
                isinstance(data, dict)
                and isinstance(data.get("models", []), list)
            )

        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ):

            return False
