"""
tools/registry.py

Central tool registry for WW-P Security AI Agent.

Python 3.10+
"""

from __future__ import annotations

import logging

from tools.base import BaseTool


logger = logging.getLogger("secure_rag.tools.registry")


class ToolRegistryError(Exception):
    """Base registry exception."""


class ToolNotFoundError(ToolRegistryError):
    """Raised when a tool does not exist."""


class ToolAlreadyRegisteredError(ToolRegistryError):
    """Raised when a tool is registered twice."""


class ToolRegistry:
    """
    Central registry of available tools.

    Responsibilities:
    - Register tools.
    - Prevent duplicate registration.
    - Retrieve tools by name.
    - Expose tool definitions.
    - Remove registered tools.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if not isinstance(tool, BaseTool):
            raise TypeError(
                "Registered object must inherit from BaseTool."
            )

        tool.validate_definition()

        if tool.name in self._tools:
            raise ToolAlreadyRegisteredError(
                f"Tool already registered: {tool.name}"
            )

        self._tools[tool.name] = tool

        logger.info("Tool registered: %s", tool.name)

    def get(self, name: str) -> BaseTool:
        tool = self._tools.get(name)

        if tool is None:
            raise ToolNotFoundError(
                f"Tool not found: {name}"
            )

        return tool

    def list_tools(self) -> list[dict]:
        """
        Return public definitions of all registered tools.

        These definitions can later be passed to an LLM
        for tool selection.
        """
        return [
            tool.get_definition()
            for tool in self._tools.values()
        ]

    def list_names(self) -> list[str]:
        """Return sorted names of registered tools."""
        return sorted(self._tools.keys())

    def unregister(self, name: str) -> bool:
        """
        Remove a registered tool.

        Raises:
            ToolNotFoundError: if the requested tool does not exist.
        """
        if name not in self._tools:
            raise ToolNotFoundError(
                f"Cannot unregister unknown tool: {name}"
            )

        del self._tools[name]

        logger.info("Tool unregistered: %s", name)

        return True

    @property
    def count(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)