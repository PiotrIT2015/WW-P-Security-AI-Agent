"""
agent/state.py

State management for WW-P Security AI Agent.

Python 3.10+
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from copy import deepcopy


class AgentStatus(str, Enum):
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class AgentState:
    """
    Represents the state of a single agent execution.

    The state is associated with one authenticated user
    and one conversation.
    """

    conversation_id: str
    user_id: str

    messages: list[dict[str, Any]] = field(default_factory=list)

    status: AgentStatus = AgentStatus.READY

    iteration: int = 0
    tool_calls_count: int = 0

    final_response: str | None = None
    error: str | None = None

    tool_results: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.conversation_id, str):
            raise ValueError("conversation_id must be a string.")

        if not self.conversation_id.strip():
            raise ValueError("conversation_id cannot be empty.")

        if not isinstance(self.user_id, str):
            raise ValueError("user_id must be a string.")

        if not self.user_id.strip():
            raise ValueError("user_id cannot be empty.")

        if not isinstance(self.messages, list):
            raise ValueError("messages must be a list.")

        if not isinstance(self.tool_results, list):
            raise ValueError("tool_results must be a list.")

        if not isinstance(self.status, AgentStatus):
            raise ValueError("Invalid agent status.")

        if self.iteration < 0:
            raise ValueError("iteration cannot be negative.")

        if self.tool_calls_count < 0:
            raise ValueError("tool_calls_count cannot be negative.")

        self.messages = deepcopy(self.messages)
        self.tool_results = deepcopy(self.tool_results)

        for message in self.messages:
            self._validate_message(message)

    @staticmethod
    def _validate_message(message: dict[str, Any]) -> None:
        if not isinstance(message, dict):
            raise ValueError("Message must be a dictionary.")

        role = message.get("role")

        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Invalid message role.")

        if not isinstance(message.get("content"), str):
            raise ValueError("Message content must be a string.")

    def append_message(self, message: dict[str, Any]) -> None:
        """Append a validated message to conversation history."""

        if self.status in {AgentStatus.COMPLETED, AgentStatus.FAILED}:
            raise ValueError("Cannot modify a finished agent state.")

        self._validate_message(message)

        self.messages.append(deepcopy(message))

    def increment_iteration(self) -> None:
        """Increase the agent execution iteration counter."""

        if self.status in {AgentStatus.COMPLETED, AgentStatus.FAILED}:
            raise ValueError("Cannot modify a finished agent state.")

        self.iteration += 1

    def increment_tool_calls(self, count: int = 1) -> None:
        """Increase the number of executed tool calls."""

        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("count must be a non-negative integer.")

        if self.status in {AgentStatus.COMPLETED, AgentStatus.FAILED}:
            raise ValueError("Cannot modify a finished agent state.")

        self.tool_calls_count += count

    def add_tool_results(
        self,
        results: list[dict[str, Any]]
    ) -> None:
        """Store tool execution results."""

        if self.status in {AgentStatus.COMPLETED, AgentStatus.FAILED}:
            raise ValueError("Cannot modify a finished agent state.")

        if not isinstance(results, list):
            raise ValueError("results must be a list.")

        for result in results:
            if not isinstance(result, dict):
                raise ValueError("Each tool result must be a dictionary.")

        self.tool_results.extend(deepcopy(results))

    def start(self) -> None:
        """Mark agent execution as running."""

        if self.status != AgentStatus.READY:
            raise ValueError("Agent can only start from READY state.")

        self.status = AgentStatus.RUNNING

    def complete(self, response: str) -> None:
        """Mark agent execution as completed."""

        if self.status != AgentStatus.RUNNING:
            raise ValueError("Agent must be running before completion.")

        if not isinstance(response, str):
            raise ValueError("Final response must be a string.")

        self.final_response = response
        self.status = AgentStatus.COMPLETED

    def fail(self, error: str) -> None:
        """Mark agent execution as failed."""

        if self.status in {AgentStatus.COMPLETED, AgentStatus.FAILED}:
            raise ValueError("Agent state is already finished.")

        if not isinstance(error, str):
            raise ValueError("Error must be a string.")

        self.error = error
        self.status = AgentStatus.FAILED

    def get_messages(self) -> list[dict[str, Any]]:
        """Return a copy of conversation history."""

        return deepcopy(self.messages)

    def is_finished(self) -> bool:
        """Check whether agent execution has finished."""

        return self.status in {
            AgentStatus.COMPLETED,
            AgentStatus.FAILED
        }