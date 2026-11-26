"""
agent/context.py

Context management for WW-P Security AI Agent.

Responsibilities:
- Limit conversation history.
- Limit context size.
- Preserve system instructions.
- Preserve the newest conversation context.
- Keep assistant tool calls and tool results together.
- Avoid modifying the original conversation history.

Compatible with Python 3.10+.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from agent.exceptions import AgentConfigurationError


@dataclass(frozen=True)
class ContextStats:
    original_messages: int
    selected_messages: int
    original_characters: int
    selected_characters: int
    removed_messages: int
    truncated: bool


class ContextManager:
    """
    Deterministic conversation context manager.

    The manager selects the newest complete conversation messages
    that fit within the configured limits.

    System messages are preserved.
    Assistant tool calls and their corresponding tool responses
    are treated as an atomic group.
    """

    def __init__(
        self,
        max_messages: int = 30,
        max_context_chars: int = 24000,
    ) -> None:

        if (
            not isinstance(max_messages, int)
            or isinstance(max_messages, bool)
            or max_messages < 2
        ):
            raise AgentConfigurationError(
                "max_messages must be an integer >= 2."
            )

        if (
            not isinstance(max_context_chars, int)
            or isinstance(max_context_chars, bool)
            or max_context_chars <= 0
        ):
            raise AgentConfigurationError(
                "max_context_chars must be a positive integer."
            )

        self.max_messages = max_messages
        self.max_context_chars = max_context_chars

        self.last_stats: ContextStats | None = None

    @staticmethod
    def _validate_messages(
        messages: list[dict[str, Any]],
    ) -> None:

        if not isinstance(messages, list):
            raise ValueError("Messages must be a list.")

        for message in messages:

            if not isinstance(message, dict):
                raise ValueError("Every message must be a dictionary.")

            if message.get("role") not in {
                "system",
                "user",
                "assistant",
                "tool",
            }:
                raise ValueError("Invalid message role.")

            if not isinstance(message.get("content"), str):
                raise ValueError(
                    "Message content must be a string."
                )

    @staticmethod
    def _message_size(message: dict[str, Any]) -> int:
        """
        Approximate message size using its string representation.

        This is a character-based limit, not an exact token count.
        """

        return len(str(message))

    def _build_groups(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
        """
        Separate system messages and conversation groups.

        An assistant message containing tool_calls and the following
        tool messages are kept together.
        """

        system_messages = []
        conversation_messages = []

        for message in messages:

            if message["role"] == "system":
                system_messages.append(message)
            else:
                conversation_messages.append(message)

        groups = []
        index = 0

        while index < len(conversation_messages):

            message = conversation_messages[index]

            group = [message]

            tool_calls = message.get("tool_calls")

            if (
                message["role"] == "assistant"
                and isinstance(tool_calls, list)
                and len(tool_calls) > 0
            ):

                index += 1

                while index < len(conversation_messages):

                    next_message = conversation_messages[index]

                    if next_message["role"] != "tool":
                        break

                    group.append(next_message)
                    index += 1

                groups.append(group)
                continue

            groups.append(group)
            index += 1

        return system_messages, groups

    def prepare(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Return a context window that fits configured limits.

        The original messages are never modified.
        """

        self._validate_messages(messages)

        copied_messages = deepcopy(messages)

        if not copied_messages:
            self.last_stats = ContextStats(
                original_messages=0,
                selected_messages=0,
                original_characters=0,
                selected_characters=0,
                removed_messages=0,
                truncated=False,
            )

            return []

        original_count = len(copied_messages)

        original_size = sum(
            self._message_size(message)
            for message in copied_messages
        )

        system_messages, groups = self._build_groups(
            copied_messages
        )

        selected_system = list(system_messages)

        selected_count = len(selected_system)

        selected_size = sum(
            self._message_size(message)
            for message in selected_system
        )

        if selected_count > self.max_messages:
            raise ValueError(
                "System messages exceed the message limit."
            )

        if selected_size > self.max_context_chars:
            raise ValueError(
                "System messages exceed the context size limit."
            )

        selected_groups = []

        # Preserve the newest complete conversation groups.
        for group in reversed(groups):

            group_size = sum(
                self._message_size(message)
                for message in group
            )

            group_count = len(group)

            if (
                selected_count + group_count > self.max_messages
                or selected_size + group_size > self.max_context_chars
            ):

                # Do not split tool-call groups.
                if not selected_groups and not selected_system:
                    raise ValueError(
                        "The newest conversation group exceeds "
                        "the context limits."
                    )

                break

            selected_groups.append(group)

            selected_count += group_count
            selected_size += group_size

        selected_groups.reverse()

        selected_messages = selected_system[:]

        for group in selected_groups:
            selected_messages.extend(group)

        selected_messages = deepcopy(selected_messages)

        final_count = len(selected_messages)

        final_size = sum(
            self._message_size(message)
            for message in selected_messages
        )

        self.last_stats = ContextStats(
            original_messages=original_count,
            selected_messages=final_count,
            original_characters=original_size,
            selected_characters=final_size,
            removed_messages=original_count - final_count,
            truncated=original_count != final_count,
        )

        return selected_messages

    def get_stats(self) -> ContextStats | None:
        """
        Return statistics from the last context preparation.
        """

        return self.last_stats