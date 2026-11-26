import pytest

from agent.context import ContextManager
from agent.exceptions import AgentConfigurationError


def test_context_manager_initialization():

    manager = ContextManager(
        max_messages=10,
        max_context_chars=5000,
    )

    assert manager.max_messages == 10
    assert manager.max_context_chars == 5000


def test_reject_invalid_message_limit():

    with pytest.raises(AgentConfigurationError):

        ContextManager(max_messages=1)


def test_reject_invalid_character_limit():

    with pytest.raises(AgentConfigurationError):

        ContextManager(max_context_chars=0)


def test_empty_context():

    manager = ContextManager()

    result = manager.prepare([])

    assert result == []


def test_preserves_system_message():

    manager = ContextManager(max_messages=3)

    messages = [
        {"role": "system", "content": "System instructions"},
        {"role": "user", "content": "Question 1"},
        {"role": "assistant", "content": "Answer 1"},
        {"role": "user", "content": "Question 2"},
        {"role": "assistant", "content": "Answer 2"},
    ]

    result = manager.prepare(messages)

    assert result[0]["role"] == "system"
    assert result[0]["content"] == "System instructions"


def test_preserves_newest_messages():

    manager = ContextManager(max_messages=3)

    messages = [
        {"role": "user", "content": "Old question"},
        {"role": "assistant", "content": "Old answer"},
        {"role": "user", "content": "Newest question"},
        {"role": "assistant", "content": "Newest answer"},
    ]

    result = manager.prepare(messages)

    assert len(result) <= 3

    assert result[-1]["content"] == "Newest answer"


def test_limits_message_count():

    manager = ContextManager(max_messages=3)

    messages = [
        {"role": "user", "content": "Question 1"},
        {"role": "assistant", "content": "Answer 1"},
        {"role": "user", "content": "Question 2"},
        {"role": "assistant", "content": "Answer 2"},
        {"role": "user", "content": "Question 3"},
    ]

    result = manager.prepare(messages)

    assert len(result) <= 3


def test_limits_context_size():

    manager = ContextManager(
        max_messages=20,
        max_context_chars=100,
    )

    messages = [
        {"role": "user", "content": "A" * 20},
        {"role": "assistant", "content": "B" * 20},
        {"role": "user", "content": "C" * 20},
    ]

    result = manager.prepare(messages)

    total_size = sum(
        manager._message_size(message)
        for message in result
    )

    assert total_size <= 100


def test_preserves_tool_call_group():

    manager = ContextManager(max_messages=3)

    messages = [
        {"role": "user", "content": "Old question"},
        {"role": "assistant", "content": "Old answer"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "system_status"}}
            ],
        },
        {
            "role": "tool",
            "tool_name": "system_status",
            "content": '{"success": true}',
        },
        {
            "role": "assistant",
            "content": "System is healthy.",
        },
    ]

    result = manager.prepare(messages)

    assistant_tool_call = next(
        message
        for message in result
        if message.get("tool_calls")
    )

    tool_message = next(
        message
        for message in result
        if message["role"] == "tool"
    )

    assert assistant_tool_call is not None
    assert tool_message is not None


def test_does_not_modify_original_messages():

    manager = ContextManager(max_messages=3)

    messages = [
        {"role": "user", "content": "Question 1"},
        {"role": "assistant", "content": "Answer 1"},
        {"role": "user", "content": "Question 2"},
        {"role": "assistant", "content": "Answer 2"},
    ]

    original = [
        message.copy()
        for message in messages
    ]

    manager.prepare(messages)

    assert messages == original


def test_returns_context_statistics():

    manager = ContextManager(max_messages=2)

    messages = [
        {"role": "user", "content": "Old question"},
        {"role": "assistant", "content": "Old answer"},
        {"role": "user", "content": "New question"},
    ]

    manager.prepare(messages)

    stats = manager.get_stats()

    assert stats is not None
    assert stats.original_messages == 3
    assert stats.selected_messages <= 2
    assert stats.removed_messages >= 1
    assert stats.truncated is True


def test_rejects_invalid_message():

    manager = ContextManager()

    messages = [
        {"role": "invalid", "content": "test"}
    ]

    with pytest.raises(ValueError):

        manager.prepare(messages)


def test_rejects_non_string_content():

    manager = ContextManager()

    messages = [
        {"role": "user", "content": 123}
    ]

    with pytest.raises(ValueError):

        manager.prepare(messages)


def test_rejects_oversized_newest_message():

    manager = ContextManager(
        max_messages=10,
        max_context_chars=10,
    )

    messages = [
        {"role": "user", "content": "This message is too long"}
    ]

    with pytest.raises(ValueError):

        manager.prepare(messages)