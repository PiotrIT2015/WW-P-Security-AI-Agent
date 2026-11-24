"""
tests/test_tool_call_parser.py

Unit tests for Ollama tool call parser.
"""

import pytest

from llm.tool_calling import (
    ToolCall,
    ToolCallParseError,
    parse_tool_calls,
)


# ============================================================
# Helpers
# ============================================================

_MISSING = object()


def create_assistant_message(tool_calls=_MISSING):
    """
    Create a sample Ollama assistant message.

    Distinguishes between:
    - missing tool_calls field
    - tool_calls explicitly set to None
    """

    message = {
        "role": "assistant",
        "content": "",
    }

    if tool_calls is not _MISSING:
        message["tool_calls"] = tool_calls

    return message


def create_tool_call(name, arguments):
    """
    Create a sample Ollama tool call.
    """

    return {
        "function": {
            "name": name,
            "arguments": arguments,
        }
    }

# ============================================================
# 1. No tool calls
# ============================================================

def test_parse_message_without_tool_calls():

    message = create_assistant_message()

    result = parse_tool_calls(message)

    assert result == []
    assert isinstance(result, list)


# ============================================================
# 2. Empty tool calls
# ============================================================

def test_parse_empty_tool_calls():

    message = create_assistant_message(
        tool_calls=[]
    )

    result = parse_tool_calls(message)

    assert result == []


# ============================================================
# 3. One valid tool call
# ============================================================

def test_parse_single_valid_tool_call():

    message = create_assistant_message(
        tool_calls=[
            create_tool_call(
                "system_status",
                {},
            )
        ]
    )

    result = parse_tool_calls(message)

    assert len(result) == 1

    assert isinstance(result[0], ToolCall)

    assert result[0].name == "system_status"

    assert result[0].arguments == {}


# ============================================================
# 4. Tool call with arguments
# ============================================================

def test_parse_tool_call_with_arguments():

    arguments = {
        "query": "network status",
        "verbose": True,
    }

    message = create_assistant_message(
        tool_calls=[
            create_tool_call(
                "network_monitor",
                arguments,
            )
        ]
    )

    result = parse_tool_calls(message)

    assert len(result) == 1

    assert result[0].name == "network_monitor"

    assert result[0].arguments == arguments


# ============================================================
# 5. Multiple valid tool calls
# ============================================================

def test_parse_multiple_tool_calls():

    message = create_assistant_message(
        tool_calls=[
            create_tool_call(
                "system_status",
                {},
            ),
            create_tool_call(
                "network_monitor",
                {"interface": "eth0"},
            ),
            create_tool_call(
                "security_scan",
                {"target": "localhost"},
            ),
        ]
    )

    result = parse_tool_calls(message)

    assert len(result) == 3

    assert result[0].name == "system_status"

    assert result[1].name == "network_monitor"

    assert result[2].name == "security_scan"

    assert result[1].arguments == {
        "interface": "eth0"
    }


# ============================================================
# 6. Invalid top-level message
# ============================================================

@pytest.mark.parametrize(
    "message",
    [
        None,
        [],
        "invalid message",
        123,
    ],
)
def test_reject_invalid_top_level_message(message):

    with pytest.raises(ToolCallParseError):
        parse_tool_calls(message)


# ============================================================
# 7. Invalid message role
# ============================================================

@pytest.mark.parametrize(
    "role",
    [
        "user",
        "system",
        "tool",
        None,
    ],
)
def test_reject_non_assistant_message(role):

    message = {
        "role": role,
        "tool_calls": [],
    }

    with pytest.raises(ToolCallParseError):
        parse_tool_calls(message)


# ============================================================
# 8. tool_calls must be a list
# ============================================================

@pytest.mark.parametrize(
    "tool_calls",
    [
        {},
        "invalid",
        123,
        None,
    ],
)
def test_reject_invalid_tool_calls_container(tool_calls):

    message = create_assistant_message(
        tool_calls=tool_calls
    )

    with pytest.raises(ToolCallParseError):
        parse_tool_calls(message)


# ============================================================
# 9. Invalid individual tool call
# ============================================================

@pytest.mark.parametrize(
    "raw_call",
    [
        None,
        [],
        "invalid",
        123,
    ],
)
def test_reject_invalid_tool_call_object(raw_call):

    message = create_assistant_message(
        tool_calls=[raw_call]
    )

    with pytest.raises(ToolCallParseError):
        parse_tool_calls(message)


# ============================================================
# 10. Missing function object
# ============================================================

def test_reject_missing_function():

    message = create_assistant_message(
        tool_calls=[
            {}
        ]
    )

    with pytest.raises(
        ToolCallParseError,
        match="function object",
    ):
        parse_tool_calls(message)


# ============================================================
# 11. Invalid function object
# ============================================================

def test_reject_invalid_function():

    message = create_assistant_message(
        tool_calls=[
            {
                "function": "system_status"
            }
        ]
    )

    with pytest.raises(
        ToolCallParseError,
        match="function object",
    ):
        parse_tool_calls(message)


# ============================================================
# 12. Missing tool name
# ============================================================

def test_reject_missing_tool_name():

    message = create_assistant_message(
        tool_calls=[
            {
                "function": {
                    "arguments": {}
                }
            }
        ]
    )

    with pytest.raises(
        ToolCallParseError,
        match="missing function name",
    ):
        parse_tool_calls(message)


# ============================================================
# 13. Invalid tool name
# ============================================================

@pytest.mark.parametrize(
    "name",
    [
        "",
        " ",
        "system status",
        "system-status",
        "system.status",
        "system/status",
        "system_status ",
        " system_status",
        None,
        123,
    ],
)
def test_reject_invalid_tool_name(name):

    message = create_assistant_message(
        tool_calls=[
            create_tool_call(
                name,
                {},
            )
        ]
    )

    with pytest.raises(ToolCallParseError):
        parse_tool_calls(message)


# ============================================================
# 14. Missing arguments
# ============================================================

def test_reject_missing_arguments():

    message = create_assistant_message(
        tool_calls=[
            {
                "function": {
                    "name": "system_status"
                }
            }
        ]
    )

    with pytest.raises(
        ToolCallParseError,
        match="missing arguments",
    ):
        parse_tool_calls(message)


# ============================================================
# 15. Arguments must be a dictionary
# ============================================================

@pytest.mark.parametrize(
    "arguments",
    [
        None,
        [],
        "invalid",
        123,
        True,
    ],
)
def test_reject_non_dictionary_arguments(arguments):

    message = create_assistant_message(
        tool_calls=[
            create_tool_call(
                "system_status",
                arguments,
            )
        ]
    )

    with pytest.raises(
        ToolCallParseError,
        match="arguments must be an object",
    ):
        parse_tool_calls(message)


# ============================================================
# 16. Arguments are copied
# ============================================================

def test_arguments_are_copied():

    original_arguments = {
        "target": "localhost"
    }

    message = create_assistant_message(
        tool_calls=[
            create_tool_call(
                "security_scan",
                original_arguments,
            )
        ]
    )

    result = parse_tool_calls(message)

    result[0].arguments["target"] = "modified"

    assert original_arguments["target"] == "localhost"


# ============================================================
# 17. Multiple calls - invalid call rejects whole message
# ============================================================

def test_reject_all_when_one_tool_call_is_invalid():

    message = create_assistant_message(
        tool_calls=[
            create_tool_call(
                "system_status",
                {},
            ),
            create_tool_call(
                "invalid tool name",
                {},
            ),
        ]
    )

    with pytest.raises(ToolCallParseError):
        parse_tool_calls(message)