import json
import pytest

from tools.base import ToolResult

from llm.tool_result_handler import (
    ToolResultHandler,
    ToolResultHandlerError,
)


@pytest.fixture
def handler():
    return ToolResultHandler()


def create_result(
    tool_name="system_status",
    success=True,
    output="System operational",
    error=None,
):
    return ToolResult(
        tool_name=tool_name,
        success=success,
        output=output,
        error=error,
    )


def test_handler_initialization(handler):
    assert isinstance(handler, ToolResultHandler)


def test_convert_successful_result(handler):

    result = create_result()

    message = handler.to_tool_message(result)

    assert message["role"] == "tool"
    assert message["tool_name"] == "system_status"

    payload = json.loads(message["content"])

    assert payload["success"] is True
    assert payload["output"] == "System operational"
    assert payload["error"] is None


def test_convert_failed_result(handler):

    result = create_result(
        success=False,
        output=None,
        error="Permission denied.",
    )

    message = handler.to_tool_message(result)

    payload = json.loads(message["content"])

    assert payload["success"] is False
    assert payload["error"] == "Permission denied."


def test_convert_result_with_dictionary_output(handler):

    result = create_result(
        output={
            "cpu": 35,
            "memory": 70,
            "status": "healthy",
        }
    )

    message = handler.to_tool_message(result)

    payload = json.loads(message["content"])

    assert payload["output"]["cpu"] == 35
    assert payload["output"]["memory"] == 70


def test_convert_result_with_list_output(handler):

    result = create_result(
        output=["Linux", "Windows", "Docker"]
    )

    message = handler.to_tool_message(result)

    payload = json.loads(message["content"])

    assert payload["output"] == [
        "Linux",
        "Windows",
        "Docker",
    ]


def test_convert_result_with_none_output(handler):

    result = create_result(output=None)

    message = handler.to_tool_message(result)

    payload = json.loads(message["content"])

    assert payload["output"] is None


def test_convert_result_with_unicode(handler):

    result = create_result(
        output="System działa poprawnie."
    )

    message = handler.to_tool_message(result)

    assert "działa" in message["content"]


def test_convert_multiple_results(handler):

    results = [
        create_result(
            tool_name="system_status",
            output="OK",
        ),
        create_result(
            tool_name="network_status",
            output="Connected",
        ),
    ]

    messages = handler.to_tool_messages(results)

    assert len(messages) == 2

    assert messages[0]["tool_name"] == "system_status"
    assert messages[1]["tool_name"] == "network_status"


def test_preserve_execution_order(handler):

    results = [
        create_result(tool_name="tool_first"),
        create_result(tool_name="tool_second"),
        create_result(tool_name="tool_third"),
    ]

    messages = handler.to_tool_messages(results)

    names = [
        message["tool_name"]
        for message in messages
    ]

    assert names == [
        "tool_first",
        "tool_second",
        "tool_third",
    ]


def test_convert_empty_results(handler):

    messages = handler.to_tool_messages([])

    assert messages == []


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        {},
        "invalid",
        123,
    ],
)
def test_reject_invalid_results_container(handler, invalid):

    with pytest.raises(ToolResultHandlerError):
        handler.to_tool_messages(invalid)


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        {},
        "invalid",
        123,
    ],
)
def test_reject_invalid_tool_result(handler, invalid):

    with pytest.raises(ToolResultHandlerError):
        handler.to_tool_message(invalid)


def test_reject_invalid_result_inside_list(handler):

    results = [
        create_result(),
        "invalid",
    ]

    with pytest.raises(ToolResultHandlerError):
        handler.to_tool_messages(results)


def test_reject_empty_tool_name(handler):

    result = create_result(tool_name="")

    with pytest.raises(ToolResultHandlerError):
        handler.to_tool_message(result)


def test_reject_whitespace_tool_name(handler):

    result = create_result(tool_name="   ")

    with pytest.raises(ToolResultHandlerError):
        handler.to_tool_message(result)


def test_reject_invalid_success_type(handler):

    result = ToolResult(
        tool_name="system_status",
        success="yes",
        output="OK",
    )

    with pytest.raises(ToolResultHandlerError):
        handler.to_tool_message(result)


def test_reject_invalid_error_type(handler):

    result = ToolResult(
        tool_name="system_status",
        success=False,
        error=123,
    )

    with pytest.raises(ToolResultHandlerError):
        handler.to_tool_message(result)


def test_reject_non_serializable_output(handler):

    result = create_result(
        output=object()
    )

    with pytest.raises(ToolResultHandlerError):
        handler.to_tool_message(result)


def test_reject_nan_output(handler):

    result = create_result(
        output=float("nan")
    )

    with pytest.raises(ToolResultHandlerError):
        handler.to_tool_message(result)


def test_handle_results_alias(handler):

    results = [
        create_result(
            tool_name="system_status",
        )
    ]

    messages = handler.handle_results(results)

    assert len(messages) == 1
    assert messages[0]["role"] == "tool"


def test_result_metadata_is_not_exposed(handler):

    result = ToolResult(
        tool_name="system_status",
        success=True,
        output="OK",
        metadata={
            "internal_debug": "secret",
            "execution_time": 0.01,
        },
    )

    message = handler.to_tool_message(result)

    assert "internal_debug" not in message["content"]
    assert "execution_time" not in message["content"]


def test_original_result_is_not_modified(handler):

    result = create_result(
        output={"status": "healthy"}
    )

    original_output = result.output.copy()

    handler.to_tool_message(result)

    assert result.output == original_output


def test_tool_message_contains_only_ollama_fields(handler):

    result = create_result()

    message = handler.to_tool_message(result)

    assert set(message.keys()) == {
        "role",
        "tool_name",
        "content",
    }