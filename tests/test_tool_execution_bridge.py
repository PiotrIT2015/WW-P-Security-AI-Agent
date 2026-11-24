"""
tests/test_tool_execution_bridge.py

Unit tests for ToolExecutionBridge.
"""

from unittest.mock import Mock, call

import pytest

from llm.tool_calling import (
    ToolCall,
    ToolCallParseError,
)

from llm.tool_bridge import (
    ToolExecutionBridge,
    ToolExecutionBridgeError,
)

from tools.base import ToolResult


# ============================================================
# Helpers
# ============================================================

def create_tool_call(
    name="system_status",
    arguments=None,
):

    return ToolCall(
        name=name,
        arguments=arguments or {},
    )


def create_result(
    name="system_status",
    success=True,
    output=None,
    error=None,
):

    return ToolResult(
        tool_name=name,
        success=success,
        output=output,
        error=error,
    )


def create_executor():

    executor = Mock()

    executor.execute.side_effect = (
        lambda tool_name, arguments, user: create_result(
            name=tool_name,
            output={"status": "ok"},
        )
    )

    return executor


def create_bridge(executor=None):

    executor = executor or create_executor()

    return ToolExecutionBridge(executor), executor


# ============================================================
# 1. Initialization
# ============================================================

def test_bridge_initialization():

    executor = create_executor()

    bridge = ToolExecutionBridge(executor)

    assert bridge.executor is executor


def test_bridge_rejects_none_executor():

    with pytest.raises(ValueError):
        ToolExecutionBridge(None)


def test_bridge_rejects_invalid_executor():

    with pytest.raises(TypeError):
        ToolExecutionBridge(object())


# ============================================================
# 2. Single tool execution
# ============================================================

def test_execute_single_tool_call():

    bridge, executor = create_bridge()

    user = object()

    tool_call = create_tool_call(
        "system_status",
        {},
    )

    result = bridge.execute_tool_call(
        tool_call,
        user,
    )

    assert isinstance(result, ToolResult)

    assert result.success is True

    assert result.tool_name == "system_status"

    executor.execute.assert_called_once()


# ============================================================
# 3. Arguments forwarding
# ============================================================

def test_execute_forwards_arguments():

    bridge, executor = create_bridge()

    user = object()

    arguments = {
        "interface": "eth0",
        "verbose": True,
    }

    tool_call = create_tool_call(
        "network_monitor",
        arguments,
    )

    bridge.execute_tool_call(
        tool_call,
        user,
    )

    executor.execute.assert_called_once_with(
        tool_name="network_monitor",
        arguments=arguments,
        user=user,
    )


# ============================================================
# 4. Arguments are copied
# ============================================================

def test_execute_passes_argument_copy():

    bridge, executor = create_bridge()

    original_arguments = {
        "target": "localhost",
    }

    tool_call = create_tool_call(
        "security_scan",
        original_arguments,
    )

    bridge.execute_tool_call(
        tool_call,
        object(),
    )

    passed_arguments = executor.execute.call_args.kwargs[
        "arguments"
    ]

    assert passed_arguments == original_arguments

    assert passed_arguments is not original_arguments


# ============================================================
# 5. Multiple tool calls
# ============================================================

def test_execute_multiple_tool_calls():

    bridge, executor = create_bridge()

    user = object()

    tool_calls = [
        create_tool_call("system_status"),
        create_tool_call("network_monitor"),
        create_tool_call("security_scan"),
    ]

    results = bridge.execute_tool_calls(
        tool_calls,
        user,
    )

    assert len(results) == 3

    assert all(
        isinstance(result, ToolResult)
        for result in results
    )

    assert executor.execute.call_count == 3


# ============================================================
# 6. Execution order
# ============================================================

def test_execution_order_is_preserved():

    executor = create_executor()

    bridge = ToolExecutionBridge(executor)

    tool_calls = [
        create_tool_call("system_status"),
        create_tool_call("network_monitor"),
        create_tool_call("security_scan"),
    ]

    bridge.execute_tool_calls(
        tool_calls,
        object(),
    )

    executed_names = [
        item.kwargs["tool_name"]
        for item in executor.execute.call_args_list
    ]

    assert executed_names == [
        "system_status",
        "network_monitor",
        "security_scan",
    ]


# ============================================================
# 7. Empty tool calls
# ============================================================

def test_execute_empty_tool_calls():

    bridge, executor = create_bridge()

    result = bridge.execute_tool_calls(
        [],
        object(),
    )

    assert result == []

    executor.execute.assert_not_called()


# ============================================================
# 8. Invalid tool calls container
# ============================================================

@pytest.mark.parametrize(
    "tool_calls",
    [
        None,
        {},
        "invalid",
        123,
    ],
)
def test_reject_invalid_tool_calls_container(tool_calls):

    bridge, executor = create_bridge()

    with pytest.raises(ToolExecutionBridgeError):
        bridge.execute_tool_calls(
            tool_calls,
            object(),
        )

    executor.execute.assert_not_called()


# ============================================================
# 9. Invalid individual tool call
# ============================================================

@pytest.mark.parametrize(
    "tool_call",
    [
        None,
        {},
        "invalid",
        123,
    ],
)
def test_reject_invalid_tool_call(tool_call):

    bridge, executor = create_bridge()

    with pytest.raises(ToolExecutionBridgeError):
        bridge.execute_tool_calls(
            [tool_call],
            object(),
        )

    executor.execute.assert_not_called()


# ============================================================
# 10. Failed execution result
# ============================================================

def test_failed_tool_execution_is_preserved():

    executor = Mock()

    executor.execute.return_value = ToolResult(
        tool_name="security_scan",
        success=False,
        error="Permission denied.",
    )

    bridge = ToolExecutionBridge(executor)

    result = bridge.execute_tool_call(
        create_tool_call("security_scan"),
        object(),
    )

    assert result.success is False

    assert result.error == "Permission denied."


# ============================================================
# 11. Continue after failed tool call
# ============================================================

def test_continue_after_failed_tool_call():

    executor = Mock()

    executor.execute.side_effect = [
        ToolResult(
            tool_name="system_status",
            success=False,
            error="Permission denied.",
        ),
        ToolResult(
            tool_name="network_monitor",
            success=True,
            output={"status": "ok"},
        ),
    ]

    bridge = ToolExecutionBridge(executor)

    results = bridge.execute_tool_calls(
        [
            create_tool_call("system_status"),
            create_tool_call("network_monitor"),
        ],
        object(),
    )

    assert len(results) == 2

    assert results[0].success is False

    assert results[1].success is True

    assert executor.execute.call_count == 2


# ============================================================
# 12. Unexpected executor exception
# ============================================================

def test_handle_executor_exception():

    executor = Mock()

    executor.execute.side_effect = RuntimeError(
        "Internal error"
    )

    bridge = ToolExecutionBridge(executor)

    result = bridge.execute_tool_call(
        create_tool_call("system_status"),
        object(),
    )

    assert result.success is False

    assert result.error == "Tool execution failed."

    assert result.metadata["bridge_error"] == "RuntimeError"


# ============================================================
# 13. Invalid executor result
# ============================================================

def test_reject_invalid_executor_result():

    executor = Mock()

    executor.execute.return_value = {
        "status": "ok"
    }

    bridge = ToolExecutionBridge(executor)

    result = bridge.execute_tool_call(
        create_tool_call("system_status"),
        object(),
    )

    assert result.success is False

    assert result.error == "Invalid tool execution result."


# ============================================================
# 14. Continue after unexpected exception
# ============================================================

def test_continue_after_unexpected_executor_exception():

    executor = Mock()

    executor.execute.side_effect = [
        RuntimeError("Internal error"),
        ToolResult(
            tool_name="network_monitor",
            success=True,
            output={"status": "ok"},
        ),
    ]

    bridge = ToolExecutionBridge(executor)

    results = bridge.execute_tool_calls(
        [
            create_tool_call("system_status"),
            create_tool_call("network_monitor"),
        ],
        object(),
    )

    assert len(results) == 2

    assert results[0].success is False

    assert results[1].success is True


# ============================================================
# 15. Full assistant message processing
# ============================================================

def test_execute_assistant_message():

    bridge, executor = create_bridge()

    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "system_status",
                    "arguments": {},
                }
            }
        ],
    }

    results = bridge.execute_message(
        message,
        object(),
    )

    assert len(results) == 1

    assert results[0].success is True

    executor.execute.assert_called_once()


# ============================================================
# 16. Multiple calls from assistant message
# ============================================================

def test_execute_multiple_calls_from_message():

    bridge, executor = create_bridge()

    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "system_status",
                    "arguments": {},
                }
            },
            {
                "function": {
                    "name": "network_monitor",
                    "arguments": {
                        "interface": "eth0"
                    },
                }
            },
        ],
    }

    results = bridge.execute_message(
        message,
        object(),
    )

    assert len(results) == 2

    assert executor.execute.call_count == 2


# ============================================================
# 17. Assistant message without tool calls
# ============================================================

def test_execute_message_without_tool_calls():

    bridge, executor = create_bridge()

    message = {
        "role": "assistant",
        "content": "Hello!",
    }

    results = bridge.execute_message(
        message,
        object(),
    )

    assert results == []

    executor.execute.assert_not_called()


# ============================================================
# 18. Invalid assistant message
# ============================================================

def test_reject_invalid_assistant_message():

    bridge, executor = create_bridge()

    message = {
        "role": "user",
        "content": "Hello!",
        "tool_calls": [],
    }

    with pytest.raises(ToolCallParseError):
        bridge.execute_message(
            message,
            object(),
        )

    executor.execute.assert_not_called()


# ============================================================
# 19. Invalid ToolCall arguments
# ============================================================

def test_reject_invalid_tool_call_arguments():

    bridge, executor = create_bridge()

    invalid_call = ToolCall(
        name="system_status",
        arguments=None,
    )

    with pytest.raises(ToolExecutionBridgeError):
        bridge.execute_tool_calls(
            [invalid_call],
            object(),
        )

    executor.execute.assert_not_called()


# ============================================================
# 20. ToolExecutor is the only execution path
# ============================================================

def test_bridge_delegates_execution_to_executor():

    bridge, executor = create_bridge()

    tool_call = create_tool_call(
        "system_status",
        {},
    )

    bridge.execute_tool_call(
        tool_call,
        object(),
    )

    assert executor.execute.call_count == 1