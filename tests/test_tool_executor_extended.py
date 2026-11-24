"""
Extended tests for ToolExecutor.

WW-P Security AI Agent
"""

import pytest

from authentication.auth import UserContext

from tools.base import BaseTool
from tools.registry import ToolRegistry
from tools.executor import ToolExecutor


# =========================================================================
# Test tools
# =========================================================================

class TestTool(BaseTool):

    name = "test_tool"
    description = "Test tool."
    required_roles = {"analyst"}

    input_schema = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string"
            }
        },
        "required": ["message"],
        "additionalProperties": False,
    }

    def execute(self, arguments, user):

        return {
            "message": arguments["message"],
            "user": user.username,
        }


class FailingTool(TestTool):

    name = "failing_tool"
    description = "Tool that raises an exception."

    def execute(self, arguments, user):
        raise RuntimeError("Internal test exception.")


class InvalidOutputTool(TestTool):

    name = "invalid_output_tool"
    description = "Tool returning invalid output."

    def execute(self, arguments, user):
        return object()


class NonSerializableTool(TestTool):

    name = "non_serializable_tool"
    description = "Tool returning non-serializable output."

    def execute(self, arguments, user):
        return {
            "result": object()
        }


# =========================================================================
# Fixtures
# =========================================================================

def create_user(
    roles=frozenset({"analyst"}),
):

    return UserContext(
        user_id="test-user",
        username="test",
        roles=roles,
        groups=frozenset(),
        authenticated_at=1234567890.0,
        session_id="test-session",
    )


def create_executor(tool=None):

    registry = ToolRegistry()

    if tool is None:
        tool = TestTool()

    registry.register(tool)

    return ToolExecutor(registry)


# =========================================================================
# Authentication tests
# =========================================================================

def test_executor_rejects_invalid_user():

    executor = create_executor()

    result = executor.execute(
        "test_tool",
        {"message": "hello"},
        None,
    )

    assert result.success is False
    assert result.error == "Authentication required."


# =========================================================================
# Authorization tests
# =========================================================================

def test_executor_allows_authorized_user():

    executor = create_executor()

    user = create_user(
        frozenset({"analyst"})
    )

    result = executor.execute(
        "test_tool",
        {"message": "hello"},
        user,
    )

    assert result.success is True
    assert result.output["message"] == "hello"


def test_executor_rejects_unauthorized_user():

    executor = create_executor()

    user = create_user(
        frozenset({"viewer"})
    )

    result = executor.execute(
        "test_tool",
        {"message": "hello"},
        user,
    )

    assert result.success is False
    assert result.error == "Permission denied."


def test_executor_allows_admin_override():

    executor = create_executor()

    user = create_user(
        frozenset({"admin"})
    )

    result = executor.execute(
        "test_tool",
        {"message": "hello"},
        user,
    )

    assert result.success is True


# =========================================================================
# Argument validation tests
# =========================================================================

def test_executor_rejects_unknown_arguments():

    executor = create_executor()

    user = create_user()

    result = executor.execute(
        "test_tool",
        {
            "message": "hello",
            "unexpected": "value",
        },
        user,
    )

    assert result.success is False
    assert result.error == "Invalid tool arguments."


def test_executor_rejects_missing_arguments():

    executor = create_executor()

    user = create_user()

    result = executor.execute(
        "test_tool",
        {},
        user,
    )

    assert result.success is False


def test_executor_rejects_non_dictionary_arguments():

    executor = create_executor()

    user = create_user()

    result = executor.execute(
        "test_tool",
        ["invalid"],
        user,
    )

    assert result.success is False
    assert result.error == "Tool arguments must be an object."


# =========================================================================
# Registry tests
# =========================================================================

def test_executor_rejects_unknown_tool():

    executor = create_executor()

    user = create_user()

    result = executor.execute(
        "unknown_tool",
        {"message": "hello"},
        user,
    )

    assert result.success is False
    assert result.error == "Tool not found."


def test_executor_rejects_invalid_tool_name():

    executor = create_executor()

    user = create_user()

    result = executor.execute(
        "",
        {"message": "hello"},
        user,
    )

    assert result.success is False
    assert result.error == "Invalid tool name."


# =========================================================================
# Execution tests
# =========================================================================

def test_executor_handles_tool_exception():

    executor = create_executor(
        FailingTool()
    )

    user = create_user()

    result = executor.execute(
        "failing_tool",
        {"message": "hello"},
        user,
    )

    assert result.success is False
    assert result.error == "Tool execution failed."


def test_executor_rejects_invalid_output():

    executor = create_executor(
        InvalidOutputTool()
    )

    user = create_user()

    result = executor.execute(
        "invalid_output_tool",
        {"message": "hello"},
        user,
    )

    assert result.success is False
    assert result.error == "Tool returned an invalid result."


def test_executor_rejects_non_serializable_output():

    executor = create_executor(
        NonSerializableTool()
    )

    user = create_user()

    result = executor.execute(
        "non_serializable_tool",
        {"message": "hello"},
        user,
    )

    assert result.success is False


def test_executor_returns_execution_metadata():

    executor = create_executor()

    user = create_user()

    result = executor.execute(
        "test_tool",
        {"message": "hello"},
        user,
    )

    assert result.success is True
    assert "execution_time_ms" in result.metadata
    assert result.metadata["execution_time_ms"] >= 0