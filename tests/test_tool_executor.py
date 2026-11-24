"""
tests/test_tool_executor.py

Unit tests for ToolExecutor.

WW-P Security AI Agent
Python 3.10+
"""

import pytest

from authentication.auth import UserContext

from tools.base import BaseTool, ToolResult
from tools.registry import ToolRegistry
from tools.executor import ToolExecutor


# =========================================================================
# Test tool
# =========================================================================

class EchoTool(BaseTool):

    name = "echo_tool"
    description = "Returns the supplied message."
    required_roles = {"analyst"}

    input_schema = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "minLength": 1,
                "maxLength": 100,
            }
        },
        "required": ["message"],
        "additionalProperties": False,
    }

    def execute(self, arguments, user):
        return {
            "message": arguments["message"],
            "executed_by": user.username,
        }


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def user():
    return UserContext(
        user_id="test-user-001",
        username="test-analyst",
        roles=frozenset({"analyst"}),
        groups=frozenset({"security"}),
        authenticated_at=1234567890.0,
        session_id="test-session-001",
    )


@pytest.fixture
def registry():
    instance = ToolRegistry()
    instance.register(EchoTool())
    return instance


@pytest.fixture
def executor(registry):
    return ToolExecutor(registry)


# =========================================================================
# Successful execution
# =========================================================================

def test_executor_initialization(registry):
    executor = ToolExecutor(registry)

    assert executor.registry is registry


def test_executor_executes_registered_tool(executor, user):

    result = executor.execute(
        tool_name="echo_tool",
        arguments={"message": "Hello WW-P"},
        user=user,
    )

    assert isinstance(result, ToolResult)
    assert result.success is True

    assert result.tool_name == "echo_tool"

    assert result.output == {
        "message": "Hello WW-P",
        "executed_by": "test-analyst",
    }

    assert result.error is None


def test_executor_returns_tool_result(executor, user):

    result = executor.execute(
        "echo_tool",
        {"message": "Test"},
        user,
    )

    assert isinstance(result, ToolResult)


# =========================================================================
# Authentication
# =========================================================================

def test_executor_rejects_missing_user(executor):

    result = executor.execute(
        "echo_tool",
        {"message": "Hello"},
        None,
    )

    assert result.success is False
    assert result.error == "Authentication required."


def test_executor_rejects_invalid_user_context(executor):

    result = executor.execute(
        "echo_tool",
        {"message": "Hello"},
        "invalid-user",
    )

    assert result.success is False
    assert result.error == "Authentication required."


# =========================================================================
# Authorization
# =========================================================================

def test_executor_allows_user_with_required_role(executor, user):

    result = executor.execute(
        "echo_tool",
        {"message": "Authorized"},
        user,
    )

    assert result.success is True


def test_executor_denies_user_without_required_role(executor):

    unauthorized_user = UserContext(
        user_id="viewer-001",
        username="viewer",
        roles=frozenset({"viewer"}),
        groups=frozenset(),
        authenticated_at=1234567890.0,
        session_id="viewer-session",
    )

    result = executor.execute(
        "echo_tool",
        {"message": "Unauthorized"},
        unauthorized_user,
    )

    assert result.success is False
    assert result.error == "Permission denied."


def test_executor_allows_admin(executor):

    admin_user = UserContext(
        user_id="admin-001",
        username="admin",
        roles=frozenset({"admin"}),
        groups=frozenset(),
        authenticated_at=1234567890.0,
        session_id="admin-session",
    )

    result = executor.execute(
        "echo_tool",
        {"message": "Admin access"},
        admin_user,
    )

    assert result.success is True


# =========================================================================
# Registry
# =========================================================================

def test_executor_rejects_unknown_tool(executor, user):

    result = executor.execute(
        "unknown_tool",
        {"message": "Hello"},
        user,
    )

    assert result.success is False
    assert result.error == "Tool not found."


def test_executor_rejects_empty_tool_name(executor, user):

    result = executor.execute(
        "",
        {"message": "Hello"},
        user,
    )

    assert result.success is False
    assert result.error == "Invalid tool name."


# =========================================================================
# Argument validation
# =========================================================================

def test_executor_rejects_missing_required_argument(executor, user):

    result = executor.execute(
        "echo_tool",
        {},
        user,
    )

    assert result.success is False
    assert result.error == "Invalid tool arguments."


def test_executor_rejects_unknown_argument(executor, user):

    result = executor.execute(
        "echo_tool",
        {
            "message": "Hello",
            "extra": "unexpected",
        },
        user,
    )

    assert result.success is False
    assert result.error == "Invalid tool arguments."


def test_executor_rejects_wrong_argument_type(executor, user):

    result = executor.execute(
        "echo_tool",
        {"message": 123},
        user,
    )

    assert result.success is False
    assert result.error == "Invalid tool arguments."


def test_executor_rejects_non_dictionary_arguments(executor, user):

    result = executor.execute(
        "echo_tool",
        ["Hello"],
        user,
    )

    assert result.success is False
    assert result.error == "Tool arguments must be an object."


# =========================================================================
# Execution metadata
# =========================================================================

def test_executor_returns_execution_time(executor, user):

    result = executor.execute(
        "echo_tool",
        {"message": "Timing test"},
        user,
    )

    assert result.success is True

    assert "execution_time_ms" in result.metadata

    assert isinstance(
        result.metadata["execution_time_ms"],
        (int, float),
    )

    assert result.metadata["execution_time_ms"] >= 0