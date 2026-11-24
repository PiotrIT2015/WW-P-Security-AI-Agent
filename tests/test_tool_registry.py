"""
Tests for WW-P Security AI Agent Tool Registry.
"""

from authentication.auth import AuthenticationManager

from tools.base import BaseTool
from tools.registry import ToolRegistry
from tools.executor import ToolExecutor


class DemoTool(BaseTool):

    name = "demo_tool"

    description = "Demo tool for testing."

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

    def execute(
        self,
        arguments,
        user,
    ):
        return {
            "message": arguments["message"],
            "executed_by": user.username,
        }


def test_tool_registry_and_execution():

    auth = AuthenticationManager()

    user = auth.create_user(
        username="demo",
        password="SecurePassword123!",
        roles={"analyst"},
        groups={"security"},
    )

    user_context, _ = auth.authenticate(
        "demo",
        "SecurePassword123!",
    )

    registry = ToolRegistry()

    registry.register(
        DemoTool()
    )

    executor = ToolExecutor(
        registry
    )

    result = executor.execute(
        tool_name="demo_tool",
        arguments={
            "message": "Hello Agent",
        },
        user=user_context,
    )

    assert result.success is True

    assert result.output["message"] == "Hello Agent"

    assert result.output["executed_by"] == "demo"


def test_tool_rejects_unknown_arguments():

    auth = AuthenticationManager()

    auth.create_user(
        username="demo",
        password="SecurePassword123!",
        roles={"analyst"},
    )

    user_context, _ = auth.authenticate(
        "demo",
        "SecurePassword123!",
    )

    registry = ToolRegistry()

    registry.register(
        DemoTool()
    )

    executor = ToolExecutor(
        registry
    )

    result = executor.execute(
        tool_name="demo_tool",
        arguments={
            "message": "Hello",
            "delete_all": True,
        },
        user=user_context,
    )

    assert result.success is False

    assert result.error == "Invalid tool arguments."


def test_tool_rejects_unauthorized_user():

    auth = AuthenticationManager()

    auth.create_user(
        username="reader",
        password="SecurePassword123!",
        roles={"reader"},
    )

    user_context, _ = auth.authenticate(
        "reader",
        "SecurePassword123!",
    )

    registry = ToolRegistry()

    registry.register(
        DemoTool()
    )

    executor = ToolExecutor(
        registry
    )

    result = executor.execute(
        tool_name="demo_tool",
        arguments={
            "message": "Hello",
        },
        user=user_context,
    )

    assert result.success is False

    assert result.error == "Permission denied."