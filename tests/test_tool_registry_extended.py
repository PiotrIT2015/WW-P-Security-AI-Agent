"""
Extended tests for ToolRegistry.

WW-P Security AI Agent
"""

import pytest

from tools.base import BaseTool
from tools.registry import (
    ToolRegistry,
    ToolNotFoundError,
    ToolAlreadyRegisteredError,
)


class DummyTool(BaseTool):

    name = "dummy_tool"
    description = "Dummy tool for registry tests."
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
            "message": arguments["message"]
        }


def create_registry():
    registry = ToolRegistry()
    return registry


def test_register_tool():
    registry = create_registry()
    tool = DummyTool()

    registry.register(tool)

    assert registry.count == 1
    assert registry.get("dummy_tool") is tool


def test_duplicate_registration_is_rejected():
    registry = create_registry()

    registry.register(DummyTool())

    with pytest.raises(ToolAlreadyRegisteredError):
        registry.register(DummyTool())


def test_get_unknown_tool():
    registry = create_registry()

    with pytest.raises(ToolNotFoundError):
        registry.get("unknown_tool")


def test_list_tools():
    registry = create_registry()

    tool = DummyTool()
    registry.register(tool)

    tools = registry.list_tools()

    assert len(tools) == 1
    assert tools[0]["name"] == "dummy_tool"
    assert tools[0]["description"] == tool.description
    assert tools[0]["input_schema"] == tool.input_schema


def test_list_names():
    registry = create_registry()

    registry.register(DummyTool())

    names = registry.list_names()

    assert "dummy_tool" in names


def test_unregister_tool():
    registry = create_registry()

    registry.register(DummyTool())

    assert registry.count == 1

    registry.unregister("dummy_tool")

    assert registry.count == 0
    assert registry.list_names() == []


def test_unregister_unknown_tool():
    registry = create_registry()

    with pytest.raises(ToolNotFoundError):
        registry.unregister("unknown_tool")


def test_registry_can_register_multiple_tools():

    class SecondDummyTool(DummyTool):
        name = "second_dummy_tool"

    registry = create_registry()

    registry.register(DummyTool())
    registry.register(SecondDummyTool())

    assert registry.count == 2

    names = registry.list_names()

    assert "dummy_tool" in names
    assert "second_dummy_tool" in names