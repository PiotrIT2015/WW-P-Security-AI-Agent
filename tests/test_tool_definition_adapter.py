"""
tests/test_tool_definition_adapter.py

Unit tests for ToolDefinitionAdapter.
"""

import pytest

from llm.tool_adapter import (
    ToolDefinitionAdapter,
    ToolDefinitionAdapterError,
)


# ============================================================
# Test registry
# ============================================================

class MockToolRegistry:

    def __init__(self, tools=None):
        self.tools = tools or []

    def list_tools(self):
        return self.tools


# ============================================================
# Helpers
# ============================================================

def create_tool_definition(
    name="system_status",
    description="Returns system status.",
    properties=None,
    required=None,
    required_roles=None,
):

    return {
        "name": name,
        "description": description,
        "required_roles": required_roles or [],
        "input_schema": {
            "type": "object",
            "properties": properties or {},
            "required": required or [],
            "additionalProperties": False,
        },
    }


def create_adapter(tools=None):

    registry = MockToolRegistry(tools)

    return ToolDefinitionAdapter(registry)


# ============================================================
# 1. Single tool conversion
# ============================================================

def test_adapt_single_tool():

    adapter = create_adapter()

    definition = create_tool_definition()

    result = adapter.adapt_tool_definition(
        definition
    )

    assert result["type"] == "function"

    assert result["function"]["name"] == "system_status"

    assert result["function"]["description"] == (
        "Returns system status."
    )

    assert result["function"]["parameters"]["type"] == "object"


# ============================================================
# 2. Tool with arguments
# ============================================================

def test_adapt_tool_with_arguments():

    adapter = create_adapter()

    definition = create_tool_definition(
        name="network_monitor",
        properties={
            "interface": {
                "type": "string",
                "description": "Network interface name.",
            },
            "verbose": {
                "type": "boolean",
            },
        },
        required=["interface"],
    )

    result = adapter.adapt_tool_definition(
        definition
    )

    parameters = result["function"]["parameters"]

    assert "interface" in parameters["properties"]

    assert "verbose" in parameters["properties"]

    assert parameters["required"] == ["interface"]


# ============================================================
# 3. Multiple tool conversion
# ============================================================

def test_adapt_multiple_tools():

    tools = [
        create_tool_definition(
            name="system_status"
        ),
        create_tool_definition(
            name="network_monitor"
        ),
        create_tool_definition(
            name="security_scan"
        ),
    ]

    adapter = create_adapter(tools)

    result = adapter.adapt_registry()

    assert len(result) == 3

    names = [
        item["function"]["name"]
        for item in result
    ]

    assert names == [
        "system_status",
        "network_monitor",
        "security_scan",
    ]


# ============================================================
# 4. Empty registry
# ============================================================

def test_adapt_empty_registry():

    adapter = create_adapter([])

    result = adapter.adapt_registry()

    assert result == []


# ============================================================
# 5. Required roles must not be exposed
# ============================================================

def test_required_roles_are_not_exposed():

    adapter = create_adapter()

    definition = create_tool_definition(
        required_roles=["admin", "security"],
    )

    result = adapter.adapt_tool_definition(
        definition
    )

    assert "required_roles" not in result

    assert "admin" not in str(result)

    assert "security" not in str(result)


# ============================================================
# 6. Invalid tool name
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
        None,
        123,
    ],
)
def test_reject_invalid_tool_name(name):

    adapter = create_adapter()

    definition = create_tool_definition(
        name=name
    )

    with pytest.raises(ToolDefinitionAdapterError):
        adapter.adapt_tool_definition(definition)


# ============================================================
# 7. Invalid description
# ============================================================

@pytest.mark.parametrize(
    "description",
    [
        "",
        " ",
        None,
        123,
    ],
)
def test_reject_invalid_description(description):

    adapter = create_adapter()

    definition = create_tool_definition(
        description=description
    )

    with pytest.raises(ToolDefinitionAdapterError):
        adapter.adapt_tool_definition(definition)


# ============================================================
# 8. Missing input schema
# ============================================================

def test_reject_missing_input_schema():

    adapter = create_adapter()

    definition = {
        "name": "system_status",
        "description": "Returns system status.",
    }

    with pytest.raises(ToolDefinitionAdapterError):
        adapter.adapt_tool_definition(definition)


# ============================================================
# 9. Invalid schema type
# ============================================================

def test_reject_invalid_schema_type():

    adapter = create_adapter()

    definition = create_tool_definition()

    definition["input_schema"]["type"] = "string"

    with pytest.raises(ToolDefinitionAdapterError):
        adapter.adapt_tool_definition(definition)


# ============================================================
# 10. Invalid properties
# ============================================================

@pytest.mark.parametrize(
    "properties",
    [
        None,
        [],
        "invalid",
        123,
    ],
)
def test_reject_invalid_properties(properties):

    adapter = create_adapter()

    definition = create_tool_definition()

    definition["input_schema"]["properties"] = properties

    with pytest.raises(ToolDefinitionAdapterError):
        adapter.adapt_tool_definition(definition)


# ============================================================
# 11. Invalid required field
# ============================================================

@pytest.mark.parametrize(
    "required",
    [
        "interface",
        None,
        123,
        [123],
    ],
)
def test_reject_invalid_required(required):

    adapter = create_adapter()

    definition = create_tool_definition()

    definition["input_schema"]["required"] = required

    with pytest.raises(ToolDefinitionAdapterError):
        adapter.adapt_tool_definition(definition)


# ============================================================
# 12. Required field missing from properties
# ============================================================

def test_reject_required_field_missing_from_properties():

    adapter = create_adapter()

    definition = create_tool_definition(
        properties={
            "interface": {
                "type": "string"
            }
        },
        required=["unknown_field"],
    )

    with pytest.raises(
        ToolDefinitionAdapterError,
        match="missing from properties",
    ):
        adapter.adapt_tool_definition(definition)


# ============================================================
# 13. Duplicate required fields
# ============================================================

def test_reject_duplicate_required_fields():

    adapter = create_adapter()

    definition = create_tool_definition(
        properties={
            "interface": {
                "type": "string"
            }
        },
        required=["interface", "interface"],
    )

    with pytest.raises(
        ToolDefinitionAdapterError,
        match="duplicate required fields",
    ):
        adapter.adapt_tool_definition(definition)


# ============================================================
# 14. Invalid additionalProperties
# ============================================================

@pytest.mark.parametrize(
    "value",
    [
        "false",
        None,
        [],
        123,
    ],
)
def test_reject_invalid_additional_properties(value):

    adapter = create_adapter()

    definition = create_tool_definition()

    definition["input_schema"]["additionalProperties"] = value

    with pytest.raises(ToolDefinitionAdapterError):
        adapter.adapt_tool_definition(definition)


# ============================================================
# 15. Schema is copied
# ============================================================

def test_schema_is_copied():

    adapter = create_adapter()

    definition = create_tool_definition(
        properties={
            "target": {
                "type": "string"
            }
        }
    )

    result = adapter.adapt_tool_definition(
        definition
    )

    result["function"]["parameters"]["properties"]["target"]["type"] = (
        "integer"
    )

    assert (
        definition["input_schema"]["properties"]["target"]["type"]
        == "string"
    )


# ============================================================
# 16. Invalid registry response
# ============================================================

def test_reject_invalid_registry_response():

    class InvalidRegistry:

        def list_tools(self):
            return {}

    adapter = ToolDefinitionAdapter(
        InvalidRegistry()
    )

    with pytest.raises(ToolDefinitionAdapterError):
        adapter.adapt_registry()


# ============================================================
# 17. Duplicate tool names
# ============================================================

def test_reject_duplicate_tool_names():

    tools = [
        create_tool_definition(
            name="system_status"
        ),
        create_tool_definition(
            name="system_status"
        ),
    ]

    adapter = create_adapter(tools)

    with pytest.raises(
        ToolDefinitionAdapterError,
        match="Duplicate tool definition",
    ):
        adapter.adapt_registry()


# ============================================================
# 18. Invalid tool definition in registry
# ============================================================

def test_reject_invalid_definition_in_registry():

    tools = [
        create_tool_definition(
            name="system_status"
        ),
        create_tool_definition(
            name="invalid tool name"
        ),
    ]

    adapter = create_adapter(tools)

    with pytest.raises(
        ToolDefinitionAdapterError,
        match="Invalid tool definition at index 1",
    ):
        adapter.adapt_registry()


# ============================================================
# 19. Count tools
# ============================================================

def test_count_tools():

    tools = [
        create_tool_definition("system_status"),
        create_tool_definition("network_monitor"),
    ]

    adapter = create_adapter(tools)

    assert adapter.count() == 2


# ============================================================
# 20. Missing schema fields receive defaults
# ============================================================

def test_schema_defaults():

    adapter = create_adapter()

    definition = {
        "name": "system_status",
        "description": "Returns system status.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    }

    result = adapter.adapt_tool_definition(
        definition
    )

    parameters = result["function"]["parameters"]

    assert parameters["required"] == []

    assert parameters["additionalProperties"] is False


# ============================================================
# 21. Reject non-JSON schema values
# ============================================================

def test_reject_non_json_schema_values():

    adapter = create_adapter()

    definition = create_tool_definition()

    definition["input_schema"]["custom"] = object()

    with pytest.raises(
        ToolDefinitionAdapterError,
        match="JSON-compatible",
    ):
        adapter.adapt_tool_definition(definition)