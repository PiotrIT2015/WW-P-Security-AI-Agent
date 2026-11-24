"""
tests/test_tool_security.py

Tests for Tool Security Manager.
"""

import pytest

from tools.security import (
    ToolSecurityConfig,
    ToolSecurityManager,
)


@pytest.fixture
def manager():

    return ToolSecurityManager()


# =============================================================================
# Tool blocking
# =============================================================================

def test_tool_can_be_blocked(manager):

    manager.block_tool("dangerous_tool")

    assert manager.is_blocked("dangerous_tool")


def test_blocked_tool_is_denied(manager):

    manager.block_tool("dangerous_tool")

    result = manager.evaluate(
        "dangerous_tool",
        "user-1",
        {},
    )

    assert result.allowed is False
    assert result.reason == "tool_blocked"


def test_tool_can_be_unblocked(manager):

    manager.block_tool("dangerous_tool")

    manager.unblock_tool("dangerous_tool")

    assert not manager.is_blocked("dangerous_tool")


# =============================================================================
# Argument validation
# =============================================================================

def test_valid_arguments(manager):

    result = manager.evaluate(
        "test_tool",
        "user-1",
        {"message": "hello"},
    )

    assert result.allowed is True


def test_non_dictionary_arguments(manager):

    result = manager.evaluate(
        "test_tool",
        "user-1",
        ["invalid"],
    )

    assert result.allowed is False
    assert result.reason == "arguments_not_object"


def test_oversized_arguments():

    manager = ToolSecurityManager(
        ToolSecurityConfig(
            max_argument_bytes=20,
        )
    )

    result = manager.evaluate(
        "test_tool",
        "user-1",
        {"message": "A" * 100},
    )

    assert result.allowed is False
    assert result.reason == "argument_size_exceeded"


def test_excessive_nesting():

    manager = ToolSecurityManager(
        ToolSecurityConfig(
            max_nesting_depth=2,
        )
    )

    arguments = {
        "level1": {
            "level2": {
                "level3": "value"
            }
        }
    }

    result = manager.evaluate(
        "test_tool",
        "user-1",
        arguments,
    )

    assert result.allowed is False
    assert result.reason == "maximum_nesting_depth_exceeded"


def test_invalid_json_value(manager):

    result = manager.evaluate(
        "test_tool",
        "user-1",
        {"value": float("nan")},
    )

    assert result.allowed is False
    assert result.reason == "invalid_json_arguments"


# =============================================================================
# Rate limiting
# =============================================================================

def test_rate_limit_allows_calls():

    manager = ToolSecurityManager(
        ToolSecurityConfig(
            max_calls=2,
            window_seconds=60,
        )
    )

    first = manager.evaluate(
        "test_tool",
        "user-1",
        {},
    )

    second = manager.evaluate(
        "test_tool",
        "user-1",
        {},
    )

    assert first.allowed is True
    assert second.allowed is True


def test_rate_limit_blocks_excessive_calls():

    manager = ToolSecurityManager(
        ToolSecurityConfig(
            max_calls=2,
            window_seconds=60,
        )
    )

    for _ in range(2):

        manager.evaluate(
            "test_tool",
            "user-1",
            {},
        )

    result = manager.evaluate(
        "test_tool",
        "user-1",
        {},
    )

    assert result.allowed is False
    assert result.reason == "rate_limit_exceeded"


def test_rate_limit_is_per_user():

    manager = ToolSecurityManager(
        ToolSecurityConfig(
            max_calls=1,
            window_seconds=60,
        )
    )

    manager.evaluate(
        "test_tool",
        "user-1",
        {},
    )

    result = manager.evaluate(
        "test_tool",
        "user-2",
        {},
    )

    assert result.allowed is True


def test_rate_limit_is_per_tool():

    manager = ToolSecurityManager(
        ToolSecurityConfig(
            max_calls=1,
            window_seconds=60,
        )
    )

    manager.evaluate(
        "tool_one",
        "user-1",
        {},
    )

    result = manager.evaluate(
        "tool_two",
        "user-1",
        {},
    )

    assert result.allowed is True


# =============================================================================
# Validation
# =============================================================================

def test_invalid_user_id(manager):

    result = manager.evaluate(
        "test_tool",
        "",
        {},
    )

    assert result.allowed is False
    assert result.reason == "invalid_user_id"


def test_invalid_tool_name(manager):

    result = manager.evaluate(
        "",
        "user-1",
        {},
    )

    assert result.allowed is False
    assert result.reason == "invalid_tool_name"


def test_case_insensitive_tool_blocking(manager):

    manager.block_tool("Dangerous_Tool")

    result = manager.evaluate(
        "DANGEROUS_TOOL",
        "user-1",
        {},
    )

    assert result.allowed is False
    assert result.reason == "tool_blocked"


def test_security_configuration_rejects_zero_limit():

    with pytest.raises(ValueError):

        ToolSecurityConfig(
            max_calls=0,
        )