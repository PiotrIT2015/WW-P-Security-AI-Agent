import pytest

from types import SimpleNamespace

from agent.core import AgentCore
from agent.state import AgentStatus
from agent.exceptions import AgentConfigurationError


class FakeOllamaClient:

    def chat(self, messages, tools=None):
        return {
            "role": "assistant",
            "content": "Test response"
        }


class FakeToolAdapter:

    def adapt_registry(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "system_status",
                    "description": "Returns system status.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False
                    }
                }
            }
        ]


class FakeToolBridge:

    def execute_message(self, message, user):
        return []


class FakeResultHandler:

    def to_tool_messages(self, results):
        return []


@pytest.fixture
def core():
    return AgentCore(
        ollama_client=FakeOllamaClient(),
        tool_adapter=FakeToolAdapter(),
        tool_bridge=FakeToolBridge(),
        result_handler=FakeResultHandler()
    )


@pytest.fixture
def user():
    return SimpleNamespace(
        user_id="test-user",
        username="test",
        roles=frozenset({"admin"}),
        groups=frozenset({"security"}),
        session_id="test-session"
    )


def test_agent_core_initialization(core):
    assert core.max_iterations == 5


def test_agent_core_custom_iteration_limit():
    core = AgentCore(
        FakeOllamaClient(),
        FakeToolAdapter(),
        FakeToolBridge(),
        FakeResultHandler(),
        max_iterations=10
    )

    assert core.max_iterations == 10


def test_reject_invalid_iteration_limit():
    with pytest.raises(AgentConfigurationError):
        AgentCore(
            FakeOllamaClient(),
            FakeToolAdapter(),
            FakeToolBridge(),
            FakeResultHandler(),
            max_iterations=0
        )


def test_reject_missing_ollama_client():
    with pytest.raises(AgentConfigurationError):
        AgentCore(
            None,
            FakeToolAdapter(),
            FakeToolBridge(),
            FakeResultHandler()
        )


def test_create_agent_state(core, user):
    state = core.create_state(user)

    assert state.user_id == "test-user"
    assert state.status == AgentStatus.READY
    assert len(state.messages) == 1
    assert state.messages[0]["role"] == "system"


def test_create_agent_state_with_conversation_id(core, user):
    state = core.create_state(
        user,
        conversation_id="custom-conversation"
    )

    assert state.conversation_id == "custom-conversation"


def test_reject_missing_user(core):
    with pytest.raises(AgentConfigurationError):
        core.create_state(None)


def test_get_tool_definitions(core):
    definitions = core.get_tool_definitions()

    assert len(definitions) == 1
    assert definitions[0]["function"]["name"] == "system_status"


def test_prepare_messages(core, user):
    state = core.create_state(user)

    messages = core.prepare_messages(state)

    assert len(messages) == 1
    assert messages[0]["role"] == "system"


def test_get_status(core, user):
    state = core.create_state(user)

    status = core.get_status(state)

    assert status["status"] == "ready"
    assert status["iteration"] == 0
    assert status["tool_calls_count"] == 0
    assert status["is_finished"] is False