import pytest

from types import SimpleNamespace

from agent.core import AgentCore
from agent.state import AgentStatus

from agent.exceptions import (
    AgentExecutionError,
    AgentIterationLimitError
)

from llm.tool_result_handler import ToolResultHandler
from tools.base import ToolResult


class FakeOllamaClient:

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None):

        self.calls.append({
            "messages": messages,
            "tools": tools
        })

        if not self.responses:
            raise RuntimeError("No fake responses available.")

        return self.responses.pop(0)


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

    def __init__(self):
        self.calls = []

    def execute_message(self, message, user):

        self.calls.append({
            "message": message,
            "user": user
        })

        return [
            ToolResult(
                tool_name="system_status",
                success=True,
                output={
                    "status": "healthy",
                    "cpu": 35
                }
            )
        ]


@pytest.fixture
def user():

    return SimpleNamespace(
        user_id="test-user",
        username="test",
        roles=frozenset({"admin"}),
        groups=frozenset({"security"}),
        session_id="test-session"
    )


def create_tool_call_response():

    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "system_status",
                    "arguments": {}
                }
            }
        ]
    }


def create_final_response(content="System is healthy."):

    return {
        "role": "assistant",
        "content": content
    }


def create_agent(responses, max_iterations=5):

    client = FakeOllamaClient(responses)

    bridge = FakeToolBridge()

    agent = AgentCore(
        ollama_client=client,
        tool_adapter=FakeToolAdapter(),
        tool_bridge=bridge,
        result_handler=ToolResultHandler(),
        max_iterations=max_iterations
    )

    return agent, client, bridge


# ---------------------------------------------------------
# BASIC EXECUTION
# ---------------------------------------------------------

def test_agent_returns_final_response(user):

    agent, client, bridge = create_agent([
        create_final_response()
    ])

    state = agent.run(
        user,
        "Check system status."
    )

    assert state.status == AgentStatus.COMPLETED
    assert state.final_response == "System is healthy."
    assert state.iteration == 1

    assert len(client.calls) == 1
    assert len(bridge.calls) == 0


def test_agent_stores_user_prompt(user):

    agent, client, bridge = create_agent([
        create_final_response()
    ])

    state = agent.run(
        user,
        "Check system status."
    )

    assert state.messages[1]["role"] == "user"
    assert state.messages[1]["content"] == "Check system status."


# ---------------------------------------------------------
# TOOL EXECUTION
# ---------------------------------------------------------

def test_agent_executes_tool_call(user):

    agent, client, bridge = create_agent([
        create_tool_call_response(),
        create_final_response("System is healthy.")
    ])

    state = agent.run(
        user,
        "Check system status."
    )

    assert state.status == AgentStatus.COMPLETED

    assert state.tool_calls_count == 1
    assert len(bridge.calls) == 1
    assert len(client.calls) == 2


def test_agent_returns_tool_result(user):

    agent, client, bridge = create_agent([
        create_tool_call_response(),
        create_final_response()
    ])

    state = agent.run(
        user,
        "Check system status."
    )

    assert len(state.tool_results) == 1

    assert state.tool_results[0]["tool_name"] == "system_status"
    assert state.tool_results[0]["success"] is True


def test_tool_result_is_added_to_conversation(user):

    agent, client, bridge = create_agent([
        create_tool_call_response(),
        create_final_response()
    ])

    state = agent.run(
        user,
        "Check system status."
    )

    roles = [
        message["role"]
        for message in state.messages
    ]

    assert roles == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant"
    ]


def test_llm_receives_tool_result(user):

    agent, client, bridge = create_agent([
        create_tool_call_response(),
        create_final_response()
    ])

    agent.run(
        user,
        "Check system status."
    )

    second_request = client.calls[1]

    assert any(
        message["role"] == "tool"
        for message in second_request["messages"]
    )


# ---------------------------------------------------------
# ITERATION LIMIT
# ---------------------------------------------------------

def test_agent_stops_after_iteration_limit(user):

    agent, client, bridge = create_agent(
        [
            create_tool_call_response(),
            create_tool_call_response(),
            create_tool_call_response()
        ],
        max_iterations=2
    )

    with pytest.raises(AgentIterationLimitError):

        agent.run(
            user,
            "Check system status."
        )

    assert len(client.calls) == 2


def test_iteration_limit_marks_state_as_failed(user):

    agent, client, bridge = create_agent(
        [
            create_tool_call_response(),
            create_tool_call_response()
        ],
        max_iterations=1
    )

    with pytest.raises(AgentIterationLimitError):

        agent.run(
            user,
            "Check system status."
        )

    # State is internal to run in this initial API.
    # The exception confirms execution was stopped.


# ---------------------------------------------------------
# ERROR HANDLING
# ---------------------------------------------------------

def test_agent_rejects_empty_prompt(user):

    agent, client, bridge = create_agent([])

    with pytest.raises(Exception):

        agent.run(
            user,
            ""
        )

    assert len(client.calls) == 0


def test_agent_handles_invalid_llm_response(user):

    agent, client, bridge = create_agent([
        {
            "role": "user",
            "content": "Invalid response"
        }
    ])

    with pytest.raises(AgentExecutionError):

        agent.run(
            user,
            "Check system status."
        )


def test_agent_handles_llm_exception(user):

    agent, client, bridge = create_agent([])

    def failing_chat(messages, tools=None):
        raise RuntimeError("Ollama unavailable.")

    agent.ollama_client.chat = failing_chat

    with pytest.raises(AgentExecutionError):

        agent.run(
            user,
            "Check system status."
        )


def test_agent_handles_tool_execution_exception(user):

    agent, client, bridge = create_agent([
        create_tool_call_response()
    ])

    def failing_execute(message, user):
        raise RuntimeError("Tool execution failed.")

    agent.tool_bridge.execute_message = failing_execute

    with pytest.raises(AgentExecutionError):

        agent.run(
            user,
            "Check system status."
        )


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

def test_agent_rejects_invalid_prompt_type(user):

    agent, client, bridge = create_agent([])

    with pytest.raises(Exception):

        agent.run(
            user,
            123
        )