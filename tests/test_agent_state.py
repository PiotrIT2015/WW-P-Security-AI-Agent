import pytest

from agent.state import AgentState, AgentStatus


@pytest.fixture
def state():
    return AgentState(
        conversation_id="test-conversation",
        user_id="test-user"
    )


def test_state_initialization(state):
    assert state.status == AgentStatus.READY
    assert state.iteration == 0
    assert state.tool_calls_count == 0
    assert state.messages == []


def test_append_message(state):
    state.append_message({
        "role": "user",
        "content": "Check system status."
    })

    assert len(state.messages) == 1
    assert state.messages[0]["role"] == "user"


def test_reject_invalid_message_role(state):
    with pytest.raises(ValueError):
        state.append_message({
            "role": "invalid",
            "content": "Test"
        })


def test_reject_non_string_content(state):
    with pytest.raises(ValueError):
        state.append_message({
            "role": "user",
            "content": 123
        })


def test_increment_iteration(state):
    state.increment_iteration()

    assert state.iteration == 1


def test_increment_tool_calls(state):
    state.increment_tool_calls(2)

    assert state.tool_calls_count == 2


def test_start_state(state):
    state.start()

    assert state.status == AgentStatus.RUNNING


def test_complete_state(state):
    state.start()
    state.complete("System is healthy.")

    assert state.status == AgentStatus.COMPLETED
    assert state.final_response == "System is healthy."
    assert state.is_finished()


def test_fail_state(state):
    state.fail("Execution error.")

    assert state.status == AgentStatus.FAILED
    assert state.error == "Execution error."
    assert state.is_finished()


def test_cannot_start_twice(state):
    state.start()

    with pytest.raises(ValueError):
        state.start()


def test_cannot_modify_completed_state(state):
    state.start()
    state.complete("Finished.")

    with pytest.raises(ValueError):
        state.append_message({
            "role": "user",
            "content": "Another message"
        })


def test_get_messages_returns_copy(state):
    state.append_message({
        "role": "user",
        "content": "Original"
    })

    messages = state.get_messages()
    messages[0]["content"] = "Modified"

    assert state.messages[0]["content"] == "Original"


def test_tool_results_are_stored(state):
    state.add_tool_results([
        {
            "tool_name": "system_status",
            "success": True
        }
    ])

    assert len(state.tool_results) == 1


def test_negative_iteration_is_rejected():
    with pytest.raises(ValueError):
        AgentState(
            conversation_id="test",
            user_id="user",
            iteration=-1
        )


def test_invalid_tool_call_count_is_rejected(state):
    with pytest.raises(ValueError):
        state.increment_tool_calls(-1)