import json

import pytest

from agent.response_generator import ResponseGenerator
from agent.exceptions import (
    AgentConfigurationError,
    AgentExecutionError,
)
from agent.rag.context_builder import BuiltRAGContext

from agent.rag.sources import RAGSource

# =========================================================
# MOCK OLLAMA
# =========================================================

class MockOllamaClient:

    def __init__(self, response=None, error=None):

        self.response = response
        self.error = error

        self.last_messages = None
        self.last_tools = None
        self.call_count = 0

    def chat(self, messages, tools=None):

        self.call_count += 1

        self.last_messages = messages
        self.last_tools = tools

        if self.error:
            raise self.error

        return self.response


# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture
def rag_context():

    return BuiltRAGContext(
        query="What is DNS?",
        context=(
            "DNS translates domain names into IP addresses."
        ),
        user_id="demo-piotr",
        candidates_count=3,
        authorized_count=2,
        denied_count=1,
        sources=(
            RAGSource(
                citation_id=1,
                document_id="networking-doc-001",
                chunk_id="chunk-001",
                score=0.95,
            ),
            RAGSource(
                citation_id=2,
                document_id="networking-doc-002",
                chunk_id="chunk-002",
                score=0.89,
            ),
        ),
    )


@pytest.fixture
def empty_rag_context():

    return BuiltRAGContext(
        query="Explain DNS.",
        context="",
        user_id="demo-piotr",
        candidates_count=0,
        authorized_count=0,
        denied_count=0,
    )


@pytest.fixture
def ollama_client():

    return MockOllamaClient(
        response={
            "role": "assistant",
            "content": (
                "DNS translates human-readable domain names "
                "into IP addresses."
            ),
        }
    )


@pytest.fixture
def generator(ollama_client):

    return ResponseGenerator(
        ollama_client=ollama_client,
    )


# =========================================================
# RAG CONTEXT VALIDATION
# =========================================================

def test_accepts_valid_rag_context(generator, rag_context):

    generator._validate_rag_context(rag_context)


def test_rejects_invalid_rag_context(generator):

    with pytest.raises(AgentConfigurationError):

        generator._validate_rag_context(
            {"query": "What is DNS?"}
        )


def test_rejects_empty_rag_query():

    context = BuiltRAGContext(
        query=" ",
        context="DNS information",
        user_id="demo-piotr",
        candidates_count=1,
        authorized_count=1,
        denied_count=0,
    )

    with pytest.raises(AgentConfigurationError):

        ResponseGenerator._validate_rag_context(
            context
        )





# =========================================================
# RAG MESSAGE PREPARATION
# =========================================================

def test_prepare_rag_messages(generator, rag_context):

    messages = generator.prepare_rag_messages(
        rag_context
    )

    assert len(messages) == 3

    assert messages[0]["role"] == "system"

    assert messages[1]["role"] == "system"

    assert messages[2]["role"] == "user"

    assert messages[2]["content"] == "What is DNS?"


def test_prepare_rag_messages_contains_context(
    generator,
    rag_context,
):

    messages = generator.prepare_rag_messages(
        rag_context
    )

    context_message = messages[1]["content"]

    assert "REFERENCE DATA" in context_message

    assert "DNS translates domain names" in context_message


def test_prepare_rag_messages_json_encodes_context(
    generator,
    rag_context,
):

    messages = generator.prepare_rag_messages(
        rag_context
    )

    context_message = messages[1]["content"]

    assert json.dumps(
        rag_context.context,
        ensure_ascii=False,
    ) in context_message


def test_prepare_rag_messages_marks_context_as_untrusted(
    generator,
    rag_context,
):

    messages = generator.prepare_rag_messages(
        rag_context
    )

    context_message = messages[1]["content"]

    assert "untrusted data" in context_message


def test_prepare_empty_rag_context(
    generator,
    empty_rag_context,
):

    messages = generator.prepare_rag_messages(
        empty_rag_context
    )

    assert "No authorized reference documents" in (
        messages[1]["content"]
    )


# =========================================================
# RAG ANSWER GENERATION
# =========================================================

def test_generate_rag_answer(
    generator,
    rag_context,
):

    result = generator.generate_rag_answer(
        rag_context
    )

    assert "DNS translates" in result


def test_generate_rag_answer_calls_ollama_once(
    generator,
    rag_context,
):

    generator.generate_rag_answer(
        rag_context
    )

    assert generator.ollama_client.call_count == 1


def test_generate_rag_answer_disables_tools(
    generator,
    rag_context,
):

    generator.generate_rag_answer(
        rag_context
    )

    assert generator.ollama_client.last_tools is None


def test_generate_rag_answer_passes_rag_context(
    generator,
    rag_context,
):

    generator.generate_rag_answer(
        rag_context
    )

    messages = generator.ollama_client.last_messages

    assert "DNS translates domain names" in (
        messages[1]["content"]
    )


def test_generate_rag_answer_passes_original_query(
    generator,
    rag_context,
):

    generator.generate_rag_answer(
        rag_context
    )

    messages = generator.ollama_client.last_messages

    assert messages[2]["content"] == rag_context.query


def test_generate_rag_answer_handles_empty_context(
    generator,
    empty_rag_context,
):

    generator.generate_rag_answer(
        empty_rag_context
    )

    messages = generator.ollama_client.last_messages

    assert "No authorized reference documents" in (
        messages[1]["content"]
    )


def test_generate_rag_answer_wraps_ollama_error(
    rag_context,
):

    client = MockOllamaClient(
        error=RuntimeError("Ollama unavailable")
    )

    generator = ResponseGenerator(client)

    with pytest.raises(AgentExecutionError):

        generator.generate_rag_answer(
            rag_context
        )


def test_generate_rag_answer_rejects_empty_response(
    rag_context,
):

    client = MockOllamaClient(
        response={
            "role": "assistant",
            "content": "",
        }
    )

    generator = ResponseGenerator(client)

    with pytest.raises(AgentExecutionError):

        generator.generate_rag_answer(
            rag_context
        )


def test_generate_rag_answer_rejects_tool_calls(
    rag_context,
):

    client = MockOllamaClient(
        response={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "example_tool",
                        "arguments": {},
                    }
                }
            ],
        }
    )

    generator = ResponseGenerator(client)

    with pytest.raises(AgentExecutionError):

        generator.generate_rag_answer(
            rag_context
        )