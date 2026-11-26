"""
tests/test_agent_hallucination_integration.py

Integration tests for HallucinationDetector and ResponseGenerator.

Stage 3.6.2.
"""

from copy import deepcopy

import pytest

from agent.exceptions import AgentExecutionError
from agent.response_generator import ResponseGenerator
from agent.rag.context_builder import BuiltRAGContext


class MockOllamaClient:

    def __init__(self, content):

        self.content = content

        self.last_messages = None

        self.call_count = 0

    def chat(self, messages, tools=None):

        self.call_count += 1

        self.last_messages = deepcopy(messages)

        return {
            "role": "assistant",
            "content": self.content,
        }


@pytest.fixture
def rag_context():

    return BuiltRAGContext(
        query="What is DNS?",
        context=(
            "DNS translates domain names into IP addresses. "
            "It allows users to access network services "
            "using readable domain names."
        ),
        user_id="demo-piotr",
        candidates_count=2,
        authorized_count=1,
        denied_count=1,
        sources=(),
    )


# ============================================================
# REPORTING MODE
# ============================================================

def test_generator_creates_grounding_report(rag_context):

    client = MockOllamaClient(
        "DNS translates domain names into IP addresses."
    )

    generator = ResponseGenerator(client)

    answer = generator.generate_rag_answer(
        rag_context
    )

    assert answer

    report = generator.last_hallucination_report

    assert report is not None

    assert report.status == "grounded"

    assert report.supported_count == 1


def test_generator_reports_unsupported_claim(rag_context):

    client = MockOllamaClient(
        "DNS uses quantum encryption."
    )

    generator = ResponseGenerator(client)

    answer = generator.generate_rag_answer(
        rag_context
    )

    assert answer

    report = generator.last_hallucination_report

    assert report is not None

    assert report.unsupported_count == 1

    assert report.is_grounded is False


def test_generator_preserves_default_non_strict_behavior(
    rag_context,
):

    client = MockOllamaClient(
        "DNS uses quantum encryption."
    )

    generator = ResponseGenerator(client)

    answer = generator.generate_rag_answer(
        rag_context
    )

    assert answer == "DNS uses quantum encryption."

    assert generator.last_hallucination_report is not None


# ============================================================
# STRICT MODE
# ============================================================

def test_strict_mode_rejects_unsupported_claim(rag_context):

    client = MockOllamaClient(
        "DNS uses quantum encryption."
    )

    generator = ResponseGenerator(
        ollama_client=client,
        strict_hallucination_check=True,
    )

    with pytest.raises(AgentExecutionError):

        generator.generate_rag_answer(
            rag_context
        )


def test_strict_mode_accepts_supported_claim(rag_context):

    client = MockOllamaClient(
        "DNS translates domain names into IP addresses."
    )

    generator = ResponseGenerator(
        ollama_client=client,
        strict_hallucination_check=True,
    )

    answer = generator.generate_rag_answer(
        rag_context
    )

    assert answer

    assert generator.last_hallucination_report.is_grounded


def test_strict_mode_rejects_claim_without_context():

    context = BuiltRAGContext(
        query="What is DNS?",
        context="",
        user_id="demo-piotr",
        candidates_count=0,
        authorized_count=0,
        denied_count=0,
        sources=(),
    )

    client = MockOllamaClient(
        "DNS translates domain names into IP addresses."
    )

    generator = ResponseGenerator(
        ollama_client=client,
        strict_hallucination_check=True,
    )

    with pytest.raises(AgentExecutionError):

        generator.generate_rag_answer(
            context
        )


# ============================================================
# SECURITY AND RELIABILITY
# ============================================================

def test_detector_receives_only_authorized_context(rag_context):

    client = MockOllamaClient(
        "DNS translates domain names into IP addresses."
    )

    generator = ResponseGenerator(client)

    generator.generate_rag_answer(
        rag_context
    )

    report = generator.last_hallucination_report

    assert report is not None

    assert "DNS" in report.claims[0].claim

    assert "unauthorized_document_content" not in (
        str(report.to_dict())
    )


def test_ollama_is_called_only_once(rag_context):

    client = MockOllamaClient(
        "DNS translates domain names into IP addresses."
    )

    generator = ResponseGenerator(client)

    generator.generate_rag_answer(
        rag_context
    )

    assert client.call_count == 1


def test_original_context_is_not_modified(rag_context):

    original_context = rag_context.context

    original_query = rag_context.query

    client = MockOllamaClient(
        "DNS translates domain names into IP addresses."
    )

    generator = ResponseGenerator(client)

    generator.generate_rag_answer(
        rag_context
    )

    assert rag_context.context == original_context

    assert rag_context.query == original_query