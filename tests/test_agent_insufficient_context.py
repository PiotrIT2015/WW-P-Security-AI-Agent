"""
tests/test_agent_insufficient_context.py

Insufficient context handling tests
for WW-P Security AI Agent.

Stage 3.6.3:
- Empty authorized context.
- Missing reference documents.
- Partial context.
- Questions outside the available knowledge.
- Explicit uncertainty.
- Hallucination assessment.
- Strict grounding policy.
- Source attribution availability.

Compatible with Python 3.10+.
"""

from copy import deepcopy

import pytest

from agent.exceptions import AgentExecutionError
from agent.response_generator import ResponseGenerator
from agent.rag.context_builder import BuiltRAGContext


# ============================================================
# MOCK OLLAMA
# ============================================================

class MockOllamaClient:

    def __init__(self, response=None):

        self.response = response

        self.last_messages = None
        self.last_tools = None

        self.call_count = 0

    def chat(self, messages, tools=None):

        self.call_count += 1

        self.last_messages = deepcopy(messages)
        self.last_tools = tools

        return deepcopy(self.response)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def empty_context():

    return BuiltRAGContext(
        query="Explain DNSSEC.",
        context="",
        user_id="demo-piotr",
        candidates_count=0,
        authorized_count=0,
        denied_count=0,
        sources=(),
    )


@pytest.fixture
def partial_context():

    return BuiltRAGContext(
        query="Explain DNS and DNSSEC.",
        context=(
            "DNS translates domain names into IP addresses. "
            "It allows users to access network services "
            "using domain names."
        ),
        user_id="demo-piotr",
        candidates_count=1,
        authorized_count=1,
        denied_count=0,
        sources=(),
    )


@pytest.fixture
def uncertainty_response():

    return (
        "The available reference material is insufficient "
        "to determine the DNSSEC configuration."
    )


@pytest.fixture
def client(uncertainty_response):

    return MockOllamaClient(
        response={
            "role": "assistant",
            "content": uncertainty_response,
        }
    )


@pytest.fixture
def generator(client):

    return ResponseGenerator(
        ollama_client=client,
    )


# ============================================================
# IC-01: EMPTY CONTEXT
# ============================================================

def test_empty_context_is_explicitly_marked(
    generator,
    empty_context,
):

    messages = generator.prepare_rag_messages(
        empty_context
    )

    combined = "\n".join(
        message["content"]
        for message in messages
    )

    assert "No authorized reference documents" in combined


def test_empty_context_preserves_original_question(
    generator,
    empty_context,
):

    messages = generator.prepare_rag_messages(
        empty_context
    )

    assert messages[-1]["role"] == "user"

    assert messages[-1]["content"] == (
        empty_context.query
    )


def test_empty_context_produces_insufficient_context_report(
    generator,
    empty_context,
):

    result = generator.generate_rag_answer(
        empty_context
    )

    report = generator.last_hallucination_report

    assert result.strip()

    assert report is not None

    assert report.status == "insufficient_context"

    assert report.unverified_count > 0

    assert report.coverage == 0.0


def test_empty_context_accepts_explicit_uncertainty(
    generator,
    empty_context,
):

    result = generator.generate_rag_answer(
        empty_context
    )

    assert "insufficient" in result.lower()

    assert "Retrieved sources:" not in result


# ============================================================
# IC-02: PARTIAL CONTEXT
# ============================================================

def test_partial_context_is_passed_to_ollama(
    generator,
    partial_context,
):

    generator.generate_rag_answer(
        partial_context
    )

    combined = "\n".join(
        message["content"]
        for message in generator.ollama_client.last_messages
    )

    assert "DNS translates domain names" in combined

    assert partial_context.query in combined


def test_partial_context_does_not_claim_full_coverage(
    partial_context,
):

    response = (
        "DNS translates domain names into IP addresses. "
        "DNSSEC uses cryptographic signatures to validate DNS data."
    )

    client = MockOllamaClient(
        response={
            "role": "assistant",
            "content": response,
        }
    )

    generator = ResponseGenerator(client)

    generator.generate_rag_answer(
        partial_context
    )

    report = generator.last_hallucination_report

    assert report is not None

    assert report.supported_count >= 1

    assert report.unsupported_count >= 1

    assert report.status == "contains_unsupported_claims"


# ============================================================
# IC-03: QUESTIONS OUTSIDE AVAILABLE KNOWLEDGE
# ============================================================

def test_out_of_scope_claim_is_detected(
    partial_context,
):

    response = (
        "DNS translates domain names into IP addresses. "
        "The production DNS server uses a 4096-bit RSA key."
    )

    client = MockOllamaClient(
        response={
            "role": "assistant",
            "content": response,
        }
    )

    generator = ResponseGenerator(client)

    generator.generate_rag_answer(
        partial_context
    )

    report = generator.last_hallucination_report

    assert report.unsupported_count >= 1

    assert report.status == "contains_unsupported_claims"


def test_unknown_configuration_is_not_automatically_verified(
    partial_context,
):

    response = (
        "The firewall uses port 8443 for DNSSEC traffic."
    )

    client = MockOllamaClient(
        response={
            "role": "assistant",
            "content": response,
        }
    )

    generator = ResponseGenerator(client)

    generator.generate_rag_answer(
        partial_context
    )

    report = generator.last_hallucination_report

    assert report.unsupported_count >= 1

    assert report.coverage == 0.0


# ============================================================
# IC-04: STRICT GROUNDING POLICY
# ============================================================

def test_strict_mode_rejects_claims_without_context(
    empty_context,
):

    client = MockOllamaClient(
        response={
            "role": "assistant",
            "content": (
                "DNSSEC uses RSA keys to validate DNS responses."
            ),
        }
    )

    generator = ResponseGenerator(
        ollama_client=client,
        strict_hallucination_check=True,
    )

    with pytest.raises(AgentExecutionError):

        generator.generate_rag_answer(
            empty_context
        )

    assert generator.last_hallucination_report is not None

    assert (
        generator.last_hallucination_report.status
        == "insufficient_context"
    )


def test_strict_mode_rejects_partially_unsupported_response(
    partial_context,
):

    client = MockOllamaClient(
        response={
            "role": "assistant",
            "content": (
                "DNS translates domain names into IP addresses. "
                "The server uses a custom 4096-bit encryption key."
            ),
        }
    )

    generator = ResponseGenerator(
        ollama_client=client,
        strict_hallucination_check=True,
    )

    with pytest.raises(AgentExecutionError):

        generator.generate_rag_answer(
            partial_context
        )

    assert generator.last_hallucination_report is not None

    assert (
        generator.last_hallucination_report.unsupported_count
        > 0
    )


# ============================================================
# IC-05: INPUT AND EXECUTION INTEGRITY
# ============================================================

def test_insufficient_context_does_not_modify_original_context(
    generator,
    empty_context,
):

    original_query = empty_context.query

    original_content = empty_context.context

    original_sources = empty_context.sources

    generator.generate_rag_answer(
        empty_context
    )

    assert empty_context.query == original_query

    assert empty_context.context == original_content

    assert empty_context.sources == original_sources


def test_insufficient_context_calls_ollama_once(
    generator,
    empty_context,
):

    generator.generate_rag_answer(
        empty_context
    )

    assert generator.ollama_client.call_count == 1


def test_insufficient_context_does_not_enable_tools(
    generator,
    empty_context,
):

    generator.generate_rag_answer(
        empty_context
    )

    assert generator.ollama_client.last_tools is None


def test_insufficient_context_does_not_invent_sources(
    generator,
    empty_context,
):

    result = generator.generate_rag_answer(
        empty_context
    )

    assert "Retrieved sources:" not in result

    assert empty_context.sources == ()