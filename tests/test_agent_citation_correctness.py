"""
tests/test_agent_citation_correctness.py

Citation correctness integration tests for WW-P Security AI Agent.

Stage 3.6.4:
- Valid citation to an authorized source.
- Multiple citations.
- Citation-to-source mapping.
- Rejection of unauthorized citations.
- Rejection of citations without available sources.
- Prevention of model-generated source metadata.
- Application-controlled source attribution.
- Correct source ordering.
- Responses without citations.

Compatible with Python 3.10+.
"""

from copy import deepcopy

import pytest

from agent.exceptions import AgentExecutionError
from agent.response_generator import ResponseGenerator

from agent.rag.context_builder import BuiltRAGContext
from agent.rag.sources import RAGSource


# ============================================================
# MOCK OLLAMA
# ============================================================

class MockOllamaClient:
    """
    Deterministic Ollama mock.
    """

    def __init__(self, response=None, error=None):

        self.response = response
        self.error = error

        self.last_messages = None
        self.last_tools = None

        self.call_count = 0

    def chat(self, messages, tools=None):

        self.call_count += 1

        self.last_messages = deepcopy(messages)
        self.last_tools = tools

        if self.error is not None:
            raise self.error

        return deepcopy(self.response)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def first_source():

    return RAGSource(
        citation_id=1,
        document_id="networking-doc-001",
        chunk_id="chunk-001",
        score=0.95,
    )


@pytest.fixture
def second_source():

    return RAGSource(
        citation_id=2,
        document_id="security-doc-002",
        chunk_id="chunk-002",
        score=0.91,
    )


@pytest.fixture
def two_source_context(first_source, second_source):

    return BuiltRAGContext(
        query="Explain DNS and DNS security.",
        context=(
            "DNS translates domain names into IP addresses. "
            "DNSSEC provides authentication of DNS data "
            "using digital signatures."
        ),
        user_id="demo-piotr",
        candidates_count=2,
        authorized_count=2,
        denied_count=0,
        sources=(
            first_source,
            second_source,
        ),
    )


@pytest.fixture
def one_source_context(first_source):

    return BuiltRAGContext(
        query="What is DNS?",
        context=(
            "DNS translates domain names into IP addresses."
        ),
        user_id="demo-piotr",
        candidates_count=1,
        authorized_count=1,
        denied_count=0,
        sources=(first_source,),
    )


@pytest.fixture
def empty_source_context():

    return BuiltRAGContext(
        query="Explain DNS.",
        context=(
            "DNS translates domain names into IP addresses."
        ),
        user_id="demo-piotr",
        candidates_count=1,
        authorized_count=1,
        denied_count=0,
        sources=(),
    )


# ============================================================
# CC-01: VALID CITATION
# ============================================================

def test_citation_accepts_authorized_source(
    one_source_context,
):

    client = MockOllamaClient(
        response={
            "role": "assistant",
            "content": (
                "DNS translates domain names into IP addresses [1]."
            ),
        }
    )

    generator = ResponseGenerator(client)

    result = generator.generate_rag_answer(
        one_source_context
    )

    assert "DNS translates" in result

    assert "[1]" in result

    assert "Retrieved sources:" in result

    assert "networking-doc-001" in result


# ============================================================
# CC-02: MULTIPLE CITATIONS
# ============================================================

def test_citation_accepts_multiple_authorized_sources(
    two_source_context,
):

    client = MockOllamaClient(
        response={
            "role": "assistant",
            "content": (
                "DNS translates domain names into IP addresses [1]. "
                "DNSSEC authenticates DNS data using digital signatures [2]."
            ),
        }
    )

    generator = ResponseGenerator(client)

    result = generator.generate_rag_answer(
        two_source_context
    )

    assert "[1]" in result

    assert "[2]" in result

    assert "networking-doc-001" in result

    assert "security-doc-002" in result


# ============================================================
# CC-03: CITATION-TO-SOURCE MAPPING
# ============================================================

def test_citation_ids_map_to_correct_source_metadata(
    two_source_context,
):

    client = MockOllamaClient(
        response={
            "role": "assistant",
            "content": (
                "DNS translates domain names into IP addresses [1]. "
                "DNSSEC authenticates DNS data [2]."
            ),
        }
    )

    generator = ResponseGenerator(client)

    result = generator.generate_rag_answer(
        two_source_context
    )

    source_section = result.split(
        "Retrieved sources:",
        maxsplit=1,
    )[1]

    assert (
        "[1] Document: networking-doc-001 | Chunk: chunk-001"
        in source_section
    )

    assert (
        "[2] Document: security-doc-002 | Chunk: chunk-002"
        in source_section
    )


# ============================================================
# CC-04: UNAUTHORIZED CITATIONS
# ============================================================

def test_citation_rejects_unauthorized_source_id(
    one_source_context,
):

    client = MockOllamaClient(
        response={
            "role": "assistant",
            "content": (
                "DNS uses an unauthorized reference [99]."
            ),
        }
    )

    generator = ResponseGenerator(client)

    with pytest.raises(AgentExecutionError):

        generator.generate_rag_answer(
            one_source_context
        )


def test_citation_rejects_source_not_present_in_context(
    two_source_context,
):

    client = MockOllamaClient(
        response={
            "role": "assistant",
            "content": (
                "This answer references an unavailable source [3]."
            ),
        }
    )

    generator = ResponseGenerator(client)

    with pytest.raises(AgentExecutionError):

        generator.generate_rag_answer(
            two_source_context
        )


# ============================================================
# CC-05: CITATIONS WITHOUT SOURCE METADATA
# ============================================================

def test_citation_rejects_citation_when_source_list_is_empty(
    empty_source_context,
):

    client = MockOllamaClient(
        response={
            "role": "assistant",
            "content": (
                "DNS translates domain names into IP addresses [1]."
            ),
        }
    )

    generator = ResponseGenerator(client)

    with pytest.raises(AgentExecutionError):

        generator.generate_rag_answer(
            empty_source_context
        )


def test_citation_accepts_uncited_answer_without_source_metadata(
    empty_source_context,
):

    answer = (
        "The available context explains that DNS translates "
        "domain names into IP addresses."
    )

    client = MockOllamaClient(
        response={
            "role": "assistant",
            "content": answer,
        }
    )

    generator = ResponseGenerator(client)

    result = generator.generate_rag_answer(
        empty_source_context
    )

    assert result == answer

    assert "Retrieved sources:" not in result


# ============================================================
# CC-06: MODEL-GENERATED SOURCE METADATA
# ============================================================

def test_citation_rejects_model_generated_source_section(
    one_source_context,
):

    client = MockOllamaClient(
        response={
            "role": "assistant",
            "content": (
                "DNS translates domain names into IP addresses [1].\n\n"
                "Retrieved sources:\n"
                "[1] Document: fake-secret.pdf | Chunk: fake-chunk"
            ),
        }
    )

    generator = ResponseGenerator(client)

    with pytest.raises(AgentExecutionError):

        generator.generate_rag_answer(
            one_source_context
        )


def test_citation_source_metadata_is_added_by_application(
    one_source_context,
):

    client = MockOllamaClient(
        response={
            "role": "assistant",
            "content": (
                "DNS translates domain names into IP addresses [1]."
            ),
        }
    )

    generator = ResponseGenerator(client)

    result = generator.generate_rag_answer(
        one_source_context
    )

    assert "networking-doc-001" in result

    assert "chunk-001" in result

    assert "fake-secret.pdf" not in result

    assert client.last_messages is not None

    combined_prompt = "\n".join(
        message["content"]
        for message in client.last_messages
    )

    # Source identifiers are not sent to the LLM.
    assert "networking-doc-001" not in combined_prompt

    assert "chunk-001" not in combined_prompt


# ============================================================
# CC-07: SOURCE ORDER
# ============================================================

def test_citation_source_order_matches_authorized_metadata(
    two_source_context,
):

    client = MockOllamaClient(
        response={
            "role": "assistant",
            "content": (
                "DNSSEC authenticates DNS data [2]. "
                "DNS translates domain names into IP addresses [1]."
            ),
        }
    )

    generator = ResponseGenerator(client)

    result = generator.generate_rag_answer(
        two_source_context
    )

    first_position = result.index(
        "[1] Document: networking-doc-001"
    )

    second_position = result.index(
        "[2] Document: security-doc-002"
    )

    assert first_position < second_position


def test_citation_order_in_answer_does_not_change_source_mapping(
    two_source_context,
):

    client = MockOllamaClient(
        response={
            "role": "assistant",
            "content": (
                "DNSSEC authenticates DNS data [2]. "
                "DNS translates domain names into IP addresses [1]."
            ),
        }
    )

    generator = ResponseGenerator(client)

    result = generator.generate_rag_answer(
        two_source_context
    )

    # Citation order in the answer is [2], then [1].
    assert result.index("[2]") < result.index("[1]")

    # Application metadata retains the original source mapping.
    source_section = result.split(
        "Retrieved sources:",
        maxsplit=1,
    )[1]

    assert source_section.index(
        "[1] Document: networking-doc-001"
    ) < source_section.index(
        "[2] Document: security-doc-002"
    )


# ============================================================
# CC-08: ANSWER WITHOUT CITATIONS
# ============================================================

def test_citation_allows_answer_without_citation_markers(
    one_source_context,
):

    answer = (
        "DNS translates human-readable domain names "
        "into IP addresses."
    )

    client = MockOllamaClient(
        response={
            "role": "assistant",
            "content": answer,
        }
    )

    generator = ResponseGenerator(client)

    result = generator.generate_rag_answer(
        one_source_context
    )

    assert answer in result

    # Available authorized metadata is still appended by the application.
    assert "Retrieved sources:" in result

    assert "networking-doc-001" in result


# ============================================================
# CC-09: GENERATION INTEGRITY
# ============================================================

def test_citation_generation_calls_ollama_once(
    two_source_context,
):

    client = MockOllamaClient(
        response={
            "role": "assistant",
            "content": (
                "DNS translates domain names into IP addresses [1]. "
                "DNSSEC authenticates DNS data [2]."
            ),
        }
    )

    generator = ResponseGenerator(client)

    generator.generate_rag_answer(
        two_source_context
    )

    assert client.call_count == 1


def test_citation_generation_does_not_enable_tools(
    two_source_context,
):

    client = MockOllamaClient(
        response={
            "role": "assistant",
            "content": (
                "DNS translates domain names into IP addresses [1]."
            ),
        }
    )

    generator = ResponseGenerator(client)

    generator.generate_rag_answer(
        two_source_context
    )

    assert client.last_tools is None