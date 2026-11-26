import pytest

from agent.rag.sources import (
    RAGSource,
    SourceExtractor,
    SourceAttributor,
    SourceValidationError,
    SourceAttributionError,
)

from retrieval.retriever import RetrievedChunk


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def chunks():

    return [
        RetrievedChunk(
            chunk_id="chunk-001",
            document_id="networking.pdf",
            text="DNS resolves domain names.",
            score=0.95,
            metadata={},
        ),
        RetrievedChunk(
            chunk_id="chunk-002",
            document_id="security.pdf",
            text="DNS queries use port 53.",
            score=0.89,
            metadata={},
        ),
    ]


@pytest.fixture
def sources():

    return (
        RAGSource(
            citation_id=1,
            document_id="networking.pdf",
            chunk_id="chunk-001",
            score=0.95,
        ),
        RAGSource(
            citation_id=2,
            document_id="security.pdf",
            chunk_id="chunk-002",
            score=0.89,
        ),
    )


# ============================================================
# RAG SOURCE MODEL
# ============================================================

def test_accepts_valid_source():

    source = RAGSource(
        citation_id=1,
        document_id="networking.pdf",
        chunk_id="chunk-001",
        score=0.95,
    )

    assert source.citation_id == 1
    assert source.document_id == "networking.pdf"


def test_rejects_invalid_citation_id():

    with pytest.raises(SourceValidationError):

        RAGSource(
            citation_id=0,
            document_id="networking.pdf",
            chunk_id="chunk-001",
            score=0.95,
        )


def test_rejects_empty_document_id():

    with pytest.raises(SourceValidationError):

        RAGSource(
            citation_id=1,
            document_id="",
            chunk_id="chunk-001",
            score=0.95,
        )


def test_rejects_empty_chunk_id():

    with pytest.raises(SourceValidationError):

        RAGSource(
            citation_id=1,
            document_id="networking.pdf",
            chunk_id="",
            score=0.95,
        )


def test_rejects_invalid_score():

    with pytest.raises(SourceValidationError):

        RAGSource(
            citation_id=1,
            document_id="networking.pdf",
            chunk_id="chunk-001",
            score=float("nan"),
        )


def test_source_to_dict():

    source = RAGSource(
        citation_id=1,
        document_id="networking.pdf",
        chunk_id="chunk-001",
        score=0.95,
    )

    result = source.to_dict()

    assert result["citation_id"] == 1
    assert result["document_id"] == "networking.pdf"
    assert result["chunk_id"] == "chunk-001"


# ============================================================
# SOURCE EXTRACTION
# ============================================================

def test_extracts_sources_from_retrieved_chunks(chunks):

    sources = SourceExtractor.extract(chunks)

    assert len(sources) == 2

    assert sources[0].citation_id == 1
    assert sources[0].document_id == "networking.pdf"

    assert sources[1].citation_id == 2
    assert sources[1].document_id == "security.pdf"


def test_assigns_sequential_citation_ids(chunks):

    sources = SourceExtractor.extract(chunks)

    assert [
        source.citation_id
        for source in sources
    ] == [1, 2]


def test_extracts_empty_source_list():

    sources = SourceExtractor.extract([])

    assert sources == ()


def test_rejects_invalid_chunk():

    with pytest.raises(SourceValidationError):

        SourceExtractor.extract(
            ["not-a-retrieved-chunk"]
        )


def test_rejects_duplicate_sources(chunks):

    duplicated = [
        chunks[0],
        chunks[0],
    ]

    with pytest.raises(SourceValidationError):

        SourceExtractor.extract(duplicated)


def test_rejects_invalid_input():

    with pytest.raises(SourceValidationError):

        SourceExtractor.extract("invalid")


# ============================================================
# SOURCE FORMATTING
# ============================================================

def test_formats_sources(sources):

    result = SourceAttributor.format_sources(
        sources
    )

    assert "Retrieved sources:" in result
    assert "[1]" in result
    assert "[2]" in result

    assert "networking.pdf" in result
    assert "security.pdf" in result


def test_formats_empty_sources():

    result = SourceAttributor.format_sources(())

    assert result == ""


def test_rejects_invalid_source_during_formatting():

    with pytest.raises(SourceValidationError):

        SourceAttributor.format_sources(
            ("invalid",)
        )


# ============================================================
# SOURCE APPENDING
# ============================================================

def test_appends_sources_to_answer(sources):

    answer = "DNS resolves domain names."

    result = SourceAttributor.append_sources(
        answer,
        sources,
    )

    assert answer in result
    assert "Retrieved sources:" in result
    assert "networking.pdf" in result


def test_returns_answer_without_sources():

    answer = "No authorized documents were retrieved."

    result = SourceAttributor.append_sources(
        answer,
        (),
    )

    assert result == answer


def test_rejects_empty_answer(sources):

    with pytest.raises(SourceAttributionError):

        SourceAttributor.append_sources(
            "",
            sources,
        )