import pytest

from unittest.mock import Mock

from authentication.auth import UserContext

from retrieval.retriever import (
    RetrievalResult,
    RetrievalError,
    RetrievedChunk,
)

from agent.rag.adapter import (
    RAGAdapter,
    AgentRAGContext,
    RAGAdapterError,
    RAGConfigurationError,
    RAGContextError,
    RAGSecurityError,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def user():
    return UserContext(
        user_id="test-user",
        username="test",
        roles=frozenset({"analyst"}),
        groups=frozenset({"security"}),
        authenticated_at=1000.0,
        session_id="test-session",
    )


@pytest.fixture
def authorized_chunks():
    return (
        RetrievedChunk(
            chunk_id="chunk-001",
            document_id="document-001",
            text="Network monitoring detects suspicious activity.",
            score=0.95,
            metadata={"source": "security-guide.pdf"},
        ),
        RetrievedChunk(
            chunk_id="chunk-002",
            document_id="document-002",
            text="Security logs provide information about incidents.",
            score=0.87,
            metadata={"source": "incident-response.pdf"},
        ),
    )


@pytest.fixture
def retrieval_result(authorized_chunks):
    return RetrievalResult(
        query="security monitoring",
        chunks=authorized_chunks,
        candidates_count=5,
        authorized_count=2,
        denied_count=3,
        user_id="test-user",
    )


@pytest.fixture
def retriever(retrieval_result):
    mock = Mock()
    mock.retrieve.return_value = retrieval_result
    return mock


@pytest.fixture
def query_processor():
    return Mock(side_effect=lambda query: query)


@pytest.fixture
def adapter(retriever, query_processor):
    return RAGAdapter(
        retriever=retriever,
        query_processor=query_processor,
        default_top_k=5,
    )


# ============================================================
# INITIALIZATION
# ============================================================

def test_adapter_initialization(adapter):
    assert adapter.default_top_k == 5


def test_reject_missing_retriever(query_processor):
    with pytest.raises(RAGConfigurationError):
        RAGAdapter(
            retriever=None,
            query_processor=query_processor,
        )


def test_reject_invalid_query_processor(retriever):
    with pytest.raises(RAGConfigurationError):
        RAGAdapter(
            retriever=retriever,
            query_processor=None,
        )


def test_reject_invalid_top_k(retriever, query_processor):
    with pytest.raises(RAGConfigurationError):
        RAGAdapter(
            retriever=retriever,
            query_processor=query_processor,
            default_top_k=0,
        )


# ============================================================
# QUERY VALIDATION
# ============================================================

def test_reject_empty_query(adapter, user):
    with pytest.raises(RAGAdapterError):
        adapter.retrieve(
            "",
            user,
        )


def test_reject_invalid_query_type(adapter, user):
    with pytest.raises(RAGAdapterError):
        adapter.retrieve(
            123,
            user,
        )


def test_reject_invalid_user(adapter):
    with pytest.raises(RAGSecurityError):
        adapter.retrieve(
            "security",
            None,
        )


# ============================================================
# SECURITY PIPELINE
# ============================================================

def test_query_processor_is_called(
    adapter,
    query_processor,
    user,
):
    adapter.retrieve(
        "security monitoring",
        user,
    )

    query_processor.assert_called_once_with(
        "security monitoring"
    )


def test_security_pipeline_rejection(
    retriever,
    user,
):
    processor = Mock(
        side_effect=ValueError("Blocked prompt")
    )

    adapter = RAGAdapter(
        retriever=retriever,
        query_processor=processor,
    )

    with pytest.raises(RAGSecurityError):
        adapter.retrieve(
            "malicious query",
            user,
        )

    retriever.retrieve.assert_not_called()


def test_reject_empty_processed_query(
    retriever,
    user,
):
    adapter = RAGAdapter(
        retriever=retriever,
        query_processor=lambda query: "",
    )

    with pytest.raises(RAGSecurityError):
        adapter.retrieve(
            "security",
            user,
        )

    retriever.retrieve.assert_not_called()


# ============================================================
# RETRIEVAL
# ============================================================

def test_retriever_receives_authenticated_user(
    adapter,
    retriever,
    user,
):
    adapter.retrieve(
        "security monitoring",
        user,
    )

    args = retriever.retrieve.call_args

    assert args.args[1] is user


def test_retriever_receives_top_k(
    adapter,
    retriever,
    user,
):
    adapter.retrieve(
        "security",
        user,
        top_k=3,
    )

    assert retriever.retrieve.call_args.kwargs[
        "top_k"
    ] == 3


def test_reject_invalid_top_k_during_retrieval(
    adapter,
    retriever,
    user,
):
    with pytest.raises(RAGAdapterError):
        adapter.retrieve(
            "security",
            user,
            top_k=-1,
        )

    retriever.retrieve.assert_not_called()


def test_retrieval_error_is_wrapped(
    retriever,
    user,
):
    retriever.retrieve.side_effect = RetrievalError(
        "Vector search failed."
    )

    adapter = RAGAdapter(
        retriever=retriever,
        query_processor=lambda query: query,
    )

    with pytest.raises(RAGContextError):
        adapter.retrieve(
            "security",
            user,
        )


def test_reject_user_identity_mismatch(
    retriever,
    user,
):
    retriever.retrieve.return_value = RetrievalResult(
        query="security",
        chunks=(),
        candidates_count=1,
        authorized_count=0,
        denied_count=1,
        user_id="another-user",
    )

    adapter = RAGAdapter(
        retriever=retriever,
        query_processor=lambda query: query,
    )

    with pytest.raises(RAGSecurityError):
        adapter.retrieve(
            "security",
            user,
        )


# ============================================================
# RESULT
# ============================================================

def test_empty_retrieval_returns_empty_context(
    retriever,
    user,
):
    retriever.retrieve.return_value = RetrievalResult(
        query="unknown",
        chunks=(),
        candidates_count=0,
        authorized_count=0,
        denied_count=0,
        user_id="test-user",
    )

    adapter = RAGAdapter(
        retriever=retriever,
        query_processor=lambda query: query,
    )

    result = adapter.retrieve(
        "unknown",
        user,
    )

    assert isinstance(
        result,
        AgentRAGContext,
    )

    assert result.empty is True
    assert result.context == ""
    assert result.authorized_count == 0
    assert result.sources == ()


def test_adapter_returns_statistics(
    retriever,
    user,
):
    chunks = tuple(
        RetrievedChunk(
            chunk_id="chunk-{0:03d}".format(index),
            document_id="document-{0:03d}".format(index),
            text="Authorized security content {0}.".format(index),
            score=0.90 - index * 0.05,
            metadata={},
        )
        for index in range(1, 5)
    )

    retriever.retrieve.return_value = RetrievalResult(
        query="security",
        chunks=chunks,
        candidates_count=10,
        authorized_count=4,
        denied_count=6,
        user_id="test-user",
    )

    adapter = RAGAdapter(
        retriever=retriever,
        query_processor=lambda query: query,
    )

    result = adapter.retrieve(
        "security",
        user,
    )

    assert result.candidates_count == 10
    assert result.authorized_count == 4
    assert result.denied_count == 6
    assert result.retrieved_chunks == 4
    assert len(result.sources) == 4