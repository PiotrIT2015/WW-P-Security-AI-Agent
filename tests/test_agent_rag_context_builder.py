import pytest

from authentication.auth import UserContext

from agent.rag.adapter import AgentRAGContext

from agent.rag.sources import RAGSource

from agent.rag.context_builder import (
    RAGContextBuilder,
    BuiltRAGContext,
    ContextValidationError,
    ContextSecurityError,
    ContextSizeError,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def user():
    return UserContext(
        user_id="test-user",
        username="tester",
        roles=frozenset({"analyst"}),
        groups=frozenset({"security"}),
        authenticated_at=1234567890.0,
        session_id="test-session",
    )


@pytest.fixture
def sources():
    return (
        RAGSource(
            citation_id=1,
            document_id="document-001",
            chunk_id="chunk-001",
            score=0.95,
        ),
        RAGSource(
            citation_id=2,
            document_id="document-002",
            chunk_id="chunk-002",
            score=0.87,
        ),
    )


@pytest.fixture
def retrieved_context(sources):
    return AgentRAGContext(
        query="What is network security?",
        context="Authorized security document content.",
        user_id="test-user",
        candidates_count=3,
        authorized_count=2,
        denied_count=1,
        sources=sources,
    )


@pytest.fixture
def builder():
    return RAGContextBuilder()


# ============================================================
# BASIC CONTEXT BUILDING
# ============================================================

def test_builder_returns_built_context(
    builder,
    retrieved_context,
    user,
):
    result = builder.build(retrieved_context, user)

    assert isinstance(result, BuiltRAGContext)

    assert result.query == retrieved_context.query

    assert result.context == retrieved_context.context

    assert result.user_id == user.user_id

    assert result.sources == retrieved_context.sources


def test_builder_preserves_retrieval_statistics(
    builder,
    retrieved_context,
    user,
):
    result = builder.build(retrieved_context, user)

    assert result.candidates_count == 3

    assert result.authorized_count == 2

    assert result.denied_count == 1


def test_builder_preserves_source_attribution(
    builder,
    retrieved_context,
    user,
):
    result = builder.build(retrieved_context, user)

    assert len(result.sources) == 2

    assert result.sources[0].citation_id == 1

    assert result.sources[0].document_id == "document-001"

    assert result.sources[0].chunk_id == "chunk-001"

    assert result.sources[1].citation_id == 2

    assert result.sources[1].document_id == "document-002"

    assert result.sources[1].chunk_id == "chunk-002"


# ============================================================
# SECURITY VALIDATION
# ============================================================

def test_builder_rejects_invalid_user(
    builder,
    retrieved_context,
):
    with pytest.raises(ContextSecurityError):
        builder.build(retrieved_context, None)


def test_builder_rejects_context_from_another_user(
    builder,
    retrieved_context,
):
    another_user = UserContext(
        user_id="another-user",
        username="another",
        roles=frozenset({"analyst"}),
        groups=frozenset({"security"}),
        authenticated_at=1234567890.0,
        session_id="another-session",
    )

    with pytest.raises(ContextSecurityError):
        builder.build(retrieved_context, another_user)


def test_builder_rejects_invalid_context_type(
    builder,
    user,
):
    with pytest.raises(ContextValidationError):
        builder.build(
            retrieved_context={"context": "invalid"},
            user=user,
        )


# ============================================================
# QUERY VALIDATION
# ============================================================

def test_builder_rejects_empty_query(user):
    context = AgentRAGContext(
        query=" ",
        context="Some content",
        user_id="test-user",
        candidates_count=1,
        authorized_count=1,
        denied_count=0,
        sources=(
            RAGSource(
                citation_id=1,
                document_id="document-001",
                chunk_id="chunk-001",
                score=0.90,
            ),
        ),
    )

    builder = RAGContextBuilder()

    with pytest.raises(ContextValidationError):
        builder.build(context, user)


# ============================================================
# CONTEXT SIZE
# ============================================================

def test_builder_rejects_context_exceeding_limit(
    retrieved_context,
    user,
):
    builder = RAGContextBuilder(
        max_context_chars=10
    )

    with pytest.raises(ContextSizeError):
        builder.build(retrieved_context, user)


def test_builder_strips_context_whitespace(user):
    context = AgentRAGContext(
        query="Security",
        context="  Authorized content.  ",
        user_id="test-user",
        candidates_count=1,
        authorized_count=1,
        denied_count=0,
        sources=(
            RAGSource(
                citation_id=1,
                document_id="document-001",
                chunk_id="chunk-001",
                score=0.90,
            ),
        ),
    )

    builder = RAGContextBuilder()

    result = builder.build(context, user)

    assert result.context == "Authorized content."


# ============================================================
# EMPTY CONTEXT
# ============================================================

def test_builder_accepts_empty_context_by_default(user):
    context = AgentRAGContext(
        query="Unknown topic",
        context="",
        user_id="test-user",
        candidates_count=0,
        authorized_count=0,
        denied_count=0,
        sources=(),
    )

    builder = RAGContextBuilder()

    result = builder.build(context, user)

    assert result.empty is True

    assert result.context == ""

    assert result.sources == ()


def test_builder_can_reject_empty_context(user):
    context = AgentRAGContext(
        query="Unknown topic",
        context="",
        user_id="test-user",
        candidates_count=0,
        authorized_count=0,
        denied_count=0,
        sources=(),
    )

    builder = RAGContextBuilder(
        allow_empty_context=False
    )

    with pytest.raises(ContextValidationError):
        builder.build(context, user)


# ============================================================
# SERIALIZATION
# ============================================================

def test_builder_to_dict(user, retrieved_context):
    builder = RAGContextBuilder()

    result = builder.build(retrieved_context, user)

    data = result.to_dict()

    assert data["user_id"] == "test-user"

    assert data["authorized_count"] == 2

    assert data["denied_count"] == 1

    assert data["empty"] is False

    assert data["context_chars"] == len(result.context)

    assert len(data["sources"]) == 2


# ============================================================
# STATISTICS VALIDATION
# ============================================================

def test_builder_rejects_inconsistent_statistics(user):
    context = AgentRAGContext(
        query="Security",
        context="Authorized content.",
        user_id="test-user",
        candidates_count=1,
        authorized_count=2,
        denied_count=0,
        sources=(
            RAGSource(
                citation_id=1,
                document_id="document-001",
                chunk_id="chunk-001",
                score=0.90,
            ),
            RAGSource(
                citation_id=2,
                document_id="document-002",
                chunk_id="chunk-002",
                score=0.85,
            ),
        ),
    )

    builder = RAGContextBuilder()

    with pytest.raises(ContextValidationError):
        builder.build(context, user)


def test_builder_rejects_negative_statistics(user):
    context = AgentRAGContext(
        query="Security",
        context="Authorized content.",
        user_id="test-user",
        candidates_count=1,
        authorized_count=-1,
        denied_count=0,
        sources=(),
    )

    builder = RAGContextBuilder()

    with pytest.raises(ContextValidationError):
        builder.build(context, user)

def test_builder_rejects_inconsistent_statistics(user):
    context = AgentRAGContext(
        query="Security",
        context="Authorized content.",
        user_id="test-user",
        candidates_count=1,
        authorized_count=2,
        denied_count=0,
    )

    builder = RAGContextBuilder()

    with pytest.raises(ContextValidationError):
        builder.build(context, user)


def test_builder_rejects_negative_statistics(user):
    context = AgentRAGContext(
        query="Security",
        context="Authorized content.",
        user_id="test-user",
        candidates_count=1,
        authorized_count=-1,
        denied_count=0,
    )

    builder = RAGContextBuilder()

    with pytest.raises(ContextValidationError):
        builder.build(context, user)