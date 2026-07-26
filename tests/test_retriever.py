from authentication.auth import (
    AuthenticationManager,
)

from authentication.permissions import (
    DocumentACL,
    PermissionManager,
)

from retrieval.retriever import (
    SecureRetriever,
)


class FakeEmbedder:
    """
    Test embedder.
    """

    def embed_query(
        self,
        text: str,
    ) -> list[float]:

        return [
            0.1,
            0.2,
            0.3,
        ]


class FakeVectorStore:
    """
    Test vector store.

    In production this is replaced by EncryptedFAISS.
    """

    def search(
        self,
        query_vector,
        top_k,
    ):

        return [

            {
                "chunk_id": "chunk-001",

                "document_id": "public-doc",

                "text": (
                    "Public information."
                ),

                "score": 0.99,

                "metadata": {
                    "classification": "public",
                },
            },

            {
                "chunk_id": "chunk-002",

                "document_id": "secret-doc",

                "text": (
                    "SECRET INFORMATION."
                ),

                "score": 0.98,

                "metadata": {
                    "classification": "secret",
                },
            },

            {
                "chunk_id": "chunk-003",

                "document_id": "internal-doc",

                "text": (
                    "Internal information."
                ),

                "score": 0.97,

                "metadata": {
                    "classification": "internal",
                },
            },

        ]


def main():

    # ---------------------------------------------------------
    # Authentication
    # ---------------------------------------------------------

    auth = AuthenticationManager()

    auth.create_user(

        "reader",

        "ReaderPassword123!",

        roles={
            "reader",
        },

    )

    user, _ = auth.authenticate(

        "reader",

        "ReaderPassword123!",

    )

    # ---------------------------------------------------------
    # Permissions
    # ---------------------------------------------------------

    permissions = PermissionManager()

    permissions.add_policy(

        DocumentACL(

            document_id="public-doc",

            classification="public",

        )
    )

    permissions.add_policy(

        DocumentACL(

            document_id="secret-doc",

            classification="secret",

        )
    )

    permissions.add_policy(

        DocumentACL(

            document_id="internal-doc",

            classification="internal",

        )
    )

    # ---------------------------------------------------------
    # Retriever
    # ---------------------------------------------------------

    retriever = SecureRetriever(

        embedder=FakeEmbedder(),

        vector_store=FakeVectorStore(),

        permission_manager=permissions,

        default_top_k=5,

    )

    # ---------------------------------------------------------
    # Query
    # ---------------------------------------------------------

    result = retriever.retrieve(

        "What information is available?",

        user,

        top_k=5,

    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print(
        "Candidates:",
        result.candidates_count,
    )

    print(
        "Authorized:",
        result.authorized_count,
    )

    print(
        "Denied:",
        result.denied_count,
    )

    print(
        "\nContext:"
    )

    print(
        result.context
    )

    # ---------------------------------------------------------
    # Security assertions
    # ---------------------------------------------------------

    assert (
        result.authorized_count == 2
    )

    assert (
        result.denied_count == 1
    )

    assert (
        "SECRET INFORMATION."
        not in result.context
    )

    print(
        "\nSecure retrieval test: OK"
    )


if __name__ == "__main__":
    main()