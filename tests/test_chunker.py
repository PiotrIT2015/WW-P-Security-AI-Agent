from pathlib import Path

from documents.extractors import ExtractedDocument
from documents.chunker import DocumentChunker


def test_chunker():

    document = ExtractedDocument(

        text="""
        Secure RAG System

        Retrieval Augmented Generation
        allows a language model to use
        external knowledge.

        Security is important because
        documents may contain untrusted
        content.

        The system should validate,
        sanitize and classify documents
        before they reach the LLM.
        """,

        pages=1,

        source=Path(
            "example.txt"
        ),

        metadata={
            "type": "txt",
        },
    )

    chunker = DocumentChunker(
        chunk_size=250,
        overlap=50,
        min_chunk_size=50,
    )

    chunks = chunker.chunk(
        document
    )

    assert chunks

    assert chunks[0].document_id

    assert chunks[0].chunk_id

    for chunk in chunks:

        print("=" * 60)

        print(
            "ID:",
            chunk.chunk_id
        )

        print(
            "Document:",
            chunk.document_id
        )

        print(
            "Position:",
            chunk.position
        )

        print(
            chunk.text
        )


if __name__ == "__main__":
    test_chunker()
