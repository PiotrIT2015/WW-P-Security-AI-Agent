from pathlib import Path

from documents.extractors import ExtractedDocument
from documents.chunker import DocumentChunker
from embeddings.embedder import Embedder


def main():

    document = ExtractedDocument(

        text="""
        Secure RAG is a retrieval augmented
        generation system designed with
        security controls.

        Documents are validated before
        they are processed.

        Embeddings transform text into
        numerical vectors used for
        semantic retrieval.
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
        chunk_size=300,
        overlap=50,
        min_chunk_size=50,
    )

    chunks = chunker.chunk(
        document
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    embedder = Embedder(
        model_name=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        batch_size=8,
        normalize=True,
    )

    result = embedder.embed_chunks(
        chunks
    )

    print(
        "Model:",
        result.model_name
    )

    print(
        "Dimension:",
        result.dimension
    )

    print(
        "Shape:",
        result.vectors.shape
    )

    print(
        "Chunk IDs:",
        result.chunk_ids
    )


if __name__ == "__main__":
    main()