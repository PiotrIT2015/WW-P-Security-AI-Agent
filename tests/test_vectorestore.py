from pathlib import Path
import tempfile

import numpy as np

from documents.chunker import DocumentChunk
from vectorstore.encrypted_faiss import (
    EncryptedFAISS,
    VectorStoreSecurityError,
)


def create_chunks():

    return [

        DocumentChunk(

            chunk_id="chunk-001",

            document_id="document-001",

            text="Python is a programming language.",

            position=0,

            source="example.txt",

            metadata={
                "classification": "internal",
                "owner": "security",
            },
        ),

        DocumentChunk(

            chunk_id="chunk-002",

            document_id="document-001",

            text="Secure RAG uses retrieval.",

            position=1,

            source="example.txt",

            metadata={
                "classification": "internal",
                "owner": "security",
            },
        ),

        DocumentChunk(

            chunk_id="chunk-003",

            document_id="document-002",

            text="FAISS provides vector similarity search.",

            position=0,

            source="faiss.txt",

            metadata={
                "classification": "public",
                "owner": "system",
            },
        ),
    ]


def main():

    dimension = 384

    rng = np.random.default_rng(
        seed=42
    )

    vectors = rng.normal(
        size=(
            3,
            dimension,
        )
    ).astype(
        np.float32
    )

    # Normalize because the index uses
    # inner product as cosine similarity.

    vectors /= np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    chunks = create_chunks()

    key = EncryptedFAISS.generate_key()

    print(
        "Key:",
        EncryptedFAISS.key_to_base64(
            key
        )
    )

    with tempfile.TemporaryDirectory() as directory:

        path = Path(
            directory
        ) / "secure_rag.faiss.enc"

        store = EncryptedFAISS(
            storage_path=path,
            key=key,
        )

        store.add_chunks(
            chunks,
            vectors,
        )

        print(
            "Vectors:",
            store.count,
        )

        store.save()

        print(
            "Encrypted file:",
            path,
        )

        # ------------------------------------------------------------------
        # Load from disk
        # ------------------------------------------------------------------

        restored = EncryptedFAISS(
            storage_path=path,
            key=key,
        )

        restored.load()

        print(
            "Restored vectors:",
            restored.count,
        )

        # ------------------------------------------------------------------
        # Search
        # ------------------------------------------------------------------

        results = restored.search(
            vectors[0],
            top_k=3,
        )

        for result in results:

            print(
                result
            )

            metadata = restored.get_metadata(
                result.chunk_id
            )

            print(
                metadata
            )

        # ------------------------------------------------------------------
        # Tamper tests
        # ------------------------------------------------------------------

        data = bytearray(
            path.read_bytes()
        )

        data[-1] ^= 0x01

        path.write_bytes(
            data
        )

        corrupted = EncryptedFAISS(
            storage_path=path,
            key=key,
        )

        try:

            corrupted.load()

        except VectorStoreSecurityError:

            print(
                "Tamper detection: OK"
            )


if __name__ == "__main__":
    main()