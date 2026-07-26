"""
embeddings/embedder.py

Local embedding generation for Secure RAG.

Python 3.12+
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from documents.chunker import DocumentChunk


class EmbedderError(Exception):
    """Raised when embedding generation fails."""

    pass


@dataclass(slots=True, frozen=True)
class EmbeddingResult:
    """
    Embeddings generated for document chunks.
    """

    vectors: np.ndarray

    chunk_ids: list[str]

    dimension: int

    model_name: str


class Embedder:
    """
    Local embedding generator.

    Uses Sentence Transformers.

    The model is loaded lazily so importing this module
    does not immediately allocate the ML model.
    """

    def __init__(
        self,
        model_name: str = (
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:

        if batch_size <= 0:

            raise ValueError(
                "batch_size must be greater than zero."
            )

        self.model_name = model_name

        self.batch_size = batch_size

        self.normalize = normalize

        self._model = None

    # =========================================================================
    # Model
    # =========================================================================

    def _load_model(self):
        """
        Load the embedding model lazily.
        """

        if self._model is not None:

            return self._model

        try:

            from sentence_transformers import (
                SentenceTransformer,
            )

        except ImportError as error:

            raise EmbedderError(
                "sentence-transformers is not installed."
            ) from error

        try:

            self._model = SentenceTransformer(
                self.model_name
            )

        except Exception as error:

            raise EmbedderError(
                "Unable to load embedding model."
            ) from error

        return self._model

    # =========================================================================
    # Single text
    # =========================================================================

    def embed_text(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Generate one embedding vector.
        """

        if not text or not text.strip():

            raise EmbedderError(
                "Cannot embed empty text."
            )

        model = self._load_model()

        try:

            vector = model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
            )

        except Exception as error:

            raise EmbedderError(
                "Embedding generation failed."
            ) from error

        vector = np.asarray(
            vector,
            dtype=np.float32,
        )

        if vector.ndim != 1:

            raise EmbedderError(
                "Expected one-dimensional embedding."
            )

        return vector

    # =========================================================================
    # Multiple texts
    # =========================================================================

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        """

        if not texts:

            return np.empty(
                (0, 0),
                dtype=np.float32,
            )

        clean_texts = [

            text.strip()

            for text in texts

        ]

        if any(
            not text
            for text in clean_texts
        ):

            raise EmbedderError(
                "Empty text found in batch."
            )

        model = self._load_model()

        try:

            vectors = model.encode(

                clean_texts,

                batch_size=self.batch_size,

                convert_to_numpy=True,

                normalize_embeddings=self.normalize,

                show_progress_bar=False,

            )

        except Exception as error:

            raise EmbedderError(
                "Batch embedding generation failed."
            ) from error

        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        if vectors.ndim != 2:

            raise EmbedderError(
                "Expected two-dimensional embedding matrix."
            )

        return vectors

    # =========================================================================
    # Document chunks
    # =========================================================================

    def embed_chunks(
        self,
        chunks: Sequence[DocumentChunk],
    ) -> EmbeddingResult:
        """
        Generate embeddings for DocumentChunk objects.
        """

        if not chunks:

            return EmbeddingResult(

                vectors=np.empty(
                    (0, 0),
                    dtype=np.float32,
                ),

                chunk_ids=[],

                dimension=0,

                model_name=self.model_name,

            )

        texts = [
            chunk.text
            for chunk in chunks
        ]

        vectors = self.embed_texts(
            texts
        )

        chunk_ids = [
            chunk.chunk_id
            for chunk in chunks
        ]

        return EmbeddingResult(

            vectors=vectors,

            chunk_ids=chunk_ids,

            dimension=vectors.shape[1],

            model_name=self.model_name,

        )

    # =========================================================================
    # Dimension
    # =========================================================================

    def dimension(self) -> int:
        """
        Return embedding vector dimension.
        """

        model = self._load_model()

        dimension = model.get_sentence_embedding_dimension()

        if dimension is None:

            raise EmbedderError(
                "Embedding dimension unavailable."
            )

        return int(dimension)


# =============================================================================
# Global instance
# =============================================================================

embedder = Embedder()