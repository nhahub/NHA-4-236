"""Embedding model wrapper around sentence-transformers.

Loads a (biomedical or general) bi-encoder once and exposes batch encoding
that returns L2-normalized float32 vectors, suitable for cosine similarity
search with a FAISS inner-product index.
"""
from __future__ import annotations

import functools

import numpy as np

from config import settings


class EmbeddingModel:
    """Thin wrapper that lazily loads a SentenceTransformer model."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self._model = None  # lazy: avoid loading torch at import time

    @property
    def model(self):
        if self._model is None:
            # Imported lazily so unit tests that stub embeddings don't need torch.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts to normalized float32 vectors of shape (n, dim)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        """Encode a single string to a (dim,) vector."""
        return self.encode([text])[0]


@functools.lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    """Process-wide singleton so the model is loaded only once."""
    return EmbeddingModel()
