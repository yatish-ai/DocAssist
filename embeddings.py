"""
embeddings.py
-------------
Generates dense vector embeddings for text chunks using SentenceTransformers.

Model: all-MiniLM-L6-v2
  - 384-dimensional output vectors
  - Lightweight & fast (ideal for local use)
  - Strong semantic similarity performance

Public API
----------
  get_embedder()           → singleton EmbeddingModel instance
  EmbeddingModel.encode()  → np.ndarray of shape (N, 384)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Union

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
# Batch size for encoding – keeps memory usage predictable
ENCODE_BATCH_SIZE = 64
# Maximum number of embeddings to cache
EMBEDDING_CACHE_SIZE = 1000


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Model wrapper
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingModel:
    """
    Thin wrapper around a SentenceTransformer model.

    Usage::

        model = EmbeddingModel()
        vectors = model.encode(["Hello world", "Another sentence"])
        # vectors.shape → (2, 384)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """
        Load the SentenceTransformer model.

        The first call downloads the model (~90 MB) and caches it in
        ~/.cache/torch/sentence_transformers/  (or HF_HOME if set).
        Subsequent calls load from cache instantly.
        """
        logger.info("Loading embedding model: %s", model_name)
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.embedding_dim: int = self._model.get_sentence_embedding_dimension()

            # LRU cache for embeddings: text -> vector
            self._embedding_cache: Dict[str, np.ndarray] = {}

            logger.info(
                "Embedding model ready. Dimension: %d", self.embedding_dim
            )
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed.\n"
                "Run: pip install sentence-transformers"
            ) from exc

    # ------------------------------------------------------------------
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = ENCODE_BATCH_SIZE,
        show_progress_bar: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode one or more texts into dense vectors.

        Args:
            texts            : A single string or a list of strings.
            batch_size       : Number of texts encoded per forward pass.
            show_progress_bar: Show tqdm progress bar (useful for large batches).
            normalize        : L2-normalize vectors (recommended for cosine sim).

        Returns:
            np.ndarray of shape (len(texts), embedding_dim) with dtype float32.
        """
        if isinstance(texts, str):
            texts = [texts]

        # Check cache for existing embeddings
        cached_embeddings = []
        texts_to_encode = []
        indices = []

        for i, text in enumerate(texts):
            if text in self._embedding_cache:
                cached_embeddings.append((i, self._embedding_cache[text]))
            else:
                texts_to_encode.append(text)
                indices.append(i)

        # Encode missing texts
        if texts_to_encode:
            vectors = self._model.encode(
                texts_to_encode,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
            ).astype(np.float32)

            # Cache the new embeddings
            for text, vector in zip(texts_to_encode, vectors):
                self._embedding_cache[text] = vector
                # Simple LRU: remove oldest if cache is full
                if len(self._embedding_cache) > EMBEDDING_CACHE_SIZE:
                    # Remove a random item (simple approximation of LRU)
                    oldest_key = next(iter(self._embedding_cache))
                    del self._embedding_cache[oldest_key]

        # Combine cached and newly encoded embeddings in original order
        result = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        cache_idx = 0
        encode_idx = 0

        for i in range(len(texts)):
            if cached_embeddings and cached_embeddings[cache_idx][0] == i:
                result[i] = cached_embeddings[cache_idx][1]
                cache_idx += 1
            else:
                result[i] = vectors[encode_idx]
                encode_idx += 1

        return result

    # ------------------------------------------------------------------
    def encode_query(self, query: str) -> np.ndarray:
        """
        Convenience method: encode a single query string.

        Returns:
            np.ndarray of shape (1, embedding_dim).
        """
        return self.encode([query])


# ─────────────────────────────────────────────────────────────────────────────
# Singleton accessor (avoids reloading the model on every call)
# ─────────────────────────────────────────────────────────────────────────────

_embedder_instance: EmbeddingModel | None = None


def get_embedder(model_name: str = DEFAULT_MODEL_NAME) -> EmbeddingModel:
    """
    Return a cached EmbeddingModel instance (loads once, reused forever).

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        EmbeddingModel singleton.
    """
    global _embedder_instance
    if _embedder_instance is None or _embedder_instance.model_name != model_name:
        _embedder_instance = EmbeddingModel(model_name)
    return _embedder_instance


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test (python embeddings.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = [
        "Gradient descent is an optimization algorithm.",
        "Neural networks learn by adjusting weights.",
        "The Eiffel Tower is in Paris.",
    ]

    model = get_embedder()
    vecs = model.encode(sample)
    print(f"Model  : {model.model_name}")
    print(f"Shape  : {vecs.shape}")          # (3, 384)
    print(f"Dtype  : {vecs.dtype}")
    print(f"Sample norms: {np.linalg.norm(vecs, axis=1)}")   # ≈ 1.0 each
