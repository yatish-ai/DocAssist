"""
vector_store.py
---------------
Manages a FAISS-based vector index for semantic similarity search.

Responsibilities:
  1. Accept text chunks + their embeddings and build / update a FAISS index.
  2. Accept a query embedding and return the top-K most similar chunks.
  3. Persist the index to disk (save / load) for session continuity.

Each stored chunk carries metadata:
    {
      "text"    : str,   # The raw chunk text
      "source"  : str,   # Filename (e.g. "report.pdf")
      "page_num": int,   # Page number inside the source document
      "chunk_id": int,   # Sequential ID across all indexed chunks
    }
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Optional FAISS import check performed at class instantiation
try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# FAISSVectorStore
# ─────────────────────────────────────────────────────────────────────────────

class FAISSVectorStore:
    """
    In-memory FAISS index with parallel metadata storage.

    The index uses IndexFlatIP (Inner Product) which, for L2-normalised
    vectors, is equivalent to cosine similarity and is very fast on CPU.

    Attributes:
        embedding_dim : Dimensionality of the embeddings (e.g. 384).
        index         : The FAISS index object.
        metadata      : List of dicts, one per stored chunk.
    """

    def __init__(self, embedding_dim: int = 384):
        if not _FAISS_AVAILABLE:
            raise ImportError(
                "faiss-cpu is not installed.\n"
                "Run: pip install faiss-cpu"
            )
        self.embedding_dim = embedding_dim
        # IndexFlatIP = brute-force cosine similarity (no quantisation)
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(embedding_dim)
        self.metadata: List[Dict] = []   # parallel list to the FAISS index
        logger.info("FAISSVectorStore initialised (dim=%d)", embedding_dim)

    # ── Indexing ──────────────────────────────────────────────────────────────

    def add_chunks(
        self,
        chunks: List[Dict],          # [{"text":…, "source":…, "page_num":…}, …]
        embeddings: np.ndarray,      # shape (N, embedding_dim), float32
    ) -> None:
        """
        Add a batch of text chunks and their pre-computed embeddings to the index.

        Args:
            chunks     : List of chunk metadata dicts.
            embeddings : (N, D) float32 array of L2-normalised vectors.
        """
        assert len(chunks) == len(embeddings), (
            f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})"
        )

        # Assign sequential chunk IDs
        start_id = len(self.metadata)
        enriched = []
        for i, chunk in enumerate(chunks):
            enriched.append({
                **chunk,
                "chunk_id": start_id + i,
            })

        # FAISS requires contiguous float32
        vecs = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(vecs)
        self.metadata.extend(enriched)
        logger.info("Added %d chunks (total: %d)", len(chunks), len(self.metadata))

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,   # shape (1, D) or (D,)
        top_k: int = 5,
        use_mmr: bool = False,
        mmr_lambda: float = 0.7,
        mmr_candidates: int = 20,
    ) -> List[Dict]:
        """
        Retrieve the top-K chunks most semantically similar to the query.

        Args:
            query_embedding : Query vector, shape (1, D) or (D,).
            top_k           : Number of results to return.
            use_mmr         : Whether to use Maximal Marginal Relevance for diversity.
            mmr_lambda      : MMR trade-off parameter (0.0 = pure diversity, 1.0 = pure relevance).
            mmr_candidates  : Number of initial candidates to consider for MMR.

        Returns:
            List of chunk dicts ordered by descending similarity (or MMR score), each
            augmented with a "score" key (cosine similarity ∈ [-1, 1]).
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty – no results returned.")
            return []

        # Ensure shape (1, D)
        vec = np.ascontiguousarray(
            query_embedding.reshape(1, -1), dtype=np.float32
        )

        if use_mmr:
            return self._search_mmr(vec, top_k, mmr_lambda, mmr_candidates)
        else:
            # Original similarity search
            k = min(top_k, self.index.ntotal)
            scores, indices = self.index.search(vec, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue  # FAISS returns -1 for padding when k > ntotal
                result = {**self.metadata[idx], "score": float(score)}
                results.append(result)

            return results

    def _search_mmr(
        self,
        query_vec: np.ndarray,
        top_k: int,
        mmr_lambda: float,
        mmr_candidates: int,
    ) -> List[Dict]:
        """
        Perform Maximal Marginal Relevance (MMR) search for diverse results.

        MMR balances relevance to query with diversity from selected items.
        Formula: MMR = λ * Rel(S_i, Q) - (1-λ) * max_{S_j ∈ Selected} Sim(S_i, S_j)
        """
        # Get more candidates than needed for MMR selection
        candidates_k = min(mmr_candidates, self.index.ntotal)
        scores, indices = self.index.search(query_vec, candidates_k)

        # Build candidate list with metadata
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            candidates.append({
                **self.metadata[idx],
                "score": float(score),
                "embedding": self.index.reconstruct(int(idx))  # Get the embedding vector
            })

        if not candidates:
            return []

        # MMR selection
        selected = []
        remaining = candidates.copy()

        for _ in range(min(top_k, len(candidates))):
            if not remaining:
                break

            best_score = -float('inf')
            best_candidate = None
            best_idx = -1

            for i, candidate in enumerate(remaining):
                # Relevance score (cosine similarity to query)
                relevance = candidate["score"]

                # Diversity penalty (max similarity to already selected items)
                if selected:
                    similarities = [
                        np.dot(candidate["embedding"], sel["embedding"]) /
                        (np.linalg.norm(candidate["embedding"]) * np.linalg.norm(sel["embedding"]))
                        for sel in selected
                    ]
                    max_similarity = max(similarities)
                else:
                    max_similarity = 0.0

                # MMR score
                mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * max_similarity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = candidate
                    best_idx = i

            if best_candidate:
                selected.append(best_candidate)
                remaining.pop(best_idx)

        return selected

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str = "faiss_index") -> None:
        """
        Persist the FAISS index and metadata to disk.

        Creates two files:
          <directory>/index.faiss   – the FAISS binary index
          <directory>/metadata.pkl  – the metadata list

        Args:
            directory: Folder path (created if it doesn't exist).
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(Path(directory) / "index.faiss"))
        with open(Path(directory) / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info("Index saved to '%s'", directory)

    def load(self, directory: str = "faiss_index") -> None:
        """
        Load a previously saved FAISS index and metadata from disk.

        Args:
            directory: Folder path containing index.faiss + metadata.pkl.

        Raises:
            FileNotFoundError if the directory or files are missing.
        """
        index_path = Path(directory) / "index.faiss"
        meta_path  = Path(directory) / "metadata.pkl"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at '{index_path}'")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found at '{meta_path}'")

        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        logger.info(
            "Index loaded from '%s' (%d chunks)", directory, len(self.metadata)
        )

    # ── Utilities ─────────────────────────────────────────────────────────────

    @property
    def total_chunks(self) -> int:
        """Total number of indexed chunks."""
        return self.index.ntotal

    def clear(self) -> None:
        """Remove all vectors and metadata from the store."""
        self.index.reset()
        self.metadata.clear()
        logger.info("Vector store cleared.")

    def get_all_sources(self) -> List[str]:
        """Return a deduplicated list of document names in the index."""
        return list({m["source"] for m in self.metadata})

    def get_chunks_by_source(self, source: str) -> List[Dict]:
        """Return all chunks belonging to a specific document."""
        return [m for m in self.metadata if m["source"] == source]


# ─────────────────────────────────────────────────────────────────────────────
# Text Chunker (lives here to keep rag_pipeline.py clean)
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[str]:
    """
    Split *text* into overlapping word-based chunks.

    Overlap preserves context across chunk boundaries, improving retrieval
    quality for sentences that straddle chunk edges.

    Args:
        text         : Input string.
        chunk_size   : Target number of words per chunk.
        chunk_overlap: Number of words shared between consecutive chunks.

    Returns:
        List of chunk strings.
    """
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    step  = max(1, chunk_size - chunk_overlap)

    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += step

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    # Minimal test without real embeddings
    store = FAISSVectorStore(embedding_dim=4)
    rng   = np.random.default_rng(42)

    fake_chunks = [{"text": f"Chunk {i}", "source": "test.txt", "page_num": 1}
                   for i in range(5)]
    fake_vecs   = rng.random((5, 4), dtype=np.float32)
    # L2-normalise
    norms = np.linalg.norm(fake_vecs, axis=1, keepdims=True)
    fake_vecs /= norms

    store.add_chunks(fake_chunks, fake_vecs)
    print(f"Total chunks: {store.total_chunks}")

    query = rng.random((1, 4), dtype=np.float32)
    query /= np.linalg.norm(query)
    results = store.search(query, top_k=3)
    for r in results:
        print(f"  [{r['score']:.4f}] chunk_id={r['chunk_id']}  text={r['text']}")
