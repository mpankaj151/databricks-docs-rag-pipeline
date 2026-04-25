"""FAISS vector index wrapper.

FAISS (Facebook AI Similarity Search) is a library for fast nearest-neighbor
search in high-dimensional vector spaces. This module wraps FAISS to provide:

- Build: Convert a list of embedding vectors into a searchable FAISS index.
- Search: Given a query vector, find the k-most-similar document chunks.
- Save/Load: Persist the index to disk so ingestion only happens once.

How FAISS search works in this pipeline:
    1. During ingestion: documents → chunks → embeddings → FAISS index.
       Each chunk is a 768-dim vector stored at position i in the index.
       Metadata for chunk i (URL, title, text) is stored separately at metadata[i].

    2. During query: question → embedding vector → FAISS search.
       FAISS returns the positions of the k nearest vectors (by cosine similarity).
       We look up the metadata at those positions to get the original chunks.

Why FAISS instead of a database?
    - FAISS searches billions of vectors in milliseconds.
    - No server needed — it's an in-process library.
    - Perfect for single-node RAG pipelines like this one.

Dimension note:
    This pipeline uses all-mpnet-base-v2 which outputs 768-dim vectors.
    FAISS.IndexFlatIP (inner product) is used because vectors are L2-normalized,
    so cosine similarity = dot product.
"""
import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Tuple, Optional


class FAISSIndex:
    """FAISS vector index with in-memory metadata.

    The index stores embedding vectors. The metadata list stores the
    corresponding text chunks — same index position in both lists.

    Example:
        index = FAISSIndex(dimension=768)
        index.build(embeddings, metadata)      # load all chunks at once
        distances, indices = index.search(query_embedding, k=5)
        for i, idx in enumerate(indices[0]):
            chunk = index.get_chunk(idx)  # get text at position idx
            print(chunk["text"])

    Attributes:
        dimension: Embedding vector dimension (must match the model, e.g. 768).
        index: The underlying FAISS index object.
        metadata: List of chunk dicts, same length as index size.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        # Metadata is kept in Python (not in FAISS) — it can hold arbitrary dicts.
        # Parallel list: metadata[i] corresponds to index vector at position i.
        self.metadata: List[dict] = []

    @property
    def ntotal(self) -> int:
        """Total number of chunks in the index."""
        if self.index is None:
            return 0
        return self.index.ntotal

    def build(self, vectors: np.ndarray, metadata: List[dict]) -> None:
        """Build the FAISS index from embedding vectors and chunk metadata.

        Args:
            vectors: 2D numpy array of shape (N, dimension), one row per chunk.
                Must be float32. Vectors should be L2-normalized.
            metadata: List of N dicts, one per chunk.
                Each dict must have at least {"text": "..."}.

        Raises:
            ValueError: If vector dimension doesn't match self.dimension.
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: got {vectors.shape[1]}, "
                f"expected {self.dimension}"
            )

        # IndexFlatIP = inner product (dot product) index.
        # Used because vectors are L2-normalized, so dot product = cosine similarity.
        # For non-normalized vectors, use IndexFlatL2 instead.
        self.index = faiss.IndexFlatIP(self.dimension)

        # FAISS requires float32 input.
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        self.index.add(vectors)
        self.metadata = metadata

    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Find the k nearest chunks to a query vector.

        Args:
            query: 1D query vector (dimension,) or 2D (1, dimension).
            k: Number of results to return.

        Returns:
            Tuple of (distances, indices):
            - distances: 2D array of similarity scores per result.
              Closer to 1 = more similar.
            - indices: 2D array of chunk positions in the metadata list.
              -1 means no result (less than k matches found).

        Raises:
            ValueError: If the index hasn't been built yet.
        """
        if self.index is None:
            raise ValueError("Index not built — call .build() or .load() first")

        # Handle both 1D (single query) and 2D (batched query) input.
        if query.ndim == 1:
            query = query.reshape(1, -1)

        query = query.astype(np.float32)
        return self.index.search(query, k)

    def get_chunk(self, index: int) -> dict:
        """Get the metadata for a chunk at a given position.

        Args:
            index: Position in the index (returned by .search()).

        Returns:
            The metadata dict for this chunk (e.g. {"text": "...", "url": "..."}).
        """
        return self.metadata[index]

    def save(self, index_path: str, metadata_path: str) -> None:
        """Save the index and metadata to disk.

        Args:
            index_path: Path to save the FAISS index binary.
            metadata_path: Path to save metadata as newline-delimited JSON.

        Creates parent directories if they don't exist.
        The FAISS binary is fast to load but opaque.
        The JSONL metadata is human-readable and easy to inspect.
        """
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)

        with open(metadata_path, "w") as f:
            for meta in self.metadata:
                f.write(json.dumps(meta) + "\n")

    @classmethod
    def load(cls, index_path: str, metadata_path: str) -> "FAISSIndex":
        """Load a saved index and metadata from disk.

        Args:
            index_path: Path to the FAISS index binary file.
            metadata_path: Path to the newline-delimited JSON metadata.

        Returns:
            A new FAISSIndex instance with the loaded index and metadata.
        """
        index = faiss.read_index(index_path)

        metadata = []
        with open(metadata_path, "r") as f:
            for line in f:
                metadata.append(json.loads(line))

        instance = cls(dimension=index.d)
        instance.index = index
        instance.metadata = metadata
        return instance