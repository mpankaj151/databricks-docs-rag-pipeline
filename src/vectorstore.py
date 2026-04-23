"""
Step 4: VECTOR DATABASE
-----------------------
Once we have thousands of embeddings (vectors) for our documents, where do we put them?
We put them in a Vector Database (Vector Store).

A standard database (like SQL) is great for finding an exact match: "SELECT * WHERE name='Delta'".
But it's terrible at finding "things that mean something similar to Delta".

A Vector Database is specifically designed to take a "query vector", and mathematically
calculate the distance between that query and every other vector in the database to 
find the nearest neighbors (the most conceptually similar texts).

In this project, we use FAISS (Facebook AI Similarity Search), an extremely fast, 
open-source library for vector similarity search.
"""
import numpy as np
import faiss
import json
from pathlib import Path
from typing import List, Tuple, Optional


class FAISSIndex:
    """FAISS index wrapper that also stores the original text/metadata for each vector."""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        
        # FAISS only stores the numbers (vectors). We need a separate list to remember 
        # which text chunk and URL corresponds to vector #1, vector #2, etc.
        self.metadata: List[dict] = []
    
    def build(self, vectors: np.ndarray, metadata: List[dict]):
        """
        Build the searchable index from our list of vectors.
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} != expected {self.dimension}")
        
        # IndexFlatIP means we use the "Inner Product" (Dot Product) to compare vectors.
        # Because we previously "normalized" our vectors to a length of 1.0, the Inner Product 
        # is mathematically identical to "Cosine Similarity" (the standard way to measure 
        # semantic similarity in AI).
        self.index = faiss.IndexFlatIP(self.dimension)
        
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        
        # Add the vectors to the FAISS database
        self.index.add(vectors)
        self.metadata = metadata
    
    @property
    def ntotal(self) -> int:
        """Number of vectors currently in the database."""
        return self.index.ntotal if self.index else 0
    
    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        The magic happens here: search the database for the top `k` most similar vectors.
        
        Returns two arrays:
        1. distances: How similar the results were (closer to 1.0 = more similar)
        2. indices: The integer ID of the vector (e.g., [42, 107, 3] means the 42nd vector is best)
        """
        if self.index is None:
            raise ValueError("Index not built yet. Call build() first.")
        
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        query = query.astype(np.float32)
        return self.index.search(query, k)
    
    def get_chunk(self, index: int) -> dict:
        """Look up the original text and metadata for a specific vector index."""
        return self.metadata[index]
    
    def save(self, index_path: str, metadata_path: str):
        """Save both the FAISS mathematical index and our text metadata to disk."""
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)
        
        with open(metadata_path, "w") as f:
            for meta in self.metadata:
                f.write(json.dumps(meta) + "\n")
    
    @classmethod
    def load(cls, index_path: str, metadata_path: str) -> "FAISSIndex":
        """Load a previously saved database from disk."""
        index = faiss.read_index(index_path)
        
        metadata = []
        with open(metadata_path, "r") as f:
            for line in f:
                if line.strip():
                    metadata.append(json.loads(line))
        
        instance = cls(dimension=index.d)
        instance.index = index
        instance.metadata = metadata
        return instance


def create_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """A simple helper to just create a raw FAISS index without our metadata wrapper."""
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    
    index.add(vectors)
    return index
