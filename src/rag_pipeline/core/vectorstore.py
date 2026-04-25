"""FAISS vector store for RAG pipeline."""
import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Tuple, Optional


class FAISSIndex:
    """FAISS vector index wrapper."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.metadata: List[dict] = []

    @property
    def ntotal(self) -> int:
        """Total number of vectors in the index."""
        if self.index is None:
            return 0
        return self.index.ntotal
    
    def build(self, vectors: np.ndarray, metadata: List[dict]):
        """Build index from vectors and metadata."""
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Dimension mismatch: {vectors.shape[1]} != {self.dimension}")
        
        self.index = faiss.IndexFlatIP(self.dimension)
        
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        
        self.index.add(vectors)
        self.metadata = metadata
    
    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search for top-k results."""
        if self.index is None:
            raise ValueError("Index not built")
        
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        query = query.astype(np.float32)
        return self.index.search(query, k)
    
    def get_chunk(self, index: int) -> dict:
        """Get metadata for a specific index."""
        return self.metadata[index]
    
    def save(self, index_path: str, metadata_path: str):
        """Persist index and metadata to disk."""
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)
        
        with open(metadata_path, "w") as f:
            for meta in self.metadata:
                f.write(json.dumps(meta) + "\n")
    
    @classmethod
    def load(cls, index_path: str, metadata_path: str) -> "FAISSIndex":
        """Load index and metadata from disk."""
        index = faiss.read_index(index_path)
        
        metadata = []
        with open(metadata_path, "r") as f:
            for line in f:
                metadata.append(json.loads(line))
        
        instance = cls(dimension=index.d)
        instance.index = index
        instance.metadata = metadata
        return instance