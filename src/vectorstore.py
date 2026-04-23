"""FAISS vector store for RAG pipeline"""
import numpy as np
import faiss
import json
from pathlib import Path
from typing import List, Tuple, Optional


class FAISSIndex:
    """FAISS index wrapper with metadata storage"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.metadata: List[dict] = []
    
    def build(self, vectors: np.ndarray, metadata: List[dict]):
        """Build index from vectors and metadata.
        
        Args:
            vectors: numpy array of shape (n, dimension)
            metadata: list of metadata dicts, one per vector
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} != expected {self.dimension}")
        
        # Create index using FlatIP for cosine similarity (assumes normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Ensure float32
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        
        self.index.add(vectors)
        self.metadata = metadata
    
    @property
    def ntotal(self) -> int:
        """Number of vectors in the index"""
        return self.index.ntotal if self.index else 0
    
    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search for top-k most similar vectors.
        
        Args:
            query: query vector(s), shape (1, dimension) or (dimension,)
            k: number of results to return
        
        Returns:
            Tuple of (distances, indices) arrays
        """
        if self.index is None:
            raise ValueError("Index not built yet. Call build() first.")
        
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        query = query.astype(np.float32)
        return self.index.search(query, k)
    
    def get_chunk(self, index: int) -> dict:
        """Get metadata for a specific vector index"""
        return self.metadata[index]
    
    def save(self, index_path: str, metadata_path: str):
        """Persist index and metadata to disk"""
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)
        
        with open(metadata_path, "w") as f:
            for meta in self.metadata:
                f.write(json.dumps(meta) + "\n")
    
    @classmethod
    def load(cls, index_path: str, metadata_path: str) -> "FAISSIndex":
        """Load index and metadata from disk"""
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
    """Create a standalone FAISS index from vectors (convenience function)"""
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    
    index.add(vectors)
    return index
