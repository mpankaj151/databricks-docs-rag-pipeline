"""Optional cross-encoder reranker for improved retrieval"""
from typing import List
import numpy as np
from src.models import RetrievedChunk


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int = 3) -> List[RetrievedChunk]:
        """Rerank chunks based on query-chunk relevance"""
        if not chunks:
            return []
        
        # Create query-chunk pairs
        pairs = [(query, chunk.text) for chunk in chunks]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Sort by score descending
        sorted_indices = np.argsort(scores)[::-1]
        
        # Reorder chunks
        reranked = [chunks[i] for i in sorted_indices[:top_k]]
        
        # Update scores
        for chunk, idx in zip(reranked, sorted_indices[:top_k]):
            chunk.score = float(scores[idx])
        
        return reranked
