"""Retrieval module for RAG pipeline"""
import numpy as np
from typing import List, Optional
from src.vectorstore import FAISSIndex
from src.embed import get_embedding_model
from src.models import RetrievedChunk


class Retriever:
    """Retrieves relevant chunks from vector store based on query"""
    
    def __init__(self, index: FAISSIndex, embedding_model=None, top_k: int = 5):
        self.index = index
        self.embedding_model = embedding_model
        self.top_k = top_k
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """Retrieve top-k chunks for a query"""
        k = top_k or self.top_k
        
        # Encode query
        if self.embedding_model is None:
            self.embedding_model = get_embedding_model()
        
        query_vector = self.embedding_model.encode_single(query)
        
        # Search index
        distances, indices = self.index.search(query_vector, k=k)
        
        # Build results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # FAISS returns -1 for empty/not-found slots
            if idx < 0:
                continue
            
            chunk_meta = self.index.get_chunk(idx)
            results.append(RetrievedChunk(
                doc_id=chunk_meta["doc_id"],
                chunk_id=chunk_meta["chunk_id"],
                text=chunk_meta["text"],
                metadata=chunk_meta["metadata"],
                score=float(dist)
            ))
        
        return results
    
    def retrieve_raw(self, query: str, top_k: Optional[int] = None) -> List[dict]:
        """Retrieve as raw dicts (useful for debugging/inspection)"""
        results = self.retrieve(query, top_k)
        return [
            {
                "doc_id": r.doc_id,
                "chunk_id": r.chunk_id,
                "text": r.text,
                "score": r.score,
                **r.metadata
            }
            for r in results
        ]
