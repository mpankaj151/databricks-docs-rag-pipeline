"""
Step 5: RETRIEVAL
-----------------
The "R" in RAG stands for Retrieval. When a user asks a question, this module orchestrates
the process of finding the right information.

The workflow is:
1. User asks "What is Delta Lake time travel?" (a string).
2. We pass that string into our Embedding Model to get a "Query Vector".
3. We pass the Query Vector into our FAISS Vector Database to search for the closest matches.
4. We look up the original text for the matching IDs and return them as `RetrievedChunk`s.
"""
import numpy as np
from typing import List, Optional
from src.vectorstore import FAISSIndex
from src.embed import get_embedding_model
from src.models import RetrievedChunk


class Retriever:
    """Retrieves relevant text chunks from the vector store based on a user's question."""
    
    def __init__(self, index: FAISSIndex, embedding_model=None, top_k: int = 5):
        self.index = index
        self.embedding_model = embedding_model
        # top_k defines how many chunks we want to fetch (e.g. the top 5 most relevant)
        self.top_k = top_k
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """
        Execute the retrieval pipeline.
        
        Args:
            query: The user's question.
            top_k: Override the default number of chunks to return.
        """
        k = top_k or self.top_k
        
        # 1. Convert the user's question into an embedding vector
        if self.embedding_model is None:
            self.embedding_model = get_embedding_model()
        
        query_vector = self.embedding_model.encode_single(query)
        
        # 2. Search the vector database for the nearest neighbors to the question vector
        distances, indices = self.index.search(query_vector, k=k)
        
        # 3. Build the final results list by looking up the text for each found vector ID
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # FAISS returns -1 if it couldn't find enough vectors (e.g., asked for 5 but DB only has 3)
            if idx < 0:
                continue
            
            # Look up the actual text using the ID
            chunk_meta = self.index.get_chunk(idx)
            
            # Package it into a neat Pydantic model
            results.append(RetrievedChunk(
                doc_id=chunk_meta["doc_id"],
                chunk_id=chunk_meta["chunk_id"],
                text=chunk_meta["text"],
                metadata=chunk_meta["metadata"],
                score=float(dist)  # The mathematical similarity score
            ))
        
        return results
    
    def retrieve_raw(self, query: str, top_k: Optional[int] = None) -> List[dict]:
        """Same as above, but returns raw dictionaries. Useful for debugging."""
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
