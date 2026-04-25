"""Embedding model for RAG pipeline."""
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wrapper for sentence-transformers embedding model."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()
    
    def encode(self, texts, batch_size: int = 64, show_progress: bool = True):
        """Encode texts into embeddings."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
    
    def encode_single(self, text: str):
        """Encode a single text."""
        return self.encode([text], show_progress=False)[0]
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension