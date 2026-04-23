"""Embedding model for RAG pipeline"""
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json


class EmbeddingModel:
    """Wrapper around SentenceTransformer for generating embeddings"""
    
    def __init__(self, model_name: str, normalize: bool = True):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        """Encode texts into embeddings.
        
        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
        
        Returns:
            numpy array of shape (n_texts, dimension)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True
        )
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string"""
        return self.encode([text], show_progress=False)[0]


_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model(model_name: str = None) -> EmbeddingModel:
    """Get or create embedding model (cached singleton for repeated calls)"""
    global _embedding_model
    
    if model_name is None:
        from src.config import get_config
        config = get_config()
        model_name = config.embedding_model
    
    # Return cached if same model
    if _embedding_model is not None and _embedding_model.model_name == model_name:
        return _embedding_model
    
    _embedding_model = EmbeddingModel(model_name)
    return _embedding_model


def embed_chunks(chunks_path: str, output_path: str, model_name: str = None) -> tuple:
    """Load chunks from JSONL and generate embeddings.
    
    Returns:
        Shape tuple of the embeddings array
    """
    model = get_embedding_model(model_name)
    
    # Load chunks
    chunks = []
    with open(chunks_path, "r") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    
    texts = [chunk["text"] for chunk in chunks]
    
    # Generate embeddings
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, batch_size=64, show_progress=True)
    
    # Save embeddings as numpy array
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings.astype("float32"))
    
    return embeddings.shape
