"""
Step 3: EMBEDDING
-----------------
How does a computer understand the "meaning" of a chunk of text? 
By using an Embedding Model.

An embedding model is a special type of neural network that reads a string of text 
and outputs an "embedding" - a long list of numbers (a vector). 
Think of this vector as coordinates in a massive high-dimensional map of concepts.

If two chunks of text have similar meanings (e.g., "The king was happy" and "The 
monarch felt joyful"), their vectors will be plotted very close to each other on this map, 
even though they share no exact words! This is the magic that makes modern AI search work.

In this file, we use `sentence-transformers`, a popular open-source library, to convert 
our text chunks into vectors.
"""
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json


class EmbeddingModel:
    """Wrapper around SentenceTransformer for generating embeddings."""
    
    def __init__(self, model_name: str, normalize: bool = True):
        self.model_name = model_name
        # The model is downloaded automatically from HuggingFace
        self.model = SentenceTransformer(model_name)
        
        # We almost always want to "normalize" embeddings.
        # This forces every vector to have a length of 1.0. 
        # It makes calculating similarity between vectors much faster and simpler later on.
        self.normalize = normalize
        
        # The dimension is how many numbers are in the vector (e.g. 384 or 768)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        """
        Convert a list of strings into a matrix of numbers (vectors).
        
        We process them in "batches" (e.g. 64 at a time) because neural networks 
        are highly optimized for parallel math operations.
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
        """Convert a single string into a vector (used during querying)."""
        return self.encode([text], show_progress=False)[0]


_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model(model_name: str = None) -> EmbeddingModel:
    """
    Get or create the embedding model.
    Loading a neural network into memory takes a few seconds, so we use a "singleton" 
    pattern to load it exactly once and keep reusing it.
    """
    global _embedding_model
    
    if model_name is None:
        from src.config import get_config
        config = get_config()
        model_name = config.embedding_model
    
    if _embedding_model is not None and _embedding_model.model_name == model_name:
        return _embedding_model
    
    _embedding_model = EmbeddingModel(model_name)
    return _embedding_model


def embed_chunks(chunks_path: str, output_path: str, model_name: str = None) -> tuple:
    """
    Read all the text chunks, compute an embedding vector for each one, 
    and save the giant matrix of numbers to a `.npy` (numpy) file.
    """
    model = get_embedding_model(model_name)
    
    chunks = []
    with open(chunks_path, "r") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    
    texts = [chunk["text"] for chunk in chunks]
    
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, batch_size=64, show_progress=True)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings.astype("float32"))
    
    return embeddings.shape
