"""Embedding model wrapper for the RAG pipeline.

Embedding models convert text into dense vector representations —
numerical "fingerprints" that capture semantic meaning. This module wraps
sentence-transformers to provide:

- encode(): Convert a list of texts into a 2D numpy array of vectors.
- encode_single(): Convert one text into a 1D vector (for querying FAISS).

Why embeddings matter:
    A query like "how to create a Delta table" and a doc chunk
    "CREATE TABLE using Delta Lake syntax..." have similar meaning but
    very different surface text. TF-IDF or keyword search would miss this.
    Embeddings capture semantic similarity — the query and the doc chunk
    will have vectors pointing in similar directions (high cosine similarity).

Model choice:
    all-mpnet-base-v2 produces 768-dim vectors with the best quality
    on sentence similarity benchmarks. It requires ~1GB RAM just for the model.
    Smaller models like all-MiniLM-L6-v2 (384-dim, ~500MB) are faster but less accurate.

Normalization:
    Vectors are L2-normalized so that dot product = cosine similarity.
    This makes FAISS search (IndexFlatIP) equivalent to cosine search,
    and is numerically more stable than computing distances separately.
"""
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wrapper for sentence-transformers embedding model.

    Loads a pre-trained sentence-transformer model and exposes encode methods.
    The model is loaded once in __init__ and reused across all encode calls.

    Example:
        model = EmbeddingModel("sentence-transformers/all-mpnet-base-v2")
        vectors = model.encode(["hello world", "how are you?"])
        print(vectors.shape)    # (2, 768)
        print(model.dimension) # 768

    Attributes:
        model_name: HuggingFace model name or local path.
        model: The underlying SentenceTransformer instance.
        dimension: Output vector dimension (e.g. 768 for all-mpnet-base-v2).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()

    def encode(
        self,
        texts: list,
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Convert a list of texts into embedding vectors.

        Args:
            texts: List of text strings to encode.
            batch_size: Texts per batch. Higher = faster, more RAM.
            show_progress: Show a progress bar for large inputs.

        Returns:
            2D numpy array of shape (len(texts), dimension).
            Each row is the embedding vector for the corresponding text.
            Vectors are float32 and L2-normalized.
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def encode_single(self, text: str) -> np.ndarray:
        """Convert a single text into an embedding vector.

        Args:
            text: One text string.

        Returns:
            1D numpy array of shape (dimension,).
            Same as encode([text])[0] but without the list wrapper.
        """
        return self.encode([text], show_progress=False)[0]

    def get_dimension(self) -> int:
        """Return the embedding vector dimension."""
        return self.dimension