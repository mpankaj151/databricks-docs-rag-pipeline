"""Tests for retrieval module"""
import pytest
import numpy as np
import faiss
from src.retrieval import Retriever
from src.vectorstore import FAISSIndex


class MockEmbeddingModel:
    """Mock model that returns deterministic vectors, just for testing"""
    def __init__(self, dimension):
        self.dimension = dimension
        
    def encode_single(self, text):
        # Create an orthogonal vector based on text length
        vec = np.zeros((1, self.dimension), dtype=np.float32)
        idx = min(len(text), self.dimension - 1)
        vec[0][idx] = 1.0
        faiss.normalize_L2(vec)
        return vec[0]


def test_retriever_finds_relevant_chunks():
    """Should retrieve relevant chunks using FAISS and embeddings"""
    dimension = 384
    index = FAISSIndex(dimension)
    
    # Create distinct orthogonal vectors
    v1 = np.zeros((1, dimension), dtype=np.float32)
    v1[0][10] = 1.0  # Length 10
    
    v2 = np.zeros((1, dimension), dtype=np.float32)
    v2[0][20] = 1.0  # Length 20
    
    v3 = np.zeros((1, dimension), dtype=np.float32)
    v3[0][30] = 1.0  # Length 30
    
    vectors = np.vstack([v1, v2, v3])
    faiss.normalize_L2(vectors)
    
    metadata = [
        {"doc_id": 1, "chunk_id": 0, "text": "A"*10, "metadata": {}},
        {"doc_id": 2, "chunk_id": 0, "text": "B"*20, "metadata": {}},
        {"doc_id": 3, "chunk_id": 0, "text": "C"*30, "metadata": {}}
    ]
    
    index.build(vectors, metadata)
    
    mock_model = MockEmbeddingModel(dimension)
    retriever = Retriever(index, embedding_model=mock_model, top_k=2)
    
    # Query length is 10, should match doc_id 1
    results = retriever.retrieve("X"*10)
    
    assert len(results) == 2
    # The first result should be the perfectly matching vector (length 10)
    assert results[0].doc_id == 1
    assert results[0].text == "A"*10
    
    # Check raw retrieve
    raw = retriever.retrieve_raw("X"*10)
    assert len(raw) == 2
    assert raw[0]["doc_id"] == 1
