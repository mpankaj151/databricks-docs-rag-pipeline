"""Tests for FAISS vector store"""
import pytest
import numpy as np
import faiss
import tempfile
import os
from src.vectorstore import FAISSIndex, create_faiss_index


def test_faiss_index_creation():
    """Should create an index with the correct number of vectors"""
    vectors = np.random.rand(10, 384).astype("float32")
    # Normalize for FlatIP
    faiss.normalize_L2(vectors)
    
    index = create_faiss_index(vectors)
    assert index.ntotal == 10


def test_faiss_index_class_build_and_search():
    """FAISSIndex class should build and search correctly"""
    dimension = 128
    index = FAISSIndex(dimension)
    
    # Create vectors and metadata
    vectors = np.random.rand(10, dimension).astype("float32")
    faiss.normalize_L2(vectors)
    
    metadata = [{"id": i, "text": f"Doc {i}"} for i in range(10)]
    
    # Build
    index.build(vectors, metadata)
    assert index.ntotal == 10
    
    # Search
    query = vectors[0:1]  # Search for the first vector
    distances, indices = index.search(query, k=3)
    
    assert distances.shape == (1, 3)
    assert indices.shape == (1, 3)
    
    # First result should be the query itself (index 0)
    assert indices[0][0] == 0
    # Score should be ~1.0 (cosine similarity of normalized vectors)
    assert np.isclose(distances[0][0], 1.0, atol=1e-5)


def test_faiss_index_get_chunk():
    """Should return correct metadata for index"""
    index = FAISSIndex(128)
    vectors = np.random.rand(2, 128).astype("float32")
    faiss.normalize_L2(vectors)
    
    metadata = [{"text": "A"}, {"text": "B"}]
    index.build(vectors, metadata)
    
    assert index.get_chunk(0) == {"text": "A"}
    assert index.get_chunk(1) == {"text": "B"}


def test_faiss_index_save_and_load(tmp_path):
    """Should persist and load from disk"""
    dimension = 64
    index = FAISSIndex(dimension)
    
    vectors = np.random.rand(5, dimension).astype("float32")
    faiss.normalize_L2(vectors)
    metadata = [{"idx": i} for i in range(5)]
    
    index.build(vectors, metadata)
    
    index_path = str(tmp_path / "test.index")
    meta_path = str(tmp_path / "test.jsonl")
    
    index.save(index_path, meta_path)
    
    assert os.path.exists(index_path)
    assert os.path.exists(meta_path)
    
    loaded = FAISSIndex.load(index_path, meta_path)
    
    assert loaded.dimension == dimension
    assert loaded.ntotal == 5
    assert len(loaded.metadata) == 5
    assert loaded.get_chunk(2) == {"idx": 2}
