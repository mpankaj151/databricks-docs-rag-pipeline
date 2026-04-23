"""Tests for embedding generation"""
import pytest
import numpy as np
from src.embed import EmbeddingModel, get_embedding_model


# Use smaller model for fast tests
TEST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def model():
    """Shared embedding model for all tests in this module"""
    return get_embedding_model(TEST_MODEL)


def test_embedding_model_initializes(model):
    """Model should initialize with correct dimension"""
    assert model is not None
    assert model.dimension == 384  # MiniLM-L6-v2 dimension


def test_embedding_generates_vector(model):
    """Should generate embeddings with correct shape"""
    embedding = model.encode(["test sentence"], show_progress=False)
    assert embedding.shape == (1, 384)


def test_embedding_batch(model):
    """Should handle multiple texts in a batch"""
    texts = ["Delta Lake", "ACID transactions", "Spark SQL"]
    embeddings = model.encode(texts, show_progress=False)
    assert embeddings.shape == (3, 384)


def test_embedding_single(model):
    """encode_single should return a 1D vector"""
    embedding = model.encode_single("test")
    assert embedding.ndim == 1
    assert embedding.shape[0] == 384


def test_similar_texts_closer(model):
    """Semantically similar texts should have higher cosine similarity"""
    emb_delta = model.encode_single("Delta Lake provides ACID transactions")
    emb_spark = model.encode_single("Apache Spark processes big data")
    emb_cooking = model.encode_single("How to bake chocolate cake")
    
    # Delta Lake and Spark are more related than Delta Lake and cooking
    sim_related = np.dot(emb_delta, emb_spark)
    sim_unrelated = np.dot(emb_delta, emb_cooking)
    assert sim_related > sim_unrelated
