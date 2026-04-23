"""Tests for token-aware chunking"""
import pytest
from src.chunking import chunk_document, tokenize, detokenize


def test_tokenize_counts_tokens():
    """Should produce non-empty token list"""
    text = "This is a test sentence for chunking."
    tokens = tokenize(text)
    assert len(tokens) > 0


def test_detokenize_round_trips():
    """Tokenize then detokenize should return original text"""
    text = "Delta Lake provides ACID transactions"
    tokens = tokenize(text)
    result = detokenize(tokens)
    assert result == text


def test_chunk_document_creates_overlapping_chunks():
    """Should create multiple overlapping chunks for large documents"""
    doc = {
        "doc_id": 1,
        "url": "https://example.com",
        "title": "Test Doc",
        "content": " ".join(["word"] * 300)
    }
    chunks = chunk_document(doc, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    # Verify chunks have correct structure
    assert chunks[0]["doc_id"] == 1
    assert chunks[0]["chunk_id"] == 0
    assert "text" in chunks[0]
    assert "metadata" in chunks[0]


def test_chunk_document_small_doc_single_chunk():
    """Small documents should produce a single chunk"""
    doc = {
        "doc_id": 1,
        "url": "https://example.com",
        "title": "Small",
        "content": "Short text"
    }
    chunks = chunk_document(doc, chunk_size=256, overlap=50)
    assert len(chunks) == 1


def test_chunk_document_no_infinite_loop():
    """Should not infinite loop with edge case parameters"""
    doc = {
        "doc_id": 1,
        "url": "https://example.com",
        "title": "Test",
        "content": " ".join(["word"] * 100)
    }
    # overlap == 0 should work fine
    chunks = chunk_document(doc, chunk_size=50, overlap=0)
    assert len(chunks) >= 1
    
    # overlap >= chunk_size should not infinite loop
    chunks = chunk_document(doc, chunk_size=50, overlap=50)
    assert len(chunks) >= 1


def test_chunk_document_metadata():
    """Each chunk should carry source metadata"""
    doc = {
        "doc_id": 42,
        "url": "https://docs.databricks.com/test",
        "title": "Delta Lake Test",
        "content": " ".join(["word"] * 100)
    }
    chunks = chunk_document(doc, chunk_size=50, overlap=10)
    for chunk in chunks:
        assert chunk["metadata"]["url"] == "https://docs.databricks.com/test"
        assert chunk["metadata"]["title"] == "Delta Lake Test"
        assert chunk["metadata"]["token_count"] > 0
