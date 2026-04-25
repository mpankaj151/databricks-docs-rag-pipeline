"""Tests for core modules: chunking, embeddings, vectorstore, ingest."""
import pytest
import numpy as np
import faiss
import tempfile
import json
from pathlib import Path

from rag_pipeline.core.chunking import chunk_text, tokenize, get_tokenizer
from rag_pipeline.core.embed import EmbeddingModel
from rag_pipeline.core.vectorstore import FAISSIndex
from rag_pipeline.core.ingest import fetch_databricks_docs, save_raw_docs, load_raw_docs, ingest_documents
from rag_pipeline.core.models import DocChunk, RetrievedChunk, QueryResult


# Use smaller model for tests
TEST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class TestChunking:
    """Text chunking with token awareness."""

    def test_tokenize_produces_tokens(self):
        text = "Delta Lake provides ACID transactions for Spark workloads."
        tokens = tokenize(text)
        assert len(tokens) > 0

    def test_chunk_text_overlapping(self):
        text = " ".join(["word"] * 200)
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        assert len(chunks) > 1

    def test_chunk_small_doc_single(self):
        text = "Delta Lake is simple."
        chunks = chunk_text(text, chunk_size=256, overlap=50)
        assert len(chunks) == 1

    def test_chunk_overlap_zero_no_infinite(self):
        text = " ".join(["word"] * 100)
        chunks = chunk_text(text, chunk_size=50, overlap=0)
        assert len(chunks) >= 1

    def test_get_tokenizer(self):
        enc = get_tokenizer()
        assert enc.encode("test") is not None


class TestEmbeddings:
    """Embedding model wrapper."""

    @pytest.fixture
    def model(self):
        return EmbeddingModel(model_name=TEST_MODEL)

    def test_model_dimension(self, model):
        assert model.get_dimension() == 384

    def test_encode_batch(self, model):
        vectors = model.encode(
            ["Delta Lake ACID", "Spark SQL", "Data engineering"],
            show_progress=False,
        )
        assert vectors.shape == (3, 384)

    def test_encode_single(self, model):
        vec = model.encode_single("Delta Lake")
        assert vec.shape == (384,)
        assert vec.ndim == 1

    def test_encode_normalizes(self, model):
        vec = model.encode_single("Delta Lake")
        norm = np.linalg.norm(vec)
        assert np.isclose(norm, 1.0, atol=0.01)


class TestVectorStore:
    """FAISS vector index."""

    def test_build_and_search(self):
        dim = 128
        index = FAISSIndex(dimension=dim)
        vectors = np.random.rand(5, dim).astype(np.float32)
        faiss.normalize_L2(vectors)

        metadata = [{"id": i, "text": f"Doc {i}"} for i in range(5)]
        index.build(vectors, metadata)

        distances, indices = index.search(vectors[0:1], k=3)
        assert distances.shape == (1, 3)
        assert indices[0][0] == 0  # Self-match

    def test_get_chunk(self):
        index = FAISSIndex(dimension=64)
        vectors = np.random.rand(3, 64).astype(np.float32)
        faiss.normalize_L2(vectors)
        metadata = [{"text": f"Text {i}"} for i in range(3)]
        index.build(vectors, metadata)

        assert index.get_chunk(0)["text"] == "Text 0"
        assert index.get_chunk(2)["text"] == "Text 2"

    def test_save_and_load(self, tmp_path):
        index = FAISSIndex(dimension=64)
        vectors = np.random.rand(4, 64).astype(np.float32)
        faiss.normalize_L2(vectors)
        metadata = [{"idx": i} for i in range(4)]
        index.build(vectors, metadata)

        index_path = str(tmp_path / "test.index")
        meta_path = str(tmp_path / "test.jsonl")
        index.save(index_path, meta_path)

        assert Path(index_path).exists()
        assert Path(meta_path).exists()

        loaded = FAISSIndex.load(index_path, meta_path)
        assert loaded.ntotal == 4
        assert len(loaded.metadata) == 4
        assert loaded.get_chunk(2)["idx"] == 2

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            FAISSIndex.load(
                str(tmp_path / "nonexistent.index"),
                str(tmp_path / "nonexistent.jsonl"),
            )

    def test_dimension_mismatch_raises(self):
        index = FAISSIndex(dimension=64)
        vectors = np.random.rand(3, 128).astype(np.float32)
        with pytest.raises(ValueError):
            index.build(vectors, [{"id": 1}])


class TestIngest:
    """Documentation ingestion."""

    def test_fetch_parses_html(self):
        html = """<html><body><main><h1>Delta Lake</h1>
        <p>Content here with actual info.</p></main></body></html>"""
        docs = fetch_databricks_docs("https://example.com/test", html)
        assert len(docs) == 1
        assert docs[0]["title"] == "Delta Lake"
        assert "Content here" in docs[0]["content"]

    def test_fetch_removes_nav(self):
        html = """<html><body><main><h1>Title</h1>
        <nav>Nav bar</nav><p>Real content</p><script>bad</script></main></body></html>"""
        docs = fetch_databricks_docs("https://example.com/nav", html)
        assert len(docs) == 1
        assert "Nav bar" not in docs[0]["content"]
        assert "bad" not in docs[0]["content"]
        assert "Real content" in docs[0]["content"]

    def test_save_and_load_docs(self, tmp_path):
        docs = [
            {"doc_id": 1, "url": "https://a.com", "title": "A", "content": "Hello"},
            {"doc_id": 2, "url": "https://b.com", "title": "B", "content": "World"},
        ]
        path = str(tmp_path / "docs.jsonl")
        save_raw_docs(docs, path)
        loaded = load_raw_docs(path)
        assert len(loaded) == 2
        assert loaded[0]["title"] == "A"

    def test_ingest_creates_chunks(self, sample_docs):
        chunks = ingest_documents(sample_docs, chunk_size=50, overlap=10)
        assert len(chunks) >= len(sample_docs)
        for chunk in chunks:
            assert "text" in chunk
            assert "doc_id" in chunk
            assert "chunk_id" in chunk


class TestModels:
    """Pydantic data models."""

    def test_doc_chunk(self):
        c = DocChunk(doc_id=1, chunk_id=0, text="Delta Lake info", metadata={})
        assert c.doc_id == 1
        assert c.chunk_id == 0

    def test_retrieved_chunk(self):
        c = RetrievedChunk(
            doc_id=1, chunk_id=0, text="Delta Lake info",
            metadata={}, score=0.95
        )
        assert c.score == 0.95

    def test_query_result(self, sample_chunks):
        rc = RetrievedChunk(
            doc_id=1, chunk_id=0,
            text="Delta Lake info", metadata={}, score=0.9
        )
        result = QueryResult(
            question="What is Delta Lake?",
            retrieved_chunks=[rc],
            answer="Delta Lake is...",
            latency_ms=150.0,
        )
        assert result.latency_ms == 150.0
        assert len(result.retrieved_chunks) == 1