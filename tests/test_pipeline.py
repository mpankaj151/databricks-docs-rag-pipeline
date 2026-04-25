"""Tests for RAG pipeline and keyword detector."""
import pytest
from unittest.mock import patch, MagicMock

from rag_pipeline.pipeline.rag import RAGPipeline
from rag_pipeline.pipeline.keyword_detector import KeywordDetector


class TestKeywordDetector:
    """Keyword-based auto-trigger detection."""

    def test_default_keywords_exist(self):
        kd = KeywordDetector()
        assert len(kd.keywords) > 0
        assert "delta lake" in kd.keywords

    def test_contains_delta_lake(self):
        kd = KeywordDetector()
        assert kd.contains_keywords("Tell me about Delta Lake") is True

    def test_not_contains_random(self):
        kd = KeywordDetector()
        assert kd.contains_keywords("What is the weather?") is False

    def test_custom_keywords(self):
        kd = KeywordDetector(keywords=["custom-keyword"])
        assert kd.contains_keywords("I need custom-keyword info") is True

    def test_should_use_rag(self):
        kd = KeywordDetector()
        assert kd.should_use_rag("How to upsert into a Delta table?") is True

    def test_extract_found(self):
        kd = KeywordDetector(keywords=["delta lake", "databricks"])
        found = kd.extract_found("Tell me about Delta Lake on Databricks")
        assert "delta lake" in found
        assert "databricks" in found


class TestRAGPipelineQuery:
    """RAG pipeline query logic."""

    def test_pipeline_initializes(self):
        pipeline = RAGPipeline()
        assert pipeline.config is not None

    @patch("rag_pipeline.pipeline.rag.FAISSIndex")
    @patch("rag_pipeline.pipeline.rag.EmbeddingModel")
    def test_query_without_index_returns_error(self, mock_embed, mock_index):
        pipeline = RAGPipeline()
        pipeline.embedding_model = MagicMock()
        pipeline.index = None
        pipeline.llm = None

        result = pipeline.query("What is Delta Lake?")

        assert result["question"] == "What is Delta Lake?"
        assert "not loaded" in result["answer"]
        assert result["sources"] == []

    @patch("rag_pipeline.pipeline.rag.FAISSIndex")
    @patch("rag_pipeline.pipeline.rag.EmbeddingModel")
    def test_query_returns_structured_response(self, mock_embed, mock_index):
        pipeline = RAGPipeline()
        pipeline.llm = None

        mock_emb_model = MagicMock()
        mock_emb_model.encode_single.return_value = MagicMock()
        pipeline.embedding_model = mock_emb_model

        mock_idx = MagicMock()
        mock_idx.search.return_value = (
            MagicMock(__getitem__=lambda s, i: [0, 1]),
            [[0, 1]],
        )
        mock_idx.get_chunk.side_effect = [
            {"text": "Delta Lake is ACID.", "doc_id": 1, "chunk_id": 0},
            {"text": "It supports time travel.", "doc_id": 1, "chunk_id": 1},
        ]
        pipeline.index = mock_idx

        result = pipeline.query("What is Delta Lake?")

        assert "question" in result
        assert "answer" in result
        assert "sources" in result
        assert isinstance(result["sources"], list)


class TestRAGPipelineConfig:
    """RAG pipeline configuration usage."""

    def test_pipeline_uses_global_config(self):
        from rag_pipeline.config import get_config
        pipeline = RAGPipeline()
        assert pipeline.config is get_config()

    def test_pipeline_accepts_custom_config(self):
        from rag_pipeline.config import Config
        cfg = Config(top_k=10)
        pipeline = RAGPipeline(config=cfg)
        assert pipeline.config.top_k == 10