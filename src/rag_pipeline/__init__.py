"""Databricks Docs RAG Pipeline."""

__version__ = "0.1.0"

from rag_pipeline.config import Config, get_config, load_config, set_config
from rag_pipeline.pipeline.rag import RAGPipeline
from rag_pipeline.pipeline.keyword_detector import KeywordDetector
from rag_pipeline.core import EmbeddingModel, FAISSIndex, chunk_text
from rag_pipeline.llm import OllamaLLM

__all__ = [
    "Config",
    "get_config",
    "load_config",
    "set_config",
    "RAGPipeline",
    "KeywordDetector",
    "EmbeddingModel",
    "FAISSIndex",
    "chunk_text",
    "OllamaLLM",
    "__version__",
]