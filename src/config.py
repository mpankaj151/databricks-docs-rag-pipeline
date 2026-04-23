"""Configuration loading for RAG pipeline"""
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
import yaml
import os


class Config(BaseModel):
    """RAG pipeline configuration with sensible defaults"""
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_batch_size: int = 64
    chunk_size_tokens: int = 256
    chunk_overlap_tokens: int = 50
    top_k: int = 5
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "glm-5.1:cloud"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: str = ""  # Set OPENROUTER_API_KEY env var or pass directly
    llm_temperature: float = 0.2
    llm_max_tokens: int = 512
    data_dir: str = "data"
    docs_raw: str = "data/docs_raw.jsonl"
    docs_chunks: str = "data/docs_chunks.jsonl"
    faiss_index: str = "data/delta_lake.index"
    vectometa: str = "data/vectometa.jsonl"
    log_level: str = "INFO"
    query_log: str = "data/query_log.jsonl"

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "Config":
        """Load config from YAML file, falling back to defaults"""
        config_path = Path(path)
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f)
                if data:
                    return cls(**data)
        return cls()


_config: Optional[Config] = None


def get_config(path: str = "config.yaml") -> Config:
    """Get or create config singleton"""
    global _config
    if _config is None:
        _config = Config.from_yaml(path)
        # Override API key from env if set
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if api_key:
            _config.llm_api_key = api_key
    return _config


def reset_config():
    """Reset config singleton (useful for testing)"""
    global _config
    _config = None
