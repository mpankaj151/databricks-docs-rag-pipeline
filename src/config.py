"""Configuration loading for RAG pipeline"""
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
import yaml
import os


class Config(BaseModel):
    """RAG pipeline configuration with sensible defaults"""
    
    # --- Embedding Settings ---
    # The neural network model used to convert text chunks into vectors. 
    # all-mpnet-base-v2 is an excellent balance of speed and accuracy for general English text.
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    # How many chunks to process at once when generating embeddings. Higher = faster but uses more RAM/VRAM.
    embedding_batch_size: int = 64
    
    # --- Chunking Settings ---
    # The maximum number of tokens (approx. 3/4 of a word) per text chunk. 256 is good for specific fact retrieval.
    chunk_size_tokens: int = 256
    # How many tokens overlap between consecutive chunks to prevent cutting context in half.
    chunk_overlap_tokens: int = 50
    
    # --- Retrieval Settings ---
    # How many chunks to retrieve from the vector database for a single query.
    top_k: int = 5
    # Whether to use the advanced Cross-Encoder reranker. (Slower, but significantly improves accuracy).
    use_reranker: bool = False
    # The model used for reranking. ms-marco is specifically trained on Bing search query relevance.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # --- LLM / Generation Settings ---
    # The name of the Large Language Model to use (must be pulled in Ollama or exist on OpenRouter).
    llm_model: str = "glm-5.1:cloud"
    # Where the LLM is hosted. Default is local Ollama.
    llm_base_url: str = "http://localhost:11434"
    # Only needed for cloud APIs like OpenRouter. If empty, defaults to local Ollama.
    llm_api_key: str = ""  
    # Controls LLM creativity. 0.0 to 0.2 is best for RAG because we want strict factual answers, not creativity.
    llm_temperature: float = 0.2
    # The maximum length of the answer the LLM is allowed to generate.
    llm_max_tokens: int = 512
    
    # --- Storage & Path Settings ---
    data_dir: str = "data"
    # Where the original, cleaned plain-text documents are stored.
    docs_raw: str = "data/docs_raw.jsonl"
    # Where the smaller, tokenized chunks are stored.
    docs_chunks: str = "data/docs_chunks.jsonl"
    # The mathematical FAISS database file containing the vectors.
    faiss_index: str = "data/delta_lake.index"
    # A companion file to the FAISS index that maps Vector IDs back to the original text chunks.
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
