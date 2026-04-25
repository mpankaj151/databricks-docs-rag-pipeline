"""Configuration loader for RAG pipeline."""
from pathlib import Path
from typing import Optional, List

import yaml
from pydantic import BaseModel, Field, ConfigDict

DEFAULT_STRICT_PROMPT = """You must answer using ONLY the provided context below.

RULES:
1. Answer ONLY from the context provided
2. If the context doesn't contain the answer, say "I don't have enough information"
3. NEVER use your own knowledge or make assumptions
4. Be concise and factual
5. If code examples are in the context, include them in your answer

Context:
{context}

Question: {question}

Answer (using ONLY context above):"""


class DataSource(BaseModel):
    """A data source for documentation."""
    model_config = ConfigDict(extra="ignore")

    type: str = "url"
    url: Optional[str] = None
    path: Optional[str] = None
    name: str = "Default"


class EmbeddingsConfig(BaseModel):
    """Embedding model configuration."""
    model_config = ConfigDict(extra="ignore")

    model: str = "sentence-transformers/all-mpnet-base-v2"
    batch_size: int = 64
    normalize: bool = True


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    model_config = ConfigDict(extra="ignore")

    provider: str = "ollama"
    model: str = "qwen3.5:cloud"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 512
    api_key: str = ""
    strict_prompt: str = ""


class RestAPIConfig(BaseModel):
    """REST API server configuration."""
    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8000


class ToolConfig(BaseModel):
    """Tool/MCP auto-trigger configuration."""
    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    auto_trigger: bool = True
    keywords: List[str] = Field(default_factory=list)


class IntegrationsConfig(BaseModel):
    """All integration configurations."""
    model_config = ConfigDict(extra="ignore")

    rest_api: RestAPIConfig = Field(default_factory=RestAPIConfig)
    tool: ToolConfig = Field(default_factory=ToolConfig)
    langchain: dict = {}
    lambda_handler: dict = {}
    cli: dict = {}


class Config(BaseModel):
    """Main configuration for the RAG pipeline.

    Supports construction from kwargs (e.g. Config(top_k=10, llm__model="custom"))
    which Pydantic interprets as nested field overrides.
    """

    model_config = ConfigDict(extra="allow")

    data_sources: List[DataSource] = Field(default_factory=list)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    integrations: IntegrationsConfig = Field(default_factory=IntegrationsConfig)
    chunk_size_tokens: int = 256
    chunk_overlap_tokens: int = 50
    top_k: int = 5
    min_similarity: float = 0.3

    # Convenience aliases (match the existing pipeline's field names)
    @property
    def embedding_model(self) -> str:
        return self.embeddings.model

    @property
    def chunk_size(self) -> int:
        return self.chunk_size_tokens

    @property
    def chunk_overlap(self) -> int:
        return self.chunk_overlap_tokens


_config: Optional[Config] = None


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from a YAML file.

    Falls back to defaults if the file doesn't exist.
    """
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
            return Config(**data)
    return Config()


def get_config() -> Config:
    """Get the global config instance (lazy-loaded)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config) -> None:
    """Set the global config instance."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global config (useful for testing)."""
    global _config
    _config = None


DEFAULT_STRICT_PROMPT = """You must answer using ONLY the provided context below.

RULES:
1. Answer ONLY from the context provided
2. If the context doesn't contain the answer, say "I don't have enough information"
3. NEVER use your own knowledge or make assumptions
4. Be concise and factual
5. If code examples are in the context, include them in your answer

Context:
{context}

Question: {question}

Answer (using ONLY context above):"""