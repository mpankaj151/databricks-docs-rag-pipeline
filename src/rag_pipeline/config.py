"""Configuration loader for the RAG pipeline.

This module loads all settings from config.yaml (or defaults) using Pydantic models.

How it works:
    1. config.yaml is parsed into nested Pydantic models (Config).
    2. Each section (embeddings, llm, integrations) has its own model.
    3. Pydantic validates types, sets defaults, and ignores unknown fields.
    4. get_config() returns a cached global instance — load once, use everywhere.

Why Pydantic?
    - Type safety: wrong types raise errors at load time, not at runtime.
    - Defaults: config.yaml can omit any field and Pydantic fills the rest.
    - Nesting: llm.model, llm.base_url etc. are nested objects, not flat dict keys.
    - Extra fields in config.yaml are silently ignored (extra="ignore").

Example config.yaml structure:
    embeddings:
        model: "sentence-transformers/all-mpnet-base-v2"
        batch_size: 64
    llm:
        provider: "ollama"
        model: "qwen3.5:cloud"
        base_url: "https://ollama.com"
    integrations:
        rest_api:
            enabled: true
            port: 8000

Each top-level key maps to a nested Pydantic model.
"""
from pathlib import Path
from typing import Optional, List

import yaml
from pydantic import BaseModel, Field, ConfigDict

# The anti-hallucination prompt injected into every LLM call.
# It forces the LLM to answer ONLY from the retrieved context.
# {context} and {question} are filled by RAGPipeline.query().
# The RULES section prevents the LLM from falling back to training knowledge.
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


# ─── Data source ────────────────────────────────────────────────────────────────

class DataSource(BaseModel):
    """A single documentation source to ingest.

    Attributes:
        type: "url" (fetch from web) or "path" (local file).
        url: URL to scrape (used when type="url").
        path: Local file path (used when type="path").
        name: Human-readable label for this source.
    """
    model_config = ConfigDict(extra="ignore")

    type: str = "url"
    url: Optional[str] = None
    path: Optional[str] = None
    name: str = "Default"


# ─── Embeddings ──────────────────────────────────────────────────────

class EmbeddingsConfig(BaseModel):
    """Configuration for the sentence-transformer embedding model.

    The embedding model converts text chunks into 768-dimensional vectors
    (one per chunk). These vectors are stored in FAISS for similarity search.

    Attributes:
        model: HuggingFace model name. Default "all-mpnet-base-v2" produces
            768-dim vectors — ideal for Apple Silicon with 64GB RAM.
            Smaller models (e.g. "all-MiniLM-L6-v2") produce 384-dim vectors.
        batch_size: Number of texts encoded per batch. Higher is faster but
            uses more RAM. 64 is a good default for 64GB systems.
        normalize: If True, vectors are unit-length (L2-normalized),
            which makes cosine similarity = dot product. Always keep True.
    """
    model_config = ConfigDict(extra="ignore")

    model: str = "sentence-transformers/all-mpnet-base-v2"
    batch_size: int = 64
    normalize: bool = True


# ─── LLM ────────────────────────────────────────────────────────────

class LLMConfig(BaseModel):
    """Configuration for the LLM provider.

    The provider is selected by config.provider and instantiated by LLMFactory.
    Each provider uses a different API endpoint and auth method.

    Attributes:
        provider: Which LLM to use — "ollama" | "anthropic" | "openai" | "bedrock".
        model: Model name (format varies by provider).
            - Ollama: "qwen3.5:cloud" (cloud model via ollama.com)
            - Anthropic: "claude-3-5-sonnet-latest"
            - OpenAI: "gpt-4o"
            - Bedrock: "anthropic.claude-3-5-sonnet-latest"
        base_url: API endpoint.
            - Ollama Cloud: "https://ollama.com"
            - Local Ollama: "http://localhost:11434"
            - Anthropic: "https://api.anthropic.com"
            - OpenAI: "https://api.openai.com/v1"
            - Bedrock: "us-east-1" (AWS region)
        temperature: How creative the LLM is. Lower = more deterministic.
            0.2 is good for factual RAG answers.
        max_tokens: Max response length. 512 is usually enough for
            short answers with code examples.
        api_key: API key or env var name. For Ollama Cloud, set the
            actual key or use OLLAMA_API_KEY env var.
        strict_prompt: Override for DEFAULT_STRICT_PROMPT. Set to ""
            to use the built-in default anti-hallucination prompt.
    """
    model_config = ConfigDict(extra="ignore")

    provider: str = "ollama"
    model: str = "qwen3.5:cloud"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 512
    api_key: str = ""
    strict_prompt: str = ""


# ─── Integrations ────────────────────────────────────────────────

class RestAPIConfig(BaseModel):
    """REST API server configuration.

    When enabled, a FastAPI server runs on host:port and exposes
    POST /rag/query for programmatic access.

    Attributes:
        enabled: Whether to start the REST API server.
        host: IP address to bind. "0.0.0.0" = all interfaces.
        port: TCP port. 8000 is the Streamlit/Gradio convention.
    """
    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8000


class ToolConfig(BaseModel):
    """Tool/MCP auto-trigger configuration.

    When enabled, the pipeline detects Databricks keywords in incoming
    queries and automatically routes them through RAG (instead of
    passing them directly to the LLM).

    Attributes:
        enabled: Whether the keyword detector is active.
        auto_trigger: If True, auto-detect and route. If False, require
            explicit RAG tool calls.
        keywords: List of phrases that trigger RAG routing.
            E.g. ["delta lake", "databricks", "spark sql"].
    """
    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    auto_trigger: bool = True
    keywords: List[str] = Field(default_factory=list)


class IntegrationsConfig(BaseModel):
    """All integration configurations.

    Groups config for each integration mode (REST API, Tool/MCP, LangChain, Lambda).
    Each integration shares the same RAGPipeline — only the access method differs.
    """
    model_config = ConfigDict(extra="ignore")

    rest_api: RestAPIConfig = Field(default_factory=RestAPIConfig)
    tool: ToolConfig = Field(default_factory=ToolConfig)
    langchain: dict = {}
    lambda_handler: dict = {}
    cli: dict = {}


# ─── Main Config ──────────────────────────────────────────────────

class Config(BaseModel):
    """Root configuration object for the entire RAG pipeline.

    Validates and holds all settings. Supports nested field overrides
    via kwargs (e.g. Config(top_k=10, llm__model="gpt-4o")) — Pydantic
    interprets "llm__model" as llm.model because __ is the default separator.

    Attributes:
        data_sources: List of documentation sources to ingest.
        embeddings: Embedding model settings.
        llm: LLM provider settings.
        integrations: Per-integration settings.
        chunk_size_tokens: Target chunk size in tokens (before embedding).
            256 tokens ≈ ~1000 chars, good for Databricks docs.
        chunk_overlap_tokens: Overlap between chunks in tokens.
            50 tokens (~200 chars) ensures context continuity.
        top_k: Number of chunks to retrieve per query.
            More chunks = more context, but risk of dilution.
        min_similarity: Minimum cosine similarity to include a chunk.
            0.3 means chunks must be at least somewhat relevant.

    Convenience properties (alias nested fields for backward compat):
        embedding_model → embeddings.model
        chunk_size    → chunk_size_tokens
        chunk_overlap → chunk_overlap_tokens
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

    @property
    def embedding_model(self) -> str:
        return self.embeddings.model

    @property
    def chunk_size(self) -> int:
        return self.chunk_size_tokens

    @property
    def chunk_overlap(self) -> int:
        return self.chunk_overlap_tokens


# ─── Global config instance ────────────────────────────────────────

# Lazy-loaded global config — loaded on first use, reused thereafter.
# reset_config() clears this (useful in tests to avoid state leakage).
_config: Optional[Config] = None


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to config.yaml. Defaults to "config.yaml"
            in the current working directory.

    Returns:
        Config instance (validated Pydantic model).

    If the file doesn't exist, returns Config() with all defaults.
    This means the pipeline works even without a config file.
    """
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
            return Config(**data)
    return Config()


def get_config() -> Config:
    """Get the global config instance (lazy-loaded).

    Returns the cached Config from load_config().
    The first call triggers load from config.yaml (or defaults).
    Subsequent calls return the same instance.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config) -> None:
    """Replace the global config instance.

    Use this to inject a test config or override defaults programmatically.
    """
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global config to None.

    Forces get_config() to re-read config.yaml on next call.
    Useful in tests to prevent state leakage between test cases.
    """
    global _config
    _config = None