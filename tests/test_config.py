"""Tests for the configuration system."""
import pytest
from rag_pipeline.config import (
    Config,
    load_config,
    get_config,
    set_config,
    reset_config,
    DataSource,
    EmbeddingsConfig,
    LLMConfig,
    RestAPIConfig,
    ToolConfig,
)


class TestConfigDefaults:
    """Config should provide sensible defaults."""

    def test_embed_model_default(self):
        assert Config().embeddings.model == "sentence-transformers/all-mpnet-base-v2"

    def test_chunk_size_default(self):
        assert Config().chunk_size_tokens == 256
        assert Config().chunk_overlap_tokens == 50

    def test_top_k_default(self):
        assert Config().top_k == 5

    def test_llm_defaults(self):
        cfg = Config().llm
        assert cfg.model == "qwen3.5:cloud"
        assert cfg.base_url == "http://localhost:11434"
        assert cfg.temperature == 0.2
        assert cfg.max_tokens == 512

    def test_api_defaults(self):
        cfg = Config().integrations.rest_api
        assert cfg.enabled is True
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8000

    def test_tool_defaults(self):
        cfg = Config().integrations.tool
        assert cfg.enabled is True
        assert cfg.auto_trigger is True


class TestConfigOverrides:
    """Config should accept custom values."""

    def test_custom_llm_model(self):
        from rag_pipeline.config import LLMConfig
        cfg = Config(llm=LLMConfig(model="llama3.2:latest", temperature=0.0))
        assert cfg.llm.model == "llama3.2:latest"
        assert cfg.llm.temperature == 0.0

    def test_custom_chunk_size(self):
        cfg = Config(chunk_size_tokens=512, top_k=10)
        assert cfg.chunk_size_tokens == 512
        assert cfg.top_k == 10

    def test_custom_api_port(self):
        from rag_pipeline.config import RestAPIConfig, IntegrationsConfig
        cfg = Config(integrations=IntegrationsConfig(
            rest_api=RestAPIConfig(port=9000)
        ))
        assert cfg.integrations.rest_api.port == 9000

    def test_custom_keywords(self):
        from rag_pipeline.config import ToolConfig, IntegrationsConfig
        cfg = Config(integrations=IntegrationsConfig(
            tool=ToolConfig(keywords=["delta", "spark"])
        ))
        assert "delta" in cfg.integrations.tool.keywords
        assert "spark" in cfg.integrations.tool.keywords


class TestConfigConvenienceAliases:
    """Convenience property aliases should work."""

    def test_embedding_model_alias(self):
        cfg = Config()
        assert cfg.embedding_model == cfg.embeddings.model

    def test_chunk_size_alias(self):
        cfg = Config()
        assert cfg.chunk_size == cfg.chunk_size_tokens

    def test_chunk_overlap_alias(self):
        cfg = Config()
        assert cfg.chunk_overlap == cfg.chunk_overlap_tokens


class TestConfigFileLoading:
    """Config should load from YAML files."""

    def test_load_missing_file_returns_defaults(self):
        cfg = load_config("/nonexistent/config.yaml")
        assert cfg.embeddings.model == "sentence-transformers/all-mpnet-base-v2"

    def test_get_config_returns_same_instance(self):
        """get_config should be idempotent."""
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_set_and_reset_config(self):
        from rag_pipeline.config import LLMConfig, set_config as _set, reset_config as _reset
        cfg = Config(llm=LLMConfig(model="custom-model"))
        _set(cfg)
        assert get_config().llm.model == "custom-model"
        _reset()


class TestDataSource:
    """DataSource configuration."""

    def test_url_source(self):
        ds = DataSource(type="url", url="https://example.com", name="Test")
        assert ds.type == "url"
        assert ds.url == "https://example.com"

    def test_local_source(self):
        ds = DataSource(type="local", path="/data/docs", name="Local Docs")
        assert ds.type == "local"
        assert ds.path == "/data/docs"


class TestLLMClientFields:
    """Tests for LLMConfig provider and strict_prompt fields."""

    def test_llm_provider_default(self):
        """provider should default to 'ollama'."""
        from rag_pipeline.config import Config, LLMConfig
        cfg = Config()
        assert cfg.llm.provider == "ollama"

    def test_llm_provider_custom(self):
        """provider can be set to 'anthropic'."""
        from rag_pipeline.config import Config, LLMConfig
        cfg = Config(llm=LLMConfig(provider="anthropic", model="claude-test",
                                  api_key="sk-test"))
        assert cfg.llm.provider == "anthropic"

    def test_strict_prompt_default_empty(self):
        """strict_prompt should default to empty string."""
        from rag_pipeline.config import Config
        cfg = Config()
        assert cfg.llm.strict_prompt == ""

    def test_strict_prompt_custom(self):
        """strict_prompt can be set in config."""
        from rag_pipeline.config import Config, LLMConfig
        cfg = Config(llm=LLMConfig(strict_prompt="Custom prompt"))
        assert "Custom prompt" in cfg.llm.strict_prompt


class TestToolConfig:
    """Tool/MCP configuration."""

    def test_keywords_from_config(self):
        from rag_pipeline.config import ToolConfig, IntegrationsConfig
        cfg = Config(integrations=IntegrationsConfig(
            tool=ToolConfig(keywords=["delta lake", "databricks"])
        ))
        assert len(cfg.integrations.tool.keywords) == 2
        assert "delta lake" in cfg.integrations.tool.keywords