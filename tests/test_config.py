"""Tests for config loading"""
import pytest
from pathlib import Path
from src.config import Config, reset_config


def test_config_loads_defaults():
    """Config should provide sensible defaults without a YAML file"""
    config = Config()
    assert config.embedding_model == "sentence-transformers/all-mpnet-base-v2"
    assert config.chunk_size_tokens == 256
    assert config.chunk_overlap_tokens == 50
    assert config.top_k == 5


def test_config_from_yaml(tmp_path):
    """Config should load values from a YAML file"""
    yaml_content = """
embedding_model: "test-model"
chunk_size_tokens: 128
top_k: 10
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)
    
    config = Config.from_yaml(str(config_file))
    assert config.embedding_model == "test-model"
    assert config.chunk_size_tokens == 128
    assert config.top_k == 10
    # Defaults should still apply for unset values
    assert config.chunk_overlap_tokens == 50


def test_config_from_missing_yaml():
    """Config should use defaults when YAML file doesn't exist"""
    config = Config.from_yaml("/nonexistent/config.yaml")
    assert config.embedding_model == "sentence-transformers/all-mpnet-base-v2"


def test_config_has_single_api_key_field():
    """Config should have exactly one llm_api_key field (regression test for duplicate bug)"""
    config = Config()
    assert hasattr(config, "llm_api_key")
    assert config.llm_api_key == ""
