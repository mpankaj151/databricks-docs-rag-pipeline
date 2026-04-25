"""Tests for LLM clients and factory."""
import pytest
from unittest.mock import patch, MagicMock

class TestLLMClientProtocol:
    """Tests for LLMClient Protocol interface."""

    def test_llm_client_protocol_exists(self):
        """LLMClient should be a runtime_checkable Protocol."""
        from rag_pipeline.llm._base import LLMClient
        from typing import runtime_checkable, Protocol
        import inspect
        assert hasattr(LLMClient, '__protocol_attrs__') or LLMClient in {p for p in dir(Protocol) if p.isupper()}

    def test_ollama_implements_protocol(self):
        """OllamaLLM should satisfy LLMClient protocol at runtime."""
        from rag_pipeline.llm._base import LLMClient
        from rag_pipeline.llm.ollama import OllamaLLM
        from typing import runtime_checkable
        llm = OllamaLLM(model="test")
        # Duck-type check: has the required methods
        assert hasattr(llm, 'generate')
        assert callable(llm.generate)
        assert hasattr(llm, 'generate_with_context')
        assert callable(llm.generate_with_context)


class TestLLMFactory:
    """Tests for LLMFactory provider routing."""

    def test_factory_returns_ollama(self):
        """Factory with provider='ollama' should return OllamaLLM."""
        from rag_pipeline.llm._factory import LLMFactory
        factory = LLMFactory(provider="ollama")
        client = factory.create(
            model="qwen3.5:cloud",
            base_url="http://localhost:11434",
            temperature=0.2,
            max_tokens=512,
            api_key="",
            strict_prompt="",
        )
        assert type(client).__name__ == "OllamaLLM"

    def test_factory_returns_anthropic(self):
        """Factory with provider='anthropic' should return AnthropicLLM."""
        from rag_pipeline.llm._factory import LLMFactory
        factory = LLMFactory(provider="anthropic")
        client = factory.create(
            model="claude-3-5-sonnet-latest",
            base_url="https://api.anthropic.com",
            temperature=0.2,
            max_tokens=512,
            api_key="sk-test",
            strict_prompt="",
        )
        assert type(client).__name__ == "AnthropicLLM"

    def test_factory_unknown_provider_error(self):
        """Unknown provider should raise UnknownLLMProviderError."""
        from rag_pipeline.llm._factory import LLMFactory, UnknownLLMProviderError
        factory = LLMFactory(provider="unknown-provider")
        with pytest.raises(UnknownLLMProviderError) as exc:
            factory.create(
                model="test",
                base_url="http://localhost:11434",
                temperature=0.2,
                max_tokens=512,
                api_key="",
                strict_prompt="",
            )
        assert "unknown-provider" in str(exc.value)
        assert "ollama" in str(exc.value)