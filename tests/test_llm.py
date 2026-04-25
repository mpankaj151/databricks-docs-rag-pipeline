"""Tests for LLM clients and factory."""
import json

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


class TestOpenAILLM:
    """Tests for OpenAILLM client."""

    def test_openai_generate(self):
        """OpenAILLM should call OpenAI-compatible /chat/completions endpoint."""
        from rag_pipeline.llm.openai import OpenAILLM
        from unittest.mock import patch, MagicMock

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Delta Lake provides ACID."}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("rag_pipeline.llm.openai.requests.post", return_value=mock_response) as mock_post:
            llm = OpenAILLM(
                model="gpt-4o",
                base_url="https://api.openai.com/v1",
                api_key="sk-test",
                temperature=0.2,
                max_tokens=512,
                strict_prompt="",
            )
            result = llm.generate("What is Delta Lake?")
            assert result == "Delta Lake provides ACID."
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["json"]["model"] == "gpt-4o"
            assert "Authorization" in call_kwargs["headers"]

    def test_openai_generate_with_context(self):
        """generate_with_context should fill strict_prompt template."""
        from rag_pipeline.llm.openai import OpenAILLM
        from unittest.mock import patch, MagicMock

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Answer"}}]}
        mock_response.raise_for_status = MagicMock()

        with patch("rag_pipeline.llm.openai.requests.post", return_value=mock_response):
            llm = OpenAILLM(
                model="gpt-4o",
                base_url="https://api.openai.com/v1",
                api_key="sk-test",
                temperature=0.2,
                max_tokens=512,
                strict_prompt="Context: {context}\nQuestion: {question}\nAnswer:",
            )
            result = llm.generate_with_context(
                "How to create a Delta table?",
                "Use CREATE TABLE my_table USING delta",
            )
            assert result == "Answer"
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


try:
    import boto3
except ImportError:
    boto3 = None


class TestBedrockLLM:
    """Tests for BedrockLLM client.

    Skipped if boto3 is not installed (Bedrock is an optional provider).
    """

    def test_bedrock_generate(self):
        """BedrockLLM should call AWS Bedrock invoke_model."""
        try:
            import boto3
        except ImportError:
            pytest.skip("boto3 not installed")

        from rag_pipeline.llm.bedrock import BedrockLLM

        mock_invoke_result = {"completion": "Delta Lake provides ACID transactions."}
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps(mock_invoke_result).encode()
        mock_response = {"body": mock_body}

        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.return_value = mock_response

        with patch("rag_pipeline.llm.bedrock.boto3", boto3):
            boto3.client.return_value = mock_bedrock
            llm = BedrockLLM(
                model="anthropic.claude-3-5-sonnet-latest",
                base_url="us-east-1",
                temperature=0.2,
                max_tokens=512,
                strict_prompt="",
            )
            result = llm.generate("What is Delta Lake?")
            assert result == "Delta Lake provides ACID transactions."
            mock_bedrock.invoke_model.assert_called_once()

    def test_bedrock_generate_with_context(self):
        """generate_with_context should fill strict_prompt and call Bedrock."""
        try:
            import boto3
        except ImportError:
            pytest.skip("boto3 not installed")

        from rag_pipeline.llm.bedrock import BedrockLLM

        mock_invoke_result = {"completion": "Use Delta Lake."}
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps(mock_invoke_result).encode()
        mock_response = {"body": mock_body}

        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.return_value = mock_response

        with patch("rag_pipeline.llm.bedrock.boto3", boto3):
            boto3.client.return_value = mock_bedrock
            llm = BedrockLLM(
                model="anthropic.claude-3-5-sonnet-latest",
                base_url="us-east-1",
                strict_prompt="Context: {context}\nQuestion: {question}\nAnswer:",
            )
            result = llm.generate_with_context(
                "How to create a table?", "Delta Lake syntax."
            )
            assert result == "Use Delta Lake."
            mock_bedrock.invoke_model.assert_called_once()


class TestAnthropicLLM:
    """Tests for AnthropicLLM client."""

    def test_anthropic_generate(self):
        """AnthropicLLM should call /v1/messages endpoint."""
        from rag_pipeline.llm.anthropic import AnthropicLLM

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"text": "Delta Lake provides ACID transactions."}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("rag_pipeline.llm.anthropic.requests.post", return_value=mock_response) as mock_post:
            llm = AnthropicLLM(
                model="claude-3-5-sonnet-latest",
                base_url="https://api.anthropic.com",
                api_key="sk-ant-test",
                temperature=0.2,
                max_tokens=512,
                strict_prompt="",
            )
            result = llm.generate("What is Delta Lake?")
            assert result == "Delta Lake provides ACID transactions."
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["json"]["model"] == "claude-3-5-sonnet-latest"
            assert "x-api-key" in call_kwargs["headers"]
            assert call_kwargs["headers"]["x-api-key"] == "sk-ant-test"

    def test_anthropic_generate_with_context(self):
        """generate_with_context should fill strict_prompt template."""
        from rag_pipeline.llm.anthropic import AnthropicLLM

        mock_response = MagicMock()
        mock_response.json.return_value = {"content": [{"text": "Answer"}]}
        mock_response.raise_for_status = MagicMock()

        with patch("rag_pipeline.llm.anthropic.requests.post", return_value=mock_response):
            llm = AnthropicLLM(
                model="claude-3-5-sonnet-latest",
                base_url="https://api.anthropic.com",
                api_key="sk-ant-test",
                temperature=0.2,
                max_tokens=512,
                strict_prompt="Context: {context}\nQuestion: {question}\nAnswer:",
            )
            result = llm.generate_with_context(
                "How to create a Delta table?",
                "Use CREATE TABLE my_table USING delta",
            )
            assert result == "Answer"


class TestLLMPackageExports:
    """Tests that llm package exports all expected names."""

    def test_llm_exports_factory(self):
        """llm package should export LLMFactory."""
        from rag_pipeline.llm import LLMFactory
        factory = LLMFactory(provider="ollama")
        assert hasattr(factory, "create")

    def test_llm_exports_all_clients(self):
        """llm package should export all LLM clients."""
        from rag_pipeline.llm import OllamaLLM, AnthropicLLM, OpenAILLM, BedrockLLM
        assert OllamaLLM is not None
        assert AnthropicLLM is not None
        assert OpenAILLM is not None
        assert BedrockLLM is not None

    def test_llm_exports_default_strict_prompt(self):
        """llm package should export DEFAULT_STRICT_PROMPT."""
        from rag_pipeline.llm import DEFAULT_STRICT_PROMPT
        assert "Answer ONLY from the context" in DEFAULT_STRICT_PROMPT
        assert "{context}" in DEFAULT_STRICT_PROMPT
        assert "{question}" in DEFAULT_STRICT_PROMPT