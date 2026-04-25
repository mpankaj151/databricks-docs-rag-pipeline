"""LLMFactory — selects the right LLM client class based on config.provider.

This is a simple factory pattern. Given a provider name (e.g. "anthropic"),
it instantiates the corresponding client class (e.g. AnthropicLLM).

Why a factory?
    The RAG pipeline works with any LLM — Ollama, Claude, GPT-4, Bedrock.
    Instead of if/elif chains everywhere, the factory centralizes
    provider-to-class mapping in one place. To add a new provider:
    1. Create the client class in a new file.
    2. Register it in _PROVIDER_MAP.
    3. Update config.yaml.

Provider → class mapping:
    "ollama"     → OllamaLLM     (local or cloud Ollama API)
    "anthropic"  → AnthropicLLM  (Anthropic Claude API)
    "openai"    → OpenAILLM    (OpenAI or OpenAI-compatible API)
    "bedrock"   → BedrockLLM   (AWS Bedrock API)
"""
from rag_pipeline.llm.ollama import OllamaLLM
from rag_pipeline.llm.anthropic import AnthropicLLM
from rag_pipeline.llm.openai import OpenAILLM
from rag_pipeline.llm.bedrock import BedrockLLM


class UnknownLLMProviderError(Exception):
    """Raised when config.provider doesn't match any known provider."""

    AVAILABLE = ["ollama", "anthropic", "openai", "bedrock"]


_PROVIDER_MAP = {
    "ollama": OllamaLLM,
    "anthropic": AnthropicLLM,
    "openai": OpenAILLM,
    "bedrock": BedrockLLM,
}


class LLMFactory:
    """Factory that creates an LLM client based on the provider name.

    Example:
        factory = LLMFactory(provider="anthropic")
        client = factory.create(
            model="claude-3-5-sonnet-latest",
            base_url="https://api.anthropic.com",
            api_key="sk-ant-...",
        )
        answer = client.generate_with_context(question, context)

    Args:
        provider: Provider name — "ollama" | "anthropic" | "openai" | "bedrock".

    Raises:
        UnknownLLMProviderError: If provider isn't in _PROVIDER_MAP.
    """

    def __init__(self, provider: str = "ollama"):
        self.provider = provider

    def create(
        self,
        model: str,
        base_url: str,
        temperature: float,
        max_tokens: int,
        api_key: str = "",
        strict_prompt: str = "",
    ):
        """Create and return an LLM client instance.

        Args:
            model: Model name (format varies by provider).
            base_url: API endpoint URL.
            temperature: Sampling temperature.
            max_tokens: Max response tokens.
            api_key: API key (if required by the provider).
            strict_prompt: Anti-hallucination prompt override.

        Returns:
            An instance of the provider's LLM client class.
            Implements LLMClient Protocol.

        Raises:
            UnknownLLMProviderError: If self.provider isn't recognized.
        """
        cls = _PROVIDER_MAP.get(self.provider)
        if cls is None:
            raise UnknownLLMProviderError(
                f"Unknown LLM provider: '{self.provider}'. "
                f"Available: {', '.join(sorted(_PROVIDER_MAP.keys()))}"
            )
        return cls(
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            strict_prompt=strict_prompt,
        )