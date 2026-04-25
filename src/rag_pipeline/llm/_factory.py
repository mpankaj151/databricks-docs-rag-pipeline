"""LLMFactory — selects the right LLM client based on config.provider."""
from rag_pipeline.llm.ollama import OllamaLLM
from rag_pipeline.llm.anthropic import AnthropicLLM
from rag_pipeline.llm.openai import OpenAILLM
from rag_pipeline.llm.bedrock import BedrockLLM


class UnknownLLMProviderError(Exception):
    """Raised when provider string is not recognized."""
    AVAILABLE = ["ollama", "anthropic", "openai", "bedrock"]


_PROVIDER_MAP = {
    "ollama": OllamaLLM,
    "anthropic": AnthropicLLM,
    "openai": OpenAILLM,
    "bedrock": BedrockLLM,
}


class LLMFactory:
    """Factory that selects the LLM client class based on provider string.

    Usage:
        factory = LLMFactory(provider="anthropic")
        client = factory.create(
            model="claude-3-5-sonnet-latest",
            base_url="https://api.anthropic.com",
            api_key="sk-ant-..."
        )
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
        """Create and return an LLM client instance."""
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