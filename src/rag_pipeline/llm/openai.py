"""OpenAI LLM client."""
from rag_pipeline.llm._base import LLMClient


class OpenAILLM(LLMClient):
    """OpenAI LLM client."""

    def __init__(
        self,
        model: str,
        base_url: str,
        temperature: float,
        max_tokens: int,
        api_key: str = "",
        strict_prompt: str = "",
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.strict_prompt = strict_prompt

    def generate(self, prompt: str) -> str:
        raise NotImplementedError()

    def generate_with_context(self, prompt: str, context: list[str]) -> str:
        raise NotImplementedError()