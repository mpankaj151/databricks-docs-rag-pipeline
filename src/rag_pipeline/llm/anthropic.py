"""Anthropic LLM client."""
import os
import requests
from rag_pipeline.config import DEFAULT_STRICT_PROMPT


class AnthropicLLM:
    """Anthropic Claude LLM client.

    Uses the Anthropic Messages API (v1/messages).
    Requires ANTHROPIC_API_KEY env var or api_key parameter.
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-latest",
        base_url: str = "https://api.anthropic.com",
        temperature: float = 0.2,
        max_tokens: int = 512,
        api_key: str = "",
        strict_prompt: str = "",
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._strict_prompt = strict_prompt or DEFAULT_STRICT_PROMPT

    @property
    def strict_prompt(self) -> str:
        return self._strict_prompt

    def _get_headers(self) -> dict:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def generate(self, prompt: str, system: str = None) -> str:
        messages = []
        if system:
            messages.append({"role": "user", "content": system + "\n\n" + prompt})
        else:
            messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        response = requests.post(
            f"{self.base_url}/v1/messages",
            headers=self._get_headers(),
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        return result["content"][0]["text"]

    def generate_with_context(
        self, question: str, context: str, strict_prompt: str = None
    ) -> str:
        template = strict_prompt or self._strict_prompt
        prompt = template.format(context=context, question=question)
        return self.generate(prompt)