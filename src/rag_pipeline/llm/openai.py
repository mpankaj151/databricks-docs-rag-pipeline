"""OpenAI-compatible LLM client.

Works with OpenAI, OpenRouter, or any OpenAI-compatible API endpoint.
Reads OPENAI_API_KEY env var if api_key is empty.
"""
import os
import requests
from rag_pipeline.config import DEFAULT_STRICT_PROMPT


class OpenAILLM:
    """OpenAI-compatible LLM client.

    Set base_url to:
      - https://api.openai.com/v1       (OpenAI)
      - https://openrouter.ai/api/v1    (OpenRouter)
      - http://localhost:8000/v1         (local OAI-compatible server)
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.2,
        max_tokens: int = 512,
        api_key: str = "",
        strict_prompt: str = "",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._strict_prompt = strict_prompt or DEFAULT_STRICT_PROMPT

    @property
    def strict_prompt(self) -> str:
        return self._strict_prompt

    def _get_headers(self) -> dict:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def generate(self, prompt: str, system: str = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._get_headers(),
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def generate_with_context(
        self, question: str, context: str, strict_prompt: str = None
    ) -> str:
        template = strict_prompt or self._strict_prompt
        prompt = template.format(context=context, question=question)
        return self.generate(prompt)