"""OpenAI-compatible LLM client.

Connects to any OpenAI-compatible API (OpenAI, OpenRouter, LM Studio, etc.)
using OpenAI's /v1/chat/completions endpoint.

Why OpenAI-compatible?
    OpenAI defined the chat completions API format that most providers now use:
    POST /v1/chat/completions with a messages array.
    OpenRouter, LM Studio, vLLM, LocalAI all use this format.

Use cases:
    - OpenAI: base_url="https://api.openai.com/v1", api_key="sk-..."
    - OpenRouter (free models): base_url="https://openrouter.ai/api/v1"
    - LM Studio (local): base_url="http://localhost:1234/v1"

Key difference from Ollama:
    OpenAI uses "max_tokens" in the payload root (same as Ollama).
    Ollama uses the Messages API format. Both are very similar here.
"""
import os
import requests
from rag_pipeline.config import DEFAULT_STRICT_PROMPT


class OpenAILLM:
    """OpenAI-compatible LLM client implementing the LLMClient Protocol.

    Supports any provider that implements the OpenAI chat completions API:
    OpenAI, OpenRouter, LM Studio, vLLM, LocalAI, etc.

    Attributes:
        model: Model name (varies by provider — check provider docs).
        base_url: API base URL ending in /v1.
        temperature: Sampling temperature.
        max_tokens: Max response tokens.
        api_key: API key. Reads OPENAI_API_KEY env var if empty.
        _strict_prompt: Anti-hallucination prompt template.
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
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._strict_prompt = strict_prompt or DEFAULT_STRICT_PROMPT

    @property
    def strict_prompt(self) -> str:
        return self._strict_prompt

    def generate(self, prompt: str, system: str = None) -> str:
        """Send a prompt to the OpenAI-compatible API and return the response.

        Args:
            prompt: User's message text.
            system: Optional system instruction.

        Returns:
            The LLM's text response.
        """
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

        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def generate_with_context(
        self, question: str, context: str, strict_prompt: str = None
    ) -> str:
        """Generate a RAG-grounded answer using retrieved context.

        Args:
            question: The user's question.
            context: Concatenated retrieved chunks from FAISS.
            strict_prompt: Override (uses instance default if None).

        Returns:
            The LLM's grounded answer.
        """
        template = strict_prompt or self._strict_prompt
        prompt = template.format(context=context, question=question)
        return self.generate(prompt)