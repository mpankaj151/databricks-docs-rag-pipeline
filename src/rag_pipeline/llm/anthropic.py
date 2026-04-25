"""Anthropic Claude LLM client.

Connects to Anthropic's Messages API (https://api.anthropic.com/v1/messages)
to generate responses.

API key:
    Set ANTHROPIC_API_KEY env var, or pass api_key directly.

Model:
    "claude-3-5-sonnet-latest" is the recommended default.
    Also: "claude-3-opus-latest", "claude-3-haiku-latest"

Key difference from Ollama:
    Anthropic uses the x-api-key header (not Authorization: Bearer).
    Anthropic also uses a separate "system" field in the payload
    (not a system message in the messages array).

The system parameter:
    Used for instructions injected before the user prompt.
    In RAG mode, the strict_prompt IS the user prompt content —
    system is used for additional instructions if needed.
"""
import os
import requests
from rag_pipeline.config import DEFAULT_STRICT_PROMPT


class AnthropicLLM:
    """Anthropic Claude LLM client implementing the LLMClient Protocol.

    Attributes:
        model: Anthropic model name (e.g. "claude-3-5-sonnet-latest").
        base_url: Always "https://api.anthropic.com" for the Messages API.
        temperature: Sampling temperature.
        max_tokens: Max response tokens. Required — Anthropic rejects 0.
        api_key: Anthropic API key. Reads ANTHROPIC_API_KEY env var if empty.
        _strict_prompt: Anti-hallucination prompt template.

    Note:
        max_tokens must be > 0. Anthropic's API requires it.
        For short RAG answers, 512 is usually sufficient.
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
        # Read from env var if not provided explicitly.
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._strict_prompt = strict_prompt or DEFAULT_STRICT_PROMPT

    @property
    def strict_prompt(self) -> str:
        return self._strict_prompt

    def _get_headers(self) -> dict:
        """Build request headers for Anthropic's API.

        Anthropic requires:
        - x-api-key: API key (not Bearer token).
        - anthropic-version: Must be "2023-06-01" for the Messages API.
        """
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def generate(self, prompt: str, system: str = None) -> str:
        """Send a prompt to Claude and return the response.

        Uses Anthropic's /v1/messages endpoint (not /v1/chat/completions).
        The system field goes in the payload root, not in messages.

        Args:
            prompt: User's message text.
            system: Optional system instruction.

        Returns:
            Claude's text response.
        """
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

        # Inject system as the separate "system" field if provided.
        if system:
            payload["system"] = system

        response = requests.post(
            f"{self.base_url}/v1/messages",
            headers=self._get_headers(),
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        # Anthropic returns content as a list: [{"type": "text", "text": "..."}]
        result = response.json()
        return result["content"][0]["text"]

    def generate_with_context(
        self, question: str, context: str, strict_prompt: str = None
    ) -> str:
        """Generate a RAG-grounded answer using retrieved context.

        Args:
            question: The user's question.
            context: Concatenated retrieved chunks from FAISS.
            strict_prompt: Override (uses instance default if None).

        Returns:
            Claude's grounded answer.
        """
        template = strict_prompt or self._strict_prompt
        prompt = template.format(context=context, question=question)
        return self.generate(prompt)