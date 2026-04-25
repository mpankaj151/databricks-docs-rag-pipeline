"""Ollama LLM client.

Connects to Ollama's API (https://ollama.com/api/chat) to generate
responses. Supports both local Ollama and Ollama Cloud.

Local mode:
    - base_url: http://localhost:11434
    - api_key: "" (no auth needed for local)
    - Must have ollama serve running
    Example model: "qwen3.5:35b"

Cloud mode:
    - base_url: https://ollama.com
    - api_key: OLLAMA_API_KEY from https://ollama.com/settings/keys
    - Works without local Ollama server
    Example model: "qwen3.5:cloud"

The client uses Ollama's /api/chat endpoint with the Messages API
format (same as OpenAI's chat completions). For cloud access, the
Authorization: Bearer header authenticates requests.

strict_prompt:
    The anti-hallucination prompt forces the LLM to answer using ONLY
    the retrieved context. {context} and {question} are substituted
    before sending to the LLM.
"""
import requests
from rag_pipeline.config import DEFAULT_STRICT_PROMPT


class OllamaLLM:
    """Ollama LLM client implementing the LLMClient Protocol.

    Attributes:
        model: Ollama model name (e.g. "qwen3.5:cloud").
        base_url: API base URL (http://localhost:11434 or https://ollama.com).
        temperature: Sampling temperature (0.2 = mostly deterministic).
        max_tokens: Max response tokens.
        api_key: Bearer token for Ollama Cloud. Empty for local.
        _strict_prompt: Anti-hallucination prompt template.
    """

    def __init__(
        self,
        model: str = "qwen3.5:cloud",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        max_tokens: int = 512,
        api_key: str = "",
        strict_prompt: str = "",
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self._strict_prompt = strict_prompt or DEFAULT_STRICT_PROMPT

    @property
    def strict_prompt(self) -> str:
        """Return the active strict_prompt (configured or default)."""
        return self._strict_prompt

    def generate(self, prompt: str, system: str = None) -> str:
        """Send a prompt to Ollama and return the response.

        Uses the /api/chat endpoint with messages format:
        - system: system message (if provided)
        - user: the prompt

        Args:
            prompt: User's message text.
            system: Optional system prompt.

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
            "stream": False,
        }

        # Auth header: only needed for Ollama Cloud (not for local).
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            headers=headers,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def generate_with_context(
        self, question: str, context: str, strict_prompt: str = None
    ) -> str:
        """Generate a RAG-grounded answer using retrieved context.

        Fills the strict_prompt template:
            {context} → retrieved doc chunks
            {question} → user's question
        Then sends to Ollama, which must answer ONLY from context.

        Args:
            question: The user's question.
            context: Concatenated retrieved chunks from FAISS.
            strict_prompt: Override (uses instance default if None).

        Returns:
            LLM's grounded answer.
        """
        template = strict_prompt or self._strict_prompt
        prompt = template.format(context=context, question=question)
        return self.generate(prompt)