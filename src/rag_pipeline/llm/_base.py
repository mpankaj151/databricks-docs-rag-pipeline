"""LLMClient Protocol — duck-typed interface for all LLM providers."""
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Interface that all LLM clients must implement.

    Any class with .generate() and .generate_with_context() can be used
    as an LLM provider. The factory selects the right class based on
    config.provider.
    """

    model: str
    temperature: float
    max_tokens: int

    def generate(self, prompt: str, system: str = None) -> str:
        """Send prompt to LLM and return response."""
        ...

    def generate_with_context(
        self, question: str, context: str, strict_prompt: str = None
    ) -> str:
        """Generate answer using retrieved context and strict_prompt.

        Args:
            question: The user's question.
            context: Concatenated retrieved chunks from FAISS.
            strict_prompt: The anti-hallucination prompt template.

        Returns:
            LLM's answer using only the provided context.
        """
        ...