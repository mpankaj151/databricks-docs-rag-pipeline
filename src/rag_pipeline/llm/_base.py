"""LLMClient Protocol — duck-typed interface that all LLM providers must implement.

This module defines the contract (Protocol) that every LLM client must follow.
Python's Protocol, combined with @runtime_checkable, lets us verify that a
class implements the interface at runtime.

Why a Protocol instead of an ABC?
    ABC requires inheritance — you'd have to subclass for every provider.
    Protocol uses structural subtyping (duck typing): any class with the right
    methods works, regardless of its inheritance chain. The LLMFactory
    doesn't care if it's OllamaLLM, AnthropicLLM, or a custom class —
    it just checks that the methods exist.

In practice:
    LLMFactory.create() instantiates the right class based on
    config.provider. The returned object has .generate() and
    .generate_with_context(). RAGPipeline doesn't know or care
    which provider it is — it calls the Protocol methods.

Adding a new provider:
    1. Create a new file: src/rag_pipeline/llm/myprovider.py
    2. Implement MyProviderLLM with .generate() and .generate_with_context()
    3. Register in _factory.py: _PROVIDER_MAP["myprovider"] = MyProviderLLM
    4. Add to config.yaml: llm.provider = "myprovider"
"""
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Interface that all LLM clients must implement.

    This is a Python Protocol — a structural interface.
    Any class with these methods is considered an LLMClient,
    regardless of inheritance.

    Required attributes:
        model: Model name string.
        temperature: Sampling temperature (float).
        max_tokens: Maximum response tokens (int).

    Methods:
        generate(prompt, system): Send a text prompt to the LLM.
        generate_with_context(question, context, strict_prompt): RAG generation.

    Example — checking if a class is an LLMClient:
        from typing import runtime_checkable
        isinstance(my_client, LLMClient)  # True if methods exist
    """

    model: str
    temperature: float
    max_tokens: int

    def generate(self, prompt: str, system: str = None) -> str:
        """Send a text prompt to the LLM and get a response.

        Args:
            prompt: The user's prompt text.
            system: Optional system prompt (e.g. "You are a helpful assistant").
                Not all providers use this — it's optional per Protocol.

        Returns:
            The LLM's text response.
        """
        ...

    def generate_with_context(
        self, question: str, context: str, strict_prompt: str = None
    ) -> str:
        """Generate a RAG-grounded answer using only the provided context.

        This is the core RAG generation method. It:
        1. Fills the strict_prompt template with context + question.
        2. Sends the filled prompt to the LLM.
        3. Returns the answer (forced to use only the context).

        Args:
            question: The user's question.
            context: Concatenated retrieved chunks from the vector store.
            strict_prompt: The anti-hallucination prompt template.
                If None, uses the provider's built-in default.

        Returns:
            The LLM's answer, grounded ONLY in the provided context.
            If the context doesn't answer the question, the LLM
            must say "I don't have enough information".
        """
        ...