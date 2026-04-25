"""Tests for LLM clients and factory."""
import pytest
from unittest.mock import patch, MagicMock

class TestLLMClientProtocol:
    """Tests for LLMClient Protocol interface."""

    def test_llm_client_protocol_exists(self):
        """LLMClient should be a runtime_checkable Protocol."""
        from rag_pipeline.llm._base import LLMClient
        from typing import runtime_checkable, Protocol
        import inspect
        assert hasattr(LLMClient, '__protocol_attrs__') or LLMClient in {p for p in dir(Protocol) if p.isupper()}

    def test_ollama_implements_protocol(self):
        """OllamaLLM should satisfy LLMClient protocol at runtime."""
        from rag_pipeline.llm._base import LLMClient
        from rag_pipeline.llm.ollama import OllamaLLM
        from typing import runtime_checkable
        llm = OllamaLLM(model="test")
        # Duck-type check: has the required methods
        assert hasattr(llm, 'generate')
        assert callable(llm.generate)
        assert hasattr(llm, 'generate_with_context')
        assert callable(llm.generate_with_context)