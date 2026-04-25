# LLM Integration Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LLM provider abstraction (factory pattern), configurable strict_prompt per provider, tool/MCP direct call, Lambda cold start cache, and rewrite docs to show the full LLM call chain.

**Architecture:** RAG retrieval (Step 1) stays identical. LLM generation (Step 2) swaps via factory. Each LLM client implements `LLMClient` Protocol, reads its own strict_prompt from config (fallback to shared default), and calls its provider API. Tool/MCP detects api_url: if set, call REST API; if None, call RAGPipeline directly. Lambda uses module-level singleton for warm invocations.

**Tech Stack:** Python 3.10+, Pydantic v2, requests, boto3 (for Bedrock), pytest

---

## File Map

| Action | File | Role |
|--------|------|------|
| Create | `src/rag_pipeline/llm/_base.py` | `LLMClient` Protocol |
| Create | `src/rag_pipeline/llm/_factory.py` | `LLMFactory` |
| Create | `src/rag_pipeline/llm/anthropic.py` | `AnthropicLLM` |
| Create | `src/rag_pipeline/llm/openai.py` | `OpenAILLM` |
| Create | `src/rag_pipeline/llm/bedrock.py` | `BedrockLLM` |
| Modify | `src/rag_pipeline/llm/ollama.py` | Remove dead code, add strict_prompt |
| Modify | `src/rag_pipeline/llm/__init__.py` | Export `LLMFactory`, `get_strict_prompt` |
| Modify | `src/rag_pipeline/config.py` | Add provider, strict_prompt fields |
| Modify | `src/rag_pipeline/pipeline/rag.py` | Use `LLMFactory` |
| Modify | `src/rag_pipeline/integrations/tool.py` | Direct call + REST API |
| Modify | `src/rag_pipeline/integrations/lambda_handler.py` | Module cache |
| Modify | `config.yaml` | Add provider, strict_prompt |
| Modify | `docs/integrations.md` | Full rewrite |
| Modify | `docs/guide.md` | Add Two-Step RAG diagram |
| Modify | `tests/test_llm.py` | Update tests for factory + new clients |

---

## Task 1: LLMClient Protocol

**Files:**
- Create: `src/rag_pipeline/llm/_base.py`
- Test: `tests/test_llm.py` (add protocol tests)

- [ ] **Step 1: Write failing test**

```python
# tests/test_llm.py — add to existing TestLLMClientProtocol class
from rag_pipeline.llm._base import LLMClient

def test_llm_client_protocol_exists():
    """LLMClient should be a Protocol defining the interface."""
    from typing import Protocol as TypingProtocol
    from rag_pipeline.llm._base import LLMClient
    # Check it's a typing Protocol (duck-typed interface)
    assert hasattr(LLMClient, '__protocol_attrs__') or hasattr(LLMClient, '__annotations__')

def test_ollama_implements_protocol():
    """OllamaLLM should satisfy LLMClient."""
    from rag_pipeline.llm._base import LLMClient
    from rag_pipeline.llm.ollama import OllamaLLM
    llm = OllamaLLM(model="test")
    assert hasattr(llm, 'generate')
    assert callable(llm.generate)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm.py::TestLLMClientProtocol -v`
Expected: FAIL — module `_base` not found

- [ ] **Step 3: Write Protocol**

```python
"""LLMClient Protocol — duck-typed interface for all LLM providers."""
from typing import Protocol, runtime_checkable

@runtime_checkable
class LLMClient(Protocol):
    """Interface that all LLM clients must implement.

    Any class with .generate() can be used as an LLM provider.
    The factory selects the right class based on config.provider.
    """

    model: str
    temperature: float
    max_tokens: int

    def generate(self, prompt: str, system: str = None) -> str:
        """Send prompt to LLM and return response."""
        ...

    def generate_with_context(self, question: str, context: str, strict_prompt: str) -> str:
        """Generate answer using retrieved context and strict_prompt.

        Args:
            question: The user's question.
            context: Concatenated retrieved chunks from FAISS.
            strict_prompt: The anti-hallucination prompt template.

        Returns:
            LLM's answer using only the provided context.
        """
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm.py::TestLLMClientProtocol -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag_pipeline/llm/_base.py tests/test_llm.py
git commit -m "feat(llm): add LLMClient Protocol"
```

---

## Task 2: Config — provider and strict_prompt fields

**Files:**
- Modify: `src/rag_pipeline/config.py:28-38`

- [ ] **Step 1: Write failing test**

```python
# tests/test_config.py — add to TestConfigOverrides
def test_llm_provider_field(self):
    cfg = Config(llm__provider="anthropic")
    assert cfg.llm.provider == "anthropic"

def test_strict_prompt_in_llm_config(self):
    cfg = Config(llm__strict_prompt="Custom prompt")
    assert "Custom prompt" in cfg.llm.strict_prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::TestConfigOverrides::test_llm_provider_field tests/test_config.py::TestConfigOverrides::test_strict_prompt_in_llm_config -v`
Expected: FAIL — no provider field in LLMConfig

- [ ] **Step 3: Update LLMConfig in config.py**

In `src/rag_pipeline/config.py`, update the `LLMConfig` class:

```python
class LLMConfig(BaseModel):
    """LLM provider configuration."""
    model_config = ConfigDict(extra="ignore")

    provider: str = "ollama"        # NEW — factory selects client class
    model: str = "qwen3.5:cloud"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 512
    api_key: str = ""             # read from env var if empty
    strict_prompt: str = ""        # NEW — per-provider override
```

Also add a shared default getter to the module:

```python
DEFAULT_STRICT_PROMPT = """You must answer using ONLY the provided context below.

RULES:
1. Answer ONLY from the context provided
2. If the context doesn't contain the answer, say "I don't have enough information"
3. NEVER use your own knowledge or make assumptions
4. Be concise and factual
5. If code examples are in the context, include them in your answer

Context:
{context}

Question: {question}

Answer (using ONLY context above):"""
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_config.py::TestConfigOverrides -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag_pipeline/config.py tests/test_config.py
git commit -m "feat(config): add provider and strict_prompt fields to LLMConfig"
```

---

## Task 3: LLMFactory

**Files:**
- Create: `src/rag_pipeline/llm/_factory.py`
- Test: `tests/test_llm.py` (add factory tests)

- [ ] **Step 1: Write failing test**

```python
# tests/test_llm.py — add TestLLMFactory class
def test_factory_returns_ollama():
    from rag_pipeline.llm._factory import LLMFactory
    factory = LLMFactory(provider="ollama")
    client = factory.create(model="test", base_url="http://localhost:11434",
                          temperature=0.2, max_tokens=512, api_key="")
    assert type(client).__name__ == "OllamaLLM"

def test_factory_returns_anthropic():
    from rag_pipeline.llm._factory import LLMFactory
    factory = LLMFactory(provider="anthropic")
    client = factory.create(model="claude", base_url="https://api.anthropic.com",
                          temperature=0.2, max_tokens=512, api_key="sk-test")
    assert type(client).__name__ == "AnthropicLLM"

def test_factory_unknown_provider():
    from rag_pipeline.llm._factory import LLMFactory, UnknownLLMProviderError
    factory = LLMFactory(provider="unknown-provider")
    with pytest.raises(UnknownLLMProviderError) as exc:
        factory.create(model="test", base_url="http://localhost:11434",
                     temperature=0.2, max_tokens=512, api_key="")
    assert "unknown-provider" in str(exc.value)
    assert "ollama" in str(exc.value)  # error shows available providers
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm.py::TestLLMFactory -v`
Expected: FAIL — `_factory` not found

- [ ] **Step 3: Write factory**

```python
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
        client = factory.create(model="claude-3-5-sonnet-latest",
                              base_url="https://api.anthropic.com",
                              api_key="sk-ant-...")
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
                f"Available: {', '.join(_PROVIDER_MAP.keys())}"
            )
        return cls(
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            strict_prompt=strict_prompt,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm.py::TestLLMFactory -v`
Expected: PASS (if clients exist) or import errors (expected — implement clients next)

- [ ] **Step 5: Commit**

```bash
git add src/rag_pipeline/llm/_factory.py tests/test_llm.py
git commit -m "feat(llm): add LLMFactory with provider routing"
```

---

## Task 4: OllamaLLM — strict_prompt + remove dead code

**Files:**
- Modify: `src/rag_pipeline/llm/ollama.py`
- Test: existing `test_llm.py` tests should pass unchanged

- [ ] **Step 1: Review current ollama.py**

Read: `src/rag_pipeline/llm/ollama.py`

- [ ] **Step 2: Update to implement LLMClient protocol**

Remove `generate_with_context` (dead code). Add `strict_prompt` init param. Implement `generate_with_context` properly:

```python
"""Ollama LLM client."""
import os
import requests
from rag_pipeline.config import DEFAULT_STRICT_PROMPT, get_config


class OllamaLLM:
    """Ollama LLM client implementing LLMClient protocol."""

    def __init__(
        self,
        model: str = "qwen3.5:cloud",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        max_tokens: int = 512,
        api_key: str = "",      # not used for local Ollama
        strict_prompt: str = "",  # per-provider override
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        # Per-provider strict_prompt, fallback to shared default
        self._strict_prompt = strict_prompt or DEFAULT_STRICT_PROMPT

    @property
    def strict_prompt(self) -> str:
        return self._strict_prompt

    def generate(self, prompt: str, system: str = None) -> str:
        """Generate response from Ollama."""
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

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def generate_with_context(
        self, question: str, context: str, strict_prompt: str = None
    ) -> str:
        """Generate answer using retrieved context.

        Args:
            question: The user's question.
            context: Concatenated retrieved chunks from FAISS.
            strict_prompt: Override template (uses instance default if None).
        """
        template = strict_prompt or self._strict_prompt
        # Fill in the template slots
        prompt = template.format(context=context, question=question)
        return self.generate(prompt)
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_llm.py -v`
Expected: PASS (OllamaLLM tests pass, others fail until implemented)

- [ ] **Step 4: Commit**

```bash
git add src/rag_pipeline/llm/ollama.py
git commit -m "feat(llm): OllamaLLM implements LLMClient, remove dead code, add strict_prompt"
```

---

## Task 5: AnthropicLLM

**Files:**
- Create: `src/rag_pipeline/llm/anthropic.py`
- Test: `tests/test_llm.py` (add Anthropic tests)

- [ ] **Step 1: Write failing test**

```python
# tests/test_llm.py — add TestAnthropicLLM class
from unittest.mock import patch, MagicMock

def test_anthropic_generate(monkeypatch):
    """AnthropicLLM should call /v1/messages endpoint."""
    from rag_pipeline.llm.anthropic import AnthropicLLM

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "content": [{"text": "Delta Lake provides ACID transactions."}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("rag_pipeline.llm.anthropic.requests.post", return_value=mock_response) as mock_post:
        llm = AnthropicLLM(
            model="claude-3-5-sonnet-latest",
            base_url="https://api.anthropic.com",
            api_key="sk-ant-test",
            temperature=0.2,
            max_tokens=512,
        )
        result = llm.generate("What is Delta Lake?")
        assert result == "Delta Lake provides ACID transactions."
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["model"] == "claude-3-5-sonnet-latest"
        assert "x-api-key" in call_kwargs["headers"]
        assert call_kwargs["headers"]["x-api-key"] == "sk-ant-test"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm.py::TestAnthropicLLM -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write AnthropicLLM**

```python
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
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_llm.py::TestAnthropicLLM -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag_pipeline/llm/anthropic.py tests/test_llm.py
git commit -m "feat(llm): add AnthropicLLM client"
```

---

## Task 6: OpenAILLM

**Files:**
- Create: `src/rag_pipeline/llm/openai.py`
- Test: `tests/test_llm.py` (add OpenAI tests)

- [ ] **Step 1: Write failing test**

```python
def test_openai_generate(monkeypatch):
    """OpenAILLM should call OpenAI-compatible /chat/completions endpoint."""
    from rag_pipeline.llm.openai import OpenAILLM

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Delta Lake provides ACID."}}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("rag_pipeline.llm.openai.requests.post", return_value=mock_response) as mock_post:
        llm = OpenAILLM(
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            temperature=0.2,
            max_tokens=512,
        )
        result = llm.generate("What is Delta Lake?")
        assert result == "Delta Lake provides ACID."
        call_kwargs = mock_post.call_args[1]
        assert "gpt-4o" in call_kwargs["json"]["model"]
        assert "Authorization" in call_kwargs["headers"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm.py::TestOpenAILLM -v`
Expected: FAIL

- [ ] **Step 3: Write OpenAILLM**

```python
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
        headers = {
            "content-type": "application/json",
        }
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
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_llm.py::TestOpenAILLM -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag_pipeline/llm/openai.py tests/test_llm.py
git commit -m "feat(llm): add OpenAILLM client (OpenAI/OpenRouter compatible)"
```

---

## Task 7: BedrockLLM

**Files:**
- Create: `src/rag_pipeline/llm/bedrock.py`
- Test: `tests/test_llm.py` (add Bedrock tests)

- [ ] **Step 1: Write failing test**

```python
def test_bedrock_generate(monkeypatch):
    """BedrockLLM should call AWS Bedrock Converse API."""
    from rag_pipeline.llm.bedrock import BedrockLLM

    mock_converse = MagicMock()
    mock_converse.return_value = {
        "output": {"message": {"content": [{"text": "Delta Lake provides ACID."}]}}
    }

    with patch("rag_pipeline.llm.bedrock.bedrock_runtime.converse", mock_converse):
        llm = BedrockLLM(
            model="anthropic.claude-3-5-sonnet-latest",
            region="us-east-1",
            temperature=0.2,
            max_tokens=512,
        )
        result = llm.generate("What is Delta Lake?")
        assert result == "Delta Lake provides ACID."
        mock_converse.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm.py::TestBedrockLLM -v`
Expected: FAIL

- [ ] **Step 3: Write BedrockLLM**

```python
"""AWS Bedrock LLM client.

Uses AWS boto3 + Bedrock Converse API.
Reads AWS credentials from environment or boto3 default credential chain.
"""
import os
import requests
from rag_pipeline.config import DEFAULT_STRICT_PROMPT


class BedrockLLM:
    """AWS Bedrock LLM client.

    Supports Anthropic models on Bedrock (claude-3-5-sonnet, etc.)
    and other Bedrock models that implement the Converse API.

    Required env vars (or boto3 default chain):
      AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
    """

    def __init__(
        self,
        model: str = "anthropic.claude-3-5-sonnet-latest",
        region: str = "us-east-1",
        temperature: float = 0.2,
        max_tokens: int = 512,
        api_key: str = "",   # not used — uses boto3 credentials
        strict_prompt: str = "",
    ):
        self.model = model
        self.region = region
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._strict_prompt = strict_prompt or DEFAULT_STRICT_PROMPT

    @property
    def strict_prompt(self) -> str:
        return self._strict_prompt

    def generate(self, prompt: str, system: str = None) -> str:
        try:
            import boto3
            from botocore.config import Config as BotoConfig
        except ImportError:
            raise ImportError(
                "boto3 is required for Bedrock. Install: pip install boto3"
            )

        boto_config = BotoConfig(region_name=self.region)
        bedrock = boto3.client("bedrock-runtime", config=boto_config)

        messages = [{"role": "user", "content": [{"text": prompt}]}]
        if system:
            messages.insert(0, {"role": "user", "content": [{"text": system}]})

        payload = {
            "modelId": self.model,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
            },
        }

        response = bedrock.converse(**payload)
        return response["output"]["message"]["content"][0]["text"]

    def generate_with_context(
        self, question: str, context: str, strict_prompt: str = None
    ) -> str:
        template = strict_prompt or self._strict_prompt
        prompt = template.format(context=context, question=question)
        return self.generate(prompt)
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_llm.py::TestBedrockLLM -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag_pipeline/llm/bedrock.py tests/test_llm.py
git commit -m "feat(llm): add BedrockLLM client"
```

---

## Task 8: llm/__init__.py exports

**Files:**
- Modify: `src/rag_pipeline/llm/__init__.py`

- [ ] **Step 1: Write module exports**

```python
"""LLM package — pluggable LLM clients."""
from rag_pipeline.llm._base import LLMClient
from rag_pipeline.llm._factory import LLMFactory, UnknownLLMProviderError
from rag_pipeline.llm.ollama import OllamaLLM
from rag_pipeline.llm.anthropic import AnthropicLLM
from rag_pipeline.llm.openai import OpenAILLM
from rag_pipeline.llm.bedrock import BedrockLLM
from rag_pipeline.config import DEFAULT_STRICT_PROMPT

__all__ = [
    "LLMClient",
    "LLMFactory",
    "UnknownLLMProviderError",
    "OllamaLLM",
    "AnthropicLLM",
    "OpenAILLM",
    "BedrockLLM",
    "DEFAULT_STRICT_PROMPT",
]
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_llm.py -v`
Expected: PASS (all 5 LLM client tests)

- [ ] **Step 3: Commit**

```bash
git add src/rag_pipeline/llm/__init__.py
git commit -m "feat(llm): export all LLM clients and factory from llm package"
```

---

## Task 9: rag.py — use LLMFactory

**Files:**
- Modify: `src/rag_pipeline/pipeline/rag.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_pipeline.py — add
def test_rag_uses_factory():
    """RAGPipeline should use LLMFactory, not hardcoded OllamaLLM."""
    from rag_pipeline.config import Config, LLMConfig, IntegrationsConfig
    from rag_pipeline.llm._factory import LLMFactory

    cfg = Config(
        llm=LLMConfig(provider="anthropic", model="claude-test",
                     api_key="sk-test")
    )

    with patch("rag_pipeline.pipeline.rag.LLMFactory") as mock_factory:
        mock_client = MagicMock()
        mock_client.generate_with_context.return_value = "Test answer"
        mock_factory.return_value.create.return_value = mock_client

        pipeline = RAGPipeline(config=cfg)
        pipeline.llm = mock_client  # factory creates and sets it

        result = pipeline.query("test question")
        mock_client.generate_with_context.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py::test_rag_uses_factory -v`
Expected: FAIL

- [ ] **Step 3: Update rag.py load() method**

Update the `load()` method in `rag.py` to use `LLMFactory`:

```python
from rag_pipeline.llm._factory import LLMFactory

def load(self) -> None:
    self.embedding_model = EmbeddingModel(
        model_name=self.config.embeddings.model
    )
    try:
        self.index = FAISSIndex.load("data/delta_lake.index", "data/vectometa.jsonl")
    except Exception:
        self.index = None

    factory = LLMFactory(provider=self.config.llm.provider)
    self.llm = factory.create(
        model=self.config.llm.model,
        base_url=self.config.llm.base_url,
        temperature=self.config.llm.temperature,
        max_tokens=self.config.llm.max_tokens,
        api_key=self.config.llm.api_key,
        strict_prompt=self.config.llm.strict_prompt,
    )
```

- [ ] **Step 4: Update rag.py query() method**

Update the `query()` method to use `generate_with_context`:

```python
def query(self, question: str) -> Dict[str, Any]:
    if self.index is None:
        return {
            "question": question,
            "answer": "Vector index not loaded. Have you ingested documentation?",
            "sources": [],
        }

    # Step 1: Retrieve
    query_embedding = self.embedding_model.encode_single(question)
    distances, indices = self.index.search(query_embedding, k=self.config.top_k)

    chunks = []
    for idx in indices[0]:
        if idx >= 0:
            chunk = self.index.get_chunk(idx)
            chunks.append(chunk)

    context = "\n\n".join([c["text"] for c in chunks])

    if not context:
        return {
            "question": question,
            "answer": "No relevant context found.",
            "sources": [],
        }

    # Step 2: Generate — use strict_prompt from the LLM client
    answer = self.llm.generate_with_context(question, context)

    return {
        "question": question,
        "answer": answer if answer else "I don't have enough information.",
        "sources": [
            {"text": c["text"][:200], "score": float(distances[0][i])}
            for i, c in enumerate(chunks)
        ],
    }
```

- [ ] **Step 5: Run pipeline tests**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS (all pipeline tests pass)

- [ ] **Step 6: Commit**

```bash
git add src/rag_pipeline/pipeline/rag.py tests/test_pipeline.py
git commit -m "feat(pipeline): use LLMFactory for pluggable LLM providers"
```

---

## Task 10: tool.py — direct call + REST API

**Files:**
- Modify: `src/rag_pipeline/integrations/tool.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_integrations.py — add TestToolDefinitionDirectCall
def test_tool_execute_direct_no_api():
    """ToolDefinition.execute should call RAGPipeline directly when api_url is None."""
    from rag_pipeline.integrations.tool import ToolDefinition

    tool = ToolDefinition(api_url=None)

    with patch("rag_pipeline.integrations.tool.RAGPipeline") as mock_cls:
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = {
            "answer": "Delta Lake provides ACID.",
            "sources": [],
        }
        mock_cls.return_value = mock_pipeline

        result = tool.execute("What is Delta Lake?")

        assert result["answer"] == "Delta Lake provides ACID."
        mock_pipeline.query.assert_called_once_with("What is Delta Lake?")

def test_tool_execute_via_api():
    """ToolDefinition.execute should call REST API when api_url is set."""
    from rag_pipeline.integrations.tool import ToolDefinition

    tool = ToolDefinition(api_url="http://localhost:8000")

    with patch("rag_pipeline.integrations.tool.requests.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "answer": "Delta Lake provides ACID.",
            "sources": [],
        }

        result = tool.execute("What is Delta Lake?")

        mock_post.assert_called_once()
        assert "tool/execute" in mock_post.call_args[0][0]
        assert result["answer"] == "Delta Lake provides ACID."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_integrations.py::TestToolDefinitionDirectCall -v`
Expected: FAIL — direct call not implemented

- [ ] **Step 3: Update tool.py**

```python
"""Tool/MCP definition for RAG pipeline."""
import requests

from rag_pipeline.pipeline.keyword_detector import KeywordDetector


class ToolDefinition:
    """RAG Tool definition for MCP/LangChain.

    Can call the RAG pipeline in two ways:
      - Direct (api_url=None): calls RAGPipeline directly — no server needed
      - REST API (api_url set): calls the REST API server — for distributed setups
    """

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.keyword_detector = KeywordDetector()

    def get_definition(self) -> dict:
        """Get tool definition for registration."""
        return {
            "name": "search_databricks_docs",
            "description": "Search Databricks Delta Lake and Lakeflow documentation. "
            "Use when user asks about: Delta Lake, Databricks, Lakeflow, Spark SQL, "
            "data pipelines, tables, CREATE TABLE, MERGE, UPSERT, or any data engineering on Databricks.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The technical question to search documentation for",
                    }
                },
                "required": ["question"],
            },
        }

    def should_auto_trigger(self, question: str) -> bool:
        """Check if should auto-trigger."""
        return self.keyword_detector.should_use_rag(question)

    def execute(self, question: str) -> dict:
        """Execute the tool.

        Calls REST API if api_url is set, otherwise calls RAGPipeline directly.
        """
        if self.api_url:
            # REST API path — for distributed setups
            response = requests.post(
                f"{self.api_url}/tool/execute",
                json={"question": question},
                timeout=120,
            )
            response.raise_for_status()
            return response.json()
        else:
            # Direct path — no server needed, runs RAGPipeline in-process
            from rag_pipeline.pipeline.rag import RAGPipeline
            pipeline = RAGPipeline()
            pipeline.load()
            return pipeline.query(question)

    def get_keywords(self) -> list:
        """Get trigger keywords."""
        return self.keyword_detector.keywords
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_integrations.py::TestToolDefinitionDirectCall -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag_pipeline/integrations/tool.py tests/test_integrations.py
git commit -m "feat(tool): add direct RAGPipeline call when api_url is None"
```

---

## Task 11: lambda_handler.py — module-level cache

**Files:**
- Modify: `src/rag_pipeline/integrations/lambda_handler.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_integrations.py — add TestLambdaCache
def test_lambda_module_cache():
    """Lambda handler should reuse pipeline across warm invocations."""
    from rag_pipeline.integrations.lambda_handler import handler

    with patch("rag_pipeline.integrations.lambda_handler.RAGPipeline") as mock_cls:
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = {
            "answer": "Delta Lake provides ACID.",
            "sources": [],
        }
        mock_cls.return_value = mock_pipeline

        # First call — cold (loads pipeline)
        result1 = handler({"question": "What is Delta Lake?"}, None)
        assert mock_pipeline.load.call_count == 1

        # Second call — warm (reuses loaded pipeline)
        result2 = handler({"question": "What is MERGE?"}, None)
        # load() should NOT be called again
        assert mock_pipeline.load.call_count == 1
        assert mock_pipeline.query.call_count == 2
        assert result2 == result1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_integrations.py::TestLambdaCache -v`
Expected: FAIL

- [ ] **Step 3: Update lambda_handler.py**

```python
"""AWS Lambda handler for RAG pipeline.

Uses module-level cache to reuse the loaded RAGPipeline across
warm Lambda invocations — avoids paying the cold-start cost
(embedding model load) on every call.
"""
import json

# Module-level singleton — persists across warm invocations
_pipeline = None


def handler(event, context):
    """AWS Lambda entry point.

    First cold call: loads embedding model + LLM (~30s).
    Subsequent warm calls: reuse already-loaded pipeline (~50ms).
    """
    global _pipeline

    question = event.get("question") or (event.get("body") or {}).get("question")

    if not question:
        return {"statusCode": 400, "body": json.dumps({"error": "question is required"})}

    try:
        from rag_pipeline.config import get_config
        from rag_pipeline.pipeline.rag import RAGPipeline

        # Load once, reuse forever
        if _pipeline is None:
            config = get_config()
            _pipeline = RAGPipeline()
            _pipeline.load()  # cold start cost paid here

        result = _pipeline.query(question)

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result),
        }

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_integrations.py::TestLambdaCache -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag_pipeline/integrations/lambda_handler.py tests/test_integrations.py
git commit -m "feat(lambda): add module-level pipeline cache for warm invocations"
```

---

## Task 12: config.yaml — add provider and strict_prompt

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: Update config.yaml**

```yaml
# ============================================
# Databricks RAG Pipeline Configuration
# ============================================

# ---------- LLM ----------
llm:
  provider: "ollama"                  # ollama | anthropic | openai | bedrock
  model: "qwen3.5:cloud"
  base_url: "http://localhost:11434"
  temperature: 0.2
  max_tokens: 512
  api_key: ""                        # set or use env var

  # Anti-hallucination prompt — per-provider override
  # Falls back to built-in default if empty
  strict_prompt: |
    You must answer using ONLY the provided context below.

    RULES:
    1. Answer ONLY from the context provided
    2. If the context doesn't contain the answer, say "I don't have enough information"
    3. NEVER use your own knowledge or make assumptions
    4. Be concise and factual
    5. If code examples are in the context, include them in your answer

    Context:
    {context}

    Question: {question}

    Answer (using ONLY context above):
```

- [ ] **Step 2: Commit**

```bash
git add config.yaml
git commit -m "chore(config): add provider and strict_prompt fields"
```

---

## Task 13: Rewrite docs/integrations.md

**Files:**
- Modify: `docs/integrations.md`

- [ ] **Step 1: Rewrite with LLM call chain for each integration**

Replace all content with the new structure:

```markdown
# Integration Options

Each integration is a **transport wrapper** — the LLM call chain is identical for all.

## The LLM Call Chain

```
User Question
    ↓
Step 1 — Retrieval
  question → EmbeddingModel.encode_single()
  → FAISSIndex.search() → top-k chunks
    ↓
Step 2 — Generation (anti-hallucination)
  chunks → LLMClient.generate_with_context(question, context, strict_prompt)
  → LLM API call (provider varies) → answer
    ↓
{answer, sources}
```

**The strict_prompt** is the key anti-hallucination piece:

```
You must answer using ONLY the provided context below.

RULES:
1. Answer ONLY from the context provided
2. If the context doesn't contain the answer, say "I don't have enough information"
3. NEVER use your own knowledge or make assumptions
4. Be concise and factual
5. If code examples are in the context, include them in your answer

Context:
{context}

Question: {question}

Answer (using ONLY context above):
```

The LLM is forced to answer **only** from retrieved context. If no relevant context exists, it says "I don't have enough information" instead of hallucinating.

## LLM Provider Options

Switch providers in `config.yaml`:

```yaml
llm:
  provider: "ollama"       # ← change this
  model: "qwen3.5:cloud"
```

| Provider | When to use |
|----------|-----------|
| Ollama (local) | Free, offline, private |
| Anthropic Claude | Best quality, paid |
| OpenAI / OpenRouter | Free models available |
| AWS Bedrock | Enterprise, no API key management |

## 1. Python (Direct)

**When to use:** Embedded in scripts, notebooks, custom applications.

```
question → RAGPipeline.query()
  → embed → FAISS → chunks
  → LLMClient.generate_with_context()  ← LLM called here
  → answer
```

```python
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.load()  # loads once at startup
result = pipeline.query("How to create a Delta table?")
print(result["answer"])
```

**Config fields:** `llm.provider`, `llm.model`, `llm.base_url`, `llm.temperature`

---

## 2. REST API

**When to use:** Web apps, agents, distributed systems, multi-user access.

```
curl → FastAPI /rag → RAGPipeline.query()
  → embed → FAISS → chunks
  → LLMClient.generate_with_context()  ← LLM called here
  → answer
```

```bash
# Start server
python -m rag_pipeline.integrations.rest_api

# Query
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"question": "How to create a Delta table?"}'
```

**Config fields:** `integrations.rest_api.enabled`, `integrations.rest_api.port`

---

## 3. CLI

**When to use:** Local development, quick testing, scripts.

```
rag-cli "question" → RAGPipeline.query()
  → embed → FAISS → chunks
  → LLMClient.generate_with_context()  ← LLM called here
  → answer
```

```bash
rag-cli "How to create a Delta table?"
```

**Config fields:** All `llm.*` fields

---

## 4. Tool/MCP

**When to use:** AI agents, MCP-compatible tools, auto-trigger.

**Two call patterns:**

**REST API path** (when `api_url` is set — distributed):
```
agent → Tool.execute() → requests.post(/tool/execute)
  → FastAPI → RAGPipeline.query()
    → embed → FAISS → chunks
    → LLMClient.generate_with_context()  ← LLM called here
    → answer
```

**Direct path** (when `api_url=None` — local, no server needed):
```
agent → Tool.execute(api_url=None) → RAGPipeline.query()
  → embed → FAISS → chunks
  → LLMClient.generate_with_context()  ← LLM called here
  → answer
```

```python
from rag_pipeline.integrations.tool import ToolDefinition

# REST API pattern
tool = ToolDefinition(api_url="http://localhost:8000")
result = tool.execute("How to create a Delta table?")

# Direct pattern (no server needed)
tool = ToolDefinition(api_url=None)
result = tool.execute("How to create a Delta table?")
```

**Config fields:** `integrations.tool.keywords`, `integrations.tool.auto_trigger`

---

## 5. LangChain

**When to use:** LangChain agents, OpenAI function-calling agents.

```
agent → get_langchain_tool() → RAGPipeline.query()
  → embed → FAISS → chunks
  → LLMClient.generate_with_context()  ← LLM called here
  → answer
```

```python
from rag_pipeline.integrations.langchain import get_langchain_tool
from langchain.agents import AgentExecutor, create_openai_functions_agent

tool = get_langchain_tool()
# ... build agent with tool
```

**Config fields:** All `llm.*` fields

---

## 6. AWS Lambda

**When to use:** Serverless production, pay-per-use deployments.

```
API GW → handler(event) → _pipeline.query()
  → embed → FAISS → chunks
  → LLMClient.generate_with_context()  ← LLM called here
  → answer
```

Lambda uses module-level cache: first cold call loads pipeline (~30s), subsequent warm calls reuse it (~50ms).

Deploy `src/rag_pipeline/integrations/lambda_handler.py` as the Lambda handler.

**Config fields:** All `llm.*` fields

---

## Swapping LLM Providers

The only thing that changes between providers is `config.yaml`:

```yaml
# Ollama (local)
llm:
  provider: "ollama"
  model: "qwen3.5:cloud"
  base_url: "http://localhost:11434"

# Anthropic Claude
llm:
  provider: "anthropic"
  model: "claude-3-5-sonnet-latest"
  base_url: "https://api.anthropic.com"
  api_key: "sk-ant-..."          # from ANTHROPIC_API_KEY env var

# OpenRouter (free models)
llm:
  provider: "openai"
  model: "deepseek-ai/DeepSeek-V3"
  base_url: "https://openrouter.ai/api/v1"
  api_key: "sk-or-..."           # from OPENAI_API_KEY env var

# AWS Bedrock
llm:
  provider: "bedrock"
  model: "anthropic.claude-3-5-sonnet-latest"
  region: "us-east-1"            # AWS credentials from boto3 chain
```

The RAG retrieval (Step 1) is identical for all providers. Only the LLM API call (Step 2) changes.
```

- [ ] **Step 2: Commit**

```bash
git add docs/integrations.md
git commit -m "docs: rewrite integrations.md showing LLM call chain for each option"
```

---

## Task 14: docs/guide.md — Two-Step RAG diagram

**Files:**
- Modify: `docs/guide.md`

- [ ] **Step 1: Add architecture diagram after Overview section**

Add after the "## Overview" section:

```markdown
## Workflow

```
User Question: "How to create a Delta table?"
    ↓
Step 1: Embedding + Retrieval
  → EmbeddingModel.encode_single(question)
  → FAISSIndex.search(query_embedding, k=5)
  → top-5 relevant chunks
    ↓
Step 2: Generate with Strict Prompt
  → chunks formatted into strict_prompt template
  → LLMClient.generate_with_context(question, context, strict_prompt)
  → LLM API call (provider from config.yaml)
  → answer using ONLY the retrieved context
    ↓
{sources: [...], answer: "..."}
```

**Anti-hallucination mechanism:** The `strict_prompt` forces the LLM to answer only from retrieved context. If no relevant context exists, the LLM says "I don't have enough information" instead of hallucinating.

See [docs/integrations.md](integrations.md) for the full LLM call chain per integration option.
```

- [ ] **Step 2: Commit**

```bash
git add docs/guide.md
git commit -m "docs: add Two-Step RAG architecture diagram to guide.md"
```

---

## Task 15: Final test run

**Files:**
- All tests

- [ ] **Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: all 62+ tests pass (62 existing + new LLM client tests)

- [ ] **Final commit**

```bash
git add -A && git commit -m "feat: complete LLM integration — factory pattern, per-provider strict_prompt, direct tool call, Lambda cache, docs rewrite"
```

---

## Spec Coverage Check

| Spec requirement | Task |
|----------------|------|
| LLM provider abstraction (factory) | Task 1, 3 |
| Per-provider strict_prompt | Task 2, 4, 5, 6, 7 |
| Tool/MCP direct call | Task 10 |
| Lambda cold start cache | Task 11 |
| Config update | Task 12 |
| Docs rewrite | Task 13, 14 |
| Remove dead code (generate_with_context) | Task 4 |
| Tests | All tasks (TDD per step) |