# LLM Integration Fix — Design Spec

**Date:** 2026-04-25
**Goal:** Fix LLM integration clarity and provider pluggability in `databricks-docs-rag-pipeline`

---

## Problems

1. `provider` field in `LLMConfig` is decorative — never read
2. No LLM provider abstraction — hardcoded to Ollama
3. `strict_prompt` hardcoded in `rag.py` — cannot be customized per deployment or per provider
4. `generate_with_context` in `ollama.py` is dead code — never called
5. `docs/integrations.md` is integration-centric, not LLM-integration-centric — no LLM call chain shown
6. `tool.py` double-hops (Tool → REST API → LLM) with no direct call option
7. Lambda cold starts load everything every call — no reuse of loaded pipeline

---

## Decisions

1. **LLM Provider Abstraction:** Option C — Factory pattern with per-provider override possible
2. **strict_prompt:** Configurable per-provider in `config.yaml` (shared default + per-provider override)
3. **Tool/MCP direct call:** Add direct `RAGPipeline` call when REST API server not needed
4. **Lambda cold start:** Module-level cache to reuse loaded pipeline across warm invocations
5. **Docs diagram:** ASCII in markdown

---

## Architecture

```
User Question
    ↓
[Step 1: Retrieval]
  question → EmbeddingModel.encode_single() → FAISSIndex.search()
    ↓
[Step 2: Generation]
  chunks + strict_prompt → LLMClient.generate() → answer
    ↓
{sources}
```

### LLM Provider Pattern

```
config.yaml
  llm:
    provider: "anthropic"          # factory selects class
    model: "claude-3-5-sonnet-latest"
    api_key: "sk-ant-..."
    strict_prompt: |             # per-provider override
      Anthropic-optimized prompt...
```

```
rag_pipeline/llm/
├── __init__.py               # exports LLMFactory, strict_prompt getter
├── _base.py                # LLMClient ABC (Protocol)
├── ollama.py              # OllamaLLM
├── anthropic.py           # AnthropicLLM (new)
├── openai.py             # OpenAILLM (new)
├── bedrock.py            # AWS Bedrock (new, lightweight)
└── _factory.py            # LLMFactory
```

**Factory behavior:**
- `provider: "ollama"` → `OllamaLLM`
- `provider: "anthropic"` → `AnthropicLLM`
- `provider: "openai"` → `OpenAILLM`
- `provider: "bedrock"` → `BedrockLLM`
- Unknown provider → raises clear error listing available options

**Per-provider strict_prompt:**
- Each client reads its own `strict_prompt` from config
- Falls back to shared `llm.strict_prompt` default
- Falls back to built-in default if neither set

### Tool/MCP Direct Call

```python
class ToolDefinition:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url   # None = direct call

    def execute(self, question: str) -> dict:
        if self.api_url:
            # REST API path
            return requests.post(f"{self.api_url}/tool/execute", json={"question": question}).json()
        else:
            # Direct path — no server needed
            from rag_pipeline.pipeline.rag import RAGPipeline
            pipeline = RAGPipeline()
            pipeline.load()
            return pipeline.query(question)
```

### Lambda Module Cache

```python
# Module-level singleton
_pipeline = None

def handler(event, context):
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
        _pipeline.load()  # cold start: ~30s
    return _pipeline.query(event["question"])  # warm: ~50ms
```

---

## File Changes

### New files
- `src/rag_pipeline/llm/_base.py` — `LLMClient` Protocol
- `src/rag_pipeline/llm/_factory.py` — `LLMFactory`
- `src/rag_pipeline/llm/anthropic.py` — `AnthropicLLM`
- `src/rag_pipeline/llm/openai.py` — `OpenAILLM`
- `src/rag_pipeline/llm/bedrock.py` — `BedrockLLM` (lightweight)

### Modified files
- `src/rag_pipeline/llm/ollama.py` — implement `LLMClient` protocol, remove `generate_with_context` dead code
- `src/rag_pipeline/llm/__init__.py` — export `LLMFactory`, `strict_prompt` getter
- `src/rag_pipeline/config.py` — add `LLMConfig.provider`, `llm.strict_prompt`, `llm.api_key`
- `src/rag_pipeline/pipeline/rag.py` — use `LLMFactory` instead of hardcoded `OllamaLLM`
- `src/rag_pipeline/integrations/tool.py` — add direct call pattern
- `src/rag_pipeline/integrations/lambda_handler.py` — add module-level cache
- `config.yaml` — add `provider`, `strict_prompt` field
- `docs/integrations.md` — LLM-integration-centric rewrite with ASCII diagrams
- `docs/guide.md` — add Two-Step RAG architecture diagram

### Deleted code
- `src/rag_pipeline/llm/ollama.py`: Remove `generate_with_context` method (dead code, never called)

### Note on new LLM providers
`AnthropicLLM`, `OpenAILLM`, and `BedrockLLM` implement the HTTP API call pattern. They are functional but **API credentials are required for live use**. For unit testing, they are mocked at the HTTP layer. Do NOT add real API keys to `config.yaml` — use environment variables:

```yaml
llm:
  api_key: ""          # read from ANTHROPIC_API_KEY env var
  aws_access_key: ""   # read from AWS_ACCESS_KEY_ID env var
```

Each client reads the env var if the config field is empty. This keeps secrets out of the codebase.

---

## strict_prompt Default

Shared default (used when provider doesn't override):

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

Per-provider overrides in config.yaml use this same template with provider-specific phrasing.

---

## Testing

- Unit tests for each LLM client (mocked at HTTP layer)
- Factory returns correct client class per provider string
- Per-provider strict_prompt override works
- ToolDefinition direct call works (no API server)
- Lambda module cache reuses pipeline across calls
- All existing tests pass (62 tests)

---

## Docs Rewrite Plan

`docs/integrations.md` sections (one per integration):

```
## N. [Integration Name]

**When to use:** [one sentence]

**LLM call chain:**
```
user question
  → EmbeddingModel.encode_single()
  → FAISSIndex.search()
  → LLMClient.generate(strict_prompt)
  → LLM API call
  → answer
```

**Code example:** [minimal, complete]
**Config fields:** [LLMConfig fields used]
```

Each section shows the full data flow with the strict_prompt visible.