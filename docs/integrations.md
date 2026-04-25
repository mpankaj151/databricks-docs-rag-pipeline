# Integration Options

Each integration is a **transport wrapper** — the LLM call chain is identical for all.

## The LLM Call Chain

```
User Question
    ↓
Step 1 — Embedding + Retrieval
  question → EmbeddingModel.encode_single()
  → FAISSIndex.search() → top-k chunks
    ↓
Step 2 — Generation (anti-hallucination)
  chunks → LLMClient.generate_with_context(question, context, strict_prompt)
  → LLM API call (provider varies) → answer
    ↓
{answer, sources}
```

**The strict_prompt** is the key anti-hallucination piece — it forces the LLM to answer only from retrieved context:

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

If no relevant context exists, the LLM says "I don't have enough information" instead of hallucinating.

## LLM Provider Options

Switch providers by changing one line in `config.yaml`:

```yaml
llm:
  provider: "ollama"       # ← change this
  model: "qwen3.5:cloud"
```

| Provider | When to use |
|----------|-------------|
| Ollama (local) | Free, offline, private |
| Anthropic Claude | Best quality, paid |
| OpenAI / OpenRouter | Free models available |
| AWS Bedrock | Enterprise, no API key management |

## 1. Python (Direct)

**When to use:** Scripts, notebooks, custom applications. No server needed.

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

## 4. Tool/MCP

**When to use:** AI agents, MCP-compatible tools, auto-trigger.

**Two call patterns:**

**REST API path** (when `api_url` is set):
```
agent → Tool.execute() → requests.post(/tool/execute)
  → FastAPI → RAGPipeline.query()
    → embed → FAISS → chunks
    → LLMClient.generate_with_context()  ← LLM called here
    → answer
```

**Direct path** (when `api_url=None` — no server needed):
```
agent → Tool.execute(api_url=None) → RAGPipeline.query()
  → embed → FAISS → chunks
  → LLMClient.generate_with_context()  ← LLM called here
  → answer
```

```python
from rag_pipeline.integrations.tool import ToolDefinition

# REST API pattern (requires server running)
tool = ToolDefinition(api_url="http://localhost:8000")
result = tool.execute("How to create a Delta table?")

# Direct pattern (no server needed)
tool = ToolDefinition(api_url=None)
result = tool.execute("How to create a Delta table?")
```

**Config fields:** `integrations.tool.keywords`, `integrations.tool.auto_trigger`

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
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.load()
tool = get_langchain_tool(rag_pipeline=pipeline)
# ... build agent with tool
```

**Config fields:** All `llm.*` fields

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
  base_url: "us-east-1"          # AWS credentials from boto3 chain
```

The RAG retrieval (Step 1) is identical for all providers. Only the LLM API call (Step 2) changes.