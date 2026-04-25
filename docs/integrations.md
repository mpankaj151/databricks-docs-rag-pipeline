# Integration Options

This pipeline supports five integration patterns. All share the same RAG engine — pick the one that fits your infrastructure.

## 1. REST API (FastAPI)

**Best for**: Web applications, agents, webhooks, multi-user access

### Start the server

```bash
python -m rag_pipeline.integrations.rest_api
```

Server starts on `http://0.0.0.0:8000` (configurable in `config.yaml`).

### Endpoints

| Method | Path | Description |
|-------|------|-------------|
| `GET` | `/` | API info |
| `GET` | `/health` | Health check |
| `GET` | `/tool` | Tool definition for LLM integration |
| `POST` | `/tool/execute` | Execute tool directly |
| `POST` | `/rag` | Query with RAG |

### Query

```bash
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"question": "How to create a Delta table?"}'
```

### Python client

```python
import requests

response = requests.post(
    "http://localhost:8000/rag",
    json={"question": "How to create a Delta table?"}
)
result = response.json()
print(result["answer"])
```

---

## 2. CLI

**Best for**: Local development, scripts, quick testing

### Install

The CLI is automatically installed as `rag-cli` when you `pip install -e .`

### Usage

```bash
# Direct query (loads pipeline in-process)
rag-cli "How to create a Delta table?"

# Use REST API (requires server running)
rag-cli "How to create a Delta table?" --use-api

# Custom API URL
rag-cli "How to create a Delta table?" --api-url http://localhost:9000
```

### Programmatic

```python
from rag_pipeline.integrations.cli import main
main()
```

---

## 3. Tool/MCP

**Best for**: AI agents, MCP-compatible tools, auto-trigger in LLM applications

### Tool Definition

```python
from rag_pipeline.integrations.tool import ToolDefinition

tool = ToolDefinition(api_url="http://localhost:8000")
definition = tool.get_definition()
# Returns JSON schema for tool registration
```

### Auto-Trigger

The tool auto-triggers when keywords are detected:

```python
tool = ToolDefinition()
if tool.should_auto_trigger("Tell me about Delta Lake merge"):
    result = tool.execute("Tell me about Delta Lake merge")
```

### MCP Server Integration

The `/tool` endpoint returns a JSON schema compatible with MCP servers:

```json
{
  "tool": {
    "name": "search_databricks_docs",
    "description": "Search Databricks Delta Lake documentation",
    "input_schema": {
      "type": "object",
      "properties": {
        "question": {"type": "string"}
      }
    }
  }
}
```

---

## 4. LangChain

**Best for**: LangChain agents, OpenAI function-calling agents

### Setup

```bash
pip install -e ".[langchain]"
```

### Usage

```python
from rag_pipeline.integrations.langchain import get_langchain_tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI

# Initialize pipeline
from rag_pipeline import RAGPipeline
pipeline = RAGPipeline()
pipeline.load()

# Create tool
tool = get_langchain_tool(rag_pipeline=pipeline)

# Build agent
llm = ChatOpenAI(model="gpt-4", temperature=0)
agent = create_openai_functions_agent(llm, [tool], prompt)
executor = AgentExecutor(agent=agent, tools=[tool])

# Run
result = executor.invoke({"input": "How to create a Delta table?"})
```

### LangChain Tool Properties

| Property | Value |
|----------|-------|
| Name | `search_databricks_docs` |
| Description | Search Databricks Delta Lake documentation |
| Input | `question` (string) |

---

## 5. AWS Lambda

**Best for**: Serverless production deployments

### Deploy

1. Package the source: `pip install -t package -e .`
2. Zip and upload to AWS Lambda
3. Set handler to `rag_pipeline.integrations.lambda_handler.handler`

### Event Format

```json
{
  "question": "How to create a Delta table?"
}
```

### Response

```json
{
  "statusCode": 200,
  "headers": {"Content-Type": "application/json"},
  "body": "{\"question\": \"How to create a Delta table?\", \"answer\": \"...\", \"sources\": [...]}"
}
```

### Notes

- Cold start: ~30s (first load of embedding model + LLM)
- Use provisioned concurrency for production
- Vector index (`data/delta_lake.index`) must be included in the deployment package

---

## Comparison

| Integration | Cold Start | Stateful | Best For |
|------------|-----------|----------|---------|
| REST API | Fast (after first load) | Yes (in-memory pipeline) | Production web services |
| CLI | Fast | No (per-call load) | Local scripts, testing |
| Tool/MCP | Fast | Yes (if API server running) | AI agents |
| LangChain | Medium | Yes | LangChain agent frameworks |
| Lambda | Slow (~30s) | No (stateless) | Serverless, pay-per-use |

## Switching Integrations

Since all integrations share the same `RAGPipeline`, switching is purely a configuration change — no code changes needed:

```yaml
# config.yaml — enable/disable integrations
integrations:
  rest_api:
    enabled: true    # REST API
  tool:
    enabled: true   # Tool/MCP
  langchain:
    enabled: false
  lambda:
    enabled: false
  cli:
    enabled: true
```