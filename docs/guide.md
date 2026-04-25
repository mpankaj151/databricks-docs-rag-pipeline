# Databricks Docs RAG Pipeline — User Guide

## Overview

This pipeline answers questions about Databricks Delta Lake documentation using a RAG (Retrieval-Augmented Generation) approach. It retrieves relevant documentation chunks and generates answers using a local LLM (qwen3.5:cloud via Ollama).

**Key design principle**: The LLM is forced to answer using ONLY retrieved context. This is the enterprise-standard "Two-Step RAG" pattern that prevents hallucination.

## Workflow

```
User Question
    ↓
Step 1: Embed question → FAISS vector search → retrieve top-k chunks
    ↓
Step 2: Send chunks + strict prompt → LLM → answer
    ↓
Return { question, answer, sources }
```

## Setup

### Prerequisites

- Python 3.10+
- Ollama running (`ollama serve`) with `qwen3.5:cloud` model
- 64GB RAM recommended for `all-mpnet-base-v2` (768-dim embeddings)

### Installation

```bash
git clone <repo>
cd databricks-docs-rag-pipeline
pip install -e .
```

### Configuration

Edit `config.yaml`:

```yaml
llm:
  model: "qwen3.5:cloud"        # Your Ollama model
  base_url: "http://localhost:11434"

embeddings:
  model: "sentence-transformers/all-mpnet-base-v2"
```

## Usage

### REST API (Recommended for production)

```bash
# Start server
python -m rag_pipeline.integrations.rest_api

# Query
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"question": "How to create a Delta table?"}'
```

Response:
```json
{
  "question": "How to create a Delta table?",
  "answer": "CREATE TABLE Delta Lake table...",
  "sources": [{"text": "...", "score": 0.92}],
  "latency_ms": 0.0
}
```

### CLI (Recommended for local testing)

```bash
rag-cli "How to create a Delta table?"
```

### Python (Recommended for scripts)

```python
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.load()

result = pipeline.query("How to create a Delta table?")
print(result["answer"])
```

### LangChain

```python
from rag_pipeline.integrations.langchain import get_langchain_tool
from langchain.agents import AgentExecutor, create_openai_functions_agent

tool = get_langchain_tool()
# ... build agent with tool
```

### AWS Lambda

Deploy `src/rag_pipeline/integrations/lambda_handler.py` to AWS Lambda.

Event format: `{"question": "How to create a Delta table?"}`

## Data Ingestion

To ingest your own documentation:

```python
from rag_pipeline.core.ingest import fetch_databricks_docs, load_raw_docs, ingest_documents
from rag_pipeline.core import EmbeddingModel, FAISSIndex
import numpy as np

# 1. Fetch docs
docs = fetch_databricks_docs("https://docs.databricks.com/en/delta/index.html")

# 2. Chunk
chunks = ingest_documents(docs, chunk_size=256, overlap=50)

# 3. Embed
model = EmbeddingModel()
vectors = model.encode([c["text"] for c in chunks])

# 4. Build index
index = FAISSIndex(dimension=model.get_dimension())
index.build(vectors, chunks)
index.save("data/delta_lake.index", "data/vectometa.jsonl")
```

## Configuration Reference

| Field | Default | Description |
|-------|---------|-------------|
| `embeddings.model` | `all-mpnet-base-v2` | Sentence-transformers model |
| `llm.model` | `qwen3.5:cloud` | Ollama model name |
| `llm.temperature` | `0.2` | LLM temperature (lower = more focused) |
| `chunk_size_tokens` | `256` | Target chunk size in tokens |
| `chunk_overlap_tokens` | `50` | Token overlap between chunks |
| `top_k` | `5` | Number of chunks to retrieve |

## Keywords Auto-Trigger

The pipeline auto-triggers when these keywords are detected:

```
delta lake, delta table, databricks, lakeflow, spark sql,
pyspark, lakehouse, create table, merge into, upsert
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Empty answer | Run ingestion to build the vector index first |
| LLM timeout | Ensure Ollama is running (`ollama serve`) |
| Import errors | Reinstall: `pip install -e .` |
| Memory errors | Use `all-MiniLM-L6-v2` (384-dim) instead of `all-mpnet-base-v2` |