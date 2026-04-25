# Databricks Docs RAG Pipeline - Implementation Plan

> **Goal:** Build a modular, configurable RAG pipeline that integrates with any LLM setup via REST API, Tool/MCP, LangChain, Lambda, or CLI.

**Architecture:** Config-driven integration layer with pluggable backends.

---

## Part 1: Project Structure

### Directories and Files

```
databricks-docs-rag-pipeline/
├── pyproject.toml                      # Package metadata (pip install)
├── setup.py                           # Package installer
├── requirements.txt                   # Python dependencies
├── requirements-dev.txt                # Dev dependencies (pytest, etc.)
├── config.yaml                        # All configuration
├── README.md                          # Quick start guide
├── CONTRIBUTING.md                     # Contribution guide
├── LICENSE                            # MIT License
├── docs/
│   ├── guide.md                       # Full documentation
│   ├── integrations.md                # Integration guide
│   └── api.md                         # API reference
├── src/
│   └── rag_pipeline/
│       ├── __init__.py               # Package init
│       ├── __version__.py             # Version
│       ├── config.py                  # Config loader
│       ├── core/
│       │   ├── __init__.py
│       │   ├── ingest.py             # Data ingestion
│       │   ├── chunking.py           # Text chunking
│       │   ├── embed.py              # Embeddings
│       │   ├── vectorstore.py         # FAISS store
│       │   └── models.py             # Data models
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── rag.py                 # Main RAG pipeline
│       │   └── keyword_detector.py     # Keyword detection
│       ├── integrations/
│       │   ├── __init__.py
│       │   ├── rest_api.py             # FastAPI server
│       │   ├── tool.py               # Tool/MCP definition
│       │   ├── langchain.py         # LangChain tool
│       │   ├── lambda_handler.py    # AWS Lambda
│       │   └── cli.py                # CLI wrapper
│       └── llm/
│           ├── __init__.py
│           └── ollama.py              # LLM client
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_core.py
│   ├── test_pipeline.py
│   ├── test_integrations.py
│   └── conftest.py                    # Pytest fixtures
└── data/                             # Data directory (gitignored)
    ├── docs_raw.jsonl
    ├── docs_chunks.jsonl
    ├── embeddings.npy
    └── delta_lake.index
```

---

## Part 2: Configuration System

### config.yaml Structure

```yaml
# ============================================
# RAG Pipeline Configuration
# ============================================

# ---------- DATA SOURCES ----------
data_sources:
  # Add any documentation sources here
  - type: "url"                      # URL-based docs
    url: "https://docs.databricks.com/en/delta/index.html"
    name: "Delta Lake Docs"
  - type: "local"                   # Local files
    path: "./docs/custom/"
    name: "Custom Docs"

# ---------- INGESTION ----------
ingestion:
  chunk_size_tokens: 256
  chunk_overlap_tokens: 50
  min_chunk_length: 50

# ---------- EMBEDDINGS ----------
embeddings:
  model: "sentence-transformers/all-mpnet-base-v2"
  batch_size: 64
  normalize: true

# ---------- VECTOR STORE ----------
vectorstore:
  type: "faiss"                      # faiss, chroma, pgvector
  index_type: "flat"                # flat, ivf, hnsw
  metric: "cosine"                  # cosine, euclidean

# ---------- RETRIEVAL ----------
retrieval:
  top_k: 5
  min_similarity: 0.3

# ---------- LLM ----------
llm:
  provider: "ollama"               # ollama, openai, anthropic, bedrock
  model: "qwen3.5:cloud"
  base_url: "http://localhost:11434"
  temperature: 0.2
  max_tokens: 512

# ---------- INTEGRATIONS ----------
integrations:
  rest_api:
    enabled: true
    host: "0.0.0.0"
    port: 8000
  
  tool:
    enabled: true
    auto_trigger: true
    keywords:
      - "delta lake"
      - "delta table"
      - "databricks"
      - "lakeflow"
      - "spark sql"
      - "pyspark"
      - "lakehouse"
      - "create table"
      - "merge into"
      - "upsert"
  
  langchain:
    enabled: false
  
  lambda:
    enabled: false
  
  cli:
    enabled: true

# ---------- LOGGING ----------
logging:
  level: "INFO"
  file: "rag_pipeline.log"
```

---

## Part 3: Implementation Tasks

### Task 1: Package Setup

**Files:**
- Create: `pyproject.toml`
- Create: `setup.py`
- Create: `requirements.txt`
- Create: `requirements-dev.txt`

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "databricks-docs-rag-pipeline"
version = "0.1.0"
description = "Modular RAG pipeline for Databricks documentation"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your@email.com"}
]
requires-python = ">=3.10"
dependencies = [
    "sentence-transformers>=2.2.0",
    "faiss-cpu>=1.9.0",
    "fastapi>=0.100.0",
    "uvicorn>=0.22.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
    "requests>=2.31.0",
    "tiktoken>=0.5.0",
    "beautifulsoup4>=4.12.0",
    "numpy>=1.24.0",
]
```

```txt
# requirements.txt
sentence-transformers>=2.2.2
faiss-cpu>=1.9.0
fastapi>=0.100.0
uvicorn>=0.22.0
pydantic>=2.0.0
pyyaml>=6.0
requests>=2.31.0
tiktoken>=0.5.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
numpy>=1.24.0
torch>=2.0.0
```

### Task 2: Core Modules

**Files:**
- Create: `src/rag_pipeline/__init__.py`
- Create: `src/rag_pipeline/__version__.py`
- Create: `src/rag_pipeline/config.py`
- Create: `src/rag_pipeline/core/__init__.py`
- Create: `src/rag_pipeline/core/models.py`
- Create: `src/rag_pipeline/core/ingest.py`
- Create: `src/rag_pipeline/core/chunking.py`
- Create: `src/rag_pipeline/core/embed.py`
- Create: `src/rag_pipeline/core/vectorstore.py`

```python
# src/rag_pipeline/config.py
"""Configuration loader."""
import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class DataSourcesConfig(BaseModel):
    type: str
    url: Optional[str] = None
    path: Optional[str] = None
    name: str


class EmbeddingsConfig(BaseModel):
    model: str = "sentence-transformers/all-mpnet-base-v2"
    batch_size: int = 64
    normalize: bool = True


class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "qwen3.5:cloud"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 512
    api_key: Optional[str] = None


class RestAPIConfig(BaseModel):
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8000


class ToolConfig(BaseModel):
    enabled: bool = True
    auto_trigger: bool = True
    keywords: list[str] = []


class IntegrationsConfig(BaseModel):
    rest_api: RestAPIConfig = RestAPIConfig()
    tool: ToolConfig = ToolConfig()
    langchain: dict = {}
    lambda: dict = {}
    cli: dict = {}


class Config(BaseModel):
    data_sources: list[DataSourcesConfig] = []
    ingestion: dict = {}
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    vectorstore: dict = {}
    retrieval: dict = {}
    llm: LLMConfig = LLMConfig()
    integrations: IntegrationsConfig = IntegrationsConfig()
    logging: dict = {}


_config: Optional[Config] = None


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f)
            return Config(**data)
    return Config()


def get_config() -> Config:
    """Get global config instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
```

### Task 3: Core RAG Pipeline

**Files:**
- Create: `src/rag_pipeline/pipeline/__init__.py`
- Create: `src/rag_pipeline/pipeline/rag.py`
- Create: `src/rag_pipeline/pipeline/keyword_detector.py`

```python
# src/rag_pipeline/pipeline/rag.py
"""Main RAG Pipeline."""
from typing import Optional

from ..config import get_config, Config
from ..core.embed import EmbeddingModel
from ..core.vectorstore import FAISSIndex
from ..core.chunking import chunk_document
from ..llm.ollama import OllamaLLM


class RAGPipeline:
    """Two-step RAG pipeline to prevent hallucination."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.embedding_model = None
        self.index = None
        self.llm = None
    
    def load(self):
        """Load all components."""
        self.embedding_model = EmbeddingModel(
            model_name=self.config.embeddings.model
        )
        self.index = FAISSIndex.load(
            self.config.vectorstore.get("index_path", "data/delta_lake.index"),
            self.config.vectorstore.get("metadata_path", "data/vectometa.jsonl")
        )
        self.llm = OllamaLLM(
            model=self.config.llm.model,
            base_url=self.config.llm.base_url,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens
        )
    
    def query(self, question: str) -> dict:
        """Query the RAG pipeline."""
        # Step 1: Retrieve
        query_embedding = self.embedding_model.encode([question])
        distances, indices = self.index.search(query_embedding, k=self.config.retrieval.get("top_k", 5))
        
        chunks = [self.index.get_chunk(idx) for idx in indices[0] if idx >= 0]
        context = "\n\n".join([chunk["text"] for chunk in chunks])
        
        # Step 2: Generate with strict prompt
        strict_prompt = f"""Answer using ONLY the context below.
If context doesn't have answer, say "I don't know".

Context:
{context}

Question: {question}

Answer:"""
        
        answer = self.llm.generate(strict_prompt)
        
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {"text": chunk["text"][:200], "score": distances[0][i]}
                for i, chunk in enumerate(chunks)
            ]
        }
```

### Task 4: Integrations

**Files:**
- Create: `src/rag_pipeline/integrations/__init__.py`
- Create: `src/rag_pipeline/integrations/rest_api.py`
- Create: `src/rag_pipeline/integrations/tool.py`
- Create: `src/rag_pipeline/integrations/langchain.py`
- Create: `src/rag_pipeline/integrations/lambda_handler.py`
- Create: `src/rag_pipeline/integrations/cli.py`

```python
# src/rag_pipeline/integrations/rest_api.py
"""FastAPI REST API server."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from ..config import get_config
from ..pipeline.rag import RAGPipeline


app = FastAPI(title="Databricks RAG Pipeline API")
pipeline = None


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class Source(BaseModel):
    text: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]
    latency_ms: float


@app.on_event("startup")
async def startup():
    global pipeline
    pipeline = RAGPipeline()
    pipeline.load()


@app.get("/health")
def health():
    return {"status": "healthy", "pipeline_loaded": pipeline is not None}


@app.post("/rag", response_model=QueryResponse)
def query(request: QueryRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    
    result = pipeline.query(request.question)
    
    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
        latency_ms=0.0
    )


def main():
    import uvicorn
    config = get_config()
    uvicorn.run(
        app,
        host=config.integrations.rest_api.host,
        port=config.integrations.rest_api.port
    )


if __name__ == "__main__":
    main()
```

### Task 5: Tests

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`
- Create: `tests/test_pipeline.py`
- Create: `tests/test_integrations.py`

```python
# tests/conftest.py
"""Pytest fixtures."""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_pipeline.config import Config


@pytest.fixture
def config():
    """Default test config."""
    return Config()


@pytest.fixture
def sample_chunks():
    """Sample document chunks."""
    return [
        {"text": "Delta Lake is the optimized storage layer.", "doc_id": 1, "chunk_id": 0},
        {"text": "Databricks uses Delta Lake by default.", "doc_id": 1, "chunk_id": 1},
    ]
```

### Task 6: Documentation

**Files:**
- Create: `README.md`
- Create: `docs/guide.md`
- Create: `docs/integrations.md`
- Create: `docs/api.md`

```markdown
# Databricks Docs RAG Pipeline

A modular, configurable RAG pipeline for Databricks documentation.

## Quick Start

### Installation

```bash
pip install databricks-docs-rag-pipeline
```

### Usage

```python
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.load()

result = pipeline.query("How to create a Delta table?")
print(result["answer"])
```

## Integration Options

| Integration | Description |
|------------|-------------|
| [REST API](docs/integrations.md#rest-api) | HTTP server |
| [Tool/MCP](docs/integrations.md#toolmcp) | Auto-trigger tool |
| [LangChain](docs/integrations.md#langchain) | LangChain tool |
| [Lambda](docs/integrations.md#lambda) | AWS Lambda |
| [CLI](docs/integrations.md#cli) | Command line |

See [docs/guide.md](docs/guide.md) for full documentation.
```

---

## Part 4: GitHub Setup

### Steps

1. Initialize git repo
2. Create GitHub repo
3. Push code
4. Create release

```bash
# Initialize
git init
git add .
git commit -m "feat: initial modular RAG pipeline

- Modular config-driven architecture
- REST API integration
- Tool/MCP integration
- LangChain integration
- Lambda handler
- CLI wrapper"

# Create GitHub repo (requires gh CLI)
gh repo create databricks-docs-rag-pipeline --public --push

# Tag and release
git tag v0.1.0
git push origin v0.1.0
```

---

## Part 5: Execution Order

| Order | Task | Description |
|-------|------|-------------|
| 1 | Package Setup | Create `pyproject.toml`, `requirements.txt` |
| 2 | Core Config | Create config loader |
| 3 | Core Modules | Create `core/` modules |
| 4 | RAG Pipeline | Create `pipeline/` module |
| 5 | Integrations | Create `integrations/` modules |
| 6 | Tests | Create test suite |
| 7 | Documentation | Create README, docs |
| 8 | GitHub Push | Push to GitHub |
| 9 | Package Test | Test pip install |

---

## Plan Complete

**Ready for execution?** 

1. **Subagent-Driven** - I dispatch subagents per task, fast iteration
2. **Inline** - Execute here with checkpoints