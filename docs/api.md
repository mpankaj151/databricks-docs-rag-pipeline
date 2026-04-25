# API Reference

## Package: `rag_pipeline`

### Top-Level Imports

```python
from rag_pipeline import (
    RAGPipeline,     # Main pipeline
    Config,          # Configuration model
    get_config,      # Load global config
    KeywordDetector, # Keyword auto-trigger
    EmbeddingModel, # Embedding wrapper
    FAISSIndex,      # Vector store
    OllamaLLM,       # LLM client
    __version__,
)
```

---

## Class: `RAGPipeline`

The main pipeline class implementing Two-Step RAG.

### `__init__(config=None)`

Create a pipeline instance.

```python
pipeline = RAGPipeline(config=None)  # Uses global config
```

### `load()`

Load the embedding model, vector index, and LLM. Call once at startup.

```python
pipeline.load()
```

### `query(question: str) -> dict`

Query the pipeline.

```python
result = pipeline.query("How to create a Delta table?")
# Returns:
# {
#     "question": "How to create a Delta table?",
#     "answer": "To create a Delta table...",
#     "sources": [{"text": "...", "score": 0.92}]
# }
```

---

## Class: `EmbeddingModel`

Wrapper for sentence-transformers.

### `__init__(model_name="sentence-transformers/all-mpnet-base-v2")`

```python
model = EmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

### `encode(texts: List[str], batch_size=64, show_progress=True) -> np.ndarray`

Encode multiple texts.

```python
vectors = model.encode(["Delta Lake provides ACID transactions",
                      "Spark SQL for data engineering"])
# Shape: (2, 768) for all-mpnet-base-v2
```

### `encode_single(text: str) -> np.ndarray`

Encode one text.

```python
vec = model.encode_single("What is Delta Lake?")
# Shape: (768,)
```

### `get_dimension() -> int`

Get embedding dimension.

```python
dim = model.get_dimension()  # 768
```

---

## Class: `FAISSIndex`

FAISS vector index wrapper.

### `__init__(dimension: int)`

```python
index = FAISSIndex(dimension=768)
```

### `build(vectors: np.ndarray, metadata: List[dict])`

Build index from embeddings and metadata.

```python
index.build(embeddings, chunks_metadata)
```

### `search(query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]`

Search for top-k results. Returns `(distances, indices)`.

```python
distances, indices = index.search(query_vector, k=5)
```

### `get_chunk(index: int) -> dict`

Get metadata for a result.

```python
chunk = index.get_chunk(indices[0][0])
```

### `save(index_path: str, metadata_path: str)`

Persist to disk.

```python
index.save("data/delta_lake.index", "data/vectometa.jsonl")
```

### `load(index_path: str, metadata_path: str) -> FAISSIndex`

Load from disk (class method).

```python
index = FAISSIndex.load("data/delta_lake.index", "data/vectometa.jsonl")
```

---

## Class: `OllamaLLM`

Ollama LLM client.

### `__init__(model="qwen3.5:cloud", base_url="http://localhost:11434", temperature=0.2, max_tokens=512)`

```python
llm = OllamaLLM(model="qwen3.5:cloud")
```

### `generate(prompt: str, system: str = None) -> str`

Generate from a prompt.

```python
answer = llm.generate("What is Delta Lake?", system="You are a Databricks expert.")
```

### `generate_with_context(question: str, context: str) -> str`

Generate using retrieved context.

```python
answer = llm.generate_with_context(
    "How to create a Delta table?",
    "CREATE TABLE uses DELTA format..."
)
```

---

## Class: `KeywordDetector`

Auto-trigger keyword detector.

### `__init__(keywords: List[str] = None)`

```python
detector = KeywordDetector()  # Uses defaults
detector = KeywordDetector(keywords=["delta lake", "databricks"])
```

### `contains_keywords(text: str) -> bool`

Check if text contains any keywords.

```python
if detector.contains_keywords("Tell me about Delta Lake merge"):
    # True
```

### `should_use_rag(text: str) -> bool`

Determine if RAG should be used.

```python
detector.should_use_rag("How do I upsert into a Delta table?")  # True
```

---

## Config Functions

### `load_config(config_path="config.yaml") -> Config`

Load from YAML file.

```python
config = load_config("config.yaml")
```

### `get_config() -> Config`

Get global config (lazy-loaded).

```python
config = get_config()
print(config.llm.model)  # qwen3.5:cloud
```

### `set_config(config: Config) -> None`

Set global config.

### `reset_config() -> None`

Reset global config (testing).

---

## Integrations

### REST API

```python
from rag_pipeline.integrations.rest_api import app, main

# Run server
main()

# Or embed in your own FastAPI app
from fastapi import FastAPI
my_app = FastAPI()
my_app.include_router(app.router, prefix="/rag-api")
```

### CLI

```python
from rag_pipeline.integrations.cli import main
main()
```

### LangChain

```python
from rag_pipeline.integrations.langchain import get_langchain_tool

tool = get_langchain_tool(rag_pipeline=pipeline)
```

### Tool/MCP

```python
from rag_pipeline.integrations.tool import ToolDefinition

tool_def = ToolDefinition(api_url="http://localhost:8000")
schema = tool_def.get_definition()
trigger = tool_def.should_auto_trigger("Delta Lake question")
result = tool_def.execute("How to merge into Delta table?")
```

### Lambda

```python
from rag_pipeline.integrations.lambda_handler import handler

result = handler({"question": "What is Delta Lake?"}, None)
```