# Databricks Delta Lake & Lakeflow RAG Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local RAG pipeline that answers developer questions about Databricks Delta Lake and Lakeflow Declarative Pipeline documentation using a local embedding model, FAISS vector store, and GLM-5.1 via Ollama Cloud (OpenRouter API).

**Architecture:** Simple vector retrieval pipeline: docs → chunk (256 tokens, 50 overlap) → embed (all-mpnet-base-v2) → FAISS index → retrieve top-k → augment prompt → query LLM via OpenRouter API (GLM-5.1).

**Tech Stack:**
- Python 3.10+
- sentence-transformers (all-mpnet-base-v2)
- FAISS (faiss-cpu)
- BeautifulSoup4 (scraping)
- OpenRouter API (GLM-5.1)
- PyYAML (config)
- Pydantic (data models)

---

## File Structure

```
rag-practice/
├── config.yaml                    # Configuration parameters
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview and usage
├── docs/
│   └── superpowers/
│       └── plans/
│           └── 2026-04-22-databricks-rag-pipeline.md
├── src/
│   ├── __init__.py
│   ├── config.py                  # Config loading
│   ├── models.py                  # Pydantic data models
│   ├── ingest.py                  # Documentation crawler
│   ├── chunking.py                # Token-aware text chunking
│   ├── embed.py                   # Embedding generation
│   ├── vectorstore.py             # FAISS index management
│   ├── retrieval.py               # Query retrieval
│   ├── reranker.py                # Optional cross-encoder reranker
│   ├── llm.py                     # Ollama LLM integration
│   ├── pipeline.py                # End-to-end RAG pipeline
│   └── repl.py                    # Interactive REPL
├── tests/
│   ├── __init__.py
│   ├── test_chunking.py
│   ├── test_embed.py
│   ├── test_vectorstore.py
│   └── test_retrieval.py
├── data/
│   ├── docs_raw.jsonl             # Raw fetched documentation
│   ├── docs_chunks.jsonl          # Chunked documents
│   ├── delta_lake.index           # FAISS index file
│   └── vectometa.jsonl            # Vector metadata mapping
└── scripts/
    ├── run_repl.sh                # REPL launcher
    └── benchmark.sh               # Performance benchmarking
```

---

### Task 1: Project Setup & Configuration

**Files:**
- Create: `rag-practice/config.yaml`
- Create: `rag-practice/requirements.txt`
- Create: `rag-practice/README.md`
- Create: `rag-practice/src/__init__.py`

- [ ] **Step 1: Create config.yaml**

```yaml
# Databricks RAG Pipeline Configuration

# Embedding Model
embedding_model: "sentence-transformers/all-mpnet-base-v2"
embedding_batch_size: 64

# Chunking
chunk_size_tokens: 256
chunk_overlap_tokens: 50

# Retrieval
top_k: 5

# Reranker (optional)
use_reranker: false
reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

# LLM (Ollama Cloud - GLM 5.1)
llm_model: "glm-5.1:cloud"
llm_base_url: "http://localhost:11434"
llm_temperature: 0.2
llm_max_tokens: 512

# Paths
data_dir: "data"
docs_raw: "data/docs_raw.jsonl"
docs_chunks: "data/docs_chunks.jsonl"
faiss_index: "data/delta_lake.index"
vectometa: "data/vectometa.jsonl"

# Logging
log_level: "INFO"
query_log: "data/query_log.jsonl"
```

- [ ] **Step 2: Create requirements.txt**

```txt
sentence-transformers>=2.2.2
faiss-cpu>=1.9.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
requests>=2.31.0
pydantic>=2.0.0
pyyaml>=6.0
tiktoken>=0.5.0
numpy>=1.24.0
torch>=2.0.0
```

- [ ] **Step 3: Create README.md**

```markdown
# Databricks Delta Lake & Lakeflow RAG Pipeline

Local RAG system to answer developer questions about Databricks Delta Lake and Lakeflow Declarative Pipeline documentation.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install and start Ollama:
   ```bash
   brew install ollama
   ollama serve
   ollama pull llama3
   ```

3. Run the REPL:
   ```bash
   python -m src.repl
   ```

## Pipeline Steps

1. **Ingest** - Fetch Databricks docs
2. **Chunk** - Split into 256-token chunks with 50-token overlap
3. **Embed** - Generate embeddings using all-mpnet-base-v2
4. **Index** - Build FAISS vector index
5. **Query** - Retrieve relevant chunks and ask LLM
```

- [ ] **Step 4: Create src/__init__.py**

```python
"""Databricks Delta Lake RAG Pipeline"""
__version__ = "0.1.0"
```

- [ ] **Step 5: Commit**

```bash
cd ~/Document/self/rag-practice
git init
git add config.yaml requirements.txt README.md src/__init__.py
git commit -m "feat: initial project setup with config and dependencies"
```

---

### Task 2: Configuration & Data Models

**Files:**
- Create: `rag-practice/src/config.py`
- Create: `rag-practice/src/models.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_config.py
import pytest
from pathlib import Path
from src.config import Config

def test_config_loads_defaults():
    config = Config()
    assert config.embedding_model == "sentence-transformers/all-mpnet-base-v2"
    assert config.chunk_size_tokens == 256
    assert config.chunk_overlap_tokens == 50
    assert config.top_k == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Document/self/rag-practice && python -m pytest tests/test_config.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src'"

- [ ] **Step 3: Write config.py**

```python
"""Configuration loading for RAG pipeline"""
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
import yaml


class Config(BaseModel):
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_batch_size: int = 64
    chunk_size_tokens: int = 256
    chunk_overlap_tokens: int = 50
    top_k: int = 5
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "glm-5.1:cloud"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: str = ""
    llm_api_key: str = ""  # Set OLLAMA_API_KEY env var  # Set OPENROUTER_API_KEY env var
    llm_temperature: float = 0.2
    llm_max_tokens: int = 512
    data_dir: str = "data"
    docs_raw: str = "data/docs_raw.jsonl"
    docs_chunks: str = "data/docs_chunks.jsonl"
    faiss_index: str = "data/delta_lake.index"
    vectometa: str = "data/vectometa.jsonl"
    log_level: str = "INFO"
    query_log: str = "data/query_log.jsonl"

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "Config":
        config_path = Path(path)
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f)
                return cls(**data)
        return cls()


_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config.from_yaml()
    return _config
```

- [ ] **Step 4: Write models.py**

```python
"""Pydantic data models for RAG pipeline"""
from typing import Optional
from pydantic import BaseModel


class DocChunk(BaseModel):
    doc_id: int
    chunk_id: int
    text: str
    metadata: dict


class RetrievedChunk(BaseModel):
    doc_id: int
    chunk_id: int
    text: str
    metadata: dict
    score: float


class QueryResult(BaseModel):
    question: str
    retrieved_chunks: list[RetrievedChunk]
    answer: Optional[str] = None
    latency_ms: Optional[float] = None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd ~/Document/self/rag-practice && python -m pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
cd ~/Document/self/rag-practice
git add src/config.py src/models.py tests/test_config.py
git commit -m "feat: add config and data models"
```

---

### Task 3: Documentation Ingestion

**Files:**
- Create: `rag-practice/src/ingest.py`
- Test: `rag-practice/tests/test_ingest.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_ingest.py
import pytest
from pathlib import Path
import tempfile
import json
from src.ingest import fetch_databricks_docs, save_raw_docs


def test_fetch_databricks_docs_parses_html():
    # Test with mock HTML content
    html = "<html><body><main><article><h1>Delta Lake</h1><p>Content here</p></article></main></body></html>"
    docs = fetch_databricks_docs("https://example.com/test", html)
    assert len(docs) > 0
    assert docs[0]["title"] == "Delta Lake"
    assert "Content here" in docs[0]["content"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Document/self/rag-practice && python -m pytest tests/test_ingest.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write ingest.py**

```python
"""Documentation ingestion from Databricks docs"""
import json
import hashlib
from typing import List, Dict
from pathlib import Path
from bs4 import BeautifulSoup
import requests
from src.models import Config
from src.config import get_config


def fetch_databricks_docs(url: str, html: str = None) -> List[Dict]:
    """Parse Databricks documentation HTML and extract content"""
    if html is None:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        html = response.text
    
    soup = BeautifulSoup(html, "lxml")
    
    # Find main content area
    main = soup.find("main") or soup.find("article") or soup.find("div", {"role": "main"})
    if not main:
        return []
    
    # Remove navigation, footer, sidebar
    for unwanted in main.find_all(["nav", "footer", "aside", "script", "style"]):
        unwanted.decompose()
    
    # Extract title
    title_tag = main.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "Untitled"
    
    # Get text content
    content = main.get_text(separator="\n", strip=True)
    
    # Generate ID from URL
    doc_id = int(hashlib.md5(url.encode()).hexdigest()[:8], 16)
    
    return [{
        "doc_id": doc_id,
        "url": url,
        "title": title,
        "content": content,
    }]


def save_raw_docs(docs: List[Dict], output_path: str):
    """Save raw docs to JSONL file"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Document/self/rag-practice && python -m pytest tests/test_ingest.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd ~/Document/self/rag-practice
git add src/ingest.py tests/test_ingest.py
git commit -m "feat: add documentation ingestion module"
```

---

### Task 4: Token-Aware Chunking

**Files:**
- Create: `rag-practice/src/chunking.py`
- Test: `rag-practice/tests/test_chunking.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_chunking.py
import pytest
from src.chunking import chunk_document, tokenize


def test_tokenize_counts_tokens():
    text = "This is a test sentence for chunking."
    tokens = tokenize(text)
    assert len(tokens) > 0


def test_chunk_document_creates_overlapping_chunks():
    doc = {
        "doc_id": 1,
        "url": "https://example.com",
        "title": "Test Doc",
        "content": " ".join(["word"] * 300)  # ~300 words
    }
    chunks = chunk_document(doc, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    # Verify overlap: first chunk end should appear in second chunk start
    first_text = chunks[0]["text"]
    second_text = chunks[1]["text"]
    # Last few tokens of first should appear in second
    first_end = " ".join(first_text.split()[-5:])
    assert first_end.split()[0] in second_text or len(chunks) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Document/self/rag-practice && python -m pytest tests/test_chunking.py -v`
Expected: FAIL

- [ ] **Step 3: Write chunking.py**

```python
"""Token-aware text chunking for RAG pipeline"""
import json
from typing import List, Dict
from pathlib import Path
import tiktoken
from src.models import DocChunk


def get_tokenizer(model_name: str = "cl100k_base"):
    """Get tiktoken tokenizer for token counting"""
    return tiktoken.get_encoding(model_name)


def tokenize(text: str, tokenizer=None) -> List[str]:
    """Convert text to tokens"""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    return tokenizer.encode(text, disallowed_special=())


def detokenize(tokens: List[str], tokenizer=None) -> str:
    """Convert tokens back to text"""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    return tokenizer.decode(tokens)


def chunk_document(
    doc: Dict,
    chunk_size: int = 256,
    overlap: int = 50,
    tokenizer=None
) -> List[DocChunk]:
    """Split a document into overlapping chunks of specified token size"""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    
    tokens = tokenize(doc["content"], tokenizer)
    chunks = []
    
    start = 0
    chunk_id = 0
    
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = detokenize(chunk_tokens, tokenizer)
        
        chunks.append(DocChunk(
            doc_id=doc["doc_id"],
            chunk_id=chunk_id,
            text=chunk_text,
            metadata={
                "url": doc["url"],
                "title": doc["title"],
                "chunk_index": chunk_id,
                "token_count": len(chunk_tokens)
            }
        ))
        
        # Move start position with overlap
        start = start + chunk_size - overlap
        chunk_id += 1
        
        # Safety: prevent infinite loop
        if overlap == 0:
            break
    
    return chunks


def chunk_all_documents(
    input_path: str,
    output_path: str,
    chunk_size: int = 256,
    overlap: int = 50
):
    """Load raw docs and chunk them all"""
    tokenizer = get_tokenizer()
    
    all_chunks = []
    
    with open(input_path, "r") as f:
        for line in f:
            doc = json.loads(line)
            chunks = chunk_document(
                doc,
                chunk_size=chunk_size,
                overlap=overlap,
                tokenizer=tokenizer
            )
            all_chunks.extend(chunks)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for chunk in all_chunks:
            f.write(chunk.model_dump_json() + "\n")
    
    return len(all_chunks)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Document/self/rag-practice && python -m pytest tests/test_chunking.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd ~/Document/self/rag-practice
git add src/chunking.py tests/test_chunking.py
git commit -m "feat: add token-aware chunking module"
```

---

### Task 5: Embedding Generation

**Files:**
- Create: `rag-practice/src/embed.py`
- Test: `rag-practice/tests/test_embed.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_embed.py
import pytest
import numpy as np
from src.embed import EmbeddingModel, get_embedding_model


def test_embedding_model_initializes():
    model = get_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
    assert model is not None


def test_embedding_generates_vector():
    model = get_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
    embedding = model.encode(["test sentence"])
    assert embedding.shape[1] == 384  # MiniLM dimension
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Document/self/rag-practice && python -m pytest tests/test_embed.py -v`
Expected: FAIL (model not downloaded)

- [ ] **Step 3: Write embed.py**

```python
"""Embedding model for RAG pipeline"""
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json


class EmbeddingModel:
    def __init__(self, model_name: str, normalize: bool = True):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        """Encode texts into embeddings"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True
        )
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text"""
        return self.encode([text], show_progress=False)[0]


def get_embedding_model(model_name: str = None) -> EmbeddingModel:
    """Get or create embedding model singleton"""
    if model_name is None:
        from src.config import get_config
        config = get_config()
        model_name = config.embedding_model
    
    return EmbeddingModel(model_name)


def embed_chunks(chunks_path: str, output_path: str, model_name: str = None):
    """Load chunks and generate embeddings"""
    model = get_embedding_model(model_name)
    
    # Load chunks
    chunks = []
    with open(chunks_path, "r") as f:
        for line in f:
            chunks.append(json.loads(line))
    
    texts = [chunk["text"] for chunk in chunks]
    
    # Generate embeddings
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, batch_size=64, show_progress=True)
    
    # Save embeddings as numpy array
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path.replace(".npy", ".npy"), embeddings.astype("float32"))
    
    return embeddings.shape
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Document/self/rag-practice && python -m pytest tests/test_embed.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd ~/Document/self/rag-practice
git add src/embed.py tests/test_embed.py
git commit -m "feat: add embedding generation module"
```

---

### Task 6: FAISS Vector Store

**Files:**
- Create: `rag-practice/src/vectorstore.py`
- Test: `rag-practice/tests/test_vectorstore.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_vectorstore.py
import pytest
import numpy as np
import faiss
from src.vectorstore import FAISSIndex, create_faiss_index


def test_faiss_index_creation():
    vectors = np.random.rand(10, 768).astype("float32")
    index = create_faiss_index(vectors)
    assert index.ntotal == 10


def test_faiss_search():
    vectors = np.random.rand(10, 768).astype("float32")
    index = create_faiss_index(vectors)
    query = np.random.rand(1, 768).astype("float32")
    distances, indices = index.search(query, k=3)
    assert distances.shape == (1, 3)
    assert indices.shape == (1, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Document/self/rag-practice && python -m pytest tests/test_vectorstore.py -v`
Expected: FAIL

- [ ] **Step 3: Write vectorstore.py**

```python
"""FAISS vector store for RAG pipeline"""
import numpy as np
import faiss
import json
from pathlib import Path
from typing import List, Tuple, Optional
from src.models import RetrievedChunk


class FAISSIndex:
    def __init__(self, dimension: int, use_gpu: bool = False):
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.metadata: List[dict] = []
        self.use_gpu = use_gpu
    
    def build(self, vectors: np.ndarray, metadata: List[dict]):
        """Build index from vectors and metadata"""
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} != expected {self.dimension}")
        
        # Create index - using FlatIP for cosine similarity (normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Convert to float32 if needed
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        
        self.index.add(vectors)
        self.metadata = metadata
    
    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search for top-k results"""
        if self.index is None:
            raise ValueError("Index not built yet")
        
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        query = query.astype(np.float32)
        return self.index.search(query, k)
    
    def get_chunk(self, index: int) -> dict:
        """Get metadata for a specific index"""
        return self.metadata[index]
    
    def save(self, index_path: str, metadata_path: str):
        """Persist index and metadata to disk"""
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)
        
        with open(metadata_path, "w") as f:
            for meta in self.metadata:
                f.write(json.dumps(meta) + "\n")
    
    @classmethod
    def load(cls, index_path: str, metadata_path: str) -> "FAISSIndex":
        """Load index and metadata from disk"""
        index = faiss.read_index(index_path)
        
        metadata = []
        with open(metadata_path, "r") as f:
            for line in f:
                metadata.append(json.loads(line))
        
        instance = cls(dimension=index.d)
        instance.index = index
        instance.metadata = metadata
        return instance


def create_faiss_index(vectors: np.ndarray, use_gpu: bool = False) -> faiss.Index:
    """Create a FAISS index from vectors"""
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    
    index.add(vectors)
    return index


def build_index_from_chunks(
    chunks_path: str,
    embeddings_path: str,
    index_path: str,
    metadata_path: str
) -> FAISSIndex:
    """Build FAISS index from chunked docs and embeddings"""
    # Load embeddings
    embeddings = np.load(embeddings_path)
    
    # Load metadata
    metadata = []
    with open(chunks_path, "r") as f:
        for line in f:
            chunk = json.loads(line)
            metadata.append({
                "doc_id": chunk["doc_id"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"]
            })
    
    # Create and build index
    faiss_index = FAISSIndex(dimension=embeddings.shape[1])
    faiss_index.build(embeddings, metadata)
    
    # Save
    faiss_index.save(index_path, metadata_path)
    
    return faiss_index
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Document/self/rag-practice && python -m pytest tests/test_vectorstore.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd ~/Document/self/rag-practice
git add src/vectorstore.py tests/test_vectorstore.py
git commit -m "feat: add FAISS vector store module"
```

---

### Task 7: Retrieval Module

**Files:**
- Create: `rag-practice/src/retrieval.py`
- Test: `rag-practice/tests/test_retrieval.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_retrieval.py
import pytest
import numpy as np
from src.retrieval import Retriever
from src.vectorstore import FAISSIndex


def test_retriever_finds_relevant_chunks():
    # Create a simple index with known content
    dimension = 384
    index = FAISSIndex(dimension)
    
    vectors = np.array([
        [1.0] * dimension,
        [0.0] * dimension,
        [0.5] * dimension
    ], dtype=np.float32)
    
    metadata = [
        {"doc_id": 1, "chunk_id": 0, "text": "Delta Lake is ACID compliant"},
        {"doc_id": 2, "chunk_id": 0, "text": "Python is a programming language"},
        {"doc_id": 3, "chunk_id": 0, "text": "Spark processes big data"}
    ]
    
    index.build(vectors, metadata)
    
    retriever = Retriever(index, top_k=2)
    results = retriever.retrieve("What is Delta Lake?")
    
    assert len(results) <= 2
    assert results[0].text.startswith("Delta Lake")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Document/self/rag-practice && python -m pytest tests/test_retrieval.py -v`
Expected: FAIL

- [ ] **Step 3: Write retrieval.py**

```python
"""Retrieval module for RAG pipeline"""
import numpy as np
from typing import List
from src.vectorstore import FAISSIndex
from src.embed import get_embedding_model
from src.models import RetrievedChunk


class Retriever:
    def __init__(self, index: FAISSIndex, embedding_model=None, top_k: int = 5):
        self.index = index
        self.embedding_model = embedding_model
        self.top_k = top_k
    
    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """Retrieve top-k chunks for a query"""
        # Encode query
        if self.embedding_model is None:
            self.embedding_model = get_embedding_model()
        
        query_vector = self.embedding_model.encode_single(query)
        
        # Search
        distances, indices = self.index.search(query_vector, k=self.top_k)
        
        # Build results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            
            chunk_meta = self.index.get_chunk(idx)
            results.append(RetrievedChunk(
                doc_id=chunk_meta["doc_id"],
                chunk_id=chunk_meta["chunk_id"],
                text=chunk_meta["text"],
                metadata=chunk_meta["metadata"],
                score=float(dist)
            ))
        
        return results
    
    def retrieve_raw(self, query: str) -> List[dict]:
        """Retrieve as raw dicts (for debugging)"""
        results = self.retrieve(query)
        return [
            {
                "doc_id": r.doc_id,
                "chunk_id": r.chunk_id,
                "text": r.text,
                "score": r.score,
                **r.metadata
            }
            for r in results
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Document/self/rag-practice && python -m pytest tests/test_retrieval.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd ~/Document/self/rag-practice
git add src/retrieval.py tests/test_retrieval.py
git commit -m "feat: add retrieval module"
```

---

### Task 8: LLM Integration (Ollama)

**Files:**
- Create: `rag-practice/src/llm.py`

- [ ] **Step 1: Write llm.py**

```python
"""LLM integration via Ollama (GLM-5.1:cloud)"""
import requests
from src.config import get_config


class OllamaLLM:
    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        temperature: float = 0.2,
        max_tokens: int = 512
    ):
        config = get_config()
        self.model = model or config.llm_model
        self.base_url = base_url or config.llm_base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate(self, prompt: str, system: str = None) -> str:
        """Generate response from LLM via Ollama API"""
        messages = []
        
        if system:
            messages.append({"role": "system", "content": system})
        
        messages.append({"role": "user", "content": prompt})
        
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False
            },
            timeout=120
        )
        
        response.raise_for_status()
        result = response.json()
        
        return result["message"]["content"]
    
    def generate_with_context(self, question: str, context: str) -> str:
        """Generate answer given retrieved context"""
        system_prompt = """You are a helpful assistant that answers questions about Databricks Delta Lake and Lakeflow Declarative Pipeline using ONLY the provided documentation excerpts. If the answer cannot be found in the excerpts, say you don't know."""
        
        prompt = f"""Context:
{context}

Question: {question}
Answer:"""
        
        return self.generate(prompt, system=system_prompt)


def get_llm() -> OllamaLLM:
    """Get Ollama LLM instance"""
    return OllamaLLM()
```

- [ ] **Step 2: Commit**

```bash
cd ~/Document/self/rag-practice
git add src/llm.py
git commit -m "feat: add Ollama LLM integration"
```

---

### Task 9: Optional Reranker

**Files:**
- Create: `rag-practice/src/reranker.py`

- [ ] **Step 1: Write reranker.py**

```python
"""Optional cross-encoder reranker for improved retrieval"""
from typing import List
import numpy as np
from sentence_transformers import CrossEncoder
from src.models import RetrievedChunk


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int = 3) -> List[RetrievedChunk]:
        """Rerank chunks based on query-chunk relevance"""
        if not chunks:
            return []
        
        # Create query-chunk pairs
        pairs = [(query, chunk.text) for chunk in chunks]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Sort by score descending
        sorted_indices = np.argsort(scores)[::-1]
        
        # Reorder chunks
        reranked = [chunks[i] for i in sorted_indices[:top_k]]
        
        # Update scores
        for chunk, idx in zip(reranked, sorted_indices[:top_k]):
            chunk.score = float(scores[idx])
        
        return reranked
```

- [ ] **Step 2: Commit**

```bash
cd ~/Document/self/rag-practice
git add src/reranker.py
git commit -m "feat: add optional cross-encoder reranker"
```

---

### Task 10: End-to-End Pipeline

**Files:**
- Create: `rag-practice/src/pipeline.py`

- [ ] **Step 1: Write pipeline.py**

```python
"""End-to-end RAG pipeline"""
import json
import time
from pathlib import Path
from typing import List, Optional
from src.config import get_config
from src.embed import get_embedding_model
from src.vectorstore import FAISSIndex
from src.retrieval import Retriever
from src.reranker import Reranker
from src.llm import OllamaLLM
from src.models import QueryResult, RetrievedChunk


class RAGPipeline:
    def __init__(self, config=None):
        self.config = config or get_config()
        self.embedding_model = None
        self.index = None
        self.retriever = None
        self.reranker = None
        self.llm = None
    
    def load(self):
        """Load all components"""
        # Load embedding model
        self.embedding_model = get_embedding_model(self.config.embedding_model)
        
        # Load FAISS index
        self.index = FAISSIndex.load(
            self.config.faiss_index,
            self.config.vectometa
        )
        
        # Create retriever
        self.retriever = Retriever(
            self.index,
            self.embedding_model,
            self.config.top_k
        )
        
        # Load reranker if enabled
        if self.config.use_reranker:
            self.reranker = Reranker(self.config.reranker_model)
        
        # Load LLM
        self.llm = OllamaLLM()
    
    def query(self, question: str) -> QueryResult:
        """Query the RAG pipeline"""
        start_time = time.time()
        
        # Retrieve chunks
        chunks = self.retriever.retrieve(question)
        
        # Rerank if enabled
        if self.reranker and len(chunks) > self.config.top_k // 2:
            chunks = self.reranker.rerank(question, chunks, top_k=self.config.top_k)
        
        # Build context
        context = "\n\n---\n\n".join([chunk.text for chunk in chunks])
        
        # Generate answer
        answer = self.llm.generate_with_context(question, context)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return QueryResult(
            question=question,
            retrieved_chunks=chunks,
            answer=answer,
            latency_ms=latency_ms
        )
    
    def query_with_sources(self, question: str) -> dict:
        """Query and return answer with source references"""
        result = self.query(question)
        
        return {
            "question": result.question,
            "answer": result.answer,
            "sources": [
                {
                    "title": chunk.metadata.get("title", ""),
                    "url": chunk.metadata.get("url", ""),
                    "score": chunk.score
                }
                for chunk in result.retrieved_chunks
            ],
            "latency_ms": result.latency_ms
        }
    
    def log_query(self, question: str, result: QueryResult):
        """Log query to file"""
        Path(self.config.query_log).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.query_log, "a") as f:
            f.write(json.dumps({
                "question": question,
                "answer": result.answer,
                "latency_ms": result.latency_ms,
                "num_sources": len(result.retrieved_chunks)
            }) + "\n")


def get_pipeline() -> RAGPipeline:
    """Get or create pipeline singleton"""
    pipeline = RAGPipeline()
    pipeline.load()
    return pipeline
```

- [ ] **Step 2: Commit**

```bash
cd ~/Document/self/rag-practice
git add src/pipeline.py
git commit -m "feat: add end-to-end RAG pipeline"
```

---

### Task 11: Interactive REPL

**Files:**
- Create: `rag-practice/src/repl.py`
- Create: `rag-practice/scripts/run_repl.sh`

- [ ] **Step 1: Write repl.py**

```python
"""Interactive REPL for querying the RAG pipeline"""
import sys
from src.pipeline import get_pipeline


def main():
    print("Databricks Delta Lake RAG - Interactive REPL")
    print("Type 'quit' or 'exit' to exit\n")
    
    pipeline = get_pipeline()
    
    while True:
        try:
            question = input("> ").strip()
            
            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            result = pipeline.query(question)
            
            print(f"\nAnswer: {result.answer}\n")
            print(f"Sources ({len(result.retrieved_chunks)}):")
            for i, chunk in enumerate(result.retrieved_chunks, 1):
                title = chunk.metadata.get("title", "Untitled")
                score = chunk.score
                print(f"  {i}. {title} (score: {score:.3f})")
            print(f"\nLatency: {result.latency_ms:.0f}ms\n")
            
            # Log query
            pipeline.log_query(question, result)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write run_repl.sh**

```bash
#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
python -m src.repl
```

- [ ] **Step 3: Commit**

```bash
cd ~/Document/self/rag-practice
git add src/repl.py scripts/run_repl.sh
git commit -m "feat: add interactive REPL"
```

---

### Task 12: Sample Data & Testing

**Files:**
- Create: `rag-practice/scripts/setup_sample_data.py`

- [ ] **Step 1: Write setup_sample_data.py**

```python
"""Setup sample data for testing the pipeline"""
import json
from pathlib import Path
from src.ingest import save_raw_docs
from src.chunking import chunk_all_documents
from src.embed import get_embedding_model
from src.vectorstore import build_index_from_chunks


SAMPLE_DOCS = [
    {
        "doc_id": 1,
        "url": "https://docs.databricks.com/delta/introduction.html",
        "title": "What is Delta Lake?",
        "content": """Delta Lake is an open source storage layer that brings ACID 
transactions to Apache Spark and Big Data workloads. Delta Lake provides 
serializable isolation levels between concurrent reads and writes. This ensures 
that readers see consistent snapshots of data while writers can modify the table 
without conflicting with each other. Delta Lake stores data in Parquet format 
and maintains a transaction log that records every commit made to the table.

Key features of Delta Lake include:
1. ACID transactions - Ensures data consistency
2. Time travel - Query previous versions of data
3. Schema enforcement - Prevents bad data from being written
4. Audit logging - Track all changes made to the table
5. Merge, update, and delete operations - CRUD operations on tables"""
    },
    {
        "doc_id": 2,
        "url": "https://docs.databricks.com/delta/tutorial.html",
        "title": "Getting Started with Delta Lake",
        "content": """To create a Delta Lake table, you can use either Spark SQL or 
the Delta Lake API in Python, Scala, or Java. In Python, use the following code:

spark.sql("CREATE TABLE my_table (id INT, name STRING) USING delta")

Or using the DataFrame API:

df.write.format("delta").save("/mnt/delta/table")

Delta Lake tables support various operations including:
- INSERT: Add new rows
- UPDATE: Modify existing rows
- DELETE: Remove rows
- MERGE: Upsert data from another table

You can also use time travel to query previous versions:
df = spark.read.format("delta").option("versionAsOf", 5).load("/mnt/table")"""
    },
    {
        "doc_id": 3,
        "url": "https://docs.databricks.com/lakeflow/introduction.html",
        "title": "What is Lakeflow?",
        "content": """Lakeflow is Databricks' declarative pipeline framework for 
building ETL pipelines on top of Delta Lake. Lakeflow allows you to define 
data transformation pipelines using a declarative syntax, handling orchestration, 
error handling, and monitoring automatically.

Key components of Lakeflow:
1. Pipelines - Define your ETL logic declaratively
2. Targets - Define where data should be written
3. Sources - Define input data sources
4. Transformations - Define data transformations

Lakeflow automatically manages:
- Task scheduling and orchestration
- Data quality validation
- Error handling and retries
- Monitoring and alerting"""
    }
]


def main():
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Save raw docs
    print("Saving sample docs...")
    save_raw_docs(SAMPLE_DOCS, "data/docs_raw.jsonl")
    
    # Chunk docs
    print("Chunking documents...")
    num_chunks = chunk_all_documents(
        "data/docs_raw.jsonl",
        "data/docs_chunks.jsonl",
        chunk_size=256,
        overlap=50
    )
    print(f"Created {num_chunks} chunks")
    
    # Generate embeddings
    print("Generating embeddings...")
    model = get_embedding_model()
    
    chunks = []
    with open("data/docs_chunks.jsonl", "r") as f:
        for line in f:
            chunks.append(json.loads(line))
    
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress=True)
    
    # Save embeddings
    import numpy as np
    np.save("data/embeddings.npy", embeddings.astype("float32"))
    
    # Build FAISS index
    print("Building FAISS index...")
    from src.vectorstore import FAISSIndex
    
    metadata = []
    for chunk in chunks:
        metadata.append({
            "doc_id": chunk["doc_id"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "metadata": chunk["metadata"]
        })
    
    index = FAISSIndex(dimension=embeddings.shape[1])
    index.build(embeddings, metadata)
    index.save("data/delta_lake.index", "data/vectometa.jsonl")
    
    print("Done! Run 'python -m src.repl' to query the RAG.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run sample data setup**

```bash
cd ~/Document/self/rag-practice && python scripts/setup_sample_data.py
```

- [ ] **Step 3: Commit**

```bash
cd ~/Document/self/rag-practice
git add scripts/setup_sample_data.py
git commit -m "feat: add sample data setup script"
```

---

### Task 13: Verify Full Pipeline

- [ ] **Step 1: Run REPL and test a query**

```bash
cd ~/Document/self/rag-practice
echo "What is Delta Lake?" | python -m src.repl
```

- [ ] **Step 2: Verify retrieval and answer**

Expected: Pipeline returns relevant chunks about Delta Lake and generates a coherent answer.

- [ ] **Step 3: Commit**

```bash
cd ~/Document/self/rag-practice
git add data/
git commit -m "feat: add sample data and verify pipeline"
```

---

## Plan Complete

**Plan saved to:** `~/Document/self/rag-practice/docs/superpowers/plans/2026-04-22-databricks-rag-pipeline.md`

**Two execution options:**

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
