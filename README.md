# Databricks Delta Lake & Lakeflow RAG Pipeline

Local RAG system to answer developer questions about Databricks Delta Lake and Lakeflow Declarative Pipeline documentation.

## Architecture

```
docs → chunk (256 tokens, 50 overlap) → embed (all-mpnet-base-v2) → FAISS index → retrieve top-k → augment prompt → query LLM
```

## Setup

1. Install dependencies:
   ```bash
   cd ~/Document/self/rag-practice
   pip install -r requirements.txt
   ```

2. Install and start Ollama:
   ```bash
   brew install ollama
   ollama serve
   ollama pull glm-5.1
   ```

3. Build the index with sample data:
   ```bash
   python scripts/setup_sample_data.py
   ```

4. Run the REPL:
   ```bash
   python -m src.repl
   ```

## Pipeline Steps

1. **Ingest** - Fetch Databricks docs (or use sample data)
2. **Chunk** - Split into 256-token chunks with 50-token overlap
3. **Embed** - Generate embeddings using all-mpnet-base-v2
4. **Index** - Build FAISS vector index
5. **Query** - Retrieve relevant chunks and ask LLM

## Testing

Run all tests:
```bash
python -m pytest tests/ -v
```

## Project Structure

```
rag-practice/
├── config.yaml          # Configuration parameters
├── requirements.txt     # Python dependencies
├── src/
│   ├── config.py        # Config loading
│   ├── models.py        # Pydantic data models
│   ├── ingest.py        # Documentation crawler
│   ├── chunking.py      # Token-aware text chunking
│   ├── embed.py         # Embedding generation
│   ├── vectorstore.py   # FAISS index management
│   ├── retrieval.py     # Query retrieval
│   ├── reranker.py      # Optional cross-encoder reranker
│   ├── llm.py           # Ollama LLM integration
│   ├── pipeline.py      # End-to-end RAG pipeline
│   └── repl.py          # Interactive REPL
├── tests/               # Unit tests
├── data/                # Generated data (index, embeddings)
└── scripts/             # Setup and benchmark scripts
```
