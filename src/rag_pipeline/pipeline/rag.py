"""Two-Step RAG Pipeline — prevents hallucination by forcing grounded answers.

This module is the heart of the RAG pipeline. It implements the "Two-Step RAG"
pattern used in enterprise deployments to prevent LLM hallucination:

    Step 1 — RETRIEVE:  Find relevant docs from the vector store (FAISS).
                Your question is converted to an embedding vector, then
                FAISS searches for the k-nearest chunks by cosine similarity.

    Step 2 — GENERATE:  Inject the retrieved chunks into the LLM prompt with
                a STRICT prompt that forces the LLM to answer using ONLY
                the provided context. The LLM cannot use its own training
                knowledge — it must ground every answer in the retrieved docs.

Why Two-Step instead of One-Step?
    In a single-step approach, the LLM might ignore the retrieved context
    and answer from memory, especially for common topics. The strict_prompt
    and the RULES listed in the prompt template make it harder for the LLM
    to hallucinate — if it doesn't know the answer from the context, it must
    say "I don't have enough information" instead of guessing.

Data flow:
    Question → Embed (sentence-transformers) → FAISS search → top-k chunks
             → concatenate chunks into context → strict_prompt template
             → LLM.generate() → grounded answer
"""
from pathlib import Path
from typing import Optional, List, Dict, Any

from rag_pipeline.config import get_config
from rag_pipeline.core.embed import EmbeddingModel
from rag_pipeline.core.vectorstore import FAISSIndex
from rag_pipeline.llm._factory import LLMFactory


class RAGPipeline:
    """Two-step RAG pipeline that prevents hallucination.

    Example usage:
        pipeline = RAGPipeline()
        pipeline.load()                       # load model + index + LLM
        result = pipeline.query(
            "How do I create a Delta table?"
        )
        print(result["answer"])             # grounded answer from docs
        print(result["sources"])            # source chunks used

    The pipeline is stateful — once loaded, it keeps the embedding model,
    vector index, and LLM client in memory for fast subsequent queries.
    Load once, query many times.
    """

    def __init__(self, config: Optional[Any] = None):
        # Accept a pre-built config or load from config.yaml
        self.config = config or get_config()

        # Components are set in .load(), kept for reuse across queries.
        # Storing them here avoids re-initializing on every query.
        self.embedding_model: Optional[EmbeddingModel] = None
        self.index: Optional[FAISSIndex] = None
        self.llm: Optional[Any] = None

    def load(self) -> None:
        """Load all components (embedding model, vector index, LLM).

        Call this once before any queries. It:
        1. Loads the sentence-transformer model for encoding queries.
        2. Loads the FAISS index + metadata from disk.
        3. Creates the LLM client (Ollama, Anthropic, etc.) from config.

        If the index can't be loaded (e.g. data/ missing), self.index is None
        and .query() returns an error message instead of crashing.
        """
        # 1. Embedding model — encodes text → 768-dim vectors
        self.embedding_model = EmbeddingModel(
            model_name=self.config.embeddings.model
        )

        # Resolve data/ relative to this package, not cwd.
        # pipeline/rag.py → parent.parent = rag_pipeline/ → parent.parent = src/
        # So parent.parent.parent.parent = repo root where data/ lives.
        data_dir = Path(__file__).parent.parent.parent.parent / "data"
        index_path = str(data_dir / "delta_lake.index")
        meta_path = str(data_dir / "vectometa.jsonl")

        # 2. FAISS vector index — loaded once, reused for all queries.
        # If data/ was never created (ingestion step skipped), this fails silently.
        # query() checks self.index and returns a helpful error if None.
        try:
            self.index = FAISSIndex.load(index_path, meta_path)
        except Exception:
            self.index = None

        # 3. LLM client — created by the factory based on config.provider.
        # The factory returns an OllamaLLM, AnthropicLLM, OpenAILLM, or BedrockLLM.
        # Each implements the same .generate() and .generate_with_context() interface.
        factory = LLMFactory(provider=self.config.llm.provider)
        self.llm = factory.create(
            model=self.config.llm.model,
            base_url=self.config.llm.base_url,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            api_key=self.config.llm.api_key,
            strict_prompt=self.config.llm.strict_prompt,
        )

    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG pipeline with a question.

        Implements the Two-Step pattern:

        STEP 1 — RETRIEVE:
            - Encode the question to a 768-dim vector (embedding model).
            - Search FAISS for top-k nearest chunks (cosine similarity).
            - Collect the chunk texts into a single "context" string.

        STEP 2 — GENERATE:
            - Fill the strict_prompt template with context + question.
            - Call the LLM, which must answer using ONLY the context.
            - Return the answer along with source references.

        Args:
            question: A natural-language question about Databricks.

        Returns:
            dict with keys:
            - "question": the original question
            - "answer": LLM's grounded answer (or error message)
            - "sources": list of source chunks with relevance scores

        If self.index is None (not loaded), returns an error instead of crashing.
        If no chunks match, returns "No relevant context found."
        If the LLM returns empty, returns "I don't have enough information."
        """
        # Guard: index must be loaded (run pipeline.load() first).
        if self.index is None:
            return {
                "question": question,
                "answer": "Vector index not loaded. Have you ingested documentation?",
                "sources": [],
            }

        # ===== STEP 1: RETRIEVE =====
        # Convert question text → embedding vector (768 dimensions).
        query_embedding = self.embedding_model.encode_single(question)

        # Search FAISS for top-k most similar chunks.
        # distances: cosine similarity scores (closer to 1 = more similar).
        # indices: integer positions of matching chunks in the metadata array.
        distances, indices = self.index.search(query_embedding, k=self.config.top_k)

        # Build context string from retrieved chunks.
        # Each chunk is a dict with at least {"text": ...}.
        chunks = []
        for idx in indices[0]:
            if idx >= 0:  # -1 means no result at this position
                chunk = self.index.get_chunk(idx)
                chunks.append(chunk)

        # Join all chunks into one context string, separated by newlines.
        # This becomes the {context} placeholder in the strict_prompt template.
        context = "\n\n".join([c["text"] for c in chunks])

        if not context:
            return {
                "question": question,
                "answer": "No relevant context found.",
                "sources": [],
            }

        # ===== STEP 2: GENERATE =====
        # The LLM client fills the strict_prompt template and calls the API.
        # strict_prompt forces the LLM to answer ONLY from the retrieved context.
        # If the context doesn't contain the answer, the LLM must say so.
        answer = self.llm.generate_with_context(question, context) if self.llm else "LLM not available"

        return {
            "question": question,
            "answer": answer if answer else "I don't have enough information.",
            "sources": [
                {"text": c["text"][:200], "score": float(distances[0][i])}
                for i, c in enumerate(chunks)
            ],
        }