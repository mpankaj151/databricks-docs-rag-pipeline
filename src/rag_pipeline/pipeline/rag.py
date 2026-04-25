"""Main RAG Pipeline - Two-Step RAG to prevent hallucination."""
from typing import Optional, List, Dict, Any

from rag_pipeline.config import get_config
from rag_pipeline.core.embed import EmbeddingModel
from rag_pipeline.core.vectorstore import FAISSIndex
from rag_pipeline.llm._factory import LLMFactory


class RAGPipeline:
    """Two-step RAG pipeline that prevents hallucination.

    Step 1: Retrieve relevant chunks from the vector store.
    Step 2: Generate answer using ONLY the retrieved context.
    """

    def __init__(self, config: Optional[Any] = None):
        self.config = config or get_config()
        self.embedding_model: Optional[EmbeddingModel] = None
        self.index: Optional[FAISSIndex] = None
        self.llm: Optional[Any] = None

    def load(self) -> None:
        """Load all components (embedding model, vector index, LLM)."""
        self.embedding_model = EmbeddingModel(
            model_name=self.config.embeddings.model
        )

        index_path = "data/delta_lake.index"
        meta_path = "data/vectometa.jsonl"

        try:
            self.index = FAISSIndex.load(index_path, meta_path)
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

    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG pipeline with a question.

        Returns a dict with question, answer, and sources.
        """
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

        # Step 2: Generate — strict_prompt comes from the LLM client
        answer = self.llm.generate_with_context(question, context) if self.llm else "LLM not available"

        return {
            "question": question,
            "answer": answer if answer else "I don't have enough information.",
            "sources": [
                {"text": c["text"][:200], "score": float(distances[0][i])}
                for i, c in enumerate(chunks)
            ],
        }