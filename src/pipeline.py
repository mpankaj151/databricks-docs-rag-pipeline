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
    
    def load(self, skip_llm: bool = False):
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
        if not skip_llm:
            self.llm = OllamaLLM()
    
    def query(self, question: str, skip_llm: bool = False) -> QueryResult:
        """Query the RAG pipeline"""
        start_time = time.time()
        
        # Retrieve chunks
        chunks = self.retriever.retrieve(question)
        
        # Rerank if enabled
        if self.reranker and len(chunks) > self.config.top_k // 2:
            chunks = self.reranker.rerank(question, chunks, top_k=self.config.top_k)
        
        # Generate answer
        answer = None
        if not skip_llm and self.llm:
            # Build context
            context = "\n\n---\n\n".join([chunk.text for chunk in chunks])
            try:
                answer = self.llm.generate_with_context(question, context)
            except Exception as e:
                answer = f"Error generating answer: {e}"
        
        latency_ms = (time.time() - start_time) * 1000
        
        return QueryResult(
            question=question,
            retrieved_chunks=chunks,
            answer=answer,
            latency_ms=latency_ms
        )
    
    def query_with_sources(self, question: str, skip_llm: bool = False) -> dict:
        """Query and return answer with source references"""
        result = self.query(question, skip_llm)
        
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


def get_pipeline(skip_llm: bool = False) -> RAGPipeline:
    """Get or create pipeline singleton"""
    pipeline = RAGPipeline()
    pipeline.load(skip_llm)
    return pipeline
