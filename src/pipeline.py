"""
ORCHESTRATION: The Pipeline
---------------------------
This file is the "Manager". It ties all the separate RAG steps together into one easy-to-use
system. When a user asks a question, this file coordinates the Retriever and the LLM.

The full flow of a user query:
1. User calls `pipeline.query("How do I make a table?")`
2. Pipeline asks `retriever` to find chunks matching the question.
3. (Optional) Pipeline asks `reranker` to re-score and re-order those chunks to ensure the 
   best ones are at the top.
4. Pipeline takes the text from those chunks, bundles them with the question, and sends 
   them to the `llm`.
5. The `llm` reads the chunks, formulates an answer, and returns it.
6. Pipeline packages the answer, the chunks (so we can show the user our sources!), and 
   timing data into a `QueryResult` object.
"""
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
        """Load all components into memory (Database, Models, etc.)"""
        # Load embedding model (Step 3)
        self.embedding_model = get_embedding_model(self.config.embedding_model)
        
        # Load FAISS index database (Step 4)
        self.index = FAISSIndex.load(
            self.config.faiss_index,
            self.config.vectometa
        )
        
        # Create retriever (Step 5)
        self.retriever = Retriever(
            self.index,
            self.embedding_model,
            self.config.top_k
        )
        
        # Load reranker if enabled (Advanced Retrieval)
        if self.config.use_reranker:
            self.reranker = Reranker(self.config.reranker_model)
        
        # Load AI Generator (Step 6)
        if not skip_llm:
            self.llm = OllamaLLM()
    
    def query(self, question: str, skip_llm: bool = False) -> QueryResult:
        """
        The main function. Takes a question, finds documents, and generates an answer.
        """
        start_time = time.time()
        
        # Step 5: Retrieve relevant document chunks from the vector database
        chunks = self.retriever.retrieve(question)
        
        # Advanced Step: Rerank chunks using a more powerful AI model to improve accuracy
        if self.reranker and len(chunks) > self.config.top_k // 2:
            chunks = self.reranker.rerank(question, chunks, top_k=self.config.top_k)
        
        # Step 6: Generate answer using the chunks as an "open book"
        answer = None
        if not skip_llm and self.llm:
            # Combine all chunk texts into one big string separated by dashed lines
            context = "\n\n---\n\n".join([chunk.text for chunk in chunks])
            try:
                answer = self.llm.generate_with_context(question, context)
            except Exception as e:
                answer = f"Error generating answer: {e}"
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Return everything, including the sources we used, so the user can verify the AI's claims
        return QueryResult(
            question=question,
            retrieved_chunks=chunks,
            answer=answer,
            latency_ms=latency_ms
        )
    
    def query_with_sources(self, question: str, skip_llm: bool = False) -> dict:
        """A convenience function to return the answer and a simplified list of source URLs."""
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
        """Keep a log of what users are asking so we can improve the system later."""
        Path(self.config.query_log).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.query_log, "a") as f:
            f.write(json.dumps({
                "question": question,
                "answer": result.answer,
                "latency_ms": result.latency_ms,
                "num_sources": len(result.retrieved_chunks)
            }) + "\n")


def get_pipeline(skip_llm: bool = False) -> RAGPipeline:
    """Helper to initialize and get the pipeline."""
    pipeline = RAGPipeline()
    pipeline.load(skip_llm)
    return pipeline
