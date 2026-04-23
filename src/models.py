"""Pydantic data models for RAG pipeline"""
from typing import Optional
from pydantic import BaseModel


class DocChunk(BaseModel):
    """A chunk of text from a document"""
    doc_id: int
    chunk_id: int
    text: str
    metadata: dict


class RetrievedChunk(BaseModel):
    """A chunk retrieved from the vector store with similarity score"""
    doc_id: int
    chunk_id: int
    text: str
    metadata: dict
    score: float


class QueryResult(BaseModel):
    """Result of a RAG query including answer and sources"""
    question: str
    retrieved_chunks: list[RetrievedChunk]
    answer: Optional[str] = None
    latency_ms: Optional[float] = None
