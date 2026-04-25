"""Chunking utilities for RAG pipeline."""
import tiktoken
from typing import List


def get_tokenizer(model: str = "cl100k_base"):
    """Get tiktoken tokenizer."""
    return tiktoken.get_encoding(model)


def tokenize(text: str, tokenizer=None) -> List[int]:
    """Convert text to tokens."""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    return tokenizer.encode(text, disallowed_special=())


def chunk_text(text: str, chunk_size: int = 256, overlap: int = 50, tokenizer=None) -> List[str]:
    """Split text into overlapping chunks."""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    
    tokens = tokenize(text, tokenizer)
    chunks = []
    
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = start + chunk_size - overlap
        
        if overlap == 0:
            break
    
    return chunks