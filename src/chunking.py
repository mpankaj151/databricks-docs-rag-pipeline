"""Token-aware text chunking for RAG pipeline"""
import json
from typing import List, Dict
from pathlib import Path
import tiktoken
from src.models import DocChunk


def get_tokenizer(model_name: str = "cl100k_base"):
    """Get tiktoken tokenizer for token counting"""
    return tiktoken.get_encoding(model_name)


def tokenize(text: str, tokenizer=None) -> List[int]:
    """Convert text to token IDs"""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    return tokenizer.encode(text, disallowed_special=())


def detokenize(tokens: List[int], tokenizer=None) -> str:
    """Convert token IDs back to text"""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    return tokenizer.decode(tokens)


def chunk_document(
    doc: Dict,
    chunk_size: int = 256,
    overlap: int = 50,
    tokenizer=None
) -> List[Dict]:
    """Split a document into overlapping chunks of specified token size.
    
    Args:
        doc: Document dict with doc_id, url, title, content
        chunk_size: Max tokens per chunk
        overlap: Number of overlapping tokens between consecutive chunks
        tokenizer: Optional tiktoken tokenizer instance
    
    Returns:
        List of chunk dicts with doc_id, chunk_id, text, metadata
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
    
    tokens = tokenize(doc["content"], tokenizer)
    chunks = []
    
    if not tokens:
        return chunks
    
    start = 0
    chunk_id = 0
    step = chunk_size - overlap
    
    # Safety: step must be positive to avoid infinite loop
    if step <= 0:
        step = max(1, chunk_size)
    
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = detokenize(chunk_tokens, tokenizer)
        
        chunks.append({
            "doc_id": doc["doc_id"],
            "chunk_id": chunk_id,
            "text": chunk_text,
            "metadata": {
                "url": doc.get("url", ""),
                "title": doc.get("title", ""),
                "chunk_index": chunk_id,
                "token_count": len(chunk_tokens)
            }
        })
        
        chunk_id += 1
        start += step
        
        # If we've consumed all tokens, stop
        if end >= len(tokens):
            break
    
    return chunks


def chunk_all_documents(
    input_path: str,
    output_path: str,
    chunk_size: int = 256,
    overlap: int = 50
) -> int:
    """Load raw docs and chunk them all.
    
    Returns:
        Number of chunks created
    """
    tokenizer = get_tokenizer()
    
    all_chunks = []
    
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
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
            f.write(json.dumps(chunk) + "\n")
    
    return len(all_chunks)
