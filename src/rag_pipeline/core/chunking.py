"""Text chunking utilities for the RAG pipeline.

Before documents can be indexed, they must be split into small pieces ("chunks")
that:
1. Fit within the LLM's context window (typically 4K–128K tokens).
2. Are small enough to be relevant when retrieved (a 10-page doc is too noisy).
3. Overlap slightly so context isn't cut mid-sentence.

This module uses tiktoken to split text by tokens (not characters).
Token-based chunking is more accurate than char-based because:
- A token is ~0.75 characters on average, but varies by text.
- LLMs count tokens, not characters.
- 256 tokens ≈ 200 words ≈ 1 paragraph of Databricks docs.

Overlap strategy:
    Chunks slide across the text in steps of (chunk_size - overlap).
    With 256 tokens and 50 overlap, the step is 206 tokens.
    So chunk 1 covers tokens 0–255, chunk 2 covers 206–461, etc.
    Overlap ensures a phrase split across chunks still appears intact
    in at least one chunk.
"""
import tiktoken
from typing import List


def get_tokenizer(model: str = "cl100k_base") -> tiktoken.Encoding:
    """Get a tiktoken tokenizer for token counting.

    Args:
        model: Tiktoken encoding name. "cl100k_base" is the GPT-4/Claude
            tokenizer — a good general-purpose choice.

    Returns:
        A tiktoken Encoding object.
    """
    return tiktoken.get_encoding(model)


def tokenize(text: str, tokenizer=None) -> List[int]:
    """Convert text to a list of token IDs.

    Args:
        text: Input text string.
        tokenizer: tiktoken Encoding. If None, creates one (cached by tiktoken).

    Returns:
        List of integer token IDs.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
    return tokenizer.encode(text, disallowed_special=())


def chunk_text(
    text: str,
    chunk_size: int = 256,
    overlap: int = 50,
    tokenizer=None,
) -> List[str]:
    """Split text into overlapping token-based chunks.

    Algorithm: slide a window of chunk_size tokens across the text,
    yielding each window as a chunk. Move forward by (chunk_size - overlap)
    tokens between chunks.

    Args:
        text: Input text string to chunk.
        chunk_size: Target tokens per chunk.
        overlap: Tokens shared between consecutive chunks.
        tokenizer: tiktoken Encoding. If None, creates one.

    Returns:
        List of text strings, one per chunk.

    Example:
        chunks = chunk_text("hello world " * 100, chunk_size=10, overlap=2)
        # ["hello world hello world ...", "world hello world hello ...", ...]
        # Each chunk has 10 tokens, overlaps by 2 tokens.
    """
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
        # Move forward: step = chunk_size - overlap
        start = start + chunk_size - overlap

        # Non-overlapping mode: stop when overlap=0 and we've processed all tokens.
        if overlap == 0:
            break

    return chunks