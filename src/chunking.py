"""
Step 2: DATA CHUNKING
---------------------
After we extract the plain text (Ingestion), we have a problem: documents are often too long!
AI models have a "context window" (a maximum amount of text they can process at once). Also, 
if we pass a whole 10-page document to the AI just to answer a specific question, it gets confused.

The solution is "Chunking" - slicing the long document into smaller, bite-sized pieces (chunks).
When the user asks a question, we only find the 3-5 most relevant chunks and give those to the AI.

To ensure we don't accidentally cut a sentence exactly in half and lose its meaning, we use 
an "overlap". E.g., if Chunk 1 ends with "Delta Lake provides", Chunk 2 might start with 
"Delta Lake provides ACID transactions", ensuring the context is preserved across the cut.
"""
import json
from typing import List, Dict
from pathlib import Path
import tiktoken


def get_tokenizer(model_name: str = "cl100k_base"):
    """
    Get a tokenizer to count "tokens". 
    A token is roughly 3/4 of a word. AI models don't read words, they read tokens.
    We chunk by token count, not character count, to accurately match the AI's limits.
    """
    return tiktoken.get_encoding(model_name)


def tokenize(text: str, tokenizer=None) -> List[int]:
    """Convert a string of text into a list of integer token IDs."""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    return tokenizer.encode(text, disallowed_special=())


def detokenize(tokens: List[int], tokenizer=None) -> str:
    """Convert a list of token IDs back into human-readable text."""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    return tokenizer.decode(tokens)


def chunk_document(
    doc: Dict,
    chunk_size: int = 256,
    overlap: int = 50,
    tokenizer=None
) -> List[Dict]:
    """
    Split a document into overlapping chunks.
    
    Args:
        doc: The document dictionary from the ingestion step.
        chunk_size: How many tokens each chunk should contain.
        overlap: How many tokens should overlap between consecutive chunks.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
    
    # 1. Convert the entire document text into a massive list of tokens
    tokens = tokenize(doc["content"], tokenizer)
    chunks = []
    
    if not tokens:
        return chunks
    
    start = 0
    chunk_id = 0
    
    # The "step" is how far we move the window forward for the next chunk.
    # If chunk_size is 256 and overlap is 50, we move forward 206 tokens each time.
    step = chunk_size - overlap
    if step <= 0:
        step = max(1, chunk_size)  # Prevent infinite loops if overlap is configured badly
    
    # 2. Slide a window over the tokens to create the chunks
    while start < len(tokens):
        # Calculate where this chunk ends
        end = min(start + chunk_size, len(tokens))
        
        # Slice out the tokens for this chunk
        chunk_tokens = tokens[start:end]
        
        # Convert tokens back into plain text
        chunk_text = detokenize(chunk_tokens, tokenizer)
        
        # 3. Store the chunk along with metadata
        # Metadata is CRITICAL in RAG. We need to remember exactly which document 
        # this chunk came from, so we can cite our sources later!
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
        start += step  # Slide the window forward
        
        if end >= len(tokens):
            break
    
    return chunks


def chunk_all_documents(
    input_path: str,
    output_path: str,
    chunk_size: int = 256,
    overlap: int = 50
) -> int:
    """Read all ingested documents, chunk them, and save the chunks to a file."""
    tokenizer = get_tokenizer()
    all_chunks = []
    
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                chunks = chunk_document(doc, chunk_size, overlap, tokenizer)
                all_chunks.extend(chunks)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")
    
    return len(all_chunks)
