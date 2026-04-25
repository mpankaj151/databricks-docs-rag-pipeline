"""Data ingestion for Databricks documentation.

Fetches HTML pages, extracts clean text content, and saves to JSONL.
"""
import json
import hashlib
from typing import List, Dict, Optional
from pathlib import Path

from bs4 import BeautifulSoup
import requests

from rag_pipeline.core.chunking import chunk_text


def fetch_databricks_docs(url: str, html: Optional[str] = None) -> List[Dict]:
    """Parse a documentation HTML page and extract core knowledge content.

    Args:
        url: The web address of the documentation page.
        html: Optional pre-fetched HTML (useful for testing).

    Returns:
        A list with a dictionary per document containing cleaned text and metadata.
    """
    if html is None:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        html = response.text

    soup = BeautifulSoup(html, "lxml")

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", {"role": "main"})
        or soup.find("body")
    )
    if not main:
        return []

    for unwanted in main.find_all(["nav", "footer", "aside", "script", "style"]):
        unwanted.decompose()

    title_tag = main.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "Untitled"

    content = main.get_text(separator="\n", strip=True)

    if not content.strip():
        return []

    doc_id = int(hashlib.md5(url.encode()).hexdigest()[:8], 16)

    return [
        {
            "doc_id": doc_id,
            "url": url,
            "title": title,
            "content": content,
        }
    ]


def save_raw_docs(docs: List[Dict], output_path: str) -> None:
    """Save extracted documents to a JSONL file.

    JSONL: each line is a separate JSON object.
    Good for processing large datasets without loading everything into memory.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")


def load_raw_docs(input_path: str) -> List[Dict]:
    """Load raw documents from a JSONL file."""
    docs = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def ingest_documents(
    docs: List[Dict],
    chunk_size: int = 256,
    overlap: int = 50,
) -> List[Dict]:
    """Split documents into overlapping chunks for embedding.

    Args:
        docs: List of documents from load_raw_docs.
        chunk_size: Target chunk size in tokens.
        overlap: Token overlap between chunks.

    Returns:
        List of chunk dictionaries ready for embedding.
    """
    chunks = []
    for doc in docs:
        text_chunks = chunk_text(
            doc["content"],
            chunk_size=chunk_size,
            overlap=overlap,
        )
        for i, text in enumerate(text_chunks):
            chunks.append(
                {
                    "doc_id": doc["doc_id"],
                    "chunk_id": i,
                    "text": text,
                    "metadata": {
                        "url": doc["url"],
                        "title": doc["title"],
                        "source": "databricks_docs",
                    },
                }
            )
    return chunks