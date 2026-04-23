"""
Step 1: DATA INGESTION
----------------------
In a RAG (Retrieval-Augmented Generation) pipeline, the first step is gathering the data 
you want the AI to know about. This could be PDFs, Notion pages, databases, or in this 
case, HTML documentation from a website.

The goal of this module is to take raw HTML pages and convert them into clean, 
plain text that the AI can easily read. We remove things like navigation bars, 
footers, and scripts because they add "noise" and don't contain actual knowledge.
"""
import json
import hashlib
from typing import List, Dict
from pathlib import Path
from bs4 import BeautifulSoup
import requests
from src.config import get_config


def fetch_databricks_docs(url: str, html: str = None) -> List[Dict]:
    """
    Parse a documentation HTML page and extract its core knowledge content.
    
    Args:
        url: The web address of the documentation page.
        html: Optional pre-fetched HTML (useful if we already downloaded it or for testing).
    
    Returns:
        A list containing a dictionary with the cleaned document text and metadata.
    """
    # 1. Fetch the raw HTML if it wasn't provided
    if html is None:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        html = response.text
    
    # 2. Use BeautifulSoup to parse the HTML tree structure
    soup = BeautifulSoup(html, "lxml")
    
    # 3. Find the main content area (heuristics for documentation sites)
    # We don't want the whole page, just the actual article.
    main = (
        soup.find("main") or 
        soup.find("article") or 
        soup.find("div", {"role": "main"}) or
        soup.find("body")
    )
    if not main:
        return []
    
    # 4. Clean up the content by removing "noisy" HTML tags
    # Navigation bars and scripts confuse the AI, so we delete them.
    for unwanted in main.find_all(["nav", "footer", "aside", "script", "style"]):
        unwanted.decompose()  # This completely removes the tag from the HTML tree
    
    # 5. Extract the title so we can tell the user which document answered their question
    title_tag = main.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "Untitled"
    
    # 6. Extract the plain text from the remaining HTML
    content = main.get_text(separator="\n", strip=True)
    
    if not content.strip():
        return []
    
    # 7. Generate a unique ID for this document based on its URL
    # This ensures if we ingest the same URL twice, it gets the same ID.
    doc_id = int(hashlib.md5(url.encode()).hexdigest()[:8], 16)
    
    # Return the structured data
    return [{
        "doc_id": doc_id,
        "url": url,
        "title": title,
        "content": content,
    }]


def save_raw_docs(docs: List[Dict], output_path: str):
    """
    Save the extracted plain text documents to a JSONL file on disk.
    JSONL means "JSON Lines" - each line is a separate, complete JSON object.
    This is great for processing large amounts of data without loading everything into memory.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")


def load_raw_docs(input_path: str) -> List[Dict]:
    """Load raw documents back from the JSONL file."""
    docs = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs
