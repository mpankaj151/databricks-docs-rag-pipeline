"""Documentation ingestion from Databricks docs"""
import json
import hashlib
from typing import List, Dict
from pathlib import Path
from bs4 import BeautifulSoup
import requests
from src.config import get_config


def fetch_databricks_docs(url: str, html: str = None) -> List[Dict]:
    """Parse Databricks documentation HTML and extract content.
    
    Args:
        url: URL of the documentation page
        html: Optional pre-fetched HTML content (for testing/offline use)
    
    Returns:
        List of document dicts with doc_id, url, title, content
    """
    if html is None:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        html = response.text
    
    soup = BeautifulSoup(html, "lxml")
    
    # Find main content area — try multiple selectors for Databricks docs
    main = (
        soup.find("main") or 
        soup.find("article") or 
        soup.find("div", {"role": "main"}) or
        soup.find("body")
    )
    if not main:
        return []
    
    # Remove navigation, footer, sidebar, scripts
    for unwanted in main.find_all(["nav", "footer", "aside", "script", "style"]):
        unwanted.decompose()
    
    # Extract title
    title_tag = main.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "Untitled"
    
    # Get text content
    content = main.get_text(separator="\n", strip=True)
    
    if not content.strip():
        return []
    
    # Generate stable ID from URL
    doc_id = int(hashlib.md5(url.encode()).hexdigest()[:8], 16)
    
    return [{
        "doc_id": doc_id,
        "url": url,
        "title": title,
        "content": content,
    }]


def save_raw_docs(docs: List[Dict], output_path: str):
    """Save raw docs to JSONL file"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")


def load_raw_docs(input_path: str) -> List[Dict]:
    """Load raw docs from JSONL file"""
    docs = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs
