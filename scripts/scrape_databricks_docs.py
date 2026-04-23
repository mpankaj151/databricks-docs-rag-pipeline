"""Scrape a larger set of Databricks documentation for RAG evaluation"""
import sys
from pathlib import Path
import time
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import fetch_databricks_docs, save_raw_docs
from src.chunking import chunk_all_documents
from src.embed import embed_chunks
from src.vectorstore import FAISSIndex


def get_internal_links(url: str, base_domain: str, html: str) -> set:
    """Find all internal links on a page."""
    soup = BeautifulSoup(html, "lxml")
    links = set()
    
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        # Resolve relative URLs
        full_url = urljoin(url, href)
        
        # Strip fragment identifiers (#section)
        full_url = full_url.split('#')[0]
        
        # Only keep links within the same domain and under /en/
        if urlparse(full_url).netloc == base_domain and "/en/" in full_url:
            # Ignore PDF, ZIP, etc.
            if not full_url.endswith((".pdf", ".zip", ".tar.gz", ".png", ".jpg")):
                links.add(full_url)
                
    return links


def crawl_databricks_docs(start_url: str, max_pages: int = 50) -> list:
    """Crawl databricks docs starting from a URL up to max_pages."""
    print(f"Starting crawl at: {start_url} (Target: {max_pages} pages)")
    
    domain = urlparse(start_url).netloc
    visited = set()
    to_visit = [start_url]
    all_docs = []
    
    # Use a session for connection pooling
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    })
    
    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        
        if url in visited:
            continue
            
        visited.add(url)
        print(f"[{len(visited)}/{max_pages}] Scraping: {url}")
        
        try:
            # Fetch page
            response = session.get(url, timeout=10)
            if response.status_code != 200:
                continue
                
            html = response.text
            
            # Parse content using our ingest module
            docs = fetch_databricks_docs(url, html=html)
            if docs:
                all_docs.extend(docs)
            
            # Find new links to crawl
            if len(visited) < max_pages:
                new_links = get_internal_links(url, domain, html)
                for link in new_links:
                    if link not in visited and link not in to_visit:
                        to_visit.append(link)
            
            # Be nice to the server
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  Error fetching {url}: {e}")
            
    return all_docs


def main():
    # 1. Scrape 50 pages from the Databricks Delta documentation
    seed_url = "https://docs.databricks.com/en/delta/index.html"
    max_pages_to_scrape = 50
    
    docs = crawl_databricks_docs(seed_url, max_pages=max_pages_to_scrape)
    print(f"\nSuccessfully scraped {len(docs)} documents.")
    
    if not docs:
        print("No documents found. Exiting.")
        return
        
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # 2. Save raw docs
    raw_path = "data/docs_raw_large.jsonl"
    print(f"\nSaving raw docs to {raw_path}...")
    save_raw_docs(docs, raw_path)
    
    # 3. Chunk docs
    chunks_path = "data/docs_chunks_large.jsonl"
    print("Chunking documents...")
    num_chunks = chunk_all_documents(
        raw_path,
        chunks_path,
        chunk_size=256,
        overlap=50
    )
    print(f"Created {num_chunks} chunks.")
    
    # 4. Generate embeddings
    embeddings_path = "data/embeddings_large.npy"
    print("\nGenerating embeddings (this may take a minute)...")
    embed_chunks(chunks_path, embeddings_path)
    
    # 5. Build FAISS index
    print("\nBuilding FAISS index...")
    import numpy as np
    embeddings = np.load(embeddings_path)
    
    chunks = []
    with open(chunks_path, "r") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
                
    metadata = []
    for chunk in chunks:
        metadata.append({
            "doc_id": chunk["doc_id"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "metadata": chunk.get("metadata", {})
        })
    
    index = FAISSIndex(dimension=embeddings.shape[1])
    index.build(embeddings, metadata)
    
    # Save as the active index in config
    index_path = "data/delta_lake.index"
    meta_path = "data/vectometa.jsonl"
    index.save(index_path, meta_path)
    
    print(f"\nSuccess! Database updated with {index.ntotal} vector chunks.")
    print("Run 'python -m src.repl' to test your newly expanded knowledge base!")


if __name__ == "__main__":
    main()
