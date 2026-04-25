"""Scrape Databricks documentation for RAG."""
import json
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from rag_pipeline.core.ingest import fetch_databricks_docs, save_raw_docs, load_raw_docs, ingest_documents
from rag_pipeline.core.embed import EmbeddingModel
from rag_pipeline.core.vectorstore import FAISSIndex
import faiss
import numpy as np


def get_internal_links(url: str, base_domain: str, html: str) -> set:
    """Find all internal links on a page."""
    soup = BeautifulSoup(html, "lxml")
    links = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(url, href)
        full_url = full_url.split("#")[0]
        if urlparse(full_url).netloc == base_domain and "/en/" in full_url:
            if not full_url.endswith((".pdf", ".zip", ".tar.gz", ".png", ".jpg")):
                links.add(full_url)
    return links


def crawl_databricks_docs(start_url: str, max_pages: int = 50) -> list:
    """Crawl Databricks docs starting from a URL."""
    print(f"Starting crawl at: {start_url} (max {max_pages} pages)")
    domain = urlparse(start_url).netloc
    visited = set()
    to_visit = [start_url]
    all_docs = []
    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)
        print(f"[{len(visited)}/{max_pages}] Scraping: {url}")
        try:
            response = session.get(url, timeout=10)
            if response.status_code != 200:
                continue
            docs = fetch_databricks_docs(url, html=response.text)
            if docs:
                all_docs.extend(docs)
            if len(visited) < max_pages:
                new_links = get_internal_links(url, domain, response.text)
                for link in new_links:
                    if link not in visited and link not in to_visit:
                        to_visit.append(link)
            time.sleep(0.5)
        except Exception as e:
            print(f"  Error: {e}")

    return all_docs


def main():
    seed_url = "https://docs.databricks.com/en/delta/index.html"
    max_pages = 50

    docs = crawl_databricks_docs(seed_url, max_pages=max_pages)
    print(f"\nScraped {len(docs)} documents.")

    if not docs:
        print("No documents found. Exiting.")
        return

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    raw_path = "data/docs_raw_large.jsonl"
    print(f"\nSaving raw docs to {raw_path}...")
    save_raw_docs(docs, raw_path)

    chunks_path = "data/docs_chunks_large.jsonl"
    print("Chunking documents...")
    chunks = ingest_documents(docs, chunk_size=256, overlap=50)
    with open(chunks_path, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")
    print(f"Created {len(chunks)} chunks.")

    embeddings_path = "data/embeddings_large.npy"
    print("\nGenerating embeddings...")
    model = EmbeddingModel()
    vectors = model.encode([c["text"] for c in chunks], show_progress=True)
    np.save(embeddings_path, vectors)

    print("\nBuilding FAISS index...")
    faiss.normalize_L2(vectors)
    index = FAISSIndex(dimension=vectors.shape[1])
    index.build(vectors, chunks)
    index.save("data/delta_lake.index", "data/vectometa.jsonl")

    print(f"\nSuccess! Index built with {index.ntotal} vector chunks.")
    print("Run 'rag-cli \"What is Delta Lake?\"' to test.")


if __name__ == "__main__":
    main()