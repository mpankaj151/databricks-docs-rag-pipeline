"""Tests for documentation ingestion"""
import pytest
import json
from pathlib import Path
from src.ingest import fetch_databricks_docs, save_raw_docs, load_raw_docs


def test_fetch_databricks_docs_parses_html():
    """Should extract title and content from HTML"""
    html = "<html><body><main><article><h1>Delta Lake</h1><p>Content here</p></article></main></body></html>"
    docs = fetch_databricks_docs("https://example.com/test", html)
    assert len(docs) == 1
    assert docs[0]["title"] == "Delta Lake"
    assert "Content here" in docs[0]["content"]


def test_fetch_databricks_docs_removes_nav():
    """Should strip navigation and script elements"""
    html = "<html><body><main><h1>Title</h1><nav>Nav</nav><p>Body text</p><script>var x=1;</script></main></body></html>"
    docs = fetch_databricks_docs("https://example.com/nav-test", html)
    assert len(docs) == 1
    assert "Nav" not in docs[0]["content"]
    assert "var x" not in docs[0]["content"]
    assert "Body text" in docs[0]["content"]


def test_fetch_databricks_docs_generates_stable_id():
    """Same URL should produce the same doc_id"""
    html = "<html><body><main><h1>Test</h1><p>Content</p></main></body></html>"
    docs1 = fetch_databricks_docs("https://example.com/stable", html)
    docs2 = fetch_databricks_docs("https://example.com/stable", html)
    assert docs1[0]["doc_id"] == docs2[0]["doc_id"]


def test_save_and_load_raw_docs(tmp_path):
    """Should round-trip docs through JSONL"""
    docs = [
        {"doc_id": 1, "url": "https://example.com", "title": "Test", "content": "Hello"},
        {"doc_id": 2, "url": "https://example.com/2", "title": "Test 2", "content": "World"},
    ]
    output = str(tmp_path / "docs.jsonl")
    save_raw_docs(docs, output)
    loaded = load_raw_docs(output)
    assert len(loaded) == 2
    assert loaded[0]["title"] == "Test"
    assert loaded[1]["content"] == "World"
