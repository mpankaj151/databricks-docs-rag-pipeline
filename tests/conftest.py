"""Pytest fixtures and shared configuration."""
import sys
from pathlib import Path

import pytest

# Add src/ to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_pipeline.config import Config, reset_config


@pytest.fixture(autouse=True)
def reset_global_config():
    """Reset config before each test."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def sample_docs():
    """Sample Databricks documentation snippets."""
    return [
        {
            "doc_id": 1,
            "url": "https://docs.databricks.com/delta/intro.html",
            "title": "What is Delta Lake?",
            "content": "Delta Lake is an open source storage layer that brings ACID transactions "
            "to Apache Spark. It provides serializable isolation levels between concurrent "
            "reads and writes.",
        },
        {
            "doc_id": 2,
            "url": "https://docs.databricks.com/delta/tutorial.html",
            "title": "Delta Lake Tutorial",
            "content": "To create a Delta Lake table, use: "
            "spark.sql('CREATE TABLE my_table (id INT, name STRING) USING delta') "
            "Delta Lake supports MERGE, UPDATE, DELETE operations.",
        },
        {
            "doc_id": 3,
            "url": "https://docs.databricks.com/lakeflow/intro.html",
            "title": "What is Lakeflow?",
            "content": "Lakeflow is a declarative pipeline framework for building ETL pipelines "
            "on top of Delta Lake. It handles orchestration and error handling automatically.",
        },
    ]


@pytest.fixture
def sample_chunks(sample_docs):
    """Sample pre-chunked documents."""
    return [
        {
            "doc_id": 1,
            "chunk_id": 0,
            "text": "Delta Lake is an open source storage layer that brings ACID transactions to Apache Spark.",
            "metadata": {"url": "https://docs.databricks.com/delta/intro.html", "title": "What is Delta Lake?"},
        },
        {
            "doc_id": 1,
            "chunk_id": 1,
            "text": "It provides serializable isolation levels between concurrent reads and writes.",
            "metadata": {"url": "https://docs.databricks.com/delta/intro.html", "title": "What is Delta Lake?"},
        },
        {
            "doc_id": 2,
            "chunk_id": 0,
            "text": "To create a Delta Lake table, use: spark.sql('CREATE TABLE my_table USING delta')",
            "metadata": {"url": "https://docs.databricks.com/delta/tutorial.html", "title": "Delta Lake Tutorial"},
        },
        {
            "doc_id": 2,
            "chunk_id": 1,
            "text": "Delta Lake supports MERGE, UPDATE, DELETE operations for data manipulation.",
            "metadata": {"url": "https://docs.databricks.com/delta/tutorial.html", "title": "Delta Lake Tutorial"},
        },
        {
            "doc_id": 3,
            "chunk_id": 0,
            "text": "Lakeflow is a declarative pipeline framework for building ETL pipelines on Delta Lake.",
            "metadata": {"url": "https://docs.databricks.com/lakeflow/intro.html", "title": "What is Lakeflow?"},
        },
    ]


@pytest.fixture
def default_config():
    """Default pipeline configuration."""
    return Config()