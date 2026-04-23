"""Setup sample data for testing the pipeline"""
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import save_raw_docs
from src.chunking import chunk_all_documents
from src.embed import get_embedding_model, embed_chunks
from src.vectorstore import FAISSIndex


SAMPLE_DOCS = [
    {
        "doc_id": 1,
        "url": "https://docs.databricks.com/delta/introduction.html",
        "title": "What is Delta Lake?",
        "content": """Delta Lake is an open source storage layer that brings ACID 
transactions to Apache Spark and Big Data workloads. Delta Lake provides 
serializable isolation levels between concurrent reads and writes. This ensures 
that readers see consistent snapshots of data while writers can modify the table 
without conflicting with each other. Delta Lake stores data in Parquet format 
and maintains a transaction log that records every commit made to the table.

Key features of Delta Lake include:
1. ACID transactions - Ensures data consistency
2. Time travel - Query previous versions of data
3. Schema enforcement - Prevents bad data from being written
4. Audit logging - Track all changes made to the table
5. Merge, update, and delete operations - CRUD operations on tables"""
    },
    {
        "doc_id": 2,
        "url": "https://docs.databricks.com/delta/tutorial.html",
        "title": "Getting Started with Delta Lake",
        "content": """To create a Delta Lake table, you can use either Spark SQL or 
the Delta Lake API in Python, Scala, or Java. In Python, use the following code:

spark.sql("CREATE TABLE my_table (id INT, name STRING) USING delta")

Or using the DataFrame API:

df.write.format("delta").save("/mnt/delta/table")

Delta Lake tables support various operations including:
- INSERT: Add new rows
- UPDATE: Modify existing rows
- DELETE: Remove rows
- MERGE: Upsert data from another table

You can also use time travel to query previous versions:
df = spark.read.format("delta").option("versionAsOf", 5).load("/mnt/table")"""
    },
    {
        "doc_id": 3,
        "url": "https://docs.databricks.com/lakeflow/introduction.html",
        "title": "What is Lakeflow?",
        "content": """Lakeflow is Databricks' declarative pipeline framework for 
building ETL pipelines on top of Delta Lake. Lakeflow allows you to define 
data transformation pipelines using a declarative syntax, handling orchestration, 
error handling, and monitoring automatically.

Key components of Lakeflow:
1. Pipelines - Define your ETL logic declaratively
2. Targets - Define where data should be written
3. Sources - Define input data sources
4. Transformations - Define data transformations

Lakeflow automatically manages:
- Task scheduling and orchestration
- Data quality validation
- Error handling and retries
- Monitoring and alerting"""
    }
]


def main():
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Save raw docs
    print("Saving sample docs...")
    save_raw_docs(SAMPLE_DOCS, "data/docs_raw.jsonl")
    
    # Chunk docs
    print("Chunking documents...")
    num_chunks = chunk_all_documents(
        "data/docs_raw.jsonl",
        "data/docs_chunks.jsonl",
        chunk_size=256,
        overlap=50
    )
    print(f"Created {num_chunks} chunks")
    
    # Generate embeddings
    print("Generating embeddings...")
    embed_chunks(
        "data/docs_chunks.jsonl", 
        "data/embeddings.npy"
    )
    
    # Build FAISS index
    print("Building FAISS index...")
    import numpy as np
    embeddings = np.load("data/embeddings.npy")
    
    chunks = []
    with open("data/docs_chunks.jsonl", "r") as f:
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
    index.save("data/delta_lake.index", "data/vectometa.jsonl")
    
    print("Done! Run 'python -m src.repl --no-llm' to query the RAG without LLM, or 'python -m src.repl' to run fully.")


if __name__ == "__main__":
    main()
