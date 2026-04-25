"""Setup sample data for testing the pipeline."""
import json
import sys
from pathlib import Path

from rag_pipeline.core.ingest import save_raw_docs
from rag_pipeline.core.chunking import chunk_text
from rag_pipeline.core.embed import EmbeddingModel
from rag_pipeline.core.vectorstore import FAISSIndex


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
5. Merge, update, and delete operations - CRUD operations on tables""",
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
df = spark.read.format("delta").option("versionAsOf", 5).load("/mnt/table")""",
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
- Monitoring and alerting""",
    },
]


def chunk_all_documents(
    raw_path: str,
    chunks_path: str,
    chunk_size: int = 256,
    overlap: int = 50,
) -> int:
    """Load docs, chunk them, and write to output file."""
    from rag_pipeline.core.ingest import load_raw_docs

    docs = load_raw_docs(raw_path)
    all_chunks = []

    for doc in docs:
        text_chunks = chunk_text(doc["content"], chunk_size, overlap)
        for i, text in enumerate(text_chunks):
            all_chunks.append(
                {
                    "doc_id": doc["doc_id"],
                    "chunk_id": i,
                    "text": text,
                    "metadata": {
                        "url": doc["url"],
                        "title": doc["title"],
                    },
                }
            )

    Path(chunks_path).parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_path, "w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    return len(all_chunks)


def embed_chunks(chunks_path: str, embeddings_path: str) -> None:
    """Embed chunks from file and save to numpy file."""
    import numpy as np

    chunks = []
    with open(chunks_path) as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    model = EmbeddingModel()
    vectors = model.encode([c["text"] for c in chunks], show_progress=False)
    np.save(embeddings_path, vectors)


def main():
    """Build the full vector index from sample docs."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # 1. Save raw docs
    print("Saving sample docs...")
    save_raw_docs(SAMPLE_DOCS, "data/docs_raw.jsonl")

    # 2. Chunk
    print("Chunking documents...")
    num_chunks = chunk_all_documents(
        "data/docs_raw.jsonl",
        "data/docs_chunks.jsonl",
        chunk_size=256,
        overlap=50,
    )
    print(f"Created {num_chunks} chunks")

    # 3. Embed
    print("Generating embeddings...")
    embed_chunks("data/docs_chunks.jsonl", "data/embeddings.npy")

    # 4. Build FAISS index
    print("Building FAISS index...")
    import numpy as np
    import faiss

    embeddings = np.load("data/embeddings.npy")
    faiss.normalize_L2(embeddings)

    chunks = []
    with open("data/docs_chunks.jsonl") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    metadata = [
        {
            "doc_id": c["doc_id"],
            "chunk_id": c["chunk_id"],
            "text": c["text"],
            "metadata": c["metadata"],
        }
        for c in chunks
    ]

    index = FAISSIndex(dimension=embeddings.shape[1])
    index.build(embeddings, metadata)
    index.save("data/delta_lake.index", "data/vectometa.jsonl")

    print(f"Done! Index built with {index.ntotal} chunks.")
    print("Run 'rag-cli \"What is Delta Lake?\"' to query the pipeline.")


if __name__ == "__main__":
    main()