"""Keyword detector for auto-triggering RAG."""
from typing import List


class KeywordDetector:
    """Detects keywords to trigger RAG tool."""
    
    DEFAULT_KEYWORDS = [
        # Delta Lake
        "delta lake", "delta table", "deltatable", "delta.io",
        # Databricks
        "databricks", "dbricks",
        # Lakeflow
        "lakeflow", "lake flow", "declarative pipeline",
        # Spark
        "spark sql", "pyspark", "spark dataframe",
        # Data concepts
        "data pipeline", "lakehouse", "data lake", "etl",
        # SQL operations
        "create table", "merge into", "upsert", "delete from",
        # Table operations
        "optimize", "vacuum", "zorder", "compact",
    ]
    
    def __init__(self, keywords: List[str] = None):
        self.keywords = keywords or self.DEFAULT_KEYWORDS
    
    def contains_keywords(self, text: str) -> bool:
        """Check if text contains any keywords."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.keywords)
    
    def extract_found(self, text: str) -> List[str]:
        """Return which keywords were found."""
        text_lower = text.lower()
        return [kw for kw in self.keywords if kw in text_lower]
    
    def should_use_rag(self, text: str) -> bool:
        """Determine if RAG should be used."""
        return self.contains_keywords(text)