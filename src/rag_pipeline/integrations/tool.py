"""Tool/MCP definition for RAG pipeline."""
import requests

from rag_pipeline.pipeline.keyword_detector import KeywordDetector


class ToolDefinition:
    """RAG Tool definition for MCP/LangChain."""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.keyword_detector = KeywordDetector()

    def get_definition(self) -> dict:
        """Get tool definition for registration."""
        return {
            "name": "search_databricks_docs",
            "description": "Search Databricks Delta Lake and Lakeflow documentation. "
            "Use when user asks about: Delta Lake, Databricks, Lakeflow, Spark SQL, "
            "data pipelines, tables, CREATE TABLE, MERGE, UPSERT, or any data engineering on Databricks.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The technical question to search documentation for",
                    }
                },
                "required": ["question"],
            },
        }

    def should_auto_trigger(self, question: str) -> bool:
        """Check if should auto-trigger."""
        return self.keyword_detector.should_use_rag(question)

    def execute(self, question: str) -> dict:
        """Execute the tool.

        Calls REST API if api_url is set, otherwise calls RAGPipeline directly.
        """
        if self.api_url:
            # REST API path — for distributed setups
            response = requests.post(
                f"{self.api_url}/tool/execute",
                json={"question": question},
                timeout=120,
            )
            response.raise_for_status()
            return response.json()
        else:
            # Direct path — no server needed, runs RAGPipeline in-process
            from rag_pipeline.pipeline.rag import RAGPipeline
            pipeline = RAGPipeline()
            pipeline.load()
            return pipeline.query(question)

    def get_keywords(self) -> list:
        """Get trigger keywords."""
        return self.keyword_detector.keywords