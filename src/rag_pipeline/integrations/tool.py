"""Tool/MCP definition for the RAG pipeline.

Provides a Tool definition that LLMs (Claude, GPT) can call directly,
enabling RAG to be used as a tool in agentic workflows (LangChain, MCP, etc.).

How it fits into agentic workflows:
    1. The tool definition is registered with the LLM agent.
    2. When the user's question matches auto-trigger keywords,
       the agent decides to call the tool.
    3. execute() runs RAG and returns the grounded answer.
    4. The LLM uses the answer to respond to the user.

Two execution paths:
    - REST API (api_url set): calls POST /tool/execute on the REST API server.
      Good for distributed setups where the server runs separately.
    - Direct (api_url empty): calls RAGPipeline.query() in-process.
      No server needed — simpler for local/MCP setups.

Auto-trigger:
    The KeywordDetector checks if incoming questions contain
    Databricks-related keywords. If yes, the tool should be called.
    Registration tools use should_auto_trigger() to configure this.
"""
import requests

from rag_pipeline.pipeline.keyword_detector import KeywordDetector


class ToolDefinition:
    """RAG tool definition for MCP/LangChain agents.

    Attributes:
        api_url: REST API URL. If set, calls the API server.
            If empty, calls RAGPipeline directly in-process.
        keyword_detector: Detects Databricks keywords in questions.

    Example registration (LangChain):
        from rag_pipeline.integrations.tool import ToolDefinition
        tool = ToolDefinition()
        get_langchain_tool = get_tool(tool.get_definition(), tool.execute)
    """

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.keyword_detector = KeywordDetector()

    def get_definition(self) -> dict:
        """Return the tool definition for registration with an LLM agent.

        This dict follows the MCP tool definition schema.
        LLM agents use it to know:
        - Tool name: "search_databricks_docs"
        - When to call it: description tells the trigger conditions
        - How to call it: input_schema defines the JSON payload

        Returns:
            dict with name, description, input_schema.
        """
        return {
            "name": "search_databricks_docs",
            "description": (
                "Search Databricks Delta Lake and Lakeflow documentation. "
                "Use when user asks about: Delta Lake, Databricks, Lakeflow, Spark SQL, "
                "data pipelines, tables, CREATE TABLE, MERGE, UPSERT, or any "
                "data engineering on Databricks."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": (
                            "The technical question to search Databricks docs for. "
                            "Be specific: e.g. 'how to create a Delta table with generated columns'."
                        ),
                    }
                },
                "required": ["question"],
            },
        }

    def should_auto_trigger(self, question: str) -> bool:
        """Check if a question should trigger RAG (has Databricks keywords).

        Args:
            question: The user's raw question.

        Returns:
            True if keywords like "delta lake", "databricks" etc. are present.
            False means RAG probably isn't needed.
        """
        return self.keyword_detector.should_use_rag(question)

    def execute(self, question: str) -> dict:
        """Execute the RAG tool with a question.

        Calls REST API or RAGPipeline directly depending on api_url.

        Args:
            question: The question to answer.

        Returns:
            dict with keys: question, answer, sources.
        """
        if self.api_url:
            # Distributed: call the REST API server.
            response = requests.post(
                f"{self.api_url}/tool/execute",
                json={"question": question},
                timeout=120,
            )
            response.raise_for_status()
            return response.json()
        else:
            # Local: call RAGPipeline directly — no server needed.
            from rag_pipeline.pipeline.rag import RAGPipeline

            pipeline = RAGPipeline()
            pipeline.load()
            return pipeline.query(question)

    def get_keywords(self) -> list:
        """Return the list of auto-trigger keywords."""
        return self.keyword_detector.keywords