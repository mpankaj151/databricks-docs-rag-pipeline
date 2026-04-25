"""LangChain tool integration."""
from typing import Optional

from rag_pipeline.pipeline.rag import RAGPipeline


def get_langchain_tool(rag_pipeline: Optional[RAGPipeline] = None):
    """Get a LangChain tool from the RAG pipeline.

    Usage:
        from rag_pipeline.integrations.langchain import get_langchain_tool
        from langchain.agents import AgentExecutor, create_openai_functions_agent
        from langchain_openai import ChatOpenAI

        tool = get_langchain_tool()
        llm = ChatOpenAI(model="gpt-4")
        agent = create_openai_functions_agent(llm, [tool], prompt)
        agent_executor = AgentExecutor(agent=agent, tools=[tool])
    """

    try:
        from langchain_core.tools import BaseTool
    except ImportError:
        raise ImportError("langchain-core is required for LangChain integration. Install with: pip install langchain-core")

    class DatabricksDocsSearchTool(BaseTool):
        """Search Databricks documentation via LangChain tool interface."""

        name: str = "search_databricks_docs"
        description: str = (
            "Search Databricks Delta Lake and Lakeflow documentation. "
            "Use when user asks about: Delta Lake, Databricks, Lakeflow, Spark SQL, "
            "data pipelines, tables, CREATE TABLE, MERGE, UPSERT, or any data engineering on Databricks. "
            "Returns relevant documentation and code examples."
        )

        def _run(self, question: str):
            if rag_pipeline is None:
                return "RAG pipeline not initialized"
            result = rag_pipeline.query(question)
            return result.get("answer", "No result")

    return DatabricksDocsSearchTool()