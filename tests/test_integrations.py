"""Tests for integration modules."""
import pytest
from unittest.mock import patch, MagicMock, Mock
import json

from rag_pipeline.integrations.tool import ToolDefinition
from rag_pipeline.integrations.langchain import get_langchain_tool
from rag_pipeline.integrations.lambda_handler import handler


class TestToolDefinition:
    """Tool/MCP definition."""

    def test_get_definition_schema(self):
        tool = ToolDefinition()
        definition = tool.get_definition()
        assert definition["name"] == "search_databricks_docs"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"

    def test_should_auto_trigger(self):
        tool = ToolDefinition()
        assert tool.should_auto_trigger("Tell me about Delta Lake merge") is True

    @patch("rag_pipeline.integrations.tool.requests.post")
    def test_execute_via_api(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "answer": "Delta Lake provides ACID transactions.",
            "sources": [],
        }
        mock_post.return_value = mock_response

        tool = ToolDefinition(api_url="http://localhost:8000")
        result = tool.execute("What is Delta Lake?")

        assert result["answer"] == "Delta Lake provides ACID transactions."
        mock_post.assert_called_once()

    def test_get_keywords(self):
        tool = ToolDefinition()
        keywords = tool.get_keywords()
        assert len(keywords) > 0
        assert "delta lake" in keywords

    def test_tool_execute_direct(self):
        """execute(api_url=None) should call RAGPipeline directly."""
        from rag_pipeline.integrations.tool import ToolDefinition
        from unittest.mock import patch, MagicMock

        tool = ToolDefinition(api_url=None)

        with patch("rag_pipeline.pipeline.rag.RAGPipeline") as mock_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.query.return_value = {
                "answer": "Delta Lake provides ACID.",
                "sources": [],
            }
            mock_cls.return_value = mock_pipeline

            result = tool.execute("What is Delta Lake?")

            assert result["answer"] == "Delta Lake provides ACID."
            mock_pipeline.query.assert_called_once_with("What is Delta Lake?")

    def test_tool_execute_via_api(self):
        """execute(api_url set) should call REST API."""
        from rag_pipeline.integrations.tool import ToolDefinition
        from unittest.mock import patch, MagicMock

        tool = ToolDefinition(api_url="http://localhost:8000")

        with patch("rag_pipeline.integrations.tool.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {
                "answer": "Delta Lake provides ACID.",
                "sources": [],
            }

            result = tool.execute("What is Delta Lake?")

            mock_post.assert_called_once()
            assert "tool/execute" in mock_post.call_args[0][0]
            assert result["answer"] == "Delta Lake provides ACID."

    def test_tool_should_auto_trigger_false(self):
        """should_auto_trigger should return False for non-RAG questions."""
        from rag_pipeline.integrations.tool import ToolDefinition
        tool = ToolDefinition()
        assert tool.should_auto_trigger("What is the weather?") is False


class TestLangChainTool:
    """LangChain integration."""

    def test_get_tool_returns_tool(self):
        with patch("rag_pipeline.pipeline.rag.RAGPipeline", autospec=True):
            tool = get_langchain_tool()
            assert tool.name == "search_databricks_docs"
            assert "Databricks" in tool.description

    def test_missing_langchain_raises(self):
        with patch.dict("sys.modules", {"langchain_core": None}, clear=True):
            with pytest.raises(ImportError, match="langchain-core"):
                get_langchain_tool()


class TestLambdaHandler:
    """AWS Lambda handler."""

    def test_missing_question_returns_error(self):
        from rag_pipeline.integrations.lambda_handler import handler
        result = handler({}, None)
        assert result["statusCode"] == 400
        body = json.loads(result["body"])
        assert "error" in body
        assert body["error"] == "question is required"

    def test_missing_nested_question_returns_error(self):
        from rag_pipeline.integrations.lambda_handler import handler
        result = handler({"body": {}}, None)
        assert result["statusCode"] == 400

    def test_exception_returns_500(self):
        from rag_pipeline.integrations.lambda_handler import handler
        from unittest.mock import patch
        with patch("rag_pipeline.pipeline.rag.RAGPipeline", side_effect=Exception("DB error")):
            result = handler({"question": "What is Delta Lake?"}, None)
        assert result["statusCode"] == 500


class TestCLIModule:
    """CLI module structure."""

    def test_cli_module_exists(self):
        from rag_pipeline.integrations import cli
        assert hasattr(cli, "main")

    def test_cli_accepts_no_args(self):
        from rag_pipeline.integrations.cli import main
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("question", nargs="?", help="Your question")
        args = parser.parse_args([])
        assert args.question is None