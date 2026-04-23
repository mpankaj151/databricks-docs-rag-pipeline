"""Tests for LLM module"""
import pytest
from unittest.mock import patch, MagicMock
from src.llm import OllamaLLM, LLMError


@patch("src.llm.requests.post")
def test_llm_generate_success(mock_post):
    """Should return message content on success"""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {"content": "Test answer"}
    }
    mock_post.return_value = mock_response
    
    llm = OllamaLLM(api_key="")  # force ollama mode
    result = llm.generate("test prompt")
    
    assert result == "Test answer"
    mock_post.assert_called_once()
    assert "messages" in mock_post.call_args[1]["json"]


@patch("src.llm.requests.post")
def test_llm_openrouter_success(mock_post):
    """Should extract openrouter response correctly"""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "OpenRouter answer"}}]
    }
    mock_post.return_value = mock_response
    
    llm = OllamaLLM(api_key="test-key")
    assert llm.is_openrouter is True
    
    result = llm.generate("test prompt")
    
    assert result == "OpenRouter answer"
    mock_post.assert_called_once()
    
    # Check headers contain auth
    headers = mock_post.call_args[1]["headers"]
    assert "Authorization" in headers
    assert "test-key" in headers["Authorization"]


@patch("src.llm.requests.get")
def test_llm_check_connection_openrouter(mock_get):
    """Should verify connection to openrouter"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response
    
    llm = OllamaLLM(api_key="test-key")
    assert llm.check_connection() is True


@patch("src.llm.requests.get")
def test_llm_check_connection_ollama(mock_get):
    """Should verify connection to ollama"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"models": [{"name": "glm-5.1:cloud"}]}
    mock_get.return_value = mock_response
    
    llm = OllamaLLM(api_key="", model="glm-5.1:cloud")
    assert llm.check_connection() is True
