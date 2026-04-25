"""AWS Lambda handler for the RAG pipeline.

Deploy this as an AWS Lambda function to serve RAG queries
in a serverless, scalable production setup.

Cold start vs warm invocations:
    - Cold start: Lambda loads the embedding model (~30s overhead).
      This happens once when the container starts.
    - Warm invocations: the pipeline is already loaded.
      Subsequent calls take ~50ms.

The module-level _pipeline singleton enables warm invocation reuse:
    _pipeline is a global Python variable that persists across
    warm Lambda invocations (same container). Only the first
    call pays the cold-start cost.

Deployment:
    1. Package with dependencies (sentence-transformers, faiss-cpu, etc.).
    2. Set handler to: rag_pipeline.integrations.lambda_handler.handler
    3. Set memory: >= 2048 MB (embedding model needs ~1.5GB).
    4. Timeout: >= 60s (cold start can take 30-45s).
    5. Set OLLAMA_API_KEY (or ANTHROPIC_API_KEY) in Lambda env vars.

Event format:
    {"question": "How do I create a Delta table?"}
    or
    {"body": {"question": "How do I create a Delta table?"}}
"""
import json

from rag_pipeline.pipeline.rag import RAGPipeline
from rag_pipeline.config import get_config

# Module-level singleton — lives across warm Lambda invocations.
# Persists because Lambda containers are reused for warm calls.
_pipeline = None


def handler(event: dict, context) -> dict:
    """AWS Lambda entry point.

    Args:
        event: Dict with "question" key, or {"body": {"question": ...}}.
        context: Lambda context (unused, but required by Lambda).

    Returns:
        dict with statusCode and body (Lambda response format):
        - 200: success (body = JSON with question, answer, sources)
        - 400: missing question
        - 500: internal error
    """
    global _pipeline

    # Support both direct payload and API Gateway wrapped payloads.
    question = event.get("question") or (event.get("body") or {}).get("question")

    if not question:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "question is required"}),
        }

    try:
        # Load once on cold start, reuse on warm calls.
        if _pipeline is None:
            config = get_config()
            _pipeline = RAGPipeline()
            _pipeline.load()  # ~30s cold start cost paid here

        result = _pipeline.query(question)

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result),
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }


if __name__ == "__main__":
    test_event = {"question": "What is Delta Lake?"}
    print(handler(test_event, None))