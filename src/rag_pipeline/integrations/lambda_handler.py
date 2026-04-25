"""AWS Lambda handler for RAG pipeline.

Uses module-level cache to reuse the loaded RAGPipeline across
warm Lambda invocations — avoids paying the cold-start cost
(embedding model load) on every call.
"""
import json

from rag_pipeline.pipeline.rag import RAGPipeline
from rag_pipeline.config import get_config

# Module-level singleton — persists across warm invocations
_pipeline = None


def handler(event, context):
    """AWS Lambda entry point.

    First cold call: loads embedding model + LLM (~30s).
    Subsequent warm calls: reuse already-loaded pipeline (~50ms).
    """
    global _pipeline

    question = event.get("question") or (event.get("body") or {}).get("question")

    if not question:
        return {"statusCode": 400, "body": json.dumps({"error": "question is required"})}

    try:
        # Load once, reuse forever
        if _pipeline is None:
            config = get_config()
            _pipeline = RAGPipeline()
            _pipeline.load()  # cold start cost paid here

        result = _pipeline.query(question)

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result),
        }

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}


if __name__ == "__main__":
    test_event = {"question": "What is Delta Lake?"}
    print(handler(test_event, None))