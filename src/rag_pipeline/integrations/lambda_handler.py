"""AWS Lambda handler for RAG pipeline."""
import json


def handler(event, context):
    """AWS Lambda entry point."""
    question = event.get("question") or (event.get("body") or {}).get("question")

    if not question:
        return {"statusCode": 400, "body": json.dumps({"error": "question is required"})}

    try:
        from rag_pipeline.config import get_config
        from rag_pipeline.pipeline.rag import RAGPipeline

        config = get_config()
        pipeline = RAGPipeline()
        pipeline.load()
        result = pipeline.query(question)

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