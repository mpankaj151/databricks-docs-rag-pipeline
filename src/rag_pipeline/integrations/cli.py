"""CLI wrapper for RAG pipeline."""
import argparse
import sys

from rag_pipeline.config import get_config
from rag_pipeline.pipeline.rag import RAGPipeline


def main():
    """CLI main entry point."""
    parser = argparse.ArgumentParser(description="Databricks RAG Pipeline CLI")
    parser.add_argument("question", nargs="?", help="Your question")
    parser.add_argument("--api-url", default="http://localhost:8000", help="RAG API URL")
    parser.add_argument("--use-api", action="store_true", help="Use REST API instead of loading pipeline")

    args = parser.parse_args()

    if not args.question:
        parser.print_help()
        return

    if args.use_api:
        import requests

        response = requests.post(
            f"{args.api_url}/rag",
            json={"question": args.question},
            timeout=120,
        )
        result = response.json()
    else:
        config = get_config()
        pipeline = RAGPipeline()
        pipeline.load()
        result = pipeline.query(args.question)

    print(f"\n{'='*60}")
    print(f"QUESTION: {result['question']}")
    print(f"{'='*60}")
    print(f"\nANSWER:\n{result['answer']}")
    print(f"\nSources: {len(result.get('sources', []))}")


if __name__ == "__main__":
    main()