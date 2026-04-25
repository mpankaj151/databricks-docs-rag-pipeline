"""CLI wrapper for RAG pipeline."""
import argparse
import sys

from rag_pipeline.config import get_config
from rag_pipeline.pipeline.rag import RAGPipeline


def _run_loop(pipeline, use_api, api_url):
    """Interactive chat loop."""
    print("\nDatabricks RAG Chat — type 'exit' to quit\n")
    while True:
        try:
            question = input("> ").strip()
            if question.lower() in ("exit", "quit", "q"):
                break
            if not question:
                continue

            if use_api:
                import requests

                response = requests.post(
                    f"{api_url}/rag",
                    json={"question": question},
                    timeout=120,
                )
                result = response.json()
            else:
                result = pipeline.query(question)

            print(f"\n{'─'*60}")
            print(f"ANSWER: {result['answer']}")
            print(f"Sources: {len(result.get('sources', []))}")
            print(f"{'─'*60}\n")
        except (KeyboardInterrupt, EOFError):
            break


def main():
    """CLI main entry point."""
    parser = argparse.ArgumentParser(description="Databricks RAG Pipeline CLI")
    parser.add_argument("question", nargs="?", help="Your question")
    parser.add_argument("--api-url", default="http://localhost:8000", help="RAG API URL")
    parser.add_argument("--use-api", action="store_true", help="Use REST API instead of loading pipeline")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive chat mode")

    args = parser.parse_args()

    if args.interactive:
        pipeline = RAGPipeline()
        pipeline.load()
        _run_loop(pipeline, args.use_api, args.api_url)
        return

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