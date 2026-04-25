"""CLI wrapper for the RAG pipeline.

Provides a command-line interface for querying Databricks documentation.

Usage:
    # Single query
    rag-cli "How do I create a Delta table?"

    # Interactive chat mode
    rag-cli -i
    > How do I create a Delta table?
    > exit

    # Via REST API (server must be running)
    rag-cli "How do I create a Delta table?" --use-api

Installation:
    Installed as the `rag-cli` entry point via pyproject.toml.
    Works from any directory after `pip install -e .`.
"""
import argparse
import sys

from rag_pipeline.config import get_config
from rag_pipeline.pipeline.rag import RAGPipeline


def _run_loop(pipeline, use_api: bool, api_url: str) -> None:
    """Interactive chat loop — reads questions from stdin, prints answers.

    Args:
        pipeline: Loaded RAGPipeline instance.
        use_api: If True, call the REST API instead of direct pipeline.
        api_url: REST API base URL (used when use_api=True).
    """
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


def main() -> None:
    """CLI entry point — parse args and run query or interactive loop."""
    parser = argparse.ArgumentParser(
        description="Databricks RAG Pipeline CLI",
        prog="rag-cli",
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Your question about Databricks",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="RAG REST API URL (for --use-api mode)",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Call the REST API server instead of loading pipeline directly",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Interactive chat mode — prompts for questions until 'exit'",
    )

    args = parser.parse_args()

    # Interactive mode: load once, loop forever.
    if args.interactive:
        pipeline = RAGPipeline()
        pipeline.load()
        _run_loop(pipeline, args.use_api, args.api_url)
        return

    # No question provided — show help.
    if not args.question:
        parser.print_help()
        return

    # Single query.
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