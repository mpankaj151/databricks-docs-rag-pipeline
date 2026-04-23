"""Interactive REPL for querying the RAG pipeline"""
import sys
import argparse
from src.pipeline import get_pipeline


def main():
    parser = argparse.ArgumentParser(description="Databricks Delta Lake RAG REPL")
    parser.add_argument("--no-llm", action="store_true", help="Run without LLM (show retrieved chunks only)")
    args = parser.parse_args()

    print("Databricks Delta Lake RAG - Interactive REPL")
    if args.no_llm:
        print("Running in --no-llm mode (retrieval only)")
    print("Type 'quit' or 'exit' to exit\n")
    
    try:
        pipeline = get_pipeline(skip_llm=args.no_llm)
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        print("Have you run 'python scripts/setup_sample_data.py' yet?")
        return
    
    if not args.no_llm and pipeline.llm:
        if not pipeline.llm.check_connection():
            print("WARNING: LLM backend not available. Responses may fail.")
            print("If using Ollama, ensure it is running (`ollama serve`).")
    
    while True:
        try:
            question = input("> ").strip()
            
            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            result = pipeline.query(question, skip_llm=args.no_llm)
            
            if not args.no_llm:
                print(f"\nAnswer: {result.answer}\n")
            
            print(f"Sources ({len(result.retrieved_chunks)}):")
            for i, chunk in enumerate(result.retrieved_chunks, 1):
                title = chunk.metadata.get("title", "Untitled")
                score = chunk.score
                print(f"  {i}. {title} (score: {score:.3f})")
                if args.no_llm:
                    print(f"     Preview: {chunk.text[:100]}...")
            
            print(f"\nLatency: {result.latency_ms:.0f}ms\n")
            
            # Log query
            pipeline.log_query(question, result)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
