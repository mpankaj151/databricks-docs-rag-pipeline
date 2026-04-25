"""REST API server for the RAG pipeline.

Exposes FastAPI endpoints so external tools and services can query
the RAG pipeline over HTTP — no Python code needed on the client side.

Endpoints:
    GET  /         — API info
    GET  /health  — Health check (is pipeline loaded?)
    GET  /tool    — Tool/MCP definition for LLM registration
    POST /tool/execute — Execute tool directly
    POST /rag      — RAG query

Usage:
    # Terminal 1 — start the server
    python -m rag_pipeline.integrations.rest_api

    # Terminal 2 — query it
    curl -X POST http://localhost:8000/rag \
      -H "Content-Type: application/json" \
      -d '{"question": "How do I create a Delta table?"}'

The server loads the pipeline on startup (lifespan manager).
Each request calls pipeline.query() — no per-request loading cost.

For distributed setups: the server is the single RAG instance.
Clients (Slack bots, web apps, other CLIs) send HTTP requests.
"""
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_pipeline.config import get_config
from rag_pipeline.pipeline.rag import RAGPipeline
from rag_pipeline.pipeline.keyword_detector import KeywordDetector


# ─── Request / Response models ────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for POST /rag and POST /tool/execute."""

    question: str = Field(..., description="The question to answer")
    top_k: Optional[int] = Field(
        default=5,
        description="Number of chunks to retrieve (overrides config.top_k)",
    )


class Source(BaseModel):
    """A retrieved source chunk in the response."""

    text: str = Field(..., description="Source text (truncated to 200 chars)")
    score: float = Field(..., description="Cosine similarity score (0–1)")


class QueryResponse(BaseModel):
    """Response body from POST /rag."""

    question: str
    answer: str
    sources: List[Source]
    latency_ms: float = Field(..., description="Processing time in milliseconds")


# ─── Global state (loaded once on startup) ────────────────────────────────

pipeline: Optional[RAGPipeline] = None
keyword_detector: Optional[KeywordDetector] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pipeline and keyword detector once when the server starts.

    The lifespan context manager ensures this runs on startup and cleanup
    runs on shutdown. Between, the global pipeline instance is reused
    for all requests — no per-request loading.
    """
    global pipeline, keyword_detector

    config = get_config()
    keyword_detector = KeywordDetector(keywords=config.integrations.tool.keywords)
    pipeline = RAGPipeline()
    pipeline.load()

    yield


app = FastAPI(
    title="Databricks RAG Pipeline API",
    description="REST API for Delta Lake documentation RAG",
    version="0.1.0",
    lifespan=lifespan,
)


# ─── Endpoints ────────────────────────────────────────────────────────

@app.get("/")
def root() -> dict:
    """API info endpoint."""
    return {"message": "Databricks RAG Pipeline API"}


@app.get("/health")
def health() -> dict:
    """Health check — tells callers if the pipeline is loaded and ready.

    Returns:
        status: "healthy" if pipeline is loaded.
        pipeline_loaded: True if ready to handle requests.
    """
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None,
    }


@app.get("/tool")
def get_tool_definition() -> dict:
    """Return the tool definition for MCP/LangChain registration.

    The calling LLM uses this definition to know when and how
    to invoke the RAG tool. Includes the auto-trigger keywords.
    """
    config = get_config()
    return {
        "tool": {
            "name": "search_databricks_docs",
            "description": (
                "Search Databricks Delta Lake and Lakeflow documentation. "
                "Use when user asks about: Delta Lake, Databricks, Lakeflow, "
                "Spark SQL, data pipelines, tables, CREATE TABLE, MERGE, UPSERT."
            ),
            "auto_trigger": config.integrations.tool.auto_trigger,
            "keywords": config.integrations.tool.keywords,
        }
    }


@app.post("/tool/execute")
def execute_tool(request: dict) -> dict:
    """Execute the RAG tool directly (no wrapping).

    Used by MCP servers and LangChain agents that call tools
    by name rather than through /rag.
    """
    question = request.get("question", "")
    if not question:
        raise HTTPException(status_code=400, detail="question required")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    return pipeline.query(question)


@app.post("/rag", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Main RAG endpoint — retrieve and generate a grounded answer.

    Args:
        request.question: The user's question.

    Returns:
        QueryResponse with question, answer, sources, and latency.

    Error handling:
        503 if the pipeline hasn't been loaded yet.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    result = pipeline.query(request.question)
    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
        latency_ms=0.0,
    )


def main() -> None:
    """Run the FastAPI server with uvicorn.

    Uses config.integrations.rest_api.host and .port from config.yaml.
    """
    import uvicorn

    config = get_config()
    uvicorn.run(
        app,
        host=config.integrations.rest_api.host,
        port=config.integrations.rest_api.port,
        reload=False,
    )


if __name__ == "__main__":
    main()