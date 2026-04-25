"""REST API server for RAG pipeline."""
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_pipeline.config import get_config
from rag_pipeline.pipeline.rag import RAGPipeline
from rag_pipeline.pipeline.keyword_detector import KeywordDetector


# Request/Response models
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class Source(BaseModel):
    text: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]
    latency_ms: float


# Global state
pipeline = None
keyword_detector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pipeline on startup."""
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


@app.get("/")
def root():
    """Root endpoint."""
    return {"message": "Databricks RAG Pipeline API"}


@app.get("/health")
def health():
    """Health check."""
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None,
    }


@app.get("/tool")
def get_tool_definition():
    """Get tool definition for LLM integration."""
    config = get_config()
    return {
        "tool": {
            "name": "search_databricks_docs",
            "description": "Search Databricks Delta Lake documentation",
            "auto_trigger": config.integrations.tool.auto_trigger,
            "keywords": config.integrations.tool.keywords,
        }
    }


@app.post("/tool/execute")
def execute_tool(request: dict):
    """Execute tool directly."""
    question = request.get("question", "")
    if not question:
        raise HTTPException(status_code=400, detail="question required")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    return pipeline.query(question)


@app.post("/rag", response_model=QueryResponse)
def query(request: QueryRequest):
    """RAG endpoint."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    result = pipeline.query(request.question)
    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
        latency_ms=0.0,
    )


def main():
    """Run the server."""
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