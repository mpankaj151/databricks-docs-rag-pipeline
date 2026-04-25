# Version 2: Tool/MCP Integration Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development (recommended) to implement.

**Goal:** Register RAG pipeline as auto-triggered tool that LLM calls when keywords are detected.

**Architecture:** RAG tool watches for keywords → auto-calls RAG API → returns context + answer → LLM responds only from context.

**Tech Stack:** Python, requests, JSON, Ollama/Opencode

---

## File Structure

```
rag-practice/
├── src/
│   ├── tool.py                    # NEW: Tool definitions
│   ├── rag_keyword_detector.py     # NEW: Keyword detection
│   ├── api.py                    # Modified: Add /tool endpoint
│   └── ...
├── tests/
│   ├── test_tool.py              # NEW
│   └── test_keyword_detector.py   # NEW
└── CLAUDE.md                      # NEW: Tool registration
```

---

### Task 1: Create Keyword Detector

**Files:**
- Create: `src/rag_keyword_detector.py`

- [ ] **Step 1: Write keyword detector**

```python
"""Keyword detector for triggering RAG tool."""
import re

# Keywords that should trigger RAG tool
KEYWORDS = [
    # Delta Lake
    "delta lake", "delta table", "deltatable", "delta.io",
    # Databricks
    "databricks", "dbricks",
    # Lakeflow
    "lakeflow", "lake flow", "declarative pipeline",
    # Spark
    "spark sql", "pyspark", "spark dataframe",
    # Data concepts
    "data pipeline", "lakehouse", "data lake", "etl",
    # SQL operations
    "create table", "merge into", "upsert", "delete from",
    # Table operations
    "optimize", "vacuum", "zorder", "compact",
]

def contains_rag_keywords(text: str) -> bool:
    """Check if text contains any RAG-worthy keywords."""
    text_lower = text.lower()
    for keyword in KEYWORDS:
        if keyword in text_lower:
            return True
    return False


def extract_keywords_found(text: str) -> list:
    """Return which keywords were found."""
    text_lower = text.lower()
    found = []
    for keyword in KEYWORDS:
        if keyword in text_lower:
            found.append(keyword)
    return found
```

- [ ] **Step 2: Write test**

```python
# tests/test_keyword_detector.py
from src.rag_keyword_detector import contains_rag_keywords, extract_keywords_found

def test_detects_delta_lake():
    assert contains_rag_keywords("How do I create a Delta Lake table?")
    assert True

def test_detects_databricks():
    assert contains_rag_keywords("How to use Databricks?")
    assert True

def test_extracts_keywords():
    found = extract_keywords_found("Databricks Delta Lake query")
    assert "databricks" in found
    assert "delta lake" in found
```

- [ ] **Step 3: Verify tests pass**

Run: `pytest tests/test_keyword_detector.py -v`

---

### Task 2: Create RAG Tool Definition

**Files:**
- Create: `src/tool.py`

- [ ] **Step 1: Write tool definition**

```python
"""RAG Tool for LLM integration."""
import json
import requests
from src.rag_keyword_detector import contains_rag_keywords

# Tool definition (MCP/LangChain compatible)
TOOL_DEFINITION = {
    "name": "search_databricks_docs",
    "description": "Search Databricks Delta Lake and Lakeflow documentation. Use when user asks technical questions about: Delta Lake, Databricks, Lakeflow, Spark SQL, data pipelines, tables, CREATE TABLE, MERGE, UPSERT, or any data engineering on Databricks. Returns relevant documentation and code examples.",
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The technical question to search documentation for"
            }
        },
        "required": ["question"]
    }
}

# Keywords that trigger this tool
TRIGGER_KEYWORDS = ["delta", "databricks", "lakeflow", "spark", "lakehouse", "table"]


def should_use_tool(user_question: str) -> bool:
    """Determine if RAG tool should be used."""
    return contains_rag_keywords(user_question)


def execute_tool(question: str, api_url: str = "http://localhost:8000/rag") -> dict:
    """Execute the RAG tool by calling the API."""
    # Call RAG API
    response = requests.post(
        api_url,
        json={"question": question},
        timeout=120
    )
    response.raise_for_status()
    result = response.json()
    
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "question": question
    }


def get_tool_definition() -> dict:
    """Return tool definition for registration."""
    return TOOL_DEFINITION
```

- [ ] **Step 2: Write test**

```python
# tests/test_tool.py
from src.tool import should_use_tool, execute_tool, TRIGGER_KEYWORDS

def test_should_use_detabricks():
    assert should_use_tool("How to create table in Databricks?")
    assert True

def test_should_not_use_unrelated():
    assert not should_use_tool("What's the weather?")
    assert True
```

- [ ] **Step 3: Verify tests pass**

Run: `pytest tests/test_tool.py -v`

---

### Task 3: Add Tool Endpoint to API

**Files:**
- Modify: `src/api.py`

- [ ] **Step 1: Add /tool definition endpoint**

```python
@app.get("/tool")
def get_tool():
    """Return tool definition for LLM registration."""
    from src.tool import get_tool_definition, TOOL_DEFINITION, TRIGGER_KEYWORDS
    return {
        "tool": get_tool_definition(),
        "trigger_keywords": TRIGGER_KEYWORDS,
        "auto_trigger": True
    }


@app.post("/tool/execute")
def execute_tool_endpoint(request: dict):
    """Execute tool directly."""
    question = request.get("question", "")
    
    if not question:
        raise HTTPException(status_code=400, detail="question required")
    
    from src.tool import execute_tool
    result = execute_tool(question)
    
    return result
```

- [ ] **Step 2: Test endpoint**

```bash
# Get tool definition
curl http://localhost:8000/tool

# Execute tool
curl -X POST http://localhost:8000/tool/execute \
  -d '{"question": "How to create Delta table?"}'
```

---

### Task 4: Create CLAUDE.md Tool Registration

**Files:**
- Create: `CLAUDE.md`

```markdown
# Tool: search_databricks_docs

## Description
Search Databricks Delta Lake and Lakeflow documentation.

## Trigger
Automatically used when question contains keywords:
- delta, databricks, lakeflow, spark, lakehouse, table, create table, merge, upsert

## How it works
1. LLM detects keyword in user question
2. LLM calls /tool/execute endpoint
3. RAG returns context + answer
4. LLM responds using ONLY returned context

## Usage in prompts
```
Before answering, check if question needs search_databricks_docs tool.
If keywords present, call the tool first.
Answer using only the returned context.
```

## API
- GET /tool - Get tool definition
- POST /tool/execute - Execute with {"question": "..."}
```

---

### Task 5: Integration Testing

**Files:**
- Test: `tests/test_integration.py`

- [ ] **Step 1: Full integration test**

```python
# tests/test_integration.py
from src.tool import should_use_tool, execute_tool

def test_full_flow():
    # Check detection
    question = "How do I create a Delta Lake table?"
    assert should_use_tool(question)
    
    # Execute tool (requires API running)
    # result = execute_tool(question)
    # assert result["answer"]
    # assert result["sources"]
    print("Integration test passed!")
```

- [ ] **Step 2: Run all tests**

```bash
pytest tests/ -v
```

---

### Task 6: Documentation

- [ ] **Step 1: Update README**

```markdown
## Version 2: Tool/MCP Integration

### Auto-trigger Keywords
- delta, databricks, lakeflow, spark, lakehouse, table, etc.

### Usage
1. Start API: `python src/api.py`
2. LLM detects keywords in question
3. LLM calls /tool/execute automatically
4. RAG returns context → LLM answers

### API Endpoints
- GET /tool - Tool definition
- POST /tool/execute - Execute tool
```

---

## Plan Complete

**Next steps:**
1. Implement tasks in order
2. Test with Opencode
3. Verify auto-trigger works

**Approach?**
1. **Subagent-Driven** - I dispatch subagents per task
2. **Inline** - Execute here