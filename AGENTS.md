# Xagent Agent System

Xagent is a powerful and flexible framework for building and running AI-powered agents with support for various execution patterns, tools, memory management, and observability.

## Features

- **Agent Patterns**: ReAct, DAG plan-execute
- **Nested Agents**: Hierarchical agent execution with parent-child relationships
- **Tool System**: Built-in tools with auto-discovery mechanism
- **Memory Management**: LanceDB-based vector storage with semantic search
- **Observability**: Langfuse integration for tracing and monitoring
- **Real-time Communication**: WebSocket support for agent execution monitoring

## Architecture Overview

### Entry Points

Xagent has one main entrypoint:

**Web Interface (`src/xagent/web/`):**
- FastAPI-based web application with WebSocket support
- Real-time agent execution monitoring
- File upload and management
- DAG visualization
- API endpoints for agent operations

## Architecture Overview

### Core Components (`src/xagent/core/`)

**Agent System:**
- `agent.py` - Main Agent class with nested agent support and execution history
- `pattern/` - Agent execution patterns (ReAct, DAG plan-execute)
- `runner.py` - Agent execution engine
- `context.py` - Agent context management

**Graph System:**
- `graph.py` - Graph workflow execution engine with validation
- `node.py` - Node types (Start, End, Agent, Tool, etc.)
- `node_factory.py` - Node creation factory

**Tools System:**
- `adapters/` - Tool adapters for different frameworks
- `core/` - Core tool implementations (calculator, file operations, web search, etc.)
- Tool auto-discovery using `get_{tool_name}_tool()` naming convention

**Model Integration:**
- `llm/` - LLM provider implementations (OpenAI, Zhipu)
- Support for embedding models and reranking models

**Memory Management:**
- `storage/` - Storage manager and database operations
- `workspace.py` - Task workspace management with isolated working directories

**Observability:**
- Langfuse integration for tracing and monitoring
- Execution history and message tracking

### Available Tools

Xagent has two categories of tools:

**Basic Tools** (`src/xagent/core/tools/core/`):
- `calculator` - Mathematical expression evaluation
- `file_tool` - File operations (read, write, list, edit, delete)
- `workspace_file_tool` - Workspace file operations
- `python_executor` - Dynamic Python code execution
- `browser_use` - Browser automation
- `excel` - Excel file operations
- `document_parser` - Document parsing (PDF, DOCX, etc.)
- `image_tool` - Image processing

**Web & Search Tools** (`src/xagent/core/tools/core/`):
- `web_search` - Generic web search
- `image_web_search` - Image search functionality
- `zhipu_web_search` - Zhipu search integration
- `web_crawler` - Web crawling and content extraction

**RAG Tools** (`src/xagent/core/tools/core/RAG_tools/`):
- Document parsing and chunking
- Vector storage and retrieval (LanceDB)
- Knowledge base management
- Semantic search capabilities

**MCP Server Tools** (`src/xagent/core/tools/core/mcp/`):
- Model Context Protocol (MCP) server integration
- Standardized tool access via MCP protocol

### Custom Tools

Create custom tools by adding Python files following the naming convention:

```python
from langchain_core.tools import BaseTool, tool

def get_my_tool(_info: Optional[dict[str, str]] = None) -> BaseTool:
    """My custom tool description"""
    return tool(my_tool_function)
```

**Requirements:**
- Function name pattern: `get_{tool_name}_tool()`
- File location: `src/xagent/core/tools/core/`
- Return type: `BaseTool` instance from langchain_core
- No manual registration needed - auto-discovery on load

## Environment Configuration

Create a `.env` file based on `example.env` with required API keys:
```bash
OPENAI_API_KEY="your-openai-key"
DEEPSEEK_API_KEY="your-deepseek-key"
GOOGLE_API_KEY="your-google-api-key"
GOOGLE_CSE_ID="your-google-cse-id"
LANGFUSE_PUBLIC_KEY="your-langfuse-public-key"
LANGFUSE_SECRET_KEY="your-langfuse-secret-key"
```

### Optional Dependencies for Presentation Generation

If you plan to use the presentation generator feature (JavaScript-based PowerPoint creation via `execute_javascript_code` tool), you need to install Node.js and pptxgenjs:

```bash
# Ensure Node.js 20+ is installed
node --version

# Install pptxgenjs globally for presentation generation
npm install -g pptxgenjs@4.0.1

# Verify installation
npm root -g  # Should show path to global node_modules
ls $(npm root -g)/pptxgenjs  # Should show the package directory
```

**Note:** Without this installation, the `javascript_executor` tool will fail with "Cannot find module 'pptxgenjs'" when generating presentations. The pptxgenjs package is automatically installed in Docker/CI environments.

## Development Commands

### Installation and Setup
```bash
# Install the package in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src/xagent --cov-report=html

# Run specific test categories
pytest -m integration  # Integration tests
pytest -m slow         # Slow tests

# Run specific test files
pytest tests/core/agent/test_agent.py
pytest tests/web_integration/test_comprehensive.py
```

### Code Quality and Linting
```bash
# Format code with ruff
ruff format .

# Lint code with ruff
ruff check .

# Type checking with mypy
mypy src/xagent

# Run pre-commit hooks
pre-commit run --all-files
```

### Running the Application

Xagent has separate frontend and backend components:

**Backend (Web API):**
```bash
python -m xagent.web.__main__
# Runs on http://localhost:8000
```

**Frontend (Web UI):**
```bash
cd frontend
npm run dev    # Development mode with hot-reload
npm run build  # Production build
npm run start  # Production mode
# Frontend runs on http://localhost:3000
```

**Development Mode:**
Run both backend and frontend in separate terminals for full-stack development.
