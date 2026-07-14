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

**Agent System (`src/xagent/core/agent/`):**
- `service.py` - `AgentService` facade used by web/chat/preview/builder entry points.
- `agent.py` - Core `Agent` definition for the only supported execution runtime.
- `execution_adapter.py` - Adapts `AgentService` calls into `AgentRunner` executions.
- `runner.py` - Execution engine with pause/resume/interruption and checkpoint loading.
- `runtime.py` - Cross-cutting pattern services: LLM calls, tool calls, tracing, checkpoints, outbound messages, and context compaction.
- `context/` - Message and execution context management.
- `pattern/` - Execution patterns: `single_call`, ReAct, DAG plan-execute, and auto routing.
- `checkpoint.py` - Trace-backed checkpoint persistence for resumable executions.

There is no v1/v2 runtime switch. Do not add new code under `agent_v2` or `agent_runtime`; both concepts have been collapsed into `core.agent`.

Execution mode mapping:
- `flash` -> `single_call`
- `balanced` -> ReAct
- `think` -> DAG plan-execute
- `auto` -> auto pattern selection between final-answer, ReAct, and DAG

Agent tasks built in the agent builder should default to memory disabled unless a future product change adds an explicit switch. Knowledge base/RAG grounding is separate from memory.

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

### Configuration System

**IMPORTANT**: All path-related and configuration settings SHALL try their best to use the unified configuration module at `src/xagent/config.py`.

**Core Principles:**
1. **Single source of truth** - All configuration goes through `config.py`
2. **No hardcoded paths** - Never use string literals like `"uploads"` or `"./data"`
3. **Environment variable support** - All paths must be configurable via `XAGENT_*` env vars
4. **No circular dependencies** - `config.py` has no dependencies on other core submodules

**Import Style:**
- **Source code** (`src/xagent/`): Use relative imports
  ```python
  from ..config import get_uploads_dir, get_storage_root
  ```
- **Test files** (`tests/`): Use absolute imports
  ```python
  from xagent.config import get_uploads_dir, get_storage_root
  ```

**Configuration Pattern:**
```python
def get_<config_name>() -> ReturnType:
    """Get <config> with environment variable override.

    Priority:
        1. <ENV_VAR> environment variable
        2. Computed default

    Returns:
        Description of return value
    """
    env_value = os.getenv(ENV_VAR)
    if env_value:
        return <process_env_value>(env_value)

    # Default computation
    return <compute_default>()
```

**Adding New Configuration:**
1. Add function to `src/xagent/config.py`
2. Add env var constant: `<ENV_VAR> = "XAGENT_<NAME>"`
3. Follow env var → default priority pattern
4. Update `example.env` with documentation
5. Add tests to `tests/core/test_config.py`

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

**Skill Documentation Access Tools** (`src/xagent/core/tools/adapters/vibe/skill_tools.py`):
- `read_skill_doc` - Read documentation from skill directories (SKILL.md, examples, etc.)
- `list_skill_docs` - List documentation files in skill directories (returns names and sizes)
- `fetch_skill_file` - Copy resource files from skill directories to workspace

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

### Local Development Conventions

**Commit and PR titles:**
- Use a short conventional prefix: `feat:`, `fix:`, `enh:`, `ref:`, or `chore:`.
- Prefer the same prefix in PR titles so split PRs are easy to scan.
- Keep branch names meaningful and task-oriented, for example `fix/remove-agent-v1` or `feat/agent-builder-preview`. Avoid generic agent/tool prefixes that do not describe the work.

**Local data locations:**
- The default storage root is `~/.xagent`, configured by `XAGENT_STORAGE_ROOT`.
- The default SQLite database is `~/.xagent/xagent.db`, unless `DATABASE_URL` is set.
- The default KB/RAG LanceDB path is `~/.xagent/data/lancedb`, unless `LANCEDB_PATH`/`LANCEDB_DIR` is set by the specific code path.
- User memory and knowledge base storage are different concepts. Agent/user memory uses `DynamicMemoryStoreManager`; when a LanceDB memory store is active it defaults to `~/.xagent/memory_store` after checking the legacy project `memory_store/` directory. KB/RAG collections use the RAG storage layer and the LanceDB path above.
- Uploaded files default to `src/xagent/web/uploads`, unless `XAGENT_UPLOADS_DIR` is set. External upload roots must go through `XAGENT_EXTERNAL_UPLOAD_DIRS`.
- Do not hardcode these paths in source or tests. Use `src/xagent/config.py` helpers or explicitly override env vars in tests.

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
# Install the package with core dependencies only (SQLite, basic PDF support)
pip install -e .

# Install development dependencies (requires pip >= 25.1 or uv)
pip install -e . --group dev

# Install optional extras for additional features
pip install -e ".[document-processing]" # Document processing libraries
pip install -e ".[ai-document]"         # AI-related document processing (docling)
pip install -e ".[postgresql]"          # PostgreSQL database driver
pip install -e ".[browser]"             # Browser automation (playwright)
pip install -e ".[chromadb]"            # ChromaDB vector database
pip install -e ".[milvus]"              # Milvus vector database
pip install -e ".[all]"                 # Install all optional extras

# For development with all features:
pip install -e ".[all]" --group dev

# For older pip versions, use uv instead:
# uv sync --group dev --extra all
```

**Optional Extras:**
| Extra | Description |
|-------|-------------|
| `document-processing` | document processing libraries (pdfplumber, unstructured, pymupdf, etc.) |
| `ai-document` | AI-related document processing (docling) |
| `postgresql` | PostgreSQL driver (uses psycopg2-binary; for production consider psycopg2) |
| `browser` | Browser automation (playwright) |
| `chromadb` | ChromaDB vector database (alternative to LanceDB) |
| `milvus` | Milvus vector database (alternative to LanceDB) |
| `all` | All optional extras combined |

**Note**: Pre-commit hooks are installed via `--group dev`, not as an optional extra.

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

### Docker Release Image Bumps

When bumping Docker release image tags, update all fixed Xagent images together:
- All fixed Xagent service images present in `docker-compose.yml`
- `docker/docker-compose.sandbox.boxlite.yml` `SANDBOX_IMAGE`
- `docker/docker-compose.sandbox.docker.yml` `SANDBOX_IMAGE`

Validate both the base compose file and sandbox overlays:
```bash
docker compose config --quiet
docker compose -f docker-compose.yml -f docker/docker-compose.sandbox.boxlite.yml config --quiet
docker compose -f docker-compose.yml -f docker/docker-compose.sandbox.docker.yml config --quiet
```

## Skills Configuration

Skills directories can be extended using the `XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS` environment variable:
- External directories are **appended** to default built-in and user directories
- Comma-separated list of paths
- Supports local directories, home directory expansion, and environment variables
- Non-existent paths are skipped with warnings
- Default directories are always loaded

Load order: built-in → user → external (later skills override earlier ones with the same name)

Examples:
```bash
# Single directory (appended to defaults)
XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS="/path/to/custom/skills"

# Multiple directories
XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS="/path/to/skills1,/path/to/skills2,~/skills"

# With path expansion
XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS="~/skills,$HOME/custom_skills,./local_skills"
```

See `src/xagent/skills/README.md` for details.
Run both backend and frontend in separate terminals for full-stack development.
