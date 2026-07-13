import json
import os

# Fix pathlib.Path compatibility issue for pytest
# In Python 3.11, pathlib.Path doesn't have _flavour but PosixPath does
# This is needed for pytest internal usage
import pathlib
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from dotenv import load_dotenv
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import (
    Function as ToolCallFunction,
)

from xagent.core.model import ChatModelConfig, EmbeddingModelConfig, RerankModelConfig
from xagent.core.tools.core.RAG_tools.storage import reset_rag_storage_for_tests
from xagent.core.tracing.langfuse import reset_langfuse_client

# YAML entrypoint has been removed, commenting out these imports
# from xagent.entrypoint.yaml.parser import MigrationManager
# from xagent.entrypoint.yaml.server import set_yaml_migration_manager

# ==========================================
# ENVIRONMENT AND PROJECT SETUP
# ==========================================


# On Windows this would shadow WindowsPath._flavour (Path precedes
# PureWindowsPath in WindowsPath's MRO) and break every Path() call with
# NotImplementedError, so it only applies on POSIX.
if (
    os.name != "nt"
    and not hasattr(pathlib.Path, "_flavour")
    and hasattr(pathlib.PosixPath, "_flavour")
):
    pathlib.Path._flavour = pathlib.PosixPath._flavour

# Load environment variables - try .env first, fallback to example.env
project_root = Path(__file__).parent.parent
env_file = project_root / ".env"
example_env_file = project_root / "example.env"

if env_file.exists():
    load_dotenv(env_file, override=True)  # Force override existing env vars
elif example_env_file.exists():
    # Don't override existing env vars (especially API keys from user's shell)
    load_dotenv(example_env_file, override=False)
else:
    print("Warning: Neither .env nor example.env file found")


def pytest_addoption(parser):
    parser.addoption(
        "--run-special",
        action="store_true",
        default=False,
        help="Run tests that require special conditions",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "docker: tests that require Docker daemon (run with --run-special)"
    )
    config.addinivalue_line(
        "markers",
        "real_rag: tests that require DashScope embedding API config",
    )
    config.addinivalue_line(
        "markers",
        "requires_network: tests that require network access (run with --run-special or set XAGENT_TESTS_ALLOW_NETWORK=1)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip Docker tests unless --run-special is specified.

    Also automatically skip tests that require unavailable external dependencies.
    Tests marked with `pytest.mark.real_rag` require DashScope embedding
    configuration. If this is not configured in the environment, the tests are
    automatically skipped rather than failing.

    Tests marked with `pytest.mark.requires_network` require network access.
    These tests are skipped unless --run-special is specified or
    XAGENT_TESTS_ALLOW_NETWORK=1 is set.
    """
    # Skip Docker tests unless --run-special is specified
    if not config.getoption("--run-special", default=False):
        skip_docker = pytest.mark.skip(
            reason="Requires --run-special flag (Docker needed)"
        )
        for item in items:
            if "docker" in item.keywords:
                item.add_marker(skip_docker)

    # Skip real_rag tests when DashScope embedding config is unavailable.
    # Check for actual values, not placeholder values from example.env.
    dashscope_key = os.getenv("DASHSCOPE_EMBEDDING_API_KEY") or os.getenv(
        "DASHSCOPE_API_KEY", ""
    )
    dashscope_model = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "")

    # Common placeholder patterns that indicate the key is not set
    placeholder_patterns = [
        "your-dashscope-api-key",
        "your-dashscope-embedding-model",
        "your-api-key",
        "test-key",
    ]

    def is_valid_value(value: str) -> bool:
        """Check if the config value is not a placeholder value."""
        if not value:
            return False
        key_lower = value.lower().strip()
        for pattern in placeholder_patterns:
            if pattern in key_lower:
                return False
        return True

    has_dashscope_embedding_config = is_valid_value(dashscope_key) and is_valid_value(
        dashscope_model
    )
    if not has_dashscope_embedding_config:
        skip_real_rag = pytest.mark.skip(
            reason="Requires (DASHSCOPE_API_KEY or DASHSCOPE_EMBEDDING_API_KEY) + DASHSCOPE_EMBEDDING_MODEL environment variables"
        )
        for item in items:
            if "real_rag" in item.keywords:
                item.add_marker(skip_real_rag)

    # Skip requires_network tests unless --run-special or XAGENT_TESTS_ALLOW_NETWORK=1
    run_special = config.getoption("--run-special", default=False)
    allow_network = os.getenv("XAGENT_TESTS_ALLOW_NETWORK", "0").strip() == "1"
    has_network_access = run_special or allow_network

    if not has_network_access:
        skip_network = pytest.mark.skip(
            reason="Requires --run-special flag or XAGENT_TESTS_ALLOW_NETWORK=1 (network access needed)"
        )
        for item in items:
            if "requires_network" in item.keywords:
                item.add_marker(skip_network)


# ==========================================
# CORE FIXTURES
# ==========================================


def _security_test_subdir(tmp_path: Path, name: str) -> str:
    """Create ``tmp_path / name`` and return its path as a string."""
    subdir = tmp_path / name
    subdir.mkdir()
    return str(subdir)


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    with TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture(autouse=True, scope="function")
def isolate_rag_storage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Isolate per-test RAG/KB storage paths and reset global storage state.

    By default, ``LANCEDB_DIR`` is set to a fresh directory under ``tmp_path``
    (current default vector backend is LanceDB). This avoids stale on-disk
    state from a developer ``.env`` or a fixed path. Parallel workers
    (pytest-xdist) each use their own process-local ``tmp_path``.

    If the environment sets ``XAGENT_PYTEST_RESPECT_LANCEDB_DIR=1``, the
    existing ``LANCEDB_DIR`` from the environment is left unchanged (for CI or
    local workflows that intentionally pin a path).

    Calls :func:`xagent.core.tools.core.RAG_tools.storage.reset_rag_storage_for_tests`
    before and after each test (backend-specific caches + storage factory reset).
    """
    respect_env = os.environ.get("XAGENT_PYTEST_RESPECT_LANCEDB_DIR") == "1"
    if not respect_env:
        lancedb_dir = tmp_path / "lancedb"
        lancedb_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("LANCEDB_DIR", str(lancedb_dir))

    reset_rag_storage_for_tests()
    yield
    reset_rag_storage_for_tests()


@pytest.fixture
def test_workspace_dir(tmp_path: Path) -> str:
    """Directory used as workspace root in ``test_service_security``."""
    return _security_test_subdir(tmp_path, "test_workspace")


@pytest.fixture
def test_access_dir(tmp_path: Path) -> str:
    """Directory used for access-restriction scenarios in security tests."""
    return _security_test_subdir(tmp_path, "test_access_restriction")


@pytest.fixture
def test_security_dir(tmp_path: Path) -> str:
    """Directory used for outside-access rejection scenarios in security tests."""
    return _security_test_subdir(tmp_path, "test_security")


@pytest.fixture(autouse=True, scope="function")
def mock_workspace_db():
    """Mock database operations for workspace to avoid DB access in tests.

    This fixture is automatically applied to all tests to prevent database
    access during testing. Tests can override this by explicitly creating
    real database connections if needed.
    """

    from unittest.mock import patch

    from xagent.core.workspace import TaskWorkspace

    # Mock _create_file_record to do nothing (avoid DB access)
    def mock_create_record(self, file_id, file_path, db_session=None):
        # Store file_id in cache for retrieval
        path_str = str(file_path)
        resolved_str = str(file_path.resolve())
        self._recently_registered_files[path_str] = file_id
        self._recently_registered_files[resolved_str] = file_id
        self._file_id_to_path[file_id] = file_path

    with patch.object(TaskWorkspace, "_create_file_record", mock_create_record):
        yield


@pytest.fixture
def temp_tool_dir():
    """Create a temporary directory with a single sample tool file.

    Generic fixture for tool directory testing across all test modules.
    """
    with TemporaryDirectory() as tmpdir:
        tool_file = Path(tmpdir) / "test_tool.py"
        tool_file.write_text("""
def get_test_tool():
    '''A test tool.'''
    return {'name': 'test_tool', 'description': 'A test tool'}
""")
        yield tmpdir


@pytest.fixture
def sample_tool_dir():
    """Create a temporary directory with sample tools for integration testing.

    Generic fixture for tool directory testing with multiple tool files.
    """
    with TemporaryDirectory() as tmpdir:
        # Create tool1.py
        tool1 = Path(tmpdir) / "tool1.py"
        tool1.write_text("""
'''Tool 1 module.'''

def get_tool1():
    '''First tool function.'''
    return {
        'name': 'tool1',
        'description': 'First test tool',
        'function': 'do_something',
        'parameters': {}
    }
""")

        # Create tool2.py
        tool2 = Path(tmpdir) / "tool2.py"
        tool2.write_text("""
'''Tool 2 module.'''

def get_tool2():
    '''Second tool function.'''
    return {
        'name': 'tool2',
        'description': 'Second test tool',
        'function': 'do_another_thing',
        'parameters': {}
    }
""")

        # Create __init__.py
        init_file = Path(tmpdir) / "__init__.py"
        init_file.write_text("")

        # Create a subdirectory with another tool
        subdir = Path(tmpdir) / "subdir"
        subdir.mkdir()
        (subdir / "__init__.py").write_text("")
        (subdir / "tool3.py").write_text("""
'''Tool 3 module.'''

def get_tool3():
    '''Third tool function.'''
    return {
        'name': 'tool3',
        'description': 'Third test tool',
        'function': 'do_third_thing',
    }
""")

        yield tmpdir


@pytest.fixture
def tool_dir_with_errors():
    """Create a directory with invalid tool files for error testing.

    Generic fixture for testing error handling in tool directories.
    """
    with TemporaryDirectory() as tmpdir:
        # Invalid tool file - syntax error
        invalid_tool = Path(tmpdir) / "invalid_tool.py"
        invalid_tool.write_text("""
def get_invalid_tool(
    # Missing closing parenthesis - syntax error
        return {'name': 'invalid'}
""")

        # Valid tool
        valid_tool = Path(tmpdir) / "valid_tool.py"
        valid_tool.write_text("""
def get_valid_tool():
    return {'name': 'valid_tool'}
""")

        yield tmpdir


@pytest.fixture
def initialized_tool_registry(temp_dir):
    """Fixture that provides a properly initialized tool registry."""
    # Initialize storage manager first
    import xagent.core.storage.manager as storage_manager
    from xagent.core.storage import initialize_storage_manager

    upload_dir = os.path.join(temp_dir, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    initialize_storage_manager(temp_dir, upload_dir)

    try:
        yield temp_dir
    finally:
        # Cleanup - reset global storage manager and remove temp directory
        storage_manager._storage_manager = None
        import shutil

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


# ==========================================
# MOCK RESPONSES AND COMPLETIONS
# ==========================================


@pytest.fixture
def mock_chat_completion():
    """Mock ChatCompletion response."""
    return ChatCompletion(
        id="test-completion-id",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content="Hello World",
                    role="assistant",
                    tool_calls=None,
                ),
            )
        ],
        created=1234567890,
        model="gpt-4o-mini",
        object="chat.completion",
        usage=None,
    )


@pytest.fixture
def mock_tool_call_completion():
    """Mock ChatCompletion response with tool call."""
    return ChatCompletion(
        id="test-tool-completion-id",
        choices=[
            Choice(
                finish_reason="tool_calls",
                index=0,
                message=ChatCompletionMessage(
                    content=None,
                    role="assistant",
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_test",
                            type="function",
                            function=ToolCallFunction(
                                name="get_weather",
                                arguments='{"location": "Boston"}',
                            ),
                        )
                    ],
                ),
            )
        ],
        created=1234567890,
        model="gpt-4o-mini",
        object="chat.completion",
        usage=None,
    )


@pytest.fixture
def mock_json_completion():
    """Mock ChatCompletion response with JSON content."""
    return ChatCompletion(
        id="test-json-completion",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content='{"name": "John", "age": 30}',
                    role="assistant",
                    tool_calls=None,
                ),
            )
        ],
        created=1234567890,
        model="gpt-4o-mini",
        object="chat.completion",
        usage=None,
    )


@pytest.fixture
def openai_llm_config():
    """Fixture providing OpenAI LLM configuration for testing."""
    return {
        "model_name": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key": "test-api-key",
        "default_temperature": 0.7,
        "default_max_tokens": 1024,
        "timeout": 30.0,
    }


@pytest.fixture
def azure_openai_llm_config():
    """Fixture providing Azure OpenAI LLM configuration for testing."""
    return {
        "model_name": "gpt-4o",
        "azure_endpoint": "https://test.openai.azure.com/",
        "api_key": "test-api-key",
        "api_version": "2024-08-01-preview",
        "default_temperature": 0.7,
        "default_max_tokens": 1024,
        "timeout": 30.0,
    }


@pytest.fixture
def gemini_llm_config():
    """Fixture providing Gemini LLM configuration for testing."""
    return {
        "model_name": "gemini-2.0-flash-exp",
        "api_key": "test-gemini-api-key",
        "default_temperature": 0.7,
        "default_max_tokens": 1024,
        "timeout": 30.0,
    }


@pytest.fixture
def claude_llm_config():
    """Fixture providing Claude LLM configuration for testing."""
    return {
        "model_name": "claude-3-5-sonnet-20241022",
        "api_key": "test-claude-api-key",
        "default_temperature": 0.7,
        "default_max_tokens": 1024,
        "timeout": 30.0,
    }


@pytest.fixture
def sample_openai_model():
    """Provide a sample OpenAI model for testing."""
    return ChatModelConfig(
        id="test_model",
        model_provider="test",
        model="gpt-3.5-turbo",
        temperature=0.7,
        api_key="test_api_key",
        base_url="https://api.openai.com/v1",
    )


@pytest.fixture
def langfuse_client_reset():
    """Fixture to reset the shared Langfuse client before and after each test."""
    reset_langfuse_client()
    yield
    reset_langfuse_client()


@pytest.fixture
def google_api_setting(monkeypatch, mocker):
    """Mock Google API settings for search tests."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test_key")
    monkeypatch.setenv("GOOGLE_CSE_ID", "test_cse_id")
    mock_data = {
        "items": [
            {
                "title": "Test Title",
                "link": "https://example.com",
                "snippet": "Test snippet",
            }
        ],
    }

    mock_response = mocker.Mock()
    mock_response.json.return_value = mock_data
    mock_response.raise_for_status.return_value = None

    mocker.patch("httpx.AsyncClient.get", return_value=mock_response)


# ==========================================
# LEGACY FIXTURES (used by existing tests)
# ==========================================


@pytest.fixture()
def team_dict():
    """Provide team configuration from JSON file for AutoGen tests."""
    json_path = (
        Path(__file__).parent / "core" / "frontend_adapter" / "autogen" / "team.json"
    )
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def modelhub():
    """Legacy fixture - creates model hub with pre-populated models."""
    with TemporaryDirectory() as temp_dir:
        model_dir = Path(temp_dir) / "model"
        model_dir.mkdir()

        # Initialize storage manager first
        from sqlalchemy import create_engine
        from sqlalchemy.ext.declarative import declarative_base
        from sqlalchemy.orm import sessionmaker

        from xagent.core.model.storage.db.adapter import SQLAlchemyModelHub
        from xagent.core.model.storage.db.db_models import create_model_table
        from xagent.core.storage import initialize_storage_manager

        upload_dir = os.path.join(temp_dir, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        initialize_storage_manager(temp_dir, upload_dir)

        # Create in-memory database for model storage
        engine = create_engine("sqlite:///:memory:")
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base = declarative_base()
        Model = create_model_table(Base)
        db = SessionLocal()
        Base.metadata.create_all(engine)

        hub = SQLAlchemyModelHub(db, Model)

        # Initialize tool registry
        from xagent.core.tools.adapters.langgraph import initialize_registry

        initialize_registry()

        openai_model = ChatModelConfig(
            id="openai-chat",
            model_provider="openai",
            model_name="openai-chat",
            api_key=os.getenv("OPENAI_API_KEY", "test_key"),
        )
        deepseek_model = ChatModelConfig(
            id="deepseek",
            model_provider="deepseek",
            model_name="deepseek-v4-flash",
            api_key=os.getenv("DEEPSEEK_API_KEY", "test_key"),
            base_url="https://api.deepseek.com",
        )
        # Add embedding model for embedding node tests
        embedding_model = EmbeddingModelConfig(
            id="embedding_model",
            model_provider="openai",
            model_name="text-embedding-ada-002",
            api_key=os.getenv("OPENAI_API_KEY", "test_key"),
        )
        # Add dashscope rerank model for rerank node tests
        dashscope_rerank_model = RerankModelConfig(
            id="dashscope-rerank",
            model_name="bge-reranker-v2-m3",
            api_key=os.getenv("DASHSCOPE_API_KEY", "test-dashscope-key"),
        )
        # Add Azure OpenAI model for Azure OpenAI tests
        azure_openai_model = ChatModelConfig(
            id="azure-openai-chat",
            model_provider="azure_openai",
            model_name="gpt-4o",
            base_url="https://test.openai.azure.com",
            api_key=os.getenv("AZURE_OPENAI_API_KEY", "test-azure-key"),
        )
        hub.store(openai_model)
        hub.store(deepseek_model)
        hub.store(embedding_model)
        hub.store(dashscope_rerank_model)
        hub.store(azure_openai_model)

        yield hub

        # Cleanup - reset global tool registry
        import xagent.core.storage.manager as storage_manager
        import xagent.core.tools.adapters.langgraph as tool_module

        tool_module._registry = None
        storage_manager._storage_manager = None


# ==========================================
# UTILITY FUNCTIONS (for integration tests)
# ==========================================


def check_langfuse_env():
    """Check required Langfuse environment variables - used by integration tests."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    base_url = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")

    if not public_key:
        pytest.fail("LANGFUSE_PUBLIC_KEY environment variable is required")
    if not secret_key:
        pytest.fail("LANGFUSE_SECRET_KEY environment variable is required")
    if not base_url:
        pytest.fail(
            "LANGFUSE_BASE_URL or LANGFUSE_HOST environment variable is required"
        )

    return public_key, secret_key, base_url


@pytest.fixture
def clear_langfuse_traces(request):
    """Clear all traces from Langfuse before starting integration tests."""
    if not request.config.getoption("--run-special"):
        pytest.skip("Run only with --run-special")

    public_key, secret_key, host = check_langfuse_env()
    yield


# YAML entrypoint has been removed, commenting out this fixture
# @pytest.fixture
# def mock_migration_manager(tmp_path):
#     """Initialize a temporary MigrationManager."""
#     test_migrations_dir = tmp_path / "test_migrations"
#     test_migrations_dir.mkdir()
#
#     print("user mock")
#     manager = MigrationManager(migrations_dir=str(test_migrations_dir))
#
#     set_yaml_migration_manager(manager)
#
#     yield manager
