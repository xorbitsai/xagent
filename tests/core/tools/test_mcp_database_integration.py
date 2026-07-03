"""
Test MCP database integration for web and agent entry points.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.agent.service import AgentService
from xagent.core.tools.adapters.vibe.factory import ToolFactory
from xagent.core.tools.core.mcp.manager.db import DatabaseMCPServerManager, MCPServer
from xagent.core.utils.encryption import encrypt_value
from xagent.core.workspace import TaskWorkspace
from xagent.web.models import MCPOAuthClient, MCPOAuthGrant
from xagent.web.models.database import Base
from xagent.web.models.mcp import UserMCPServer
from xagent.web.models.user import User


@pytest.fixture
def test_db():
    """Create test database."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    # Create test user
    test_user = User(username="test_user", id=1, password_hash="hashed_password")
    db.add(test_user)
    db.commit()

    yield db

    db.close()


@pytest.fixture
def test_workspace(tmp_path):
    """Create test workspace."""
    workspace_dir = tmp_path / "workspaces"
    workspace_dir.mkdir(exist_ok=True)
    return TaskWorkspace("test_workspace", base_dir=str(workspace_dir))


@pytest.fixture
def sample_stdio_config():
    """Sample STDIO MCP server configuration."""
    return {
        "name": "test_stdio_server",
        "transport": "stdio",
        "managed": "external",
        "description": "Test STDIO server",
        "command": "python",
        "args": ["test_server.py"],
        "env": {"TEST_VAR": "test_value"},
        "cwd": "/tmp",
    }


@pytest.fixture
def sample_websocket_config():
    """Sample WebSocket MCP server configuration."""
    return {
        "name": "test_websocket_server",
        "transport": "websocket",
        "managed": "external",
        "description": "Test WebSocket server",
        "url": "ws://localhost:8080/ws",
        "headers": {"Authorization": "Bearer token123"},
    }


@pytest.fixture
def sample_streamable_http_bearer_auth_config():
    """Sample Streamable HTTP MCP server config using bearer auth."""
    return {
        "name": "test_http_bearer_server",
        "transport": "streamable_http",
        "managed": "external",
        "description": "Test HTTP MCP server with bearer auth",
        "url": "http://localhost:18000/mcp",
        "headers": {"X-Test": "true"},
        "auth": {"type": "bearer", "bearer_token": "secret-token"},
    }


@pytest.fixture
def sample_streamable_http_api_key_auth_config():
    """Sample Streamable HTTP MCP server config using API key auth."""
    return {
        "name": "test_http_api_key_server",
        "transport": "streamable_http",
        "managed": "external",
        "description": "Test HTTP MCP server with API key auth",
        "url": "http://localhost:18000/mcp",
        "auth": {
            "type": "api_key",
            "api_key_name": "X-API-Key",
            "api_key_value": "secret-api-key",
        },
    }


class TestUserEnvOverrides:
    """Per-user env override loading (batched + decrypted) and merge semantics."""

    def test_load_user_env_overrides_decrypts_and_keys_by_server(self, test_db):
        from xagent.core.utils.encryption import encrypt_env_dict
        from xagent.web.services.mcp_runtime import (
            load_user_env_overrides,
            merge_stdio_env,
        )

        manager = DatabaseMCPServerManager(test_db)
        manager.add_server(
            manager.create_config(
                name="srv", transport="stdio", managed="external", command="python"
            )
        )
        server = test_db.query(MCPServer).filter(MCPServer.name == "srv").first()
        test_db.add(
            UserMCPServer(
                user_id=1,
                mcpserver_id=server.id,
                is_active=True,
                env=encrypt_env_dict({"API_KEY": "mine"}),
            )
        )
        test_db.commit()

        overrides = load_user_env_overrides(test_db, 1)
        assert overrides == {server.id: {"API_KEY": "mine"}}  # decrypted, keyed by id
        # user override wins; global env is the fallback
        assert merge_stdio_env(
            {"API_KEY": "global", "REGION": "us"}, overrides[server.id]
        ) == {
            "API_KEY": "mine",
            "REGION": "us",
        }

    def test_load_user_env_overrides_empty_without_user(self, test_db):
        from xagent.web.services.mcp_runtime import load_user_env_overrides

        assert load_user_env_overrides(test_db, None) == {}


class TestDatabaseMCPServerManager:
    """Test DatabaseMCPServerManager functionality."""

    def test_load_config_empty(self, test_db):
        """Test loading configuration from empty database."""
        manager = DatabaseMCPServerManager(test_db)
        config = manager.load_config()

        assert config == {"servers": {}}

    def test_add_stdio_server(self, test_db, sample_stdio_config):
        """Test adding STDIO MCP server."""
        manager = DatabaseMCPServerManager(test_db)

        # Create config object
        config = manager.create_config(**sample_stdio_config)

        # Add server
        manager.add_server(config)

        # Verify server was added
        servers = manager.list_servers()
        assert len(servers) == 1
        assert servers[0].config.name == sample_stdio_config["name"]
        assert servers[0].config.transport == sample_stdio_config["transport"]
        assert servers[0].config.command == sample_stdio_config["command"]
        assert servers[0].config.args == sample_stdio_config["args"]

    def test_get_connections(
        self, test_db, sample_stdio_config, sample_websocket_config
    ):
        """Test getting MCP connections."""
        manager = DatabaseMCPServerManager(test_db)

        # Add servers
        stdio_config = manager.create_config(**sample_stdio_config)
        manager.add_server(stdio_config)

        ws_config = manager.create_config(**sample_websocket_config)
        manager.add_server(ws_config)

        # Get connections
        connections = manager.get_connections()

        assert len(connections) == 2
        assert sample_stdio_config["name"] in connections
        assert sample_websocket_config["name"] in connections

        # Check STDIO connection format
        stdio_conn = connections[sample_stdio_config["name"]]
        assert stdio_conn["transport"] == "stdio"
        assert stdio_conn["command"] == "python"
        assert stdio_conn["args"] == ["test_server.py"]
        assert stdio_conn["env"] == {"TEST_VAR": "test_value"}

        # Check WebSocket connection format
        ws_conn = connections[sample_websocket_config["name"]]
        assert ws_conn["transport"] == "websocket"
        assert ws_conn["url"] == "ws://localhost:8080/ws"
        assert ws_conn["headers"] == {"Authorization": "Bearer token123"}

    def test_get_connections_includes_mcp_concurrency_config(
        self, test_db, sample_stdio_config
    ):
        """Database-backed MCP tool loading receives scheduler metadata."""
        manager = DatabaseMCPServerManager(test_db)
        sample_stdio_config = {
            **sample_stdio_config,
            "concurrency_safe": True,
            "concurrent_tools": ["list_messages"],
        }
        config = manager.create_config(**sample_stdio_config)
        manager.add_server(config)

        connections = manager.get_connections()

        stdio_conn = connections[sample_stdio_config["name"]]
        assert stdio_conn["concurrency_safe"] is True
        assert stdio_conn["concurrent_tools"] == ["list_messages"]

    def test_get_connections_skips_mcp_oauth_without_runtime_resolver(self, test_db):
        """Direct DB manager must not execute MCP OAuth via static headers."""
        manager = DatabaseMCPServerManager(test_db)
        config = manager.create_config(
            name="test_mcp_oauth_server",
            transport="streamable_http",
            managed="external",
            description="MCP OAuth server",
            url="https://mcp.example.com/mcp",
            headers={"Authorization": "Bearer static-token"},
            auth={
                "type": "mcp_oauth",
                "resource": "https://mcp.example.com/mcp",
                "issuer": "https://auth.example.com",
                "client_id": "client-123",
            },
        )
        manager.add_server(config)

        connections = manager.get_connections()

        assert "test_mcp_oauth_server" not in connections

    def test_runtime_oauth_detection_fails_closed_without_auth_decrypter(self):
        server = SimpleNamespace(
            transport="streamable_http",
            auth={"type": "mcp_oauth"},
        )

        assert DatabaseMCPServerManager._requires_runtime_mcp_oauth(server) is False

    def test_get_server(self, test_db, sample_stdio_config):
        """Test getting specific server."""
        manager = DatabaseMCPServerManager(test_db)

        # Add server
        config = manager.create_config(**sample_stdio_config)
        manager.add_server(config)

        # Get server
        server_data = manager.get_server(sample_stdio_config["name"])

        assert server_data is not None
        assert server_data.config.name == sample_stdio_config["name"]
        assert server_data.config.transport == sample_stdio_config["transport"]

    def test_remove_server(self, test_db, sample_stdio_config):
        """Test removing MCP server."""
        manager = DatabaseMCPServerManager(test_db)

        # Add server
        config = manager.create_config(**sample_stdio_config)
        manager.add_server(config)

        # Verify server exists
        servers = manager.list_servers()
        assert len(servers) == 1

        # Remove server
        result = manager.remove_server(sample_stdio_config["name"])

        assert result is True

        # Verify server is removed
        servers = manager.list_servers()
        assert len(servers) == 0

    def test_list_servers(self, test_db, sample_stdio_config, sample_websocket_config):
        """Test listing all servers."""
        manager = DatabaseMCPServerManager(test_db)

        # Add servers
        stdio_config = manager.create_config(**sample_stdio_config)
        manager.add_server(stdio_config)

        ws_config = manager.create_config(**sample_websocket_config)
        manager.add_server(ws_config)

        # List servers
        servers = manager.list_servers()

        assert len(servers) == 2
        server_names = [s.config.name for s in servers]
        assert sample_stdio_config["name"] in server_names
        assert sample_websocket_config["name"] in server_names

    def test_create_config_stdio(self, sample_stdio_config):
        """Test creating STDIO configuration."""

        manager = DatabaseMCPServerManager(MagicMock())
        config = manager.create_config(**sample_stdio_config)

        assert config.name == sample_stdio_config["name"]
        assert config.transport == sample_stdio_config["transport"]
        assert config.command == sample_stdio_config["command"]
        assert config.args == sample_stdio_config["args"]
        assert config.env == sample_stdio_config["env"]

    def test_create_config_websocket(self, sample_websocket_config):
        """Test creating WebSocket configuration."""

        manager = DatabaseMCPServerManager(MagicMock())
        config = manager.create_config(**sample_websocket_config)

        assert config.name == sample_websocket_config["name"]
        assert config.transport == sample_websocket_config["transport"]
        assert config.url == sample_websocket_config["url"]
        assert config.headers == sample_websocket_config["headers"]

    def test_get_connections_maps_bearer_auth_to_authorization_header(
        self, test_db, sample_streamable_http_bearer_auth_config
    ):
        """Test bearer auth is converted into Authorization header."""
        manager = DatabaseMCPServerManager(test_db)
        config = manager.create_config(**sample_streamable_http_bearer_auth_config)
        manager.add_server(config)

        connection = manager.get_connections()[config.name]

        assert connection["transport"] == "streamable_http"
        assert connection["url"] == sample_streamable_http_bearer_auth_config["url"]
        assert connection["headers"]["X-Test"] == "true"
        assert connection["headers"]["Authorization"] == "Bearer secret-token"
        assert "auth" not in connection

    def test_get_connections_maps_api_key_auth_to_header(
        self, test_db, sample_streamable_http_api_key_auth_config
    ):
        """Test API key auth is converted into the configured header."""
        manager = DatabaseMCPServerManager(test_db)
        config = manager.create_config(**sample_streamable_http_api_key_auth_config)
        manager.add_server(config)

        connection = manager.get_connections()[config.name]

        assert connection["headers"]["X-API-Key"] == "secret-api-key"
        assert "auth" not in connection

    def test_get_connections_preserves_explicit_headers_over_auto_auth(
        self, test_db, sample_streamable_http_bearer_auth_config
    ):
        """Test explicit custom headers take precedence over generated auth headers."""
        manager = DatabaseMCPServerManager(test_db)
        overridden_config = {
            **sample_streamable_http_bearer_auth_config,
            "name": "test_http_custom_auth_header_server",
            "headers": {"Authorization": "Bearer custom-header-token"},
        }
        config = manager.create_config(**overridden_config)
        manager.add_server(config)

        connection = manager.get_connections()[config.name]

        assert connection["headers"]["Authorization"] == "Bearer custom-header-token"


class TestToolFactoryMCPIntegration:
    """Test ToolFactory MCP integration."""

    @staticmethod
    def _mcp_oauth_config(name: str = "test_mcp_oauth_server"):
        return {
            "name": name,
            "transport": "streamable_http",
            "managed": "external",
            "description": "MCP OAuth server",
            "url": "https://mcp.example.com/mcp",
            "headers": {"Authorization": "Bearer static-token"},
            "auth": {
                "type": "mcp_oauth",
                "resource": "https://mcp.example.com/mcp",
                "issuer": "https://auth.example.com",
                "client_id": "client-123",
            },
        }

    @patch("xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools")
    async def test_create_mcp_tools_success(
        self, mock_load_mcp, test_db, sample_stdio_config
    ):
        """Test successful MCP tools creation."""
        # Setup mock
        mock_tools = [MagicMock(), MagicMock()]
        mock_load_mcp.return_value = mock_tools

        # Add MCP server to database
        manager = DatabaseMCPServerManager(test_db)
        config = manager.create_config(**sample_stdio_config)
        manager.add_server(config)

        # Create MCP tools
        tools = await ToolFactory.create_mcp_tools(test_db)

        # Verify
        assert len(tools) == 2
        mock_load_mcp.assert_called_once()

        # Check call arguments
        call_args = mock_load_mcp.call_args
        connections_arg = call_args[0][0]  # First positional argument
        assert sample_stdio_config["name"] in connections_arg

    @patch("xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools")
    async def test_create_mcp_tools_skips_mcp_oauth_but_loads_other_servers(
        self, mock_load_mcp, test_db, sample_stdio_config
    ):
        mock_tools = [MagicMock()]
        mock_load_mcp.return_value = mock_tools
        manager = DatabaseMCPServerManager(test_db)
        manager.add_server(manager.create_config(**sample_stdio_config))
        manager.add_server(manager.create_config(**self._mcp_oauth_config()))

        tools = await ToolFactory.create_mcp_tools(test_db)

        assert tools == mock_tools
        connections_arg = mock_load_mcp.call_args[0][0]
        assert sample_stdio_config["name"] in connections_arg
        assert "test_mcp_oauth_server" not in connections_arg

    @patch("xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools")
    async def test_create_mcp_tools_all_mcp_oauth_returns_empty_without_loading(
        self, mock_load_mcp, test_db
    ):
        manager = DatabaseMCPServerManager(test_db)
        manager.add_server(manager.create_config(**self._mcp_oauth_config()))

        tools = await ToolFactory.create_mcp_tools(test_db)

        assert tools == []
        mock_load_mcp.assert_not_called()

    @patch("xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools")
    async def test_create_mcp_tools_routes_mcp_oauth_through_runtime_builder(
        self, mock_load_mcp, test_db
    ):
        mock_tools = [MagicMock()]
        mock_load_mcp.return_value = mock_tools
        manager = DatabaseMCPServerManager(test_db)
        manager.add_server(manager.create_config(**self._mcp_oauth_config()))
        server = test_db.query(MCPServer).filter_by(name="test_mcp_oauth_server").one()
        test_db.add(
            UserMCPServer(
                user_id=1,
                mcpserver_id=server.id,
                is_owner=True,
                is_active=True,
            )
        )
        oauth_client = MCPOAuthClient(
            mcp_server_id=server.id,
            issuer="https://auth.example.com",
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
            client_id="client-123",
            token_endpoint_auth_method="none",
            redirect_uri="https://xagent.example.com/api/mcp/oauth/callback",
        )
        test_db.add(oauth_client)
        test_db.flush()
        test_db.add(
            MCPOAuthGrant(
                mcp_server_id=server.id,
                user_id=1,
                mcp_oauth_client_id=oauth_client.id,
                resource_owner_key="xagent:user:1",
                issuer="https://auth.example.com",
                resource="https://mcp.example.com/mcp",
                scope="",
                access_token=encrypt_value("runtime-access-token"),
            )
        )
        test_db.commit()

        tools = await ToolFactory.create_mcp_tools(test_db, user_id=1)

        assert tools == mock_tools
        connections_arg = mock_load_mcp.call_args[0][0]
        assert connections_arg["test_mcp_oauth_server"]["headers"] == {
            "Authorization": "Bearer runtime-access-token"
        }

    @patch("xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools")
    async def test_create_mcp_tools_no_connections(self, mock_load_mcp, test_db):
        """Test MCP tools creation with no connections."""
        tools = await ToolFactory.create_mcp_tools(test_db)

        assert tools == []
        mock_load_mcp.assert_not_called()

    @patch("xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools")
    async def test_create_mcp_tools_error_handling(
        self, mock_load_mcp, test_db, sample_stdio_config
    ):
        """Test MCP tools creation error handling."""
        # Setup mock to raise exception
        mock_load_mcp.side_effect = Exception("MCP connection failed")

        # Add MCP server to database
        manager = DatabaseMCPServerManager(test_db)
        config = manager.create_config(**sample_stdio_config)
        manager.add_server(config)

        # Create MCP tools (should handle error gracefully)
        tools = await ToolFactory.create_mcp_tools(test_db, user_id=1)

        assert tools == []

    def test_create_mcp_tools_sync_wrapper(self, test_db, sample_stdio_config):
        """Test synchronous wrapper for MCP tools creation."""
        # Add an MCP server to the database first
        manager = DatabaseMCPServerManager(test_db)
        config = manager.create_config(**sample_stdio_config)
        manager.add_server(config)

        # Now test the sync wrapper
        tools = ToolFactory._create_mcp_tools(test_db, user_id=1)

        # The result should be an empty list (no actual MCP tools available)
        assert isinstance(tools, list)


class TestAgentServiceMCPIntegration:
    """Test AgentService MCP integration."""

    @patch("xagent.core.tools.adapters.vibe.factory.ToolFactory._create_mcp_tools")
    def test_agent_service_init_with_mcp_enabled(
        self, mock_create_mcp, test_db, test_workspace
    ):
        """Test AgentService initialization with MCP tools enabled."""
        # Setup mock
        mock_tools = [MagicMock()]
        mock_create_mcp.return_value = mock_tools

        # Create AgentService with workspace
        agent_service = AgentService(
            name="test_agent",
            id="test_agent",
            workspace=test_workspace,
        )

        assert agent_service.workspace == test_workspace

    @patch("xagent.core.tools.adapters.vibe.factory.ToolFactory._create_mcp_tools")
    def test_agent_service_init_with_mcp_disabled(
        self, mock_create_mcp, test_db, test_workspace
    ):
        """Test AgentService initialization with MCP tools disabled."""
        # Create AgentService
        AgentService(
            name="test_agent",
            id="test_agent",
            workspace=test_workspace,
        )

        # Verify MCP tools were not loaded
        mock_create_mcp.assert_not_called()

    @patch("xagent.core.tools.adapters.vibe.factory.ToolFactory._create_mcp_tools")
    def test_agent_service_init_without_db(self, mock_create_mcp, test_workspace):
        """Test AgentService initialization without database."""
        # Create AgentService without database
        AgentService(name="test_agent", id="test_agent", workspace=test_workspace)

        # Verify MCP tools were not loaded
        mock_create_mcp.assert_not_called()

    @patch("xagent.core.tools.adapters.vibe.factory.ToolFactory._create_mcp_tools")
    def test_setup_mcp_tools_error_handling(
        self, mock_create_mcp, test_db, test_workspace
    ):
        """Test MCP tools setup error handling."""
        # Setup mock to raise exception
        mock_create_mcp.side_effect = Exception("Database connection failed")

        # Create AgentService (should handle error gracefully)
        AgentService(
            name="test_agent",
            id="test_agent",
            workspace=test_workspace,
        )

        # Verify MCP tools were not called directly by AgentService
        mock_create_mcp.assert_not_called()


class TestMCPServerModel:
    """Test MCPServer database model."""

    def test_to_connection_dict_stdio(self, test_db):
        """Test to_connection_dict method for STDIO transport."""

        config = {
            "command": "python",
            "args": ["server.py"],
            "env": {"API_KEY": "secret"},
            "cwd": "/tmp",
        }

        server = MCPServer(
            name="test_server",
            transport="stdio",
            command=config["command"],
            args=config["args"],
            env=config["env"],
            cwd=config["cwd"],
            managed="external",
        )

        connection_dict = server.to_connection_dict()

        assert connection_dict["name"] == "test_server"
        assert connection_dict["transport"] == "stdio"
        assert connection_dict["command"] == "python"
        assert connection_dict["args"] == ["server.py"]
        assert connection_dict["env"] == {"API_KEY": "secret"}
        assert connection_dict["cwd"] == "/tmp"

    def test_to_connection_dict_websocket(self, test_db):
        """Test to_connection_dict method for WebSocket transport."""

        config = {
            "url": "ws://localhost:8080/ws",
            "headers": {"Authorization": "Bearer token"},
        }

        server = MCPServer(
            name="test_server",
            transport="websocket",
            url=config["url"],
            headers=config["headers"],
            managed="external",
        )

        connection_dict = server.to_connection_dict()

        assert connection_dict["name"] == "test_server"
        assert connection_dict["transport"] == "websocket"
        assert connection_dict["url"] == "ws://localhost:8080/ws"
        assert connection_dict["headers"] == {"Authorization": "Bearer token"}

    def test_transport_display_property(self, test_db):
        """Test transport_display property."""

        stdio_server = MCPServer(
            name="stdio_test", transport="stdio", managed="external"
        )
        websocket_server = MCPServer(
            name="ws_test", transport="websocket", managed="external"
        )
        unknown_server = MCPServer(
            name="unknown_test", transport="unknown", managed="external"
        )

        assert stdio_server.transport_display == "STDIO"
        assert websocket_server.transport_display == "WebSocket"
        assert unknown_server.transport_display == "UNKNOWN"

    def test_to_config_dict(self, test_db):
        """Test to_config_dict method."""

        server = MCPServer(
            name="test_server",
            transport="stdio",
            command="python",
            args=["server.py"],
            env={"API_KEY": "secret"},
            managed="external",
            description="Test server",
        )

        config_dict = server.to_config_dict()

        assert config_dict["name"] == "test_server"
        assert config_dict["transport"] == "stdio"
        assert config_dict["command"] == "python"
        assert config_dict["args"] == ["server.py"]
        assert config_dict["env"] == {"API_KEY": "secret"}
        assert config_dict["managed"] == "external"
        assert config_dict["description"] == "Test server"

    def test_from_config(self, test_db):
        """Test from_config class method."""

        config = {
            "name": "test_server",
            "transport": "stdio",
            "command": "python",
            "args": ["server.py"],
            "env": {"API_KEY": "secret"},
            "managed": "external",
            "description": "Test server",
        }

        server = MCPServer.from_config(config)

        assert server.name == "test_server"
        assert server.transport == "stdio"
        assert server.command == "python"
        assert server.args == ["server.py"]
        # env is encrypted at rest but decrypts back for consumption
        assert server.env["API_KEY"].startswith("gAAAAAB")
        assert server.to_connection_dict()["env"] == {"API_KEY": "secret"}
        assert server.managed == "external"
        assert server.description == "Test server"
