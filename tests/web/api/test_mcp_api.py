"""
Test MCP API endpoints and functions
"""

from unittest.mock import MagicMock

import pytest

from xagent.web.api.mcp import _db_server_to_response, get_supported_transports
from xagent.web.models.mcp import MCPServer


class TestMCPServerModel:
    """Test MCPServer database model."""

    def test_to_connection_dict_stdio(self):
        """Test to_connection_dict method for STDIO transport."""
        server = MCPServer(
            name="test_server",
            transport="stdio",
            managed="external",
            command="python",
            args=["server.py"],
            env={"API_KEY": "secret"},
            cwd="/tmp",
        )

        connection_dict = server.to_connection_dict()

        assert connection_dict["name"] == "test_server"
        assert connection_dict["transport"] == "stdio"
        assert connection_dict["command"] == "python"
        assert connection_dict["args"] == ["server.py"]
        assert connection_dict["env"] == {"API_KEY": "secret"}
        assert connection_dict["cwd"] == "/tmp"

    def test_to_connection_dict_websocket(self):
        """Test to_connection_dict method for WebSocket transport."""
        server = MCPServer(
            name="test_websocket_server",
            transport="websocket",
            managed="external",
            url="ws://localhost:8080/ws",
            headers={"Authorization": "Bearer token"},
        )

        connection_dict = server.to_connection_dict()

        assert connection_dict["name"] == "test_websocket_server"
        assert connection_dict["transport"] == "websocket"
        assert connection_dict["url"] == "ws://localhost:8080/ws"
        assert connection_dict["headers"] == {"Authorization": "Bearer token"}

    def test_to_config_dict_external(self):
        """Test to_config_dict method for external server."""
        server = MCPServer(
            name="external_server",
            transport="stdio",
            managed="external",
            description="Test external server",
            command="python",
            args=["server.py"],
            env={"KEY": "value"},
            cwd="/app",
        )

        config = server.to_config_dict()

        assert config["name"] == "external_server"
        assert config["description"] == "Test external server"
        assert config["managed"] == "external"
        assert config["transport"] == "stdio"
        assert config["command"] == "python"
        assert config["args"] == ["server.py"]
        assert config["env"] == {"KEY": "value"}
        assert config["cwd"] == "/app"
        # Internal-only fields should not be present
        assert "docker_url" not in config
        assert "docker_image" not in config

    def test_to_config_dict_internal(self):
        """Test to_config_dict method for internal server."""
        server = MCPServer(
            name="internal_server",
            transport="stdio",
            managed="internal",
            description="Test internal server",
            docker_url="unix:///var/run/docker.sock",
            docker_image="mcp-server:latest",
            docker_environment={"ENV_VAR": "value"},
            docker_working_dir="/app",
            volumes=["/host:/container"],
            bind_ports={"8080": 8080},
            restart_policy="unless-stopped",
            auto_start=True,
        )

        config = server.to_config_dict()

        assert config["name"] == "internal_server"
        assert config["managed"] == "internal"
        assert config["docker_url"] == "unix:///var/run/docker.sock"
        assert config["docker_image"] == "mcp-server:latest"
        assert config["docker_environment"] == {"ENV_VAR": "value"}
        assert config["docker_working_dir"] == "/app"
        assert config["volumes"] == ["/host:/container"]
        assert config["bind_ports"] == {"8080": 8080}
        assert config["restart_policy"] == "unless-stopped"
        assert config["auto_start"] is True

    def test_from_config_external(self):
        """Test from_config class method for external server."""
        config = {
            "name": "test_server",
            "description": "Test server",
            "managed": "external",
            "transport": "stdio",
            "command": "python",
            "args": ["server.py"],
            "env": {"KEY": "value"},
            "cwd": "/app",
        }

        server = MCPServer.from_config(config)

        assert server.name == "test_server"
        assert server.description == "Test server"
        assert server.managed == "external"
        assert server.transport == "stdio"
        assert server.command == "python"
        assert server.args == ["server.py"]
        assert server.env == {"KEY": "value"}
        assert server.cwd == "/app"

    def test_from_config_internal(self):
        """Test from_config class method for internal server."""
        config = {
            "name": "internal_server",
            "managed": "internal",
            "transport": "stdio",
            "docker_url": "unix:///var/run/docker.sock",
            "docker_image": "mcp-server:latest",
            "docker_environment": {"ENV": "value"},
            "restart_policy": "always",
            "auto_start": True,
        }

        server = MCPServer.from_config(config)

        assert server.name == "internal_server"
        assert server.managed == "internal"
        assert server.docker_url == "unix:///var/run/docker.sock"
        assert server.docker_image == "mcp-server:latest"
        assert server.docker_environment == {"ENV": "value"}
        assert server.restart_policy == "always"
        assert server.auto_start is True

    def test_from_config_encrypts_oauth_access_token(self):
        """Test from_config encrypts OAuth access tokens at rest."""
        server = MCPServer.from_config(
            {
                "name": "oauth_server",
                "managed": "external",
                "transport": "streamable_http",
                "url": "https://example.com/mcp",
                "auth": {
                    "type": "oauth2",
                    "access_token": "plain-access-token",
                    "token_type": "Bearer",
                },
            }
        )

        assert server.auth["access_token"] != "plain-access-token"
        assert server.to_config_dict()["auth"]["access_token"] == "plain-access-token"

    def test_transport_display_property(self):
        """Test transport_display property for different transports."""
        stdio_server = MCPServer(
            name="stdio_test", transport="stdio", managed="external"
        )
        websocket_server = MCPServer(
            name="ws_test", transport="websocket", managed="external"
        )
        sse_server = MCPServer(name="sse_test", transport="sse", managed="external")
        streamable_server = MCPServer(
            name="http_test", transport="streamable_http", managed="external"
        )
        unknown_server = MCPServer(
            name="unknown_test", transport="unknown", managed="external"
        )

        assert stdio_server.transport_display == "STDIO"
        assert websocket_server.transport_display == "WebSocket"
        assert sse_server.transport_display == "Server-Sent Events"
        assert streamable_server.transport_display == "Streamable HTTP"
        assert unknown_server.transport_display == "UNKNOWN"

    def test_repr_method(self):
        """Test __repr__ method."""
        server = MCPServer(
            id=1, name="test_server", transport="stdio", managed="external"
        )

        repr_str = repr(server)
        assert "MCPServer" in repr_str
        assert "id=1" in repr_str
        assert "name='test_server'" in repr_str
        assert "transport='stdio'" in repr_str
        assert "managed='external'" in repr_str


class TestMCPApiFunctions:
    """Test MCP API utility functions."""

    def test_get_supported_transports_data(self):
        """Test get_supported_transports_data function."""
        transports_data = get_supported_transports()

        assert "transports" in transports_data
        assert isinstance(transports_data["transports"], list)
        assert len(transports_data["transports"]) > 0

        # Check required transports
        transport_ids = [t["id"] for t in transports_data["transports"]]
        assert "stdio" in transport_ids
        assert "websocket" in transport_ids
        assert "sse" in transport_ids
        assert "streamable_http" in transport_ids

        # Check stdio transport structure
        stdio_transport = next(
            t for t in transports_data["transports"] if t["id"] == "stdio"
        )
        assert stdio_transport["name"] == "STDIO"
        assert "Standard input/output transport" in stdio_transport["description"]
        assert "config_fields" in stdio_transport
        assert isinstance(stdio_transport["config_fields"], list)

        # Check required stdio config fields
        config_fields = {f["name"]: f for f in stdio_transport["config_fields"]}
        assert "command" in config_fields
        assert config_fields["command"]["required"] is True
        assert "args" in config_fields
        assert config_fields["args"]["required"] is False
        assert "env" in config_fields
        assert "cwd" in config_fields

    def test_db_server_to_response_masks_oauth_access_token(self):
        """Test API responses mask OAuth access tokens like other auth secrets."""
        server = MCPServer.from_config(
            {
                "name": "oauth_server",
                "managed": "external",
                "transport": "streamable_http",
                "url": "https://example.com/mcp",
                "auth": {
                    "type": "oauth2",
                    "access_token": "plain-access-token",
                    "token_type": "Bearer",
                },
            }
        )
        server.id = 1

        user_mcp = MagicMock()
        user_mcp.user_id = 1
        user_mcp.is_active = True
        user_mcp.is_default = False

        response = _db_server_to_response(
            server=server,
            user_mcp=user_mcp,
            manager=MagicMock(),
        )

        assert response.config["auth"]["type"] == "oauth2"
        assert response.config["auth"]["token_type"] == "Bearer"
        assert response.config["auth"]["access_token"] == "********"


class TestMCPApiModels:
    """Test MCP API Pydantic models."""

    def test_mcp_server_create_model(self):
        """Test MCPServerCreate model validation."""
        from xagent.web.api.mcp import MCPServerCreate

        # Valid data
        valid_data = {
            "name": "test_server",
            "transport": "stdio",
            "description": "Test server",
            "config": {"command": "echo", "args": ["hello"]},
        }

        server = MCPServerCreate(**valid_data)
        assert server.name == "test_server"
        assert server.transport == "stdio"
        assert server.config == {"command": "echo", "args": ["hello"]}

        # Test required fields
        invalid_data = {"transport": "stdio", "config": {}}

        with pytest.raises(ValueError):
            MCPServerCreate(**invalid_data)

    def test_mcp_server_update_model(self):
        """Test MCPServerUpdate model validation."""
        from xagent.web.api.mcp import MCPServerUpdate

        # Partial update data
        partial_data = {"name": "updated_server", "description": "Updated"}

        server = MCPServerUpdate(**partial_data)
        assert server.name == "updated_server"
        assert server.description == "Updated"

        # Empty update data
        empty_data = {}
        server = MCPServerUpdate(**empty_data)
        assert server.name is None
        assert server.transport is None

    def test_mcp_server_response_model(self):
        """Test MCPServerResponse model."""
        from xagent.web.api.mcp import MCPServerResponse

        response_data = {
            "id": 1,
            "user_id": 1,
            "is_default": True,
            "name": "test_server",
            "transport": "stdio",
            "description": "Test server",
            "is_active": True,
            "config": {
                "command": "python",
                "args": ["server.py"],
            },
            "transport_display": "STDIO",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }

        response = MCPServerResponse(**response_data)
        assert response.id == 1
        assert response.name == "test_server"
        assert response.transport == "stdio"
        assert response.is_active is True
        assert response.config["command"] == "python"
        assert response.config["args"] == ["server.py"]

    def test_mcp_connection_test_models(self):
        """Test MCP connection test models."""
        from xagent.web.api.mcp import MCPConnectionTest, MCPConnectionTestResponse

        # Test request model
        test_data = {
            "name": "test_connection",
            "transport": "stdio",
            "config": {
                "command": "echo",
            },
        }

        test_request = MCPConnectionTest(**test_data)
        assert test_request.name == "test_connection"
        assert test_request.transport == "stdio"
        assert test_request.config["command"] == "echo"

        # Test response model
        response_data = {
            "success": True,
            "message": "Connection successful",
            "details": {"tool_count": 5},
        }

        response = MCPConnectionTestResponse(**response_data)
        assert response.success is True
        assert response.message == "Connection successful"
        assert response.details == {"tool_count": 5}

        # Test response without details
        minimal_response = {"success": False, "message": "Connection failed"}

        response = MCPConnectionTestResponse(**minimal_response)
        assert response.success is False
        assert response.details is None
