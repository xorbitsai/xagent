"""
Test MCP API endpoints and functions
"""

from unittest.mock import MagicMock

import pytest

from xagent.web.api.mcp import (
    MCPServerCreate,
    MCPServerUpdate,
    _auth_metadata_tampered,
    _build_server_config,
    _check_mcp_permission,
    _db_server_to_response,
    _global_config_tampered,
    _mask_env,
    _merge_masked_env,
    get_supported_transports,
)
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

    def test_to_connection_dict_includes_mcp_concurrency_metadata(self):
        """MCP loader receives scheduling metadata alongside connection fields."""
        server = MCPServer(
            name="test_server",
            transport="stdio",
            managed="external",
            command="python",
            args=["server.py"],
            concurrency_safe=True,
            concurrent_tools=["list_messages"],
        )

        connection_dict = server.to_connection_dict()

        assert connection_dict["concurrency_safe"] is True
        assert connection_dict["concurrent_tools"] == ["list_messages"]

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

    def test_mcp_oauth_to_connection_dict_strips_static_authorization(self):
        """MCP OAuth runtime credentials must not fall back to stored headers."""
        server = MCPServer(
            name="test_oauth_server",
            transport="streamable_http",
            managed="external",
            url="https://mcp.example.com/mcp",
            headers={
                "Authorization": "Bearer static-token",
                "X-Request-Source": "xagent",
            },
            auth={"type": "mcp_oauth"},
        )

        connection_dict = server.to_connection_dict()

        assert connection_dict["headers"] == {"X-Request-Source": "xagent"}

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
        assert config["concurrency_safe"] is False
        assert config["concurrent_tools"] == []
        # Internal-only fields should not be present
        assert "docker_url" not in config
        assert "docker_image" not in config

    def test_to_config_dict_includes_mcp_concurrency_config(self):
        """MCP config responses include explicit concurrency opt-in settings."""
        server = MCPServer(
            name="external_server",
            transport="stdio",
            managed="external",
            command="python",
            concurrency_safe=True,
            concurrent_tools=["list_messages"],
        )

        config = server.to_config_dict()

        assert config["concurrency_safe"] is True
        assert config["concurrent_tools"] == ["list_messages"]

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
        # env is encrypted at rest but decrypts back for consumption
        assert server.env["KEY"].startswith("gAAAAAB")
        assert server.to_connection_dict()["env"] == {"KEY": "value"}
        assert server.cwd == "/app"
        assert server.concurrency_safe is False
        assert server.concurrent_tools == []

    def test_from_config_persists_mcp_concurrency_config(self):
        """MCP concurrency opt-in survives database model construction."""
        config = {
            "name": "test_server",
            "description": "Test server",
            "managed": "external",
            "transport": "stdio",
            "command": "python",
            "concurrency_safe": True,
            "concurrent_tools": ["list_messages"],
        }

        server = MCPServer.from_config(config)

        assert server.concurrency_safe is True
        assert server.concurrent_tools == ["list_messages"]

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

    def test_from_config_rejects_masked_auth_secret_without_existing_value(self):
        """Masked placeholders are response values, not creatable secrets."""
        with pytest.raises(ValueError, match="Masked auth value"):
            MCPServer.from_config(
                {
                    "name": "masked_server",
                    "managed": "external",
                    "transport": "streamable_http",
                    "url": "https://example.com/mcp",
                    "auth": {
                        "type": "mcp_oauth",
                        "client_id": "client-123",
                        "client_secret": "********",
                    },
                }
            )

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
        user_mcp.is_owner = True
        user_mcp.env = None

        response = _db_server_to_response(
            server=server,
            user_mcp=user_mcp,
            manager=MagicMock(),
        )

        assert response.config["auth"]["type"] == "oauth2"
        assert response.config["auth"]["token_type"] == "Bearer"
        assert response.config["auth"]["access_token"] == "********"

    def test_db_server_to_response_masks_env_and_returns_user_env(self):
        """Env values are masked (keys kept); per-user env is returned masked too."""
        server = MCPServer.from_config(
            {
                "name": "envy",
                "managed": "external",
                "transport": "stdio",
                "command": "python",
                "env": {"API_KEY": "global-secret", "REGION": "us"},
            }
        )
        server.id = 1

        user_mcp = MagicMock()
        user_mcp.user_id = 1
        user_mcp.is_active = True
        user_mcp.is_default = False
        user_mcp.is_owner = False
        user_mcp.env = {"API_KEY": "my-secret"}

        response = _db_server_to_response(
            server=server, user_mcp=user_mcp, manager=MagicMock()
        )

        # Keys visible, values masked
        assert set(response.config["env"]) == {"API_KEY", "REGION"}
        assert response.config["env"]["API_KEY"] == "********"
        assert response.user_env == {"API_KEY": "********"}
        # Non-owner cannot edit the global fallback
        assert response.can_edit_global is False

    def test_merge_masked_env_preserves_stored_secrets(self):
        """Masked values keep the stored secret; real values overwrite; new keys added."""
        merged = _merge_masked_env(
            {"API_KEY": "********", "TOKEN": "new", "EXTRA": "x"},
            {"API_KEY": "old-secret", "TOKEN": "old"},
        )
        assert merged == {"API_KEY": "old-secret", "TOKEN": "new", "EXTRA": "x"}

    def test_merge_masked_env_drops_masked_key_without_stored_value(self):
        """A masked key with no stored value (e.g. renamed) is dropped, never None."""
        merged = _merge_masked_env({"NEW": "********"}, {"OLD": "secret"})
        assert merged == {}
        assert None not in merged.values()

    def test_check_mcp_permission(self):
        """Owner gates edit; owner-or-can_delete gates delete; admin bypasses."""
        owner = MagicMock(is_owner=True, can_delete=True)
        guest = MagicMock(is_owner=False, can_delete=False)
        assert _check_mcp_permission(owner, is_admin=False, require="edit") is True
        assert _check_mcp_permission(owner, is_admin=False, require="delete") is True
        assert _check_mcp_permission(guest, is_admin=False, require="edit") is False
        assert _check_mcp_permission(guest, is_admin=False, require="delete") is False
        # Admin bypasses the per-row flags
        assert _check_mcp_permission(guest, is_admin=True, require="delete") is True
        # Regression: an owner whose can_delete was never set (OAuth provisioning,
        # migration-skipped rows) must still be able to delete their own server.
        legacy_owner = MagicMock(is_owner=True, can_delete=False)
        assert _check_mcp_permission(legacy_owner, is_admin=False, require="delete")
        # A non-owner explicitly granted can_delete may delete.
        grantee = MagicMock(is_owner=False, can_delete=True)
        assert _check_mcp_permission(grantee, is_admin=False, require="delete")

    def test_global_config_tampered(self):
        """Non-secret global fields are diffed; unchanged payloads pass."""
        server = MCPServer.from_config(
            {
                "name": "svc",
                "managed": "external",
                "transport": "stdio",
                "command": "python",
                "args": ["-m", "svc"],
            }
        )
        # Same values (what a non-owner's disabled-but-submitted form sends) -> ok
        unchanged = MCPServerUpdate(config={"command": "python", "args": ["-m", "svc"]})
        assert _global_config_tampered(unchanged, server) is False
        # Changed command -> tampered
        assert _global_config_tampered(
            MCPServerUpdate(config={"command": "sh"}), server
        )
        # Changed top-level name -> tampered
        assert _global_config_tampered(MCPServerUpdate(name="other"), server)

    def test_auth_metadata_tampered(self):
        """Non-secret auth metadata is diffed; secrets/masked values are ignored."""
        current = {"type": "oauth2", "client_id": "abc", "client_secret": "enc"}
        # Unchanged metadata (masked secret) -> not tampered
        assert not _auth_metadata_tampered(
            {"type": "oauth2", "client_id": "abc", "client_secret": "********"},
            current,
        )
        # Changed non-secret metadata -> tampered
        assert _auth_metadata_tampered({"client_id": "hijacked"}, current)
        assert _auth_metadata_tampered({"issuer": "https://evil"}, current)
        # Only a (masked) secret changed -> secrets can't be diffed, not tampered
        assert not _auth_metadata_tampered({"client_secret": "********"}, current)
        assert not _auth_metadata_tampered(None, current)

    def test_mask_env_keeps_keys(self):
        assert _mask_env({"A": "1", "B": ""}) == {"A": "********", "B": ""}

    def test_env_dict_encryption_roundtrip_and_no_double_encrypt(self):
        """env values encrypt at rest, decrypt back, and never double-encrypt."""
        from xagent.core.utils.encryption import decrypt_env_dict, encrypt_env_dict

        enc = encrypt_env_dict({"API_KEY": "secret", "EMPTY": ""})
        assert enc["API_KEY"].startswith("gAAAAAB")
        assert enc["EMPTY"] == ""  # empty values are not secrets
        # Re-encrypting an already-encrypted value is a no-op
        assert encrypt_env_dict(enc)["API_KEY"] == enc["API_KEY"]
        assert decrypt_env_dict(enc) == {"API_KEY": "secret", "EMPTY": ""}

    def test_build_server_config_parses_mcp_concurrency_config(self):
        """API request config accepts the explicit MCP concurrency opt-in."""
        server_data = MCPServerCreate(
            name="mail",
            transport="stdio",
            description="Mail MCP",
            config={
                "command": "python",
                "concurrency_safe": "true",
                "concurrent_tools": "list_messages, search_messages",
            },
        )

        config = _build_server_config(server_data)

        assert config.concurrency_safe is True
        assert config.concurrent_tools == ["list_messages", "search_messages"]


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
