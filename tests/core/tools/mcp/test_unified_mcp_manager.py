"""
Unit tests for the unified MCP models and database manager.

This module contains comprehensive tests for the unified data models and
DatabaseMCPServerManager functionality including load/save, CRUD operations,
validation, and error handling.
"""

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.storage.manager import Base
from xagent.core.tools.core.mcp.data_config import ContainerInfo
from xagent.core.tools.core.mcp.manager.db import DatabaseMCPServerManager, MCPServer


class TestDatabaseMCPServerManager:
    """Test cases for DatabaseMCPServerManager class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary in-memory database for testing."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    @pytest.fixture
    def manager(self, temp_db):
        """Create a DatabaseMCPServerManager instance for testing."""
        return DatabaseMCPServerManager(temp_db)

    @pytest.fixture
    def sample_stdio_config(self):
        """Create a sample stdio server configuration."""
        return {
            "name": "test-stdio",
            "transport": "stdio",
            "managed": "external",
            "description": "Test stdio server",
            "command": "python",
            "args": ["-m", "test.server"],
            "env": {"TEST_VAR": "test_value"},
            "cwd": "/tmp",
        }

    @pytest.fixture
    def sample_sse_config(self):
        """Create a sample SSE server configuration."""
        return {
            "name": "test-sse",
            "transport": "sse",
            "managed": "external",
            "description": "Test SSE server",
            "url": "http://localhost:8080/sse",
            "headers": {"Authorization": "Bearer token123"},
        }

    @pytest.fixture
    def sample_websocket_config(self):
        """Create a sample WebSocket server configuration."""
        return {
            "name": "test-websocket",
            "transport": "websocket",
            "managed": "external",
            "description": "Test WebSocket server",
            "url": "ws://localhost:8081/ws",
        }

    @pytest.fixture
    def sample_http_config(self):
        """Create a sample HTTP server configuration."""
        return {
            "name": "test-http",
            "transport": "streamable_http",
            "managed": "external",
            "description": "Test HTTP server",
            "url": "http://localhost:8082/api",
            "headers": {"Content-Type": "application/json"},
        }

    @pytest.fixture
    def sample_internal_config(self):
        """Create a sample internal managed server configuration."""
        return {
            "name": "test-internal",
            "transport": "sse",
            "managed": "internal",
            "description": "Test internal server",
            "url": "http://localhost:8083/sse",
            "docker_image": "test/server:latest",
            "volumes": ["/host/path:/container/path"],
            "restart_policy": "unless-stopped",
        }

    @pytest.fixture
    def sample_internal_config_with_env(self):
        """Create a sample internal managed server configuration with environment variables and working directory."""
        return {
            "name": "test-internal-env",
            "transport": "sse",
            "managed": "internal",
            "description": "Test internal server with environment",
            "url": "http://localhost:8084/sse",
            "docker_image": "test/server:latest",
            "docker_environment": {"NODE_ENV": "production", "PORT": "8084"},
            "docker_working_dir": "/app",
            "volumes": ["/host/path:/container/path"],
            "restart_policy": "unless-stopped",
        }

    def test_load_config_empty_db(self, manager):
        """Test loading configuration when database is empty."""
        config = manager.load_config()
        assert "servers" in config
        assert len(config["servers"]) == 0

    def test_add_and_load_config(self, manager, sample_stdio_config):
        """Test adding and loading configuration."""
        # Add a server
        config = manager.create_config(**sample_stdio_config)
        manager.add_server(config)

        # Load config
        loaded_config = manager.load_config()
        print(f"loaded: {loaded_config}")
        assert len(loaded_config["servers"]) == 1
        assert "test-stdio" in loaded_config["servers"]
        assert loaded_config["servers"]["test-stdio"]["transport"] == "stdio"

    def test_add_stdio_server(self, manager, sample_stdio_config):
        """Test adding a stdio server."""
        config = manager.create_config(**sample_stdio_config)
        manager.add_server(config)

        servers = manager.list_servers()
        assert len(servers) == 1

        server_data = servers[0]
        assert server_data.config.name == "test-stdio"
        assert server_data.config.transport == "stdio"
        assert server_data.config.managed == "external"
        assert server_data.config.command == "python"
        assert server_data.config.args == ["-m", "test.server"]

    def test_add_sse_server(self, manager, sample_sse_config):
        """Test adding an SSE server."""
        config = manager.create_config(**sample_sse_config)
        manager.add_server(config)

        server_data = manager.get_server("test-sse")
        assert server_data is not None
        assert server_data.config.transport == "sse"
        assert server_data.config.url == "http://localhost:8080/sse"
        assert server_data.config.headers == {"Authorization": "Bearer token123"}

    def test_add_websocket_server(self, manager, sample_websocket_config):
        """Test adding a WebSocket server."""
        config = manager.create_config(**sample_websocket_config)
        manager.add_server(config)

        server_data = manager.get_server("test-websocket")
        assert server_data is not None
        assert server_data.config.transport == "websocket"
        assert server_data.config.url == "ws://localhost:8081/ws"

    def test_add_http_server(self, manager, sample_http_config):
        """Test adding an HTTP server."""
        config = manager.create_config(**sample_http_config)
        manager.add_server(config)

        server_data = manager.get_server("test-http")
        assert server_data is not None
        assert server_data.config.transport == "streamable_http"
        assert server_data.config.url == "http://localhost:8082/api"

    @pytest.mark.docker
    def test_add_internal_server(self, manager, sample_internal_config):
        """Test adding an internal managed server."""
        config = manager.create_config(**sample_internal_config)
        manager.add_server(config)

        server_data = manager.get_server("test-internal")
        assert server_data is not None
        assert server_data.config.managed == "internal"
        assert server_data.config.docker_image == "test/server:latest"
        assert server_data.config.auto_start is True  # Default for internal servers

    def test_add_server_from_request(self, manager, sample_stdio_config):
        """Test adding a server from request data."""
        config = manager.add_server_from_request(sample_stdio_config)

        assert config.name == "test-stdio"
        server_data = manager.get_server("test-stdio")
        assert server_data is not None

    def test_add_duplicate_server_fails(self, manager, sample_stdio_config):
        """Test that adding a server with duplicate name fails."""
        config = manager.create_config(**sample_stdio_config)
        manager.add_server(config)

        with pytest.raises(ValueError, match="Server 'test-stdio' already exists"):
            manager.add_server(config)

    def test_remove_server(self, manager, sample_stdio_config):
        """Test removing a server."""
        config = manager.create_config(**sample_stdio_config)
        manager.add_server(config)

        assert len(manager.list_servers()) == 1

        result = manager.remove_server("test-stdio")
        assert result is True
        assert len(manager.list_servers()) == 0

    def test_remove_nonexistent_server(self, manager):
        """Test removing a server that doesn't exist."""
        result = manager.remove_server("nonexistent")
        assert result is False

    def test_list_servers_empty(self, manager):
        """Test listing servers when none exist."""
        servers = manager.list_servers()
        assert len(servers) == 0
        assert isinstance(servers, list)

    def test_list_multiple_servers(
        self, manager, sample_stdio_config, sample_sse_config
    ):
        """Test listing multiple servers."""
        config1 = manager.create_config(**sample_stdio_config)
        config2 = manager.create_config(**sample_sse_config)

        manager.add_server(config1)
        manager.add_server(config2)

        servers = manager.list_servers()
        assert len(servers) == 2
        names = [s.config.name for s in servers]
        assert "test-stdio" in names
        assert "test-sse" in names

    def test_get_server_exists(self, manager, sample_stdio_config):
        """Test getting an existing server."""
        config = manager.create_config(**sample_stdio_config)
        manager.add_server(config)

        server_data = manager.get_server("test-stdio")
        assert server_data is not None
        assert server_data.config.name == "test-stdio"

    def test_get_server_not_exists(self, manager):
        """Test getting a non-existent server."""
        server_data = manager.get_server("nonexistent")
        assert server_data is None

    def test_get_connections(self, manager, sample_stdio_config, sample_sse_config):
        """Test getting connections format."""
        config1 = manager.create_config(**sample_stdio_config)
        config2 = manager.create_config(**sample_sse_config)

        manager.add_server(config1)
        manager.add_server(config2)

        connections = manager.get_connections()
        assert len(connections) == 2
        assert "test-stdio" in connections
        assert "test-sse" in connections

        stdio_conn = connections["test-stdio"]
        assert stdio_conn["transport"] == "stdio"
        assert stdio_conn["command"] == "python"

        sse_conn = connections["test-sse"]
        assert sse_conn["transport"] == "sse"
        assert sse_conn["url"] == "http://localhost:8080/sse"

    def test_get_connections_with_filter(
        self, manager, sample_stdio_config, sample_sse_config
    ):
        """Test getting connections with filter function."""
        config1 = manager.create_config(**sample_stdio_config)
        config2 = manager.create_config(**sample_sse_config)

        manager.add_server(config1)
        manager.add_server(config2)

        # Filter to only get stdio servers
        def filter_stdio(query):
            return query.filter(MCPServer.transport == "stdio")

        connections = manager.get_connections(filter_stdio)
        assert len(connections) == 1
        assert "test-stdio" in connections
        assert "test-sse" not in connections

    def test_create_config_stdio_validation(self, manager):
        """Test stdio configuration validation."""
        # Missing command should fail
        with pytest.raises(
            ValueError, match="stdio transport requires 'command' parameter"
        ):
            manager.create_config(name="test", transport="stdio", managed="external")

    def test_create_config_http_validation(self, manager):
        """Test HTTP-based transport validation."""
        # Missing URL should fail for SSE
        with pytest.raises(ValueError, match="sse transport requires 'url' parameter"):
            manager.create_config(name="test", transport="sse", managed="external")

        # Missing URL should fail for WebSocket
        with pytest.raises(
            ValueError, match="websocket transport requires 'url' parameter"
        ):
            manager.create_config(
                name="test", transport="websocket", managed="external"
            )

    def test_create_config_internal_validation(self, manager):
        """Test internal server validation."""
        # Internal server without docker config should fail
        with pytest.raises(
            ValueError,
            match="internal managed servers require 'docker_image'",
        ):
            manager.create_config(
                name="test",
                transport="sse",
                managed="internal",
                url="http://localhost:8080",
            )

        # Internal server with stdio transport should fail
        with pytest.raises(
            ValueError, match="internal managed servers cannot use 'stdio' transport"
        ):
            manager.create_config(
                name="test",
                transport="stdio",
                managed="internal",
                docker_image="test:latest",
            )

    def test_create_config_external_validation(self, manager):
        """Test external server validation."""
        # External server with docker config should fail
        with pytest.raises(
            ValueError,
            match="external managed servers should not have docker configuration",
        ):
            manager.create_config(
                name="test",
                transport="sse",
                managed="external",
                url="http://localhost:8080",
                docker_image="test:latest",
            )

        # External server with auto_start should fail
        with pytest.raises(
            ValueError,
            match="external managed servers should not have auto_start parameter",
        ):
            manager.create_config(
                name="test",
                transport="sse",
                managed="external",
                url="http://localhost:8080",
                auto_start=True,
            )

    def test_add_server_from_request_with_extra_params(self, manager):
        """Test that add_server_from_request forwards extra parameters correctly."""
        # Test internal server with auto_start=False
        request_data = {
            "name": "test-extra-params",
            "transport": "sse",
            "managed": "internal",
            "url": "http://localhost:8080/sse",
            "docker_image": "test/server:latest",
            "auto_start": False,  # This should be forwarded
        }

        config = manager.add_server_from_request(request_data)

        assert config.name == "test-extra-params"
        assert config.transport == "sse"
        assert config.managed == "internal"
        assert not config.auto_start  # Should be False, not default True

        # Test internal server without auto_start (should default to True)
        request_data_default = {
            "name": "test-default-autostart",
            "transport": "sse",
            "managed": "internal",
            "url": "http://localhost:8081/sse",
            "docker_image": "test/server:latest",
            # No auto_start specified
        }

        config_default = manager.add_server_from_request(request_data_default)
        assert config_default.auto_start  # Should default to True

    def test_add_server_from_request_missing_required_params(self, manager):
        """Test that KeyError is raised when required parameters are missing."""
        # Test missing 'name'
        with pytest.raises(KeyError, match="Required parameter 'name' is missing"):
            manager.add_server_from_request(
                {"transport": "stdio", "command": "echo", "args": ["hello"]}
            )

        # Test missing 'transport'
        with pytest.raises(KeyError, match="Required parameter 'transport' is missing"):
            manager.add_server_from_request(
                {"name": "test-server", "command": "echo", "args": ["hello"]}
            )

        # Test missing both 'name' and 'transport'
        with pytest.raises(KeyError, match="Required parameter 'name' is missing"):
            manager.add_server_from_request({"command": "echo", "args": ["hello"]})

    def test_save_config_with_path_objects(self, manager):
        """Test that Path objects in configuration are properly handled."""
        # Create a config with Path object for cwd
        config = manager.create_config(
            name="test-path",
            transport="stdio",
            command="echo",
            args=["hello"],
            cwd=Path("/tmp/test"),
        )

        # Add the server - this should not raise an error
        manager.add_server(config)

        # Get the server back
        server_data = manager.get_server("test-path")
        assert server_data is not None
        assert server_data.config.name == "test-path"
        assert server_data.config.cwd == "/tmp/test"  # Should be converted to string

    @pytest.mark.docker
    def test_start_server_internal_success(
        self, manager, sample_internal_config, mocker
    ):
        """Test successfully starting an internal server."""
        # Add server configuration
        manager.add_server_from_request(sample_internal_config)

        # Mock DockerManager methods
        mock_docker_manager = mocker.patch.object(manager, "_docker_manager")
        container_info = ContainerInfo(
            container_id="test-container-id",
            container_name="mcp-test-internal",
        )
        mock_docker_manager.start_container.return_value = container_info
        # Mock get_container_status for status update
        from xagent.core.tools.core.mcp.data_config import ContainerStatus

        mock_docker_manager.get_container_status.return_value = (
            ContainerStatus.RUNNING,
            None,
            "1h 30m",
        )

        # Start server
        result = manager.start_server("test-internal")

        # Verify result
        assert result is True
        mock_docker_manager.start_container.assert_called_once_with(
            name="test-internal",
            docker_url=None,
            image="test/server:latest",
            volumes=["/host/path:/container/path"],
            bind_ports=None,
            environment=None,
            working_dir=None,
            restart_policy="unless-stopped",
        )

        # Verify server data was updated
        server_data = manager.get_server("test-internal")
        assert server_data.status.status.value == "running"
        assert server_data.container_info == container_info

    @pytest.mark.docker
    def test_start_server_internal_with_environment_success(
        self, manager, sample_internal_config_with_env, mocker
    ):
        """Test successfully starting an internal server with environment variables and working directory."""
        # Add server configuration
        manager.add_server_from_request(sample_internal_config_with_env)

        # Mock DockerManager methods
        mock_docker_manager = mocker.patch.object(manager, "_docker_manager")
        container_info = ContainerInfo(
            container_id="test-container-env-id",
            container_name="mcp-test-internal-env",
        )
        mock_docker_manager.start_container.return_value = container_info

        # Mock get_container_status for status update
        from xagent.core.tools.core.mcp.data_config import ContainerStatus

        mock_docker_manager.get_container_status.return_value = (
            ContainerStatus.RUNNING,
            None,
            "1h 30m",
        )

        # Start server
        result = manager.start_server("test-internal-env")

        # Verify result
        assert result is True
        mock_docker_manager.start_container.assert_called_once_with(
            name="test-internal-env",
            docker_url=None,
            image="test/server:latest",
            volumes=["/host/path:/container/path"],
            bind_ports=None,
            environment={"NODE_ENV": "production", "PORT": "8084"},
            working_dir="/app",
            restart_policy="unless-stopped",
        )

        # Verify server data was updated
        server_data = manager.get_server("test-internal-env")
        assert server_data.status.status.value == "running"
        assert server_data.container_info == container_info

    def test_start_server_external_fails(self, manager, sample_sse_config):
        """Test that starting an external server fails."""
        # Add external server configuration
        manager.add_server_from_request(sample_sse_config)

        # Attempt to start external server should fail
        with pytest.raises(ValueError, match="Cannot start external server 'test-sse'"):
            manager.start_server("test-sse")

    def test_start_server_not_found(self, manager):
        """Test starting a non-existent server fails."""
        with pytest.raises(ValueError, match="Server 'non-existent' not found"):
            manager.start_server("non-existent")

    @pytest.mark.docker
    def test_stop_server_internal_success(
        self, manager, sample_internal_config, mocker
    ):
        """Test successfully stopping an internal server."""
        # Add server configuration
        manager.add_server_from_request(sample_internal_config)

        # Mock DockerManager methods
        mock_docker_manager = mocker.patch.object(manager, "_docker_manager")
        mock_docker_manager.stop_container.return_value = True
        from xagent.core.tools.core.mcp.data_config import ContainerStatus

        mock_docker_manager.get_container_status.return_value = (
            ContainerStatus.STOPPED,
            None,
            None,
        )

        # Stop server
        result = manager.stop_server("test-internal")

        # Verify result
        assert result is True
        mock_docker_manager.stop_container.assert_called_once_with(
            "test-internal", None
        )

        # Verify server status was updated
        server_data = manager.get_server("test-internal")
        assert server_data.status.status.value == "stopped"

    def test_stop_server_external_fails(self, manager, sample_sse_config):
        """Test that stopping an external server fails."""
        # Add external server configuration
        manager.add_server_from_request(sample_sse_config)

        # Attempt to stop external server should fail
        with pytest.raises(ValueError, match="Cannot stop external server 'test-sse'"):
            manager.stop_server("test-sse")

    @pytest.mark.docker
    def test_restart_server_internal_success(
        self, manager, sample_internal_config, mocker
    ):
        """Test successfully restarting an internal server."""
        # Add server configuration
        manager.add_server_from_request(sample_internal_config)

        # Mock DockerManager methods
        mock_docker_manager = mocker.patch.object(manager, "_docker_manager")
        mock_docker_manager.restart_container.return_value = True
        from xagent.core.tools.core.mcp.data_config import ContainerStatus

        mock_docker_manager.get_container_status.return_value = (
            ContainerStatus.RUNNING,
            None,
            "1h 30m",
        )

        # Restart server
        result = manager.restart_server("test-internal")

        # Verify result
        assert result is True
        mock_docker_manager.restart_container.assert_called_once_with(
            "test-internal", None
        )

        # Verify server status was updated
        server_data = manager.get_server("test-internal")
        assert server_data.status.status.value == "running"

    def test_restart_server_external_fails(self, manager, sample_sse_config):
        """Test that restarting an external server fails."""
        # Add external server configuration
        manager.add_server_from_request(sample_sse_config)

        # Attempt to restart external server should fail
        with pytest.raises(
            ValueError, match="Cannot restart external server 'test-sse'"
        ):
            manager.restart_server("test-sse")

    @pytest.mark.docker
    def test_get_logs_internal_success(self, manager, sample_internal_config, mocker):
        """Test successfully getting logs from an internal server."""
        # Add server configuration
        manager.add_server_from_request(sample_internal_config)

        # Mock DockerManager methods
        mock_docker_manager = mocker.patch.object(manager, "_docker_manager")
        mock_logs = ["Log line 1", "Log line 2", "Log line 3"]
        mock_docker_manager.get_container_logs.return_value = mock_logs
        from xagent.core.tools.core.mcp.data_config import ContainerStatus

        mock_docker_manager.get_container_status.return_value = (
            ContainerStatus.RUNNING,
            None,
            "1h 30m",
        )

        # Get logs
        result = manager.get_logs("test-internal")

        # Verify result
        assert result == mock_logs
        mock_docker_manager.get_container_logs.assert_called_once_with(
            "test-internal", None, 100
        )

        # Verify logs were cached
        server_data = manager.get_server("test-internal")
        assert server_data.status.container_logs.logs == mock_logs

    def test_get_logs_external_returns_none(self, manager, sample_sse_config):
        """Test that getting logs from an external server returns None."""
        # Add external server configuration
        manager.add_server_from_request(sample_sse_config)

        # Get logs from external server should return None
        result = manager.get_logs("test-sse")
        assert result is None

    @pytest.mark.docker
    def test_get_server_updates_status_internal(
        self, manager, sample_internal_config, mocker
    ):
        """Test that get_server updates status for internal servers."""
        # Add server configuration
        manager.add_server_from_request(sample_internal_config)

        # Mock DockerManager methods
        mock_docker_manager = mocker.patch.object(manager, "_docker_manager")
        from xagent.core.tools.core.mcp.data_config import ContainerStatus

        mock_docker_manager.get_container_status.return_value = (
            ContainerStatus.RUNNING,
            None,
            "1h 30m",
        )

        # Get server (should update status)
        server_data = manager.get_server("test-internal")

        # Verify DockerManager was called to update status
        mock_docker_manager.get_container_status.assert_called_once_with(
            "test-internal", None
        )

        # Verify status was updated
        assert server_data.status.status.value == "running"
        assert server_data.status.uptime == "1h 30m"

    def test_get_server_updates_status_external(
        self, manager, sample_sse_config, mocker
    ):
        """Test that get_server updates status for external servers."""
        # Add server configuration
        manager.add_server_from_request(sample_sse_config)

        # Mock DockerManager methods
        mock_docker_manager = mocker.patch.object(manager, "_docker_manager")
        from xagent.core.tools.core.mcp.data_config import ContainerStatus

        mock_docker_manager.check_external_server_health.return_value = (
            ContainerStatus.REACHABLE,
            "healthy",
        )

        # Get server (should update status)
        server_data = manager.get_server("test-sse")

        # Verify DockerManager was called to check health
        mock_docker_manager.check_external_server_health.assert_called_once_with(
            "http://localhost:8080/sse"
        )

        # Verify status was updated
        assert server_data.status.status.value == "reachable"
        assert server_data.status.health_status == "healthy"

    def test_get_server_not_found(self, manager):
        """Test getting a non-existent server returns None."""
        result = manager.get_server("non-existent")
        assert result is None

    @pytest.mark.docker
    def test_start_server_docker_error_handling(
        self, manager, sample_internal_config, mocker
    ):
        """Test error handling when Docker operations fail."""
        # Add server configuration
        manager.add_server_from_request(sample_internal_config)

        # Mock DockerManager to raise an exception
        mock_docker_manager = mocker.patch.object(manager, "_docker_manager")
        mock_docker_manager.start_container.side_effect = ValueError(
            "Docker connection failed"
        )
        from xagent.core.tools.core.mcp.data_config import ContainerStatus

        mock_docker_manager.get_container_status.return_value = (
            ContainerStatus.ERROR,
            None,
            None,
        )

        # Start server should propagate the error
        with pytest.raises(ValueError, match="Docker connection failed"):
            manager.start_server("test-internal")

        # Verify server status was set to ERROR
        server_data = manager.get_server("test-internal")
        assert server_data.status.status.value == "error"

    @pytest.mark.docker
    def test_get_logs_with_custom_lines(self, manager, sample_internal_config, mocker):
        """Test getting logs with custom number of lines."""
        # Add server configuration
        manager.add_server_from_request(sample_internal_config)

        # Mock DockerManager methods
        mock_docker_manager = mocker.patch.object(manager, "_docker_manager")
        mock_logs = ["Log line 1", "Log line 2", "Log line 3"]
        mock_docker_manager.get_container_logs.return_value = mock_logs
        from xagent.core.tools.core.mcp.data_config import ContainerStatus

        mock_docker_manager.get_container_status.return_value = (
            ContainerStatus.RUNNING,
            None,
            "1h 30m",
        )

        # Get logs with custom lines
        result = manager.get_logs("test-internal", lines=500)

        # Verify result
        assert result == mock_logs
        mock_docker_manager.get_container_logs.assert_called_once_with(
            "test-internal", None, 500
        )

    @pytest.mark.docker
    def test_get_logs_invalid_lines_range(self, manager, sample_internal_config):
        """Test that invalid lines parameter raises ValueError."""
        # Add server configuration
        manager.add_server_from_request(sample_internal_config)

        # Test lines too low
        with pytest.raises(ValueError, match="lines must be between 1 and 1000"):
            manager.get_logs("test-internal", lines=0)

        # Test lines too high
        with pytest.raises(ValueError, match="lines must be between 1 and 1000"):
            manager.get_logs("test-internal", lines=1001)

    def test_db_session_rollback_on_error(self, manager, sample_stdio_config):
        """Test that database session is rolled back on error."""
        # Add a server successfully
        config = manager.create_config(**sample_stdio_config)
        manager.add_server(config)

        # Try to add duplicate - should rollback
        try:
            manager.add_server(config)
        except ValueError:
            pass

        # Verify database is in consistent state
        servers = manager.list_servers()
        assert len(servers) == 1
