import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.storage.manager import Base
from xagent.core.tools.core.mcp.data_config import ContainerStatus
from xagent.core.tools.core.mcp.manager.db import MCPServer
from xagent.core.tools.core.mcp.manager.migration import MCPServerMigrator, try_migrate


@pytest.fixture(autouse=True)
def mock_docker_manager():
    """Mock DockerManager to avoid Docker connection."""
    with patch("xagent.core.tools.core.mcp.manager.file.DockerManager") as mock:
        mock_instance = Mock()
        mock_instance.get_container_status.return_value = (
            ContainerStatus.UNKNOWN,
            None,
            None,
        )
        mock_instance.check_external_server_health.return_value = (
            ContainerStatus.UNKNOWN,
            None,
        )
        mock_instance.close_all = Mock()
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def db_session():
    """Create in-memory SQLite database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def sample_yaml_config():
    """Sample YAML configuration data."""
    return {
        "servers": {
            "test-server-1": {
                "transport": "stdio",
                "command": "python",
                "args": ["-m", "test_server"],
                "env": {"TEST_VAR": "value1"},
                "managed": "external",
            },
            "test-server-2": {
                "transport": "sse",
                "url": "http://localhost:8080/sse",
                "headers": {"Authorization": "Bearer token"},
                "managed": "external",
            },
            "docker-server": {
                "transport": "websocket",
                "url": "ws://localhost:9090/ws",
                "docker_image": "test/server:latest",
                "docker_url": "unix:///var/run/docker.sock",
                "volumes": ["/data:/app/data"],
                "bind_ports": {"8080": 8080},
                "restart_policy": "unless-stopped",
                "managed": "internal",
            },
        }
    }


@pytest.fixture
def yaml_config_file(temp_storage, sample_yaml_config):
    """Create YAML config file."""
    config_file = temp_storage / "mcp_servers.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(sample_yaml_config, f)
    return config_file


class TestMCPServerMigrator:
    """Test cases for MCPServerMigrator."""

    def test_init(self, temp_storage, db_session):
        """Test migrator initialization."""
        migrator = MCPServerMigrator(temp_storage, db_session)

        assert migrator.storage_root == temp_storage
        assert migrator.yaml_config_file == temp_storage / "mcp_servers.yaml"
        assert migrator.db_session == db_session

    def test_should_migrate_no_yaml_file(self, temp_storage, db_session):
        """Test should_migrate when YAML file doesn't exist."""
        migrator = MCPServerMigrator(temp_storage, db_session)

        assert not migrator.should_migrate()

    def test_should_migrate_yaml_exists_empty_db(self, yaml_config_file, db_session):
        """Test should_migrate when YAML exists and DB is empty."""
        migrator = MCPServerMigrator(yaml_config_file.parent, db_session)

        assert migrator.should_migrate()

    def test_should_migrate_yaml_exists_db_has_data(self, yaml_config_file, db_session):
        """Test should_migrate when YAML exists but DB has data."""
        # Add a server to DB
        server = MCPServer(
            name="existing-server",
            transport="stdio",
            command="test",
            managed="external",
        )
        db_session.add(server)
        db_session.commit()

        migrator = MCPServerMigrator(yaml_config_file.parent, db_session)

        assert not migrator.should_migrate()

    @patch("sqlalchemy.inspect")
    def test_should_migrate_no_table(self, mock_inspect, yaml_config_file, db_session):
        """Test should_migrate when database table doesn't exist."""
        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = []
        mock_inspect.return_value = mock_inspector

        migrator = MCPServerMigrator(yaml_config_file.parent, db_session)

        assert migrator.should_migrate()

    def test_migrate_no_migration_needed(self, temp_storage, db_session):
        """Test migrate when no migration is needed."""
        migrator = MCPServerMigrator(temp_storage, db_session)

        result = migrator.migrate()

        assert result["migrated"] is False
        assert result["reason"] == "Migration not needed"
        assert result["servers_migrated"] == 0
        assert result["errors"] == []

    def test_migrate_successful(self, yaml_config_file, db_session, sample_yaml_config):
        """Test successful migration."""
        migrator = MCPServerMigrator(yaml_config_file.parent, db_session)

        result = migrator.migrate()

        assert result["migrated"] is True
        assert result["servers_migrated"] == 3
        assert result["total_servers"] == 3
        assert result["errors"] == []

        # Verify servers were added to database
        servers = db_session.query(MCPServer).all()
        assert len(servers) == 3

        server_names = {s.name for s in servers}
        assert server_names == {"test-server-1", "test-server-2", "docker-server"}

        # Verify YAML file was backed up
        backup_file = yaml_config_file.with_suffix(".yaml.backup")
        assert backup_file.exists()
        assert not yaml_config_file.exists()

    def test_migrate_with_existing_server(self, yaml_config_file, db_session):
        """Test skips migration when some servers already exist in DB."""
        existing_server = MCPServer(
            name="test-server-1",
            transport="stdio",
            command="existing",
            managed="external",
        )
        db_session.add(existing_server)
        db_session.commit()

        migrator = MCPServerMigrator(yaml_config_file.parent, db_session)

        result = migrator.migrate()

        assert result["migrated"] is False

    @patch("xagent.core.tools.core.mcp.manager.migration.YamlMCPServerManager")
    def test_migrate_with_errors(self, mock_yaml_manager, yaml_config_file, db_session):
        """Test migration with some errors."""
        # Mock YAML manager to return servers, but make one fail
        mock_server_data = Mock()
        mock_server_data.config.name = "failing-server"
        mock_yaml_manager.return_value.list_servers.return_value = [mock_server_data]

        # Mock database manager to raise exception
        with patch(
            "xagent.core.tools.core.mcp.manager.migration.DatabaseMCPServerManager"
        ) as mock_db_manager:
            mock_db_manager.return_value.add_server.side_effect = Exception("DB Error")

            migrator = MCPServerMigrator(yaml_config_file.parent, db_session)

            result = migrator.migrate()

            assert result["migrated"] is True
            assert result["servers_migrated"] == 0
            assert result["total_servers"] == 1
            assert len(result["errors"]) == 1
            assert "failing-server" in result["errors"][0]


class TestAutoMigrateIfNeeded:
    """Test cases for try_migrate function."""

    def test_auto_migrate_not_needed(self, temp_storage, db_session):
        """Test auto migrate when migration is not needed."""
        result = try_migrate(temp_storage, db_session)

        assert result is None

    def test_auto_migrate_needed(self, yaml_config_file, db_session):
        """Test auto migrate when migration is needed."""
        result = try_migrate(yaml_config_file.parent, db_session)

        assert result is not None
        assert result["migrated"] is True
        assert result["servers_migrated"] == 3


class TestMigrationIntegration:
    """Integration tests for migration functionality."""

    def test_full_migration_workflow(self, temp_storage, db_session):
        """Test complete migration workflow."""
        # Create YAML config
        yaml_config = {
            "servers": {
                "integration-server": {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["-m", "integration_test"],
                    "managed": "external",
                    "description": "Integration test server",
                }
            }
        }

        config_file = temp_storage / "mcp_servers.yaml"
        with open(config_file, "w") as f:
            yaml.safe_dump(yaml_config, f)

        # Perform migration
        migrator = MCPServerMigrator(temp_storage, db_session)

        # Check migration is needed
        assert migrator.should_migrate()

        # Perform migration
        result = migrator.migrate()

        # Verify results
        assert result["migrated"] is True
        assert result["servers_migrated"] == 1

        # Verify database state
        server = (
            db_session.query(MCPServer)
            .filter(MCPServer.name == "integration-server")
            .first()
        )

        assert server is not None
        assert server.transport == "stdio"
        assert server.command == "python"
        assert server.args == ["-m", "integration_test"]
        assert server.managed == "external"
        assert server.description == "Integration test server"

        # Verify YAML file was backed up
        backup_file = config_file.with_suffix(".yaml.backup")
        assert backup_file.exists()
        assert not config_file.exists()

        # Verify second migration is not needed
        assert not migrator.should_migrate()

    def test_migration_preserves_all_config_fields(self, temp_storage, db_session):
        """Test that migration preserves all configuration fields."""
        # Create comprehensive YAML config
        yaml_config = {
            "servers": {
                "comprehensive-server": {
                    "transport": "websocket",
                    "url": "ws://localhost:8080/ws",
                    "headers": {"Authorization": "Bearer test-token"},
                    "managed": "internal",
                    "description": "Comprehensive test server",
                    "docker_image": "test/comprehensive:latest",
                    "docker_url": "unix:///var/run/docker.sock",
                    "docker_environment": {"ENV_VAR": "test_value"},
                    "docker_working_dir": "/app",
                    "volumes": ["/host:/container"],
                    "bind_ports": {"8080": "8080", "9090": 9090},
                    "restart_policy": "always",
                    "auto_start": True,
                }
            }
        }

        config_file = temp_storage / "mcp_servers.yaml"
        with open(config_file, "w") as f:
            yaml.safe_dump(yaml_config, f)

        # Perform migration
        migrator = MCPServerMigrator(temp_storage, db_session)
        result = migrator.migrate()

        assert result["migrated"] is True

        # Verify all fields were preserved
        server = db_session.query(MCPServer).first()

        assert server.name == "comprehensive-server"
        assert server.transport == "websocket"
        assert server.url == "ws://localhost:8080/ws"
        assert server.headers == {"Authorization": "Bearer test-token"}
        assert server.managed == "internal"
        assert server.description == "Comprehensive test server"
        assert server.docker_image == "test/comprehensive:latest"
        assert server.docker_url == "unix:///var/run/docker.sock"
        assert server.docker_environment == {"ENV_VAR": "test_value"}
        assert server.docker_working_dir == "/app"
        assert server.volumes == ["/host:/container"]
        assert server.bind_ports == {"8080": "8080", "9090": 9090}
        assert server.restart_policy == "always"
        assert server.auto_start is True


@pytest.mark.parametrize(
    "yaml_exists,db_empty,expected",
    [
        (False, True, False),  # No YAML file
        (True, False, False),  # YAML exists but DB has data
        (True, True, True),  # YAML exists and DB is empty
    ],
)
def test_should_migrate_scenarios(
    temp_storage, db_session, yaml_exists, db_empty, expected
):
    """Test should_migrate under different scenarios."""
    # Create YAML file if needed
    if yaml_exists:
        config_file = temp_storage / "mcp_servers.yaml"
        with open(config_file, "w") as f:
            yaml.safe_dump({"servers": {"test": {"transport": "stdio"}}}, f)

    # Add DB data if needed
    if not db_empty:
        server = MCPServer(name="existing", transport="stdio", managed="external")
        db_session.add(server)
        db_session.commit()

    migrator = MCPServerMigrator(temp_storage, db_session)

    assert migrator.should_migrate() == expected
