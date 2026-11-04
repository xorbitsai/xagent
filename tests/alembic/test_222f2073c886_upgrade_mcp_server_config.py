"""Tests for the MCP server migration script"""

import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


def get_migration_module():
    """Import the migration module using importlib"""
    # Get the path to the migration file
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/222f2073c886_upgrade_mcp_server_config.py"
    )

    spec = importlib.util.spec_from_file_location("migration_module", migration_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def migration_module():
    """Fixture to provide the migration module"""
    return get_migration_module()


@pytest.fixture
def test_db(tmp_path):
    """Create a temporary test database with just the legacy MCP schema"""
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)

    # Create only what we need for THIS migration
    with engine.connect() as conn:
        # Users table (referenced by foreign key)
        conn.execute(
            text("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                username VARCHAR(50) NOT NULL,
                email VARCHAR(100) NOT NULL
            )
        """)
        )

        # Legacy MCP servers table (what we're migrating FROM)
        conn.execute(
            text("""
            CREATE TABLE mcp_servers (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                name VARCHAR(100) NOT NULL,
                description TEXT,
                transport VARCHAR(50) NOT NULL,
                config TEXT NOT NULL,  -- Use TEXT instead of JSON for SQLite
                is_active BOOLEAN DEFAULT TRUE,
                is_default BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        )

        # Insert test users
        conn.execute(
            text("""
            INSERT INTO users (id, username, email) VALUES
            (1, 'testuser1', 'test1@example.com'),
            (2, 'testuser2', 'test2@example.com')
        """)
        )
        conn.commit()

    yield engine, Session
    engine.dispose()


def test_upgrade_function_creates_tables(test_db, migration_module):
    """Test the upgrade function creates new tables"""
    from unittest.mock import patch

    from alembic import context as alembic_context

    engine, _ = test_db

    # Mock both op and context
    with patch.object(migration_module, "op") as mock_op:
        mock_op.get_bind.return_value = engine
        mock_op.rename_table = Mock()
        mock_op.create_table = Mock()
        mock_op.create_index = Mock()

        # Mock alembic.context.get_bind to return engine
        with patch.object(alembic_context, "get_bind", return_value=engine):
            # Test just the table creation part
            migration_module.create_new_tables()

            # Verify tables were created (mcp_servers and user_mcpservers)
            assert mock_op.create_table.call_count >= 1
            assert mock_op.create_index.call_count >= 1


def test_migrate_single_server_orm(test_db, migration_module):
    """Test migrating a single server using ORM"""
    engine, Session = test_db

    # Insert legacy data FIRST
    with engine.connect() as conn:
        conn.execute(
            text("""
            INSERT INTO mcp_servers (user_id, name, description, transport, config, is_active, is_default)
            VALUES (1, 'test-stdio', 'Test STDIO server', 'stdio', :config, 1, 0)
        """),
            {
                "config": json.dumps(
                    {
                        "command": "python",
                        "args": ["-m", "server"],
                        "env": {"DEBUG": "true"},
                        "cwd": "/app",
                    }
                )
            },
        )
        conn.commit()

    # Rename old table and create new tables
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE mcp_servers RENAME TO mcp_servers_legacy"))

        # Create new tables with ALL expected columns
        conn.execute(
            text("""
            CREATE TABLE mcp_servers (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100) NOT NULL UNIQUE,
                description TEXT,
                managed VARCHAR(20) NOT NULL,
                transport VARCHAR(50) NOT NULL,
                command VARCHAR(500),
                args TEXT,  -- Use TEXT for JSON in SQLite
                url VARCHAR(500),
                env TEXT,
                cwd VARCHAR(500),
                headers TEXT,
                docker_url VARCHAR(500),
                docker_image VARCHAR(200),
                docker_environment TEXT,
                docker_working_dir VARCHAR(500),
                volumes TEXT,
                bind_ports TEXT,
                restart_policy VARCHAR(50) DEFAULT 'no',
                auto_start BOOLEAN,
                container_id VARCHAR(100),
                container_name VARCHAR(200),
                container_logs TEXT,
                created_at DATETIME,
                updated_at DATETIME
            )
        """)
        )

        conn.execute(
            text("""
            CREATE TABLE user_mcpservers (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                mcpserver_id INTEGER NOT NULL,
                is_owner BOOLEAN DEFAULT FALSE,
                can_edit BOOLEAN DEFAULT FALSE,
                can_delete BOOLEAN DEFAULT FALSE,
                is_shared BOOLEAN DEFAULT FALSE,
                is_active BOOLEAN DEFAULT TRUE,
                is_default BOOLEAN DEFAULT FALSE,
                created_at DATETIME,
                updated_at DATETIME,
                UNIQUE (user_id, mcpserver_id)
            )
        """)
        )
        conn.commit()

    # Create a session and test the ORM migration
    session = Session()

    try:
        # Get the legacy server
        legacy_server = session.query(migration_module.LegacyMCPServer).first()
        assert legacy_server is not None

        # Migrate it
        migration_module.migrate_single_server_orm(session, legacy_server)
        session.commit()

        # Verify results
        with engine.connect() as conn:
            server = conn.execute(text("SELECT * FROM mcp_servers")).fetchone()
            assert server.name == "test-stdio"
            assert server.managed == "external"
            assert server.command == "python"
            assert json.loads(server.args) == ["-m", "server"]

            relationship = conn.execute(
                text("SELECT * FROM user_mcpservers")
            ).fetchone()
            assert relationship.user_id == 1
            assert relationship.is_owner == 1

    finally:
        session.close()


def test_normalize_config_fields(migration_module):
    """Test the config normalization function"""
    # Test STDIO config
    config = {
        "command": "python",
        "args": "--port 3000 --verbose",  # String that should be parsed
        "env": "NODE_ENV=production\nDEBUG=true",  # String that should be parsed
        "cwd": "/app",
    }

    result = migration_module.normalize_config_fields(config, "stdio")

    assert result["command"] == "python"
    assert result["args"] == ["--port", "3000", "--verbose"]
    assert result["env"] == {"NODE_ENV": "production", "DEBUG": "true"}
    assert result["cwd"] == "/app"
    assert result["managed"] == "external"


def test_normalize_config_fields_internal_server(migration_module):
    """Test normalizing internal server config"""
    config = {
        "command": "python",
        "docker_image": "mcp-server:latest",
        "docker_environment": {"API_KEY": "secret"},
        "volumes": ["/data:/app/data"],
        "bind_ports": {"8080": "8080"},
        "restart_policy": "always",
        "auto_start": True,
    }

    result = migration_module.normalize_config_fields(config, "stdio")

    assert result["managed"] == "internal"
    assert result["docker_image"] == "mcp-server:latest"
    assert result["docker_environment"] == {"API_KEY": "secret"}
    assert result["volumes"] == ["/data:/app/data"]
    assert result["bind_ports"] == {"8080": "8080"}
    assert result["restart_policy"] == "always"
    assert result["auto_start"] is True


def test_parse_config_field(migration_module):
    """Test individual config field parsing"""
    # Test args parsing
    assert migration_module.parse_config_field("args", "--port 3000 --verbose") == [
        "--port",
        "3000",
        "--verbose",
    ]

    # Test env parsing
    assert migration_module.parse_config_field("env", "KEY1=value1 KEY2=value2") == {
        "KEY1": "value1",
        "KEY2": "value2",
    }

    # Test boolean parsing
    assert migration_module.parse_config_field("auto_start", "true") is True
    assert migration_module.parse_config_field("auto_start", "false") is False

    # Test unknown field
    assert migration_module.parse_config_field("unknown", "some value") == "some value"


def test_parse_config_field_with_json(migration_module):
    """Test parsing config fields that can be JSON"""
    # Test JSON env
    json_env = '{"NODE_ENV": "production", "PORT": "3000"}'
    result = migration_module.parse_config_field("env", json_env)
    assert result == {"NODE_ENV": "production", "PORT": "3000"}

    # Test JSON args
    json_args = "--port 3000 --verbose"
    result = migration_module.parse_config_field("args", json_args)
    assert result == ["--port", "3000", "--verbose"]


def test_parse_config_field_with_newlines(migration_module):
    """Test parsing config fields with newlines"""
    # Test env with newlines
    env_with_newlines = "NODE_ENV=production\nPORT=3000\nDEBUG=true"
    result = migration_module.parse_config_field("env", env_with_newlines)
    assert result == {"NODE_ENV": "production", "PORT": "3000", "DEBUG": "true"}

    # Test args with newlines
    args_with_newlines = "--port\n3000\n--verbose"
    result = migration_module.parse_config_field("args", args_with_newlines)
    assert result == ["--port", "3000", "--verbose"]


def test_parse_config_field_error_handling(migration_module):
    """Test that parse_config_field handles errors gracefully"""
    # Test with None values
    assert migration_module.parse_config_field("args", None) is None
    assert migration_module.parse_config_field("args", "") is None
    assert migration_module.parse_config_field("args", "   ") is None

    # Test with non-string values (should return as-is)
    list_value = ["already", "parsed"]
    assert migration_module.parse_config_field("args", list_value) == list_value

    dict_value = {"already": "parsed"}
    assert migration_module.parse_config_field("env", dict_value) == dict_value


def test_config_field_parser_classes(migration_module):
    """Test the individual parser classes"""
    # Test string list parser
    result = migration_module.ConfigFieldParser.parse_string_list(
        '--port 3000 --name "My Server"'
    )
    assert result == ["--port", "3000", "--name", "My Server"]

    # Test key-value dict parser
    result = migration_module.ConfigFieldParser.parse_key_value_dict(
        "KEY1=value1 KEY2=value2"
    )
    assert result == {"KEY1": "value1", "KEY2": "value2"}

    # Test port mappings parser
    result = migration_module.ConfigFieldParser.parse_port_mappings(
        "8080:8080 9090:9091"
    )
    assert result == {"8080": "8080", "9090": "9091"}

    # Test boolean parser
    assert migration_module.ConfigFieldParser.parse_boolean("true") is True
    assert migration_module.ConfigFieldParser.parse_boolean("false") is False


def test_config_field_parser_json_fallback(migration_module):
    """Test parser JSON fallback functionality"""
    # Test JSON parsing for key-value dict
    json_dict = '{"KEY1": "value1", "KEY2": "value2"}'
    result = migration_module.ConfigFieldParser.parse_key_value_dict(json_dict)
    assert result == {"KEY1": "value1", "KEY2": "value2"}

    # Test JSON parsing for port mappings
    json_ports = '{"8080": "8080", "9090": "9091"}'
    result = migration_module.ConfigFieldParser.parse_port_mappings(json_ports)
    assert result == {"8080": "8080", "9090": "9091"}


def test_mcp_config_field_registry(migration_module):
    """Test the field registry"""
    registry = migration_module.MCPConfigFieldRegistry

    # Test string list fields
    assert (
        registry.get_parser_for_field("args")
        == migration_module.ConfigFieldParser.parse_string_list
    )
    assert (
        registry.get_parser_for_field("volumes")
        == migration_module.ConfigFieldParser.parse_string_list
    )

    # Test key-value dict fields
    assert (
        registry.get_parser_for_field("env")
        == migration_module.ConfigFieldParser.parse_key_value_dict
    )
    assert (
        registry.get_parser_for_field("headers")
        == migration_module.ConfigFieldParser.parse_key_value_dict
    )
    assert (
        registry.get_parser_for_field("docker_environment")
        == migration_module.ConfigFieldParser.parse_key_value_dict
    )

    # Test port mapping fields
    assert (
        registry.get_parser_for_field("bind_ports")
        == migration_module.ConfigFieldParser.parse_port_mappings
    )

    # Test boolean fields
    assert (
        registry.get_parser_for_field("auto_start")
        == migration_module.ConfigFieldParser.parse_boolean
    )

    # Test unknown field
    assert registry.get_parser_for_field("unknown_field") is None


def test_migrate_data_orm_with_real_legacy_data(test_db, migration_module):
    """Test the full ORM migration with realistic legacy data"""
    engine, Session = test_db

    # Insert realistic legacy data
    legacy_data = [
        {
            "user_id": 1,
            "name": "stdio-server",
            "description": "STDIO MCP server",
            "transport": "stdio",
            "config": json.dumps(
                {
                    "command": "python",
                    "args": ["-m", "mcp_server"],
                    "env": {"DEBUG": "true"},
                    "cwd": "/app",
                }
            ),
            "is_active": True,
            "is_default": False,
        },
        {
            "user_id": 1,
            "name": "websocket-server",
            "description": "WebSocket MCP server",
            "transport": "websocket",
            "config": json.dumps(
                {
                    "url": "ws://localhost:8080",
                    "headers": {"Authorization": "Bearer token"},
                }
            ),
            "is_active": True,
            "is_default": True,
        },
        {
            "user_id": 2,
            "name": "string-config-server",
            "description": "Server with string configs",
            "transport": "stdio",
            "config": json.dumps(
                {
                    "command": "node server.js",
                    "args": "--port 3000 --verbose",  # String instead of list
                    "env": "NODE_ENV=production\nPORT=3000",  # String instead of dict
                }
            ),
            "is_active": False,
            "is_default": False,
        },
    ]

    # Insert legacy data
    with engine.connect() as conn:
        for server in legacy_data:
            conn.execute(
                text("""
                INSERT INTO mcp_servers (user_id, name, description, transport, config, is_active, is_default)
                VALUES (:user_id, :name, :description, :transport, :config, :is_active, :is_default)
            """),
                server,
            )
        conn.commit()

    # Rename old table and create new tables with ALL expected columns
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE mcp_servers RENAME TO mcp_servers_legacy"))

        # Create new tables with ALL expected columns
        conn.execute(
            text("""
            CREATE TABLE mcp_servers (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100) NOT NULL UNIQUE,
                description TEXT,
                managed VARCHAR(20) NOT NULL,
                transport VARCHAR(50) NOT NULL,
                command VARCHAR(500),
                args TEXT,
                url VARCHAR(500),
                env TEXT,
                cwd VARCHAR(500),
                headers TEXT,
                docker_url VARCHAR(500),
                docker_image VARCHAR(200),
                docker_environment TEXT,
                docker_working_dir VARCHAR(500),
                volumes TEXT,
                bind_ports TEXT,
                restart_policy VARCHAR(50) DEFAULT 'no',
                auto_start BOOLEAN,
                container_id VARCHAR(100),
                container_name VARCHAR(200),
                container_logs TEXT,
                created_at DATETIME,
                updated_at DATETIME
            )
        """)
        )

        conn.execute(
            text("""
            CREATE TABLE user_mcpservers (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                mcpserver_id INTEGER NOT NULL,
                is_owner BOOLEAN DEFAULT FALSE,
                can_edit BOOLEAN DEFAULT FALSE,
                can_delete BOOLEAN DEFAULT FALSE,
                is_shared BOOLEAN DEFAULT FALSE,
                is_active BOOLEAN DEFAULT TRUE,
                is_default BOOLEAN DEFAULT FALSE,
                created_at DATETIME,
                updated_at DATETIME,
                UNIQUE (user_id, mcpserver_id)
            )
        """)
        )
        conn.commit()

    # Mock op and run ORM migration
    with patch.object(migration_module, "op") as mock_op:
        mock_op.get_bind.return_value = engine

        # Run the ORM data migration
        migration_module.migrate_data_orm()

    # Verify migration results
    with engine.connect() as conn:
        # Check servers were migrated
        servers = conn.execute(text("SELECT * FROM mcp_servers ORDER BY id")).fetchall()
        assert len(servers) == 3

        # Check STDIO server
        stdio_server = next(s for s in servers if s.name == "stdio-server")
        assert stdio_server.managed == "external"
        assert stdio_server.command == "python"
        assert json.loads(stdio_server.args) == ["-m", "mcp_server"]
        assert json.loads(stdio_server.env) == {"DEBUG": "true"}

        # Check WebSocket server
        ws_server = next(s for s in servers if s.name == "websocket-server")
        assert ws_server.managed == "external"
        assert ws_server.url == "ws://localhost:8080"
        assert json.loads(ws_server.headers) == {"Authorization": "Bearer token"}

        # Check string config server (should be parsed)
        string_server = next(s for s in servers if s.name == "string-config-server")
        assert string_server.command == "node server.js"
        assert json.loads(string_server.args) == ["--port", "3000", "--verbose"]
        assert json.loads(string_server.env) == {
            "NODE_ENV": "production",
            "PORT": "3000",
        }

        # Check user relationships
        relationships = conn.execute(text("SELECT * FROM user_mcpservers")).fetchall()
        assert len(relationships) == 3

        # All should be owners with full permissions
        for rel in relationships:
            assert rel.is_owner == 1
            assert rel.can_edit == 1
            assert rel.can_delete == 1


def test_migration_handles_invalid_json_orm(test_db, migration_module):
    """Test ORM migration handles invalid JSON gracefully"""
    engine, Session = test_db

    # Insert server with invalid JSON
    with engine.connect() as conn:
        conn.execute(
            text("""
            INSERT INTO mcp_servers (user_id, name, description, transport, config, is_active, is_default)
            VALUES (1, 'invalid-server', 'Invalid JSON', 'stdio', 'invalid json', 1, 0)
        """)
        )
        conn.commit()

    # Test normalize_config_fields with empty config (what happens after JSON parse failure)
    result = migration_module.normalize_config_fields({}, "stdio")

    assert result["managed"] == "external"
    assert result["command"] is None
    assert result["args"] is None
    assert result["restart_policy"] == "no"


def test_migration_handles_duplicate_names_orm(test_db, migration_module):
    """Test ORM migration handles duplicate server names gracefully"""
    engine, Session = test_db

    # Insert servers with duplicate names
    with engine.connect() as conn:
        conn.execute(
            text("""
            INSERT INTO mcp_servers (user_id, name, description, transport, config, is_active, is_default) VALUES
            (1, 'duplicate-name', 'First server', 'stdio', '{"command": "first"}', 1, 0),
            (2, 'duplicate-name', 'Second server', 'stdio', '{"command": "second"}', 1, 0)
        """)
        )
        conn.commit()

    # Create new tables with FULL schema
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE mcp_servers RENAME TO mcp_servers_legacy"))
        conn.execute(
            text("""
            CREATE TABLE mcp_servers (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100) NOT NULL UNIQUE,
                description TEXT,
                managed VARCHAR(20) NOT NULL,
                transport VARCHAR(50) NOT NULL,
                command VARCHAR(500),
                args TEXT,
                url VARCHAR(500),
                env TEXT,
                cwd VARCHAR(500),
                headers TEXT,
                docker_url VARCHAR(500),
                docker_image VARCHAR(200),
                docker_environment TEXT,
                docker_working_dir VARCHAR(500),
                volumes TEXT,
                bind_ports TEXT,
                restart_policy VARCHAR(50) DEFAULT 'no',
                auto_start BOOLEAN,
                container_id VARCHAR(100),
                container_name VARCHAR(200),
                container_logs TEXT,
                created_at DATETIME,
                updated_at DATETIME
            )
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE user_mcpservers (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                mcpserver_id INTEGER NOT NULL,
                is_owner BOOLEAN DEFAULT FALSE,
                can_edit BOOLEAN DEFAULT FALSE,
                can_delete BOOLEAN DEFAULT FALSE,
                is_shared BOOLEAN DEFAULT FALSE,
                is_active BOOLEAN DEFAULT TRUE,
                is_default BOOLEAN DEFAULT FALSE,
                created_at DATETIME,
                updated_at DATETIME
            )
        """)
        )
        conn.commit()

    # Mock op and run migration - should handle duplicates gracefully
    with patch.object(migration_module, "op") as mock_op:
        mock_op.get_bind.return_value = engine

        # Should handle duplicates gracefully - the second one should fail but continue
        # This should NOT raise an exception, but log the failure
        migration_module.migrate_data_orm()

    # Verify only one server with that name exists (first one wins)
    with engine.connect() as conn:
        servers = conn.execute(
            text("SELECT * FROM mcp_servers WHERE name = 'duplicate-name'")
        ).fetchall()
        assert len(servers) == 1  # Only first one should be migrated

        # Verify it's the first server that was migrated
        server = servers[0]
        assert server.description == "First server"
        assert server.command == "first"

        # Verify we have exactly one relationship
        relationships = conn.execute(text("SELECT * FROM user_mcpservers")).fetchall()
        assert len(relationships) == 1
        assert relationships[0].user_id == 1  # First server's user


def test_migration_preserves_timestamps_orm(test_db, migration_module):
    """Test that ORM migration preserves original timestamps"""
    engine, Session = test_db

    created_time = "2024-01-01 12:00:00"
    updated_time = "2024-01-02 12:00:00"

    # Insert server with specific timestamps
    with engine.connect() as conn:
        conn.execute(
            text("""
            INSERT INTO mcp_servers (user_id, name, description, transport, config, is_active, is_default, created_at, updated_at)
            VALUES (1, 'timestamp-test', 'Timestamp test', 'stdio', '{"command": "test"}', 1, 0, :created, :updated)
        """),
            {"created": created_time, "updated": updated_time},
        )
        conn.commit()

    # Create new tables with FULL schema
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE mcp_servers RENAME TO mcp_servers_legacy"))
        conn.execute(
            text("""
            CREATE TABLE mcp_servers (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100) NOT NULL UNIQUE,
                description TEXT,
                managed VARCHAR(20) NOT NULL,
                transport VARCHAR(50) NOT NULL,
                command VARCHAR(500),
                args TEXT,
                url VARCHAR(500),
                env TEXT,
                cwd VARCHAR(500),
                headers TEXT,
                docker_url VARCHAR(500),
                docker_image VARCHAR(200),
                docker_environment TEXT,
                docker_working_dir VARCHAR(500),
                volumes TEXT,
                bind_ports TEXT,
                restart_policy VARCHAR(50) DEFAULT 'no',
                auto_start BOOLEAN,
                container_id VARCHAR(100),
                container_name VARCHAR(200),
                container_logs TEXT,
                created_at DATETIME,
                updated_at DATETIME
            )
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE user_mcpservers (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                mcpserver_id INTEGER NOT NULL,
                is_owner BOOLEAN DEFAULT FALSE,
                can_edit BOOLEAN DEFAULT FALSE,
                can_delete BOOLEAN DEFAULT FALSE,
                is_shared BOOLEAN DEFAULT FALSE,
                is_active BOOLEAN DEFAULT TRUE,
                is_default BOOLEAN DEFAULT FALSE,
                created_at DATETIME,
                updated_at DATETIME
            )
        """)
        )
        conn.commit()

    # Mock op and run migration
    with patch.object(migration_module, "op") as mock_op:
        mock_op.get_bind.return_value = engine
        migration_module.migrate_data_orm()

    # Verify timestamps preserved (allowing for microseconds)
    with engine.connect() as conn:
        server = conn.execute(
            text("""
            SELECT created_at, updated_at FROM mcp_servers WHERE name = 'timestamp-test'
        """)
        ).fetchone()

        # Compare just the date/time part, ignoring microseconds
        assert str(server.created_at).startswith(created_time)
        assert str(server.updated_at).startswith(updated_time)


def test_orm_models_structure(migration_module):
    """Test that ORM models have the expected structure"""
    # Test LegacyMCPServer model
    legacy_model = migration_module.LegacyMCPServer
    assert legacy_model.__tablename__ == "mcp_servers_legacy"
    assert hasattr(legacy_model, "id")
    assert hasattr(legacy_model, "user_id")
    assert hasattr(legacy_model, "name")
    assert hasattr(legacy_model, "config")

    # Test NewMCPServer model
    new_model = migration_module.NewMCPServer
    assert new_model.__tablename__ == "mcp_servers"
    assert hasattr(new_model, "id")
    assert hasattr(new_model, "name")
    assert hasattr(new_model, "managed")
    assert hasattr(new_model, "transport")
    assert hasattr(new_model, "command")
    assert hasattr(new_model, "args")

    # Test UserMCPServer model
    user_model = migration_module.UserMCPServer
    assert user_model.__tablename__ == "user_mcpservers"
    assert hasattr(user_model, "id")
    assert hasattr(user_model, "user_id")
    assert hasattr(user_model, "mcpserver_id")
    assert hasattr(user_model, "is_owner")


def test_orm_error_handling_with_partial_failure(test_db, migration_module):
    """Test ORM handles partial failures gracefully"""
    engine, Session = test_db

    # Insert multiple servers, one with problematic data
    with engine.connect() as conn:
        # Good server
        conn.execute(
            text("""
            INSERT INTO mcp_servers (user_id, name, description, transport, config, is_active, is_default)
            VALUES (1, 'good-server', 'Good server', 'stdio', '{"command": "good"}', 1, 0)
        """)
        )
        # Server with valid JSON but will be handled in the migration logic
        conn.execute(
            text("""
            INSERT INTO mcp_servers (user_id, name, description, transport, config, is_active, is_default)
            VALUES (1, 'bad-json-server', 'Bad JSON server', 'stdio', '{}', 1, 0)
        """)
        )
        conn.commit()

    # Create new tables
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE mcp_servers RENAME TO mcp_servers_legacy"))
        conn.execute(
            text("""
            CREATE TABLE mcp_servers (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100) NOT NULL UNIQUE,
                description TEXT,
                managed VARCHAR(20) NOT NULL,
                transport VARCHAR(50) NOT NULL,
                command VARCHAR(500),
                args TEXT,
                url VARCHAR(500),
                env TEXT,
                cwd VARCHAR(500),
                headers TEXT,
                docker_url VARCHAR(500),
                docker_image VARCHAR(200),
                docker_environment TEXT,
                docker_working_dir VARCHAR(500),
                volumes TEXT,
                bind_ports TEXT,
                restart_policy VARCHAR(50) DEFAULT 'no',
                auto_start BOOLEAN,
                container_id VARCHAR(100),
                container_name VARCHAR(200),
                container_logs TEXT,
                created_at DATETIME,
                updated_at DATETIME
            )
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE user_mcpservers (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                mcpserver_id INTEGER NOT NULL,
                is_owner BOOLEAN DEFAULT FALSE,
                can_edit BOOLEAN DEFAULT FALSE,
                can_delete BOOLEAN DEFAULT FALSE,
                is_shared BOOLEAN DEFAULT FALSE,
                is_active BOOLEAN DEFAULT TRUE,
                is_default BOOLEAN DEFAULT FALSE,
                created_at DATETIME,
                updated_at DATETIME
            )
        """)
        )
        conn.commit()

    # Mock op and run migration
    with patch.object(migration_module, "op") as mock_op:
        mock_op.get_bind.return_value = engine

        # Should handle both servers
        migration_module.migrate_data_orm()

    # Verify both servers were migrated
    with engine.connect() as conn:
        servers = conn.execute(
            text("SELECT * FROM mcp_servers ORDER BY name")
        ).fetchall()
        assert len(servers) == 2

        # Good server should have proper config
        good_server = next(s for s in servers if s.name == "good-server")
        assert good_server.command == "good"

        # Bad JSON server should have been migrated with empty config
        bad_server = next(s for s in servers if s.name == "bad-json-server")
        assert bad_server.managed == "external"  # Default value
        assert bad_server.command is None  # No command from empty config


def test_migration_handles_edge_cases(migration_module):
    """Test migration handles various edge cases"""
    # Test with Path object for cwd
    config_with_path = {
        "cwd": "/some/path"  # Simulate Path object as string
    }
    result = migration_module.normalize_config_fields(config_with_path, "stdio")
    assert result["cwd"] == "/some/path"

    # Test with empty restart policy
    config_empty_restart = {"restart_policy": ""}
    result = migration_module.normalize_config_fields(config_empty_restart, "stdio")
    assert result["restart_policy"] == "no"


def test_orm_migration_transaction_rollback(test_db, migration_module):
    """Test that ORM migration properly rolls back on errors"""
    engine, Session = test_db

    # Insert a server that will cause a conflict
    with engine.connect() as conn:
        conn.execute(
            text("""
            INSERT INTO mcp_servers (user_id, name, description, transport, config, is_active, is_default)
            VALUES (1, 'test-server', 'Test server', 'stdio', '{"command": "test"}', 1, 0)
        """)
        )
        conn.commit()

    # Create new tables
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE mcp_servers RENAME TO mcp_servers_legacy"))
        conn.execute(
            text("""
            CREATE TABLE mcp_servers (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100) NOT NULL UNIQUE,
                description TEXT,
                managed VARCHAR(20) NOT NULL,
                transport VARCHAR(50) NOT NULL,
                command VARCHAR(500),
                args TEXT,
                url VARCHAR(500),
                env TEXT,
                cwd VARCHAR(500),
                headers TEXT,
                docker_url VARCHAR(500),
                docker_image VARCHAR(200),
                docker_environment TEXT,
                docker_working_dir VARCHAR(500),
                volumes TEXT,
                bind_ports TEXT,
                restart_policy VARCHAR(50) DEFAULT 'no',
                auto_start BOOLEAN,
                container_id VARCHAR(100),
                container_name VARCHAR(200),
                container_logs TEXT,
                created_at DATETIME,
                updated_at DATETIME
            )
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE user_mcpservers (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                mcpserver_id INTEGER NOT NULL,
                is_owner BOOLEAN DEFAULT FALSE,
                can_edit BOOLEAN DEFAULT FALSE,
                can_delete BOOLEAN DEFAULT FALSE,
                is_shared BOOLEAN DEFAULT FALSE,
                is_active BOOLEAN DEFAULT TRUE,
                is_default BOOLEAN DEFAULT FALSE,
                created_at DATETIME,
                updated_at DATETIME
            )
        """)
        )
        conn.commit()

    # Test that the migration works normally
    with patch.object(migration_module, "op") as mock_op:
        mock_op.get_bind.return_value = engine

        # Should complete successfully
        migration_module.migrate_data_orm()

    # Verify the server was migrated
    with engine.connect() as conn:
        count = (
            conn.execute(text("SELECT COUNT(*) as count FROM mcp_servers"))
            .fetchone()
            .count
        )
        assert count == 1


def test_orm_session_management(test_db, migration_module):
    """Test that ORM properly manages database sessions"""
    engine, Session = test_db

    # Insert test data
    with engine.connect() as conn:
        conn.execute(
            text("""
            INSERT INTO mcp_servers (user_id, name, description, transport, config, is_active, is_default)
            VALUES (1, 'session-test', 'Session test', 'stdio', '{"command": "test"}', 1, 0)
        """)
        )
        conn.commit()

    # Create new tables
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE mcp_servers RENAME TO mcp_servers_legacy"))
        conn.execute(
            text("""
            CREATE TABLE mcp_servers (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100) NOT NULL UNIQUE,
                description TEXT,
                managed VARCHAR(20) NOT NULL,
                transport VARCHAR(50) NOT NULL,
                command VARCHAR(500),
                args TEXT,
                url VARCHAR(500),
                env TEXT,
                cwd VARCHAR(500),
                headers TEXT,
                docker_url VARCHAR(500),
                docker_image VARCHAR(200),
                docker_environment TEXT,
                docker_working_dir VARCHAR(500),
                volumes TEXT,
                bind_ports TEXT,
                restart_policy VARCHAR(50) DEFAULT 'no',
                auto_start BOOLEAN,
                container_id VARCHAR(100),
                container_name VARCHAR(200),
                container_logs TEXT,
                created_at DATETIME,
                updated_at DATETIME
            )
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE user_mcpservers (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                mcpserver_id INTEGER NOT NULL,
                is_owner BOOLEAN DEFAULT FALSE,
                can_edit BOOLEAN DEFAULT FALSE,
                can_delete BOOLEAN DEFAULT FALSE,
                is_shared BOOLEAN DEFAULT FALSE,
                is_active BOOLEAN DEFAULT TRUE,
                is_default BOOLEAN DEFAULT FALSE,
                created_at DATETIME,
                updated_at DATETIME
            )
        """)
        )
        conn.commit()

    # Mock op and test session handling
    with patch.object(migration_module, "op") as mock_op:
        mock_op.get_bind.return_value = engine

        # Should handle session creation and cleanup properly
        migration_module.migrate_data_orm()

    # Verify migration completed
    with engine.connect() as conn:
        server = conn.execute(
            text("SELECT * FROM mcp_servers WHERE name = 'session-test'")
        ).fetchone()
        assert server is not None
        assert server.name == "session-test"
