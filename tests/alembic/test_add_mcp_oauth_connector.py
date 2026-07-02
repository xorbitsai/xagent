"""Tests for the MCP OAuth connector migration."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/5bb3df522a7d_add_mcp_oauth_connector.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mcp_oauth_connector_migration", migration_file
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    context = MigrationContext.configure(connection)
    return Operations(context)


def _create_base_schema(connection):
    connection.execute(
        text(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                username VARCHAR(50) NOT NULL,
                email VARCHAR(100) NOT NULL
            )
            """
        )
    )
    connection.execute(
        text(
            """
            CREATE TABLE mcp_servers (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                managed VARCHAR(20) NOT NULL,
                transport VARCHAR(50) NOT NULL
            )
            """
        )
    )


def test_upgrade_creates_table_and_adds_columns(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()

    with engine.begin() as connection:
        _create_base_schema(connection)

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        inspector = inspect(connection)

        mcp_server_columns = {
            column["name"] for column in inspector.get_columns("mcp_servers")
        }
        assert "oauth_client" in mcp_server_columns
        assert "auth_server_metadata" in mcp_server_columns

        assert "mcp_user_oauth_tokens" in inspector.get_table_names()

        token_columns = {
            column["name"]
            for column in inspector.get_columns("mcp_user_oauth_tokens")
        }
        assert token_columns == {
            "id",
            "user_id",
            "mcpserver_id",
            "access_token",
            "refresh_token",
            "expires_at",
            "token_type",
            "scope",
            "status",
            "pkce_verifier",
            "state",
            "created_at",
            "updated_at",
        }

        index_names = {
            index["name"] for index in inspector.get_indexes("mcp_user_oauth_tokens")
        }
        assert "ix_mcp_user_oauth_tokens_user_id" in index_names
        assert "ix_mcp_user_oauth_tokens_mcpserver_id" in index_names
        assert "ix_mcp_user_oauth_tokens_state" in index_names


def test_downgrade_removes_table_and_columns(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()

    with engine.begin() as connection:
        _create_base_schema(connection)

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()

        inspector = inspect(connection)

        assert "mcp_user_oauth_tokens" not in inspector.get_table_names()

        mcp_server_columns = {
            column["name"] for column in inspector.get_columns("mcp_servers")
        }
        assert "oauth_client" not in mcp_server_columns
        assert "auth_server_metadata" not in mcp_server_columns
