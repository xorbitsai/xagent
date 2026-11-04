"""Tests for the MCP concurrency config migration."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260624_add_mcp_concurrency_config.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mcp_concurrency_migration", migration_file
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    context = MigrationContext.configure(connection)
    return Operations(context)


def test_upgrade_adds_mcp_concurrency_columns(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()

    with engine.begin() as connection:
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

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        columns = {
            column["name"] for column in inspect(connection).get_columns("mcp_servers")
        }
        assert "concurrency_safe" in columns
        assert "concurrent_tools" in columns


def test_downgrade_removes_mcp_concurrency_columns(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE mcp_servers (
                    id INTEGER PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    managed VARCHAR(20) NOT NULL,
                    transport VARCHAR(50) NOT NULL,
                    concurrency_safe BOOLEAN NOT NULL DEFAULT 0,
                    concurrent_tools JSON
                )
                """
            )
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()

        columns = {
            column["name"] for column in inspect(connection).get_columns("mcp_servers")
        }
        assert "concurrency_safe" not in columns
        assert "concurrent_tools" not in columns
