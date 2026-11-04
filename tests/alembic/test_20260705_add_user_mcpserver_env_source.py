"""Tests for the user_mcpservers.env_source migration."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260705_add_user_mcpserver_env_source.py"
    )
    spec = importlib.util.spec_from_file_location(
        "user_mcpserver_env_source_migration", migration_file
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


def _create_table(connection, with_source: bool):
    source_col = ",\n                env_source VARCHAR(16)" if with_source else ""
    connection.execute(
        text(
            f"""
            CREATE TABLE user_mcpservers (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                mcpserver_id INTEGER NOT NULL,
                is_active BOOLEAN NOT NULL DEFAULT 1,
                env JSON{source_col}
            )
            """
        )
    )


def _columns(connection):
    return {c["name"] for c in inspect(connection).get_columns("user_mcpservers")}


def test_upgrade_adds_env_source_column(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, with_source=False)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert "env_source" in _columns(connection)


def test_downgrade_removes_env_source_column(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, with_source=True)
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        assert "env_source" not in _columns(connection)
