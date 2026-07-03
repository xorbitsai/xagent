"""Tests for the per-user MCP env override migration."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260703_add_user_mcpserver_env.py"
    )
    spec = importlib.util.spec_from_file_location(
        "user_mcpserver_env_migration", migration_file
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


def _create_table(connection, with_env: bool):
    env_col = ",\n                    env JSON" if with_env else ""
    connection.execute(
        text(
            f"""
            CREATE TABLE user_mcpservers (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                mcpserver_id INTEGER NOT NULL,
                is_owner BOOLEAN NOT NULL DEFAULT 0,
                can_edit BOOLEAN NOT NULL DEFAULT 0,
                can_delete BOOLEAN NOT NULL DEFAULT 0,
                is_active BOOLEAN NOT NULL DEFAULT 1{env_col}
            )
            """
        )
    )


def _columns(connection):
    return {c["name"] for c in inspect(connection).get_columns("user_mcpservers")}


def test_upgrade_adds_env_column(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, with_env=False)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert "env" in _columns(connection)


def test_upgrade_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, with_env=True)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # env already present -> no-op, must not raise
        assert "env" in _columns(connection)


def test_upgrade_backfills_ownership(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, with_env=False)
        # Regressed owner row (is_owner=0) and an already-correct owner row.
        connection.execute(
            text(
                "INSERT INTO user_mcpservers "
                "(id, user_id, mcpserver_id, is_owner, can_edit, can_delete) VALUES "
                "(1, 10, 100, 0, 0, 0), (2, 20, 200, 1, 1, 1)"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        rows = connection.execute(
            text(
                "SELECT id, is_owner, can_edit, can_delete FROM user_mcpservers ORDER BY id"
            )
        ).fetchall()
        # Both rows end up as owners with edit/delete rights.
        assert rows[0] == (1, 1, 1, 1)
        assert rows[1] == (2, 1, 1, 1)


def test_downgrade_removes_env_column(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, with_env=True)
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        assert "env" not in _columns(connection)
