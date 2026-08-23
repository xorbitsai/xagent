"""Tests for the users.preferences column migration."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260823_add_preferences_to_users.py"
    )
    spec = importlib.util.spec_from_file_location(
        "user_preferences_migration", migration_file
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


def _create_users_table(connection, with_preferences: bool) -> None:
    preferences_col = ",\n                preferences JSON" if with_preferences else ""
    connection.execute(
        text(
            f"""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                username VARCHAR(50) NOT NULL{preferences_col}
            )
            """
        )
    )


def _columns(connection):
    return {c["name"] for c in inspect(connection).get_columns("users")}


def test_upgrade_adds_preferences_column(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_users_table(connection, with_preferences=False)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert "preferences" in _columns(connection)


def test_upgrade_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_users_table(connection, with_preferences=True)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # preferences already present -> no-op, must not raise
        assert "preferences" in _columns(connection)


def test_upgrade_is_a_no_op_when_users_table_is_absent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # no users table yet -> must not raise


def test_downgrade_removes_preferences_column(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_users_table(connection, with_preferences=True)
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        assert "preferences" not in _columns(connection)


def test_downgrade_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_users_table(connection, with_preferences=False)
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()  # already absent -> no-op, must not raise
        assert "preferences" not in _columns(connection)
