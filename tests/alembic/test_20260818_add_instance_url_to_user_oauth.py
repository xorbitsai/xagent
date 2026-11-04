"""Tests for the user_oauth.instance_url column migration."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260818_add_instance_url_to_user_oauth.py"
    )
    spec = importlib.util.spec_from_file_location(
        "add_instance_url_migration", migration_file
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


def _create_table(connection):
    connection.execute(
        text(
            """
            CREATE TABLE user_oauth (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                provider VARCHAR(50) NOT NULL,
                access_token TEXT NOT NULL,
                refresh_token TEXT,
                expires_at DATETIME,
                token_type VARCHAR(50),
                scope TEXT,
                provider_user_id TEXT,
                email TEXT
            )
            """
        )
    )


def _columns(connection):
    return {
        row[1]
        for row in connection.execute(text("PRAGMA table_info(user_oauth)")).fetchall()
    }


def test_upgrade_adds_instance_url_column(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert "instance_url" in _columns(connection)

        connection.execute(
            text(
                "INSERT INTO user_oauth"
                " (user_id, provider, access_token, instance_url)"
                " VALUES (1, 'salesforce', 'tok', 'https://acme.my.salesforce.com')"
            )
        )
        row = connection.execute(
            text("SELECT instance_url FROM user_oauth WHERE provider='salesforce'")
        ).first()
        assert row[0] == "https://acme.my.salesforce.com"


def test_upgrade_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()  # second run must not raise or duplicate the column
        assert "instance_url" in _columns(connection)


def test_downgrade_removes_instance_url_column(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert "instance_url" not in _columns(connection)


def test_upgrade_and_downgrade_no_op_without_table(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        table_names = set(
            connection.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).scalars()
        )
        assert "user_oauth" not in table_names
