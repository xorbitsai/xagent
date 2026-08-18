"""Tests for the Linear MCP connector seed migration."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260818_seed_linear_mcp_app.py"
    )
    spec = importlib.util.spec_from_file_location(
        "seed_linear_migration", migration_file
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


def _create_tables(connection):
    connection.execute(
        text(
            """
            CREATE TABLE oauth_providers (
                id INTEGER PRIMARY KEY,
                provider_name VARCHAR(50) NOT NULL UNIQUE,
                name VARCHAR(100) NOT NULL,
                client_id VARCHAR(500) NOT NULL,
                client_secret VARCHAR(500) NOT NULL,
                auth_url VARCHAR(500) NOT NULL,
                token_url VARCHAR(500) NOT NULL,
                redirect_uri VARCHAR(500),
                userinfo_url VARCHAR(500),
                user_id_path VARCHAR(100),
                email_path VARCHAR(100),
                default_scopes JSON
            )
            """
        )
    )
    connection.execute(
        text(
            """
            CREATE TABLE public_mcp_apps (
                id INTEGER PRIMARY KEY,
                app_id VARCHAR(100) NOT NULL UNIQUE,
                name VARCHAR(200) NOT NULL,
                description TEXT,
                icon VARCHAR(1000),
                transport VARCHAR(50) NOT NULL DEFAULT 'oauth',
                provider_name VARCHAR(50),
                category VARCHAR(100),
                oauth_scopes JSON,
                is_visible_in_connector BOOLEAN NOT NULL DEFAULT 1,
                launch_config JSON
            )
            """
        )
    )


def _app_ids(connection):
    return set(connection.execute(text("SELECT app_id FROM public_mcp_apps")).scalars())


def _provider_names(connection):
    return set(
        connection.execute(text("SELECT provider_name FROM oauth_providers")).scalars()
    )


def test_upgrade_inserts_provider_and_app(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert "linear" in _provider_names(connection)
        assert "linear" in _app_ids(connection)

        row = connection.execute(
            text(
                "SELECT transport, provider_name, launch_config FROM public_mcp_apps"
                " WHERE app_id='linear'"
            )
        ).first()
        assert row[0] == "oauth"
        assert row[1] == "linear"
        assert "xagent.web.tools.mcp.linear" in str(row[2])
        assert "LINEAR_ACCESS_TOKEN" in str(row[2])


def test_upgrade_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()  # second run must not raise or duplicate
        app_count = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='linear'")
        ).scalar()
        assert app_count == 1
        provider_count = connection.execute(
            text("SELECT COUNT(*) FROM oauth_providers WHERE provider_name='linear'")
        ).scalar()
        assert provider_count == 1


def test_seed_rows_match_registry(tmp_path):
    """The migration snapshot and the runtime registry must define the same
    linear rows (the migration is a frozen copy; this catches drift)."""
    from xagent.web.builtin_mcp_registry import (
        get_builtin_oauth_provider_rows,
        get_builtin_public_mcp_app_rows,
    )

    migration = _load_migration_module()

    registry_app = next(
        row for row in get_builtin_public_mcp_app_rows() if row["app_id"] == "linear"
    )
    assert migration._linear_app_row() == registry_app

    registry_provider = next(
        row
        for row in get_builtin_oauth_provider_rows()
        if row["provider_name"] == "linear"
    )
    assert migration._linear_provider_row() == registry_provider


def test_downgrade_removes_provider_and_app(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert "linear" not in _app_ids(connection)
        assert "linear" not in _provider_names(connection)


def test_downgrade_keeps_provider_when_custom_linear_app_exists(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            connection.execute(
                text(
                    "INSERT INTO public_mcp_apps (app_id, name, transport, provider_name)"
                    " VALUES ('custom-linear', 'Custom Linear', 'oauth', 'linear')"
                )
            )
            migration.downgrade()
        assert "linear" in _provider_names(connection)


def test_downgrade_preserves_admin_created_linear_provider(tmp_path):
    """A pre-existing admin-created "linear" provider (different shape than
    the seeded row) must survive downgrade even when no linear apps remain."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        connection.execute(
            text(
                "INSERT INTO oauth_providers"
                " (provider_name, name, client_id, client_secret, auth_url, token_url)"
                " VALUES ('linear', 'Custom Linear', 'cid', 'secret',"
                " 'https://custom.example.com/authorize',"
                " 'https://custom.example.com/token')"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert "linear" not in _app_ids(connection)
        assert "linear" in _provider_names(connection)


def test_upgrade_and_downgrade_no_op_without_tables(tmp_path):
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
        assert "oauth_providers" not in table_names
        assert "public_mcp_apps" not in table_names
