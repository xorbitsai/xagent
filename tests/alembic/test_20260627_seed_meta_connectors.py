"""Tests for the Meta connector registry seed migration."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260627_seed_meta_connectors.py"
    )
    spec = importlib.util.spec_from_file_location(
        "seed_meta_connectors_migration", migration_file
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    context = MigrationContext.configure(connection)
    return Operations(context)


def _create_tables(connection):
    connection.execute(
        text(
            """
            CREATE TABLE oauth_providers (
                id INTEGER PRIMARY KEY,
                provider_name VARCHAR(50) UNIQUE NOT NULL,
                name VARCHAR(100) NOT NULL,
                client_id VARCHAR(500) NOT NULL,
                client_secret VARCHAR(500) NOT NULL,
                auth_url VARCHAR(1000) NOT NULL,
                token_url VARCHAR(1000) NOT NULL,
                redirect_uri VARCHAR(1000),
                userinfo_url VARCHAR(1000),
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
                app_id VARCHAR(100) UNIQUE NOT NULL,
                name VARCHAR(200) NOT NULL,
                description TEXT,
                icon VARCHAR(1000),
                transport VARCHAR(50) NOT NULL,
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


def test_upgrade_inserts_meta_provider_and_public_apps(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()

    with engine.begin() as connection:
        _create_tables(connection)

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()

        providers = connection.execute(
            text("SELECT provider_name, name FROM oauth_providers")
        ).fetchall()
        apps = connection.execute(
            text("SELECT app_id, provider_name, category FROM public_mcp_apps")
        ).fetchall()

    assert providers == [("meta", "Meta")]
    assert {
        (row.app_id, row.provider_name, row.category)
        for row in apps
        if row.app_id in {"facebook", "instagram"}
    } == {
        ("facebook", "meta", "Marketing"),
        ("instagram", "meta", "Marketing"),
    }


def test_downgrade_removes_provider_and_apps(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert not {"facebook", "instagram"} & _app_ids(connection)
        assert "meta" not in _provider_names(connection)


def test_downgrade_preserves_colliding_custom_app(tmp_path):
    """An operator's custom app that reuses one of this migration's app_ids
    (e.g. a hand-created "facebook" connector with a different config) must
    survive downgrade, since upgrade() itself no-ops on that collision."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps (app_id, name, transport, provider_name)"
                " VALUES ('facebook', 'Custom Facebook Bridge', 'stdio', NULL)"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert "facebook" in _app_ids(connection)
        row = connection.execute(
            text("SELECT name, transport FROM public_mcp_apps WHERE app_id='facebook'")
        ).first()
        assert row[0] == "Custom Facebook Bridge"
        assert row[1] == "stdio"
        assert "instagram" not in _app_ids(connection)
