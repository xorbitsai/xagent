"""Tests for the Google Docs, Google Slides, and HubSpot CRM MCP connector seed migration."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260720_seed_docs_slides_hubspot.py"
    )
    spec = importlib.util.spec_from_file_location(
        "seed_docs_slides_hubspot_migration", migration_file
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


def test_upgrade_inserts_provider_and_apps(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert "hubspot" in _provider_names(connection)
        assert {"google-docs", "google-slides", "hubspot"}.issubset(
            _app_ids(connection)
        )

        docs_row = connection.execute(
            text(
                "SELECT transport, provider_name, launch_config FROM public_mcp_apps"
                " WHERE app_id='google-docs'"
            )
        ).first()
        assert docs_row[0] == "oauth"
        assert docs_row[1] == "google"
        assert "xagent.web.tools.mcp.google_docs" in str(docs_row[2])

        slides_row = connection.execute(
            text(
                "SELECT transport, provider_name, launch_config FROM public_mcp_apps"
                " WHERE app_id='google-slides'"
            )
        ).first()
        assert slides_row[0] == "oauth"
        assert slides_row[1] == "google"
        assert "xagent.web.tools.mcp.google_slides" in str(slides_row[2])

        hubspot_row = connection.execute(
            text(
                "SELECT transport, provider_name, launch_config FROM public_mcp_apps"
                " WHERE app_id='hubspot'"
            )
        ).first()
        assert hubspot_row[0] == "oauth"
        assert hubspot_row[1] == "hubspot"
        assert "xagent.web.tools.mcp.hubspot" in str(hubspot_row[2])


def test_upgrade_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()  # second run must not raise or duplicate
        app_count = connection.execute(
            text(
                "SELECT COUNT(*) FROM public_mcp_apps"
                " WHERE app_id IN ('google-docs', 'google-slides', 'hubspot')"
            )
        ).scalar()
        assert app_count == 3
        provider_count = connection.execute(
            text("SELECT COUNT(*) FROM oauth_providers WHERE provider_name='hubspot'")
        ).scalar()
        assert provider_count == 1


def test_seed_rows_match_registry(tmp_path):
    """The migration snapshot and the runtime registry must define the same
    rows (the migration is a frozen copy; this catches drift)."""
    from xagent.web.builtin_mcp_registry import (
        get_builtin_oauth_provider_rows,
        get_builtin_public_mcp_app_rows,
    )

    migration = _load_migration_module()

    registry_apps = {
        row["app_id"]: row
        for row in get_builtin_public_mcp_app_rows()
        if row["app_id"] in migration.NEW_APP_IDS
    }
    migration_apps = {row["app_id"]: row for row in migration._new_app_rows()}
    assert migration_apps == registry_apps

    registry_provider = next(
        row
        for row in get_builtin_oauth_provider_rows()
        if row["provider_name"] == "hubspot"
    )
    assert migration._hubspot_provider_row() == registry_provider


def test_downgrade_removes_provider_and_apps(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert not {"google-docs", "google-slides", "hubspot"} & _app_ids(connection)
        assert "hubspot" not in _provider_names(connection)


def test_downgrade_keeps_provider_when_custom_hubspot_app_exists(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            connection.execute(
                text(
                    "INSERT INTO public_mcp_apps (app_id, name, transport, provider_name)"
                    " VALUES ('custom-hubspot', 'Custom HubSpot', 'oauth', 'hubspot')"
                )
            )
            migration.downgrade()
        assert "hubspot" in _provider_names(connection)


def test_downgrade_preserves_admin_created_hubspot_provider(tmp_path):
    """A pre-existing admin-created "hubspot" provider (different shape than the
    seeded row) must survive downgrade even when no hubspot apps remain."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        connection.execute(
            text(
                "INSERT INTO oauth_providers"
                " (provider_name, name, client_id, client_secret, auth_url, token_url)"
                " VALUES ('hubspot', 'Custom HubSpot', 'cid', 'secret',"
                " 'https://custom.example.com/authorize',"
                " 'https://custom.example.com/token')"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert not {"google-docs", "google-slides", "hubspot"} & _app_ids(connection)
        assert "hubspot" in _provider_names(connection)


def test_upgrade_and_downgrade_with_reduced_column_schema(tmp_path):
    """Exercise _filter_row's column dropping against tables missing optional
    columns."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
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
                    token_url VARCHAR(500) NOT NULL
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
                    transport VARCHAR(50) NOT NULL DEFAULT 'oauth',
                    provider_name VARCHAR(50)
                )
                """
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            assert "hubspot" in _provider_names(connection)
            assert {"google-docs", "google-slides", "hubspot"}.issubset(
                _app_ids(connection)
            )
            migration.downgrade()
        assert not {"google-docs", "google-slides", "hubspot"} & _app_ids(connection)
        assert "hubspot" not in _provider_names(connection)


def test_downgrade_guard_falls_back_when_guard_columns_absent(tmp_path):
    """The downgrade provenance guard skips comparisons for auth_url/token_url
    when those columns do not exist, still deleting the seeded provider by the
    remaining name match."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE oauth_providers (
                    id INTEGER PRIMARY KEY,
                    provider_name VARCHAR(50) NOT NULL UNIQUE,
                    name VARCHAR(100) NOT NULL,
                    client_id VARCHAR(500) NOT NULL,
                    client_secret VARCHAR(500) NOT NULL
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
                    transport VARCHAR(50) NOT NULL DEFAULT 'oauth',
                    provider_name VARCHAR(50)
                )
                """
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            assert "hubspot" in _provider_names(connection)
            migration.downgrade()
        assert "hubspot" not in _provider_names(connection)


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
