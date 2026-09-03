"""Tests for the Employment Hero MCP connector seed migration."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260826_seed_employment_hero_mcp_app.py"
    )
    spec = importlib.util.spec_from_file_location(
        "seed_employment_hero_migration", migration_file
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
        assert "employment-hero" in _provider_names(connection)
        assert "employment-hero" in _app_ids(connection)

        row = connection.execute(
            text(
                "SELECT transport, provider_name, launch_config FROM public_mcp_apps"
                " WHERE app_id='employment-hero'"
            )
        ).first()
        assert row[0] == "oauth"
        assert row[1] == "employment-hero"
        assert "xagent.web.tools.mcp.employment_hero" in str(row[2])
        assert "EMPLOYMENT_HERO_ACCESS_TOKEN" in str(row[2])


def test_upgrade_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()  # second run must not raise or duplicate
        app_count = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='employment-hero'")
        ).scalar()
        assert app_count == 1
        provider_count = connection.execute(
            text(
                "SELECT COUNT(*) FROM oauth_providers"
                " WHERE provider_name='employment-hero'"
            )
        ).scalar()
        assert provider_count == 1


def test_seed_rows_match_registry(tmp_path):
    """The migration snapshot and the runtime registry must define the same
    employment-hero rows (the migration is a frozen copy; this catches drift)."""
    from xagent.web.builtin_mcp_registry import (
        get_builtin_oauth_provider_rows,
        get_builtin_public_mcp_app_rows,
    )

    migration = _load_migration_module()

    registry_app = next(
        row
        for row in get_builtin_public_mcp_app_rows()
        if row["app_id"] == "employment-hero"
    )
    assert migration._employment_hero_app_row() == registry_app

    registry_provider = next(
        row
        for row in get_builtin_oauth_provider_rows()
        if row["provider_name"] == "employment-hero"
    )
    assert migration._employment_hero_provider_row() == registry_provider


def test_downgrade_removes_provider_and_app(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert "employment-hero" not in _app_ids(connection)
        assert "employment-hero" not in _provider_names(connection)


def test_downgrade_keeps_provider_when_custom_employment_hero_app_exists(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            connection.execute(
                text(
                    "INSERT INTO public_mcp_apps (app_id, name, transport, provider_name)"
                    " VALUES ('custom-employment-hero', 'Custom Employment Hero',"
                    " 'oauth', 'employment-hero')"
                )
            )
            migration.downgrade()
        assert "employment-hero" in _provider_names(connection)


def test_downgrade_preserves_admin_created_employment_hero_provider(tmp_path):
    """A pre-existing admin-created "employment-hero" provider (different
    shape than the seeded row) must survive downgrade even when no
    employment-hero apps remain."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        connection.execute(
            text(
                "INSERT INTO oauth_providers"
                " (provider_name, name, client_id, client_secret, auth_url, token_url)"
                " VALUES ('employment-hero', 'Custom Employment Hero', 'cid', 'secret',"
                " 'https://custom.example.com/authorize',"
                " 'https://custom.example.com/token')"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert "employment-hero" not in _app_ids(connection)
        assert "employment-hero" in _provider_names(connection)


def test_downgrade_preserves_pre_existing_employment_hero_app(tmp_path):
    """A pre-existing "employment-hero" app row (different shape than the
    seeded one) must survive downgrade -- upgrade()'s own `app_id not in
    existing_app_ids` check skipped inserting over it, so it was never
    "this migration's row" to remove. Deleting it unconditionally would
    also make the remaining-employment-hero-apps count wrongly read as
    zero, letting the oauth_providers row underneath it be deleted too."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps (app_id, name, transport, provider_name)"
                " VALUES ('employment-hero', 'Custom Employment Hero App', 'oauth',"
                " 'employment-hero')"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert "employment-hero" in _app_ids(connection)
        assert "employment-hero" in _provider_names(connection)


def test_downgrade_preserves_app_row_admin_edited_beyond_structural_fields(tmp_path):
    """An admin who PATCHed the seeded app row's oauth_scopes without
    touching app_id/name/transport/provider_name must not have that edit
    silently discarded by downgrade -- matching only those four structural
    columns isn't enough to prove this is still "this migration's row"
    once anything else about it has been customized."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        connection.execute(
            text(
                "UPDATE public_mcp_apps SET oauth_scopes = '[\"custom_scope\"]'"
                " WHERE app_id = 'employment-hero'"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        assert "employment-hero" in _app_ids(connection)


def test_downgrade_preserves_provider_row_admin_edited_beyond_structural_fields(
    tmp_path,
):
    """An admin who edited the seeded provider row's default_scopes without
    touching provider_name/name/auth_url/token_url must not have that edit
    silently discarded by downgrade -- the app row is removed as usual (it
    still matches the seeded shape exactly), but the provider row it
    depends on must survive since its own shape no longer matches."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        connection.execute(
            text(
                "UPDATE oauth_providers SET default_scopes = '[\"custom_scope\"]'"
                " WHERE provider_name = 'employment-hero'"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        assert "employment-hero" not in _app_ids(connection)
        assert "employment-hero" in _provider_names(connection)


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
