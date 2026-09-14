"""Tests for the Microsoft Graph MCP registry seed migration."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260526_seed_builtin_microsoft_graph_mcp_apps.py"
    )
    spec = importlib.util.spec_from_file_location(
        "seed_microsoft_graph_migration", migration_file
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_normalize_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260715_normalize_builtin_mcp_launch.py"
    )
    spec = importlib.util.spec_from_file_location(
        "normalize_builtin_mcp_launch_migration", migration_file
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


def test_upgrade_inserts_microsoft_provider_and_apps(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert "microsoft" in _provider_names(connection)
        assert {"teams", "outlook", "onedrive"}.issubset(_app_ids(connection))

        teams_row = connection.execute(
            text(
                "SELECT transport, provider_name, launch_config FROM public_mcp_apps"
                " WHERE app_id='teams'"
            )
        ).first()
        assert teams_row[0] == "oauth"
        assert teams_row[1] == "microsoft"
        assert "xagent.web.tools.mcp.teams" in str(teams_row[2])


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
                " WHERE app_id IN ('teams', 'outlook', 'onedrive')"
            )
        ).scalar()
        assert app_count == 3
        provider_count = connection.execute(
            text("SELECT COUNT(*) FROM oauth_providers WHERE provider_name='microsoft'")
        ).scalar()
        assert provider_count == 1


def test_seed_rows_match_registry(tmp_path):
    """The migration snapshot and the runtime registry must define the same
    rows. Drift here (e.g. a later normalization migration rewriting these
    apps' launch_config) means a downgrade chain that passes through the
    normalization migration would see the seeded rows as "modified" and
    incorrectly preserve them -- see
    test_downgrade_cleans_up_after_descendant_normalization_migration."""
    from xagent.web.builtin_mcp_registry import (
        get_builtin_oauth_provider_rows,
        get_builtin_public_mcp_app_rows,
    )

    migration = _load_migration_module()

    registry_apps = {
        row["app_id"]: row
        for row in get_builtin_public_mcp_app_rows()
        if row["app_id"] in {"teams", "outlook", "onedrive"}
    }
    migration_apps = {row["app_id"]: row for row in migration._microsoft_app_rows()}
    assert migration_apps == registry_apps

    registry_provider = next(
        row
        for row in get_builtin_oauth_provider_rows()
        if row["provider_name"] == "microsoft"
    )
    assert migration._microsoft_provider_row() == registry_provider


def test_downgrade_cleans_up_after_descendant_normalization_migration(tmp_path):
    """20260715_normalize_builtin_mcp_launch runs after this migration and
    unconditionally rewrites teams/outlook/onedrive's launch_config (its own
    downgrade is a deliberate no-op). A downgrade chain that passes through
    it and then back through this migration must still remove the seeded
    apps and provider, not preserve them as if an operator had edited them."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    normalize_migration = _load_normalize_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        with patch.object(normalize_migration, "op", _operations(connection)):
            normalize_migration.upgrade()
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        assert not {"teams", "outlook", "onedrive"} & _app_ids(connection)
        assert "microsoft" not in _provider_names(connection)


def test_downgrade_removes_provider_and_apps(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert not {"teams", "outlook", "onedrive"} & _app_ids(connection)
        assert "microsoft" not in _provider_names(connection)


def test_downgrade_preserves_colliding_custom_app(tmp_path):
    """An operator's custom app that reuses one of this migration's app_ids
    (e.g. a hand-created "teams" connector with a different config) must
    survive downgrade, since upgrade() itself no-ops on that collision."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps (app_id, name, transport, provider_name)"
                " VALUES ('teams', 'Custom Teams Bridge', 'stdio', NULL)"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert "teams" in _app_ids(connection)
        row = connection.execute(
            text("SELECT name, transport FROM public_mcp_apps WHERE app_id='teams'")
        ).first()
        assert row[0] == "Custom Teams Bridge"
        assert row[1] == "stdio"
        assert not {"outlook", "onedrive"} & _app_ids(connection)


def test_downgrade_preserves_admin_created_microsoft_provider(tmp_path):
    """A pre-existing admin-created "microsoft" provider (different shape
    than the seeded row) must survive downgrade even when no microsoft apps
    remain."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        connection.execute(
            text(
                "INSERT INTO oauth_providers"
                " (provider_name, name, client_id, client_secret, auth_url, token_url)"
                " VALUES ('microsoft', 'Custom Microsoft', 'cid', 'secret',"
                " 'https://custom.example.com/authorize',"
                " 'https://custom.example.com/token')"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert not {"teams", "outlook", "onedrive"} & _app_ids(connection)
        assert "microsoft" in _provider_names(connection)


def test_downgrade_keeps_provider_when_custom_microsoft_app_exists(tmp_path):
    """The shared "microsoft" oauth_providers row must survive downgrade if a
    non-seeded microsoft app is still present."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            connection.execute(
                text(
                    "INSERT INTO public_mcp_apps (app_id, name, transport, provider_name)"
                    " VALUES ('custom-teams', 'Custom Teams', 'oauth', 'microsoft')"
                )
            )
            migration.downgrade()
        assert "microsoft" in _provider_names(connection)


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
