"""Tests for the MYOB MCP connector seed migration."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260903_seed_myob_mcp_app.py"
    )
    spec = importlib.util.spec_from_file_location("seed_myob_migration", migration_file)
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
        assert "myob" in _provider_names(connection)
        assert "myob" in _app_ids(connection)

        provider_row = connection.execute(
            text(
                "SELECT auth_url, token_url, default_scopes FROM oauth_providers"
                " WHERE provider_name='myob'"
            )
        ).first()
        assert provider_row[0] == "https://secure.myob.com/oauth2/account/authorize/"
        assert provider_row[1] == "https://secure.myob.com/oauth2/v1/authorize/"
        assert provider_row[2] == "[]"

        app_row = connection.execute(
            text(
                "SELECT transport, provider_name, category, oauth_scopes,"
                " launch_config FROM public_mcp_apps WHERE app_id='myob'"
            )
        ).first()
        assert app_row[0] == "oauth"
        assert app_row[1] == "myob"
        assert app_row[2] == "Operations"
        assert "sme-contacts-customer" in str(app_row[3])
        assert "sme-banking" not in str(app_row[3])
        assert "xagent.web.tools.mcp.myob" in str(app_row[4])
        assert "MYOB_ACCESS_TOKEN" in str(app_row[4])
        assert "MYOB_BUSINESS_ID" in str(app_row[4])
        assert "MYOB_API_KEY" in str(app_row[4])


def test_upgrade_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()  # second run must not raise or duplicate
        app_count = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='myob'")
        ).scalar()
        assert app_count == 1
        provider_count = connection.execute(
            text("SELECT COUNT(*) FROM oauth_providers WHERE provider_name='myob'")
        ).scalar()
        assert provider_count == 1


def test_seed_rows_match_registry(tmp_path):
    """The migration snapshot and the runtime registry must define the same
    myob rows (the migration is a frozen copy; this catches drift)."""
    from xagent.web.builtin_mcp_registry import (
        get_builtin_oauth_provider_rows,
        get_builtin_public_mcp_app_rows,
    )

    migration = _load_migration_module()

    registry_app = next(
        row for row in get_builtin_public_mcp_app_rows() if row["app_id"] == "myob"
    )
    assert migration._myob_app_row() == registry_app

    registry_provider = next(
        row
        for row in get_builtin_oauth_provider_rows()
        if row["provider_name"] == "myob"
    )
    assert migration._myob_provider_row() == registry_provider


def test_downgrade_removes_provider_and_app(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert "myob" not in _app_ids(connection)
        assert "myob" not in _provider_names(connection)


def test_downgrade_keeps_provider_when_custom_myob_app_exists(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            connection.execute(
                text(
                    "INSERT INTO public_mcp_apps (app_id, name, transport, provider_name)"
                    " VALUES ('custom-myob', 'Custom MYOB', 'oauth', 'myob')"
                )
            )
            migration.downgrade()
        assert "myob" in _provider_names(connection)


def test_downgrade_preserves_pre_existing_myob_app(tmp_path):
    """A pre-existing "myob" app row (different shape than the seeded one)
    must survive downgrade -- upgrade()'s own `app_id not in
    existing_app_ids` check skipped inserting over it, so it was never
    "this migration's row" to remove. Deleting it unconditionally would
    also make the remaining-myob-apps count wrongly read as zero, letting
    the oauth_providers row underneath it be deleted too."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps (app_id, name, transport, provider_name)"
                " VALUES ('myob', 'Custom MYOB App', 'oauth', 'myob')"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert "myob" in _app_ids(connection)
        assert "myob" in _provider_names(connection)


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
                " WHERE app_id = 'myob'"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        assert "myob" in _app_ids(connection)


def test_downgrade_preserves_admin_created_myob_provider(tmp_path):
    """A pre-existing admin-created "myob" provider (different shape than
    the seeded row) must survive downgrade even when no myob apps
    remain."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        connection.execute(
            text(
                "INSERT INTO oauth_providers"
                " (provider_name, name, client_id, client_secret, auth_url, token_url)"
                " VALUES ('myob', 'Custom MYOB', 'cid', 'secret',"
                " 'https://custom.myob.com/oauth2/account/authorize/',"
                " 'https://custom.myob.com/oauth2/v1/authorize/')"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert "myob" not in _app_ids(connection)
        assert "myob" in _provider_names(connection)


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
                " WHERE provider_name = 'myob'"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        assert "myob" not in _app_ids(connection)
        assert "myob" in _provider_names(connection)


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


def test_down_revision_matches_current_head():
    """Pin down_revision to the confirmed true head as of this branch's
    last rebase onto upstream/main, so a future migration insertion
    between them would be caught here rather than only surfacing as a
    confusing multiple-heads error from `alembic heads`."""
    migration = _load_migration_module()

    assert migration.down_revision == "20260903_model_management"
