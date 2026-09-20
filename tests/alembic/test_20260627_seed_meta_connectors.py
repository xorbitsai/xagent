"""Tests for the Meta connector registry seed migration."""

import importlib.util
import json
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


def _load_facebook_scope_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260728_add_facebook_pages_read_user_content_scope.py"
    )
    spec = importlib.util.spec_from_file_location(
        "add_facebook_pages_read_user_content_scope_migration", migration_file
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


def _launch_config(connection, app_id):
    value = connection.execute(
        text("SELECT launch_config FROM public_mcp_apps WHERE app_id=:app_id"),
        {"app_id": app_id},
    ).scalar_one()
    return json.loads(value) if isinstance(value, str) else value


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
                " WHERE app_id IN ('facebook', 'instagram')"
            )
        ).scalar()
        assert app_count == 2
        provider_count = connection.execute(
            text("SELECT COUNT(*) FROM oauth_providers WHERE provider_name='meta'")
        ).scalar()
        assert provider_count == 1


def test_downgrade_cleans_up_after_descendant_normalization_migration(tmp_path):
    """20260715_normalize_builtin_mcp_launch runs after this migration and
    unconditionally rewrites facebook/instagram's launch_config (its own
    downgrade is a deliberate no-op).
    20260728_add_facebook_pages_read_user_content_scope runs later still and
    adds a scope to facebook's oauth_scopes, but -- unlike 20260715 -- it DOES
    revert that on its own downgrade. A downgrade chain that runs both of
    those descendant migrations' downgrades (in reverse revision order, as
    Alembic would) and then back through this migration must still remove
    the seeded apps and provider, not preserve them as if an operator had
    edited them.

    (No test here asserts this migration's frozen snapshot equals the live
    registry: 20260728 proves that isn't always true by design -- a
    downstream migration can own a reversible delta on a field this
    migration seeded, in which case the frozen snapshot correctly stays at
    the *original* value forever, matching what a full downgrade chain
    actually restores, not the live registry's current value.)"""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    normalize_migration = _load_normalize_migration_module()
    facebook_scope_migration = _load_facebook_scope_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        with patch.object(normalize_migration, "op", _operations(connection)):
            normalize_migration.upgrade()
        with patch.object(facebook_scope_migration, "op", _operations(connection)):
            facebook_scope_migration.upgrade()
            facebook_scope_migration.downgrade()
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        assert not {"facebook", "instagram"} & _app_ids(connection)
        assert "meta" not in _provider_names(connection)


def test_downgrade_removes_provider_and_apps(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            assert _launch_config(connection, "facebook")["command"] == "uv"
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


def test_downgrade_preserves_admin_created_meta_provider(tmp_path):
    """A pre-existing admin-created "meta" provider (different shape than the
    seeded row) must survive downgrade even when no meta apps remain."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        connection.execute(
            text(
                "INSERT INTO oauth_providers"
                " (provider_name, name, client_id, client_secret, auth_url, token_url)"
                " VALUES ('meta', 'Custom Meta', 'cid', 'secret',"
                " 'https://custom.example.com/authorize',"
                " 'https://custom.example.com/token')"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert not {"facebook", "instagram"} & _app_ids(connection)
        assert "meta" in _provider_names(connection)


def test_downgrade_keeps_provider_when_custom_meta_app_exists(tmp_path):
    """The shared "meta" oauth_providers row must survive downgrade if a
    non-seeded meta app is still present."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            connection.execute(
                text(
                    "INSERT INTO public_mcp_apps (app_id, name, transport, provider_name)"
                    " VALUES ('custom-facebook', 'Custom Facebook', 'oauth', 'meta')"
                )
            )
            migration.downgrade()
        assert "meta" in _provider_names(connection)


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
