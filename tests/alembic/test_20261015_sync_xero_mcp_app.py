"""Tests for syncing the persisted Xero connector configuration."""

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
        / "src/xagent/migrations/versions/20261015_sync_xero_mcp_app.py"
    )
    spec = importlib.util.spec_from_file_location("sync_xero_mcp_app", migration_file)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


def _create_tables(connection, migration):
    connection.execute(
        text(
            """
            CREATE TABLE public_mcp_apps (
                id INTEGER PRIMARY KEY,
                app_id VARCHAR(100) NOT NULL UNIQUE,
                oauth_scopes JSON,
                launch_config JSON
            )
            """
        )
    )
    connection.execute(
        text(
            """
            CREATE TABLE user_oauth (
                id INTEGER PRIMARY KEY,
                provider VARCHAR(50) NOT NULL,
                access_token VARCHAR NOT NULL,
                refresh_token VARCHAR
            )
            """
        )
    )
    connection.execute(
        text(
            """
            INSERT INTO public_mcp_apps (app_id, oauth_scopes, launch_config)
            VALUES ('xero', :scopes, :launch_config)
            """
        ),
        {
            "scopes": json.dumps(migration.PREVIOUS_SCOPES),
            "launch_config": json.dumps(migration.PREVIOUS_LAUNCH_CONFIG),
        },
    )
    connection.execute(
        text(
            """
            INSERT INTO user_oauth (provider, access_token, refresh_token)
            VALUES ('xero', 'old-access', 'old-refresh'),
                   ('hubspot', 'other-access', 'other-refresh')
            """
        )
    )


def _xero_app(connection):
    row = connection.execute(
        text(
            "SELECT oauth_scopes, launch_config FROM public_mcp_apps "
            "WHERE app_id='xero'"
        )
    ).one()
    return tuple(
        json.loads(value) if isinstance(value, str) else value for value in row
    )


def test_upgrade_syncs_config_and_invalidates_only_xero_grants(tmp_path):
    migration = _load_migration_module()
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    with engine.begin() as connection:
        _create_tables(connection, migration)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        assert _xero_app(connection) == (
            migration.CURRENT_SCOPES,
            migration.CURRENT_LAUNCH_CONFIG,
        )
        grants = {
            row.provider: (row.access_token, row.refresh_token)
            for row in connection.execute(
                text("SELECT provider, access_token, refresh_token FROM user_oauth")
            )
        }
        assert grants["xero"] == ("", None)
        assert grants["hubspot"] == ("other-access", "other-refresh")


def test_downgrade_restores_previous_persisted_config(tmp_path):
    migration = _load_migration_module()
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    with engine.begin() as connection:
        _create_tables(connection, migration)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert _xero_app(connection) == (
            migration.PREVIOUS_SCOPES,
            migration.PREVIOUS_LAUNCH_CONFIG,
        )


def test_upgrade_and_downgrade_no_op_without_tables(tmp_path):
    migration = _load_migration_module()
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()


def test_current_values_match_builtin_registry():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration_module()
    registry_row = next(
        row for row in get_builtin_public_mcp_app_rows() if row["app_id"] == "xero"
    )
    assert migration.CURRENT_SCOPES == registry_row["oauth_scopes"]
    assert migration.CURRENT_LAUNCH_CONFIG == registry_row["launch_config"]
