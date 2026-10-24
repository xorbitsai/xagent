"""Tests for adding the Slack users:read and users:read.email OAuth scopes."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text

_VERSIONS = Path(__file__).parent.parent.parent / "src/xagent/migrations/versions"
ADDED_SCOPES = {"users:read", "users:read.email"}


def _load(file_name: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _VERSIONS / file_name)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_migration_module():
    return _load(
        "20261023_add_slack_users_read_scopes.py",
        "add_slack_users_read_scopes_migration",
    )


def _load_prior_migration_module():
    return _load(
        "20260825_add_slack_channels_join_scope.py",
        "add_slack_channels_join_scope_migration",
    )


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


def _create_table(connection, scopes: list[str], description: str | None = None):
    connection.execute(
        text(
            """
            CREATE TABLE public_mcp_apps (
                id INTEGER PRIMARY KEY,
                app_id VARCHAR(100) NOT NULL UNIQUE,
                description TEXT,
                oauth_scopes JSON
            )
            """
        )
    )
    connection.execute(
        text(
            "INSERT INTO public_mcp_apps (app_id, description, oauth_scopes) "
            "VALUES ('slack', :description, :scopes)"
        ),
        {"description": description, "scopes": json.dumps(scopes)},
    )


def _create_oauth_providers_table(connection, default_scopes: list[str]):
    connection.execute(
        text(
            """
            CREATE TABLE oauth_providers (
                id INTEGER PRIMARY KEY,
                provider_name VARCHAR(50) NOT NULL UNIQUE,
                default_scopes JSON
            )
            """
        )
    )
    connection.execute(
        text(
            "INSERT INTO oauth_providers (provider_name, default_scopes) "
            "VALUES ('slack', :scopes), ('hubspot', :other_scopes)"
        ),
        {
            "scopes": json.dumps(default_scopes),
            "other_scopes": json.dumps(["oauth"]),
        },
    )


def _create_user_oauth_table(connection):
    connection.execute(
        text(
            """
            CREATE TABLE user_oauth (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                provider VARCHAR(50) NOT NULL,
                access_token VARCHAR NOT NULL
            )
            """
        )
    )
    connection.execute(
        text(
            "INSERT INTO user_oauth (user_id, provider, access_token) "
            "VALUES (1, 'slack', 'old-slack-token'), (1, 'hubspot', 'old-hubspot-token')"
        )
    )


def _access_tokens(connection):
    return dict(
        connection.execute(text("SELECT provider, access_token FROM user_oauth")).all()
    )


def _row(connection):
    row = connection.execute(
        text(
            "SELECT description, oauth_scopes FROM public_mcp_apps WHERE app_id='slack'"
        )
    ).one()
    description, scopes = row[0], row[1]
    return description, json.loads(scopes) if isinstance(scopes, str) else scopes


def _scopes(connection):
    return _row(connection)[1]


def _provider_default_scopes(connection, provider_name: str):
    scopes = connection.execute(
        text("SELECT default_scopes FROM oauth_providers WHERE provider_name = :p"),
        {"p": provider_name},
    ).scalar()
    return json.loads(scopes) if isinstance(scopes, str) else scopes


def _engine(tmp_path):
    return create_engine(f"sqlite:///{tmp_path / 'test.db'}")


def test_upgrade_adds_users_read_scopes_and_description(tmp_path):
    migration = _load_migration_module()
    with _engine(tmp_path).begin() as connection:
        _create_table(
            connection, migration.PREVIOUS_SCOPES, migration.PREVIOUS_DESCRIPTION
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        description, scopes = _row(connection)
        assert scopes == migration.CURRENT_SCOPES
        assert ADDED_SCOPES <= set(scopes)
        assert description == migration.CURRENT_DESCRIPTION


def test_upgrade_is_idempotent(tmp_path):
    """Two separate migration runs (each its own transaction), so the second
    run re-reads already-persisted state."""
    engine = _engine(tmp_path)
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(
            connection, migration.PREVIOUS_SCOPES, migration.PREVIOUS_DESCRIPTION
        )
        _create_oauth_providers_table(connection, migration.PREVIOUS_SCOPES)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert _scopes(connection) == migration.CURRENT_SCOPES
        assert _provider_default_scopes(connection, "slack") == migration.CURRENT_SCOPES


def test_upgrade_does_not_disconnect_existing_user_oauth_grants(tmp_path):
    """Existing tokens keep every tool they already supported; only the new
    lookup tool asks for a reconnect, through its missing_scope message."""
    migration = _load_migration_module()
    with _engine(tmp_path).begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        _create_user_oauth_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert _access_tokens(connection) == {
            "slack": "old-slack-token",
            "hubspot": "old-hubspot-token",
        }


def test_downgrade_restores_previous_scopes_and_description(tmp_path):
    migration = _load_migration_module()
    with _engine(tmp_path).begin() as connection:
        _create_table(
            connection, migration.PREVIOUS_SCOPES, migration.PREVIOUS_DESCRIPTION
        )
        _create_oauth_providers_table(connection, migration.PREVIOUS_SCOPES)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        description, scopes = _row(connection)
        assert scopes == migration.PREVIOUS_SCOPES
        assert description == migration.PREVIOUS_DESCRIPTION
        assert (
            _provider_default_scopes(connection, "slack") == migration.PREVIOUS_SCOPES
        )
        # The other provider's row is never touched in either direction.
        assert _provider_default_scopes(connection, "hubspot") == ["oauth"]


def test_upgrade_without_tables_is_a_noop(tmp_path):
    migration = _load_migration_module()
    with _engine(tmp_path).begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()


def test_upgrade_preserves_customized_description(tmp_path):
    migration = _load_migration_module()
    with _engine(tmp_path).begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES, "Our own Slack text")
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        description, scopes = _row(connection)
        assert description == "Our own Slack text"
        assert scopes == migration.CURRENT_SCOPES


def test_upgrade_merges_into_customized_provider_default_scopes(tmp_path):
    """Only this migration's own delta is added, so an operator's custom list
    keeps its shape instead of being replaced by CURRENT_SCOPES."""
    migration = _load_migration_module()
    with _engine(tmp_path).begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        _create_oauth_providers_table(connection, ["chat:write", "custom:scope"])
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert set(_provider_default_scopes(connection, "slack")) == {
            "chat:write",
            "custom:scope",
            *ADDED_SCOPES,
        }
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        assert _provider_default_scopes(connection, "slack") == [
            "chat:write",
            "custom:scope",
        ]


def test_upgrade_seeds_full_scopes_into_null_provider_default_scopes(tmp_path):
    """A NULL default_scopes has no delta to merge into, so the full set is
    written rather than only this migration's two scopes."""
    migration = _load_migration_module()
    with _engine(tmp_path).begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        _create_oauth_providers_table(connection, [])
        connection.execute(
            text(
                "UPDATE oauth_providers SET default_scopes = NULL "
                "WHERE provider_name = 'slack'"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert _provider_default_scopes(connection, "slack") == migration.CURRENT_SCOPES


def test_migration_fields_match_registry():
    from xagent.web.builtin_mcp_registry import (
        get_builtin_oauth_provider_rows,
        get_builtin_public_mcp_app_rows,
    )

    migration = _load_migration_module()
    registry_row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "slack"
    )
    assert migration.CURRENT_SCOPES == registry_row["oauth_scopes"]
    assert migration.CURRENT_DESCRIPTION == registry_row["description"]
    registry_provider = next(
        r for r in get_builtin_oauth_provider_rows() if r["provider_name"] == "slack"
    )
    assert migration.CURRENT_SCOPES == registry_provider["default_scopes"]


def test_previous_fields_match_prior_migrations_current_fields():
    """PREVIOUS_* are hand-copied from 20260825's CURRENT_*. A stale copy
    would pass every other test while permanently breaking the description
    guard (WHERE description == expected_current) on real databases."""
    prior = _load_prior_migration_module()
    migration = _load_migration_module()
    assert migration.PREVIOUS_SCOPES == prior.CURRENT_SCOPES
    assert migration.PREVIOUS_DESCRIPTION == prior.CURRENT_DESCRIPTION


def test_revises_the_previous_head_and_fits_alembic_version_column():
    migration = _load_migration_module()
    assert migration.down_revision == "20261022_task_cmd_fair_index"
    # alembic_version.version_num is VARCHAR(32).
    assert len(migration.revision) <= 32


def test_upgrade_after_actually_running_20260825_upgrade(tmp_path):
    """Run the real two-migration sequence rather than starting from this
    migration's own PREVIOUS_* constants."""
    prior = _load_prior_migration_module()
    migration = _load_migration_module()
    with _engine(tmp_path).begin() as connection:
        _create_table(connection, prior.PREVIOUS_SCOPES, prior.PREVIOUS_DESCRIPTION)
        _create_oauth_providers_table(connection, prior.PREVIOUS_SCOPES)
        with patch.object(prior, "op", _operations(connection)):
            prior.upgrade()
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        description, scopes = _row(connection)
        assert scopes == migration.CURRENT_SCOPES
        assert description == migration.CURRENT_DESCRIPTION
        # Set equality: delta merges append rather than reproduce the
        # canonical order.
        assert set(_provider_default_scopes(connection, "slack")) == set(
            migration.CURRENT_SCOPES
        )
