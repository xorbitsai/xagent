"""Tests for adding the Slack channels:join OAuth scope."""

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
        / "src/xagent/migrations/versions/20260825_add_slack_channels_join_scope.py"
    )
    spec = importlib.util.spec_from_file_location(
        "add_slack_channels_join_scope_migration", migration_file
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def test_upgrade_adds_channels_join_scope_and_description(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(
            connection, migration.PREVIOUS_SCOPES, migration.PREVIOUS_DESCRIPTION
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        description, scopes = _row(connection)
        assert scopes == migration.CURRENT_SCOPES
        assert "channels:join" in scopes
        assert description == migration.CURRENT_DESCRIPTION


def test_upgrade_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(
            connection, migration.PREVIOUS_SCOPES, migration.PREVIOUS_DESCRIPTION
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()
        assert _scopes(connection) == migration.CURRENT_SCOPES


def test_upgrade_does_not_disconnect_existing_user_oauth_grants(tmp_path):
    """Slack doesn't retroactively grant a newly-requested scope to an
    already-issued token, but it also doesn't revoke what that token could
    already do — adding channels:join must not force every existing user to
    reconnect just to keep using the tools they already had access to."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        _create_user_oauth_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        tokens = _access_tokens(connection)
        assert tokens["slack"] == "old-slack-token"
        assert tokens["hubspot"] == "old-hubspot-token"


def test_downgrade_restores_previous_scopes_and_description(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(
            connection, migration.PREVIOUS_SCOPES, migration.PREVIOUS_DESCRIPTION
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        description, scopes = _row(connection)
        assert scopes == migration.PREVIOUS_SCOPES
        assert description == migration.PREVIOUS_DESCRIPTION


def test_upgrade_downgrade_upgrade_round_trip(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
            migration.upgrade()
        assert _scopes(connection) == migration.CURRENT_SCOPES


def test_upgrade_without_table_is_a_noop(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # must not raise when the table doesn't exist


def test_upgrade_preserves_customized_description(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES, "A custom description")
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        description, scopes = _row(connection)
        assert description == "A custom description"
        # oauth_scopes is builtin-protected, so it always updates regardless
        # of the description guard — see _set_slack_scopes's own docstring.
        assert scopes == migration.CURRENT_SCOPES


def test_downgrade_preserves_customized_description(tmp_path):
    """Mirrors test_downgrade_preserves_customized_provider_default_scopes:
    an operator customization made after upgrade() must survive downgrade()
    too, not just upgrade()."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(
            connection, migration.PREVIOUS_SCOPES, migration.PREVIOUS_DESCRIPTION
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            connection.execute(
                text(
                    "UPDATE public_mcp_apps SET description = :d WHERE app_id = 'slack'"
                ),
                {"d": "A custom description"},
            )
            migration.downgrade()
        description, _ = _row(connection)
        assert description == "A custom description"


def test_upgrade_updates_provider_default_scopes(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        _create_oauth_providers_table(connection, migration.PREVIOUS_SCOPES)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert _provider_default_scopes(connection, "slack") == migration.CURRENT_SCOPES
        # A different provider row must be untouched.
        assert _provider_default_scopes(connection, "hubspot") == ["oauth"]


def test_downgrade_restores_provider_default_scopes(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        _create_oauth_providers_table(connection, migration.PREVIOUS_SCOPES)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert (
            _provider_default_scopes(connection, "slack") == migration.PREVIOUS_SCOPES
        )


def test_upgrade_without_oauth_providers_table_is_a_noop(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # must not raise when oauth_providers is missing


def test_upgrade_preserves_customized_provider_default_scopes(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        _create_oauth_providers_table(connection, ["chat:write", "custom:scope"])
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert _provider_default_scopes(connection, "slack") == [
            "chat:write",
            "custom:scope",
        ]
        # The app-facing oauth_scopes column is unaffected by this guard —
        # it always updates (see _set_slack_scopes's own docstring).
        assert _scopes(connection) == migration.CURRENT_SCOPES


def test_downgrade_preserves_customized_provider_default_scopes(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        _create_oauth_providers_table(connection, migration.PREVIOUS_SCOPES)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            connection.execute(
                text(
                    "UPDATE oauth_providers SET default_scopes = :s "
                    "WHERE provider_name = 'slack'"
                ),
                {"s": json.dumps(["chat:write", "custom:scope"])},
            )
            migration.downgrade()
        assert _provider_default_scopes(connection, "slack") == [
            "chat:write",
            "custom:scope",
        ]


def test_upgrade_updates_provider_default_scopes_when_no_row_exists(tmp_path):
    """No 'slack' provider row is not a customization to protect — the
    if-unchanged guard must not turn a previously-unconditional no-op into a
    silent skip once a row is later created some other way."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
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
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # must not raise with no matching row
        assert _provider_default_scopes(connection, "slack") is None


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
