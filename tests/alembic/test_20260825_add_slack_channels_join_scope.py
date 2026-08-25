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


def _create_table(connection, scopes: list[str]):
    connection.execute(
        text(
            """
            CREATE TABLE public_mcp_apps (
                id INTEGER PRIMARY KEY,
                app_id VARCHAR(100) NOT NULL UNIQUE,
                oauth_scopes JSON
            )
            """
        )
    )
    connection.execute(
        text(
            "INSERT INTO public_mcp_apps (app_id, oauth_scopes) VALUES ('slack', :scopes)"
        ),
        {"scopes": json.dumps(scopes)},
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


def _create_user_oauth_table(connection, with_refresh_token_column: bool = True):
    refresh_column = "refresh_token VARCHAR," if with_refresh_token_column else ""
    connection.execute(
        text(
            f"""
            CREATE TABLE user_oauth (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                provider VARCHAR(50) NOT NULL,
                {refresh_column}
                access_token VARCHAR NOT NULL
            )
            """
        )
    )
    refresh_col = ", refresh_token" if with_refresh_token_column else ""
    refresh_slack = ", 'old-slack-refresh'" if with_refresh_token_column else ""
    refresh_other = ", 'old-hubspot-refresh'" if with_refresh_token_column else ""
    connection.execute(
        text(
            f"INSERT INTO user_oauth (user_id, provider, access_token{refresh_col}) "
            f"VALUES (1, 'slack', 'old-slack-token'{refresh_slack}), "
            f"(1, 'hubspot', 'old-hubspot-token'{refresh_other})"
        )
    )


def _access_tokens(connection):
    return dict(
        connection.execute(text("SELECT provider, access_token FROM user_oauth")).all()
    )


def _refresh_tokens(connection):
    return dict(
        connection.execute(text("SELECT provider, refresh_token FROM user_oauth")).all()
    )


def _scopes(connection):
    scopes = connection.execute(
        text("SELECT oauth_scopes FROM public_mcp_apps WHERE app_id='slack'")
    ).scalar()
    return json.loads(scopes) if isinstance(scopes, str) else scopes


def _provider_default_scopes(connection, provider_name: str):
    scopes = connection.execute(
        text("SELECT default_scopes FROM oauth_providers WHERE provider_name = :p"),
        {"p": provider_name},
    ).scalar()
    return json.loads(scopes) if isinstance(scopes, str) else scopes


def test_upgrade_adds_channels_join_scope(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert _scopes(connection) == migration.CURRENT_SCOPES
        assert "channels:join" in _scopes(connection)


def test_upgrade_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()
        assert _scopes(connection) == migration.CURRENT_SCOPES


def test_downgrade_restores_previous_scopes(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert _scopes(connection) == migration.PREVIOUS_SCOPES


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


def test_upgrade_invalidates_existing_slack_grant_only(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        _create_user_oauth_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        tokens = _access_tokens(connection)
        assert tokens["slack"] == ""
        assert tokens["hubspot"] == "old-hubspot-token"
        refresh_tokens = _refresh_tokens(connection)
        assert refresh_tokens["slack"] is None
        assert refresh_tokens["hubspot"] == "old-hubspot-refresh"


def test_upgrade_does_not_reinvalidate_grants_when_scopes_already_current(tmp_path):
    """A second upgrade() run against an already-migrated DB (a direct
    re-invocation, an ops retry, or a stamp+reapply) must not wipe a grant
    the user already reconnected after the first run — upgrade() must only
    invalidate grants the first time channels:join is actually added."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        _create_user_oauth_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            # Simulate the user reconnecting after the first run.
            connection.execute(
                text(
                    "UPDATE user_oauth SET access_token = 'new-slack-token', "
                    "refresh_token = 'new-slack-refresh' WHERE provider = 'slack'"
                )
            )
            migration.upgrade()
        tokens = _access_tokens(connection)
        assert tokens["slack"] == "new-slack-token"
        refresh_tokens = _refresh_tokens(connection)
        assert refresh_tokens["slack"] == "new-slack-refresh"


def test_upgrade_logs_a_warning_when_grants_are_disconnected(tmp_path, caplog):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        _create_user_oauth_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            with caplog.at_level("WARNING", logger=migration.logger.name):
                migration.upgrade()
    assert any(
        "Disconnected" in r.message and "Slack" in r.message for r in caplog.records
    )


def test_upgrade_does_not_log_when_no_grants_exist(tmp_path, caplog):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        with patch.object(migration, "op", _operations(connection)):
            with caplog.at_level("WARNING", logger=migration.logger.name):
                migration.upgrade()  # no user_oauth table at all
    assert caplog.records == []


def test_upgrade_clears_access_token_without_refresh_token_column(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        _create_user_oauth_table(connection, with_refresh_token_column=False)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # must not raise when refresh_token is missing
        tokens = _access_tokens(connection)
        assert tokens["slack"] == ""


def test_downgrade_does_not_touch_user_oauth(tmp_path):
    """Cleared access tokens are gone for good; downgrade only reverts
    public_mcp_apps, mirroring 20260812's own downgrade."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        _create_user_oauth_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        tokens = _access_tokens(connection)
        assert tokens["slack"] == ""


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
    registry_provider = next(
        r for r in get_builtin_oauth_provider_rows() if r["provider_name"] == "slack"
    )
    assert migration.CURRENT_SCOPES == registry_provider["default_scopes"]
