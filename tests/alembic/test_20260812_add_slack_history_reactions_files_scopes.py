"""Tests for adding Slack history, reaction, and file-upload OAuth scopes."""

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
        / "src/xagent/migrations/versions/20260812_add_slack_history_reactions_files_scopes.py"
    )
    spec = importlib.util.spec_from_file_location(
        "add_slack_history_reactions_files_scopes_migration", migration_file
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


def _create_table(connection, description: str, with_description_column: bool = True):
    description_column = "description TEXT," if with_description_column else ""
    connection.execute(
        text(
            f"""
            CREATE TABLE public_mcp_apps (
                id INTEGER PRIMARY KEY,
                app_id VARCHAR(100) NOT NULL UNIQUE,
                {description_column}
                oauth_scopes JSON
            )
            """
        )
    )
    description_col = ", description" if with_description_column else ""
    description_val = ", :description" if with_description_column else ""
    connection.execute(
        text(
            f"INSERT INTO public_mcp_apps (app_id, oauth_scopes{description_col}) "
            f"VALUES ('slack', :scopes{description_val})"
        ),
        {
            "scopes": json.dumps(["chat:write", "chat:write.public", "channels:read"]),
            "description": description,
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


def _row(connection):
    row = connection.execute(
        text(
            "SELECT description, oauth_scopes FROM public_mcp_apps WHERE app_id='slack'"
        )
    ).first()
    description, scopes = row[0], row[1]
    return description, json.loads(scopes) if isinstance(scopes, str) else scopes


def test_upgrade_adds_new_scopes_and_description(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, description=migration.PREVIOUS_DESCRIPTION)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        description, scopes = _row(connection)
        assert description == migration.CURRENT_DESCRIPTION
        assert scopes == migration.CURRENT_SCOPES


def test_upgrade_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, description=migration.PREVIOUS_DESCRIPTION)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()
        description, scopes = _row(connection)
        assert description == migration.CURRENT_DESCRIPTION
        assert scopes == migration.CURRENT_SCOPES


def test_downgrade_restores_previous_scopes_and_description(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, description=migration.PREVIOUS_DESCRIPTION)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        description, scopes = _row(connection)
        assert description == migration.PREVIOUS_DESCRIPTION
        assert scopes == migration.PREVIOUS_SCOPES


def test_upgrade_downgrade_upgrade_round_trip(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, description=migration.PREVIOUS_DESCRIPTION)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
            migration.upgrade()
        description, scopes = _row(connection)
        assert description == migration.CURRENT_DESCRIPTION
        assert scopes == migration.CURRENT_SCOPES


def test_upgrade_preserves_admin_customized_description(tmp_path):
    """description is not in _BUILTIN_PROTECTED_FIELDS (admin_mcp.py), so an
    operator can have edited it via the admin PATCH endpoint. The migration
    must not clobber a value that no longer matches the last-known default.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, description="Our internal Slack connector")
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        description, scopes = _row(connection)
        assert description == "Our internal Slack connector"
        assert scopes == migration.CURRENT_SCOPES


def test_downgrade_preserves_admin_customized_description(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, description=migration.PREVIOUS_DESCRIPTION)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            connection.execute(
                text(
                    "UPDATE public_mcp_apps SET description = :d WHERE app_id = 'slack'"
                ),
                {"d": "Our internal Slack connector"},
            )
            migration.downgrade()
        description, scopes = _row(connection)
        assert description == "Our internal Slack connector"
        assert scopes == migration.PREVIOUS_SCOPES


def test_upgrade_without_description_column_still_updates_scopes(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(
            connection,
            description=migration.PREVIOUS_DESCRIPTION,
            with_description_column=False,
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # must not raise when description is missing
        scopes = connection.execute(
            text("SELECT oauth_scopes FROM public_mcp_apps WHERE app_id='slack'")
        ).scalar()
        assert json.loads(scopes) == migration.CURRENT_SCOPES


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
        _create_table(connection, description=migration.PREVIOUS_DESCRIPTION)
        _create_user_oauth_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        tokens = _access_tokens(connection)
        assert tokens["slack"] == ""
        assert tokens["hubspot"] == "old-hubspot-token"
        refresh_tokens = _refresh_tokens(connection)
        assert refresh_tokens["slack"] is None
        assert refresh_tokens["hubspot"] == "old-hubspot-refresh"


def test_upgrade_logs_a_warning_when_grants_are_disconnected(tmp_path, caplog):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, description=migration.PREVIOUS_DESCRIPTION)
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
        _create_table(connection, description=migration.PREVIOUS_DESCRIPTION)
        with patch.object(migration, "op", _operations(connection)):
            with caplog.at_level("WARNING", logger=migration.logger.name):
                migration.upgrade()  # no user_oauth table at all
    assert caplog.records == []


def test_upgrade_clears_access_token_without_refresh_token_column(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, description=migration.PREVIOUS_DESCRIPTION)
        _create_user_oauth_table(connection, with_refresh_token_column=False)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # must not raise when refresh_token is missing
        tokens = _access_tokens(connection)
        assert tokens["slack"] == ""


def test_upgrade_without_user_oauth_table_is_a_noop(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, description=migration.PREVIOUS_DESCRIPTION)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # must not raise when user_oauth doesn't exist


def test_downgrade_does_not_touch_user_oauth(tmp_path):
    """Cleared access tokens are gone for good; downgrade only reverts
    public_mcp_apps, mirroring the HubSpot scope migration's approach."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, description=migration.PREVIOUS_DESCRIPTION)
        _create_user_oauth_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        tokens = _access_tokens(connection)
        assert tokens["slack"] == ""


def test_migration_fields_match_registry():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration_module()
    registry_row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "slack"
    )
    assert migration.CURRENT_SCOPES == registry_row["oauth_scopes"]
    assert migration.CURRENT_DESCRIPTION == registry_row["description"]
