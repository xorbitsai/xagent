"""Tests for narrowing the Google Calendar connector's OAuth scope."""

import importlib.util
import json
import sqlite3
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20260817_narrow_google_calendar_scope.py"
)

OLD_SCOPES = ["https://www.googleapis.com/auth/calendar"]
NEW_SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "narrow_google_calendar_scope_migration", MIGRATION_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection) -> Operations:
    return Operations(MigrationContext.configure(connection))


def _public_mcp_apps(metadata: sa.MetaData) -> sa.Table:
    return sa.Table(
        "public_mcp_apps",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("app_id", sa.String(100), nullable=False, unique=True),
        sa.Column("oauth_scopes", sa.JSON),
    )


def _scopes_by_app_id(connection, table: sa.Table) -> dict[str, object]:
    rows = connection.execute(sa.select(table.c.app_id, table.c.oauth_scopes))
    return {row.app_id: row.oauth_scopes for row in rows}


def test_upgrade_narrows_google_calendar_scope_without_touching_other_apps(
    tmp_path,
) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            sa.insert(table),
            [
                {"app_id": "google-calendar", "oauth_scopes": OLD_SCOPES},
                {"app_id": "gmail", "oauth_scopes": ["gmail-scope"]},
            ],
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        scopes = _scopes_by_app_id(connection, table)

    assert scopes["google-calendar"] == NEW_SCOPES
    assert scopes["gmail"] == ["gmail-scope"]


def test_upgrade_is_idempotent(tmp_path) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            sa.insert(table),
            {"app_id": "google-calendar", "oauth_scopes": OLD_SCOPES},
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()

        scopes = _scopes_by_app_id(connection, table)

    assert scopes["google-calendar"] == NEW_SCOPES


def test_upgrade_skips_when_row_is_absent(tmp_path) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        scopes = _scopes_by_app_id(connection, table)

    assert scopes == {}


def test_upgrade_skips_when_catalog_table_is_absent() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        # Not just "didn't raise": confirm it truly no-op'd rather than, say,
        # creating the table it was supposed to find missing.
        assert sa.inspect(connection).get_table_names() == []


def test_upgrade_skips_when_oauth_scopes_column_is_absent() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    table = sa.Table(
        "public_mcp_apps",
        metadata,
        sa.Column("app_id", sa.String(100), primary_key=True),
    )
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(sa.insert(table), {"app_id": "google-calendar"})

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        stored = connection.execute(sa.select(table)).mappings().one()

    assert stored["app_id"] == "google-calendar"


def test_downgrade_restores_the_full_calendar_scope(tmp_path) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            sa.insert(table),
            {"app_id": "google-calendar", "oauth_scopes": OLD_SCOPES},
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()

        scopes = _scopes_by_app_id(connection, table)

    assert scopes["google-calendar"] == OLD_SCOPES


def test_downgrade_restores_the_full_calendar_scope_without_touching_other_apps(
    tmp_path,
) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            sa.insert(table),
            [
                {"app_id": "google-calendar", "oauth_scopes": NEW_SCOPES},
                {"app_id": "gmail", "oauth_scopes": ["gmail-scope"]},
            ],
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()

        scopes = _scopes_by_app_id(connection, table)

    assert scopes["google-calendar"] == OLD_SCOPES
    assert scopes["gmail"] == ["gmail-scope"]


def test_offline_postgresql_upgrade_emits_literal_update_only_sql() -> None:
    migration = _load_migration_module()
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql",
        opts={"as_sql": True, "output_buffer": output},
    )

    with Operations.context(context):
        migration.upgrade()

    sql = output.getvalue()
    assert sql.count("UPDATE public_mcp_apps SET") == 1
    assert "oauth_scopes=" in sql
    assert "public_mcp_apps.app_id = 'google-calendar'" in sql
    assert "calendar.events" in sql
    assert "INSERT INTO public_mcp_apps" not in sql
    assert "DELETE FROM public_mcp_apps" not in sql
    assert "%(" not in sql


def test_offline_sqlite_upgrade_round_trips_json_scope_value() -> None:
    migration = _load_migration_module()
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="sqlite",
        opts={"as_sql": True, "output_buffer": output},
    )

    with Operations.context(context):
        migration.upgrade()

    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE public_mcp_apps (app_id TEXT PRIMARY KEY, oauth_scopes JSON)"
    )
    connection.execute(
        "INSERT INTO public_mcp_apps (app_id, oauth_scopes) VALUES (?, ?)",
        ("google-calendar", json.dumps(OLD_SCOPES)),
    )
    connection.executescript(output.getvalue())

    stored = connection.execute(
        "SELECT oauth_scopes FROM public_mcp_apps WHERE app_id = 'google-calendar'"
    ).fetchone()
    connection.close()

    assert stored is not None
    assert json.loads(stored[0]) == NEW_SCOPES


def test_offline_postgresql_downgrade_emits_literal_update_only_sql() -> None:
    migration = _load_migration_module()
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql",
        opts={"as_sql": True, "output_buffer": output},
    )

    with Operations.context(context):
        migration.downgrade()

    sql = output.getvalue()
    assert sql.count("UPDATE public_mcp_apps SET") == 1
    assert "oauth_scopes=" in sql
    assert "public_mcp_apps.app_id = 'google-calendar'" in sql
    # "auth/calendar" alone is a substring of both OLD_SCOPES and NEW_SCOPES
    # ("auth/calendar.events"), so it can't distinguish which one was
    # emitted; assert the narrower scope is absent to catch a swapped
    # OLD_SCOPES/NEW_SCOPES argument in downgrade()'s _set_calendar_scopes()
    # call.
    assert "auth/calendar" in sql
    assert "calendar.events" not in sql
    assert "INSERT INTO public_mcp_apps" not in sql
    assert "DELETE FROM public_mcp_apps" not in sql
    assert "%(" not in sql


def test_offline_sqlite_downgrade_round_trips_json_scope_value() -> None:
    migration = _load_migration_module()
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="sqlite",
        opts={"as_sql": True, "output_buffer": output},
    )

    with Operations.context(context):
        migration.downgrade()

    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE public_mcp_apps (app_id TEXT PRIMARY KEY, oauth_scopes JSON)"
    )
    connection.execute(
        "INSERT INTO public_mcp_apps (app_id, oauth_scopes) VALUES (?, ?)",
        ("google-calendar", json.dumps(NEW_SCOPES)),
    )
    connection.executescript(output.getvalue())

    stored = connection.execute(
        "SELECT oauth_scopes FROM public_mcp_apps WHERE app_id = 'google-calendar'"
    ).fetchone()
    connection.close()

    assert stored is not None
    assert json.loads(stored[0]) == OLD_SCOPES


def test_migration_fields_match_registry() -> None:
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration_module()
    registry_row = next(
        row
        for row in get_builtin_public_mcp_app_rows()
        if row["app_id"] == "google-calendar"
    )

    assert list(migration.NEW_SCOPES) == registry_row["oauth_scopes"]


def test_revision_metadata() -> None:
    migration = _load_migration_module()

    assert migration.revision == "20260817_narrow_google_calendar_scope"
    assert migration.down_revision == "20260825_add_slack_channels_join_scope"
