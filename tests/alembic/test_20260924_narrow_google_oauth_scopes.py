"""Tests for the temporary Google restricted-scope removal."""

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
    / "src/xagent/migrations/versions/20260924_narrow_google_oauth_scopes.py"
)

GMAIL_OLD_SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
GMAIL_NEW_SCOPES: list[str] = []
DRIVE_OLD_SCOPES = ["https://www.googleapis.com/auth/drive"]
DRIVE_NEW_SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "narrow_google_oauth_scopes_migration", MIGRATION_PATH
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
        sa.Column("is_visible_in_connector", sa.Boolean, nullable=False),
    )


def _rows_by_app_id(connection, table: sa.Table) -> dict[str, dict[str, object]]:
    return {
        row["app_id"]: dict(row)
        for row in connection.execute(sa.select(table)).mappings()
    }


def test_upgrade_removes_gmail_scope_hides_gmail_and_narrows_drive(tmp_path) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            sa.insert(table),
            [
                {
                    "app_id": "gmail",
                    "oauth_scopes": GMAIL_OLD_SCOPES,
                    "is_visible_in_connector": True,
                },
                {
                    "app_id": "google-drive",
                    "oauth_scopes": DRIVE_OLD_SCOPES,
                    "is_visible_in_connector": True,
                },
                {
                    "app_id": "other",
                    "oauth_scopes": ["other-scope"],
                    "is_visible_in_connector": True,
                },
            ],
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()

        rows = _rows_by_app_id(connection, table)

    assert rows["gmail"]["oauth_scopes"] == GMAIL_NEW_SCOPES
    assert rows["gmail"]["is_visible_in_connector"] is False
    assert rows["google-drive"]["oauth_scopes"] == DRIVE_NEW_SCOPES
    assert rows["google-drive"]["is_visible_in_connector"] is True
    assert rows["other"]["oauth_scopes"] == ["other-scope"]
    assert rows["other"]["is_visible_in_connector"] is True


def test_downgrade_restores_previous_scopes_and_gmail_visibility(tmp_path) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            sa.insert(table),
            [
                {
                    "app_id": "gmail",
                    "oauth_scopes": GMAIL_NEW_SCOPES,
                    "is_visible_in_connector": False,
                },
                {
                    "app_id": "google-drive",
                    "oauth_scopes": DRIVE_NEW_SCOPES,
                    "is_visible_in_connector": True,
                },
            ],
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()

        rows = _rows_by_app_id(connection, table)

    assert rows["gmail"]["oauth_scopes"] == GMAIL_OLD_SCOPES
    assert rows["gmail"]["is_visible_in_connector"] is True
    assert rows["google-drive"]["oauth_scopes"] == DRIVE_OLD_SCOPES


def test_upgrade_is_a_noop_when_catalog_table_is_absent() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        assert sa.inspect(connection).get_table_names() == []


def test_upgrade_handles_individually_missing_catalog_columns() -> None:
    migration = _load_migration_module()

    for included_column in ("oauth_scopes", "is_visible_in_connector"):
        engine = sa.create_engine("sqlite:///:memory:")
        metadata = sa.MetaData()
        columns = [sa.Column("app_id", sa.String(100), primary_key=True)]
        if included_column == "oauth_scopes":
            columns.append(sa.Column("oauth_scopes", sa.JSON))
        else:
            columns.append(sa.Column("is_visible_in_connector", sa.Boolean))
        table = sa.Table("public_mcp_apps", metadata, *columns)
        metadata.create_all(engine)

        with engine.begin() as connection:
            values: dict[str, object] = {"app_id": "gmail"}
            if included_column == "oauth_scopes":
                values["oauth_scopes"] = GMAIL_OLD_SCOPES
            else:
                values["is_visible_in_connector"] = True
            connection.execute(sa.insert(table), values)

            with patch.object(migration, "op", _operations(connection)):
                migration.upgrade()

            stored = connection.execute(sa.select(table)).mappings().one()

        if included_column == "oauth_scopes":
            assert stored["oauth_scopes"] == GMAIL_NEW_SCOPES
        else:
            assert stored["is_visible_in_connector"] is False


def test_offline_sqlite_upgrade_round_trips_scopes_and_visibility() -> None:
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
        "CREATE TABLE public_mcp_apps ("
        "app_id TEXT PRIMARY KEY, oauth_scopes JSON, "
        "is_visible_in_connector BOOLEAN)"
    )
    connection.executemany(
        "INSERT INTO public_mcp_apps VALUES (?, ?, ?)",
        [
            ("gmail", json.dumps(GMAIL_OLD_SCOPES), 1),
            ("google-drive", json.dumps(DRIVE_OLD_SCOPES), 1),
        ],
    )
    connection.executescript(output.getvalue())
    rows = {
        app_id: (json.loads(scopes), bool(visible))
        for app_id, scopes, visible in connection.execute(
            "SELECT app_id, oauth_scopes, is_visible_in_connector FROM public_mcp_apps"
        )
    }
    connection.close()

    assert rows["gmail"] == (GMAIL_NEW_SCOPES, False)
    assert rows["google-drive"] == (DRIVE_NEW_SCOPES, True)


def test_offline_postgresql_upgrade_contains_only_literal_updates() -> None:
    migration = _load_migration_module()
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql",
        opts={"as_sql": True, "output_buffer": output},
    )

    with Operations.context(context):
        migration.upgrade()

    sql = output.getvalue()
    assert sql.count("UPDATE public_mcp_apps SET") == 3
    assert "gmail.modify" not in sql
    assert "drive.file" in sql
    assert "is_visible_in_connector=false" in sql
    assert "%(" not in sql


def test_registry_and_migration_values_match() -> None:
    migration = _load_migration_module()
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    rows = {row["app_id"]: row for row in get_builtin_public_mcp_app_rows()}

    assert rows["gmail"]["oauth_scopes"] == list(migration.CURRENT_GMAIL_SCOPES)
    assert rows["gmail"]["is_visible_in_connector"] is False
    assert rows["google-drive"]["oauth_scopes"] == list(
        migration.CURRENT_GOOGLE_DRIVE_SCOPES
    )


def test_revision_metadata() -> None:
    migration = _load_migration_module()

    assert migration.revision == "20260924_narrow_google_oauth_scopes"
    assert migration.down_revision == "20260923_seed_freshdesk_mcp_app"
