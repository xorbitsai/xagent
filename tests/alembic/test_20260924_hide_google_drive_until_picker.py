"""Tests for gating Google Drive until the drive.file Picker flow exists."""

import importlib.util
import sqlite3
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20260924_hide_google_drive_until_picker.py"
)


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "hide_google_drive_until_picker_migration", MIGRATION_PATH
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
        sa.Column("description", sa.Text),
        sa.Column("is_visible_in_connector", sa.Boolean, nullable=False),
    )


def test_upgrade_hides_drive_updates_canonical_description_and_preserves_custom(
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
                {
                    "app_id": "google-drive",
                    "description": migration.PREVIOUS_DESCRIPTION,
                    "is_visible_in_connector": True,
                },
                {
                    "app_id": "custom-drive",
                    "description": migration.PREVIOUS_DESCRIPTION,
                    "is_visible_in_connector": True,
                },
            ],
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()

        rows = {row["app_id"]: dict(row) for row in connection.execute(sa.select(table)).mappings()}

    assert rows["google-drive"]["description"] == migration.CURRENT_DESCRIPTION
    assert rows["google-drive"]["is_visible_in_connector"] is False
    assert rows["custom-drive"]["description"] == migration.PREVIOUS_DESCRIPTION
    assert rows["custom-drive"]["is_visible_in_connector"] is True


def test_downgrade_restores_only_canonical_drive_description(tmp_path) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            sa.insert(table),
            {
                "app_id": "google-drive",
                "description": migration.CURRENT_DESCRIPTION,
                "is_visible_in_connector": False,
            },
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        row = connection.execute(sa.select(table)).mappings().one()

    assert row["description"] == migration.PREVIOUS_DESCRIPTION
    assert row["is_visible_in_connector"] is True


def test_upgrade_is_noop_when_catalog_columns_are_missing() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    table = sa.Table(
        "public_mcp_apps",
        metadata,
        sa.Column("app_id", sa.String(100), primary_key=True),
        sa.Column("description", sa.Text),
    )
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            sa.insert(table),
            {"app_id": "google-drive", "description": migration.PREVIOUS_DESCRIPTION},
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        row = connection.execute(sa.select(table)).mappings().one()

    assert row["description"] == migration.CURRENT_DESCRIPTION


def test_offline_sqlite_upgrade_round_trips_visibility_and_description() -> None:
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
        "app_id TEXT PRIMARY KEY, description TEXT, "
        "is_visible_in_connector BOOLEAN)"
    )
    connection.execute(
        "INSERT INTO public_mcp_apps VALUES (?, ?, ?)",
        ("google-drive", migration.PREVIOUS_DESCRIPTION, 1),
    )
    connection.executescript(output.getvalue())
    row = connection.execute(
        "SELECT description, is_visible_in_connector FROM public_mcp_apps"
    ).fetchone()
    connection.close()

    assert row == (migration.CURRENT_DESCRIPTION, 0)


def test_offline_postgresql_upgrade_uses_literal_updates() -> None:
    migration = _load_migration_module()
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql",
        opts={"as_sql": True, "output_buffer": output},
    )
    with Operations.context(context):
        migration.upgrade()

    sql = output.getvalue()
    assert sql.count("UPDATE public_mcp_apps SET") == 2
    assert "drive.file" not in sql
    assert "%(" not in sql


def test_registry_matches_migration() -> None:
    migration = _load_migration_module()
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app

    app = get_builtin_public_mcp_app("google-drive")
    assert app is not None
    assert app["description"] == migration.CURRENT_DESCRIPTION
    # The follow-up Picker migration re-enables this seeded row after the
    # restricted-scope flow is available; this migration only owns the
    # temporary hidden state.


def test_revision_metadata() -> None:
    migration = _load_migration_module()

    assert migration.revision == "20260924_hide_google_drive_until_picker"
    assert migration.down_revision == "20260924_narrow_google_oauth_scopes"
