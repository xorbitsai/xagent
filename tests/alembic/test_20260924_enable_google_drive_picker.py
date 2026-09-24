"""Tests for re-enabling Google Drive after Picker support is available."""

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
    / "src/xagent/migrations/versions/20260924_enable_google_drive_picker.py"
)


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "enable_google_drive_picker_migration", MIGRATION_PATH
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
        sa.Column("is_visible_in_connector", sa.Boolean, nullable=False),
    )


def test_upgrade_enables_google_drive_idempotently(tmp_path) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            sa.insert(table),
            [
                {"app_id": "google-drive", "is_visible_in_connector": False},
                {"app_id": "other", "is_visible_in_connector": False},
            ],
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()
        rows = {
            row["app_id"]: dict(row)
            for row in connection.execute(sa.select(table)).mappings()
        }

    assert rows["google-drive"]["is_visible_in_connector"] is True
    assert rows["other"]["is_visible_in_connector"] is False


def test_downgrade_hides_google_drive(tmp_path) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            sa.insert(table),
            {"app_id": "google-drive", "is_visible_in_connector": True},
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        row = connection.execute(sa.select(table)).mappings().one()

    assert row["is_visible_in_connector"] is False


def test_upgrade_is_noop_when_catalog_columns_are_missing() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    sa.Table(
        "public_mcp_apps",
        metadata,
        sa.Column("app_id", sa.String(100), primary_key=True),
    )
    metadata.create_all(engine)

    with (
        engine.begin() as connection,
        patch.object(migration, "op", _operations(connection)),
    ):
        migration.upgrade()


def test_offline_sqlite_upgrade_round_trips_visibility() -> None:
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
        "CREATE TABLE public_mcp_apps (app_id TEXT PRIMARY KEY, is_visible_in_connector BOOLEAN)"
    )
    connection.execute("INSERT INTO public_mcp_apps VALUES (?, ?)", ("google-drive", 0))
    connection.executescript(output.getvalue())
    row = connection.execute(
        "SELECT is_visible_in_connector FROM public_mcp_apps WHERE app_id = 'google-drive'"
    ).fetchone()
    connection.close()

    assert row == (1,)


def test_revision_metadata() -> None:
    migration = _load_migration_module()

    assert migration.revision == "20260924_enable_google_drive_picker"
    assert migration.down_revision == "20260924_hide_google_drive_until_picker"
