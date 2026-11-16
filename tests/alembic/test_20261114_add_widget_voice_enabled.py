"""The widget voice switch migration on agents and deployments."""

import importlib.util
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20261114_add_widget_voice_enabled.py"
)
TABLES = ("agents", "deployments")
COLUMN = "widget_voice_enabled"


def _migration():
    spec = importlib.util.spec_from_file_location(
        "add_widget_voice_enabled_migration", MIGRATION_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run(connection, step) -> None:
    with Operations.context(MigrationContext.configure(connection)):
        step()


def _legacy_schema(connection, *, with_column: bool = False) -> None:
    extra = f", {COLUMN} BOOLEAN NOT NULL DEFAULT 0" if with_column else ""
    for table in TABLES:
        connection.exec_driver_sql(
            f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, name VARCHAR(50){extra})"
        )
        connection.exec_driver_sql(f"INSERT INTO {table} (id, name) VALUES (1, 'a')")


def _has_column(connection, table: str) -> bool:
    return COLUMN in {c["name"] for c in sa.inspect(connection).get_columns(table)}


def test_upgrade_downgrade_upgrade_round_trip() -> None:
    migration = _migration()
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        _legacy_schema(connection)
        _run(connection, migration.upgrade)
        for table in TABLES:
            assert _has_column(connection, table)
            # Existing rows read false; new rows default to false too.
            connection.exec_driver_sql(f"INSERT INTO {table} (id) VALUES (2)")
            values = connection.exec_driver_sql(
                f"SELECT {COLUMN} FROM {table} ORDER BY id"
            ).scalars()
            assert [bool(v) for v in values] == [False, False]
            column = next(
                c
                for c in sa.inspect(connection).get_columns(table)
                if c["name"] == COLUMN
            )
            assert column["nullable"] is False
        _run(connection, migration.downgrade)
        for table in TABLES:
            assert not _has_column(connection, table)
        _run(connection, migration.upgrade)
        for table in TABLES:
            assert _has_column(connection, table)


@pytest.mark.parametrize("state", ["column_exists", "tables_missing"])
def test_upgrade_tolerates_existing_column_and_missing_tables(state) -> None:
    migration = _migration()
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        if state == "column_exists":
            _legacy_schema(connection, with_column=True)
        _run(connection, migration.upgrade)
        _run(connection, migration.upgrade)
        tables = set(sa.inspect(connection).get_table_names())
        assert tables == (set(TABLES) if state == "column_exists" else set())
        _run(connection, migration.downgrade)
        _run(connection, migration.downgrade)
