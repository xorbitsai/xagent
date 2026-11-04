"""Tests for the workforce_runs.last_activity_at migration (preview-run reaper)."""

import importlib.util
from pathlib import Path

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20260802_add_workforce_run_last_activity_at.py"
)
TABLE = "workforce_runs"
COLUMN = "last_activity_at"


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "add_workforce_run_last_activity_at_migration", MIGRATION_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection) -> Operations:
    return Operations(MigrationContext.configure(connection))


def _create_legacy_schema(connection) -> None:
    """Migration-only schema (no create_all()) -- workforce_runs has FKs to
    all three of workforces/tasks/users, the same shape as the migration
    this one is chained after."""
    connection.exec_driver_sql("CREATE TABLE workforces (id INTEGER PRIMARY KEY)")
    connection.exec_driver_sql("CREATE TABLE tasks (id INTEGER PRIMARY KEY)")
    connection.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY)")
    connection.exec_driver_sql(
        f"""
        CREATE TABLE {TABLE} (
            id INTEGER PRIMARY KEY,
            workforce_id INTEGER REFERENCES workforces(id) ON DELETE CASCADE,
            task_id INTEGER UNIQUE REFERENCES tasks(id) ON DELETE SET NULL,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            status VARCHAR(20) NOT NULL DEFAULT 'pending',
            is_preview BOOLEAN NOT NULL DEFAULT 0,
            idempotency_key VARCHAR(128),
            snapshot JSON NOT NULL,
            created_at DATETIME,
            completed_at DATETIME
        )
        """
    )
    connection.execute(sa.text("INSERT INTO workforces (id) VALUES (7)"))
    connection.execute(sa.text("INSERT INTO users (id) VALUES (1)"))
    connection.execute(sa.text("INSERT INTO tasks (id) VALUES (100)"))
    connection.execute(
        sa.text(
            f"INSERT INTO {TABLE} "
            "(id, workforce_id, task_id, user_id, status, is_preview, snapshot) "
            "VALUES (1, 7, 100, 1, 'running', 1, '{}')"
        )
    )


def _column_names(connection) -> set[str]:
    return {column["name"] for column in sa.inspect(connection).get_columns(TABLE)}


def test_upgrade_adds_a_nullable_timestamp_column_backfilled_for_existing_rows() -> (
    None
):
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_legacy_schema(connection)
        with Operations.context(_operations(connection).get_context()):
            migration.upgrade()

        assert COLUMN in _column_names(connection)

        # server_default=func.now() backfills the pre-existing row rather
        # than leaving it NULL -- the reaper's COALESCE fallback exists for
        # defense in depth, not because this is expected to be NULL.
        value = connection.execute(
            sa.text(f"SELECT {COLUMN} FROM {TABLE} WHERE id = 1")
        ).scalar()
        assert value is not None


def test_upgrade_is_idempotent() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_legacy_schema(connection)
        with Operations.context(_operations(connection).get_context()):
            migration.upgrade()
            migration.upgrade()

        assert COLUMN in _column_names(connection)


def test_downgrade_drops_the_column() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_legacy_schema(connection)
        operations = _operations(connection)
        with Operations.context(operations.get_context()):
            migration.upgrade()
        with Operations.context(operations.get_context()):
            migration.downgrade()

        assert COLUMN not in _column_names(connection)


def test_downgrade_is_idempotent_when_column_already_absent() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_legacy_schema(connection)
        with Operations.context(_operations(connection).get_context()):
            migration.downgrade()

        assert COLUMN not in _column_names(connection)


def test_upgrade_and_downgrade_skip_when_table_missing() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        with Operations.context(_operations(connection).get_context()):
            migration.upgrade()
            migration.downgrade()

        assert TABLE not in sa.inspect(connection).get_table_names()
