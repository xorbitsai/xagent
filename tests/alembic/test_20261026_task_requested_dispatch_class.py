"""requested_dispatch_class column on tasks."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from alembic import command
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text

from tests.alembic.lock_retry_support import counting, driver_engines
from tests.shared.postgres_disposable import (
    disposable_database_factory,
    load_migration_module,
)
from xagent.db.config import create_alembic_config
from xagent.web.models.task import Task

REVISION = "20261026_task_requested_dispatch_class"
DOWN_REVISION = "20261026_task_cmd_unfinished_idx"
TABLE = "tasks"
COLUMN = "requested_dispatch_class"
ADD_SQL = f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} VARCHAR(16)"
DROP_SQL = f"ALTER TABLE {TABLE} DROP COLUMN {COLUMN}"
MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20261026_task_requested_dispatch_class.py"
)


def _create_table(connection, dialect: str) -> None:
    serial = "SERIAL" if dialect == "postgresql" else "INTEGER"
    connection.execute(
        text(f"CREATE TABLE {TABLE} (id {serial} PRIMARY KEY, title VARCHAR(200))")
    )
    connection.execute(text(f"INSERT INTO {TABLE} (id, title) VALUES (1, 'old')"))


def _stamp(connection) -> None:
    connection.execute(
        text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
    )
    connection.execute(
        text("INSERT INTO alembic_version (version_num) VALUES (:revision)"),
        {"revision": DOWN_REVISION},
    )


def _columns(connection) -> dict[str, dict]:
    return {c["name"]: c for c in inspect(connection).get_columns(TABLE)}


def _run(migration, connection, operation: str) -> None:
    connection.commit()
    context = MigrationContext.configure(
        connection, opts={"transaction_per_migration": True}
    )
    with Operations.context(context):
        with context.begin_transaction(_per_migration=True):
            getattr(migration, operation)()


def test_sqlite_upgrade_adds_a_nullable_column_and_downgrade_drops_it() -> None:
    engine = create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)

    with engine.begin() as connection:
        _stamp(connection)
        _create_table(connection, "sqlite")
        config.attributes["connection"] = connection

        command.upgrade(config, REVISION)

        column = _columns(connection)[COLUMN]
        assert column["nullable"] is True
        assert column["default"] is None
        assert (
            connection.execute(
                text(f"SELECT {COLUMN} FROM {TABLE} WHERE id = 1")
            ).scalar_one()
            is None
        )

        # Running the revision body again, as after create_all, is a no-op.
        migration = load_migration_module(MIGRATION_PATH, "requested_class_twice")
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
        assert COLUMN in _columns(connection)

        command.downgrade(config, DOWN_REVISION)
        assert COLUMN not in _columns(connection)


def test_sqlite_upgrade_and_downgrade_skip_without_the_table() -> None:
    engine = create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)

    with engine.begin() as connection:
        _stamp(connection)
        config.attributes["connection"] = connection

        command.upgrade(config, REVISION)
        command.downgrade(config, DOWN_REVISION)

        assert TABLE not in inspect(connection).get_table_names()


def test_model_column_matches_the_migration() -> None:
    column = Task.__table__.c[COLUMN]
    assert column.nullable is True
    assert column.type.length == 16
    assert column.server_default is None
    assert column.default is None


def _offline_sql(dialect: str, operation: str) -> str:
    migration = load_migration_module(MIGRATION_PATH, f"requested_class_off_{dialect}")
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name=dialect, opts={"as_sql": True, "output_buffer": output}
    )
    with Operations.context(context):
        getattr(migration, operation)()
    return output.getvalue()


@pytest.mark.parametrize(
    "dialect,operation,statement",
    [
        ("postgresql", "upgrade", ADD_SQL),
        ("postgresql", "downgrade", DROP_SQL),
        ("sqlite", "upgrade", f"ADD COLUMN {COLUMN} VARCHAR(16)"),
        ("sqlite", "downgrade", f"DROP COLUMN {COLUMN}"),
    ],
)
def test_offline_sql(dialect, operation, statement) -> None:
    sql = _offline_sql(dialect, operation)
    assert statement in sql
    if dialect == "postgresql":
        # A session lock timeout, set and reset around the ALTER.
        assert "SET LOCAL" not in sql
        set_at = sql.index("SET lock_timeout = '5s'")
        alter_at = sql.index(statement)
        reset_at = sql.index("RESET lock_timeout")
        assert set_at < alter_at < reset_at
    else:
        assert "lock_timeout" not in sql


@pytest.fixture
def pg_factory():
    with disposable_database_factory("xagent_requested_class") as make:
        yield make


@pytest.fixture(params=["psycopg2", "psycopg"])
def pg_engine(request, pg_factory):
    engines = driver_engines(pg_factory, request.param)
    with engines["psycopg2"].begin() as setup:
        _create_table(setup, "postgresql")
    try:
        yield engines[request.param]
    finally:
        if request.param != "psycopg2":
            engines[request.param].dispose()


@pytest.mark.postgresql
def test_postgresql_upgrade_skip_and_downgrade_go_through_the_lock_retry_helper(
    pg_engine, monkeypatch
) -> None:
    migration = load_migration_module(MIGRATION_PATH, "requested_class_pg")
    calls: list[tuple[str, str]] = []
    real = migration.run_ddl_with_lock_retry

    def recording(statement, *, table, **kwargs):
        calls.append((statement, table))
        return real(statement, table=table, **kwargs)

    monkeypatch.setattr(migration, "run_ddl_with_lock_retry", recording)

    with pg_engine.connect() as connection:
        _run(migration, connection, "upgrade")
        column = _columns(connection)[COLUMN]
        assert column["nullable"] is True
        assert column["default"] is None
        assert calls == [(ADD_SQL, "tasks")]

        # The column already exists: no statement runs.
        with counting(pg_engine, "ADD COLUMN") as adds:
            _run(migration, connection, "upgrade")
        assert adds == []
        assert len(calls) == 1

        _run(migration, connection, "downgrade")
        assert COLUMN not in _columns(connection)
        assert calls[-1] == (DROP_SQL, "tasks")

        # The column is already gone: downgrade is a no-op.
        _run(migration, connection, "downgrade")
        assert len(calls) == 2
