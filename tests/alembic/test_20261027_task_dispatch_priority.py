"""dispatch priority columns on tasks and task_execution_commands, and the index."""

from __future__ import annotations

import re
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
from xagent.web.models.database import Base
from xagent.web.models.task import Task
from xagent.web.models.task_command import TaskExecutionCommand

REVISION = "20261027_task_dispatch_priority"
DOWN_REVISION = "20261026_auto_sequential_failover"
TASKS = "tasks"
COMMANDS = "task_execution_commands"
INDEX = "ix_task_commands_claimable_fair_priority"
ADD_TASK_SQL = f"ALTER TABLE {TASKS} ADD COLUMN dispatch_priority SMALLINT"
ADD_COMMAND_SQL = (
    f"ALTER TABLE {COMMANDS} ADD COLUMN priority SMALLINT NOT NULL DEFAULT 0"
)
CREATE_INDEX_SQL = (
    f"CREATE INDEX {INDEX} ON {COMMANDS} "
    "(fairness_key, dispatch_class, priority DESC, id) "
    "WHERE status IN ('pending', 'processing')"
)
DROP_INDEX_SQL = f"DROP INDEX IF EXISTS {INDEX}"
DROP_COMMAND_SQL = f"ALTER TABLE {COMMANDS} DROP COLUMN priority"
DROP_TASK_SQL = f"ALTER TABLE {TASKS} DROP COLUMN dispatch_priority"
MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20261027_task_dispatch_priority.py"
)


def _create_tables(connection, dialect: str, *, with_key: bool = True) -> None:
    serial = "SERIAL" if dialect == "postgresql" else "INTEGER"
    key = "fairness_key VARCHAR(64) NOT NULL DEFAULT '', " if with_key else ""
    connection.execute(
        text(f"CREATE TABLE {TASKS} (id {serial} PRIMARY KEY, title VARCHAR(200))")
    )
    connection.execute(
        text(
            f"CREATE TABLE {COMMANDS} (id {serial} PRIMARY KEY, "
            f"task_id INTEGER NOT NULL, status VARCHAR(32) NOT NULL, {key}"
            "dispatch_class VARCHAR(16) NOT NULL DEFAULT 'batch')"
        )
    )
    connection.execute(text(f"INSERT INTO {TASKS} (id, title) VALUES (1, 'old')"))
    connection.execute(
        text(f"INSERT INTO {COMMANDS} (id, task_id, status) VALUES (1, 1, 'pending')")
    )


def _stamp(connection) -> None:
    connection.execute(
        text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
    )
    connection.execute(
        text("INSERT INTO alembic_version (version_num) VALUES (:revision)"),
        {"revision": DOWN_REVISION},
    )


def _columns(connection, table: str) -> dict[str, dict]:
    return {c["name"]: c for c in inspect(connection).get_columns(table)}


def _index_names(connection) -> set[str]:
    return {index["name"] for index in inspect(connection).get_indexes(COMMANDS)}


def _sqlite_index_sql(connection) -> str:
    return " ".join(
        connection.execute(
            text("SELECT sql FROM sqlite_master WHERE name = :name"), {"name": INDEX}
        )
        .scalar_one()
        .split()
    )


def _run(migration, connection, operation: str) -> None:
    connection.commit()
    context = MigrationContext.configure(
        connection, opts={"transaction_per_migration": True}
    )
    with Operations.context(context):
        with context.begin_transaction(_per_migration=True):
            getattr(migration, operation)()


def test_sqlite_upgrade_adds_both_columns_and_the_index_and_downgrade_removes_them() -> (
    None
):
    engine = create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)

    with engine.begin() as connection:
        _stamp(connection)
        _create_tables(connection, "sqlite")
        config.attributes["connection"] = connection

        command.upgrade(config, REVISION)

        task_column = _columns(connection, TASKS)["dispatch_priority"]
        assert task_column["nullable"] is True
        assert task_column["default"] is None
        assert str(task_column["type"]) == "SMALLINT"
        command_column = _columns(connection, COMMANDS)["priority"]
        assert command_column["nullable"] is False
        assert str(command_column["default"]).strip("'") == "0"
        assert str(command_column["type"]) == "SMALLINT"
        old_rows = connection.execute(
            text(f"SELECT t.dispatch_priority, c.priority FROM {TASKS} t, {COMMANDS} c")
        ).one()
        assert tuple(old_rows) == (None, 0)
        # The index is built on a table that had no priority column before the
        # revision added it.
        index_sql = _sqlite_index_sql(connection)
        assert "(fairness_key, dispatch_class, priority DESC, id)" in index_sql
        assert "WHERE status IN ('pending', 'processing')" in index_sql

        # Running the revision body again, as after create_all, is a no-op.
        migration = load_migration_module(MIGRATION_PATH, "priority_twice")
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
        assert INDEX in _index_names(connection)

        command.downgrade(config, DOWN_REVISION)
        assert "dispatch_priority" not in _columns(connection, TASKS)
        assert "priority" not in _columns(connection, COMMANDS)
        assert INDEX not in _index_names(connection)


def test_sqlite_upgrade_and_downgrade_skip_without_the_tables() -> None:
    engine = create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)

    with engine.begin() as connection:
        _stamp(connection)
        config.attributes["connection"] = connection

        command.upgrade(config, REVISION)
        command.downgrade(config, DOWN_REVISION)

        assert not {TASKS, COMMANDS} & set(inspect(connection).get_table_names())


def test_sqlite_upgrade_adds_the_columns_but_no_index_without_the_key_column() -> None:
    engine = create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)

    with engine.begin() as connection:
        _stamp(connection)
        _create_tables(connection, "sqlite", with_key=False)
        config.attributes["connection"] = connection

        command.upgrade(config, REVISION)

        assert "priority" in _columns(connection, COMMANDS)
        assert INDEX not in _index_names(connection)


def test_model_columns_match_the_migration() -> None:
    task_column = Task.__table__.c["dispatch_priority"]
    assert task_column.nullable and task_column.server_default is None
    assert task_column.default is None
    command_column = TaskExecutionCommand.__table__.c["priority"]
    assert not command_column.nullable
    assert (command_column.server_default.arg, command_column.default.arg) == ("0", 0)
    for column in (task_column, command_column):
        assert type(column.type).__name__ == "SmallInteger"


def test_create_all_builds_the_same_index_and_the_migration_is_then_a_noop() -> None:
    migration = load_migration_module(MIGRATION_PATH, "priority_parity")
    assert migration.INDEX == INDEX
    model_index = next(
        i for i in TaskExecutionCommand.__table__.indexes if i.name == INDEX
    )
    for dialect in ("postgresql", "sqlite"):
        predicate = model_index.dialect_options[dialect]["where"]
        assert str(predicate) == str(migration.PREDICATE)

    built = create_engine("sqlite:///:memory:")
    with built.begin() as connection:
        Base.metadata.create_all(connection)
        from_model = _sqlite_index_sql(connection).replace(" ", "").lower()
    migrated = create_engine("sqlite:///:memory:")
    config = create_alembic_config(migrated)
    with migrated.begin() as connection:
        _stamp(connection)
        _create_tables(connection, "sqlite")
        config.attributes["connection"] = connection
        command.upgrade(config, REVISION)
        from_migration = _sqlite_index_sql(connection).replace(" ", "").lower()
    assert "priorityDESC".lower() in from_model
    assert from_model == from_migration

    with built.begin() as connection:
        _stamp(connection)
        config = create_alembic_config(built)
        config.attributes["connection"] = connection
        command.upgrade(config, REVISION)
        assert INDEX in _index_names(connection)


def _offline_sql(dialect: str, operation: str) -> str:
    migration = load_migration_module(MIGRATION_PATH, f"priority_off_{dialect}")
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name=dialect, opts={"as_sql": True, "output_buffer": output}
    )
    with Operations.context(context):
        getattr(migration, operation)()
    return output.getvalue()


@pytest.mark.parametrize(
    "operation,statements",
    [
        ("upgrade", [ADD_TASK_SQL, ADD_COMMAND_SQL, CREATE_INDEX_SQL]),
        ("downgrade", [DROP_INDEX_SQL, DROP_COMMAND_SQL, DROP_TASK_SQL]),
    ],
)
def test_offline_postgresql_runs_each_statement_under_a_lock_timeout(
    operation, statements
) -> None:
    sql = _offline_sql("postgresql", operation)

    positions = []
    for statement in statements:
        assert sql.count(statement) == 1
        positions.append(sql.index(statement))
        # A session lock timeout, set and reset around every statement.
        assert sql.rfind("SET lock_timeout = '5s'", 0, positions[-1]) != -1
        assert sql.index("RESET lock_timeout", positions[-1]) != -1
    assert positions == sorted(positions)
    assert sql.count("SET lock_timeout = '5s'") == 3
    assert "SET LOCAL" not in sql
    assert "CONCURRENTLY" not in sql


@pytest.mark.parametrize("operation", ["upgrade", "downgrade"])
def test_offline_sqlite_has_no_lock_timeout(operation) -> None:
    sql = _offline_sql("sqlite", operation)

    assert "lock_timeout" not in sql
    expected = {
        "upgrade": [
            "ADD COLUMN dispatch_priority SMALLINT",
            "ADD COLUMN priority SMALLINT DEFAULT '0' NOT NULL",
            f"CREATE INDEX {INDEX} ON {COMMANDS} "
            "(fairness_key, dispatch_class, priority DESC, id) "
            "WHERE status IN ('pending', 'processing')",
        ],
        "downgrade": [
            f"DROP INDEX {INDEX}",
            "DROP COLUMN priority",
            "DROP COLUMN dispatch_priority",
        ],
    }
    assert all(part in sql for part in expected[operation])


# --- PostgreSQL ---------------------------------------------------------------


@pytest.fixture
def pg_factory():
    with disposable_database_factory("xagent_dispatch_priority") as make:
        yield make


@pytest.fixture(params=["psycopg2", "psycopg"])
def pg_engine(request, pg_factory):
    engines = driver_engines(pg_factory, request.param)
    with engines["psycopg2"].begin() as setup:
        _create_tables(setup, "postgresql")
    try:
        yield engines[request.param]
    finally:
        if request.param != "psycopg2":
            engines[request.param].dispose()


def _index_state(connection) -> tuple[bool, str] | None:
    row = connection.execute(
        text(
            "SELECT i.indisvalid, pg_catalog.pg_get_indexdef(i.indexrelid) "
            "FROM pg_catalog.pg_index AS i "
            "WHERE i.indexrelid = pg_catalog.to_regclass(:name)"
        ),
        {"name": INDEX},
    ).one_or_none()
    connection.commit()
    return None if row is None else (row[0], row[1])


@pytest.mark.postgresql
def test_postgresql_upgrade_skip_and_downgrade_go_through_the_lock_retry_helper(
    pg_engine, monkeypatch
) -> None:
    migration = load_migration_module(MIGRATION_PATH, "priority_pg")
    calls: list[tuple[str, str]] = []
    real = migration.run_ddl_with_lock_retry

    def recording(statement, *, table, **kwargs):
        calls.append((statement, table))
        return real(statement, table=table, **kwargs)

    monkeypatch.setattr(migration, "run_ddl_with_lock_retry", recording)

    with pg_engine.connect() as connection:
        _run(migration, connection, "upgrade")
        assert _columns(connection, TASKS)["dispatch_priority"]["nullable"] is True
        command_column = _columns(connection, COMMANDS)["priority"]
        assert command_column["nullable"] is False
        assert "0" in str(command_column["default"])
        valid, definition = _index_state(connection)
        assert valid is True
        assert "(fairness_key, dispatch_class, priority DESC, id)" in definition
        assert calls == [
            (ADD_TASK_SQL, TASKS),
            (ADD_COMMAND_SQL, COMMANDS),
            (CREATE_INDEX_SQL, COMMANDS),
        ]

        # Everything exists already: no statement runs.
        with counting(pg_engine, "ALTER TABLE") as alters:
            with counting(pg_engine, "CREATE INDEX") as creates:
                _run(migration, connection, "upgrade")
        assert alters == [] and creates == []
        assert len(calls) == 3

        _run(migration, connection, "downgrade")
        assert "dispatch_priority" not in _columns(connection, TASKS)
        assert "priority" not in _columns(connection, COMMANDS)
        assert _index_state(connection) is None
        assert calls[3:] == [
            (DROP_INDEX_SQL, COMMANDS),
            (DROP_COMMAND_SQL, COMMANDS),
            (DROP_TASK_SQL, TASKS),
        ]

        # Everything is gone already: downgrade is a no-op.
        _run(migration, connection, "downgrade")
        assert len(calls) == 6


@pytest.mark.postgresql
def test_postgresql_index_from_create_all_and_from_the_migration_are_the_same(
    pg_factory, monkeypatch
) -> None:
    built = pg_factory("built")
    Base.metadata.create_all(built)
    migrated = pg_factory("migrated")
    with migrated.begin() as setup:
        _create_tables(setup, "postgresql")
    migration = load_migration_module(MIGRATION_PATH, "priority_pg_parity")
    with migrated.connect() as connection:
        _run(migration, connection, "upgrade")
        from_migration = _index_state(connection)
    with built.connect() as connection:
        from_model = _index_state(connection)

    def bare(state: tuple[bool, str]) -> str:
        # The schema prefix depends on the search path, not on the index.
        return re.sub(r" ON \w+\.", " ON ", state[1])

    assert from_model is not None and from_migration is not None
    assert from_model[0] is True and from_migration[0] is True
    assert "priority DESC" in bare(from_model)
    assert bare(from_model) == bare(from_migration)
