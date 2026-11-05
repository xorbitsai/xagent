"""tasks.structured_output_record: the migration, the model column, and the
round trip of a stored record."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from alembic import command
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.types import JSON

from tests.alembic.lock_retry_support import counting
from tests.shared.postgres_disposable import (
    disposable_database_factory,
    load_migration_module,
)
from xagent.db.config import create_alembic_config
from xagent.web.models.database import Base
from xagent.web.models.task import Task

REVISION = "20261105_task_structured_output_record"
DOWN_REVISION = "20261107_split_tool_categories"
COLUMN = "structured_output_record"
MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20261105_task_structured_output_record.py"
)
ADD_COLUMN_SQL = f"ALTER TABLE tasks ADD COLUMN {COLUMN} JSON"
DROP_COLUMN_SQL = f"ALTER TABLE tasks DROP COLUMN {COLUMN}"


def _migration(name: str):
    return load_migration_module(MIGRATION_PATH, name)


def _create_tasks(connection, dialect: str) -> None:
    key = "SERIAL" if dialect == "postgresql" else "INTEGER"
    tail = " AUTOINCREMENT" if dialect == "sqlite" else ""
    connection.execute(
        text(
            f"CREATE TABLE tasks (id {key} PRIMARY KEY{tail}, "
            "title VARCHAR(200) NOT NULL)"
        )
    )
    connection.execute(text("INSERT INTO tasks (id, title) VALUES (1, 'fixture')"))


def _stamp(connection) -> None:
    connection.execute(
        text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
    )
    connection.execute(
        text("INSERT INTO alembic_version (version_num) VALUES (:revision)"),
        {"revision": DOWN_REVISION},
    )


def test_sqlite_upgrade_adds_a_nullable_json_column_and_downgrade_drops_it() -> None:
    engine = create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)

    with engine.begin() as connection:
        _stamp(connection)
        _create_tasks(connection, "sqlite")
        config.attributes["connection"] = connection

        command.upgrade(config, REVISION)

        column = {c["name"]: c for c in inspect(connection).get_columns("tasks")}[
            COLUMN
        ]
        assert column["nullable"] is True
        assert column["default"] is None
        assert connection.execute(text(f"SELECT {COLUMN} FROM tasks")).all() == [
            (None,)
        ]
        # One ADD COLUMN: a batch rebuild would have dropped this.
        tasks_sql = connection.execute(
            text("SELECT sql FROM sqlite_master WHERE name = 'tasks'")
        ).scalar_one()
        assert "AUTOINCREMENT" in tasks_sql

        # Running the revision body again, as after create_all, is a no-op.
        with Operations.context(MigrationContext.configure(connection)):
            _migration("record_twice").upgrade()

        command.downgrade(config, DOWN_REVISION)

        names = {c["name"] for c in inspect(connection).get_columns("tasks")}
        assert COLUMN not in names

        # Dropping an absent column is a no-op as well.
        with Operations.context(MigrationContext.configure(connection)):
            _migration("record_drop_twice").downgrade()


def test_model_column_matches_the_migration() -> None:
    column = Task.__table__.c[COLUMN]
    assert column.nullable and column.default is None and column.server_default is None
    assert type(column.type) is JSON
    assert column.type.none_as_null is True
    # A plain json column on PostgreSQL: the stored text keeps object key order.
    assert column.type.compile(dialect=postgresql.dialect()) == "JSON"

    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        Base.metadata.create_all(connection)
        names = {c["name"] for c in inspect(connection).get_columns("tasks")}
    assert COLUMN in names


def test_python_none_is_stored_as_sql_null_and_a_record_round_trips_in_order() -> None:
    engine = create_engine("sqlite:///:memory:")
    record = {
        "run_id": "run-1",
        "status": "ready",
        "output": {"zeta": 1, "alpha": [3, 2, 1], "mid": {"b": 1, "a": 2}},
        "reason": None,
    }
    with engine.begin() as connection:
        Base.metadata.create_all(connection)
        connection.execute(
            Task.__table__.insert().values(
                id=1, user_id=1, title="t", **{COLUMN: record}
            )
        )
        connection.execute(
            Task.__table__.insert().values(id=2, user_id=1, title="t", **{COLUMN: None})
        )
        rows = dict(
            connection.execute(
                text(f"SELECT id, {COLUMN} FROM tasks ORDER BY id")
            ).all()
        )
        assert rows[2] is None
        stored = connection.execute(
            Task.__table__.select()
            .with_only_columns(Task.__table__.c[COLUMN])
            .where(Task.__table__.c.id == 1)
        ).scalar_one()
    assert stored == record
    assert list(stored["output"]) == ["zeta", "alpha", "mid"]
    assert list(stored["output"]["mid"]) == ["b", "a"]
    assert json.loads(rows[1])["output"].keys() == record["output"].keys()


def _offline_sql(dialect: str, operation: str) -> str:
    migration = _migration(f"record_off_{dialect}_{operation}")
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name=dialect, opts={"as_sql": True, "output_buffer": output}
    )
    with Operations.context(context):
        getattr(migration, operation)()
    return output.getvalue()


@pytest.mark.parametrize(
    ("operation", "statement"),
    [("upgrade", ADD_COLUMN_SQL), ("downgrade", DROP_COLUMN_SQL)],
)
def test_offline_postgresql_runs_the_statement_under_a_lock_timeout(
    operation, statement
) -> None:
    sql = _offline_sql("postgresql", operation)

    assert sql.count(statement) == 1
    position = sql.index(statement)
    assert sql.rfind("SET lock_timeout = '5s'", 0, position) != -1
    assert sql.index("RESET lock_timeout", position) != -1


def test_offline_sqlite_adds_and_drops_the_column() -> None:
    assert f"ADD COLUMN {COLUMN} JSON" in " ".join(
        _offline_sql("sqlite", "upgrade").split()
    )
    assert f"DROP COLUMN {COLUMN}" in _offline_sql("sqlite", "downgrade")


# --- PostgreSQL ---------------------------------------------------------------


@pytest.fixture
def pg_factory():
    with disposable_database_factory("xagent_structured_output_record") as make:
        yield make


def _run(migration, connection, operation: str) -> None:
    connection.commit()
    context = MigrationContext.configure(
        connection, opts={"transaction_per_migration": True}
    )
    with Operations.context(context):
        with context.begin_transaction(_per_migration=True):
            getattr(migration, operation)()
    connection.commit()


@pytest.mark.postgresql
def test_postgresql_upgrade_and_downgrade_go_through_the_lock_retry_helper(
    pg_factory, monkeypatch
) -> None:
    engine = pg_factory("lock_retry")
    with engine.begin() as setup:
        _create_tasks(setup, "postgresql")
    migration = _migration("record_pg")
    calls: list[tuple[str, str]] = []
    real = migration.run_ddl_with_lock_retry

    def recording(statement, *, table, **kwargs):
        calls.append((statement, table))
        return real(statement, table=table, **kwargs)

    monkeypatch.setattr(migration, "run_ddl_with_lock_retry", recording)

    with engine.connect() as connection:
        _run(migration, connection, "upgrade")
        assert calls == [(ADD_COLUMN_SQL, "tasks")]
        column_type = connection.execute(
            text(
                "SELECT data_type FROM information_schema.columns "
                f"WHERE table_name = 'tasks' AND column_name = '{COLUMN}'"
            )
        ).scalar_one()
        connection.commit()
        assert column_type == "json"

        with counting(engine, "ALTER TABLE") as alters:
            _run(migration, connection, "upgrade")
        assert alters == []

        _run(migration, connection, "downgrade")
        assert calls[1:] == [(DROP_COLUMN_SQL, "tasks")]
        names = {c["name"] for c in inspect(connection).get_columns("tasks")}
        assert COLUMN not in names


@pytest.mark.postgresql
def test_postgresql_record_keeps_key_order_and_reports_the_stored_size(
    pg_factory,
) -> None:
    engine = pg_factory("round_trip")
    Base.metadata.create_all(engine)
    record = {
        "run_id": "run-1",
        "status": "ready",
        "output": {"zeta": 1, "alpha": [3, 2, 1], "mid": {"b": 1, "a": 2}},
        "reason": None,
    }
    # A compact JSON size close to 2 MiB made only of DEL, which the JSON column
    # writes as the six-character escape.
    big_output = {"text": "\x7f" * (2 * 1024 * 1024 - 64)}
    big = {"run_id": "run-2", "status": "ready", "output": big_output, "reason": None}
    with engine.begin() as connection:
        for task_id, value in ((1, record), (2, big)):
            connection.execute(
                Task.__table__.insert().values(
                    id=task_id, user_id=1, title="t", **{COLUMN: value}
                )
            )
    with engine.connect() as connection:
        stored = connection.execute(
            text(f"SELECT {COLUMN} FROM tasks WHERE id = 1")
        ).scalar_one()
        assert list(stored["output"]) == ["zeta", "alpha", "mid"]
        assert list(stored["output"]["mid"]) == ["b", "a"]
        raw = connection.execute(
            text(f"SELECT {COLUMN}::text FROM tasks WHERE id = 1")
        ).scalar_one()
        assert raw.index('"zeta"') < raw.index('"alpha"') < raw.index('"mid"')

        sizes = connection.execute(
            text(
                f"SELECT pg_column_size({COLUMN}), length({COLUMN}::text) "
                "FROM tasks WHERE id = 2"
            )
        ).one()
        # Printed so a run records the numbers; the bounds are the worst case.
        print("pg_column_size, length(::text):", tuple(sizes))
        assert sizes[1] >= 6 * (2 * 1024 * 1024 - 64)
        assert sizes[0] > 0
