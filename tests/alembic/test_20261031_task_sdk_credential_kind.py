"""sdk_credential_kind on tasks and expired_task_tombstones, and its rollback."""

from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pytest
from alembic import command
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from tests.alembic.lock_retry_support import counting, driver_engines
from tests.shared.postgres_disposable import (
    disposable_database_factory,
    load_migration_module,
)
from xagent.db.config import create_alembic_config
from xagent.web.models.database import Base
from xagent.web.models.expired_task import ExpiredTaskTombstone
from xagent.web.models.task import Task

REVISION = "20261031_task_sdk_credential_kind"
DOWN_REVISION = "20261102_zendesk_side_conv"
COLUMN = "sdk_credential_kind"
TABLES = ("tasks", "expired_task_tombstones")
KEYS = {"tasks": "id", "expired_task_tombstones": "task_id"}
CONSTRAINTS = {
    "tasks": "ck_tasks_sdk_credential_kind",
    "expired_task_tombstones": "ck_expired_task_tombstones_sdk_credential_kind",
}
CONDITION = "sdk_credential_kind IS NULL OR sdk_credential_kind = 'personal'"
MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20261031_task_sdk_credential_kind.py"
)
# The agent-key scope of code that predates the column, and the personal-key
# scope of the code that reads it.
OLD_AGENT_KEY_SCOPE = "source = 'sdk' AND agent_id = 7"
PERSONAL_SCOPE = (
    "source = 'sdk' AND user_id = 1 AND sdk_credential_kind = 'personal' "
    "AND agent_id IS NOT NULL"
)


def _migration(name: str):
    return load_migration_module(MIGRATION_PATH, name)


def _create_tables(connection, dialect: str) -> None:
    """The columns these statements read, plus each table's NOT NULL ones."""
    key = "SERIAL" if dialect == "postgresql" else "INTEGER"
    tail = " AUTOINCREMENT" if dialect == "sqlite" else ""
    shared = "user_id INTEGER NOT NULL, agent_id INTEGER, source VARCHAR(20)"
    connection.execute(
        text(
            f"CREATE TABLE tasks (id {key} PRIMARY KEY{tail}, {shared}, "
            "title VARCHAR(200) NOT NULL)"
        )
    )
    connection.execute(
        text(
            f"CREATE TABLE expired_task_tombstones (task_id INTEGER PRIMARY KEY, "
            f"{shared}, is_visible BOOLEAN NOT NULL, "
            "is_channel_plumbing BOOLEAN NOT NULL, expired_at TIMESTAMP NOT NULL)"
        )
    )
    for table in TABLES:
        _insert(connection, table, 1)


def _insert(connection, table: str, row_id: int, *, source="sdk", kind=...) -> None:
    """One row in ``table``; ``kind`` is left out of the INSERT when omitted."""
    values = {KEYS[table]: row_id, "user_id": 1, "agent_id": 7, "source": source}
    if table == "tasks":
        values["title"] = "fixture"
    else:
        values.update(
            is_visible=False,
            is_channel_plumbing=False,
            expired_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        )
    if kind is not ...:
        values[COLUMN] = kind
    connection.execute(
        text(
            f"INSERT INTO {table} ({', '.join(values)}) "
            f"VALUES ({', '.join(':' + name for name in values)})"
        ),
        values,
    )


def _insert_kinds(connection) -> None:
    """Row 2 created by a personal key and row 3 by a runtime key, per table."""
    for table in TABLES:
        _insert(connection, table, 2, kind="personal")
        _insert(connection, table, 3, kind=None)


def _sources(connection, table: str) -> dict[int, str]:
    return dict(
        connection.execute(text(f"SELECT {KEYS[table]}, source FROM {table}")).all()
    )


def _assert_check_rejects_runtime(connection) -> None:
    for table in TABLES:
        with pytest.raises(IntegrityError, match=CONSTRAINTS[table]):
            with connection.begin_nested():
                _insert(connection, table, 99, kind="runtime")


def _stamp(connection) -> None:
    connection.execute(
        text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
    )
    connection.execute(
        text("INSERT INTO alembic_version (version_num) VALUES (:revision)"),
        {"revision": DOWN_REVISION},
    )


def test_sqlite_upgrade_adds_checked_columns_and_downgrade_rewrites_then_drops() -> (
    None
):
    engine = create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)

    with engine.begin() as connection:
        _stamp(connection)
        _create_tables(connection, "sqlite")
        config.attributes["connection"] = connection

        command.upgrade(config, REVISION)

        for table in TABLES:
            column = {c["name"]: c for c in inspect(connection).get_columns(table)}
            assert COLUMN in column, table
            assert column[COLUMN]["nullable"] is True
            assert column[COLUMN]["default"] is None
            assert str(column[COLUMN]["type"]) == "VARCHAR(16)"
            stored = connection.execute(text(f"SELECT {COLUMN} FROM {table}")).all()
            assert stored == [(None,)]
        _assert_check_rejects_runtime(connection)
        # One ADD COLUMN: a batch rebuild would have dropped this.
        tasks_sql = connection.execute(
            text("SELECT sql FROM sqlite_master WHERE name = 'tasks'")
        ).scalar_one()
        assert "AUTOINCREMENT" in tasks_sql

        # Running the revision body again, as after create_all, is a no-op.
        with Operations.context(MigrationContext.configure(connection)):
            _migration("kind_twice").upgrade()

        _insert_kinds(connection)
        command.downgrade(config, DOWN_REVISION)

        for table in TABLES:
            columns = {c["name"] for c in inspect(connection).get_columns(table)}
            assert COLUMN not in columns
            assert _sources(connection, table) == {1: "sdk", 2: "internal", 3: "sdk"}


def test_model_columns_match_the_migration() -> None:
    migration = _migration("kind_model")
    assert migration.CONSTRAINTS == CONSTRAINTS
    assert migration.CHECK_CONDITION == CONDITION
    for model in (Task, ExpiredTaskTombstone):
        assert COLUMN in model.__table__.c, model.__tablename__
        column = model.__table__.c[COLUMN]
        assert column.nullable and column.default is None
        assert column.server_default is None
        assert type(column.type).__name__ == "String" and column.type.length == 16
        checks = {c.name: str(c.sqltext) for c in column.constraints}
        assert checks == {CONSTRAINTS[model.__tablename__]: CONDITION}

    built = create_engine("sqlite:///:memory:")
    with built.begin() as connection:
        Base.metadata.create_all(connection)
        _assert_check_rejects_runtime(connection)


def test_rollback_rewrite_hides_personal_tasks_from_older_scopes() -> None:
    migration = _migration("kind_rewrite")
    engine = create_engine("sqlite:///:memory:")

    def matching(connection, where: str) -> dict[str, set[int]]:
        return {
            table: set(
                connection.execute(
                    text(f"SELECT {KEYS[table]} FROM {table} WHERE {where}")
                ).scalars()
            )
            for table in TABLES
        }

    with engine.begin() as connection:
        Base.metadata.create_all(connection)
        _insert_kinds(connection)
        assert matching(connection, OLD_AGENT_KEY_SCOPE) == {
            table: {2, 3} for table in TABLES
        }

        # Repeatable: the second run changes nothing more.
        for _ in range(2):
            for table in TABLES:
                connection.execute(text(migration.ROLLBACK_REWRITE_SQL[table]))
        assert matching(connection, OLD_AGENT_KEY_SCOPE) == {
            table: {3} for table in TABLES
        }
        assert matching(connection, PERSONAL_SCOPE) == {
            table: set() for table in TABLES
        }


def _offline_sql(dialect: str, operation: str) -> str:
    migration = _migration(f"kind_off_{dialect}")
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name=dialect, opts={"as_sql": True, "output_buffer": output}
    )
    with Operations.context(context):
        getattr(migration, operation)()
    return output.getvalue()


def _pg_statements(operation: str) -> list[tuple[str, str]]:
    """(statement, table) in the order the revision runs them on PostgreSQL."""
    if operation == "upgrade":
        return [
            pair
            for table in TABLES
            for pair in (
                (f"ALTER TABLE {table} ADD COLUMN {COLUMN} VARCHAR(16)", table),
                (
                    f"ALTER TABLE {table} ADD CONSTRAINT {CONSTRAINTS[table]} "
                    f"CHECK ({CONDITION}) NOT VALID",
                    table,
                ),
            )
        ]
    rewrites = _migration("kind_statements").ROLLBACK_REWRITE_SQL
    return [(rewrites[table], table) for table in TABLES] + [
        pair
        for table in TABLES
        for pair in (
            (
                f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {CONSTRAINTS[table]}",
                table,
            ),
            (f"ALTER TABLE {table} DROP COLUMN {COLUMN}", table),
        )
    ]


@pytest.mark.parametrize("operation", ["upgrade", "downgrade"])
def test_offline_postgresql_runs_each_statement_under_a_lock_timeout(
    operation,
) -> None:
    sql = _offline_sql("postgresql", operation)

    positions = []
    for statement, _table in _pg_statements(operation):
        assert sql.count(statement) == 1, statement
        positions.append(sql.index(statement))
        assert sql.rfind("SET lock_timeout = '5s'", 0, positions[-1]) != -1
        assert sql.index("RESET lock_timeout", positions[-1]) != -1
    assert positions == sorted(positions)
    assert sql.count("SET lock_timeout = '5s'") == len(positions)


def test_offline_sqlite_adds_the_check_with_the_column_and_rewrites_first() -> None:
    upgrade = " ".join(_offline_sql("sqlite", "upgrade").split())
    for table in TABLES:
        assert (
            f"ALTER TABLE {table} ADD COLUMN {COLUMN} VARCHAR(16) "
            f"CONSTRAINT {CONSTRAINTS[table]} CHECK ({CONDITION})"
        ) in upgrade
    downgrade = _offline_sql("sqlite", "downgrade")
    rewrites = _migration("kind_offline_rewrite").ROLLBACK_REWRITE_SQL
    assert max(downgrade.index(sql) for sql in rewrites.values()) < downgrade.index(
        "DROP COLUMN"
    )


# --- PostgreSQL ---------------------------------------------------------------


@pytest.fixture
def pg_factory():
    with disposable_database_factory("xagent_sdk_credential_kind") as make:
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


def _run(migration, connection, operation: str) -> None:
    connection.commit()
    context = MigrationContext.configure(
        connection, opts={"transaction_per_migration": True}
    )
    with Operations.context(context):
        with context.begin_transaction(_per_migration=True):
            getattr(migration, operation)()
    connection.commit()


def _pg_constraints(connection) -> dict[str, tuple[bool, str]]:
    rows = connection.execute(
        text(
            "SELECT conname, convalidated, pg_catalog.pg_get_constraintdef(oid) "
            "FROM pg_catalog.pg_constraint WHERE conname = ANY(:names)"
        ),
        {"names": list(CONSTRAINTS.values())},
    ).all()
    connection.commit()
    return {name: (valid, definition) for name, valid, definition in rows}


@pytest.mark.postgresql
def test_postgresql_upgrade_and_downgrade_go_through_the_lock_retry_helper(
    pg_engine, monkeypatch
) -> None:
    migration = _migration("kind_pg")
    calls: list[tuple[str, str]] = []
    real = migration.run_ddl_with_lock_retry

    def recording(statement, *, table, **kwargs):
        calls.append((statement, table))
        return real(statement, table=table, **kwargs)

    monkeypatch.setattr(migration, "run_ddl_with_lock_retry", recording)
    upgrade, downgrade = _pg_statements("upgrade"), _pg_statements("downgrade")

    with pg_engine.connect() as connection:
        _run(migration, connection, "upgrade")
        assert calls == upgrade
        # NOT VALID: adding it scanned nothing; new writes are still checked.
        assert {
            name: valid for name, (valid, _) in _pg_constraints(connection).items()
        } == {name: False for name in CONSTRAINTS.values()}
        _assert_check_rejects_runtime(connection)
        connection.commit()

        with counting(pg_engine, "ALTER TABLE") as alters:
            _run(migration, connection, "upgrade")
        assert alters == [] and calls == upgrade

        _insert_kinds(connection)
        _run(migration, connection, "downgrade")
        assert calls[len(upgrade) :] == downgrade
        for table in TABLES:
            columns = {c["name"] for c in inspect(connection).get_columns(table)}
            assert COLUMN not in columns
            assert _sources(connection, table) == {1: "sdk", 2: "internal", 3: "sdk"}
        assert _pg_constraints(connection) == {}

        _run(migration, connection, "downgrade")
        assert len(calls) == len(upgrade) + len(downgrade)


@pytest.mark.postgresql
def test_postgresql_check_from_create_all_and_from_the_migration_are_the_same(
    pg_factory,
) -> None:
    built = pg_factory("built")
    Base.metadata.create_all(built)
    migrated = pg_factory("migrated")
    with migrated.begin() as setup:
        _create_tables(setup, "postgresql")
    with migrated.connect() as connection:
        _run(_migration("kind_pg_parity"), connection, "upgrade")
        from_migration = _pg_constraints(connection)
    with built.connect() as connection:
        from_model = _pg_constraints(connection)

    assert set(from_model) == set(from_migration) == set(CONSTRAINTS.values())
    for name in CONSTRAINTS.values():
        assert from_model[name] == (
            True,
            from_migration[name][1].removesuffix(" NOT VALID"),
        )
