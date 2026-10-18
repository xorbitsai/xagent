"""dispatch_class column on task_execution_commands.

The column is added with a constant default; on PostgreSQL the ALTER takes a
short ACCESS EXCLUSIVE lock, so the migration runs under a session
``lock_timeout`` and retries a bounded number of times.
"""

from __future__ import annotations

import os
from io import StringIO
from pathlib import Path

import pytest
from alembic import command
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import make_url

from tests.shared.postgres_disposable import (
    disposable_database_factory,
    load_migration_module,
)
from xagent.db.config import create_alembic_config
from xagent.web.models.database import Base
from xagent.web.models.task_command import TaskExecutionCommand

REVISION = "20261017_task_cmd_dispatch_class"
DOWN_REVISION = "20261016_email_session_merge"
TABLE = "task_execution_commands"
COLUMN = "dispatch_class"
MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20261017_task_command_dispatch_class.py"
)


def _create_legacy_table(connection) -> None:
    connection.execute(
        text(
            f"CREATE TABLE {TABLE} ("
            "id INTEGER PRIMARY KEY, task_id INTEGER NOT NULL, "
            "kind VARCHAR(32) NOT NULL, status VARCHAR(16) NOT NULL)"
        )
    )
    connection.execute(
        text(
            f"INSERT INTO {TABLE} (id, task_id, kind, status) VALUES (1, 1, 'start', 'pending')"
        )
    )


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


def test_sqlite_upgrade_adds_the_column_twice_safe_and_downgrade_drops_it() -> None:
    engine = create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)

    with engine.begin() as connection:
        _stamp(connection)
        _create_legacy_table(connection)
        config.attributes["connection"] = connection

        command.upgrade(config, REVISION)

        column = _columns(connection)[COLUMN]
        assert column["nullable"] is False
        assert "batch" in str(column["default"])
        assert (
            connection.execute(
                text(f"SELECT {COLUMN} FROM {TABLE} WHERE id = 1")
            ).scalar_one()
            == "batch"
        )

        # Running the revision body again is a no-op.
        migration = load_migration_module(MIGRATION_PATH, "dispatch_class_twice")
        context = MigrationContext.configure(connection)
        with Operations.context(context):
            migration.upgrade()
        assert COLUMN in _columns(connection)

        command.downgrade(config, DOWN_REVISION)
        assert COLUMN not in _columns(connection)


def test_sqlite_upgrade_skips_without_the_table() -> None:
    engine = create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)

    with engine.begin() as connection:
        _stamp(connection)
        config.attributes["connection"] = connection

        command.upgrade(config, REVISION)

        assert TABLE not in inspect(connection).get_table_names()


def test_create_all_builds_the_column_and_the_migration_is_then_a_noop() -> None:
    engine = create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)

    with engine.begin() as connection:
        Base.metadata.create_all(connection)
        _stamp(connection)
        config.attributes["connection"] = connection
        assert COLUMN in _columns(connection)

        command.upgrade(config, REVISION)

        assert COLUMN in _columns(connection)


def test_model_column_matches_the_migration() -> None:
    column = TaskExecutionCommand.__table__.c[COLUMN]
    assert column.nullable is False
    assert column.type.length == 16
    assert str(column.server_default.arg) == "batch"
    assert column.default.arg == "batch"


def _offline_sql(dialect: str, operation: str) -> str:
    migration = load_migration_module(MIGRATION_PATH, f"dispatch_class_off_{dialect}")
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name=dialect, opts={"as_sql": True, "output_buffer": output}
    )
    with Operations.context(context):
        getattr(migration, operation)()
    return output.getvalue()


def test_offline_postgresql_sets_a_session_lock_timeout_around_the_alter() -> None:
    sql = _offline_sql("postgresql", "upgrade")
    assert "SET lock_timeout = '5s'" in sql
    assert "SET LOCAL" not in sql
    set_at = sql.index("SET lock_timeout")
    alter_at = sql.index(f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} VARCHAR(16)")
    reset_at = sql.index("RESET lock_timeout")
    assert set_at < alter_at < reset_at
    assert "NOT NULL DEFAULT 'batch'" in sql


def test_offline_sqlite_adds_the_column_and_drops_it_on_downgrade() -> None:
    assert f"ADD COLUMN {COLUMN} VARCHAR(16)" in _offline_sql("sqlite", "upgrade")
    assert f"DROP COLUMN {COLUMN}" in _offline_sql("sqlite", "downgrade")
    assert f"DROP COLUMN {COLUMN}" in _offline_sql("postgresql", "downgrade")


@pytest.fixture
def postgresql_engine_factory():
    with disposable_database_factory("xagent_dispatch_class") as make:
        yield make


def _run(migration, connection, operation: str) -> None:
    connection.commit()
    context = MigrationContext.configure(
        connection, opts={"transaction_per_migration": True}
    )
    with Operations.context(context):
        with context.begin_transaction(_per_migration=True):
            getattr(migration, operation)()


def _create_pg_table(connection) -> None:
    connection.execute(
        text(
            f"CREATE TABLE {TABLE} ("
            "id SERIAL PRIMARY KEY, task_id INTEGER NOT NULL, "
            "kind VARCHAR(32) NOT NULL, status VARCHAR(16) NOT NULL)"
        )
    )
    connection.execute(
        text(
            f"INSERT INTO {TABLE} (task_id, kind, status) VALUES (1, 'start', 'pending')"
        )
    )
    connection.commit()


def _driver_engines(factory, tag: str):
    """One engine per installed PostgreSQL driver on the same fresh database."""
    base = make_url(os.environ["XAGENT_TEST_POSTGRES_URL"])
    first = factory(tag)
    with first.connect() as connection:
        dbname = connection.execute(text("SELECT current_database()")).scalar_one()
    engines = {"psycopg2": first}
    try:
        import psycopg  # noqa: F401
    except ImportError:
        return engines
    engines["psycopg"] = create_engine(
        base.set(drivername="postgresql+psycopg", database=dbname)
    )
    return engines


@pytest.mark.postgresql
@pytest.mark.parametrize("driver", ["psycopg2", "psycopg"])
def test_postgresql_lock_timeout_is_retried_then_reported(
    postgresql_engine_factory, monkeypatch, driver
) -> None:
    migration = load_migration_module(MIGRATION_PATH, f"dispatch_class_lock_{driver}")
    monkeypatch.setattr(migration, "_LOCK_TIMEOUT", "200ms")
    monkeypatch.setattr(migration, "_LOCK_RETRY_DELAYS", (0.01, 0.01))
    engines = _driver_engines(postgresql_engine_factory, f"lock_{driver}")
    if driver not in engines:
        pytest.skip(f"{driver} is not installed")
    engine = engines[driver]

    holder = engines["psycopg2"].connect()
    try:
        with engine.connect() as connection:
            _create_pg_table(connection)
            lock_timeout_before = connection.execute(
                text("SHOW lock_timeout")
            ).scalar_one()
            holder.execute(text(f"LOCK TABLE {TABLE} IN ACCESS SHARE MODE"))

            attempts: list[str] = []

            def record(_conn, _cursor, statement, *_args) -> None:
                if "ADD COLUMN" in statement:
                    attempts.append(statement)

            event.listen(engine, "before_cursor_execute", record)
            try:
                with pytest.raises(RuntimeError, match="could not lock"):
                    _run(migration, connection, "upgrade")
            finally:
                event.remove(engine, "before_cursor_execute", record)

            assert len(attempts) == 3
            assert connection.execute(text("SHOW lock_timeout")).scalar_one() == (
                lock_timeout_before
            )
            assert COLUMN not in _columns(connection)

            holder.rollback()
            holder.close()
            holder = None

            _run(migration, connection, "upgrade")
            assert COLUMN in _columns(connection)
            assert connection.execute(text("SHOW lock_timeout")).scalar_one() == (
                lock_timeout_before
            )
    finally:
        if holder is not None:
            holder.rollback()
            holder.close()
        if driver != "psycopg2":
            engine.dispose()


@pytest.mark.postgresql
def test_postgresql_a_non_lock_error_is_not_retried(
    postgresql_engine_factory, monkeypatch
) -> None:
    migration = load_migration_module(MIGRATION_PATH, "dispatch_class_other_error")
    monkeypatch.setattr(migration, "_LOCK_RETRY_DELAYS", (0.01, 0.01))
    engine = postgresql_engine_factory("other")
    # A column of the same name with a different shape is not our concern; an
    # unrelated failure must surface at once, so provoke one with a bad type.
    monkeypatch.setattr(
        migration, "_ADD_COLUMN_SQL", f"ALTER TABLE {TABLE} ADD COLUMN x nosuchtype"
    )
    with engine.connect() as connection:
        _create_pg_table(connection)
        lock_timeout_before = connection.execute(text("SHOW lock_timeout")).scalar_one()
        attempts: list[str] = []

        def record(_conn, _cursor, statement, *_args) -> None:
            if "ADD COLUMN" in statement:
                attempts.append(statement)

        event.listen(engine, "before_cursor_execute", record)
        try:
            with pytest.raises(Exception) as caught:
                _run(migration, connection, "upgrade")
        finally:
            event.remove(engine, "before_cursor_execute", record)
        assert not isinstance(caught.value, RuntimeError)
        assert len(attempts) == 1
        assert connection.execute(text("SHOW lock_timeout")).scalar_one() == (
            lock_timeout_before
        )
