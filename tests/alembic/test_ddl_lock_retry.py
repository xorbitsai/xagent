"""Running one PostgreSQL DDL statement under a lock timeout, retrying lock timeouts only."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from io import StringIO

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, event, text

from tests.alembic.lock_retry_support import (
    counting,
    driver_engines,
    held_lock,
    release_before_nth,
)
from tests.shared.postgres_disposable import disposable_database_factory
from xagent.migrations.ddl_lock_retry import (
    offline_ddl_with_lock_timeout,
    run_ddl_with_lock_retry,
)

TABLE = "ddl_probe"
INDEX = "ix_ddl_probe_k"
CREATE = f"CREATE INDEX {INDEX} ON {TABLE} (k)"
FAST = {"lock_timeout": "200ms", "retry_delays": (0.01, 0.01)}


@contextmanager
def migration_context(connection) -> Iterator[None]:
    connection.commit()
    context = MigrationContext.configure(
        connection, opts={"transaction_per_migration": True}
    )
    with Operations.context(context):
        with context.begin_transaction(_per_migration=True):
            yield


def run(connection, statement: str, **kwargs) -> None:
    with migration_context(connection):
        run_ddl_with_lock_retry(statement, table=TABLE, **kwargs)


def index_exists(connection) -> bool:
    connection.commit()
    return (
        connection.execute(
            text("SELECT count(*) FROM pg_indexes WHERE indexname = :n"), {"n": INDEX}
        ).scalar_one()
        == 1
    )


@pytest.fixture
def pg_factory():
    with disposable_database_factory("xagent_ddl_retry") as make:
        yield make


@pytest.fixture(params=["psycopg2", "psycopg"])
def pg(request, pg_factory) -> Iterator[tuple[sa.Engine, sa.Engine]]:
    """(engine for the driver under test, psycopg2 engine for the lock holder)."""
    engines = driver_engines(pg_factory, request.param)
    with engines["psycopg2"].begin() as setup:
        setup.execute(
            text(f"CREATE TABLE {TABLE} (id SERIAL PRIMARY KEY, k VARCHAR(8))")
        )
    try:
        yield engines[request.param], engines["psycopg2"]
    finally:
        if request.param != "psycopg2":
            engines[request.param].dispose()


@pytest.mark.postgresql
def test_a_lock_that_is_never_released_is_retried_then_reported(pg) -> None:
    engine, holder_engine = pg
    with engine.connect() as connection, held_lock(holder_engine, TABLE):
        with counting(engine) as attempts:
            with pytest.raises(RuntimeError, match=f"could not lock {TABLE}"):
                run(connection, CREATE, **FAST)
        assert len(attempts) == 3
        assert not index_exists(connection)


@pytest.mark.postgresql
def test_a_lock_released_before_the_second_attempt_lets_it_succeed(pg) -> None:
    engine, holder_engine = pg
    with engine.connect() as connection, held_lock(holder_engine, TABLE) as held:
        with release_before_nth(engine, held, "CREATE INDEX", 2):
            with counting(engine) as attempts:
                run(connection, CREATE, **FAST)
        assert len(attempts) == 2
        assert index_exists(connection)


@pytest.mark.postgresql
def test_readers_do_not_block_the_statement(pg) -> None:
    engine, holder_engine = pg
    with engine.connect() as connection:
        with held_lock(holder_engine, TABLE, "ACCESS SHARE"):
            with counting(engine) as attempts:
                run(connection, CREATE, **FAST)
        assert len(attempts) == 1
        assert index_exists(connection)


@pytest.mark.postgresql
def test_an_error_that_is_not_a_lock_timeout_is_not_retried(pg) -> None:
    engine, _ = pg
    with engine.connect() as connection:
        with counting(engine) as attempts:
            with pytest.raises(sa.exc.DBAPIError):
                run(connection, "CREATE INDEX ix_missing ON no_such_table (k)", **FAST)
        assert len(attempts) == 1


@pytest.mark.postgresql
@pytest.mark.parametrize("outcome", ["exhausted", "succeeded", "other_error"])
def test_the_lock_timeout_before_the_call_is_restored(pg, outcome) -> None:
    engine, holder_engine = pg
    with engine.connect() as connection:
        connection.execute(text("SET lock_timeout = '1234ms'"))
        connection.commit()
        if outcome == "exhausted":
            with held_lock(holder_engine, TABLE):
                with pytest.raises(RuntimeError):
                    run(connection, CREATE, **FAST)
        elif outcome == "succeeded":
            run(connection, CREATE, **FAST)
        else:
            with pytest.raises(sa.exc.DBAPIError):
                run(connection, "CREATE INDEX ix_missing ON no_such_table (k)", **FAST)
        connection.rollback()
        assert connection.execute(text("SHOW lock_timeout")).scalar_one() == "1234ms"


@pytest.mark.postgresql
def test_a_set_the_server_refuses_still_runs_the_restore(pg) -> None:
    engine, _ = pg
    with engine.connect() as connection:
        connection.execute(text("SET lock_timeout = '1234ms'"))
        connection.commit()
        with counting(engine, "set_config") as restores:
            with pytest.raises(sa.exc.DBAPIError):
                # Well formed, but beyond what the server accepts.
                run(connection, CREATE, lock_timeout="99999999999999s")
        assert len(restores) == 1
        connection.rollback()
        assert connection.execute(text("SHOW lock_timeout")).scalar_one() == "1234ms"


@pytest.mark.parametrize(
    "bad", ["0s", "0ms", "5s; DROP TABLE x", "", "-1s", "5", "5m", "5s\n"]
)
def test_a_lock_timeout_that_is_not_a_positive_number_of_ms_or_s_is_refused(bad):
    with pytest.raises(ValueError, match="lock_timeout"):
        offline_ddl_with_lock_timeout("SELECT 1", lock_timeout=bad)
    engine = create_engine("sqlite://")
    with engine.connect() as connection:
        with pytest.raises(ValueError):
            run(connection, "SELECT 1", lock_timeout=bad)


def test_it_refuses_a_connection_that_is_not_postgresql_and_runs_nothing() -> None:
    engine = create_engine("sqlite://")
    seen: list[str] = []
    event.listen(
        engine, "before_cursor_execute", lambda _c, _cur, stmt, *_a: seen.append(stmt)
    )
    with engine.connect() as connection:
        with pytest.raises(ValueError, match="PostgreSQL"):
            run(connection, "CREATE TABLE t (id INTEGER)")
    assert seen == []


def test_offline_sql_is_refused_before_anything_is_emitted() -> None:
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql", opts={"as_sql": True, "output_buffer": output}
    )
    with Operations.context(context):
        with pytest.raises(ValueError, match="offline_ddl_with_lock_timeout"):
            run_ddl_with_lock_retry(CREATE, table=TABLE)
    assert output.getvalue() == ""


def test_offline_output_is_the_statement_between_set_and_reset_in_that_order() -> None:
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql", opts={"as_sql": True, "output_buffer": output}
    )
    with Operations.context(context):
        offline_ddl_with_lock_timeout(CREATE)
    sql = output.getvalue()
    positions = [
        sql.index(part)
        for part in ("COMMIT", "SET lock_timeout = '5s'", CREATE, "RESET lock_timeout")
    ]
    assert positions == sorted(positions)
    assert "CONCURRENTLY" not in sql
