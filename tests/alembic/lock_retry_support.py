"""Shared pieces for tests that run a DDL statement against a locked PostgreSQL table."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from sqlalchemy import Connection, Engine, create_engine, event, text
from sqlalchemy.engine import make_url

TABLE_LOCK_MODES = frozenset(
    {
        "ACCESS SHARE",
        "ROW SHARE",
        "ROW EXCLUSIVE",
        "SHARE UPDATE EXCLUSIVE",
        "SHARE",
        "SHARE ROW EXCLUSIVE",
        "EXCLUSIVE",
        "ACCESS EXCLUSIVE",
    }
)
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


def driver_engines(factory: Callable[[str], Engine], tag: str) -> dict[str, Engine]:
    """One engine per PostgreSQL driver on the same fresh database.

    Raises when psycopg 3 is missing: the CI job treats a skipped cell as a
    failure, and a silently shorter result would hide one driver.
    """
    import psycopg  # noqa: F401  (ImportError is the failure)

    base = make_url(os.environ["XAGENT_TEST_POSTGRES_URL"])
    first = factory(tag)
    with first.connect() as connection:
        dbname = connection.execute(text("SELECT current_database()")).scalar_one()
    return {
        "psycopg2": first,
        "psycopg": create_engine(
            base.set(drivername="postgresql+psycopg", database=dbname)
        ),
    }


@contextmanager
def held_lock(
    engine: Engine, table: str, mode: str = "ROW EXCLUSIVE"
) -> Iterator[Connection]:
    """Hold a table lock on another connection, in an open transaction.

    The default is the lock an uncommitted INSERT or UPDATE holds, which blocks
    CREATE INDEX; ACCESS SHARE, which a plain SELECT holds, does not.
    """
    if mode not in TABLE_LOCK_MODES:
        raise ValueError(f"Unknown table lock mode: {mode!r}")
    if not _IDENTIFIER.match(table):
        raise ValueError(f"Not a plain table name: {table!r}")
    holder = engine.connect()
    try:
        holder.execute(text(f"LOCK TABLE {table} IN {mode} MODE"))
        yield holder
    finally:
        holder.rollback()
        holder.close()


@contextmanager
def counting(engine: Engine, marker: str = "CREATE INDEX") -> Iterator[list[str]]:
    """Record every statement on ``engine`` whose text contains ``marker``."""
    seen: list[str] = []

    def record(_conn, _cursor, statement, *_args) -> None:
        if marker in statement:
            seen.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        yield seen
    finally:
        event.remove(engine, "before_cursor_execute", record)


@contextmanager
def release_before_nth(
    engine: Engine, held: Connection, marker: str, n: int
) -> Iterator[None]:
    """Release ``held`` just before the ``n``-th statement containing ``marker``.

    Only statements whose text contains ``marker`` are counted, so the
    statements that read and restore the lock timeout do not shift the count.
    """
    seen = 0

    def before(_conn, _cursor, statement, *_args) -> None:
        nonlocal seen
        if marker in statement:
            seen += 1
            if seen == n:
                held.rollback()

    event.listen(engine, "before_cursor_execute", before)
    try:
        yield
    finally:
        event.remove(engine, "before_cursor_execute", before)
