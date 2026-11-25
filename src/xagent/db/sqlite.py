"""SQLite engine hardening for concurrency and relational integrity.

Enables WAL journaling (concurrent readers alongside a single writer) and a
``busy_timeout`` (a writer waits for the lock instead of failing immediately
with "database is locked") on SQLite engines. Foreign-key enforcement is also
enabled per connection so declared relationships behave like databases that
enforce them by default.

No effect on non-SQLite engines (e.g. Postgres handles this with MVCC and
row-level locks).

:func:`begin_sqlite_write_transaction` takes SQLite's write lock before a
transaction reads anything, for code that reads, decides and then writes:
SQLite ignores ``SELECT ... FOR UPDATE``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import Engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# 5s gives concurrent writers room to wait out a short-lived write lock without
# unbounded blocking.
DEFAULT_BUSY_TIMEOUT_MS = 5000


def ensure_sqlite_parent_directory(database_url: str) -> str:
    """Create the SQLite database file's parent directory if it is missing.

    sqlite3 creates a missing database file on connect but not missing parent
    directories: on a fresh install the default ``~/.xagent`` storage root does
    not exist yet, so the first connection fails with "unable to open database
    file". Call this before creating an engine for a file-backed SQLite URL.

    Returns the URL to hand to ``create_engine``: sqlite3 does not expand
    ``~`` (it would open a literal ``./~/...`` path), so a ``~``-prefixed
    database path is rewritten to its expanded absolute form — the same path
    whose parent was just created. Other URLs are returned unchanged;
    non-SQLite, in-memory, and ``file:`` URI databases are ignored.
    """
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite":
        return database_url
    database = url.database
    if not database or database == ":memory:" or database.startswith("file:"):
        return database_url
    path = Path(database).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(path) != database:
        return str(url.set(database=str(path)))
    return database_url


def apply_sqlite_concurrency_pragmas(
    engine: Engine, *, busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS
) -> None:
    """Register SQLite WAL, busy-timeout, and foreign-key connection pragmas.

    Args:
        engine: The SQLAlchemy engine to harden. Non-SQLite engines are ignored.
        busy_timeout_ms: How long (ms) a blocked writer waits for the lock.
    """
    if engine.dialect.name != "sqlite":
        return

    timeout_ms = max(0, int(busy_timeout_ms))

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _connection_record):  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        try:
            _apply_concurrency_pragmas(cursor, timeout_ms)
        finally:
            cursor.close()


def _apply_concurrency_pragmas(cursor, timeout_ms: int) -> None:  # type: ignore[no-untyped-def]
    """Set optional concurrency pragmas and required relational integrity.

    WAL and ``busy_timeout`` failures degrade concurrency but do not invalidate
    stored relationships, so they remain best-effort. Foreign-key enforcement
    is a required connection invariant: ORM relationships intentionally depend
    on database cascades for rows hidden from filtered collections.
    """
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not enable SQLite WAL journal_mode (the database or its "
            "directory may be read-only); continuing without it: %s",
            exc,
        )
    try:
        cursor.execute(f"PRAGMA busy_timeout={timeout_ms}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not set SQLite busy_timeout: %s", exc)
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA foreign_keys")
    enabled = cursor.fetchone()
    if enabled is None or int(enabled[0]) != 1:
        raise RuntimeError("SQLite foreign-key enforcement could not be enabled")


def begin_sqlite_write_transaction(session: Session) -> None:
    """Take SQLite's write lock before *session* reads anything.

    SQLite ignores ``SELECT ... FOR UPDATE``, and pysqlite opens a transaction
    only before the first write, so reads that decide a write are under no
    lock. ``BEGIN IMMEDIATE`` takes the database's single write lock at once
    and keeps it until the session commits or rolls back. While another
    connection holds that lock, this waits for it up to the connection's
    ``busy_timeout`` and then raises ``OperationalError`` ("database is
    locked").

    It must be the session's first statement: a session that has begun a
    transaction is refused with ``RuntimeError``. On any other dialect this
    does nothing and takes no connection.
    """
    if session.get_bind().dialect.name != "sqlite":
        return
    if session.in_transaction():
        raise RuntimeError("BEGIN IMMEDIATE must be the session's first statement")
    session.connection().exec_driver_sql("BEGIN IMMEDIATE")
