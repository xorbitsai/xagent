"""SQLite engine hardening for concurrency and relational integrity.

Enables WAL journaling (concurrent readers alongside a single writer) and a
``busy_timeout`` (a writer waits for the lock instead of failing immediately
with "database is locked") on SQLite engines. Foreign-key enforcement is also
enabled per connection so declared relationships behave like databases that
enforce them by default.

No effect on non-SQLite engines (e.g. Postgres handles this with MVCC and
row-level locks).
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import Engine, event
from sqlalchemy.engine import make_url

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
    """Set the SQLite runtime pragmas, best-effort.

    A connect hook that raises breaks every connection, so a pragma failure must
    never propagate. On a read-only database (or a directory where the -wal/-shm
    sidecars cannot be created) ``PRAGMA journal_mode=WAL`` raises; we log and
    continue. ``busy_timeout`` is connection-local (no disk write) and is set
    independently so it still applies when WAL is unavailable.
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
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not enable SQLite foreign keys: %s", exc)
