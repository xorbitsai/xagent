"""Inc.7 — SQLite concurrency hardening (design §5.5-B1, issue #639).

Concurrent tool execution can drive concurrent writes through the shared
engine. On SQLite the default rollback journal + no busy timeout surfaces as
"database is locked". Enabling WAL (concurrent readers + one writer) and a
busy_timeout (writers wait for the lock instead of failing immediately) makes
the engine tolerate that contention. Postgres is unaffected.
"""

from __future__ import annotations

import logging

from sqlalchemy import create_engine


def test_apply_pragmas_enables_wal_busy_timeout_and_foreign_keys(tmp_path) -> None:
    from xagent.db.sqlite import apply_sqlite_concurrency_pragmas

    engine = create_engine(f"sqlite:///{tmp_path / 'wal.db'}")
    apply_sqlite_concurrency_pragmas(engine, busy_timeout_ms=4000)

    with engine.connect() as conn:
        journal_mode = conn.exec_driver_sql("PRAGMA journal_mode").scalar()
        busy_timeout = conn.exec_driver_sql("PRAGMA busy_timeout").scalar()
        foreign_keys = conn.exec_driver_sql("PRAGMA foreign_keys").scalar()

    assert str(journal_mode).lower() == "wal"
    assert busy_timeout == 4000
    assert foreign_keys == 1
    engine.dispose()


def test_apply_pragmas_default_busy_timeout(tmp_path) -> None:
    from xagent.db.sqlite import apply_sqlite_concurrency_pragmas

    engine = create_engine(f"sqlite:///{tmp_path / 'default.db'}")
    apply_sqlite_concurrency_pragmas(engine)

    with engine.connect() as conn:
        busy_timeout = conn.exec_driver_sql("PRAGMA busy_timeout").scalar()

    assert busy_timeout and busy_timeout > 0
    engine.dispose()


def test_apply_pragmas_swallows_pragma_failures(caplog) -> None:
    # A read-only database/directory makes ``PRAGMA journal_mode=WAL`` raise (the
    # -wal/-shm sidecars cannot be created). The connect hook must degrade with a
    # warning instead of crashing every connection, and still attempt the
    # connection-local busy_timeout.
    from xagent.db.sqlite import _apply_concurrency_pragmas

    class _FailWALCursor:
        def __init__(self) -> None:
            self.executed: list[str] = []

        def execute(self, sql: str) -> None:
            self.executed.append(sql)
            if "journal_mode" in sql:
                raise RuntimeError("attempt to write a readonly database")

    cursor = _FailWALCursor()
    with caplog.at_level(logging.WARNING):
        _apply_concurrency_pragmas(cursor, 5000)  # must not raise

    assert any("journal_mode" in sql for sql in cursor.executed)
    assert any("busy_timeout=5000" in sql for sql in cursor.executed)
    assert any("WAL" in record.message for record in caplog.records)


def test_apply_pragmas_noop_for_non_sqlite() -> None:
    from xagent.db.sqlite import apply_sqlite_concurrency_pragmas

    # create_engine does not connect, so no Postgres server is required; the
    # helper must return without registering a connect hook or raising.
    engine = create_engine("postgresql://user:pass@localhost:5432/db")
    apply_sqlite_concurrency_pragmas(engine)
    engine.dispose()
