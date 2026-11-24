"""Inc.7 — SQLite concurrency hardening (design §5.5-B1, issue #639).

Concurrent tool execution can drive concurrent writes through the shared
engine. On SQLite the default rollback journal + no busy timeout surfaces as
"database is locked". Enabling WAL (concurrent readers + one writer) and a
busy_timeout (writers wait for the lock instead of failing immediately) makes
the engine tolerate that contention. Postgres is unaffected.
"""

from __future__ import annotations

import logging

import pytest
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


def test_apply_pragmas_swallows_concurrency_pragma_failures(caplog) -> None:
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

        def fetchone(self) -> tuple[int]:
            return (1,)

    cursor = _FailWALCursor()
    with caplog.at_level(logging.WARNING):
        _apply_concurrency_pragmas(cursor, 5000)  # must not raise

    assert any("journal_mode" in sql for sql in cursor.executed)
    assert any("busy_timeout=5000" in sql for sql in cursor.executed)
    assert any("WAL" in record.message for record in caplog.records)


def test_apply_pragmas_fails_when_foreign_keys_cannot_be_enabled() -> None:
    from xagent.db.sqlite import _apply_concurrency_pragmas

    class _FailForeignKeysCursor:
        def execute(self, sql: str) -> None:
            if sql == "PRAGMA foreign_keys=ON":
                raise RuntimeError("foreign keys unavailable")

    with pytest.raises(RuntimeError, match="foreign keys unavailable"):
        _apply_concurrency_pragmas(_FailForeignKeysCursor(), 5000)


def test_apply_pragmas_fails_when_foreign_keys_remain_disabled() -> None:
    from xagent.db.sqlite import _apply_concurrency_pragmas

    class _DisabledForeignKeysCursor:
        def execute(self, _sql: str) -> None:
            pass

        def fetchone(self) -> tuple[int]:
            return (0,)

    with pytest.raises(RuntimeError, match="foreign-key enforcement"):
        _apply_concurrency_pragmas(_DisabledForeignKeysCursor(), 5000)


def test_apply_pragmas_noop_for_non_sqlite() -> None:
    from xagent.db.sqlite import apply_sqlite_concurrency_pragmas

    # create_engine does not connect, so no Postgres server is required; the
    # helper must return without registering a connect hook or raising.
    engine = create_engine("postgresql://user:pass@localhost:5432/db")
    apply_sqlite_concurrency_pragmas(engine)
    engine.dispose()


def test_write_transaction_must_be_the_first_statement(tmp_path) -> None:
    from sqlalchemy import text
    from sqlalchemy.orm import Session

    from xagent.db.sqlite import begin_sqlite_write_transaction

    engine = create_engine(f"sqlite:///{tmp_path / 'fence.db'}")
    with Session(engine) as session:
        session.execute(text("SELECT 1"))
        with pytest.raises(RuntimeError, match="first statement"):
            begin_sqlite_write_transaction(session)
    engine.dispose()


def test_write_transaction_noop_for_non_sqlite() -> None:
    from sqlalchemy.orm import Session

    from xagent.db.sqlite import begin_sqlite_write_transaction

    # create_engine does not connect; taking a connection would open a
    # transaction (and fail with no server), so none may be open afterwards.
    engine = create_engine("postgresql://user:pass@localhost:5432/db")
    with Session(engine) as session:
        begin_sqlite_write_transaction(session)
        assert not session.in_transaction()
    engine.dispose()


def test_write_transaction_takes_the_write_lock(tmp_path) -> None:
    from sqlalchemy.exc import OperationalError
    from sqlalchemy.orm import Session

    from xagent.db.sqlite import begin_sqlite_write_transaction

    # A blocked writer gives up after 0.1 seconds, not the default 5.
    engine = create_engine(
        f"sqlite:///{tmp_path / 'lock.db'}", connect_args={"timeout": 0.1}
    )
    with Session(engine) as first, Session(engine) as second:
        begin_sqlite_write_transaction(first)
        with pytest.raises(OperationalError, match="database is locked"):
            begin_sqlite_write_transaction(second)
        first.rollback()
        second.rollback()
        # Once the first session's transaction ends, the lock is free.
        begin_sqlite_write_transaction(second)
    engine.dispose()
