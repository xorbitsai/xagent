"""Bounded same-identity retries for authoritative execution-event writes.

A V2 execution fact is committed under a fixed identity, so a transient
database failure (deadlock, lock timeout, dropped connection) can be answered
by running the same write again instead of aborting the run. The driver here
retries only the database write it is handed: callers must make that
operation an idempotent rerun under the same fact identity, and must treat a
retryable failure as possibly committed.

Exhaustion re-raises the last error unchanged, so the callers' existing
failure handling (``ExecutionEventPersistenceError`` -> runner abort ->
settlement) is the same with retries enabled, disabled, or used up.
"""

from __future__ import annotations

import asyncio
import random
import sqlite3
from dataclasses import dataclass
from typing import Awaitable, Callable, Iterator, TypeVar

from sqlalchemy import exc as sa_exc
from sqlalchemy import text
from sqlalchemy.orm import Session

from ...config import (
    get_db_write_retry_attempts,
    get_db_write_retry_base_delay_ms,
    get_db_write_retry_max_delay_ms,
)
from ...core.agent.interruption import is_database_unavailable
from ...core.runtime_performance import increment_counter
from .db_runtime import is_database_pool_timeout

T = TypeVar("T")

_RETRY_COUNTER = "xagent.execution_event.write.retries"
# PostgreSQL query_canceled, which also covers statement_timeout.
_QUERY_CANCELED_SQLSTATE = "57014"
# SQLite reports BUSY only after waiting out its busy timeout, so a retry
# already costs a full timeout; one is enough.
_SQLITE_MAX_ATTEMPTS = 2
# A rerun's first statement waits for the task row lock. After a dropped
# connection the server may still hold it in an abandoned transaction, so a
# rerun waits a bounded time and then fails as a retryable lock timeout.
_RERUN_LOCK_TIMEOUT = "5s"


def cause_links(error: BaseException) -> Iterator[BaseException]:
    """Yield ``error`` and its explicit ``__cause__`` chain, cycle-safe."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__


def _sqlstate(link: BaseException) -> str | None:
    orig = getattr(link, "orig", None)
    if orig is None:
        return None
    return getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)


def database_sqlstate(error: BaseException) -> str | None:
    """The first SQLSTATE found on ``error`` or any link of its cause chain."""
    for link in cause_links(error):
        state = _sqlstate(link)
        if state:
            return str(state)
    return None


def connection_was_lost(error: BaseException) -> bool:
    """Whether ``error`` or a link of its cause chain is a dropped connection.

    The one failure that can hide an applied COMMIT: psycopg2 reports a lost
    COMMIT acknowledgement as a dropped connection (SQLAlchemy marks it
    ``connection_invalidated``) and carries no SQLSTATE, since the loss is
    client-side. Every other COMMIT error -- a deadlock, a serialization
    failure, SQLite's "database is locked", an integrity violation -- is
    the server refusing it, so the commit did not apply. This is narrower
    than :func:`is_database_unavailable`, which counts every
    ``OperationalError`` as transient. ``InterfaceError`` (the DBAPI
    connection-level class, raised for an already-closed connection) counts
    too.
    """
    return any(
        getattr(link, "connection_invalidated", False)
        or isinstance(link, sa_exc.InterfaceError)
        for link in cause_links(error)
    )


def is_retryable_database_error(error: BaseException) -> bool:
    """Whether ``error`` is a database failure worth rerunning the write for.

    This narrows the taxonomy settlement uses to decide a pause
    (:func:`is_database_unavailable`), so an in-run retry and a cross-run
    pause agree on what counts as transient. Two cases are carved out:

    - The pool checkout timeout keeps its TTL-retention path; waiting a few
      milliseconds does not free a saturated pool.
    - SQLSTATE 57014 (query canceled, including ``statement_timeout``) is a
      cancellation, not a connectivity failure, and would only repeat.

    Every retryable error must be treated as possibly committed. psycopg2
    reports a lost COMMIT acknowledgement and an aborted COMMIT identically,
    so callers confirm by an idempotent rerun or a witness read-back.
    """
    if not is_database_unavailable(error):
        return False
    if is_database_pool_timeout(error):
        return False
    return not any(
        _sqlstate(link) == _QUERY_CANCELED_SQLSTATE for link in cause_links(error)
    )


def _is_sqlite_error(error: BaseException) -> bool:
    """Whether any link in the cause chain is, or wraps, a ``sqlite3.Error``."""
    return any(
        isinstance(link, sqlite3.Error)
        or isinstance(getattr(link, "orig", None), sqlite3.Error)
        for link in cause_links(error)
    )


def bound_rerun_lock_wait_no_commit(db: Session) -> None:
    """Bound how long a rerun's transaction waits for a row lock.

    Call it first in the transaction of every attempt after the first. Only
    PostgreSQL needs it; its lock timeout (55P03) is retryable. The first
    attempt keeps its unbounded wait, which a long legitimate holder may need.
    """
    if db.get_bind().dialect.name == "postgresql":
        db.execute(
            text("SELECT set_config('lock_timeout', :lock, true)"),
            {"lock": _RERUN_LOCK_TIMEOUT},
        )


@dataclass(frozen=True)
class DbWriteRetryPolicy:
    """How many times, and how patiently, an execution-event write is retried."""

    attempts: int
    base_delay_seconds: float
    max_delay_seconds: float

    @classmethod
    def from_config(cls) -> DbWriteRetryPolicy:
        return cls(
            attempts=get_db_write_retry_attempts(),
            base_delay_seconds=get_db_write_retry_base_delay_ms() / 1000,
            max_delay_seconds=get_db_write_retry_max_delay_ms() / 1000,
        )

    def attempts_for(self, error: BaseException) -> int:
        """Total attempts allowed after ``error``, first try included.

        SQLite reports BUSY only after waiting out its 5 s busy timeout, so at
        most one retry is allowed there.
        """
        if _is_sqlite_error(error):
            return min(self.attempts, _SQLITE_MAX_ATTEMPTS)
        return self.attempts

    def delay(self, retry: int, *, rng: random.Random | None = None) -> float:
        """Full-jitter wait before retry number ``retry`` (1-based).

        Uniform in ``[0, min(max_delay, base_delay * 2 ** (retry - 1))]``.
        """
        ceiling = min(
            self.max_delay_seconds, self.base_delay_seconds * 2 ** (retry - 1)
        )
        return (rng or random).uniform(0, ceiling)


def _count(site: str, outcome: str) -> None:
    increment_counter(_RETRY_COUNTER, attributes={"site": site, "outcome": outcome})


async def retry_db_write_async(
    operation: Callable[[], Awaitable[T]],
    *,
    site: str,
    policy: DbWriteRetryPolicy | None = None,
) -> T:
    """Run ``operation`` again after a retryable database error, within bounds.

    The original exception is re-raised untouched (cause chain intact) when it
    is not retryable, when the allowed attempts are used up, or when the
    running task is already being cancelled. A cancellation during the
    backoff sleep propagates and the operation is not called again.

    ``site`` labels the retry counter. A write retried at least once ends as
    ``recovered`` or ``failed``, whatever stopped it: attempts used up, an
    error that is not retryable, or the task being cancelled.
    """
    policy = policy or DbWriteRetryPolicy.from_config()
    attempt = 0
    while True:
        attempt += 1
        try:
            result = await operation()
        except asyncio.CancelledError:
            if attempt > 1:
                _count(site, "failed")
            raise
        except Exception as exc:
            task = asyncio.current_task()
            if (
                not is_retryable_database_error(exc)
                or attempt >= policy.attempts_for(exc)
                or (task is not None and task.cancelling())
            ):
                if attempt > 1:
                    _count(site, "failed")
                raise
            _count(site, "retried")
            try:
                await asyncio.sleep(policy.delay(attempt))
            except asyncio.CancelledError:
                _count(site, "failed")
                raise
        else:
            if attempt > 1:
                _count(site, "recovered")
            return result
