"""Bounded same-identity retries for authoritative execution-event writes.

A V2 execution fact is committed under a fixed identity, so a transient
database failure (deadlock, lock timeout, dropped connection) can be answered
by running the same write again instead of aborting the run. The drivers here
retry only the database write they are handed, and treat a retryable
failure as possibly committed:

- ``retry_db_write_async`` reruns the write as is; callers make
  that operation an idempotent rerun under the same fact identity.
- ``run_with_commit_witness_sync`` is for fenced writes (settlements), whose
  rerun after an applied COMMIT would miss its own fence. The attempt
  records the witness of the result fact it staged; after a retryable error
  the witness is read back, and only an attempt proven not to have
  committed is rerun.

Exhaustion re-raises the last error unchanged, so the callers' existing
failure handling (``ExecutionEventPersistenceError`` -> runner abort ->
settlement) is the same with retries enabled, disabled, or used up.
"""

from __future__ import annotations

import asyncio
import logging
import random
import sqlite3
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Iterator, TypeVar, cast

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

if TYPE_CHECKING:
    from .task_execution_event_writer import FactWitness

logger = logging.getLogger(__name__)

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


@dataclass
class CommitAttempt:
    """What one attempt of a witnessed write leaves for its retry driver.

    ``witness`` is the fact row this attempt's transaction inserted (from
    ``stage_result_fact_no_commit``); it stands for every write in that
    atomic transaction. ``confirm`` is what the attempt does after its
    COMMIT: the driver calls it instead when a lost acknowledgement's COMMIT
    is proven applied, so set it before committing.
    """

    number: int = 1
    witness: FactWitness | None = None
    confirm: Callable[[], Any] | None = None

    @property
    def rerun(self) -> bool:
        """Whether an earlier attempt failed; its transaction must bound its
        lock waits (:func:`bound_rerun_lock_wait_no_commit`)."""
        return self.number > 1


def run_with_commit_witness_sync(
    operation: Callable[[CommitAttempt], T],
    *,
    site: str,
    task_id: int,
    run_id: str | None,
    policy: DbWriteRetryPolicy | None = None,
) -> T:
    """Run a witnessed write, confirming or rerunning it after a lost COMMIT.

    For writes that cannot simply be rerun because they make a fenced state
    transition (a settlement): a rerun after an applied COMMIT would miss its
    own fence and report the outcome as someone else's. So, after a
    retryable error, an attempt that inserted a witness is read back first
    (``fact_witness_committed``, which waits for the task row lock):

    - found: the COMMIT applied. The attempt's ``confirm`` is returned, so
      the caller reports its own committed outcome once.
    - absent: nothing applied; the operation is rerun in a new attempt,
      within the policy's bounds.
    - the read fails: the outcome is undecided and the error is re-raised.

    An error that is not retryable, or one from an attempt without a witness
    (a legacy task, a replayed fact, or a failure before the fact was
    staged), is re-raised unchanged, as without this driver. Every error
    re-raised is the operation's own, cause chain intact. Runs in a worker
    thread: the backoff sleeps it.

    Operator signals: ``site`` labels the retry counter, as in
    :func:`retry_db_write_async`. ``settlement_ack_reconciled`` (INFO) is
    reported when a read-back proves a COMMIT applied, and
    ``settlement_ack_unresolved`` when the read-back fails after a dropped
    connection (``connection_was_lost``), the one error that can hide an
    applied COMMIT.
    """
    from .execution_event_signals import report_unknown_outcome
    from .task_execution_event_writer import fact_witness_committed

    policy = policy or DbWriteRetryPolicy.from_config()
    number = 0
    while True:
        number += 1
        attempt = CommitAttempt(number=number)
        try:
            result = operation(attempt)
        except Exception as exc:
            witness = attempt.witness
            if witness is None or not is_retryable_database_error(exc):
                if number > 1:
                    _count(site, "failed")
                raise
            committed: bool | None
            try:
                committed = fact_witness_committed(witness)
            except Exception:
                logger.warning(
                    "Task %s %s commit read-back failed", task_id, site, exc_info=True
                )
                committed = None
            if committed is None:
                if connection_was_lost(exc):
                    report_unknown_outcome(
                        "settlement_ack_unresolved",
                        task_id=task_id,
                        run_id=run_id,
                        storage="v2",
                    )
                if number > 1:
                    _count(site, "failed")
                raise
            if committed and attempt.confirm is not None:
                report_unknown_outcome(
                    "settlement_ack_reconciled",
                    task_id=task_id,
                    run_id=run_id,
                    storage="v2",
                    level=logging.INFO,
                )
                if number > 1:
                    _count(site, "recovered")
                return cast(T, attempt.confirm())
            if committed or number >= policy.attempts_for(exc):
                # ``committed`` without ``confirm``: an attempt that committed
                # outside its contract; never claim an outcome for it.
                if number > 1:
                    _count(site, "failed")
                raise
            _count(site, "retried")
            time.sleep(policy.delay(number))
        else:
            if number > 1:
                _count(site, "recovered")
            return result
