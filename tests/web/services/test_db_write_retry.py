from __future__ import annotations

import asyncio
import random
import sqlite3
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy
from sqlalchemy import exc as sa_exc

from xagent.core.agent.checkpoint import ExecutionEventPersistenceError
from xagent.web.services import db_write_retry
from xagent.web.services.db_write_retry import (
    DbWriteRetryPolicy,
    is_retryable_database_error,
    retry_db_write_async,
)

_COUNTER = "xagent.execution_event.write.retries"


class _PgOrig(Exception):
    def __init__(self, message: str, *, pgcode: str | None = None) -> None:
        super().__init__(message)
        self.pgcode = pgcode


class _SqlstateOrig(Exception):
    def __init__(self, message: str, *, sqlstate: str) -> None:
        super().__init__(message)
        self.sqlstate = sqlstate


def _operational(pgcode: str | None = None) -> sa_exc.OperationalError:
    return sa_exc.OperationalError("stmt", {}, _PgOrig("boom", pgcode=pgcode))


def _sqlite_locked() -> sa_exc.OperationalError:
    return sa_exc.OperationalError(
        "stmt", {}, sqlite3.OperationalError("database is locked")
    )


def _persistence_error_from(cause: BaseException) -> ExecutionEventPersistenceError:
    try:
        raise ExecutionEventPersistenceError("fact failed to commit") from cause
    except ExecutionEventPersistenceError as exc:
        return exc


def _invalidated() -> sa_exc.DBAPIError:
    return sa_exc.DBAPIError("stmt", {}, Exception("gone"), connection_invalidated=True)


@pytest.mark.parametrize(
    "build",
    [
        lambda: _operational("40P01"),
        lambda: _operational("40001"),
        lambda: _operational("55P03"),
        lambda: _operational("57P01"),
        lambda: _operational(None),
        _invalidated,
        lambda: sa_exc.InterfaceError("stmt", {}, Exception("closed")),
        _sqlite_locked,
        lambda: _persistence_error_from(_operational("40P01")),
    ],
    ids=[
        "deadlock",
        "serialization-failure",
        "lock-not-available",
        "admin-shutdown",
        "lost-connection",
        "connection-invalidated",
        "interface-error",
        "sqlite-locked",
        "persistence-error-from-retryable",
    ],
)
def test_retryable_errors(build) -> None:
    assert is_retryable_database_error(build()) is True


@pytest.mark.parametrize(
    "build",
    [
        lambda: _operational("57014"),
        lambda: sa_exc.OperationalError(
            "stmt", {}, _SqlstateOrig("canceled", sqlstate="57014")
        ),
        lambda: sa_exc.TimeoutError("QueuePool limit reached"),
        lambda: sa_exc.IntegrityError("stmt", {}, Exception("dup")),
        lambda: sa_exc.DataError("stmt", {}, Exception("bad value")),
        lambda: sa_exc.ProgrammingError("stmt", {}, Exception("no table")),
        lambda: _persistence_error_from(
            sa_exc.IntegrityError("stmt", {}, Exception("dup"))
        ),
        lambda: _persistence_error_from(_operational("57014")),
        lambda: RuntimeError("Execution event producer lost its task lease"),
        lambda: ValueError("nope"),
    ],
    ids=[
        "statement-timeout-pgcode",
        "statement-timeout-sqlstate",
        "pool-timeout",
        "integrity-error",
        "data-error",
        "programming-error",
        "persistence-error-from-integrity",
        "persistence-error-from-canceled",
        "lease-lost",
        "value-error",
    ],
)
def test_non_retryable_errors(build) -> None:
    assert is_retryable_database_error(build()) is False


def test_attempts_for_caps_sqlite_at_one_retry() -> None:
    policy = DbWriteRetryPolicy(attempts=5, base_delay_seconds=0, max_delay_seconds=0)

    assert policy.attempts_for(_sqlite_locked()) == 2
    assert policy.attempts_for(sqlite3.OperationalError("database is locked")) == 2
    assert policy.attempts_for(_persistence_error_from(_sqlite_locked())) == 2


def test_attempts_for_keeps_postgres_attempts() -> None:
    policy = DbWriteRetryPolicy(attempts=5, base_delay_seconds=0, max_delay_seconds=0)

    assert policy.attempts_for(_operational("40P01")) == 5


def test_attempts_for_never_raises_the_configured_attempts() -> None:
    policy = DbWriteRetryPolicy(attempts=1, base_delay_seconds=0, max_delay_seconds=0)

    assert policy.attempts_for(_sqlite_locked()) == 1
    assert policy.attempts_for(_operational("40P01")) == 1


def test_delay_is_full_jitter_below_the_capped_exponential_ceiling() -> None:
    policy = DbWriteRetryPolicy(
        attempts=6, base_delay_seconds=0.05, max_delay_seconds=0.3
    )
    rng = random.Random(1234)

    for retry in range(1, 6):
        ceiling = min(0.3, 0.05 * 2 ** (retry - 1))
        samples = [policy.delay(retry, rng=rng) for _ in range(50)]
        assert all(0 <= sample <= ceiling for sample in samples)
        assert max(samples) > 0


def test_delay_is_zero_without_a_base_delay() -> None:
    policy = DbWriteRetryPolicy(attempts=3, base_delay_seconds=0, max_delay_seconds=1)

    assert policy.delay(1, rng=random.Random(0)) == 0
    assert policy.delay(4, rng=random.Random(0)) == 0


def test_from_config_converts_milliseconds_to_seconds(monkeypatch) -> None:
    monkeypatch.setenv("XAGENT_DB_WRITE_RETRY_ATTEMPTS", "4")
    monkeypatch.setenv("XAGENT_DB_WRITE_RETRY_BASE_DELAY_MS", "20")
    monkeypatch.setenv("XAGENT_DB_WRITE_RETRY_MAX_DELAY_MS", "500")

    assert DbWriteRetryPolicy.from_config() == DbWriteRetryPolicy(
        attempts=4, base_delay_seconds=0.02, max_delay_seconds=0.5
    )


@pytest.fixture
def counters(monkeypatch) -> list[tuple[str, dict[str, str]]]:
    recorded: list[tuple[str, dict[str, str]]] = []

    def record(metric: str, *, amount: int = 1, attributes=None) -> None:
        recorded.append((metric, dict(attributes or {})))

    monkeypatch.setattr(db_write_retry, "increment_counter", record)
    return recorded


def _outcomes(counters: list[tuple[str, dict[str, str]]]) -> list[str]:
    assert all(metric == _COUNTER for metric, _ in counters)
    return [attributes["outcome"] for _, attributes in counters]


_FAST = DbWriteRetryPolicy(attempts=3, base_delay_seconds=0, max_delay_seconds=0)


class _FlakyOperation:
    def __init__(self, failures: list[BaseException], result: Any = "ok") -> None:
        self._failures = list(failures)
        self._result = result
        self.calls = 0

    async def __call__(self) -> Any:
        self.calls += 1
        if self._failures:
            raise self._failures.pop(0)
        return self._result


@pytest.mark.asyncio
async def test_retryable_failure_then_success(counters) -> None:
    operation = _FlakyOperation([_operational("40P01")], result="committed")

    result = await retry_db_write_async(operation, site="test", policy=_FAST)

    assert result == "committed"
    assert operation.calls == 2
    assert counters == [
        (_COUNTER, {"site": "test", "outcome": "retried"}),
        (_COUNTER, {"site": "test", "outcome": "recovered"}),
    ]


@pytest.mark.asyncio
async def test_first_attempt_success_is_not_counted(counters) -> None:
    operation = _FlakyOperation([])

    assert await retry_db_write_async(operation, site="test", policy=_FAST) == "ok"

    assert operation.calls == 1
    assert counters == []


@pytest.mark.asyncio
async def test_non_retryable_error_is_raised_unchanged(counters) -> None:
    error = sa_exc.IntegrityError("stmt", {}, Exception("dup"))
    operation = _FlakyOperation([error])

    with pytest.raises(sa_exc.IntegrityError) as caught:
        await retry_db_write_async(operation, site="test", policy=_FAST)

    assert caught.value is error
    assert operation.calls == 1
    assert counters == []


@pytest.mark.asyncio
async def test_exhaustion_raises_the_last_error_with_its_cause(counters) -> None:
    cause = _operational("40P01")
    last = _persistence_error_from(cause)
    operation = _FlakyOperation([_operational("40P01"), _operational("40P01"), last])

    with pytest.raises(ExecutionEventPersistenceError) as caught:
        await retry_db_write_async(operation, site="test", policy=_FAST)

    assert caught.value is last
    assert caught.value.__cause__ is cause
    assert operation.calls == 3
    assert _outcomes(counters) == ["retried", "retried", "failed"]


@pytest.mark.asyncio
async def test_sqlite_error_is_retried_once(counters) -> None:
    operation = _FlakyOperation([_sqlite_locked() for _ in range(5)])

    with pytest.raises(sa_exc.OperationalError):
        await retry_db_write_async(operation, site="test", policy=_FAST)

    assert operation.calls == 2
    assert _outcomes(counters) == ["retried", "failed"]


@pytest.mark.asyncio
async def test_single_attempt_policy_disables_retries(counters) -> None:
    policy = DbWriteRetryPolicy(attempts=1, base_delay_seconds=0, max_delay_seconds=0)
    operation = _FlakyOperation([_operational("40P01")])

    with pytest.raises(sa_exc.OperationalError):
        await retry_db_write_async(operation, site="test", policy=policy)

    assert operation.calls == 1
    assert counters == []


@pytest.mark.asyncio
async def test_cancellation_during_backoff_stops_retrying(
    counters, monkeypatch
) -> None:
    # Jitter may draw near zero; pin the wait so the backoff is really long.
    monkeypatch.setattr(
        DbWriteRetryPolicy, "delay", lambda self, retry, *, rng=None: 10.0
    )
    policy = DbWriteRetryPolicy(attempts=3, base_delay_seconds=10, max_delay_seconds=10)
    failed = asyncio.Event()
    calls = 0

    async def operation() -> None:
        nonlocal calls
        calls += 1
        failed.set()
        raise _operational("40P01")

    task = asyncio.create_task(
        retry_db_write_async(operation, site="test", policy=policy)
    )
    await failed.wait()
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert calls == 1
    assert _outcomes(counters) == ["retried"]


@pytest.mark.asyncio
async def test_already_cancelling_caller_is_not_retried(counters) -> None:
    operation = _FlakyOperation([_operational("40P01")])
    error = operation._failures[0]

    async def body() -> None:
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            assert task.cancelling() > 0
            with pytest.raises(sa_exc.OperationalError) as caught:
                await retry_db_write_async(
                    operation,
                    site="test",
                    policy=DbWriteRetryPolicy(
                        attempts=3, base_delay_seconds=10, max_delay_seconds=10
                    ),
                )
            assert caught.value is error

    await asyncio.wait_for(asyncio.create_task(body()), timeout=5)

    assert operation.calls == 1
    assert counters == []


def test_real_sqlite_busy_error_is_retryable_once(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'busy.db'}"
    holder = sqlalchemy.create_engine(url, connect_args={"timeout": 0.05})
    writer = sqlalchemy.create_engine(url, connect_args={"timeout": 0.05})
    try:
        with holder.begin() as connection:
            connection.execute(sqlalchemy.text("CREATE TABLE facts (id INTEGER)"))

        raw = holder.raw_connection()
        try:
            raw.cursor().execute("BEGIN IMMEDIATE")
            with pytest.raises(sa_exc.OperationalError) as caught:
                with writer.begin() as connection:
                    connection.execute(sqlalchemy.text("INSERT INTO facts VALUES (1)"))
        finally:
            raw.rollback()
            raw.close()

        policy = DbWriteRetryPolicy(
            attempts=3, base_delay_seconds=0, max_delay_seconds=0
        )
        assert is_retryable_database_error(caught.value) is True
        assert policy.attempts_for(caught.value) == 2
    finally:
        holder.dispose()
        writer.dispose()
