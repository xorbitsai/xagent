"""Monitoring queries run on a worker thread under a slot limit and a budget.

The monitoring endpoints scan large tables. Running those scans on the event
loop thread stalls every other request the process serves, so each endpoint
hands its queries to ``_run_monitor_query``, which executes them on a worker
thread with their own session, refuses new work when every slot is taken, and
maps budget / pool-timeout failures to a fixed 503.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute
from sqlalchemy import event
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import Session

from xagent.web.api import monitor as monitor_module
from xagent.web.api.monitor import (
    get_agents,
    get_dashboard_stats,
    get_model_stats,
    get_monitoring_stats,
    get_popular_tools,
)
from xagent.web.models.database import get_db, get_engine
from xagent.web.models.task import Task, TraceEvent
from xagent.web.models.user import User

from .conftest import _direct_db_session

pytestmark = pytest.mark.usefixtures("_test_db")

UNAVAILABLE = "Monitoring statistics are temporarily unavailable"


def _seed_admin(db: Session, *, username: str) -> User:
    """An administrator with one task and one event of each kind counted."""
    user = User(username=username, password_hash="hash", is_admin=True)
    db.add(user)
    db.commit()
    db.refresh(user)

    task = Task(user_id=user.id, title=f"{username}-task")
    db.add(task)
    db.commit()
    db.refresh(task)

    now = datetime.now(timezone.utc)
    for index, (event_type, data) in enumerate(
        [
            ("llm_call_start", {"model_name": "gpt-4o"}),
            ("llm_call_end", {"total_tokens": 5}),
            ("tool_execution_start", {"tool_name": "calculator"}),
        ]
    ):
        db.add(
            TraceEvent(
                task_id=task.id,
                event_id=f"{username}-{index}",
                event_type=event_type,
                timestamp=now,
                data=data,
            )
        )
    db.commit()
    return user


OFFLOADED_ENDPOINTS = [
    get_agents,
    get_monitoring_stats,
    get_popular_tools,
    get_model_stats,
    get_dashboard_stats,
]


def test_offloaded_endpoint_list_matches_router() -> None:
    """Every database-backed monitoring route is covered by the thread test."""
    db_routes = {
        route.endpoint
        for route in monitor_module.monitor_router.routes
        if isinstance(route, APIRoute)
        and any(
            dependency.call is get_db
            for dependency in getattr(route, "depend" + "ant").dependencies
        )
    }

    assert db_routes == set(OFFLOADED_ENDPOINTS)


@pytest.mark.parametrize(
    "endpoint",
    OFFLOADED_ENDPOINTS,
    ids=["agents", "stats", "popular_tools", "model_stats", "dashboard_stats"],
)
async def test_monitor_queries_run_off_the_event_loop(
    endpoint: Callable[..., Awaitable[Any]],
) -> None:
    db = _direct_db_session()
    try:
        user = _seed_admin(db, username="offload-admin")
        # Load the user's columns now so the call itself never needs a lazy
        # refresh on the event loop.
        db.refresh(user)

        executed_on: set[int] = set()

        def record(*_args: Any, **_kwargs: Any) -> None:
            executed_on.add(threading.get_ident())

        engine = get_engine()
        loop_thread = threading.get_ident()
        event.listen(engine, "before_cursor_execute", record)
        try:
            await endpoint(db=db, current_user=user)
        finally:
            event.remove(engine, "before_cursor_execute", record)

        assert executed_on
        assert loop_thread not in executed_on
    finally:
        db.close()


async def test_request_is_refused_while_every_slot_is_taken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With every slot taken the next request fails at once instead of queueing."""
    monkeypatch.setattr(monitor_module, "_MONITOR_QUERY_SLOTS", asyncio.Semaphore(3))
    db = _direct_db_session()
    try:
        release = threading.Event()
        entered = threading.Semaphore(0)
        refused_compute_calls: list[int] = []

        def blocking(_session: Session) -> str:
            entered.release()
            release.wait(timeout=30)
            return "done"

        def refused(_session: Session) -> str:
            refused_compute_calls.append(1)
            return "should not run"

        holders = [
            asyncio.create_task(monitor_module._run_monitor_query("held", db, blocking))
            for _ in range(3)
        ]
        try:
            for _ in range(3):
                assert await asyncio.to_thread(entered.acquire, True, 10)

            with pytest.raises(HTTPException) as caught:
                await asyncio.wait_for(
                    monitor_module._run_monitor_query("fourth", db, refused),
                    timeout=5,
                )

            assert caught.value.status_code == 503
            assert caught.value.detail == UNAVAILABLE
            assert refused_compute_calls == []
        finally:
            release.set()
        assert await asyncio.gather(*holders) == ["done"] * 3
    finally:
        db.close()


class _PgcodeOrig(Exception):
    """psycopg2-style driver error: carries ``pgcode``."""

    def __init__(self, code: str) -> None:
        super().__init__(f"driver error {code}")
        self.pgcode = code


class _SqlstateOrig(Exception):
    """psycopg3-style driver error: carries ``sqlstate``."""

    def __init__(self, code: str) -> None:
        super().__init__(f"driver error {code}")
        self.sqlstate = code


def _raise_operational(orig: Exception) -> None:
    raise OperationalError("SELECT secret_column FROM t", {}, orig)


def _raise_wrapped_budget_error() -> None:
    try:
        _raise_operational(_PgcodeOrig("57014"))
    except OperationalError as inner:
        raise RuntimeError("wrapper") from inner


@pytest.mark.parametrize(
    "raiser",
    [
        pytest.param(lambda: _raise_operational(_PgcodeOrig("57014")), id="pgcode"),
        pytest.param(lambda: _raise_operational(_SqlstateOrig("57014")), id="sqlstate"),
        pytest.param(_raise_wrapped_budget_error, id="cause-chain"),
        pytest.param(
            lambda: (_ for _ in ()).throw(SQLAlchemyTimeoutError("pool exhausted")),
            id="pool-timeout",
        ),
    ],
)
async def test_budget_and_pool_errors_map_to_503(
    raiser: Callable[[], None],
) -> None:
    db = _direct_db_session()
    try:

        def compute(_session: Session) -> None:
            raiser()

        with pytest.raises(HTTPException) as caught:
            await monitor_module._run_monitor_query("budget", db, compute)

        assert caught.value.status_code == 503
        assert caught.value.detail == UNAVAILABLE
    finally:
        db.close()


async def test_other_sqlstate_stays_500() -> None:
    db = _direct_db_session()
    try:
        raised: list[BaseException] = []

        def compute(_session: Session) -> None:
            try:
                _raise_operational(_PgcodeOrig("40001"))
            except OperationalError as exc:
                raised.append(exc)
                raise

        with pytest.raises(HTTPException) as caught:
            await monitor_module._run_monitor_query("serialization", db, compute)

        assert caught.value.status_code == 500
        assert caught.value.detail == str(raised[0])
    finally:
        db.close()


async def test_non_budget_error_keeps_500_contract(
    caplog: pytest.LogCaptureFixture,
) -> None:
    db = _direct_db_session()
    try:

        def compute(_session: Session) -> None:
            raise ValueError("boom")

        with pytest.raises(HTTPException) as caught:
            await monitor_module._run_monitor_query("failing", db, compute)

        assert caught.value.status_code == 500
        assert caught.value.detail == "boom"
        assert "Get failing failed: boom" in caplog.text
    finally:
        db.close()


async def test_cancelled_request_keeps_slot_until_worker_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled request holds its slot while its worker is still running."""
    slots = asyncio.Semaphore(3)
    monkeypatch.setattr(monitor_module, "_MONITOR_QUERY_SLOTS", slots)
    db = _direct_db_session()
    try:
        started = threading.Event()
        release = threading.Event()

        def blocking(_session: Session) -> str:
            started.set()
            release.wait(timeout=30)
            return "done"

        task = asyncio.create_task(
            monitor_module._run_monitor_query("cancelled", db, blocking)
        )
        try:
            assert await asyncio.to_thread(started.wait, 10)
            task.cancel()
            await asyncio.sleep(0.2)

            assert slots._value == 2
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert slots._value == 3
    finally:
        db.close()


async def test_slot_is_released_when_compute_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slots = asyncio.Semaphore(3)
    monkeypatch.setattr(monitor_module, "_MONITOR_QUERY_SLOTS", slots)
    db = _direct_db_session()
    try:

        def compute(_session: Session) -> None:
            raise ValueError("boom")

        with pytest.raises(HTTPException):
            await monitor_module._run_monitor_query("failing", db, compute)

        assert slots._value == 3
    finally:
        db.close()
