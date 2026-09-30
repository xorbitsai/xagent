"""How the retention purge loop is attached to the application (#2563).

The loop's own behaviour is tested in
``tests/web/services/test_task_retention_purge.py``. What is pinned here is
narrower and, for a feature that deletes conversations, the part worth pinning
separately: that an unconfigured deployment starts nothing at all, and that
shutdown signals the loop before it cancels it.
"""

from __future__ import annotations

import asyncio
from importlib import import_module

import pytest
from fastapi import FastAPI


@pytest.fixture
def app_module():
    return import_module("xagent.web.app")


@pytest.fixture
def no_retention_env(monkeypatch):
    for name in (
        "XAGENT_CONVERSATION_RETENTION_DAYS",
        "XAGENT_TRACE_RETENTION_DAYS",
        "XAGENT_RETENTION_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_no_configured_period_starts_nothing(app_module, no_retention_env) -> None:
    """The shipping default. Nothing is scheduled and no state is left behind."""
    app = FastAPI()

    assert app_module.start_retention_purge_task(app) is None
    assert getattr(app.state, "retention_purge_task", None) is None
    assert getattr(app.state, "retention_purge_stop", None) is None


def test_the_kill_switch_alone_prevents_the_start(app_module, no_retention_env) -> None:
    no_retention_env.setenv("XAGENT_CONVERSATION_RETENTION_DAYS", "365")
    no_retention_env.setenv("XAGENT_RETENTION_ENABLED", "false")
    app = FastAPI()

    assert app_module.start_retention_purge_task(app) is None


@pytest.mark.asyncio
async def test_a_configured_period_starts_the_loop(
    app_module, no_retention_env, monkeypatch
) -> None:
    """The positive path, which nothing else here exercises.

    Opts past the ``PYTEST_CURRENT_TEST`` guard explicitly. The guard is there
    because ``tests/conftest.py`` loads a developer's ``.env`` with
    ``override=True``, so a machine configured with a retention period and a
    PostgreSQL ``DATABASE_URL`` would otherwise run a real, deleting sweep
    during its own test suite. Opting in here keeps that protection while
    still exercising the starter for real -- one that returned ``None`` in
    every test would be indistinguishable from one that was broken.
    """
    import sqlalchemy as sa
    from sqlalchemy.orm import sessionmaker

    import xagent.web.models.database as database_module

    monkeypatch.setenv("XAGENT_CONVERSATION_RETENTION_DAYS", "365")
    monkeypatch.setattr(
        database_module,
        "get_session_local",
        lambda: sessionmaker(bind=sa.create_engine("sqlite://")),
    )
    app = FastAPI()
    app.state.retention_purge_allowed_in_tests = True

    task = app_module.start_retention_purge_task(app)

    assert task is not None
    assert app.state.retention_purge_task is task
    assert isinstance(app.state.retention_purge_stop, asyncio.Event)
    # It ends itself: SQLite cannot fence the purge, so the loop refuses the
    # store and returns instead of sweeping.
    await asyncio.wait_for(task, timeout=5)
    await app_module.stop_retention_purge_task(app)


@pytest.mark.asyncio
async def test_stopping_signals_before_cancelling(app_module) -> None:
    """A batch in flight gets to finish its current task's transaction.

    Cancelling without signalling first would tear the loop out of a delete
    it is midway through; the transaction would roll back cleanly, but the
    sweep would have no way to stop *between* tasks, which is the only place
    stopping is free.
    """
    app = FastAPI()
    signalled = asyncio.Event()
    stop_event = asyncio.Event()

    async def loop_body() -> None:
        await stop_event.wait()
        signalled.set()

    app.state.retention_purge_stop = stop_event
    app.state.retention_purge_task = asyncio.create_task(loop_body())

    await app_module.stop_retention_purge_task(app)

    assert signalled.is_set(), "the loop was cancelled without being asked to stop"
    assert getattr(app.state, "retention_purge_task", None) is None
    assert getattr(app.state, "retention_purge_stop", None) is None


@pytest.mark.asyncio
async def test_stopping_drains_a_loop_that_ignores_the_signal(
    app_module, monkeypatch
) -> None:
    """Signalling is the polite path, not the only one.

    The grace period is shortened here rather than waited out: what is under
    test is that the cancel happens once it elapses, not how long it is.
    """
    monkeypatch.setattr(app_module, "RETENTION_PURGE_STOP_GRACE_SECONDS", 0.05)
    app = FastAPI()

    async def unresponsive() -> None:
        await asyncio.Event().wait()

    app.state.retention_purge_stop = asyncio.Event()
    task = asyncio.create_task(unresponsive())
    app.state.retention_purge_task = task

    await asyncio.wait_for(app_module.stop_retention_purge_task(app), timeout=5)

    # ``done()`` would also be true of a loop that simply returned, which is
    # what this test must not accept: the point is that the grace period
    # elapsed and the cancel happened.
    assert task.cancelled()


@pytest.mark.asyncio
async def test_stopping_is_safe_when_nothing_was_started(app_module) -> None:
    """Shutdown runs unconditionally, including after a startup that never got here."""
    await app_module.stop_retention_purge_task(FastAPI())


def test_the_pytest_guard_keeps_a_configured_period_from_sweeping(
    app_module, no_retention_env, monkeypatch
) -> None:
    """A developer's own `.env` must not turn `pytest` into a purge.

    `tests/conftest.py` loads it with `override=True`, so a period configured
    there reaches every test process. Without this guard, a developer whose
    `DATABASE_URL` also points at PostgreSQL would have had their own database
    swept -- the dialect refusal that was argued as sufficient does not fire
    in that case.
    """
    monkeypatch.setenv("XAGENT_CONVERSATION_RETENTION_DAYS", "365")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_guard (call)")
    app = FastAPI()

    assert app_module.start_retention_purge_task(app) is None
    assert getattr(app.state, "retention_purge_task", None) is None
