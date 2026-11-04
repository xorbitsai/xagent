"""Tests for src/xagent/web/jobs/trigger_tasks.py.

PR #1060 review, F4: the stale-preview-run reaper (reap_stale_preview_workforce_runs)
must run from every trigger-scan entrypoint, not just the Celery Beat one
(scan_due_triggers) -- handle_trigger_scan is the BackgroundJob-driven variant of
the same scan, and had no reaper wiring at all before this fix.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, event, text

from xagent.web.jobs import trigger_tasks
from xagent.web.models.background_job import BackgroundJob, BackgroundJobType
from xagent.web.models.database import get_session_local, init_db
from xagent.web.models.user import User
from xagent.web.services import trace_database
from xagent.web.services.workforce_runtime import WorkforceRunPauseTarget


def _init_test_db(path: Path):
    init_db(f"sqlite:///{path}")
    return get_session_local()


def _create_user(db, username: str = "trigger-scan-test") -> User:
    user = User(username=username, password_hash="x")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def test_handle_trigger_scan_reaps_stale_preview_workforce_runs(
    tmp_path, monkeypatch
) -> None:
    """handle_trigger_scan must call the reaper and dispatch PAUSE for any
    reaped run still RUNNING, the same as scan_due_triggers already does."""
    SessionLocal = _init_test_db(tmp_path / "trigger-scan.db")
    db = SessionLocal()
    user = _create_user(db)

    pause_target = WorkforceRunPauseTarget(run_id=1, task_id=2, actor_user_id=3)
    reap_mock = MagicMock(return_value=[pause_target])
    dispatch_calls: list[tuple[list[WorkforceRunPauseTarget], str]] = []

    async def fake_dispatch(pause_targets, *, reason="archive"):
        dispatch_calls.append((pause_targets, reason))

    monkeypatch.setattr(trigger_tasks, "reap_stale_preview_workforce_runs", reap_mock)
    monkeypatch.setattr(
        trigger_tasks, "pause_workforce_tasks_after_archive", fake_dispatch
    )
    monkeypatch.setattr(trigger_tasks, "requeue_stale_background_jobs", lambda _db: [])
    monkeypatch.setattr(trigger_tasks, "scan_due_scheduled_triggers", lambda _db: [])

    job = BackgroundJob(
        user_id=int(user.id),
        job_type=BackgroundJobType.TRIGGER_SCAN.value,
        payload={},
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    result = trigger_tasks.handle_trigger_scan(db, job)

    reap_mock.assert_called_once_with(db)
    assert dispatch_calls == [([pause_target], "preview-reap")]
    assert result["reaped_preview_run_pause_dispatches"] == 1


def test_handle_trigger_scan_skips_dispatch_when_nothing_reaped(
    tmp_path, monkeypatch
) -> None:
    SessionLocal = _init_test_db(tmp_path / "trigger-scan-empty.db")
    db = SessionLocal()
    user = _create_user(db)

    reap_mock = MagicMock(return_value=[])
    dispatch_mock = MagicMock()

    monkeypatch.setattr(trigger_tasks, "reap_stale_preview_workforce_runs", reap_mock)
    monkeypatch.setattr(
        trigger_tasks, "pause_workforce_tasks_after_archive", dispatch_mock
    )
    monkeypatch.setattr(trigger_tasks, "requeue_stale_background_jobs", lambda _db: [])
    monkeypatch.setattr(trigger_tasks, "scan_due_scheduled_triggers", lambda _db: [])

    job = BackgroundJob(
        user_id=int(user.id),
        job_type=BackgroundJobType.TRIGGER_SCAN.value,
        payload={},
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    result = trigger_tasks.handle_trigger_scan(db, job)

    reap_mock.assert_called_once_with(db)
    dispatch_mock.assert_not_called()
    assert result["reaped_preview_run_pause_dispatches"] == 0


@pytest.mark.parametrize("outcome", ["success", "error", "cancel", "detached"])
def test_reaper_closes_trace_resources_before_private_loop_exits(
    tmp_path, monkeypatch, outcome
):
    source = create_engine(f"sqlite:///{tmp_path / 'trace-lifetime.db'}")
    with source.begin() as db:
        db.execute(text("CREATE TABLE writes (id INTEGER PRIMARY KEY)"))
    monkeypatch.setattr(trace_database, "get_engine", lambda: source)
    monkeypatch.setenv("XAGENT_ASYNC_TRACE_DB_ENABLED", "true")
    monkeypatch.setattr(
        trigger_tasks, "reap_stale_preview_workforce_runs", lambda db: [object()]
    )
    runtimes, loops, closed = [], [], []

    async def write_trace():
        runtime = trace_database.get_trace_database_runtime()
        runtimes.append(runtime)
        loop = asyncio.get_running_loop()
        loops.append(loop)
        event.listen(
            runtime.engine.sync_engine,
            "close",
            lambda *_: closed.append(not loop.is_closed()),
        )

        def transaction(db):
            db.execute(text("INSERT INTO writes DEFAULT VALUES"))
            db.commit()

        await runtime.run(
            lambda: pytest.fail("expected async persistence"),
            transaction,
            prepare=lambda: transaction,
        )

    async def dispatch(*args, **kwargs):
        if outcome == "detached":

            async def background():
                try:
                    await asyncio.Future()
                finally:
                    # A runtime first created during cancellation must also
                    # be closed, not missed by early cleanup of the parent.
                    await write_trace()

            asyncio.create_task(background())
            await asyncio.sleep(0)
            return
        await write_trace()
        if outcome == "error":
            raise ValueError("dispatch failed")
        if outcome == "cancel":
            raise asyncio.CancelledError()

    monkeypatch.setattr(trigger_tasks, "pause_workforce_tasks_after_archive", dispatch)
    try:
        # Repeated invocations each own a distinct loop and pool.
        for _ in range(2):
            if outcome in {"error", "cancel"}:
                with pytest.raises(
                    ValueError if outcome == "error" else asyncio.CancelledError
                ):
                    trigger_tasks._reap_and_pause_stale_preview_runs(None)
            else:
                assert trigger_tasks._reap_and_pause_stale_preview_runs(None) == 1
        assert len(runtimes) == 2
        assert runtimes[0] is not runtimes[1]
        assert closed == [True, True]
        assert all(loop.is_closed() for loop in loops)
        assert all(runtime._close_task.done() for runtime in runtimes)
        assert all(
            not thread.is_alive()
            for runtime in runtimes
            for thread in runtime._preparation_pool._threads
        )
        with source.connect() as db:
            assert db.scalar(text("SELECT COUNT(*) FROM writes")) == 2
    finally:
        # Also release resources when running this regression against the
        # unfixed implementation, without hiding the assertions above.
        for runtime in runtimes:
            asyncio.run(runtime.close())
        source.dispose()
