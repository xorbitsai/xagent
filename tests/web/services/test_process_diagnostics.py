"""Per-process runtime diagnostics line and the execution counts behind it."""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import sys
from types import ModuleType

import pytest

from xagent.web.services import process_diagnostics
from xagent.web.services.task_dispatch_cap import DispatchPolicy
from xagent.web.services.task_execution import BackgroundTaskManager

ONE_SLOT = DispatchPolicy(limit=1, interactive_reserve=0)

LOGGER = process_diagnostics.__name__
MANAGER_LOGGER = "xagent.web.services.task_execution"


@pytest.fixture
def manager(monkeypatch):
    fresh = BackgroundTaskManager()
    monkeypatch.setattr(
        "xagent.web.services.task_execution.background_task_manager", fresh
    )
    return fresh


def _counts(manager: BackgroundTaskManager) -> tuple[int, int, int, int]:
    return manager.drain_window_counts()


async def _settle() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


class _FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True], ids=["idle", "busy"])
async def test_flush_logs_one_line_per_window(manager, monkeypatch, caplog, busy):
    monkeypatch.setenv("XAGENT_TASK_EXECUTION_ROLE", "combined")
    pending = asyncio.create_task(asyncio.sleep(3600))
    done = asyncio.create_task(asyncio.sleep(0))
    await done
    if busy:
        manager.register_task(1, pending)
        manager.register_task(2, done)
        await _settle()
    clock = _FakeClock()
    flushes = 0

    async def fake_sleep(seconds: float) -> None:
        nonlocal flushes
        clock.now += seconds
        if flushes == 1:
            raise asyncio.CancelledError
        flushes += 1

    caplog.set_level(logging.INFO, logger=LOGGER)
    with pytest.raises(asyncio.CancelledError):
        await process_diagnostics.run_process_diagnostics(
            flush_interval_s=1.0, sample_interval_s=1.0, clock=clock, sleep=fake_sleep
        )
    pending.cancel()

    lines = [r for r in caplog.records if r.name == LOGGER]
    assert len(lines) == 1
    assert lines[0].levelno == logging.INFO
    message = lines[0].getMessage()
    assert message.startswith("runtime diagnostics role=combined ")
    assert f" pid={os.getpid()} window_s=1.0 " in message
    if busy:
        assert " inflight=1 " in message
        assert " started_1m=2 finished_1m=1 ok=1 failed=0 cancelled=0 " in message
    else:
        assert " inflight=0 " in message
        assert " started_1m=0 finished_1m=0 " in message
    assert " gate_held_s=0.0 " in message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "bucket"),
    [
        ("ok", "ok"),
        ("raised", "failed"),
        ("flagged", "failed"),
        ("cancelled", "cancelled"),
        ("flagged_and_cancelled", "cancelled"),
        ("flag_raises", "failed"),
    ],
)
async def test_finish_outcome_bucket(manager, caplog, outcome, bucket):
    caplog.set_level(logging.ERROR, logger=MANAGER_LOGGER)
    handled_failure = outcome.startswith("flagged")
    release = asyncio.Event()

    async def run() -> None:
        await release.wait()
        if outcome == "raised":
            raise RuntimeError("boom")

    def flag() -> bool:
        if outcome == "flag_raises":
            raise RuntimeError("flag broke")
        return handled_failure

    task = asyncio.create_task(run())
    manager.register_task(1, task, failure_flag=flag)
    await asyncio.sleep(0)
    if outcome.endswith("cancelled"):
        task.cancel()
    release.set()
    await asyncio.gather(task, return_exceptions=True)
    await _settle()

    started, ok, failed, cancelled = _counts(manager)
    assert started == 1
    assert {"ok": ok, "failed": failed, "cancelled": cancelled} == {
        name: int(name == bucket) for name in ("ok", "failed", "cancelled")
    }
    raised_reports = [
        r
        for r in caplog.records
        if r.getMessage() == "Background execution for task 1 raised"
    ]
    assert len(raised_reports) == int(outcome == "raised")


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["register", "promote"])
async def test_unawaited_execution_error_is_logged_once_with_traceback(
    manager, caplog, entry
):
    failure = RuntimeError("boom")

    async def run() -> None:
        raise failure

    task = asyncio.create_task(run())
    if entry == "register":
        manager.register_task(7, task)
    else:
        assert manager.reserve_resume(7)
        manager.register_reserved_resume(7, task, run_id="run-7")
        manager.promote_resume_task(7, task)
    caplog.set_level(logging.ERROR)
    # asyncio.wait does not retrieve the exception: nothing awaits the handle.
    await asyncio.wait({task})
    await _settle()
    del task
    gc.collect()

    # Every record carrying this exception, from any logger: asyncio's own
    # "never retrieved" report would be a second one.
    reports = [
        (r.name, r.levelno, r.getMessage())
        for r in caplog.records
        if r.exc_info is not None and r.exc_info[1] is failure
    ]
    assert reports == [
        (MANAGER_LOGGER, logging.ERROR, "Background execution for task 7 raised")
    ]
    assert _counts(manager) == (1, 0, 1, 0)


@pytest.mark.asyncio
async def test_drain_resets_and_shutdown_keeps_counts(manager):
    task = asyncio.create_task(asyncio.sleep(3600))
    manager.register_task(1, task)
    await manager.shutdown()
    await _settle()

    assert _counts(manager) == (1, 0, 0, 1)
    assert _counts(manager) == (0, 0, 0, 0)


@pytest.mark.asyncio
async def test_new_lifespan_starts_from_zero(manager):
    task = asyncio.create_task(asyncio.sleep(0))
    manager.register_task(1, task)
    await task
    await _settle()

    manager.start_accepting()

    assert _counts(manager) == (0, 0, 0, 0)


@pytest.mark.asyncio
async def test_same_task_registered_twice_counted_once(manager):
    task = asyncio.create_task(asyncio.sleep(0))
    manager.register_task(1, task)
    manager.register_task(1, task)
    await task
    await _settle()

    assert _counts(manager) == (1, 1, 0, 0)


@pytest.mark.asyncio
async def test_replaced_registration_still_counts_inflight(manager):
    older = asyncio.create_task(asyncio.sleep(3600))
    newer = asyncio.create_task(asyncio.sleep(3600))
    manager.register_task(1, older)
    manager.register_task(1, newer)

    assert manager.inflight_counts() == (2, 0)
    for task in (older, newer):
        task.cancel()
    await asyncio.gather(older, newer, return_exceptions=True)


@pytest.mark.asyncio
async def test_promoted_resume_counted_and_fields_disjoint(manager):
    coordinator = asyncio.create_task(asyncio.sleep(3600))
    assert manager.reserve_resume(1)
    manager.register_reserved_resume(1, coordinator, run_id="run-1")

    assert manager.inflight_counts() == (0, 1)
    assert _counts(manager) == (0, 0, 0, 0)

    manager.promote_resume_task(1, coordinator)

    assert manager.inflight_counts() == (1, 0)
    coordinator.cancel()
    await asyncio.gather(coordinator, return_exceptions=True)
    await _settle()
    assert _counts(manager) == (1, 0, 0, 1)


@pytest.mark.asyncio
async def test_registration_rejected_while_shutting_down_is_not_counted(manager):
    await manager.shutdown()
    task = asyncio.create_task(asyncio.sleep(3600))

    with pytest.raises(RuntimeError, match="shutting down"):
        manager.register_task(1, task)
    await asyncio.gather(task, return_exceptions=True)
    await _settle()

    assert _counts(manager) == (0, 0, 0, 0)


@pytest.mark.asyncio
async def test_lag_sampler_counts_overrun(manager, monkeypatch):
    clock = _FakeClock()
    sleeps = 0

    async def fake_sleep(seconds: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps == 3:
            raise asyncio.CancelledError
        # The second sleep overruns by 0.8 s.
        clock.now += seconds + (0.8 if sleeps == 2 else 0.0)

    monkeypatch.setattr(process_diagnostics, "_window", process_diagnostics._Window())
    with pytest.raises(asyncio.CancelledError):
        await process_diagnostics.run_process_diagnostics(
            flush_interval_s=3600.0,
            sample_interval_s=1.0,
            clock=clock,
            sleep=fake_sleep,
        )

    counters = process_diagnostics.drain_execution_counters(now=clock.now)
    assert counters.loop_lag_samples == 1
    assert counters.loop_lag_max_ms == pytest.approx(800.0)


@pytest.mark.asyncio
async def test_flush_failure_keeps_loop_running(manager, monkeypatch, caplog):
    clock = _FakeClock()
    real_drain = process_diagnostics.drain_execution_counters
    calls = 0

    def flaky_drain(now=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("flush broke")
        return real_drain(now)

    async def fake_sleep(seconds: float) -> None:
        if calls == 2:
            raise asyncio.CancelledError
        clock.now += seconds

    monkeypatch.setattr(process_diagnostics, "drain_execution_counters", flaky_drain)
    caplog.set_level(logging.INFO, logger=LOGGER)
    with pytest.raises(asyncio.CancelledError):
        await process_diagnostics.run_process_diagnostics(
            flush_interval_s=1.0, sample_interval_s=1.0, clock=clock, sleep=fake_sleep
        )

    records = [r for r in caplog.records if r.name == LOGGER]
    assert [r.levelno for r in records] == [logging.ERROR, logging.INFO]
    assert records[1].getMessage().startswith("runtime diagnostics ")


@pytest.mark.asyncio
async def test_bad_role_still_starts_and_stop_logs_final_line(
    manager, monkeypatch, caplog
):
    monkeypatch.setenv("XAGENT_TASK_EXECUTION_ROLE", "nonsense")
    caplog.set_level(logging.INFO, logger=LOGGER)

    task = process_diagnostics.start_process_diagnostics()
    assert task is not None and not task.done()
    execution = asyncio.create_task(asyncio.sleep(3600))
    manager.register_task(1, execution)
    await manager.shutdown()
    await process_diagnostics.stop_process_diagnostics(task)

    assert task.cancelled()
    lines = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert len(lines) == 1
    assert lines[0].startswith("runtime diagnostics role=unknown ")
    assert " started_1m=1 finished_1m=1 ok=0 failed=0 cancelled=1 " in lines[0]


@pytest.mark.asyncio
async def test_stop_keeps_the_callers_own_cancellation(manager):
    task = process_diagnostics.start_process_diagnostics()
    assert task is not None
    gate = asyncio.Event()

    async def caller() -> None:
        await gate.wait()
        await process_diagnostics.stop_process_diagnostics(task)

    outer = asyncio.create_task(caller())
    await asyncio.sleep(0)
    gate.set()
    await asyncio.sleep(0)
    outer.cancel()

    with pytest.raises(asyncio.CancelledError):
        await outer


@pytest.mark.asyncio
async def test_app_lifespan_starts_one_diagnostics_task_and_stops_it(manager):
    from fastapi import FastAPI

    from xagent.web import app as app_module

    test_app = FastAPI()
    app_module.start_app_process_diagnostics(test_app)
    first = test_app.state.process_diagnostics_task
    app_module.start_app_process_diagnostics(test_app)

    assert first is not None and not first.done()
    assert test_app.state.process_diagnostics_task is first

    await app_module.stop_app_process_diagnostics(test_app)
    assert test_app.state.process_diagnostics_task is None
    assert first.cancelled()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failing_step", [None, "drain"], ids=["drain_returns", "drain_raises"]
)
async def test_app_shutdown_stops_diagnostics_after_the_drain(
    monkeypatch, failing_step
):
    from xagent.web import app as app_module

    order: list[str] = []

    async def record(name: str) -> None:
        order.append(name)
        if name == failing_step:
            raise RuntimeError("drain failed")

    class _Channel:
        enabled = False

        async def stop(self) -> None:
            return None

    class _Sandboxes:
        async def cleanup(self) -> None:
            return None

    for module_name, attribute in (
        ("xagent.web.channels.telegram.bot", "get_telegram_channel"),
        ("xagent.web.channels.feishu.bot", "get_feishu_channel"),
        ("xagent.web.channels.slack.bot", "get_slack_channel"),
    ):
        fake = ModuleType(module_name)
        setattr(fake, attribute, lambda: _Channel())
        monkeypatch.setitem(sys.modules, module_name, fake)
    monkeypatch.setattr(app_module, "flush_langfuse", lambda: None)
    for name in (
        "stop_runtime_performance_monitor",
        "stop_orphan_upload_gc_task",
        "stop_retention_purge_task",
        "stop_task_cleanup_retry_task",
        "stop_uploaded_file_recovery_task",
        "stop_task_auto_resume_task",
        "stop_task_lease_recovery_task",
    ):
        monkeypatch.setattr(app_module, name, lambda _app: record("other"))
    monkeypatch.setattr(
        app_module, "stop_app_process_diagnostics", lambda _app: record("diagnostics")
    )
    monkeypatch.setattr(
        "xagent.web.services.task_execution.background_task_manager.shutdown",
        lambda: record("drain"),
    )
    monkeypatch.setattr(
        "xagent.web.services.task_lease_service.wait_for_heartbeat_manager_idle",
        lambda: record("other"),
    )
    for name in (
        "_task_command_dispatcher_task",
        "_sandbox_idle_sweep_task",
        "_file_storage_startup_sync_task",
        "_trigger_dispatcher_task",
        "_migration_task",
    ):
        monkeypatch.setattr(app_module, name, None)
    monkeypatch.setattr(
        "xagent.web.sandbox_manager.get_sandbox_manager", lambda: _Sandboxes()
    )
    app_module.app.state.metadata_rebuild_task = None
    for attribute in ("telegram_task", "slack_task"):
        if hasattr(app_module.app.state, attribute):
            monkeypatch.delattr(app_module.app.state, attribute)

    if failing_step is None:
        await app_module.shutdown_event()
    else:
        with pytest.raises(RuntimeError, match="drain failed"):
            await app_module.shutdown_event()

    stops = [name for name in order if name != "other"]
    assert stops == ["drain", "diagnostics"]


def _gate_probe(manager, monkeypatch, *, window_started_at: float = 0.0):
    """Return (clock, close, open_, drain) helpers around one manager.

    ``close`` / ``open_`` make the process full / empty at a clock reading and
    build a cap snapshot with limit 1; ``drain`` is the diagnostics drain.
    """

    monkeypatch.setenv("XAGENT_TASK_EXECUTION_ROLE", "combined")
    monkeypatch.setattr(process_diagnostics._window, "started_at", window_started_at)
    clock = _FakeClock()
    manager._cap_clock = clock

    def close(at: float) -> None:
        clock.now = at
        manager._live_executions[1] = {object()}  # type: ignore[assignment]
        manager.dispatch_cap_snapshot(ONE_SLOT)

    def open_(at: float) -> None:
        clock.now = at
        manager._live_executions.pop(1, None)
        manager.dispatch_cap_snapshot(ONE_SLOT)

    def drain(at: float) -> float:
        return process_diagnostics.drain_execution_counters(now=at).gate_held_s

    return clock, close, open_, drain


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row",
    ["never_closed", "closed_within_window", "split_across_drain"]
    + ["bounded_by_window", "drain_observes", "zeroed_after_drain"],
)
async def test_gate_held_seconds_measures_closed_time(manager, monkeypatch, row):
    started_at = 100.0 if row == "bounded_by_window" else 0.0
    _, close, open_, drain = _gate_probe(
        manager, monkeypatch, window_started_at=started_at
    )

    if row == "never_closed":
        open_(1)
        open_(5)
        assert drain(10) == 0.0
    elif row == "closed_within_window":
        open_(1)
        close(3)
        open_(5)
        assert drain(10) == 2.0
    elif row == "split_across_drain":
        close(8)
        assert drain(10) == 2.0
        open_(11)
        assert drain(13) == 1.0
    elif row == "bounded_by_window":
        close(90)
        assert drain(110) == 10.0
    elif row == "drain_observes":
        open_(0)
        manager._live_executions[1] = {object()}  # type: ignore[assignment]
        assert drain(10) == 0.0
        assert drain(15) == 5.0
    else:
        close(3)
        open_(5)
        assert drain(10) == 2.0
        assert drain(20) == 0.0


@pytest.mark.asyncio
async def test_gate_held_counts_a_full_process_with_nothing_queued(
    manager, monkeypatch
):
    clock, _, _, drain = _gate_probe(manager, monkeypatch)
    running = asyncio.create_task(asyncio.sleep(3600))
    try:
        manager.register_task(1, running)
        clock.now = 0.0
        manager.dispatch_cap_snapshot(ONE_SLOT)
        assert drain(60) == 60.0
    finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("row", ["shutdown_closes_interval", "new_lifespan_resets"])
async def test_gate_held_after_shutdown_and_new_lifespan(manager, monkeypatch, row):
    clock, _, _, drain = _gate_probe(manager, monkeypatch)
    clock.now = 0.0
    # A reservation outlives cancellation, so only the observation that
    # follows shutdown's clearing of the tables can end the interval at 4.0.
    assert (
        manager.reserve_dispatch_slot(1, policy=ONE_SLOT, interactive=False) is not None
    )
    manager.dispatch_cap_snapshot(ONE_SLOT)
    clock.now = 4.0
    await manager.shutdown()

    if row == "shutdown_closes_interval":
        assert drain(10) == 4.0
    else:
        clock.now = 6.0
        manager.start_accepting()
        assert drain(10) == 0.0
