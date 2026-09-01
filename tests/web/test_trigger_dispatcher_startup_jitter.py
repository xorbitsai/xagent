from __future__ import annotations

import asyncio

import pytest

from xagent.web import app as app_module


@pytest.mark.asyncio
async def test_trigger_dispatcher_delays_first_tick_by_startup_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A container restart brings every backlogged trigger due at once, and
    the dispatcher's first tick otherwise fires immediately on startup. The
    startup jitter must delay that first tick -- before touching the DB or
    scanning anything -- so a restart-time burst gets spread out instead."""

    order: list[str] = []

    async def fake_sleep(seconds: float) -> None:
        order.append(f"sleep:{seconds}")

    def fake_uniform(a: float, b: float) -> float:
        assert (a, b) == (0, 30)
        return 12.5

    class FakeSession:
        def close(self) -> None:
            return None

    def fake_scan_due_scheduled_triggers(_db):
        order.append("scan")
        return []

    def fake_reap_stale_preview_workforce_runs(_db):
        order.append("reap")
        return []

    async def fake_dispatch(_db, *, limit: int) -> int:
        order.append("dispatch")
        raise asyncio.CancelledError

    monkeypatch.setattr(app_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(app_module.random, "uniform", fake_uniform)
    monkeypatch.setattr(
        app_module, "get_gmail_watch_enabled", lambda: False, raising=False
    )
    monkeypatch.setattr(
        "xagent.web.models.database.get_session_local",
        lambda: FakeSession,
    )
    monkeypatch.setattr(
        "xagent.web.services.triggers.scan_due_scheduled_triggers",
        fake_scan_due_scheduled_triggers,
    )
    monkeypatch.setattr(
        "xagent.web.services.workforce_runtime.reap_stale_preview_workforce_runs",
        fake_reap_stale_preview_workforce_runs,
    )
    monkeypatch.setattr(
        "xagent.web.services.triggers.dispatch_pending_trigger_runs",
        fake_dispatch,
    )

    with pytest.raises(asyncio.CancelledError):
        await app_module._run_trigger_dispatcher(
            poll_interval_seconds=60,
            batch_size=25,
            startup_jitter_seconds=30,
        )

    assert order[0] == "sleep:12.5"
    assert order.index("sleep:12.5") < order.index("scan")
    assert order.index("sleep:12.5") < order.index("dispatch")


@pytest.mark.asyncio
async def test_trigger_dispatcher_skips_startup_delay_when_jitter_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_calls: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    class FakeSession:
        def close(self) -> None:
            return None

    async def fake_dispatch(_db, *, limit: int) -> int:
        raise asyncio.CancelledError

    monkeypatch.setattr(app_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        app_module, "get_gmail_watch_enabled", lambda: False, raising=False
    )
    monkeypatch.setattr(
        "xagent.web.models.database.get_session_local",
        lambda: FakeSession,
    )
    monkeypatch.setattr(
        "xagent.web.services.triggers.scan_due_scheduled_triggers",
        lambda _db: [],
    )
    monkeypatch.setattr(
        "xagent.web.services.workforce_runtime.reap_stale_preview_workforce_runs",
        lambda _db: [],
    )
    monkeypatch.setattr(
        "xagent.web.services.triggers.dispatch_pending_trigger_runs",
        fake_dispatch,
    )

    with pytest.raises(asyncio.CancelledError):
        await app_module._run_trigger_dispatcher(
            poll_interval_seconds=60,
            batch_size=25,
            startup_jitter_seconds=0,
        )

    assert sleep_calls == []


@pytest.mark.asyncio
async def test_start_trigger_dispatcher_task_passes_configured_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_trigger_dispatcher(**kwargs):
        captured.update(kwargs)

        async def _noop() -> None:
            return None

        return _noop()

    monkeypatch.setattr(app_module, "get_trigger_dispatcher_enabled", lambda: True)
    monkeypatch.setattr(
        app_module, "get_trigger_dispatcher_interval_seconds", lambda: 5
    )
    monkeypatch.setattr(app_module, "get_trigger_dispatcher_batch_size", lambda: 20)
    monkeypatch.setattr(
        app_module, "get_trigger_dispatcher_startup_jitter_seconds", lambda: 45
    )
    monkeypatch.setattr(
        app_module, "_run_trigger_dispatcher", fake_run_trigger_dispatcher
    )
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    class FakeAppState:
        trigger_dispatcher_task = None

    class FakeApp:
        state = FakeAppState()

    task = app_module.start_trigger_dispatcher_task(FakeApp())
    try:
        assert captured["startup_jitter_seconds"] == 45
    finally:
        if task is not None:
            task.cancel()
