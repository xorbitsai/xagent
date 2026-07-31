from __future__ import annotations

import asyncio

import pytest

from xagent.web import app as app_module


@pytest.mark.asyncio
async def test_trigger_dispatcher_skips_gmail_scan_when_watch_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeSession:
        def close(self) -> None:
            return None

    def fake_get_session_local():
        return FakeSession

    def fake_scan_due_gmail_watch_renewals(_db) -> int:
        return 0

    def fake_scan_due_scheduled_triggers(_db):
        return []

    def fake_reap_stale_preview_workforce_runs(_db):
        return []

    async def fake_dispatch_pending_trigger_runs(_db, *, limit: int) -> int:
        raise asyncio.CancelledError

    async def fake_to_thread(func, /, *args, **kwargs):
        calls.append(func.__name__)
        return func(*args, **kwargs)

    monkeypatch.setattr(
        app_module, "get_gmail_watch_enabled", lambda: False, raising=False
    )
    monkeypatch.setattr(app_module.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(
        "xagent.web.models.database.get_session_local",
        fake_get_session_local,
    )
    monkeypatch.setattr(
        "xagent.web.services.gmail_triggers.scan_due_gmail_watch_renewals",
        fake_scan_due_gmail_watch_renewals,
    )
    monkeypatch.setattr(
        "xagent.web.services.triggers.scan_due_scheduled_triggers",
        fake_scan_due_scheduled_triggers,
    )
    # Not mocking this (and reaping via a real FakeSession, which has no
    # .query) would raise inside the loop body before dispatch is ever
    # reached -- the outer except swallows it and the loop just spins/sleeps
    # forever instead of ending via fake_dispatch_pending_trigger_runs below.
    monkeypatch.setattr(
        "xagent.web.services.workforce_runtime.reap_stale_preview_workforce_runs",
        fake_reap_stale_preview_workforce_runs,
    )
    monkeypatch.setattr(
        "xagent.web.services.triggers.dispatch_pending_trigger_runs",
        fake_dispatch_pending_trigger_runs,
    )

    with pytest.raises(asyncio.CancelledError):
        await app_module._run_trigger_dispatcher(
            poll_interval_seconds=60,
            batch_size=25,
        )

    # Scheduled triggers are scanned in-process every tick (no Celery needed),
    # but the Gmail watch-renewal scan stays gated off when watch is disabled.
    assert "_scan_due_scheduled_triggers_tick" in calls
    assert "_scan_due_gmail_watch_renewals_tick" not in calls
    assert "_reap_stale_preview_workforce_runs_tick" in calls


@pytest.mark.asyncio
async def test_trigger_dispatcher_survives_scheduled_scan_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scheduled-scan tick that raises must not kill the loop: it is caught
    at the loop level, logged, and the loop survives to the next tick."""

    class FakeSession:
        def close(self) -> None:
            return None

    scan_calls = {"n": 0}
    dispatch_calls = {"n": 0}

    def flaky_scan(_db):
        scan_calls["n"] += 1
        if scan_calls["n"] == 1:
            raise RuntimeError("scan blew up")
        return []

    async def fake_dispatch(_db, *, limit: int) -> int:
        dispatch_calls["n"] += 1
        # Only reached on a surviving tick; stop the loop here.
        raise asyncio.CancelledError

    def fake_reap_stale_preview_workforce_runs(_db):
        return []

    monkeypatch.setattr(
        app_module, "get_gmail_watch_enabled", lambda: False, raising=False
    )
    monkeypatch.setattr(
        "xagent.web.models.database.get_session_local",
        lambda: FakeSession,
    )
    monkeypatch.setattr(
        "xagent.web.services.triggers.scan_due_scheduled_triggers",
        flaky_scan,
    )
    # Not mocking this (and reaping via FakeSession, which has no .query)
    # would raise inside the loop body before fake_dispatch is ever reached
    # -- with poll_interval_seconds=0 below, the outer except swallowing it
    # would spin the loop as fast as possible forever instead of ending.
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
            poll_interval_seconds=0,
            batch_size=25,
        )

    # First tick's scan raised; the loop caught it and ran a second tick where
    # dispatch was finally reached.
    assert scan_calls["n"] == 2
    assert dispatch_calls["n"] == 1


@pytest.mark.asyncio
async def test_trigger_dispatcher_throttles_preview_run_reap_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Self-review follow-up: the reap tick must not run on every dispatcher
    poll -- only at its own, much coarser interval
    (get_background_job_sweep_interval_seconds) -- since the staleness
    threshold it acts on (get_workforce_preview_run_stale_seconds, default
    7200s) is hours-scale, unlike the dispatcher's own poll interval (as
    low as a few seconds)."""

    class FakeSession:
        def close(self) -> None:
            return None

    reap_calls = {"n": 0}
    dispatch_calls = {"n": 0}

    def fake_reap(_db):
        reap_calls["n"] += 1
        return []

    async def fake_dispatch(_db, *, limit: int) -> int:
        dispatch_calls["n"] += 1
        if dispatch_calls["n"] >= 3:
            raise asyncio.CancelledError
        return 0

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
        fake_reap,
    )
    monkeypatch.setattr(
        "xagent.web.services.triggers.dispatch_pending_trigger_runs",
        fake_dispatch,
    )

    with pytest.raises(asyncio.CancelledError):
        await app_module._run_trigger_dispatcher(
            poll_interval_seconds=0,
            batch_size=25,
        )

    # 3 loop iterations happened (dispatch_calls reaches 3 before raising),
    # but the reap tick's own throttle interval hasn't elapsed between them
    # (poll_interval_seconds=0, so barely any wall-clock time passes across
    # iterations) -- it must have run only once, not on every iteration.
    assert dispatch_calls["n"] == 3
    assert reap_calls["n"] == 1
