"""The in-process Milvus cleanup must never delay or stop trigger dispatch."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from xagent.web import app as app_module

_CLEANUP = (
    "xagent.core.tools.core.RAG_tools.kb.milvus_cleanup.purge_leftover_milvus_rows"
)


def _install_harness(
    monkeypatch: pytest.MonkeyPatch, purge: Any, dispatch: Any
) -> None:
    class FakeSession:
        def close(self) -> None:
            return None

    monkeypatch.setattr(
        app_module, "get_gmail_watch_enabled", lambda: False, raising=False
    )
    # Make every tick a due interval; the tests below prove the running-task gate.
    monkeypatch.setattr(app_module, "get_kb_milvus_cleanup_interval_seconds", lambda: 0)
    monkeypatch.setattr(
        "xagent.web.models.database.get_session_local", lambda: FakeSession
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
        "xagent.web.services.triggers.dispatch_pending_trigger_runs", dispatch
    )
    monkeypatch.setattr(_CLEANUP, purge)


def _find_cleanup_task() -> asyncio.Task[Any]:
    """Return the dispatcher's in-flight Milvus cleanup task, if any."""
    for task in asyncio.all_tasks():
        if getattr(task.get_coro(), "__qualname__", "").endswith(
            "_run_kb_milvus_cleanup"
        ):
            return task
    raise AssertionError("dispatcher did not start the Milvus cleanup task")


@pytest.mark.asyncio
async def test_a_running_purge_does_not_delay_dispatch_or_start_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    purge_calls = {"n": 0}
    purge_started = threading.Event()
    purge_release = threading.Event()
    dispatch_calls = {"n": 0}

    def blocking_purge() -> dict[str, int]:
        purge_calls["n"] += 1
        purge_started.set()
        purge_release.wait(timeout=10)
        return {}

    async def fake_dispatch(_db: Any, *, limit: int) -> int:
        dispatch_calls["n"] += 1
        if dispatch_calls["n"] >= 3:
            raise asyncio.CancelledError
        return 0

    _install_harness(monkeypatch, blocking_purge, fake_dispatch)

    try:
        with pytest.raises(asyncio.CancelledError):
            await app_module._run_trigger_dispatcher(
                poll_interval_seconds=0,
                batch_size=25,
            )
    finally:
        purge_release.set()
    await asyncio.sleep(0)

    assert purge_started.wait(timeout=5)
    # Dispatch ran on every tick while the purge was still blocked, and neither
    # of the later due intervals started a second purge.
    assert dispatch_calls["n"] == 3
    assert purge_calls["n"] == 1


@pytest.mark.asyncio
async def test_a_failed_purge_does_not_stop_dispatch_or_block_the_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    purge_calls = {"n": 0}
    dispatch_calls = {"n": 0}

    def failing_purge() -> dict[str, int]:
        purge_calls["n"] += 1
        if purge_calls["n"] == 1:
            raise RuntimeError("milvus is down")
        return {}

    async def fake_dispatch(_db: Any, *, limit: int) -> int:
        dispatch_calls["n"] += 1
        if dispatch_calls["n"] >= 3:
            raise asyncio.CancelledError
        return 0

    async def fake_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    _install_harness(monkeypatch, failing_purge, fake_dispatch)
    # Run the purge inline so each background task settles before the next
    # interval check; the background-task structure is what is under test.
    monkeypatch.setattr(app_module.asyncio, "to_thread", fake_to_thread)

    with pytest.raises(asyncio.CancelledError):
        await app_module._run_trigger_dispatcher(
            poll_interval_seconds=0,
            batch_size=25,
        )

    # The first purge raised, dispatch still ran anyway, and the next due
    # interval started a fresh purge.
    assert dispatch_calls["n"] == 3
    assert purge_calls["n"] == 2


@pytest.mark.asyncio
async def test_b_cancelling_dispatcher_in_a_tick_cancels_a_blocked_purge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    purge_started = threading.Event()
    purge_release = threading.Event()
    dispatch_entered = asyncio.Event()

    def blocking_purge() -> dict[str, int]:
        purge_started.set()
        purge_release.wait(timeout=10)
        return {}

    async def blocking_dispatch(_db: Any, *, limit: int) -> int:
        # Cancellation is raised here, inside the dispatcher tick's try.
        dispatch_entered.set()
        await asyncio.Event().wait()
        return 0

    _install_harness(monkeypatch, blocking_purge, blocking_dispatch)

    dispatcher = asyncio.create_task(
        app_module._run_trigger_dispatcher(poll_interval_seconds=0, batch_size=25)
    )
    cleanup_task: asyncio.Task[Any] | None = None
    try:
        assert await asyncio.to_thread(purge_started.wait, 5)
        await asyncio.wait_for(dispatch_entered.wait(), timeout=5)
        cleanup_task = _find_cleanup_task()

        dispatcher.cancel()
        with pytest.raises(asyncio.CancelledError):
            await dispatcher

        for _ in range(20):
            if cleanup_task.done():
                break
            await asyncio.sleep(0)
        assert cleanup_task.cancelled()
    finally:
        purge_release.set()


@pytest.mark.asyncio
async def test_c_cancelling_dispatcher_in_poll_sleep_cancels_a_blocked_purge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    purge_started = threading.Event()
    purge_release = threading.Event()

    def blocking_purge() -> dict[str, int]:
        purge_started.set()
        purge_release.wait(timeout=10)
        return {}

    async def fake_dispatch(_db: Any, *, limit: int) -> int:
        return 0

    _install_harness(monkeypatch, blocking_purge, fake_dispatch)

    dispatcher = asyncio.create_task(
        app_module._run_trigger_dispatcher(poll_interval_seconds=3600, batch_size=25)
    )
    cleanup_task: asyncio.Task[Any] | None = None
    try:
        # purge_started implies the dispatcher already reached the poll sleep.
        assert await asyncio.to_thread(purge_started.wait, 5)
        cleanup_task = _find_cleanup_task()

        dispatcher.cancel()
        with pytest.raises(asyncio.CancelledError):
            await dispatcher

        for _ in range(20):
            if cleanup_task.done():
                break
            await asyncio.sleep(0)
        assert cleanup_task.cancelled()
    finally:
        purge_release.set()
