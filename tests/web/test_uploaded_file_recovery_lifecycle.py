"""Backend lifecycle wiring for uploaded-file compensation recovery."""

from __future__ import annotations

import asyncio

import pytest

from xagent.web import app as app_module


@pytest.mark.asyncio
async def test_uploaded_file_recovery_start_and_stop_owns_background_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def fake_recovery_loop(
        *,
        poll_interval_seconds: int,
        stale_after_seconds: int,
        batch_size: int,
    ) -> None:
        assert poll_interval_seconds == 13
        assert stale_after_seconds == 29
        assert batch_size == 17
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(
        app_module,
        "get_uploaded_file_recovery_interval_seconds",
        lambda: 13,
    )
    monkeypatch.setattr(
        app_module,
        "get_uploaded_file_recovery_stale_seconds",
        lambda: 29,
    )
    monkeypatch.setattr(
        app_module,
        "get_uploaded_file_recovery_batch_size",
        lambda: 17,
    )
    monkeypatch.setattr(
        app_module,
        "run_uploaded_file_compensation_recovery_loop",
        fake_recovery_loop,
    )

    task = app_module.start_uploaded_file_recovery_task(app_module.app)
    assert task is app_module.app.state.uploaded_file_recovery_task
    await asyncio.wait_for(started.wait(), timeout=1)

    await app_module.stop_uploaded_file_recovery_task(app_module.app)

    assert stopped.is_set()
    assert task.cancelled()
    assert app_module.app.state.uploaded_file_recovery_task is None


@pytest.mark.asyncio
async def test_uploaded_file_recovery_start_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = 0
    stop = asyncio.Event()

    async def fake_recovery_loop(**_kwargs) -> None:
        nonlocal started
        started += 1
        await stop.wait()

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(
        app_module,
        "run_uploaded_file_compensation_recovery_loop",
        fake_recovery_loop,
    )
    app_module.app.state.uploaded_file_recovery_task = None

    first = app_module.start_uploaded_file_recovery_task(app_module.app)
    second = app_module.start_uploaded_file_recovery_task(app_module.app)
    await asyncio.sleep(0)

    assert first is second
    assert started == 1

    stop.set()
    await app_module.stop_uploaded_file_recovery_task(app_module.app)


def test_uploaded_file_recovery_start_skips_automatic_task_under_pytest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test")
    app_module.app.state.uploaded_file_recovery_task = None

    assert app_module.start_uploaded_file_recovery_task(app_module.app) is None
    assert app_module.app.state.uploaded_file_recovery_task is None
