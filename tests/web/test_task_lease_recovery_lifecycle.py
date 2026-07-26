"""Backend lifecycle wiring for automatic task lease recovery."""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType

import pytest

from xagent.web import app as app_module


@pytest.mark.asyncio
async def test_task_lease_recovery_start_and_stop_owns_background_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def fake_recovery_loop(
        *,
        poll_interval_seconds: int,
        batch_size: int,
    ) -> None:
        assert poll_interval_seconds == 13
        assert batch_size == 17
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(
        app_module,
        "get_task_lease_recovery_interval_seconds",
        lambda: 13,
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "get_task_lease_recovery_batch_size",
        lambda: 17,
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "run_task_lease_recovery_loop",
        fake_recovery_loop,
        raising=False,
    )

    task = app_module.start_task_lease_recovery_task(app_module.app)
    assert task is app_module.app.state.task_lease_recovery_task
    await asyncio.wait_for(started.wait(), timeout=1)

    await app_module.stop_task_lease_recovery_task(app_module.app)

    assert stopped.is_set()
    assert task.cancelled()
    assert app_module.app.state.task_lease_recovery_task is None


@pytest.mark.asyncio
async def test_task_lease_recovery_start_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = 0
    stop = asyncio.Event()

    async def fake_recovery_loop(
        *,
        poll_interval_seconds: int,
        batch_size: int,
    ) -> None:
        nonlocal started
        started += 1
        await stop.wait()

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(
        app_module,
        "run_task_lease_recovery_loop",
        fake_recovery_loop,
        raising=False,
    )
    app_module.app.state.task_lease_recovery_task = None

    first = app_module.start_task_lease_recovery_task(app_module.app)
    second = app_module.start_task_lease_recovery_task(app_module.app)
    await asyncio.sleep(0)

    assert first is second
    assert started == 1

    stop.set()
    await app_module.stop_task_lease_recovery_task(app_module.app)


@pytest.mark.asyncio
async def test_task_lease_recovery_stop_consumes_completed_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = RuntimeError("recovery loop failed")

    async def failed_recovery_loop() -> None:
        raise failure

    task = asyncio.create_task(failed_recovery_loop())
    await asyncio.sleep(0)
    assert task.done()
    app_module.app.state.task_lease_recovery_task = task
    logged: list[BaseException] = []

    def fake_log_exception(
        _message: str,
        *_args: object,
        exc_info: BaseException | bool | None = None,
        **_kwargs: object,
    ) -> None:
        if isinstance(exc_info, BaseException):
            logged.append(exc_info)

    monkeypatch.setattr(app_module.logger, "error", fake_log_exception)

    await app_module.stop_task_lease_recovery_task(app_module.app)

    assert logged == [failure]
    assert app_module.app.state.task_lease_recovery_task is None


def test_task_lease_recovery_start_skips_automatic_task_under_pytest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test")

    assert app_module.start_task_lease_recovery_task(app_module.app) is None
    assert app_module.app.state.task_lease_recovery_task is None


@pytest.mark.asyncio
async def test_application_shutdown_stops_task_lease_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped_for = []
    shutdown_order: list[str] = []

    async def fake_stop(app_instance) -> None:
        stopped_for.append(app_instance)

    async def fake_stop_uploaded_file_recovery(app_instance) -> None:
        assert app_instance is app_module.app

    async def fake_shutdown_background_tasks() -> None:
        shutdown_order.append("background_tasks")

    async def fake_wait_for_heartbeat_idle() -> None:
        shutdown_order.append("heartbeat_idle")

    class _FakeChannel:
        enabled = False

        async def stop(self) -> None:
            return None

    class _FakeSandboxManager:
        async def cleanup(self) -> None:
            shutdown_order.append("sandbox")

    fake_telegram_bot = ModuleType("xagent.web.channels.telegram.bot")
    fake_telegram_bot.get_telegram_channel = lambda: _FakeChannel()
    fake_feishu_bot = ModuleType("xagent.web.channels.feishu.bot")
    fake_feishu_bot.get_feishu_channel = lambda: _FakeChannel()
    monkeypatch.setitem(
        sys.modules,
        "xagent.web.channels.telegram.bot",
        fake_telegram_bot,
    )
    monkeypatch.setitem(
        sys.modules,
        "xagent.web.channels.feishu.bot",
        fake_feishu_bot,
    )
    monkeypatch.setattr(app_module, "flush_langfuse", lambda: None)
    monkeypatch.setattr(app_module, "stop_task_lease_recovery_task", fake_stop)
    monkeypatch.setattr(
        app_module,
        "stop_uploaded_file_recovery_task",
        fake_stop_uploaded_file_recovery,
    )
    monkeypatch.setattr(
        "xagent.web.api.websocket.background_task_manager.shutdown",
        fake_shutdown_background_tasks,
    )
    monkeypatch.setattr(
        "xagent.web.services.task_lease_service.wait_for_heartbeat_manager_idle",
        fake_wait_for_heartbeat_idle,
    )
    monkeypatch.setattr(app_module, "_task_command_dispatcher_task", None)
    monkeypatch.setattr(app_module, "_sandbox_idle_sweep_task", None)
    monkeypatch.setattr(app_module, "_file_storage_startup_sync_task", None)
    monkeypatch.setattr(app_module, "_trigger_dispatcher_task", None)
    monkeypatch.setattr(app_module, "_migration_task", None)
    monkeypatch.setattr(
        "xagent.web.sandbox_manager.get_sandbox_manager",
        lambda: _FakeSandboxManager(),
    )
    app_module.app.state.metadata_rebuild_task = None
    if hasattr(app_module.app.state, "telegram_task"):
        delattr(app_module.app.state, "telegram_task")

    await app_module.shutdown_event()

    assert stopped_for == [app_module.app]
    assert shutdown_order == [
        "background_tasks",
        "heartbeat_idle",
        "sandbox",
    ]
