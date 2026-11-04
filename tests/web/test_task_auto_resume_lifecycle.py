"""Backend lifecycle wiring for the auto-resume sweeper."""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType

import pytest
from fastapi import FastAPI

from xagent.web import app as app_module


@pytest.mark.asyncio
async def test_auto_resume_start_and_stop_owns_background_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def fake_loop(*, poll_interval_seconds: int) -> None:
        assert poll_interval_seconds == 11
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(app_module, "get_task_auto_resume_poll_seconds", lambda: 11)
    monkeypatch.setattr(app_module, "run_task_auto_resume_loop", fake_loop)
    app = FastAPI()

    task = app_module.start_task_auto_resume_task(app)
    assert task is app.state.task_auto_resume_task
    # Idempotent while running.
    assert app_module.start_task_auto_resume_task(app) is task
    await asyncio.wait_for(started.wait(), timeout=1)

    await app_module.stop_task_auto_resume_task(app)

    assert stopped.is_set()
    assert task.cancelled()
    assert app.state.task_auto_resume_task is None


@pytest.mark.asyncio
async def test_auto_resume_stop_consumes_completed_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = RuntimeError("sweeper failed")

    async def failed() -> None:
        raise failure

    task = asyncio.create_task(failed())
    await asyncio.sleep(0)
    app = FastAPI()
    app.state.task_auto_resume_task = task
    logged: list[BaseException] = []
    monkeypatch.setattr(
        app_module.logger,
        "error",
        lambda *_a, exc_info=None, **_k: logged.append(exc_info),
    )

    await app_module.stop_task_auto_resume_task(app)

    assert logged == [failure]
    assert app.state.task_auto_resume_task is None


def test_auto_resume_start_skips_under_pytest() -> None:
    app = FastAPI()
    assert app_module.start_task_auto_resume_task(app) is None
    assert getattr(app.state, "task_auto_resume_task", None) is None


def test_web_role_does_not_run_the_sweeper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("XAGENT_TASK_EXECUTION_ROLE", "web")
    assert app_module.start_task_auto_resume_task(FastAPI()) is None


@pytest.mark.asyncio
async def test_application_shutdown_stops_the_sweeper_before_lease_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    async def record(name: str) -> None:
        order.append(name)

    class _FakeChannel:
        enabled = False

        async def stop(self) -> None:
            return None

    class _FakeSandboxManager:
        async def cleanup(self) -> None:
            return None

    for module_name, attribute in (
        ("xagent.web.channels.telegram.bot", "get_telegram_channel"),
        ("xagent.web.channels.feishu.bot", "get_feishu_channel"),
        ("xagent.web.channels.slack.bot", "get_slack_channel"),
    ):
        fake = ModuleType(module_name)
        setattr(fake, attribute, lambda: _FakeChannel())
        monkeypatch.setitem(sys.modules, module_name, fake)
    monkeypatch.setattr(app_module, "flush_langfuse", lambda: None)
    for name in (
        "stop_runtime_performance_monitor",
        "stop_orphan_upload_gc_task",
        "stop_retention_purge_task",
        "stop_task_cleanup_retry_task",
        "stop_uploaded_file_recovery_task",
    ):
        monkeypatch.setattr(app_module, name, lambda _app: record("other"))
    monkeypatch.setattr(
        app_module, "stop_task_auto_resume_task", lambda _app: record("auto_resume")
    )
    monkeypatch.setattr(
        app_module,
        "stop_task_lease_recovery_task",
        lambda _app: record("lease_recovery"),
    )
    monkeypatch.setattr(
        "xagent.web.services.task_execution.background_task_manager.shutdown",
        lambda: record("other"),
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
        "xagent.web.sandbox_manager.get_sandbox_manager",
        lambda: _FakeSandboxManager(),
    )
    app_module.app.state.metadata_rebuild_task = None
    for attribute in ("telegram_task", "slack_task"):
        if hasattr(app_module.app.state, attribute):
            monkeypatch.delattr(app_module.app.state, attribute)

    await app_module.shutdown_event()

    stops = [name for name in order if name != "other"]
    assert stops == ["auto_resume", "lease_recovery"]
