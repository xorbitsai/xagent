import asyncio
from importlib import import_module
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient


def test_startup_file_storage_sync_skips_when_disabled(monkeypatch):
    app_module = import_module("xagent.web.app")

    monkeypatch.setattr(
        app_module, "get_file_storage_startup_sync_enabled", lambda: False
    )
    sync_mock = Mock()
    monkeypatch.setattr(
        "xagent.web.services.startup_file_storage_sync.sync_registered_files_to_durable_storage",
        sync_mock,
    )

    app_module.run_startup_file_storage_sync()

    sync_mock.assert_not_called()


def test_startup_file_storage_sync_runs_when_enabled(monkeypatch):
    app_module = import_module("xagent.web.app")
    from xagent.web.services.startup_file_storage_sync import (
        StartupFileStorageSyncResult,
    )

    sync_mock = Mock(return_value=StartupFileStorageSyncResult())
    monkeypatch.setattr(
        app_module, "get_file_storage_startup_sync_enabled", lambda: True
    )
    monkeypatch.setattr(
        "xagent.web.services.startup_file_storage_sync.sync_registered_files_to_durable_storage",
        sync_mock,
    )

    app_module.run_startup_file_storage_sync()

    sync_mock.assert_called_once_with()


def test_startup_file_storage_sync_raises_when_registered_file_sync_fails(
    monkeypatch,
):
    app_module = import_module("xagent.web.app")
    from xagent.web.services.startup_file_storage_sync import (
        StartupFileStorageSyncResult,
    )

    monkeypatch.setattr(
        app_module, "get_file_storage_startup_sync_enabled", lambda: True
    )
    monkeypatch.setattr(
        "xagent.web.services.startup_file_storage_sync.sync_registered_files_to_durable_storage",
        Mock(return_value=StartupFileStorageSyncResult(scanned=3, failed=1)),
    )

    with pytest.raises(RuntimeError, match="failed for 1 registered file"):
        app_module.run_startup_file_storage_sync()


def test_startup_file_storage_sync_propagates_errors(monkeypatch):
    app_module = import_module("xagent.web.app")

    monkeypatch.setattr(
        app_module, "get_file_storage_startup_sync_enabled", lambda: True
    )
    monkeypatch.setattr(
        "xagent.web.services.startup_file_storage_sync.sync_registered_files_to_durable_storage",
        Mock(side_effect=RuntimeError("s3 unavailable")),
    )

    with pytest.raises(RuntimeError, match="s3 unavailable"):
        app_module.run_startup_file_storage_sync()


def test_startup_file_storage_sync_gate_waits_for_http_requests():
    app_module = import_module("xagent.web.app")

    test_app = FastAPI()
    test_app.add_middleware(app_module.FileStorageStartupSyncGateMiddleware)
    events: list[str] = []

    @test_app.on_event("startup")
    async def _startup() -> None:
        import asyncio

        async def _sync() -> None:
            events.append("sync-started")
            await asyncio.sleep(0.01)
            events.append("sync-completed")

        test_app.state.file_storage_startup_sync_task = asyncio.create_task(_sync())

    @test_app.get("/api/client")
    async def _client_route() -> dict[str, str]:
        events.append("route-called")
        return {"status": "served"}

    with TestClient(test_app) as client:
        response = client.get("/api/client")

    assert response.status_code == 200
    assert response.json() == {"status": "served"}
    assert events == ["sync-started", "sync-completed", "route-called"]


def test_startup_file_storage_sync_gate_exempts_health_while_sync_runs():
    app_module = import_module("xagent.web.app")

    test_app = FastAPI()
    test_app.add_middleware(app_module.FileStorageStartupSyncGateMiddleware)

    @test_app.on_event("startup")
    async def _startup() -> None:
        import asyncio

        async def _sync() -> None:
            await asyncio.sleep(10)

        test_app.state.file_storage_startup_sync_task = asyncio.create_task(_sync())

    @test_app.get("/health")
    async def _health() -> dict[str, str]:
        return {"status": "ok"}

    with TestClient(test_app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_startup_file_storage_sync_gate_exempts_options_while_sync_runs():
    app_module = import_module("xagent.web.app")

    test_app = FastAPI()
    test_app.add_middleware(app_module.FileStorageStartupSyncGateMiddleware)

    @test_app.on_event("startup")
    async def _startup() -> None:
        import asyncio

        async def _sync() -> None:
            await asyncio.sleep(10)

        test_app.state.file_storage_startup_sync_task = asyncio.create_task(_sync())

    @test_app.options("/api/client")
    async def _client_options() -> dict[str, str]:
        return {"status": "preflight"}

    with TestClient(test_app) as client:
        response = client.options("/api/client")

    assert response.status_code == 200
    assert response.json() == {"status": "preflight"}


def test_startup_file_storage_sync_gate_returns_503_when_sync_fails():
    app_module = import_module("xagent.web.app")

    test_app = FastAPI()
    test_app.add_middleware(app_module.FileStorageStartupSyncGateMiddleware)

    @test_app.on_event("startup")
    async def _startup() -> None:
        import asyncio

        async def _sync() -> None:
            raise RuntimeError("s3 unavailable")

        test_app.state.file_storage_startup_sync_task = asyncio.create_task(_sync())

    @test_app.get("/api/client")
    async def _client_route() -> dict[str, str]:
        return {"status": "served"}

    with TestClient(test_app, raise_server_exceptions=False) as client:
        response = client.get("/api/client")

    assert response.status_code == 503
    assert response.json()["detail"] == "Startup file storage sync failed"


@pytest.mark.asyncio
async def test_startup_file_storage_sync_wait_fails_fast_during_retry_loop():
    app_module = import_module("xagent.web.app")

    test_app = FastAPI()
    error = RuntimeError("s3 unavailable")

    async def _retry_loop() -> None:
        test_app.state.file_storage_startup_sync_error = error
        await asyncio.Event().wait()

    task = asyncio.create_task(_retry_loop())
    test_app.state.file_storage_startup_sync_task = task

    try:
        await asyncio.sleep(0)

        with pytest.raises(RuntimeError, match="s3 unavailable"):
            await asyncio.wait_for(
                app_module.wait_for_file_storage_startup_sync(test_app),
                timeout=30,
            )

        assert not task.done()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


def test_startup_file_storage_sync_gate_waits_for_websocket_connections():
    app_module = import_module("xagent.web.app")

    test_app = FastAPI()
    test_app.add_middleware(app_module.FileStorageStartupSyncGateMiddleware)
    events: list[str] = []

    @test_app.on_event("startup")
    async def _startup() -> None:
        import asyncio

        async def _sync() -> None:
            events.append("sync-started")
            await asyncio.sleep(0.01)
            events.append("sync-completed")

        test_app.state.file_storage_startup_sync_task = asyncio.create_task(_sync())

    @test_app.websocket("/ws/client")
    async def _websocket_route(websocket: WebSocket) -> None:
        events.append("websocket-called")
        await websocket.accept()
        await websocket.send_text("served")
        await websocket.close()

    with TestClient(test_app) as client:
        with client.websocket_connect("/ws/client") as websocket:
            assert websocket.receive_text() == "served"

    assert events == ["sync-started", "sync-completed", "websocket-called"]


def test_startup_file_storage_sync_task_is_scheduled_async(monkeypatch):
    app_module = import_module("xagent.web.app")

    monkeypatch.setattr(
        app_module, "get_file_storage_startup_sync_enabled", lambda: True
    )
    retry_coro = Mock()
    retry_runner_mock = Mock(return_value=retry_coro)
    sync_task = Mock()
    create_task_mock = Mock(return_value=sync_task)
    monkeypatch.setattr(
        app_module,
        "_run_file_storage_startup_sync_with_retries",
        retry_runner_mock,
    )
    monkeypatch.setattr(app_module.asyncio, "create_task", create_task_mock)

    test_app = FastAPI()
    task = app_module.start_file_storage_startup_sync_task(test_app)

    assert task == sync_task
    retry_runner_mock.assert_called_once_with(
        test_app,
        retry_interval_seconds=app_module.FILE_STORAGE_STARTUP_SYNC_RETRY_INTERVAL_SECONDS,
    )
    create_task_mock.assert_called_once_with(retry_coro)
    sync_task.add_done_callback.assert_called_once()
    assert test_app.state.file_storage_startup_sync_task == sync_task


@pytest.mark.asyncio
async def test_startup_file_storage_sync_task_cancellation_is_not_recorded_as_failure(
    monkeypatch,
):
    app_module = import_module("xagent.web.app")

    test_app = FastAPI()
    monkeypatch.setattr(
        app_module, "get_file_storage_startup_sync_enabled", lambda: True
    )

    async def _never_finishes(_fn):
        await asyncio.Event().wait()

    monkeypatch.setattr(app_module.asyncio, "to_thread", _never_finishes)

    task = app_module.start_file_storage_startup_sync_task(test_app)
    assert task is not None

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    assert test_app.state.file_storage_startup_sync_error is None
    assert test_app.state.file_storage_startup_sync_completed is False


@pytest.mark.asyncio
async def test_startup_file_storage_sync_task_retries_until_success(monkeypatch):
    app_module = import_module("xagent.web.app")

    test_app = FastAPI()
    attempts = {"count": 0}
    monkeypatch.setattr(
        app_module, "get_file_storage_startup_sync_enabled", lambda: True
    )

    def _sync_then_recover() -> None:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("s3 unavailable")

    monkeypatch.setattr(app_module, "run_startup_file_storage_sync", _sync_then_recover)

    task = app_module.start_file_storage_startup_sync_task(
        test_app,
        retry_interval_seconds=0,
    )
    assert task is not None

    await task

    assert attempts["count"] == 2
    assert test_app.state.file_storage_startup_sync_error is None
    assert test_app.state.file_storage_startup_sync_completed is True
