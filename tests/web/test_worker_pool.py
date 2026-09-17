"""The combined CLI owns startup ordering and the lifetime of its child group."""

import asyncio
import json
import multiprocessing
import os
import signal
import sys
import time
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI

from xagent import config
from xagent.web import __main__ as web_main
from xagent.web import worker_pool


@pytest.mark.parametrize("count", [None, "2"])
def test_cli_selects_pool_only_when_configured(monkeypatch, count):
    monkeypatch.delenv(config.WORKER_COUNT, raising=False)
    if count is not None:
        monkeypatch.setenv(config.WORKER_COUNT, count)
    monkeypatch.setattr(
        web_main,
        "parse_args",
        lambda: Namespace(
            host="127.0.0.2", port=8123, reload=False, debug=False, log_level="warning"
        ),
    )
    monkeypatch.setattr(web_main, "setup_logging", Mock())
    monkeypatch.setattr(web_main, "warn_if_example_jwt_config", Mock())
    run_server = Mock()
    run_pool = Mock()
    monkeypatch.setattr(web_main.uvicorn, "run", run_server)
    monkeypatch.setattr(worker_pool, "run_combined_worker_pool", run_pool)
    monkeypatch.setattr(sys, "argv", ["xagent-web"])
    web_main.main()
    if count is None:
        run_server.assert_called_once_with(
            "xagent.web.app:app",
            host="127.0.0.2",
            port=8123,
            reload=False,
            log_level="warning",
        )
        run_pool.assert_not_called()
    else:
        run_pool.assert_called_once_with(
            worker_count=2, host="127.0.0.2", port=8123, log_level="warning"
        )
        run_server.assert_not_called()


@pytest.mark.parametrize("count,reload", [("0", False), ("2", True)])
def test_cli_rejects_invalid_pool_options_before_start(monkeypatch, count, reload):
    monkeypatch.setenv(config.WORKER_COUNT, count)
    monkeypatch.setattr(
        web_main,
        "parse_args",
        lambda: Namespace(
            host="127.0.0.1", port=8000, reload=reload, debug=False, log_level=None
        ),
    )
    monkeypatch.setattr(web_main, "setup_logging", Mock())
    monkeypatch.setattr(web_main, "warn_if_example_jwt_config", Mock())
    run_server, run_pool = Mock(), Mock()
    monkeypatch.setattr(web_main.uvicorn, "run", run_server)
    monkeypatch.setattr(worker_pool, "run_combined_worker_pool", run_pool)
    monkeypatch.setattr(sys, "argv", ["xagent-web"])
    with pytest.raises(SystemExit) as error:
        web_main.main()
    assert error.value.code == 1
    run_server.assert_not_called()
    run_pool.assert_not_called()


@pytest.mark.parametrize(
    "shared,role", [(False, "combined"), (True, "web"), (True, "worker")]
)
def test_pool_rejects_incompatible_roles_before_spawning(monkeypatch, shared, role):
    monkeypatch.setenv(config.SHARED_TASK_EXECUTION_ENABLED, str(shared).lower())
    monkeypatch.setenv(config.TASK_EXECUTION_ROLE, role)
    spawn = Mock()
    monkeypatch.setattr(worker_pool.multiprocessing, "get_context", spawn)
    with pytest.raises(
        ValueError, match="requires shared execution and the combined role"
    ):
        worker_pool.run_combined_worker_pool(
            worker_count=2, host="127.0.0.1", port=8000, log_level=None
        )
    spawn.assert_not_called()


def test_web_child_signals_after_startup_and_only_serves_web(monkeypatch):
    monkeypatch.setenv(config.TASK_EXECUTION_ROLE, "combined")
    monkeypatch.setenv(config.WORKER_COUNT, "2")
    monkeypatch.setenv(config.SHARED_TASK_EXECUTION_ENABLED, "true")
    monkeypatch.setenv(config.CHANNEL_INGRESS_ENABLED, "true")
    from xagent.web.services.task_execution_host import consumes_task_commands

    order = []
    app = FastAPI()
    app.router.add_event_handler(
        "startup", lambda: order.append("database and admission")
    )
    monkeypatch.setitem(sys.modules, "xagent.web.app", SimpleNamespace(app=app))
    reader, writer = multiprocessing.get_context("spawn").Pipe(duplex=False)

    def serve(application, **kwargs):
        assert config.get_task_execution_role() == "web"
        assert config.get_worker_count() is None
        assert not consumes_task_commands()
        assert config.get_channel_ingress_enabled()
        assert kwargs == dict(host="127.0.0.1", port=8123, log_level=None, workers=1)
        assert not reader.poll()
        asyncio.run(application.router.startup())
        assert order == ["database and admission"]
        assert reader.poll()
        assert reader.recv_bytes() == b""

    monkeypatch.setattr(web_main.uvicorn, "run", serve)
    monkeypatch.setattr(worker_pool, "setup_logging", Mock())
    try:
        worker_pool._run_web("127.0.0.1", 8123, None, writer)
    finally:
        reader.close()
        writer.close()


def test_worker_child_reuses_standalone_runtime_with_isolated_identity(monkeypatch):
    monkeypatch.setenv(config.TASK_EXECUTION_ROLE, "combined")
    monkeypatch.setenv(config.WORKER_COUNT, "2")
    monkeypatch.setenv(config.SHARED_TASK_EXECUTION_ENABLED, "true")
    monkeypatch.setenv(config.CHANNEL_INGRESS_ENABLED, "true")
    monkeypatch.setenv(config.SANDBOX_WORKER_ID, "backend")
    monkeypatch.setenv(config.ENCRYPTION_KEY, "inherited-key")

    async def run():
        assert config.get_task_execution_role() == "worker"
        assert config.get_worker_count() is None
        assert not config.get_channel_ingress_enabled()
        assert config.get_sandbox_worker_id() == "backend-worker-2"
        assert os.environ[config.ENCRYPTION_KEY] == "inherited-key"

    entrypoint = AsyncMock(side_effect=run)
    monkeypatch.setitem(
        sys.modules, "xagent.web.worker", SimpleNamespace(_main=entrypoint)
    )
    monkeypatch.setattr(worker_pool, "setup_logging", Mock())
    worker_pool._run_worker("backend-worker-2", None)
    entrypoint.assert_awaited_once()


def _test_web(host, port, log_level, ready):
    directory = Path(os.environ["XAGENT_TEST_POOL_DIR"])
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    record = directory / "web.tmp"
    record.write_text(json.dumps({"pid": os.getpid()}))
    record.replace(directory / "web.json")
    while not stopped and not (directory / "allow-ready").exists():
        time.sleep(0.02)
    if stopped:
        return
    if os.environ["XAGENT_TEST_POOL_SCENARIO"] == "web-failure":
        raise SystemExit(7)
    ready.send_bytes(b"")
    ready.close()
    while not stopped:
        time.sleep(0.02)
    (directory / "web-stopped").touch()


def _test_worker(worker_id, log_level):
    directory = Path(os.environ["XAGENT_TEST_POOL_DIR"])
    scenario = os.environ["XAGENT_TEST_POOL_SCENARIO"]
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    handler = signal.SIG_IGN if scenario == "stubborn" else stop
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    record = directory / f"{worker_id}.tmp"
    record.write_text(json.dumps({"pid": os.getpid(), "id": worker_id}))
    record.replace(directory / f"{worker_id}.json")
    while not stopped:
        if (
            scenario == "worker-failure"
            and worker_id.endswith("-2")
            and (directory / "fail-worker").exists()
        ):
            raise SystemExit(7)
        time.sleep(0.02)
    (directory / f"{worker_id}-stopped").touch()


def _test_launcher(directory, scenario):
    os.environ.update(
        {
            "XAGENT_TEST_POOL_DIR": directory,
            "XAGENT_TEST_POOL_SCENARIO": scenario,
            config.SHARED_TASK_EXECUTION_ENABLED: "true",
            config.TASK_EXECUTION_ROLE: "combined",
            config.WORKER_COUNT: "2",
            config.REDIS_URL: "redis://localhost:6379/0",
            config.ENCRYPTION_KEY: Fernet.generate_key().decode(),
            config.SANDBOX_WORKER_ID: "backend",
        }
    )
    worker_pool._run_web = _test_web
    worker_pool._run_worker = _test_worker
    worker_pool.SHUTDOWN_TIMEOUT_SECONDS = 0.5
    old_handlers = {
        sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    if scenario == "spawn-failure":
        context = multiprocessing.get_context("spawn")
        created = 0

        def process(**kwargs):
            nonlocal created
            created += 1
            child = context.Process(**kwargs)
            if created == 3:

                def fail():
                    raise OSError("synthetic process start failure")

                child.start = fail
            return child

        worker_pool.multiprocessing.get_context = lambda _method: SimpleNamespace(
            Pipe=context.Pipe, Process=process
        )
    try:
        worker_pool.run_combined_worker_pool(
            worker_count=2, host="127.0.0.1", port=8123, log_level=None
        )
    finally:
        assert not multiprocessing.active_children()
        assert all(
            signal.getsignal(sig) == handler for sig, handler in old_handlers.items()
        )
        Path(directory, "reaped").touch()


def _wait_for(path, process):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if path.exists():
            return
        assert process.is_alive(), (
            f"launcher exited {process.exitcode} before {path.name}"
        )
        time.sleep(0.02)
    pytest.fail(f"Timed out waiting for {path.name}")


@pytest.mark.parametrize(
    "scenario",
    [
        "term",
        "interrupt",
        "stop-startup",
        "web-failure",
        "worker-failure",
        "spawn-failure",
        "stubborn",
    ],
)
def test_real_process_group_lifecycle(tmp_path, scenario):
    process = multiprocessing.get_context("spawn").Process(
        target=_test_launcher, args=(str(tmp_path), scenario)
    )
    process.start()
    try:
        _wait_for(tmp_path / "web.json", process)
        # No worker can start before the web startup handshake.
        assert not list(tmp_path.glob("backend-worker-*.json"))
        if scenario == "stop-startup":
            os.kill(process.pid, signal.SIGTERM)
        else:
            (tmp_path / "allow-ready").touch()
            if scenario not in {"web-failure", "spawn-failure"}:
                for index in (1, 2):
                    _wait_for(tmp_path / f"backend-worker-{index}.json", process)
                rows = [
                    json.loads(path.read_text()) for path in tmp_path.glob("*.json")
                ]
                assert len({row["pid"] for row in rows}) == 3
                assert {row.get("id") for row in rows} == {
                    None,
                    "backend-worker-1",
                    "backend-worker-2",
                }
                if scenario == "worker-failure":
                    (tmp_path / "fail-worker").touch()
                else:
                    os.kill(
                        process.pid,
                        signal.SIGINT if scenario == "interrupt" else signal.SIGTERM,
                    )
        process.join(timeout=15)
        assert not process.is_alive()
        assert (process.exitcode != 0) == (
            scenario in {"web-failure", "worker-failure", "spawn-failure"}
        )
        assert (tmp_path / "reaped").exists()
        if scenario in {"term", "interrupt"}:
            assert (tmp_path / "web-stopped").exists()
            assert all(
                (tmp_path / f"backend-worker-{i}-stopped").exists() for i in (1, 2)
            )
        if scenario in {"stop-startup", "web-failure"}:
            assert not list(tmp_path.glob("backend-worker-*.json"))
        for path in tmp_path.glob("*.json"):
            with pytest.raises(ProcessLookupError):
                os.kill(json.loads(path.read_text())["pid"], 0)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=3)
        if process.is_alive():
            process.kill()
            process.join()
        for path in tmp_path.glob("*.json"):
            try:
                os.kill(json.loads(path.read_text())["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.close()
