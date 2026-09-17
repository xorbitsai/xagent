"""One CLI-owned web process and a fixed number of standalone task workers."""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import signal
import time
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from types import FrameType
from typing import cast

from ..config import (
    CHANNEL_INGRESS_ENABLED,
    SANDBOX_WORKER_ID,
    TASK_EXECUTION_ROLE,
    WORKER_COUNT,
    get_sandbox_worker_id,
    get_shared_task_execution_enabled,
    get_task_execution_role,
    validate_task_execution_host_config,
)
from .logging_config import LogLevel, setup_logging

logger = logging.getLogger(__name__)
SHUTDOWN_TIMEOUT_SECONDS = 30.0


def _run_web(host: str, port: int, log_level: str | None, ready: Connection) -> None:
    os.environ[TASK_EXECUTION_ROLE] = "web"
    os.environ.pop(WORKER_COUNT, None)
    setup_logging(level=cast(LogLevel, log_level) if log_level else None, force=True)

    import uvicorn

    from .app import app

    def notify_ready() -> None:
        ready.send_bytes(b"")
        ready.close()

    # Existing startup completes migrations and admission before signaling.
    app.router.add_event_handler("startup", notify_ready)
    with ready:
        uvicorn.run(app, host=host, port=port, log_level=log_level, workers=1)


def _run_worker(worker_id: str | None, log_level: str | None) -> None:
    os.environ[TASK_EXECUTION_ROLE] = "worker"
    os.environ[CHANNEL_INGRESS_ENABLED] = "false"
    os.environ.pop(WORKER_COUNT, None)
    if worker_id is not None:
        os.environ[SANDBOX_WORKER_ID] = worker_id
    setup_logging(level=cast(LogLevel, log_level) if log_level else None, force=True)

    from .worker import _main

    asyncio.run(_main())


def _stop_processes(processes: list[BaseProcess]) -> None:
    # One shared deadline bounds shutdown regardless of the configured count.
    deadline = time.monotonic() + SHUTDOWN_TIMEOUT_SECONDS
    for process in processes:
        if process.pid is not None and process.is_alive():
            process.terminate()
    for process in processes:
        if process.pid is not None:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.pid is not None and process.is_alive():
            logger.warning("Force-stopping %s after shutdown timeout", process.name)
            process.kill()
    for process in processes:
        if process.pid is not None:
            process.join()
        process.close()


def run_combined_worker_pool(
    *, worker_count: int, host: str, port: int, log_level: str | None
) -> None:
    """Supervise one web host and N workers; any child exit stops the group."""
    if (
        not get_shared_task_execution_enabled()
        or get_task_execution_role() != "combined"
    ):
        raise ValueError(
            "XAGENT_WORKER_COUNT requires shared execution and the combined role"
        )
    validate_task_execution_host_config()
    base_worker_id = (
        get_sandbox_worker_id() if os.getenv(SANDBOX_WORKER_ID, "").strip() else None
    )
    context = multiprocessing.get_context("spawn")
    ready_reader, ready_writer = context.Pipe(duplex=False)
    stop_requested = False
    processes: list[BaseProcess] = []

    def request_stop(_signum: int, _frame: FrameType | None) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_handlers = {
        sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        web = context.Process(
            name="xagent-web",
            target=_run_web,
            args=(host, port, log_level, ready_writer),
        )
        processes.append(web)
        web.start()
        ready_writer.close()
        workers_started = False
        while not stop_requested:
            for process in processes:
                if process.exitcode is not None:
                    raise RuntimeError(
                        f"{process.name} exited unexpectedly (code {process.exitcode})"
                    )
            if not workers_started and ready_reader.poll():
                try:
                    ready_reader.recv_bytes()
                except EOFError:
                    raise RuntimeError(
                        "Web process exited before completing startup"
                    ) from None
                ready_reader.close()
                for index in range(1, worker_count + 1):
                    if stop_requested:
                        break
                    worker_id = (
                        f"{base_worker_id}-worker-{index}" if base_worker_id else None
                    )
                    process = context.Process(
                        name=f"xagent-worker-{index}",
                        target=_run_worker,
                        args=(worker_id, log_level),
                    )
                    processes.append(process)
                    process.start()
                workers_started = True
                logger.info("Started %s task worker processes", len(processes) - 1)
            time.sleep(0.1)
    finally:
        ready_reader.close()
        ready_writer.close()
        try:
            _stop_processes(processes)
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
