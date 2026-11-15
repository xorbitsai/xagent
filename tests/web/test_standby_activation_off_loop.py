"""Standby activation keeps the event loop serving while blocking work runs."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from unittest.mock import Mock

import pytest
from fastapi import FastAPI

from xagent.web import app as app_module
from xagent.web.services.standby import StandbyController
from xagent.web.startup_admission import (
    register_host_startup_admission,
    run_host_startup_admissions,
)

_BLOCK_SECONDS = 5.0


async def _wait_for(event: threading.Event) -> None:
    deadline = time.monotonic() + _BLOCK_SECONDS
    while not event.is_set():
        assert time.monotonic() < deadline, "blocking work never started"
        await asyncio.sleep(0.005)


def _patch_rest_of_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_admissions(_app) -> None:
        return None

    monkeypatch.setattr(app_module, "run_host_startup_admissions", no_admissions)
    monkeypatch.setattr(
        "xagent.web.services.trace_database.get_trace_database_runtime", lambda: None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("bound", [True, False])
async def test_database_init_does_not_block_the_event_loop(
    monkeypatch: pytest.MonkeyPatch, bound: bool
) -> None:
    started = threading.Event()
    release = threading.Event()
    ran_on: list[int] = []

    def blocking_init() -> None:
        ran_on.append(threading.get_ident())
        started.set()
        # Only a loop that keeps running can ever set ``release``.
        assert release.wait(_BLOCK_SECONDS), "event loop was blocked by init"

    monkeypatch.setattr(app_module, "complete_standby_db_init", blocking_init)
    monkeypatch.setattr(app_module, "init_db", blocking_init)
    _patch_rest_of_admission(monkeypatch)

    task = asyncio.ensure_future(
        app_module._admit_database_and_host(FastAPI(), bound=bound)
    )
    await _wait_for(started)
    ticks = 0
    for _ in range(5):
        await asyncio.sleep(0.01)
        ticks += 1
    assert ticks == 5 and not task.done()
    release.set()
    await asyncio.wait_for(task, _BLOCK_SECONDS)
    assert ran_on[0] != threading.get_ident()


@pytest.mark.asyncio
async def test_cancelled_activation_waits_for_database_init_to_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def init() -> None:
        started.set()
        release.wait(_BLOCK_SECONDS)
        finished.set()

    monkeypatch.setattr(app_module, "complete_standby_db_init", init)
    _patch_rest_of_admission(monkeypatch)

    task = asyncio.ensure_future(
        app_module._admit_database_and_host(FastAPI(), bound=True)
    )
    await _wait_for(started)
    task.cancel()
    await asyncio.sleep(0.05)
    # The cancel must not abandon a half-done migration/seed.
    assert not task.done() and not finished.is_set()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, _BLOCK_SECONDS)
    assert finished.is_set()


@pytest.mark.asyncio
async def test_sync_admission_runs_in_a_thread_in_registration_order() -> None:
    app = FastAPI()
    events: list[tuple[str, bool]] = []
    main = threading.get_ident()

    def sync_check() -> None:
        events.append(("sync", threading.get_ident() != main))

    async def async_check() -> None:
        events.append(("async", threading.get_ident() == main))

    register_host_startup_admission(app, sync_check)
    register_host_startup_admission(app, async_check)
    register_host_startup_admission(app, Mock())
    await run_host_startup_admissions(app)
    assert events == [("sync", True), ("async", True)]


@pytest.mark.asyncio
async def test_sync_admission_rejection_propagates_unchanged() -> None:
    app = FastAPI()

    def reject() -> None:
        raise RuntimeError("catalog refused")

    register_host_startup_admission(app, reject)
    with pytest.raises(RuntimeError, match="catalog refused"):
        await run_host_startup_admissions(app)


@pytest.mark.asyncio
async def test_activation_logs_step_duration_and_loop_lag(
    caplog: pytest.LogCaptureFixture,
) -> None:
    controller = StandbyController()

    async def stalls_the_loop() -> None:
        await asyncio.sleep(0.12)  # let the probe take a baseline tick
        time.sleep(0.3)  # synchronous work on the loop

    controller.begin()
    controller.defer("stalling step", stalls_the_loop)
    with caplog.at_level(logging.INFO, logger="xagent.web.services.standby"):
        await controller.activate()

    line = next(
        r.getMessage() for r in caplog.records if "stalling step took" in r.getMessage()
    )
    lag = float(line.rsplit("lag ", 1)[1].rstrip("s)"))
    assert lag >= 0.2, line
    assert any("activation complete in" in r.getMessage() for r in caplog.records)
