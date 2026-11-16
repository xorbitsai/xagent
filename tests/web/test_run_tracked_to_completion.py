"""Cancellation contract of ``run_tracked_to_completion``.

However many times the awaiting task is cancelled, the helper returns only once
the worker thread is finished, it consumes the worker's outcome, and the
tracking count returns to where it was.
"""

from __future__ import annotations

import asyncio
import gc
import threading
from typing import Any

import pytest

from xagent.web.services import writer_workers
from xagent.web.services.standby import StandbyController, StandbyPhase
from xagent.web.services.writer_workers import run_tracked_to_completion

_WAIT = 10.0


async def _until(predicate: Any) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), _WAIT)


class _Worker:
    def __init__(self, *, fail: bool = False) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.fail = fail

    def __call__(self) -> str:
        self.started.set()
        self.release.wait(_WAIT)
        self.finished.set()
        if self.fail:
            raise RuntimeError("worker failed after cancellation")
        return "done"


@pytest.mark.asyncio
@pytest.mark.parametrize("cancels", [1, 2, 5])
async def test_every_cancel_still_waits_for_the_worker(cancels: int) -> None:
    baseline = writer_workers._active
    worker = _Worker()
    task = asyncio.ensure_future(run_tracked_to_completion(worker))
    await _until(worker.started.is_set)
    for _ in range(cancels):
        task.cancel()
        await asyncio.sleep(0.02)
    assert not task.done() and not worker.finished.is_set()
    worker.release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, _WAIT)
    assert worker.finished.is_set()
    await _until(lambda: writer_workers._active == baseline)


@pytest.mark.asyncio
async def test_activation_failure_handling_waits_for_a_twice_cancelled_worker() -> None:
    controller = StandbyController()
    worker = _Worker()
    hooks: list[str] = []

    async def hook() -> None:
        hooks.append("failure hook")

    async def step() -> None:
        await run_tracked_to_completion(worker)

    controller.begin()
    controller.on_failure(hook)
    controller.defer("init", step)
    task = asyncio.ensure_future(controller.activate())
    await _until(worker.started.is_set)
    task.cancel()
    await asyncio.sleep(0.02)
    task.cancel()  # second cancel, while the helper is joining the worker
    await asyncio.sleep(0.05)
    assert not task.done() and hooks == []
    assert controller.phase is StandbyPhase.ACTIVATING
    worker.release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, _WAIT)
    assert worker.finished.is_set()
    assert hooks == ["failure hook"]
    assert controller.phase is StandbyPhase.FAILED


@pytest.mark.asyncio
async def test_worker_error_after_cancel_is_consumed_and_cancel_propagates() -> None:
    loop = asyncio.get_running_loop()
    seen: list[dict[str, Any]] = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, ctx: seen.append(ctx))
    try:
        worker = _Worker(fail=True)
        task = asyncio.ensure_future(run_tracked_to_completion(worker))
        await _until(worker.started.is_set)
        task.cancel()
        await asyncio.sleep(0.02)
        worker.release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, _WAIT)
        del task
        gc.collect()
        await asyncio.sleep(0.05)
    finally:
        loop.set_exception_handler(previous)
    assert seen == []


@pytest.mark.asyncio
async def test_cancel_right_after_submit_keeps_the_count_consistent() -> None:
    baseline = writer_workers._active
    ran = threading.Event()

    def work() -> None:
        ran.set()

    task = asyncio.ensure_future(run_tracked_to_completion(work))
    await asyncio.sleep(0)  # the thread was submitted, maybe not yet started
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, _WAIT)
    assert ran.is_set()  # started or not, the work is finished, never half-done
    await _until(lambda: writer_workers._active == baseline)


@pytest.mark.asyncio
async def test_normal_result_and_error_pass_through() -> None:
    assert await run_tracked_to_completion(lambda: 7) == 7

    def boom() -> None:
        raise ValueError("x")

    with pytest.raises(ValueError):
        await run_tracked_to_completion(boom)
