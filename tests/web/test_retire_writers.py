"""``retire_writers``: the writers stop once, the process keeps serving, shutdown skips it."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from contextlib import suppress

import pytest

from xagent.web import app as app_module
from xagent.web.services.release_hold import WritersRetired
from xagent.web.services.writer_workers import (
    active_workers,
    drain_workers,
    run_tracked,
)


@pytest.fixture
def steps(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    calls: list[str] = []

    def stopper(name: str):
        async def stop(*_args, **_kwargs) -> None:
            calls.append(name)

        return stop

    for name in (
        "stop_orphan_upload_gc_task",
        "stop_composio_reconcile_task",
        "stop_retention_purge_task",
        "stop_task_cleanup_retry_task",
        "stop_uploaded_file_recovery_task",
        "stop_task_auto_resume_task",
        "stop_task_lease_recovery_task",
    ):
        monkeypatch.setattr(app_module, name, stopper(name))
    monkeypatch.setattr(app_module, "_writers_retired", False)
    monkeypatch.setattr(app_module, "_task_command_dispatcher_task", None)
    monkeypatch.setattr(app_module, "_sandbox_idle_sweep_task", None)
    monkeypatch.setattr(app_module, "_file_storage_startup_sync_task", None)
    monkeypatch.setattr(app_module, "_trigger_dispatcher_task", None)
    monkeypatch.setattr(app_module, "_migration_task", None)
    monkeypatch.setattr(app_module, "_uploaded_file_reconcile_task", None)
    monkeypatch.setattr(app_module, "_retire_writers_lock", None)
    yield calls
    app_module.release_hold.reset_writers_retired()


async def test_retire_writers_stops_every_writer_once_and_is_idempotent(steps):
    assert not app_module.writers_retired()

    await app_module.retire_writers()
    first = list(steps)
    await app_module.retire_writers()

    assert app_module.writers_retired()
    # Socket writes that remain are refused from here on.
    assert app_module.release_hold.writers_retired
    assert first == [
        "stop_orphan_upload_gc_task",
        "stop_composio_reconcile_task",
        "stop_retention_purge_task",
        "stop_task_cleanup_retry_task",
        "stop_uploaded_file_recovery_task",
        "stop_task_auto_resume_task",
        "stop_task_lease_recovery_task",
    ]
    assert steps == first


async def test_a_step_that_fails_leaves_the_retire_unfinished_so_it_can_continue(
    steps, monkeypatch
):
    async def boom(*_a, **_k) -> None:
        raise RuntimeError("stop failed")

    monkeypatch.setattr(app_module, "stop_task_auto_resume_task", boom)
    with pytest.raises(RuntimeError):
        await app_module.retire_writers()
    assert not app_module.writers_retired()

    async def fine(*_a, **_k) -> None:
        steps.append("auto_resume")

    monkeypatch.setattr(app_module, "stop_task_auto_resume_task", fine)
    await app_module.retire_writers()

    assert app_module.writers_retired()
    assert steps[-2:] == ["auto_resume", "stop_task_lease_recovery_task"]


# -- socket writes are refused once the writers are retired -------------------


@pytest.fixture
def retired_hold():
    from xagent.web.services.release_hold import release_hold

    release_hold.mark_writers_retired()
    yield release_hold
    release_hold.reset_writers_retired()


def test_the_retired_flag_is_one_way_until_a_new_lifespan():
    from xagent.web.services.release_hold import release_hold

    assert not release_hold.writers_retired
    release_hold.mark_writers_retired()
    try:
        release_hold.reopen()
        assert release_hold.writers_retired
    finally:
        release_hold.reset_writers_retired()
    assert not release_hold.writers_retired


async def test_a_socket_command_is_refused_with_the_retryable_paused_error(
    retired_hold,
):
    from xagent.web.api import websocket as ws
    from xagent.web.services.client_error_messages import ClientErrorCode
    from xagent.web.services.task_command_transport import TaskCommandKind

    class _User:
        id = 1
        is_admin = False

    with pytest.raises(ws.ClientVisibleValidationError) as caught:
        await ws._enqueue_websocket_task_command(
            task_id=1, message_data={"user": _User()}, kind=TaskCommandKind.PAUSE
        )

    assert caught.value.error_code is ClientErrorCode.EXECUTION_PAUSED


async def test_execute_task_is_refused_before_it_acknowledges_or_stages(retired_hold):
    from xagent.web.api import websocket as ws

    sent: list[dict] = []

    class _User:
        id = 1
        is_admin = False

    async def record(message, _websocket):
        sent.append(message)

    original = ws.manager.send_personal_message
    ws.manager.send_personal_message = record  # type: ignore[assignment]
    try:
        await ws.handle_execute_task(object(), 1, {"user": _User()})  # type: ignore[arg-type]
    finally:
        ws.manager.send_personal_message = original  # type: ignore[assignment]

    assert all(m.get("type") != "execution_started" for m in sent)
    assert any(m.get("error_code") == "execution_paused" for m in sent)


# -- the retire waits for work that outlives the tasks it cancels ---------------


async def test_the_retire_waits_for_an_admitted_socket_write_and_fences_new_ones(steps):
    hold = app_module.release_hold
    interrupted: list[bool] = []
    token = hold.acquire_socket_write(lambda: interrupted.append(True))

    retiring = asyncio.create_task(app_module.retire_writers())
    await asyncio.sleep(0.2)

    assert not retiring.done()
    assert interrupted == [True]
    assert not app_module.writers_retired()
    with pytest.raises(WritersRetired):
        hold.acquire_socket_write()

    token.release()
    await retiring
    assert app_module.writers_retired()


async def test_a_socket_write_that_never_settles_fails_the_retire(steps, monkeypatch):
    monkeypatch.setattr(app_module, "RETIRE_SETTLE_SECONDS", 0.2)
    token = app_module.release_hold.acquire_socket_write()
    try:
        with pytest.raises(RuntimeError, match="socket writes"):
            await app_module.retire_writers()
        assert not app_module.writers_retired()
        # The fence stays up for the life of the lifespan.
        assert app_module.release_hold.writers_retired
    finally:
        token.release()

    await app_module.retire_writers()
    assert app_module.writers_retired()


async def test_a_detached_worker_thread_blocks_the_retire_until_it_returns(
    steps, monkeypatch
):
    release = threading.Event()
    started = threading.Event()

    def worker() -> None:
        started.set()
        release.wait(5)

    awaiting = asyncio.create_task(run_tracked(worker))
    while not started.is_set():
        await asyncio.sleep(0.01)
    # What stop_retention_purge_task does after its grace period.
    awaiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await awaiting
    assert active_workers() == 1

    monkeypatch.setattr(app_module, "RETIRE_SETTLE_SECONDS", 0.2)
    with pytest.raises(RuntimeError, match="writer threads"):
        await app_module.retire_writers()
    assert not app_module.writers_retired()

    release.set()
    monkeypatch.setattr(app_module, "RETIRE_SETTLE_SECONDS", 5)
    await app_module.retire_writers()
    assert app_module.writers_retired()
    assert active_workers() == 0


async def test_a_cancelled_await_before_the_thread_started_leaves_no_count():
    task = asyncio.create_task(run_tracked(lambda: None))
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    assert await drain_workers(1.0)
    assert active_workers() == 0


async def test_the_uploaded_file_reconcile_task_is_cancelled_by_the_retire(
    steps, monkeypatch
):
    started = asyncio.Event()

    async def background() -> None:
        started.set()
        await asyncio.sleep(60)

    task = asyncio.create_task(background())
    await started.wait()
    monkeypatch.setattr(app_module, "_uploaded_file_reconcile_task", task)

    await app_module.retire_writers()

    assert task.cancelled()


async def test_the_socket_command_enqueue_holds_a_token_until_it_settles(monkeypatch):
    from xagent.web.api import websocket as ws

    seen: list[int] = []

    async def fake(**_kwargs):
        seen.append(len(app_module.release_hold._socket_writes))
        raise RuntimeError("enqueue failed")

    monkeypatch.setattr(ws, "_enqueue_websocket_task_command_unguarded", fake)
    with pytest.raises(RuntimeError):
        await ws._enqueue_websocket_task_command(
            task_id=1, message_data={"user": object()}
        )

    assert seen == [1]
    assert not app_module.release_hold._socket_writes


async def test_an_asr_connection_to_a_retired_process_is_accepted_before_it_is_closed(
    monkeypatch,
):
    from starlette.websockets import WebSocketState

    from xagent.web.api import speech

    order: list[str] = []

    class _Socket:
        application_state = WebSocketState.CONNECTING

        async def close(self, code: int = 1000, reason: str = "") -> None:
            order.append(f"close:{code}")

    async def authenticated(*_a, **_k):
        return type("U", (), {"id": 1})()

    async def accept(_websocket) -> None:
        order.append("accept")

    monkeypatch.setattr(speech, "get_authenticated_user", authenticated)
    monkeypatch.setattr(speech, "accept_authenticated_websocket", accept)
    app_module.release_hold.mark_writers_retired()

    await speech.stream_asr.__wrapped__(_Socket(), token=None)  # type: ignore[attr-defined]

    assert order == ["accept", "close:1012"]


async def test_an_asr_session_is_a_socket_write_that_is_wound_down_by_the_fence():
    hold = app_module.release_hold
    task = asyncio.create_task(asyncio.sleep(60))
    token = hold.acquire_socket_write(task.cancel)

    hold.mark_writers_retired()
    with suppress(asyncio.CancelledError):
        await task

    assert task.cancelled()
    token.release()
    assert await hold.drain_socket_writes(1.0)
