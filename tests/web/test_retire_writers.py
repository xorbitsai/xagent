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

    async def stop_chat_channels() -> None:
        calls.append("stop_chat_channels")

    monkeypatch.setattr(app_module, "_stop_chat_channels", stop_chat_channels)
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
    assert app_module.release_hold.socket_writes_fenced
    assert first == [
        "stop_orphan_upload_gc_task",
        "stop_composio_reconcile_task",
        "stop_retention_purge_task",
        "stop_task_cleanup_retry_task",
        "stop_uploaded_file_recovery_task",
        "stop_task_auto_resume_task",
        "stop_task_lease_recovery_task",
        "stop_chat_channels",
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
    assert steps[-3:] == [
        "auto_resume",
        "stop_task_lease_recovery_task",
        "stop_chat_channels",
    ]


# -- socket writes are refused once the writers are retired -------------------


@pytest.fixture
def retired_hold():
    from xagent.web.services.release_hold import release_hold

    release_hold.mark_writers_retired()
    yield release_hold
    release_hold.reset_writers_retired()


def test_the_retired_flag_is_one_way_until_a_new_lifespan():
    from xagent.web.services.release_hold import release_hold

    assert not release_hold.socket_writes_fenced
    release_hold.mark_writers_retired()
    try:
        release_hold.reopen()
        assert release_hold.socket_writes_fenced
    finally:
        release_hold.reset_writers_retired()
    assert not release_hold.socket_writes_fenced


async def test_a_work_starting_socket_command_is_refused_with_the_paused_error(
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
            task_id=1, message_data={"user": _User()}, kind=TaskCommandKind.MESSAGE
        )

    assert caught.value.error_code is ClientErrorCode.EXECUTION_PAUSED


@pytest.mark.parametrize("name", ["PAUSE", "RESUME", "CANCEL"])
async def test_a_steering_command_still_gets_through_the_fence(
    retired_hold, monkeypatch, name
):
    from xagent.web.api import websocket as ws
    from xagent.web.services.task_command_transport import TaskCommandKind

    class _User:
        id = 1
        is_admin = False

    sentinel = object()
    monkeypatch.setattr(
        ws, "_enqueue_websocket_task_command_sync", lambda **_k: sentinel
    )

    result = await ws._enqueue_websocket_task_command(
        task_id=1, message_data={"user": _User()}, kind=TaskCommandKind[name]
    )

    assert result is sentinel
    assert not retired_hold._socket_writes


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
        assert app_module.release_hold.socket_writes_fenced
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


async def test_a_cancelled_await_before_the_thread_started_leaves_no_count(
    monkeypatch,
):
    from xagent.web.services import writer_workers

    # The thread never starts: the executor hand-off is replaced by a wait that
    # only the cancellation ends, so the count has been taken when it arrives.
    async def never_started(*_a, **_k):
        await asyncio.Event().wait()

    monkeypatch.setattr(writer_workers.asyncio, "to_thread", never_started)
    ran: list[bool] = []
    task = asyncio.create_task(run_tracked(lambda: ran.append(True)))
    await asyncio.sleep(0)
    assert active_workers() == 1

    task.cancel()
    with suppress(asyncio.CancelledError):
        await task

    assert active_workers() == 0
    assert await drain_workers(1.0)
    assert ran == []


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


async def test_the_socket_command_enqueue_holds_a_token_until_its_thread_settles(
    monkeypatch,
):
    from xagent.web.api import websocket as ws
    from xagent.web.services.task_command_transport import TaskCommandKind

    seen: list[tuple[int, int]] = []

    def fake_enqueue(**_kwargs):
        seen.append((len(app_module.release_hold._socket_writes), active_workers()))
        raise RuntimeError("enqueue failed")

    class _User:
        id = 1
        is_admin = False

    monkeypatch.setattr(ws, "_enqueue_websocket_task_command_sync", fake_enqueue)
    with pytest.raises(RuntimeError):
        await ws._enqueue_websocket_task_command(
            task_id=1,
            message_data={"user": _User()},
            kind=TaskCommandKind.MESSAGE,
        )

    # One token and one tracked worker while it ran; both gone afterwards.
    assert seen == [(1, 1)]
    assert not app_module.release_hold._socket_writes
    assert active_workers() == 0


async def test_an_unauthenticated_enqueue_is_refused_without_a_token():
    from xagent.web.api import websocket as ws
    from xagent.web.services.task_command_transport import TaskCommandKind

    with pytest.raises(ws.ClientVisibleValidationError):
        await ws._enqueue_websocket_task_command(
            task_id=1, message_data={}, kind=TaskCommandKind.MESSAGE
        )
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


class _AsrSocket:
    """Just enough of a WebSocket for ``stream_asr`` up to its first receive."""

    from starlette.websockets import WebSocketState as _State

    application_state = _State.CONNECTING

    def __init__(self) -> None:
        self.closed: list[int] = []

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed.append(code)
        self.application_state = self._State.DISCONNECTED


async def test_a_wound_down_asr_session_closes_1012_and_releases_its_token(
    monkeypatch,
):
    from xagent.web.api import speech

    blocked = asyncio.Event()

    async def authenticated(*_a, **_k):
        return type("U", (), {"id": 1})()

    async def accept(_websocket) -> None:
        return None

    async def receive(_websocket):
        blocked.set()
        await asyncio.sleep(60)

    monkeypatch.setattr(speech, "get_authenticated_user", authenticated)
    monkeypatch.setattr(speech, "accept_authenticated_websocket", accept)
    monkeypatch.setattr(speech, "_receive_asr_message", receive)
    socket = _AsrSocket()

    handler = asyncio.create_task(
        speech.stream_asr.__wrapped__(socket, token=None)  # type: ignore[attr-defined]
    )
    await blocked.wait()
    assert len(app_module.release_hold._socket_writes) == 1

    app_module.release_hold.mark_writers_retired()
    await handler  # the fence cancel does not escape the handler

    assert not handler.cancelled()
    assert socket.closed == [1012]
    assert await app_module.release_hold.drain_socket_writes(1.0)


async def test_the_usage_delivery_of_a_scope_is_a_tracked_worker(monkeypatch):
    from xagent.web.tracking import standalone_usage

    release = threading.Event()
    started = threading.Event()

    def blocking_deliver(_self, _details) -> None:
        started.set()
        release.wait(5)

    async def no_gate(*_a, **_k) -> None:
        return None

    monkeypatch.setattr(standalone_usage, "check_standalone_gate_async", no_gate)
    monkeypatch.setattr(
        standalone_usage.UsageScope, "_pending", lambda _self: [{"x": 1}]
    )
    monkeypatch.setattr(standalone_usage.UsageScope, "_deliver", blocking_deliver)

    async def metered() -> None:
        async with standalone_usage.usage_scope(
            1, usage_source="speech_transcribe", model_ids=()
        ):
            pass

    exiting = asyncio.create_task(metered())
    while not started.is_set():
        await asyncio.sleep(0.01)
    exiting.cancel()
    with suppress(asyncio.CancelledError):
        await exiting

    assert active_workers() == 1
    release.set()
    assert await drain_workers(5.0)


# -- the fence fires its interrupts once ----------------------------------------


def test_a_retry_of_the_fence_does_not_interrupt_a_winding_down_write_again():
    from xagent.web.services.release_hold import ReleaseHold

    hold = ReleaseHold()
    interrupted: list[bool] = []
    hold.acquire_socket_write(lambda: interrupted.append(True))

    hold.mark_writers_retired()
    hold.mark_writers_retired()

    assert interrupted == [True]


# -- the chokepoint: nothing is dispatched once the fence is up -----------------


async def test_a_fenced_dispatch_only_notifies_the_dispatcher(
    retired_hold, monkeypatch
):
    from xagent.web.services import task_command_transport as transport

    dispatched: list[int] = []
    notified: list[bool] = []

    async def dispatch_one(_executor, *, command_db_id):
        dispatched.append(command_db_id)
        return True

    monkeypatch.setattr(transport, "dispatch_one_task_command", dispatch_one)
    monkeypatch.setattr(transport, "consumes_task_commands", lambda: True)
    monkeypatch.setattr(
        transport, "notify_task_command_dispatcher", lambda: notified.append(True)
    )

    await transport.dispatch_task_command_promptly(object(), command_db_id=7)  # type: ignore[arg-type]

    assert dispatched == []
    assert notified == [True]
    assert transport.prompt_dispatch_count() == 0


async def test_an_unfenced_dispatch_still_runs_the_command(monkeypatch):
    from xagent.web.services import task_command_transport as transport

    dispatched: list[int] = []

    async def dispatch_one(_executor, *, command_db_id):
        dispatched.append(command_db_id)
        return True

    monkeypatch.setattr(transport, "dispatch_one_task_command", dispatch_one)
    monkeypatch.setattr(transport, "consumes_task_commands", lambda: True)

    await transport.dispatch_task_command_promptly(object(), command_db_id=7)  # type: ignore[arg-type]

    assert dispatched == [7]


async def test_the_retire_waits_for_a_prompt_dispatch_that_started_before_the_fence(
    steps, monkeypatch
):
    from xagent.web.services import task_command_transport as transport

    monkeypatch.setattr(app_module, "RETIRE_SETTLE_SECONDS", 0.2)
    gate = asyncio.Event()

    async def stubborn() -> bool:
        # A dispatch that is cancelled but cleans up for a moment.
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await gate.wait()
            raise
        return True

    task = asyncio.create_task(stubborn())
    await asyncio.sleep(0)
    transport._prompt_dispatch_tasks.add(task)
    try:
        with pytest.raises(RuntimeError, match="command dispatches"):
            await app_module.retire_writers()
        assert not app_module.writers_retired()

        gate.set()
        monkeypatch.setattr(app_module, "RETIRE_SETTLE_SECONDS", 5)
        await app_module.retire_writers()
        assert app_module.writers_retired()
        assert task.cancelled()
    finally:
        gate.set()
        transport._prompt_dispatch_tasks.discard(task)


# -- every startup-owned writer thread is tracked --------------------------------


async def test_the_task_cleanup_retry_batch_runs_on_tracked_workers(monkeypatch):
    from xagent.web.services import task_cleanup_obligations as cleanup

    release = threading.Event()
    started = threading.Event()

    def blocking_candidates(*_a, **_k):
        started.set()
        release.wait(5)
        return []

    monkeypatch.setattr(cleanup, "_due_candidates_sync", blocking_candidates)
    batch = asyncio.create_task(cleanup.run_cleanup_obligation_batch(lambda: None))  # type: ignore[arg-type,return-value]
    while not started.is_set():
        await asyncio.sleep(0.01)
    batch.cancel()
    with suppress(asyncio.CancelledError):
        await batch

    assert active_workers() == 1
    release.set()
    assert await drain_workers(5.0)


async def test_the_composio_reconcile_pass_runs_on_a_tracked_worker(monkeypatch):
    from xagent import web as _web  # noqa: F401
    from xagent.web import composio_auth

    release = threading.Event()
    started = threading.Event()
    stop = threading.Event()

    def blocking_pass(*, stop_event) -> None:
        started.set()
        release.wait(5)

    monkeypatch.setattr(
        composio_auth, "reconcile_orphan_accounts_isolated", blocking_pass
    )
    loop_task = asyncio.create_task(
        composio_auth.run_reconcile_loop(interval_seconds=60, stop_event=stop)
    )
    while not started.is_set():
        await asyncio.sleep(0.01)
    stop.set()
    loop_task.cancel()
    with suppress(asyncio.CancelledError):
        await loop_task

    assert active_workers() == 1
    release.set()
    assert await drain_workers(5.0)


async def test_a_trigger_run_start_runs_its_database_work_on_tracked_workers(
    monkeypatch,
):
    from xagent.web.services import triggers

    release = threading.Event()
    started = threading.Event()

    def blocking_load(_run_id):
        started.set()
        release.wait(5)

    monkeypatch.setattr(triggers, "_load_prepared_trigger_start", blocking_load)
    starting = asyncio.create_task(triggers._start_prepared_trigger_run_id(1))
    while not started.is_set():
        await asyncio.sleep(0.01)
    starting.cancel()
    with suppress(asyncio.CancelledError):
        await starting

    assert active_workers() == 1
    release.set()
    assert await drain_workers(5.0)


async def test_the_retire_sets_the_temp_file_walk_stop_without_waiting_for_it(steps):
    stop = threading.Event()
    app_module.app.state.temp_file_cleanup_stop = stop
    try:
        await app_module.retire_writers()
    finally:
        app_module.app.state.temp_file_cleanup_stop = None

    assert stop.is_set()


# -- concurrency and lifecycle ----------------------------------------------------


async def test_two_concurrent_retires_stop_the_writers_once(steps, monkeypatch):
    async def slow_stop(*_a, **_k) -> None:
        steps.append("stop_orphan_upload_gc_task")
        await asyncio.sleep(0.1)

    monkeypatch.setattr(app_module, "stop_orphan_upload_gc_task", slow_stop)

    await asyncio.gather(app_module.retire_writers(), app_module.retire_writers())

    assert steps.count("stop_orphan_upload_gc_task") == 1
    assert steps.count("stop_task_lease_recovery_task") == 1
    assert app_module.writers_retired()


async def test_a_new_lifespan_clears_the_retired_state(steps, monkeypatch):
    await app_module.retire_writers()
    assert app_module.writers_retired()
    assert app_module.release_hold.socket_writes_fenced

    class _Stop(Exception):
        pass

    def stop_here() -> None:
        raise _Stop

    monkeypatch.setattr(app_module, "validate_auth_config", stop_here)
    with pytest.raises(_Stop):
        await app_module.startup_event()

    assert not app_module.writers_retired()
    assert not app_module.release_hold.socket_writes_fenced


@pytest.fixture
def teardown(monkeypatch: pytest.MonkeyPatch, steps: list[str]) -> list[str]:
    """Stub everything ``shutdown_event`` does after the writers stop."""
    from xagent.web.services import task_coordinator_runtime

    monkeypatch.setattr(app_module, "flush_langfuse", lambda: None)

    async def record_drain() -> None:
        steps.append("close_task_coordinators")

    async def idle(*_a, **_k) -> None:
        return None

    monkeypatch.setattr(
        task_coordinator_runtime, "close_task_coordinators", record_drain
    )
    monkeypatch.setattr(
        "xagent.web.services.task_execution.background_task_manager.shutdown", idle
    )
    monkeypatch.setattr(
        "xagent.web.services.task_lease_service.wait_for_heartbeat_manager_idle", idle
    )
    monkeypatch.setattr(app_module, "stop_runtime_performance_monitor", idle)
    monkeypatch.setattr(app_module, "stop_app_process_diagnostics", idle)
    monkeypatch.setattr("xagent.web.sandbox_manager.get_sandbox_manager", lambda: None)
    monkeypatch.setattr(
        "xagent.web.services.chrome_mcp_runtime.shutdown_chrome_execution_session_pool",
        idle,
    )
    monkeypatch.setattr(
        "xagent.web.services.trace_database.close_trace_database_runtime", idle
    )
    return steps


async def test_shutdown_stops_the_writers_before_it_drains_the_coordinators(
    teardown,
):
    await app_module.shutdown_event()

    assert teardown.index("stop_task_lease_recovery_task") < teardown.index(
        "close_task_coordinators"
    )


async def test_shutdown_skips_a_retire_the_host_already_completed(teardown):
    await app_module.retire_writers()
    stops_before = list(teardown)

    await app_module.shutdown_event()

    assert teardown == [*stops_before, "close_task_coordinators"]


async def test_shutdown_still_drains_the_coordinators_when_the_retire_times_out(
    teardown, monkeypatch
):
    monkeypatch.setattr(app_module, "RETIRE_SETTLE_SECONDS", 0.1)
    token = app_module.release_hold.acquire_socket_write()
    try:
        await app_module.shutdown_event()
    finally:
        token.release()

    assert not app_module.writers_retired()
    assert "close_task_coordinators" in teardown


# -- builder chat is a socket write ------------------------------------------------


class _BuilderSocket:
    """Delivers one builder frame, then optionally runs a hook before it closes."""

    def __init__(self, after_first=None) -> None:
        self.frames = 0
        self.after_first = after_first
        self.state = type("S", (), {})()

    async def receive_text(self) -> str:
        from starlette.websockets import WebSocketDisconnect

        self.frames += 1
        if self.frames == 1:
            return "{}"
        if self.after_first is not None:
            await self.after_first()
        await asyncio.sleep(0.05)
        raise WebSocketDisconnect()


@pytest.fixture
def builder_endpoint(monkeypatch):
    from xagent.web.api import websocket as ws

    async def authenticated(*_a, **_k):
        return type("U", (), {"id": 1, "is_admin": False})()

    async def accept(_websocket) -> None:
        return None

    monkeypatch.setattr(ws, "get_authenticated_user", authenticated)
    monkeypatch.setattr(ws, "accept_authenticated_websocket", accept)
    return ws


async def test_a_builder_turn_after_the_fence_is_refused_not_started(
    builder_endpoint, retired_hold, monkeypatch
):
    ws = builder_endpoint
    paused: list[bool] = []
    started: list[bool] = []

    async def send_paused(_websocket, **_k) -> None:
        paused.append(True)

    async def turn(*_a, **_k) -> None:
        started.append(True)

    monkeypatch.setattr(ws, "send_execution_paused", send_paused)
    monkeypatch.setattr(ws, "handle_builder_chat", turn)

    await ws.websocket_builder_chat_endpoint.__wrapped__(_BuilderSocket(), token=None)  # type: ignore[attr-defined]

    assert paused == [True]
    assert started == []
    assert not retired_hold._socket_writes


async def test_a_builder_turn_holds_a_token_and_releases_it_when_the_turn_ends(
    builder_endpoint, monkeypatch
):
    ws = builder_endpoint
    seen: list[int] = []

    async def turn(*_a, **_k) -> None:
        seen.append(len(app_module.release_hold._socket_writes))

    monkeypatch.setattr(ws, "handle_builder_chat", turn)

    await ws.websocket_builder_chat_endpoint.__wrapped__(_BuilderSocket(), token=None)  # type: ignore[attr-defined]

    assert seen == [1]
    assert not app_module.release_hold._socket_writes


async def test_the_fence_cancels_a_running_builder_turn_and_the_token_follows(
    builder_endpoint, monkeypatch
):
    ws = builder_endpoint
    running = asyncio.Event()
    cancelled: list[bool] = []

    async def turn(*_a, **_k) -> None:
        running.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    async def fence() -> None:
        await running.wait()
        app_module.release_hold.mark_writers_retired()

    monkeypatch.setattr(ws, "handle_builder_chat", turn)

    await ws.websocket_builder_chat_endpoint.__wrapped__(  # type: ignore[attr-defined]
        _BuilderSocket(after_first=fence), token=None
    )

    assert cancelled == [True]
    assert not app_module.release_hold._socket_writes


# -- entries that write are refused before they write -------------------------------


async def test_the_missing_task_branch_refuses_once_the_fence_rose_after_the_enqueue(
    monkeypatch,
):
    from xagent.web.api import websocket as ws

    paused: list[bool] = []

    async def enqueue_then_fence(**_k):
        # The enqueue was admitted; the fence goes up before the branch runs.
        app_module.release_hold.mark_writers_retired()
        return None

    async def send_paused(_websocket, **_k) -> None:
        paused.append(True)

    async def replacement(*_a, **_k) -> None:
        raise AssertionError("a replacement task was created after the fence")

    monkeypatch.setattr(ws, "_enqueue_websocket_task_command", enqueue_then_fence)
    monkeypatch.setattr(ws, "send_execution_paused", send_paused)
    monkeypatch.setattr(
        ws.command_execution_service, "handle_missing_task_message", replacement
    )

    await ws.handle_chat_message(object(), 1, {"user": object()})  # type: ignore[arg-type]

    assert paused == [True]
    assert not app_module.release_hold._socket_writes
    assert not app_module.release_hold.snapshot()["responsibilities"]


async def test_a_preview_writes_no_task_row_once_the_fence_is_up(
    retired_hold, monkeypatch
):
    from xagent.web.api import websocket as ws

    paused: list[bool] = []

    async def send_paused(_websocket, **_k) -> None:
        paused.append(True)

    async def create(*_a, **_k) -> int:
        raise AssertionError("the hidden preview Task row was written")

    class _Socket:
        state = type("S", (), {})()

    monkeypatch.setattr(ws, "send_execution_paused", send_paused)
    monkeypatch.setattr(ws, "_create_preview_task", create)

    await ws.handle_build_preview_execution(
        _Socket(),  # type: ignore[arg-type]
        {"message": "hello"},
        type("U", (), {"id": 1})(),  # type: ignore[arg-type]
    )

    assert paused == [True]
    assert not app_module.release_hold.snapshot()["responsibilities"]


# -- the status a host reads ---------------------------------------------------------


async def test_execution_status_reports_the_fence_and_the_admitted_socket_writes():
    from xagent.web.services.release_hold import execution_status

    before = execution_status()
    token = app_module.release_hold.acquire_socket_write()
    try:
        during = execution_status()
        app_module.release_hold.mark_writers_retired()
        fenced = execution_status()
    finally:
        token.release()

    assert (before["socket_writes_fenced"], before["socket_writes"]) == (False, 0)
    assert (during["socket_writes_fenced"], during["socket_writes"]) == (False, 1)
    assert (fenced["socket_writes_fenced"], fenced["socket_writes"]) == (True, 1)
    # Diagnostics only: they never make the process look busy.
    assert not before["busy"]
