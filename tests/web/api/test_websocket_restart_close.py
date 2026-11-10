"""A planned restart closes registered WebSockets with 1012 Service Restart."""

import asyncio
import json
import time

import pytest
from starlette.websockets import WebSocketState

from xagent.web.api import websocket as ws_module
from xagent.web.api.websocket import ConnectionManager, _command_origins


class FakeSocket:
    def __init__(
        self,
        *,
        fail_close: bool = False,
        hang_close: bool = False,
        log: list[str] | None = None,
        send_yields: bool = False,
    ) -> None:
        self.send_yields = send_yields
        self.closed: list[tuple[int, str | None]] = []
        self.fail_close = fail_close
        self.hang_close = hang_close
        self.log = log if log is not None else []

    async def send_text(self, text: str) -> None:
        if self.send_yields:
            await asyncio.sleep(0)
        self.log.append(f"sent:{text}")

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        if self.hang_close:
            await asyncio.Event().wait()
        if self.fail_close:
            raise RuntimeError("peer already gone")
        self.log.append(f"close:{code}")
        self.closed.append((code, reason))


class FakeSse(FakeSocket):
    is_broadcast_only = True


@pytest.fixture
def manager():
    mgr = ConnectionManager()
    yield mgr
    _command_origins._origins.clear()


@pytest.mark.asyncio
async def test_observation_only_sockets_close_with_1012_and_unregister(manager):
    a, b = FakeSocket(), FakeSocket()
    manager.register_connection(a, 1)
    manager.register_connection(b, 2, public_audience=True)

    assert await manager.close_for_restart() == 2

    assert a.closed == [(1012, "Service Restart")]
    assert b.closed == [(1012, "Service Restart")]
    assert manager.connection_count() == 0


@pytest.mark.asyncio
async def test_sse_sinks_are_left_open(manager):
    sse, ws = FakeSse(), FakeSocket()
    manager.register_connection(sse, 1)
    manager.register_connection(ws, 1)

    await manager.close_for_restart()

    assert sse.closed == []
    assert ws.closed == [(1012, "Service Restart")]


@pytest.mark.asyncio
async def test_socket_with_pending_command_closes_after_reply(manager):
    waiting, idle = FakeSocket(), FakeSocket()
    manager.register_connection(waiting, 1)
    manager.register_connection(idle, 1)
    _command_origins.register("cmd-1", waiting, 1)

    closing = asyncio.create_task(manager.close_for_restart(reply_deadline_seconds=5))
    await asyncio.sleep(0.2)
    assert idle.closed == [(1012, "Service Restart")]
    assert waiting.closed == []

    _command_origins.discard_command("cmd-1", 1)
    await asyncio.wait_for(closing, timeout=2)
    assert waiting.closed == [(1012, "Service Restart")]


@pytest.mark.asyncio
async def test_pending_command_is_closed_at_the_deadline(manager):
    waiting = FakeSocket()
    manager.register_connection(waiting, 1)
    _command_origins.register("cmd-1", waiting, 1)

    await asyncio.wait_for(manager.close_for_restart(reply_deadline_seconds=0.1), 2)

    assert waiting.closed == [(1012, "Service Restart")]
    assert manager.connection_count() == 0


@pytest.mark.asyncio
async def test_close_failure_still_unregisters_and_is_idempotent(manager):
    gone = FakeSocket(fail_close=True)
    manager.register_connection(gone, 1)

    assert await manager.close_for_restart() == 1
    assert manager.connection_count() == 0
    assert await manager.close_for_restart() == 0


def test_close_code_constant():
    assert ws_module.WS_CLOSE_SERVICE_RESTART == 1012


class _FakeBridge:
    def __init__(self) -> None:
        self.pending: set[int] = set()

    def has_pending_for_recipient(self, recipient) -> bool:
        return id(recipient) in self.pending

    def discard_recipient(self, recipient) -> None:
        self.pending.discard(id(recipient))


@pytest.fixture
def shared_mode(monkeypatch):
    from xagent.web.services import task_event_bridge

    bridge = _FakeBridge()
    monkeypatch.setattr(ws_module, "get_shared_task_execution_enabled", lambda: True)
    monkeypatch.setattr(task_event_bridge, "get_task_event_bridge", lambda: bridge)
    return bridge


@pytest.mark.asyncio
async def test_shared_mode_pending_command_waits_for_its_reply(manager, shared_mode):
    waiting = FakeSocket()
    manager.register_connection(waiting, 1)
    shared_mode.pending.add(id(waiting))

    closing = asyncio.create_task(manager.close_for_restart(reply_deadline_seconds=5))
    await asyncio.sleep(0.2)
    assert waiting.closed == []

    shared_mode.pending.clear()
    await asyncio.wait_for(closing, timeout=2)
    assert waiting.closed == [(1012, "Service Restart")]


@pytest.mark.asyncio
async def test_socket_registered_while_draining_is_closed_at_once(manager):
    await manager.close_for_restart()
    late, sse = FakeSocket(), FakeSse()

    manager.register_connection(late, 1)
    manager.register_connection(sse, 1)
    await asyncio.sleep(0.05)

    assert late.closed == [(1012, "Service Restart")]
    assert sse.closed == []
    assert manager.connection_count() == 1  # only the SSE sink remains


@pytest.mark.asyncio
async def test_end_restart_drain_admits_new_sockets_again(manager):
    await manager.close_for_restart()
    manager.end_restart_drain()
    late = FakeSocket()

    manager.register_connection(late, 1)
    await asyncio.sleep(0.05)

    assert late.closed == []
    assert manager.connection_count() == 1


@pytest.mark.asyncio
async def test_stalled_close_is_bounded_and_unregisters(manager):
    stuck = FakeSocket(hang_close=True)
    manager.register_connection(stuck, 1)

    handled = await asyncio.wait_for(
        manager.close_for_restart(close_timeout_seconds=0.1), timeout=2
    )

    assert handled == 1
    assert manager.connection_count() == 0


@pytest.mark.asyncio
async def test_ingress_holds_the_close_until_the_handler_finishes(manager):
    log: list[str] = []
    ws = FakeSocket(log=log)
    manager.register_connection(ws, 1)
    release = asyncio.Event()

    async def handler() -> None:
        async with manager.ingress(ws):
            await release.wait()
            await ws.send_text("accepted")

    work = asyncio.create_task(handler())
    await asyncio.sleep(0.05)
    closing = asyncio.create_task(manager.close_for_restart(reply_deadline_seconds=5))
    await asyncio.sleep(0.2)
    assert log == []

    release.set()
    await asyncio.wait_for(asyncio.gather(work, closing), timeout=2)
    assert log == ["sent:accepted", "close:1012"]


@pytest.mark.asyncio
async def test_ingress_is_released_when_the_handler_raises(manager):
    ws = FakeSocket()
    manager.register_connection(ws, 1)

    with pytest.raises(RuntimeError):
        async with manager.ingress(ws):
            raise RuntimeError("handler failed")

    assert not manager._has_pending_command(ws)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name", ["handle_chat_message", "handle_pause_task", "handle_resume_task"]
)
async def test_command_handlers_hold_ingress_while_enqueueing(monkeypatch, name):
    log: list[str] = []
    ws = FakeSocket(log=log)
    gm = ws_module.manager
    gm.register_connection(ws, 1)
    release = asyncio.Event()

    async def blocked_enqueue(**_kwargs):
        await release.wait()
        raise ValueError("rejected")

    async def reply(message, websocket):
        await websocket.send_text("reply")

    monkeypatch.setattr(ws_module, "_enqueue_websocket_task_command", blocked_enqueue)
    monkeypatch.setattr(gm, "send_personal_message", reply)
    monkeypatch.setattr(ws_module, "send_message_delivery", _reply_delivery)
    handler = getattr(ws_module, name)
    work = asyncio.create_task(handler(ws, 1, {"command_id": "c1"}))
    await asyncio.sleep(0.05)
    try:
        closing = asyncio.create_task(gm.close_for_restart(reply_deadline_seconds=5))
        await asyncio.sleep(0.2)
        assert log == []
        release.set()
        await asyncio.wait_for(asyncio.gather(work, closing), timeout=2)
        assert log[-1] == "close:1012"
        assert log.index("sent:reply") < log.index("close:1012")
    finally:
        gm.end_restart_drain()
        gm.disconnect(ws)


async def _reply_delivery(websocket, **_kwargs):
    await websocket.send_text("reply")


@pytest.fixture
def global_manager(monkeypatch):
    """The module-level manager, which the real reply path is bound to."""

    async def passthrough(message):
        return message

    monkeypatch.setattr(ws_module, "_with_current_task_control_state", passthrough)
    gm = ws_module.manager
    yield gm
    gm.end_restart_drain()
    for ws in list(gm._connection_task_ids):
        gm.disconnect(ws)
    _command_origins._origins.clear()


@pytest.mark.asyncio
async def test_task_id_updated_reply_precedes_1012_through_the_real_reply_path(
    global_manager,
):
    # A legacy first chat: the reply re-registers the socket on the new task
    # while the manager drains, then sends ``task_id_updated``.
    log: list[str] = []
    ws = FakeSocket(log=log, send_yields=True)
    global_manager.register_connection(ws, 1)
    _command_origins.register("cmd-1", ws, 1)

    closing = asyncio.create_task(
        global_manager.close_for_restart(reply_deadline_seconds=5)
    )
    await asyncio.sleep(0.1)
    reply = _command_origins.reply_for("cmd-1", 1)
    await reply({"type": "task_id_updated", "new_task_id": 2})
    await asyncio.sleep(0.1)
    assert [e.split(":", 1)[0] for e in log] == ["sent"]
    assert json.loads(log[0].removeprefix("sent:"))["new_task_id"] == 2

    _command_origins.discard_command("cmd-1", 1)
    await asyncio.wait_for(closing, timeout=2)
    assert log[1:] == ["close:1012"]


@pytest.mark.asyncio
async def test_move_connection_while_holding_ingress_is_not_closed_at_once(manager):
    ws = FakeSocket()
    manager.register_connection(ws, 1)
    release = asyncio.Event()

    async def handler() -> None:
        async with manager.ingress(ws):
            await release.wait()

    work = asyncio.create_task(handler())
    await asyncio.sleep(0.05)
    closing = asyncio.create_task(manager.close_for_restart(reply_deadline_seconds=5))
    await asyncio.sleep(0.1)

    manager.move_connection(ws, 2)
    await asyncio.sleep(0.2)
    assert ws.closed == []
    assert manager.connection_count() == 1

    release.set()
    await asyncio.wait_for(asyncio.gather(work, closing), timeout=2)
    assert ws.closed == [(1012, "Service Restart")]


@pytest.mark.asyncio
async def test_ingress_hands_over_to_the_origin_without_a_close_gap(manager):
    # The accepted path: the handler registers the origin, then leaves ingress.
    ws = FakeSocket()
    manager.register_connection(ws, 1)
    release = asyncio.Event()

    async def handler() -> None:
        async with manager.ingress(ws):
            _command_origins.register("cmd-1", ws, 1)
            await release.wait()

    work = asyncio.create_task(handler())
    await asyncio.sleep(0.05)
    closing = asyncio.create_task(manager.close_for_restart(reply_deadline_seconds=5))
    release.set()
    await asyncio.wait_for(work, timeout=1)
    await asyncio.sleep(0.2)
    assert ws.closed == []

    _command_origins.discard_command("cmd-1", 1)
    await asyncio.wait_for(closing, timeout=2)
    assert ws.closed == [(1012, "Service Restart")]


@pytest.fixture
def real_bridge(monkeypatch):
    monkeypatch.setenv("XAGENT_REDIS_URL", "redis://127.0.0.1:1/0")
    from xagent.web.services import task_event_bridge
    from xagent.web.services.task_event_bridge import TaskEventBridge

    bridge = TaskEventBridge()
    bridge.ready.set()
    monkeypatch.setattr(ws_module, "get_shared_task_execution_enabled", lambda: True)
    monkeypatch.setattr(task_event_bridge, "_bridge", bridge)
    return bridge


@pytest.mark.asyncio
async def test_real_bridge_origin_holds_the_close_until_discarded(manager, real_bridge):
    async def reply(_message) -> None:
        return None

    ws = FakeSocket()
    manager.register_connection(ws, 1)
    token = real_bridge.register_origin(1, "cmd-1", reply, recipient=ws)

    closing = asyncio.create_task(manager.close_for_restart(reply_deadline_seconds=5))
    await asyncio.sleep(0.2)
    assert ws.closed == []

    real_bridge.discard_origin(token)
    await asyncio.wait_for(closing, timeout=2)
    assert ws.closed == [(1012, "Service Restart")]


@pytest.mark.asyncio
async def test_unavailable_bridge_counts_as_no_pending_command(manager, monkeypatch):
    from xagent.web.services import task_event_bridge

    monkeypatch.setattr(ws_module, "get_shared_task_execution_enabled", lambda: True)
    monkeypatch.setattr(task_event_bridge, "_bridge", None)  # raises ConnectionError
    ws = FakeSocket()
    manager.register_connection(ws, 1)

    assert not manager._has_pending_command(ws)
    await asyncio.wait_for(manager.close_for_restart(reply_deadline_seconds=5), 1)
    assert ws.closed == [(1012, "Service Restart")]


@pytest.mark.asyncio
async def test_end_restart_drain_during_the_wait_sends_no_further_1012(manager):
    waiting = FakeSocket()
    manager.register_connection(waiting, 1)
    _command_origins.register("cmd-1", waiting, 1)

    closing = asyncio.create_task(manager.close_for_restart(reply_deadline_seconds=5))
    await asyncio.sleep(0.2)
    manager.end_restart_drain()
    _command_origins.discard_command("cmd-1", 1)

    assert await asyncio.wait_for(closing, timeout=2) == 0
    assert waiting.closed == []
    assert manager.connection_count() == 1


@pytest.mark.asyncio
async def test_end_restart_drain_lets_a_started_close_finish(manager, monkeypatch):
    monkeypatch.setattr(ws_module, "RESTART_CLOSE_TIMEOUT_SECONDS", 0.1)
    await manager.close_for_restart()
    late = FakeSocket(hang_close=True)
    manager.register_connection(late, 1)
    await asyncio.sleep(0.02)  # the scheduled close is now inside close()
    manager.end_restart_drain()
    await asyncio.sleep(0.3)  # past its timeout: the started close unregisters

    assert manager._restart_closes == set()
    assert manager.connection_count() == 0

    fresh = FakeSocket()
    manager.register_connection(fresh, 1)
    await asyncio.sleep(0.05)
    assert fresh.closed == []
    assert manager.connection_count() == 1


@pytest.mark.asyncio
async def test_end_restart_drain_stops_a_scheduled_close_that_has_not_started(
    manager,
):
    await manager.close_for_restart()
    late = FakeSocket()
    manager.register_connection(late, 1)  # schedules the close, not yet run
    manager.end_restart_drain()
    await asyncio.sleep(0.05)

    assert late.closed == []
    assert manager._restart_closes == set()
    assert manager.connection_count() == 1


@pytest.mark.asyncio
async def test_close_is_never_invoked_after_the_drain_generation_moved(manager):
    seen: list[int] = []

    class GenerationSocket(FakeSocket):
        async def close(self, code: int = 1000, reason: str | None = None) -> None:
            seen.append(manager._restart_drain_generation)
            await super().close(code, reason)

    ws = GenerationSocket()
    manager.register_connection(ws, 1)
    generation = manager._restart_drain_generation
    closing = asyncio.create_task(manager._close_for_restart(ws, 1.0, generation))
    # Queued behind the close's first step: an abort that lands as soon as
    # the close yields for the first time.
    asyncio.get_running_loop().call_soon(manager.end_restart_drain)
    await closing
    await asyncio.sleep(0)

    # Whether or not the close got in first, it was never invoked after the
    # abort, and the socket is unregistered exactly when its close went out.
    assert all(gen == generation for gen in seen)
    assert (ws.closed == []) == (seen == [])
    assert (manager.connection_count() == 1) == (seen == [])


@pytest.mark.asyncio
async def test_zero_close_timeout_still_closes_and_unregisters(manager):
    ws = FakeSocket()
    manager.register_connection(ws, 1)

    handled = await manager.close_for_restart(close_timeout_seconds=0)

    assert handled == 1
    assert ws.closed == [(1012, "Service Restart")]
    assert manager.connection_count() == 0


@pytest.mark.asyncio
async def test_a_close_from_an_ended_drain_is_a_no_op(manager):
    ws = FakeSocket()
    manager.register_connection(ws, 1)
    stale = manager._restart_drain_generation
    manager.end_restart_drain()

    await manager._close_for_restart(ws, 1.0, stale)

    assert ws.closed == []
    assert manager.connection_count() == 1


@pytest.mark.asyncio
async def test_cancelled_close_for_restart_ends_the_drain(manager):
    waiting = FakeSocket()
    manager.register_connection(waiting, 1)
    _command_origins.register("cmd-1", waiting, 1)

    closing = asyncio.create_task(manager.close_for_restart(reply_deadline_seconds=5))
    await asyncio.sleep(0.1)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing

    late = FakeSocket()
    manager.register_connection(late, 2)
    await asyncio.sleep(0.05)
    assert late.closed == []
    assert waiting.closed == []


@pytest.mark.asyncio
async def test_failed_close_for_restart_ends_the_drain(manager, monkeypatch):
    ws = FakeSocket()
    manager.register_connection(ws, 1)

    async def boom(*_args):
        raise ValueError("boom")

    monkeypatch.setattr(manager, "_close_for_restart", boom)
    with pytest.raises(ValueError):
        await manager.close_for_restart()
    monkeypatch.undo()

    late = FakeSocket()
    manager.register_connection(late, 2)
    await asyncio.sleep(0.05)
    assert late.closed == []


@pytest.mark.asyncio
async def test_deadline_counts_from_the_call_not_after_the_first_close(manager):
    stuck = FakeSocket(hang_close=True)
    waiting = FakeSocket()
    manager.register_connection(stuck, 1)
    manager.register_connection(waiting, 1)
    _command_origins.register("cmd-1", waiting, 1)

    started = time.monotonic()
    await asyncio.wait_for(
        manager.close_for_restart(
            reply_deadline_seconds=0.4, close_timeout_seconds=0.4
        ),
        timeout=3,
    )

    # close_timeout + one more close, not close_timeout + deadline + close.
    assert time.monotonic() - started < 0.4 + 0.4 - 0.1
    assert waiting.closed == [(1012, "Service Restart")]


@pytest.mark.asyncio
async def test_execute_task_holds_ingress_while_it_runs(global_manager, monkeypatch):
    ws = FakeSocket()
    release = asyncio.Event()

    async def acknowledge(_message, _websocket):
        return None

    async def blocked_load(*_args, **_kwargs):
        await release.wait()
        return None  # task not found: the handler answers with an error

    monkeypatch.setattr(global_manager, "send_personal_message", acknowledge)
    monkeypatch.setattr(ws_module, "run_db_io_cancellation_safe", blocked_load)
    user = type("User", (), {"id": 1, "is_admin": False})()
    work = asyncio.create_task(ws_module.handle_execute_task(ws, 1, {"user": user}))
    await asyncio.sleep(0.05)
    assert global_manager._has_pending_command(ws)

    release.set()
    await work
    assert not global_manager._has_pending_command(ws)


@pytest.mark.asyncio
async def test_scheduled_close_uses_the_shared_timeout_constant(manager, monkeypatch):
    monkeypatch.setattr(ws_module, "RESTART_CLOSE_TIMEOUT_SECONDS", 0.05)
    await manager.close_for_restart()
    stuck = FakeSocket(hang_close=True)

    manager.register_connection(stuck, 1)
    await asyncio.sleep(0.3)

    assert manager.connection_count() == 0


def test_close_timeout_default_is_the_shared_constant():
    import inspect

    default = inspect.signature(ConnectionManager.close_for_restart).parameters[
        "close_timeout_seconds"
    ]
    assert default.default == ws_module.RESTART_CLOSE_TIMEOUT_SECONDS


class _StateSocket:
    def __init__(self, state) -> None:
        self.application_state = state


def test_receive_after_our_own_close_is_not_an_error(manager):
    gone = _StateSocket(WebSocketState.DISCONNECTED)
    live = _StateSocket(WebSocketState.CONNECTED)

    assert manager.was_closed_by_server(gone, RuntimeError("receive after close"))
    assert not manager.was_closed_by_server(live, RuntimeError("real fault"))
    assert not manager.was_closed_by_server(gone, ConnectionError("reset"))
