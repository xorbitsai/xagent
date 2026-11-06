"""A planned restart closes registered WebSockets with 1012 Service Restart."""

import asyncio

import pytest

from xagent.web.api import websocket as ws_module
from xagent.web.api.websocket import ConnectionManager, _command_origins


class FakeSocket:
    def __init__(self, *, fail_close: bool = False) -> None:
        self.closed: list[tuple[int, str | None]] = []
        self.fail_close = fail_close

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        if self.fail_close:
            raise RuntimeError("peer already gone")
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
