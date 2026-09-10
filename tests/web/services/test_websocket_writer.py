from __future__ import annotations

import asyncio
import json

import pytest
from starlette.datastructures import State

from xagent.web.api.websocket import ConnectionManager
from xagent.web.services.websocket_writer import (
    WebSocketWriterRetiredError,
    activate_websocket_writer,
    retire_websocket_writer,
    send_websocket_text,
)


class ControlledWebSocket:
    def __init__(self, *, fail_first: bool = False) -> None:
        self.state = State()
        self.fail_first = fail_first
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.attempts: list[str] = []
        self.completed: list[str] = []
        self.active_sends = 0
        self.max_active_sends = 0

    async def send_text(self, data: str) -> None:
        self.attempts.append(data)
        self.active_sends += 1
        self.max_active_sends = max(self.max_active_sends, self.active_sends)
        self.entered.set()
        try:
            await self.release.wait()
            if self.fail_first:
                self.fail_first = False
                raise ConnectionError("first write failed")
            self.completed.append(data)
        finally:
            self.active_sends -= 1


@pytest.mark.asyncio
async def test_one_connection_serializes_concurrent_producers_in_arrival_order() -> (
    None
):
    websocket = ControlledWebSocket()
    first = asyncio.create_task(send_websocket_text(websocket, "history"))
    await websocket.entered.wait()
    second = asyncio.create_task(send_websocket_text(websocket, "live"))
    third = asyncio.create_task(send_websocket_text(websocket, "terminal"))
    await asyncio.sleep(0)

    assert websocket.attempts == ["history"]
    websocket.release.set()
    await asyncio.gather(first, second, third)

    assert websocket.completed == ["history", "live", "terminal"]
    assert websocket.max_active_sends == 1


@pytest.mark.asyncio
async def test_failed_write_does_not_wedge_the_connection_writer() -> None:
    websocket = ControlledWebSocket(fail_first=True)
    websocket.release.set()

    with pytest.raises(ConnectionError, match="first write failed"):
        await send_websocket_text(websocket, "broken")
    await send_websocket_text(websocket, "recovered")

    assert websocket.attempts == ["broken", "recovered"]
    assert websocket.completed == ["recovered"]


@pytest.mark.asyncio
async def test_cancelled_write_does_not_wedge_the_connection_writer() -> None:
    websocket = ControlledWebSocket()
    cancelled = asyncio.create_task(send_websocket_text(websocket, "cancelled"))
    await websocket.entered.wait()
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled

    websocket.release.set()
    await send_websocket_text(websocket, "replacement")

    assert websocket.completed == ["replacement"]


@pytest.mark.asyncio
async def test_cancelled_queued_write_does_not_release_its_successor_early() -> None:
    websocket = ControlledWebSocket()
    first = asyncio.create_task(send_websocket_text(websocket, "first"))
    await websocket.entered.wait()
    cancelled = asyncio.create_task(send_websocket_text(websocket, "cancelled"))
    successor = asyncio.create_task(send_websocket_text(websocket, "successor"))
    await asyncio.sleep(0)

    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    await asyncio.sleep(0)

    assert websocket.attempts == ["first"]
    assert websocket.max_active_sends == 1

    websocket.release.set()
    await asyncio.gather(first, successor)

    assert websocket.completed == ["first", "successor"]
    assert websocket.max_active_sends == 1


@pytest.mark.asyncio
async def test_different_connections_make_progress_independently() -> None:
    first_socket = ControlledWebSocket()
    second_socket = ControlledWebSocket()
    first = asyncio.create_task(send_websocket_text(first_socket, "first"))
    await first_socket.entered.wait()
    second = asyncio.create_task(send_websocket_text(second_socket, "second"))
    await asyncio.wait_for(second_socket.entered.wait(), timeout=1)

    second_socket.release.set()
    await asyncio.wait_for(second, timeout=1)
    assert second_socket.completed == ["second"]
    assert not first.done()

    first_socket.release.set()
    await first


@pytest.mark.asyncio
async def test_retirement_rejects_queued_writes_and_same_socket_reactivation() -> None:
    websocket = ControlledWebSocket()
    first = asyncio.create_task(send_websocket_text(websocket, "in-flight"))
    await websocket.entered.wait()
    queued = asyncio.create_task(send_websocket_text(websocket, "stale"))
    await asyncio.sleep(0)
    retire_websocket_writer(websocket)

    with pytest.raises(WebSocketWriterRetiredError):
        await asyncio.wait_for(queued, timeout=1)
    with pytest.raises(WebSocketWriterRetiredError):
        await asyncio.wait_for(first, timeout=1)

    replacement = ControlledWebSocket()
    replacement.release.set()
    await send_websocket_text(replacement, "replacement")
    assert websocket.completed == []
    assert websocket.active_sends == 0
    assert replacement.completed == ["replacement"]

    # Retirement rejects the generation, not the socket object: a live socket
    # can be registered again (a task delete retires every socket on the task
    # before closing them), and the new generation writes normally.
    activate_websocket_writer(websocket)
    websocket.release.set()
    assert await send_websocket_text(websocket, "next generation")
    assert websocket.completed == ["next generation"]


@pytest.mark.asyncio
async def test_retirement_does_not_swallow_the_callers_own_cancellation() -> None:
    websocket = ControlledWebSocket()
    sender = asyncio.create_task(send_websocket_text(websocket, "in-flight"))
    await websocket.entered.wait()

    # Retirement cancels the in-flight write, and the caller is cancelled in
    # the same tick. The caller's own cancellation wins: reporting the
    # retirement instead would finish a cancelled task with a plain exception.
    retire_websocket_writer(websocket)
    sender.cancel()

    with pytest.raises(asyncio.CancelledError):
        await sender
    assert sender.cancelled()
    assert websocket.completed == []


@pytest.mark.asyncio
async def test_manager_unregister_keeps_the_physical_socket_writer_usable() -> None:
    websocket = ControlledWebSocket()
    websocket.release.set()
    manager = ConnectionManager()
    manager.register_connection(websocket, task_id=42)

    manager.unregister_connection(websocket)
    manager.register_connection(websocket, task_id=43)
    await manager.send_personal_message({"type": "preview"}, websocket)

    assert manager.connections_for_task(42) == []
    assert manager.connections_for_task(43) == [websocket]
    assert [json.loads(frame)["type"] for frame in websocket.completed] == ["preview"]


@pytest.mark.asyncio
async def test_manager_re_registers_a_disconnected_socket_without_raising() -> None:
    websocket = ControlledWebSocket()
    websocket.release.set()
    manager = ConnectionManager()
    manager.register_connection(websocket, task_id=42)
    manager.disconnect(websocket)

    # ``detach_task_connections`` retires every socket on a deleted task and
    # closes them from a separate task, so a message arriving in that window
    # re-registers a retired-but-open socket. Registration must stay total.
    manager.register_connection(websocket, task_id=43)
    await manager.send_personal_message({"type": "resumed"}, websocket)

    assert manager.connections_for_task(43) == [websocket]
    assert [json.loads(frame)["type"] for frame in websocket.completed] == ["resumed"]


@pytest.mark.asyncio
async def test_manager_and_direct_producer_share_the_connection_writer() -> None:
    websocket = ControlledWebSocket()
    manager = ConnectionManager()
    manager.register_connection(websocket, task_id=42)

    history = asyncio.create_task(
        manager.send_personal_message({"type": "history"}, websocket)
    )
    await websocket.entered.wait()
    terminal = asyncio.create_task(
        send_websocket_text(websocket, json.dumps({"type": "terminal"}))
    )
    await asyncio.sleep(0)
    assert len(websocket.attempts) == 1

    websocket.release.set()
    await asyncio.gather(history, terminal)
    assert [json.loads(frame)["type"] for frame in websocket.completed] == [
        "history",
        "terminal",
    ]
    assert websocket.max_active_sends == 1


@pytest.mark.asyncio
async def test_manager_broadcast_does_not_serialize_different_connections() -> None:
    blocked_socket = ControlledWebSocket()
    independent_socket = ControlledWebSocket()
    manager = ConnectionManager()
    manager.register_connection(blocked_socket, task_id=42)
    manager.register_connection(independent_socket, task_id=42)

    broadcast = asyncio.create_task(
        manager.broadcast_to_task({"type": "status"}, task_id=42)
    )
    await blocked_socket.entered.wait()
    await asyncio.wait_for(independent_socket.entered.wait(), timeout=1)
    independent_socket.release.set()
    await asyncio.sleep(0)
    assert independent_socket.completed
    assert not broadcast.done()

    blocked_socket.release.set()
    await broadcast


@pytest.mark.asyncio
async def test_manager_broadcast_rechecks_membership_after_waiting_for_writer() -> None:
    websocket = ControlledWebSocket()
    manager = ConnectionManager()
    manager.register_connection(websocket, task_id=42)
    in_flight = asyncio.create_task(send_websocket_text(websocket, "in-flight"))
    await websocket.entered.wait()
    broadcast = asyncio.create_task(
        manager.broadcast_to_task({"type": "old-task"}, task_id=42)
    )
    await asyncio.sleep(0)

    manager.move_connection(websocket, new_task_id=43)
    websocket.release.set()
    await asyncio.gather(in_flight, broadcast)

    assert websocket.completed == ["in-flight"]
    assert manager.connections_for_task(43) == [websocket]
