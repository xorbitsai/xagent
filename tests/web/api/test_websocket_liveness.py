"""Server heartbeat negotiation and lifecycle on the ``/ws/chat`` endpoint."""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from anyio import BrokenResourceError, ClosedResourceError
from fastapi import WebSocketDisconnect

from xagent.web.api import websocket as websocket_api

TASK_ID = 41
CAPABILITIES_FRAME = {"type": "connection_capabilities", "capabilities": ["heartbeat"]}
HEARTBEAT_FRAME = {"type": "heartbeat"}
STATUS_MARKER = {"type": "status_request_handled"}
# Mirrors ``DEAD_AFTER_MS`` in the frontend ``use-websocket`` hook.
CLIENT_DEAD_AFTER_SECONDS = 30.0
HEARTBEAT_LOGGER = "xagent.web.api.websocket"


class _FakeSocket:
    """Records sent frames; ``confirmation_delay`` slows only the capabilities
    frame so a heartbeat task started too early would overtake it.
    ``heartbeat_error`` is raised by every heartbeat send, and
    ``heartbeat_senders`` names the task that attempted each one."""

    def __init__(
        self,
        liveness: list[str],
        *,
        confirmation_delay: float = 0.0,
        heartbeat_error: BaseException | None = None,
    ) -> None:
        self.query_params = SimpleNamespace(
            getlist=lambda key: list(liveness) if key == "liveness" else []
        )
        self.accept = AsyncMock()
        self.close = AsyncMock()
        self.frames: list[dict[str, Any]] = []
        self.heartbeat_senders: list[str] = []
        self._inbox: asyncio.Queue[Any] = asyncio.Queue()
        self._confirmation_delay = confirmation_delay
        self._heartbeat_error = heartbeat_error

    async def send_text(self, text: str) -> None:
        frame = json.loads(text)
        if frame == CAPABILITIES_FRAME and self._confirmation_delay:
            await asyncio.sleep(self._confirmation_delay)
        if frame == HEARTBEAT_FRAME:
            current = asyncio.current_task()
            assert current is not None
            self.heartbeat_senders.append(current.get_coro().__qualname__)
            if self._heartbeat_error is not None:
                raise self._heartbeat_error
        self.frames.append(frame)

    async def receive_text(self) -> str:
        item = await self._inbox.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def client_sends(self, item: Any) -> None:
        self._inbox.put_nowait(item)


async def _status_marker(websocket: _FakeSocket, *_: Any) -> None:
    websocket.frames.append(STATUS_MARKER)


def _endpoint_env(manager: websocket_api.ConnectionManager, **overrides: Any) -> Any:
    """Patch the endpoint's collaborators so no database is touched."""
    return patch.multiple(
        websocket_api,
        manager=manager,
        get_authenticated_user=AsyncMock(
            return_value=SimpleNamespace(id=7, is_admin=False)
        ),
        _private_websocket_task_access_sync=lambda **_: "authorized",
        handle_status_request=_status_marker,
        **overrides,
    )


async def _finished(endpoint: asyncio.Task[None]) -> list[Any]:
    # Bounded, so an endpoint that never stops its heartbeat task fails the
    # test instead of hanging it.
    return await asyncio.wait_for(
        asyncio.gather(endpoint, return_exceptions=True), timeout=5
    )


def _run_endpoint(websocket: _FakeSocket) -> asyncio.Task[None]:
    return asyncio.ensure_future(
        websocket_api.websocket_chat_endpoint(websocket, TASK_ID, None)  # type: ignore[arg-type]
    )


def _running_heartbeats() -> list[asyncio.Task[Any]]:
    return [
        task
        for task in asyncio.all_tasks()
        if task.get_coro().__qualname__ == "_run_chat_heartbeat" and not task.done()
    ]


def _heartbeat_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == HEARTBEAT_LOGGER and "heartbeat" in record.getMessage()
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("liveness", "negotiated"),
    [
        ([], False),
        (["heartbeat"], True),
        (["ping"], False),
        (["heartbeat", "heartbeat"], False),
    ],
)
async def test_ws_chat_liveness_negotiation(
    liveness: list[str], negotiated: bool
) -> None:
    manager = websocket_api.ConnectionManager()
    websocket = _FakeSocket(liveness, confirmation_delay=0.1)

    with _endpoint_env(manager, HEARTBEAT_INTERVAL_SECONDS=0.005):
        endpoint = _run_endpoint(websocket)
        await asyncio.sleep(0.3)  # many heartbeat intervals
        websocket.client_sends(WebSocketDisconnect())
        await _finished(endpoint)

    if negotiated:
        # The acknowledgement precedes the status snapshot and every heartbeat.
        assert websocket.frames[0] == CAPABILITIES_FRAME
        assert STATUS_MARKER in websocket.frames
        assert HEARTBEAT_FRAME in websocket.frames
    else:
        assert websocket.frames == [STATUS_MARKER]


@pytest.mark.asyncio
async def test_ws_chat_heartbeat_survives_blocked_loop() -> None:
    manager = websocket_api.ConnectionManager()
    release_handler = asyncio.Event()

    async def blocked_execute_task(*_: Any) -> None:
        await release_handler.wait()

    # Phase 1: the read loop is stuck inside a long handler, and the heartbeat
    # keeps flowing; after a normal disconnect the task is gone.
    first = _FakeSocket(["heartbeat"])
    with _endpoint_env(
        manager,
        HEARTBEAT_INTERVAL_SECONDS=0.05,
        handle_execute_task=blocked_execute_task,
    ):
        endpoint = _run_endpoint(first)
        first.client_sends(json.dumps({"type": "execute_task"}))
        await asyncio.sleep(0.4)
        assert not endpoint.done()
        assert first.frames.count(HEARTBEAT_FRAME) >= 2

        release_handler.set()
        first.client_sends(WebSocketDisconnect())
        await _finished(endpoint)

        assert manager.connection_count() == 0
        beats_at_exit = first.frames.count(HEARTBEAT_FRAME)
        await asyncio.sleep(0.2)
        assert first.frames.count(HEARTBEAT_FRAME) == beats_at_exit

    # Phase 2: a stand-in heartbeat task raises past the real task's own
    # handling; teardown still removes the connection from the manager.
    async def failing_heartbeat(*_: Any) -> None:
        raise ValueError("heartbeat fault")

    second = _FakeSocket(["heartbeat"])
    with _endpoint_env(
        manager,
        _run_chat_heartbeat=failing_heartbeat,
    ):
        endpoint = _run_endpoint(second)
        await asyncio.sleep(0.1)
        assert manager.connection_count() == 1
        second.client_sends(WebSocketDisconnect())
        assert await _finished(endpoint) == [None]
        assert manager.connection_count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "send_error",
    [
        RuntimeError("socket closed"),
        ConnectionError("socket gone"),
        WebSocketDisconnect(code=1006),
        BrokenResourceError(),
        ClosedResourceError(),
    ],
    ids=lambda error: type(error).__name__,
)
async def test_ws_chat_heartbeat_send_failure_ends_quietly(
    send_error: BaseException, caplog: pytest.LogCaptureFixture
) -> None:
    manager = websocket_api.ConnectionManager()
    websocket = _FakeSocket(["heartbeat"], heartbeat_error=send_error)

    with (
        caplog.at_level(logging.WARNING, logger=HEARTBEAT_LOGGER),
        _endpoint_env(manager, HEARTBEAT_INTERVAL_SECONDS=0.01),
    ):
        endpoint = _run_endpoint(websocket)
        await asyncio.sleep(0.2)  # many heartbeat intervals

        # One failed send ended the task; the read loop still answers.
        assert websocket.heartbeat_senders == ["_run_chat_heartbeat"]
        assert _running_heartbeats() == []
        websocket.client_sends(json.dumps({"type": "status_request"}))
        await asyncio.sleep(0.05)
        assert websocket.frames.count(STATUS_MARKER) == 2

        websocket.client_sends(WebSocketDisconnect())
        assert await _finished(endpoint) == [None]

    assert _heartbeat_warnings(caplog) == []
    assert manager.connection_count() == 0


@pytest.mark.asyncio
async def test_ws_chat_heartbeat_logs_other_exceptions_while_connected(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = websocket_api.ConnectionManager()
    websocket = _FakeSocket(["heartbeat"], heartbeat_error=ValueError("bad frame"))

    with (
        caplog.at_level(logging.WARNING, logger=HEARTBEAT_LOGGER),
        _endpoint_env(manager, HEARTBEAT_INTERVAL_SECONDS=0.01),
    ):
        endpoint = _run_endpoint(websocket)
        await asyncio.sleep(0.2)

        # Logged by the task itself, with the traceback, before any teardown.
        assert not endpoint.done()
        assert _running_heartbeats() == []
        (record,) = _heartbeat_warnings(caplog)
        assert record.exc_info is not None
        assert isinstance(record.exc_info[1], ValueError)
        assert str(TASK_ID) in record.getMessage()

        websocket.client_sends(WebSocketDisconnect())
        assert await _finished(endpoint) == [None]

    # Teardown does not log the same fault a second time.
    assert len(_heartbeat_warnings(caplog)) == 1
    assert manager.connection_count() == 0


class _OrderRecordingManager(websocket_api.ConnectionManager):
    def __init__(self) -> None:
        super().__init__()
        self.heartbeats_running_at_disconnect: list[int] = []

    def disconnect(self, websocket: Any) -> None:
        self.heartbeats_running_at_disconnect.append(len(_running_heartbeats()))
        super().disconnect(websocket)


@pytest.mark.asyncio
async def test_ws_chat_heartbeat_stops_before_the_connection_is_unregistered(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = _OrderRecordingManager()
    websocket = _FakeSocket(["heartbeat"])

    with (
        caplog.at_level(logging.WARNING, logger=HEARTBEAT_LOGGER),
        _endpoint_env(manager, HEARTBEAT_INTERVAL_SECONDS=0.01),
    ):
        endpoint = _run_endpoint(websocket)
        await asyncio.sleep(0.05)
        assert len(_running_heartbeats()) == 1
        websocket.client_sends(WebSocketDisconnect())
        assert await _finished(endpoint) == [None]

    assert manager.heartbeats_running_at_disconnect == [0]
    # Cancelling the heartbeat on a normal disconnect is not a fault.
    assert _heartbeat_warnings(caplog) == []


@pytest.mark.asyncio
async def test_ws_chat_heartbeat_in_shared_mode_goes_through_the_writer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = websocket_api.ConnectionManager()
    websocket = _FakeSocket(["heartbeat"])
    bridge = SimpleNamespace(discard_recipient=lambda recipient: None)

    with (
        caplog.at_level(logging.WARNING, logger=HEARTBEAT_LOGGER),
        patch(
            "xagent.web.services.task_event_bridge.get_task_event_bridge",
            return_value=bridge,
        ),
        _endpoint_env(
            manager,
            HEARTBEAT_INTERVAL_SECONDS=0.01,
            get_shared_task_execution_enabled=lambda: True,
        ),
    ):
        endpoint = _run_endpoint(websocket)
        await asyncio.sleep(0.1)

        # Heartbeats are queued and written by the socket writer's own task.
        assert websocket.frames[0] == CAPABILITIES_FRAME
        assert websocket.heartbeat_senders
        assert set(websocket.heartbeat_senders) == {"TaskSocketWriter._run"}

        # A failed write closes the writer, which fails the heartbeat's pending
        # delivery with ``ConnectionError``; the heartbeat ends quietly.
        websocket._heartbeat_error = OSError("write failed")
        await asyncio.sleep(0.1)
        assert _running_heartbeats() == []
        assert manager.connection_count() == 0
        websocket.close.assert_awaited_with(code=1013)

        websocket.client_sends(WebSocketDisconnect())
        assert await _finished(endpoint) == [None]

    assert _heartbeat_warnings(caplog) == []


def test_heartbeat_interval_stays_within_half_the_dead_threshold() -> None:
    assert websocket_api.HEARTBEAT_INTERVAL_SECONDS * 2 <= CLIENT_DEAD_AFTER_SECONDS
