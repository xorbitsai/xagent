from __future__ import annotations

import asyncio
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from xagent.core.tools.core.RAG_tools.progress.realtime import ProgressBroadcaster
from xagent.web.api import progress_ws as progress_ws_api
from xagent.web.api import websocket as websocket_api
from xagent.web.services.websocket_writer import (
    WebSocketWriterRetiredError,
    send_websocket_text,
)

WEBSOCKET_API = Path(__file__).parents[3] / "src/xagent/web/api/websocket.py"
PROGRESS_API = Path(__file__).parents[3] / "src/xagent/web/api/progress_ws.py"


def _raw_send_text_calls(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "send_text"
        and not (
            path == PROGRESS_API
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "adapter"
        )
    ]


def test_builder_and_preview_have_no_direct_websocket_text_writes() -> None:
    assert _raw_send_text_calls(WEBSOCKET_API) == []


def test_progress_socket_writes_only_through_its_ordered_adapter() -> None:
    assert _raw_send_text_calls(PROGRESS_API) == []


class _BlockingWebSocket:
    def __init__(self) -> None:
        self.state = SimpleNamespace()
        self.receive_started = asyncio.Event()
        self.closed = False

    async def accept(self) -> None:
        return None

    async def close(self, **_kwargs: object) -> None:
        self.closed = True

    async def receive_text(self) -> str:
        self.receive_started.set()
        await asyncio.Future()
        raise AssertionError("unreachable")

    async def send_text(self, _data: str) -> None:
        return None


class _ControlledSendWebSocket(_BlockingWebSocket):
    def __init__(self) -> None:
        super().__init__()
        self.send_started = asyncio.Event()
        self.release_send = asyncio.Event()
        self.attempts: list[str] = []
        self.active_sends = 0
        self.max_active_sends = 0

    async def send_text(self, data: str) -> None:
        self.attempts.append(data)
        self.active_sends += 1
        self.max_active_sends = max(self.max_active_sends, self.active_sends)
        self.send_started.set()
        try:
            await self.release_send.wait()
        finally:
            self.active_sends -= 1


class _ProgressConnection:
    def __init__(self, *, blocked: bool) -> None:
        self.send_started = asyncio.Event()
        self.release_send = asyncio.Event()
        self.send_completed = asyncio.Event()
        if not blocked:
            self.release_send.set()

    def is_connected(self) -> bool:
        return True

    async def send_text(self, _data: str) -> None:
        self.send_started.set()
        await self.release_send.wait()
        self.send_completed.set()


@pytest.mark.asyncio
async def test_slow_progress_socket_does_not_block_another_socket() -> None:
    broadcaster = ProgressBroadcaster()
    blocked = _ProgressConnection(blocked=True)
    independent = _ProgressConnection(blocked=False)
    await broadcaster.connect("task-1", blocked)
    await broadcaster.connect("task-1", independent)

    broadcast = asyncio.create_task(
        broadcaster.broadcast_event("task-1", "progress_update")
    )
    await asyncio.wait_for(blocked.send_started.wait(), timeout=1)
    await asyncio.wait_for(independent.send_completed.wait(), timeout=1)

    assert not broadcast.done()

    blocked.release_send.set()
    await asyncio.wait_for(broadcast, timeout=1)


@pytest.mark.asyncio
async def test_progress_setup_failure_disconnects_and_retires_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = _BlockingWebSocket()
    broadcaster = SimpleNamespace(
        connect=AsyncMock(side_effect=RuntimeError("connect failed")),
        disconnect=AsyncMock(),
    )
    progress_manager = SimpleNamespace(get_task_progress=lambda _task_id: None)
    monkeypatch.setattr(
        progress_ws_api,
        "get_authenticated_user",
        AsyncMock(return_value=SimpleNamespace(id=1)),
    )
    monkeypatch.setattr(
        progress_ws_api,
        "get_progress_manager",
        lambda: progress_manager,
    )
    monkeypatch.setattr(progress_ws_api, "progress_broadcaster", broadcaster)

    await progress_ws_api.progress_websocket_endpoint(websocket, "task-1", "token")

    broadcaster.disconnect.assert_awaited_once()
    assert websocket.closed
    with pytest.raises(WebSocketWriterRetiredError):
        await send_websocket_text(websocket, "late")


@pytest.mark.asyncio
async def test_progress_cancellation_retires_before_draining_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = _BlockingWebSocket()
    disconnect_started = asyncio.Event()
    release_disconnect = asyncio.Event()

    async def disconnect(_task_id: str, _adapter: object) -> None:
        disconnect_started.set()
        await release_disconnect.wait()

    broadcaster = SimpleNamespace(
        connect=AsyncMock(),
        disconnect=disconnect,
    )
    progress_manager = SimpleNamespace(get_task_progress=lambda _task_id: None)
    monkeypatch.setattr(
        progress_ws_api,
        "get_authenticated_user",
        AsyncMock(return_value=SimpleNamespace(id=1)),
    )
    monkeypatch.setattr(
        progress_ws_api,
        "get_progress_manager",
        lambda: progress_manager,
    )
    monkeypatch.setattr(progress_ws_api, "progress_broadcaster", broadcaster)

    endpoint = asyncio.create_task(
        progress_ws_api.progress_websocket_endpoint(websocket, "task-1", "token")
    )
    await websocket.receive_started.wait()
    endpoint.cancel()
    await disconnect_started.wait()
    try:
        assert not endpoint.done()
        with pytest.raises(WebSocketWriterRetiredError):
            await send_websocket_text(websocket, "late")
    finally:
        release_disconnect.set()
    with pytest.raises(asyncio.CancelledError):
        await endpoint


@pytest.mark.asyncio
async def test_progress_initial_and_live_updates_share_one_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = _ControlledSendWebSocket()
    adapter_ready = asyncio.Event()
    adapter: object | None = None

    async def connect(_task_id: str, connected_adapter: object) -> None:
        nonlocal adapter
        adapter = connected_adapter
        adapter_ready.set()

    broadcaster = SimpleNamespace(
        connect=connect,
        disconnect=AsyncMock(),
    )
    task_progress = SimpleNamespace(
        task_id="task-1",
        user_id=1,
        task_type="ingestion",
        status="running",
        current_step="parse",
        overall_progress=0.5,
        start_time=1.0,
        end_time=None,
        metadata={},
    )
    progress_manager = SimpleNamespace(get_task_progress=lambda _task_id: task_progress)
    monkeypatch.setattr(
        progress_ws_api,
        "get_authenticated_user",
        AsyncMock(return_value=SimpleNamespace(id=1)),
    )
    monkeypatch.setattr(
        progress_ws_api,
        "get_progress_manager",
        lambda: progress_manager,
    )
    monkeypatch.setattr(progress_ws_api, "progress_broadcaster", broadcaster)

    endpoint = asyncio.create_task(
        progress_ws_api.progress_websocket_endpoint(websocket, "task-1", "token")
    )
    await adapter_ready.wait()
    await websocket.send_started.wait()
    assert adapter is not None
    live_update = asyncio.create_task(adapter.send_text("live"))
    await asyncio.sleep(0)
    assert len(websocket.attempts) == 1

    websocket.release_send.set()
    await live_update
    await websocket.receive_started.wait()
    endpoint.cancel()
    with pytest.raises(asyncio.CancelledError):
        await endpoint

    assert len(websocket.attempts) == 2
    assert websocket.max_active_sends == 1


@pytest.mark.asyncio
async def test_builder_cancellation_retires_before_draining_active_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler_started = asyncio.Event()
    handler_cleanup_started = asyncio.Event()
    release_handler_cleanup = asyncio.Event()

    class BuilderWebSocket(_BlockingWebSocket):
        def __init__(self) -> None:
            super().__init__()
            self.receive_count = 0

        async def receive_text(self) -> str:
            self.receive_count += 1
            if self.receive_count == 1:
                return "{}"
            self.receive_started.set()
            await asyncio.Future()
            raise AssertionError("unreachable")

    async def handler(*_args: object) -> None:
        handler_started.set()
        try:
            await asyncio.Future()
        finally:
            handler_cleanup_started.set()
            await release_handler_cleanup.wait()

    websocket = BuilderWebSocket()
    monkeypatch.setattr(
        websocket_api,
        "get_authenticated_user",
        AsyncMock(return_value=SimpleNamespace(id=1)),
    )
    monkeypatch.setattr(websocket_api, "handle_builder_chat", handler)
    endpoint = asyncio.create_task(
        websocket_api.websocket_builder_chat_endpoint(websocket, "token")
    )
    await handler_started.wait()
    await websocket.receive_started.wait()
    endpoint.cancel()
    await handler_cleanup_started.wait()
    try:
        assert not endpoint.done()
        with pytest.raises(WebSocketWriterRetiredError):
            await send_websocket_text(websocket, "late")
    finally:
        release_handler_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await endpoint
