"""WebSocket message handlers that reach the planned-restart execution hold.

Each test drives a real handler. Direct WS executions are refused in band
before any side effect and, when admitted, are counted until the handler
returns; follow-ups that already have a durable command path keep it. The
trigger HTTP entries are covered beside their own fixtures in
``test_agent_triggers`` / ``test_workforce_triggers``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import WebSocketDisconnect
from starlette.websockets import WebSocketState

from xagent.web.api import websocket as ws_api
from xagent.web.services import release_hold as release_hold_module
from xagent.web.services.release_hold import ReleaseHold


@pytest.fixture
def hold(monkeypatch: pytest.MonkeyPatch) -> ReleaseHold:
    fresh = ReleaseHold()
    monkeypatch.setattr(release_hold_module, "release_hold", fresh)
    monkeypatch.setattr(ws_api, "release_hold", fresh)
    return fresh


# --- WebSocket message handlers --------------------------------------------------


class _Frames:
    """Records every personal frame the handlers send to a fake socket."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.sent: list[dict[str, Any]] = []

        async def record(message: dict[str, Any], _websocket: Any) -> None:
            self.sent.append(message)

        monkeypatch.setattr(ws_api.manager, "send_personal_message", record)
        monkeypatch.setattr(ws_api.manager, "broadcast_to_task", AsyncMock())

    def codes(self) -> list[str | None]:
        return [frame.get("error_code") for frame in self.sent]


def _socket() -> Any:
    return SimpleNamespace(state=SimpleNamespace(preview_task_id=None))


def _user() -> Any:
    return SimpleNamespace(id=7, is_admin=False)


async def test_execute_task_frame_is_refused_before_side_effects(
    hold, monkeypatch
) -> None:
    frames = _Frames(monkeypatch)
    started = AsyncMock()
    monkeypatch.setattr(ws_api.task_start_service, "execute_existing_task", started)
    hold.close("release r1")
    await ws_api.handle_execute_task(_socket(), 5, {"user": _user()})
    started.assert_not_called()
    # No legacy "execution_started" acknowledgement was sent first.
    assert [frame["type"] for frame in frames.sent] == ["error"]
    assert frames.codes() == ["execution_paused"]


async def test_admitted_execute_task_is_counted_until_registered(
    hold, monkeypatch
) -> None:
    _Frames(monkeypatch)
    seen: list[list[str]] = []

    async def start(**_kwargs: Any) -> None:
        seen.append([r["kind"] for r in hold.snapshot()["responsibilities"]])

    monkeypatch.setattr(ws_api.task_start_service, "execute_existing_task", start)
    monkeypatch.setattr(
        ws_api,
        "_load_legacy_execute_task_request_sync",
        lambda *_a, **_k: SimpleNamespace(
            task_id=5,
            task_owner_user_id=7,
            task_source=None,
            task_description="d",
            task_context={},
            task_info={},
            created_at=None,
        ),
    )
    await ws_api.handle_execute_task(_socket(), 5, {"user": _user()})
    assert seen == [["ws_execute_task"]]
    assert hold.snapshot()["responsibilities"] == []


async def test_new_preview_frame_is_refused_without_creating_a_task(
    hold, monkeypatch
) -> None:
    frames = _Frames(monkeypatch)
    create = AsyncMock()
    monkeypatch.setattr("xagent.web.api.chat.create_task", create)
    hold.close("release r1")
    socket = _socket()
    await ws_api.handle_build_preview_execution(
        socket, {"type": "preview", "message": "hi"}, _user()
    )
    create.assert_not_called()
    assert socket.state.preview_task_id is None
    assert frames.codes() == ["execution_paused"]


async def test_preview_follow_up_uses_the_durable_chat_command(
    hold, monkeypatch
) -> None:
    _Frames(monkeypatch)
    chat = AsyncMock()
    monkeypatch.setattr(ws_api, "handle_chat_message", chat)
    hold.close("release r1")
    socket = _socket()
    socket.state.preview_task_id = 12
    await ws_api.handle_build_preview_execution(
        socket, {"type": "preview", "message": "again"}, _user()
    )
    # Not refused here: the durable command stays pending under the held cap
    # unless task 12 is active, which is the dispatcher's decision.
    assert chat.await_args.args[1] == 12


async def test_missing_task_chat_executes_directly_so_it_is_admitted(
    hold, monkeypatch
) -> None:
    frames = _Frames(monkeypatch)
    monkeypatch.setattr(
        ws_api, "_enqueue_websocket_task_command", AsyncMock(return_value=None)
    )
    replacement = AsyncMock()
    monkeypatch.setattr(
        ws_api.command_execution_service, "handle_missing_task_message", replacement
    )
    hold.close("release r1")
    await ws_api.handle_chat_message(
        _socket(),
        9,
        {"type": "chat", "message": "hi", "client_message_id": "m-1", "user": _user()},
    )
    replacement.assert_not_called()
    deliveries = [f for f in frames.sent if f.get("type") == "message_rejected"]
    assert deliveries, frames.sent
    assert deliveries[0].get("rejection_outcome") == "not_accepted"
    assert "execution_paused" in frames.codes()


class _OneFrameSocket:
    """Just enough of a WebSocket for the public/share endpoints' loop."""

    def __init__(self, message: dict[str, Any]) -> None:
        self._message = json.dumps(message)
        self._received = False
        self.accept = AsyncMock()
        self.close = AsyncMock()
        self.application_state = WebSocketState.CONNECTED
        self.scope = {"extensions": {}, "route": SimpleNamespace(path="/public/ws")}
        self.client = SimpleNamespace(host="127.0.0.1", port=1)
        self.headers: dict[str, str] = {}
        self.state = SimpleNamespace(preview_task_id=None)

    async def receive_text(self) -> str:
        if self._received:
            raise WebSocketDisconnect()
        self._received = True
        return self._message


@pytest.mark.parametrize("endpoint_kind", ["widget", "share"])
async def test_legacy_widget_and_share_execute_task_reach_the_hold(
    hold, monkeypatch, endpoint_kind
) -> None:
    """``/api/widget/chat/ws`` and ``/api/share/chat/ws`` run the shared handler."""
    from xagent.web.api import public_chat_access
    from xagent.web.api.websocket import WebSocketPrincipal

    frames = _Frames(monkeypatch)
    principal = WebSocketPrincipal(id=7, is_admin=False)
    for name in ("_authorize_public_chat_websocket", "_authorize_share_chat_websocket"):
        monkeypatch.setattr(
            public_chat_access, name, AsyncMock(return_value=principal), raising=False
        )
    connections = MagicMock()
    monkeypatch.setattr(public_chat_access, "manager", connections)
    monkeypatch.setattr(public_chat_access, "handle_status_request", AsyncMock())
    started = AsyncMock()
    monkeypatch.setattr(ws_api.task_start_service, "execute_existing_task", started)
    hold.close("release r1")
    socket = _OneFrameSocket({"type": "execute_task"})
    if endpoint_kind == "widget":
        await public_chat_access.public_chat_websocket_endpoint(
            websocket=socket, task_id=42, token="t", expected_auth_mode="widget"
        )
    else:
        await public_chat_access.share_chat_websocket_endpoint(
            websocket=socket, task_id=42, token="t"
        )
    started.assert_not_called()
    assert frames.codes() == ["execution_paused"]
    assert hold.snapshot()["refused"] == {"ws_execute_task": 1}


class _FramesSocket(_OneFrameSocket):
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        super().__init__({})
        self._messages = [json.dumps(message) for message in messages]
        self.sent_text: list[dict[str, Any]] = []

    async def receive_text(self) -> str:
        if not self._messages:
            raise WebSocketDisconnect()
        return self._messages.pop(0)

    async def send_text(self, text: str) -> None:
        self.sent_text.append(json.loads(text))


async def test_preview_controls_stay_reachable_while_held(hold, monkeypatch) -> None:
    """``/ws/build/preview``: a new run is refused; pause, resume (durable)
    and clear_context are still handled, and clearing cancels nothing."""
    frames = _Frames(monkeypatch)
    user = _user()
    monkeypatch.setattr(ws_api, "get_authenticated_user", AsyncMock(return_value=user))
    monkeypatch.setattr(ws_api, "accept_authenticated_websocket", AsyncMock())
    pause, resume = AsyncMock(), AsyncMock()
    monkeypatch.setattr(ws_api, "handle_pause_task", pause)
    monkeypatch.setattr(ws_api, "handle_resume_task", resume)
    from xagent.web.services.task_execution import background_task_manager

    cancel = AsyncMock()
    monkeypatch.setattr(background_task_manager, "cancel_task", cancel)
    hold.close("release r1")
    socket = _FramesSocket(
        [
            {"type": "preview", "message": "new run"},
            {"type": "pause"},
            {"type": "resume"},
            {"type": "clear_context"},
        ]
    )
    socket.state.preview_task_id = None
    endpoint = ws_api.websocket_build_preview_endpoint.__wrapped__

    async def bind_after_refusal(*_args: Any, **_kwargs: Any) -> None:
        # The old run this socket was watching (bound before the hold).
        socket.state.preview_task_id = 12

    monkeypatch.setattr(
        ws_api, "send_execution_paused", AsyncMock(side_effect=bind_after_refusal)
    )
    await endpoint(socket, token=None)

    assert pause.await_args.args[1] == 12
    assert resume.await_args.args[1] == 12
    assert [m["type"] for m in socket.sent_text] == ["context_cleared"]
    assert socket.state.preview_task_id is None
    cancel.assert_not_called()
    assert frames.sent == []
