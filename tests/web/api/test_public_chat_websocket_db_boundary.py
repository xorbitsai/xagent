from __future__ import annotations

import json
import threading
from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, WebSocketDisconnect

from xagent.web.api import public_chat_access
from xagent.web.api.websocket import WebSocketPrincipal


class _SingleMessageWebSocket:
    def __init__(self, message: dict[str, object]) -> None:
        self._message = json.dumps(message)
        self._received = False
        self.close = AsyncMock()

    async def receive_text(self) -> str:
        if self._received:
            raise WebSocketDisconnect()
        self._received = True
        return self._message


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint_kind", ["public", "share"])
async def test_public_access_websocket_authorization_uses_worker_owned_session(
    monkeypatch: pytest.MonkeyPatch,
    endpoint_kind: str,
) -> None:
    event_loop_thread = threading.get_ident()
    operation_threads: list[int] = []
    closed_sessions: list[bool] = []
    session = object()
    user = SimpleNamespace(id=73, is_admin=True)
    access_context = SimpleNamespace(user=user)

    class TrackingSession:
        def __enter__(self) -> object:
            operation_threads.append(threading.get_ident())
            return session

        def __exit__(self, _exc_type, _exc, _traceback) -> None:
            closed_sessions.append(True)

    monkeypatch.setattr(
        public_chat_access,
        "get_session_local",
        lambda: TrackingSession,
        raising=False,
    )

    def load_public_context(
        token: str,
        db: object,
        *,
        expected_auth_mode: str,
    ) -> object:
        operation_threads.append(threading.get_ident())
        assert token == "widget-token"
        assert db is session
        assert expected_auth_mode == "widget"
        return access_context

    def load_share_context(token: str, db: object) -> object:
        operation_threads.append(threading.get_ident())
        assert token == "share-token"
        assert db is session
        return access_context

    def authorize_task(db: object, task_id: int, context: object) -> None:
        operation_threads.append(threading.get_ident())
        assert db is session
        assert task_id == 41
        assert context is access_context

    monkeypatch.setattr(
        public_chat_access,
        "get_public_chat_user",
        load_public_context,
    )
    monkeypatch.setattr(
        public_chat_access,
        "get_share_chat_user",
        load_share_context,
    )
    monkeypatch.setattr(
        public_chat_access,
        "get_task_for_public_context",
        authorize_task,
    )
    monkeypatch.setattr(
        public_chat_access,
        "get_task_for_share_context",
        authorize_task,
    )

    if endpoint_kind == "public":
        principal = await public_chat_access._authorize_public_chat_websocket(
            token="widget-token",
            task_id=41,
            expected_auth_mode="widget",
        )
    else:
        principal = await public_chat_access._authorize_share_chat_websocket(
            token="share-token",
            task_id=41,
        )

    assert closed_sessions == [True]
    assert operation_threads
    assert all(thread_id != event_loop_thread for thread_id in operation_threads)
    assert is_dataclass(principal)
    assert principal.__dataclass_params__.frozen is True
    assert {field.name for field in fields(principal)} == {"id", "is_admin"}
    assert principal == WebSocketPrincipal(id=73, is_admin=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint_kind", ["public", "share"])
async def test_public_access_websocket_revalidates_with_frozen_principal(
    monkeypatch: pytest.MonkeyPatch,
    endpoint_kind: str,
) -> None:
    websocket = _SingleMessageWebSocket({"type": "chat", "message": "hello"})
    initial_principal = WebSocketPrincipal(id=7, is_admin=False)
    current_principal = WebSocketPrincipal(id=7, is_admin=False)
    authorize_public = AsyncMock(side_effect=[initial_principal, current_principal])
    authorize_share = AsyncMock(side_effect=[initial_principal, current_principal])
    connection_manager = MagicMock()
    connection_manager.connect = AsyncMock()
    connection_manager.disconnect = MagicMock()
    status = AsyncMock()
    chat = AsyncMock()

    monkeypatch.setattr(
        public_chat_access,
        "_authorize_public_chat_websocket",
        authorize_public,
        raising=False,
    )
    monkeypatch.setattr(
        public_chat_access,
        "_authorize_share_chat_websocket",
        authorize_share,
        raising=False,
    )
    monkeypatch.setattr(public_chat_access, "manager", connection_manager)
    monkeypatch.setattr(public_chat_access, "handle_status_request", status)
    monkeypatch.setattr(public_chat_access, "handle_chat_message", chat)
    monkeypatch.setattr(
        public_chat_access,
        "db_session_context",
        MagicMock(side_effect=AssertionError("request Session used on event loop")),
    )

    if endpoint_kind == "public":
        await public_chat_access.public_chat_websocket_endpoint(
            websocket=websocket,
            task_id=42,
            token="widget-token",
            expected_auth_mode="widget",
        )
        assert authorize_public.await_args_list[0].kwargs == {
            "token": "widget-token",
            "task_id": 42,
            "expected_auth_mode": "widget",
        }
        assert authorize_public.await_count == 2
        authorize_share.assert_not_awaited()
    else:
        await public_chat_access.share_chat_websocket_endpoint(
            websocket=websocket,
            task_id=42,
            token="share-token",
        )
        assert authorize_share.await_args_list[0].kwargs == {
            "token": "share-token",
            "task_id": 42,
        }
        assert authorize_share.await_count == 2
        authorize_public.assert_not_awaited()

    status.assert_awaited_once_with(websocket, 42, initial_principal)
    chat.assert_awaited_once()
    message_data = chat.await_args.args[2]
    assert message_data["user_id"] == current_principal.id
    assert message_data["user"] is current_principal
    connection_manager.disconnect.assert_called_once_with(websocket)


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint_kind", ["public", "share"])
async def test_public_access_websocket_preserves_auth_close_codes(
    monkeypatch: pytest.MonkeyPatch,
    endpoint_kind: str,
) -> None:
    principal = WebSocketPrincipal(id=7, is_admin=False)
    websocket = _SingleMessageWebSocket({"type": "chat", "message": "hello"})
    authorize = AsyncMock(
        side_effect=[
            principal,
            HTTPException(status_code=403, detail="Access revoked"),
        ]
    )
    connection_manager = MagicMock()
    connection_manager.connect = AsyncMock()
    connection_manager.disconnect = MagicMock()
    monkeypatch.setattr(public_chat_access, "manager", connection_manager)
    monkeypatch.setattr(
        public_chat_access,
        "handle_status_request",
        AsyncMock(),
    )
    monkeypatch.setattr(
        public_chat_access,
        "handle_chat_message",
        AsyncMock(),
    )
    monkeypatch.setattr(
        public_chat_access,
        "db_session_context",
        MagicMock(side_effect=AssertionError("request Session used on event loop")),
    )

    if endpoint_kind == "public":
        monkeypatch.setattr(
            public_chat_access,
            "_authorize_public_chat_websocket",
            authorize,
            raising=False,
        )
        await public_chat_access.public_chat_websocket_endpoint(
            websocket=websocket,
            task_id=42,
            token="widget-token",
            expected_auth_mode="widget",
        )
    else:
        monkeypatch.setattr(
            public_chat_access,
            "_authorize_share_chat_websocket",
            authorize,
            raising=False,
        )
        await public_chat_access.share_chat_websocket_endpoint(
            websocket=websocket,
            task_id=42,
            token="share-token",
        )

    websocket.close.assert_awaited_once_with(code=4003, reason="Access revoked")
    connection_manager.disconnect.assert_called_once_with(websocket)


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint_kind", ["public", "share"])
async def test_public_access_websocket_initial_auth_failure_stays_4001(
    monkeypatch: pytest.MonkeyPatch,
    endpoint_kind: str,
) -> None:
    websocket = _SingleMessageWebSocket({"type": "chat", "message": "hello"})
    authorize = AsyncMock(
        side_effect=HTTPException(status_code=401, detail="Invalid token")
    )
    connection_manager = MagicMock()
    connection_manager.connect = AsyncMock()
    connection_manager.disconnect = MagicMock()
    monkeypatch.setattr(public_chat_access, "manager", connection_manager)

    if endpoint_kind == "public":
        monkeypatch.setattr(
            public_chat_access,
            "_authorize_public_chat_websocket",
            authorize,
            raising=False,
        )
        await public_chat_access.public_chat_websocket_endpoint(
            websocket=websocket,
            task_id=42,
            token="widget-token",
            expected_auth_mode="widget",
        )
    else:
        monkeypatch.setattr(
            public_chat_access,
            "_authorize_share_chat_websocket",
            authorize,
            raising=False,
        )
        await public_chat_access.share_chat_websocket_endpoint(
            websocket=websocket,
            task_id=42,
            token="share-token",
        )

    websocket.close.assert_awaited_once_with(
        code=4001,
        reason="Authentication required",
    )
    connection_manager.connect.assert_not_awaited()
    connection_manager.disconnect.assert_not_called()
