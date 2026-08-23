from __future__ import annotations

import json
import logging
import threading
from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, WebSocketDisconnect
from starlette.websockets import WebSocketState

from xagent.web.api import public_chat_access
from xagent.web.api.websocket import WebSocketPrincipal


class _SingleMessageWebSocket:
    def __init__(self, message: dict[str, object]) -> None:
        self._message = json.dumps(message)
        self._received = False
        # The share endpoint accepts the handshake before its auth check (#973),
        # so a post-accept close preserves the code/reason on the wire.
        self.accept = AsyncMock()
        self.close = AsyncMock()
        self.application_state = WebSocketState.CONNECTED
        self.scope = {"extensions": {}, "route": SimpleNamespace(path="/public/ws")}

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
    # The widget entity ids are read by the public (widget) authorize path to
    # derive the principal's rate-limit key (#1056); the share path never
    # touches them.
    access_context = SimpleNamespace(
        user=user,
        guest_id="guest-73",
        widget_agent_id=7,
        widget_workforce_id=None,
    )

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
    assert {field.name for field in fields(principal)} == {
        "id",
        "is_admin",
        "guest_id",
        "widget_entity_key",
        "voice",
    }
    # Only the widget authorize path derives the entity rate-limit key (#1056).
    assert principal == WebSocketPrincipal(
        id=73,
        is_admin=True,
        guest_id="guest-73",
        widget_entity_key="agent:7" if endpoint_kind == "public" else None,
    )


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
async def test_public_access_websocket_initial_http_auth_failure_maps_to_4003(
    monkeypatch: pytest.MonkeyPatch,
    endpoint_kind: str,
) -> None:
    # Both endpoints map an auth-time HTTPException to a 4003 (#973 for share,
    # #1057 for widget): the connect-time analogue of the per-message
    # revalidation 4003, and the common sequence for a revoked widget guest who
    # reloads the page and reconnects rather than staying on a live socket.
    websocket = _SingleMessageWebSocket({"type": "chat", "message": "hello"})
    authorize = AsyncMock(
        side_effect=HTTPException(status_code=403, detail="Widget is unavailable")
    )
    connection_manager = MagicMock()
    connection_manager.connect = AsyncMock()
    connection_manager.register_connection = MagicMock()
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

    websocket.accept.assert_awaited_once()
    websocket.close.assert_awaited_once_with(
        code=4003,
        reason="Widget is unavailable",
    )
    # Rejected before registration: neither the async accept-and-register helper
    # nor the register-only path runs, so the socket never joins task broadcasts.
    connection_manager.connect.assert_not_awaited()
    connection_manager.register_connection.assert_not_called()
    connection_manager.disconnect.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint_kind", ["public", "share"])
@pytest.mark.parametrize("phase", ["initial", "revalidation"])
async def test_public_access_websocket_infrastructure_failure_closes_with_1011(
    monkeypatch: pytest.MonkeyPatch,
    endpoint_kind: str,
    phase: str,
) -> None:
    websocket = _SingleMessageWebSocket({"type": "chat", "message": "hello"})
    principal = WebSocketPrincipal(id=7, is_admin=False)
    failure = RuntimeError("database is down token=secret")
    authorize = AsyncMock(
        side_effect=failure if phase == "initial" else [principal, failure]
    )
    connection_manager = MagicMock()
    connection_manager.connect = AsyncMock()
    connection_manager.register_connection = MagicMock()
    connection_manager.disconnect = MagicMock()
    monkeypatch.setattr(public_chat_access, "manager", connection_manager)
    monkeypatch.setattr(public_chat_access, "handle_status_request", AsyncMock())

    if endpoint_kind == "public":
        monkeypatch.setattr(
            public_chat_access,
            "_authorize_public_chat_websocket",
            authorize,
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
        )
        await public_chat_access.share_chat_websocket_endpoint(
            websocket=websocket,
            task_id=42,
            token="share-token",
        )

    websocket.accept.assert_awaited_once()
    websocket.close.assert_awaited_once_with(
        code=1011,
        reason="Internal server error",
    )
    connection_manager.connect.assert_not_awaited()
    if phase == "initial":
        connection_manager.register_connection.assert_not_called()
        connection_manager.disconnect.assert_not_called()
    else:
        connection_manager.register_connection.assert_called_once_with(websocket, 42)
        connection_manager.disconnect.assert_called_once_with(websocket)


class _PublicResolverSession:
    def __init__(self, query_error: Exception | None = None) -> None:
        self._bind = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))
        self.query_error = query_error
        self.bind_count = 0
        self.query_count = 0

    def get_bind(self) -> object:
        self.bind_count += 1
        return self._bind

    def query(self, _model: object) -> "_PublicResolverSession":
        self.query_count += 1
        if self.query_error is not None:
            raise self.query_error
        raise AssertionError("malformed public IDs must be rejected before a query")


def _resolve_public_token(
    resolver_kind: str, token: str, db: _PublicResolverSession
) -> object:
    if resolver_kind == "widget":
        return public_chat_access.get_public_chat_user(
            token, db, expected_auth_mode="widget"
        )
    return public_chat_access.get_share_chat_user(token, db)


@pytest.mark.parametrize("resolver_kind", ["widget", "share"])
def test_public_token_resolvers_propagate_database_failures_by_identity(
    resolver_kind: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    failure = RuntimeError("database unavailable")
    db = _PublicResolverSession(failure)
    caplog.set_level(logging.ERROR, logger=public_chat_access.__name__)
    if resolver_kind == "widget":
        token = public_chat_access.create_public_chat_access_token(
            {
                "user_id": 1,
                "channel_id": 2,
                "guest_id": "guest",
                "auth_mode": "widget",
                "widget_agent_id": 3,
            }
        )
    else:
        token = public_chat_access.create_public_chat_access_token(
            {
                "user_id": 1,
                "guest_id": "guest",
                "auth_mode": "share",
                "share_agent_id": 3,
                "share_token": "share",
            }
        )

    with pytest.raises(RuntimeError) as raised:
        _resolve_public_token(resolver_kind, token, db)

    assert raised.value is failure
    assert db.bind_count == 1
    assert db.query_count == 1
    assert "database unavailable" not in caplog.text


@pytest.mark.parametrize(
    ("guest_id", "expected_status"),
    (("", 401), (7, 401), ("   ", None)),
    ids=("empty", "non-string", "whitespace"),
)
def test_public_widget_guest_id_requires_nonempty_string(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    guest_id: object,
    expected_status: int | None,
) -> None:
    payload = {
        "type": "widget",
        "user_id": 1,
        "channel_id": None,
        "guest_id": guest_id,
        "auth_mode": "widget",
        "widget_agent_id": 3,
    }
    db = MagicMock()
    db.get_bind.return_value = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))
    db.query.return_value.filter.return_value.first.return_value = SimpleNamespace(
        id=1,
        is_admin=False,
    )
    monkeypatch.setattr(
        public_chat_access, "_decode_public_token", lambda _token: payload
    )
    monkeypatch.setattr(
        public_chat_access, "ensure_widget_agent_available", MagicMock()
    )
    caplog.set_level(logging.INFO, logger=public_chat_access.__name__)

    if expected_status is None:
        context = public_chat_access.get_public_chat_user(
            "signed-token", db, expected_auth_mode="widget"
        )

        assert context.guest_id == guest_id
        return

    with pytest.raises(HTTPException) as raised:
        public_chat_access.get_public_chat_user(
            "signed-token", db, expected_auth_mode="widget"
        )

    assert raised.value.status_code == expected_status
    db.get_bind.assert_not_called()
    assert "reason=INVALID_CLAIMS" in caplog.text
    assert "signed-token" not in caplog.text
    assert "database unavailable" not in caplog.text


def test_public_token_failure_projection_preserves_http_exception_identity() -> None:
    expected = HTTPException(status_code=403, detail="Access denied")

    assert (
        public_chat_access._project_public_token_failure(
            expected, invalid_detail="Invalid widget token"
        )
        is expected
    )


def test_public_token_failure_projection_leaves_defects_unprojected() -> None:
    """A non-credential exception (e.g. the ShareChatAccessContext invariant
    ValueError, #1225) must NOT be laundered into a 401: the projection
    returns None so the caller re-raises the defect loudly (#1214)."""
    assert (
        public_chat_access._project_public_token_failure(
            ValueError("boom"), invalid_detail="Invalid share token"
        )
        is None
    )


@pytest.mark.parametrize(
    ("resolver_kind", "claim", "value", "expected_bind_count"),
    (
        ("widget", "user_id", "1", 0),
        ("widget", "channel_id", "2", 0),
        ("widget", "widget_agent_id", "3", 0),
        ("widget", "widget_workforce_id", "3", 0),
        ("share", "user_id", "1", 0),
        ("share", "share_agent_id", "3", 0),
        ("share", "share_workforce_id", "3", 0),
        ("widget", "user_id", 2**63, 1),
        ("widget", "channel_id", 2**63, 1),
        ("widget", "widget_agent_id", 2**63, 1),
        ("widget", "widget_workforce_id", 2**63, 1),
        ("share", "user_id", 2**63, 1),
        ("share", "share_agent_id", 2**63, 1),
        ("share", "share_workforce_id", 2**63, 1),
    ),
)
def test_public_token_resolvers_reject_invalid_ids_before_query(
    resolver_kind: str, claim: str, value: object, expected_bind_count: int
) -> None:
    db = _PublicResolverSession()
    if resolver_kind == "widget":
        claims = {
            "user_id": 1,
            "channel_id": 2,
            "guest_id": "guest",
            "auth_mode": "widget",
            "widget_agent_id": 3,
        }
    else:
        claims = {
            "user_id": 1,
            "guest_id": "guest",
            "auth_mode": "share",
            "share_agent_id": 3,
            "share_token": "share",
        }
    if "workforce" in claim:
        claims.pop("widget_agent_id" if resolver_kind == "widget" else "share_agent_id")
    claims[claim] = value
    token = public_chat_access.create_public_chat_access_token(claims)

    with pytest.raises(HTTPException) as raised:
        _resolve_public_token(resolver_kind, token, db)

    assert raised.value.status_code == 401
    assert db.bind_count == expected_bind_count
    assert db.query_count == 0
