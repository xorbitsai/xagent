"""Tests for the shared authenticated WebSocket transport owner."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import is_dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import WebSocket
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from starlette.websockets import WebSocketState

from tests.web.auth_token_cases import (
    REJECTED_ACCESS_TOKEN_CASES,
    WRONG_TYPE_ACCESS_TOKEN_CASE,
    RejectedAccessTokenCase,
    build_access_token,
)
from xagent.web.api import progress_ws
from xagent.web.api import websocket as websocket_api
from xagent.web.api import websocket_auth


class _BoundSQLiteSession:
    """Small worker Session double for real access-token resolution."""

    def __init__(self, query_exception: Exception | None = None) -> None:
        self._bind = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))
        self._query_exception = query_exception
        self.enter_count = 0
        self.exit_count = 0
        self.query_count = 0

    def __enter__(self) -> "_BoundSQLiteSession":
        self.enter_count += 1
        return self

    def __exit__(self, *_exception: object) -> None:
        self.exit_count += 1

    def get_bind(self) -> SimpleNamespace:
        return self._bind

    def query(self, _model: object) -> "_BoundSQLiteSession":
        self.query_count += 1
        if self._query_exception is not None:
            raise self._query_exception
        return self

    def filter(self, *_conditions: object) -> "_BoundSQLiteSession":
        return self

    def first(self) -> None:
        return None


def _asgi_websocket(
    *, denial_extension: bool
) -> tuple[WebSocket, list[dict[str, object]]]:
    """Create a pre-handshake WebSocket and collect its outgoing ASGI messages."""

    messages: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return {"type": "websocket.connect"}

    async def send(message: dict[str, object]) -> None:
        messages.append(message)

    extensions = {"websocket.http.response": {}} if denial_extension else {}
    websocket = WebSocket(
        {
            "type": "websocket",
            "asgi": {"version": "3.0"},
            "scheme": "ws",
            "path": "/ws/chat/1",
            "raw_path": b"/ws/chat/1",
            "query_string": b"",
            "headers": [],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
            "subprotocols": [],
            "extensions": extensions,
        },
        receive,
        send,
    )
    return websocket, messages


@pytest.mark.asyncio
async def test_missing_token_returns_none_without_starting_database_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_db_io = MagicMock()
    monkeypatch.setattr(websocket_auth, "run_db_io_cancellation_safe", run_db_io)

    assert await websocket_auth.get_authenticated_user(MagicMock(), None) is None
    run_db_io.assert_not_called()


@pytest.mark.parametrize("case", REJECTED_ACCESS_TOKEN_CASES, ids=lambda case: case.id)
def test_loader_rejects_every_credential_reason_inside_worker_session(
    monkeypatch: pytest.MonkeyPatch,
    case: RejectedAccessTokenCase,
) -> None:
    session = _BoundSQLiteSession()
    monkeypatch.setattr(websocket_auth, "get_session_local", lambda: lambda: session)

    assert websocket_auth._load_websocket_principal_sync(case.build_token()) is None
    assert session.enter_count == 1
    assert session.exit_count == 1
    assert session.query_count == (1 if case.expected_detail == "User not found" else 0)


@pytest.mark.asyncio
async def test_valid_token_returns_frozen_detached_principal_from_worker_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_loop_thread = threading.get_ident()
    auth_threads: list[int] = []
    closed: list[bool] = []

    class TrackingSession:
        def __enter__(self) -> "TrackingSession":
            return self

        def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
            closed.append(True)

    def authenticate(_token: str, _db: object) -> SimpleNamespace:
        auth_threads.append(threading.get_ident())
        return SimpleNamespace(id=73, is_admin=True)

    monkeypatch.setattr(websocket_auth, "get_session_local", lambda: TrackingSession)
    monkeypatch.setattr(websocket_auth, "get_user_from_websocket_token", authenticate)

    principal = await websocket_auth.get_authenticated_user(MagicMock(), "signed")

    assert principal == websocket_auth.WebSocketPrincipal(id=73, is_admin=True)
    assert is_dataclass(principal)
    assert principal.__dataclass_params__.frozen is True
    assert auth_threads == [auth_threads[0]]
    assert auth_threads[0] != event_loop_thread
    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("preferences", "expected_voice"),
    [
        ({"voice": "concise"}, "concise"),
        (None, None),
        ("not-a-dict", None),
    ],
)
async def test_principal_reduces_voice_from_preferences_without_a_second_query(
    monkeypatch: pytest.MonkeyPatch,
    preferences: object,
    expected_voice: str | None,
) -> None:
    """The builder-chat websocket assistant applies the same voice
    preference a saved agent does (see apply_user_voice's call sites) -
    the principal must carry `voice` from the one query that already
    authenticates the token, not a second lookup against a session that
    may be closed by the time a system prompt is assembled."""

    def authenticate(_token: str, _db: object) -> SimpleNamespace:
        return SimpleNamespace(id=73, is_admin=True, preferences=preferences)

    session = _BoundSQLiteSession()
    monkeypatch.setattr(websocket_auth, "get_session_local", lambda: lambda: session)
    monkeypatch.setattr(websocket_auth, "get_user_from_websocket_token", authenticate)

    principal = await websocket_auth.get_authenticated_user(MagicMock(), "signed")

    assert principal == websocket_auth.WebSocketPrincipal(
        id=73, is_admin=True, voice=expected_voice
    )


@pytest.mark.asyncio
async def test_cancellation_propagates_from_database_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def cancelled(_operation: object) -> None:
        raise asyncio.CancelledError()

    monkeypatch.setattr(websocket_auth, "run_db_io_cancellation_safe", cancelled)

    with pytest.raises(asyncio.CancelledError):
        await websocket_auth.get_authenticated_user(MagicMock(), "signed")


@pytest.mark.asyncio
async def test_operational_auth_failure_sends_sanitized_extension_denial(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    websocket, messages = _asgi_websocket(denial_extension=True)
    route_template = "/ws/chat/{task_id}"
    query_secret = "query-secret-value"
    websocket.scope["route"] = SimpleNamespace(path=route_template)
    websocket.scope["query_string"] = f"debug={query_secret}".encode()
    timeout = SQLAlchemyTimeoutError("database pool token=secret", None, None)
    session = _BoundSQLiteSession(query_exception=timeout)
    monkeypatch.setattr(websocket_auth, "get_session_local", lambda: lambda: session)
    token = build_access_token()
    monkeypatch.setattr(logging.root.manager, "disable", logging.NOTSET)
    monkeypatch.setattr(websocket_auth.logger, "disabled", False)
    monkeypatch.setattr(websocket_auth.logger, "propagate", True)
    caplog.set_level(logging.ERROR, logger=websocket_auth.__name__)

    with pytest.raises(websocket_auth._WebSocketAuthenticationTerminated) as raised:
        await websocket_auth.get_authenticated_user(websocket, token)

    assert raised.value.__cause__ is timeout
    assert session.enter_count == 1
    assert session.exit_count == 1
    assert messages == [
        {
            "type": "websocket.http.response.start",
            "status": 503,
            "headers": [
                (b"content-length", b"44"),
                (b"content-type", b"application/json"),
            ],
        },
        {
            "type": "websocket.http.response.body",
            "body": b'{"detail":"Service temporarily unavailable"}',
        },
    ]
    serialized_messages = json.dumps(messages, default=lambda value: value.decode())
    assert "secret" not in serialized_messages
    assert "database pool" not in serialized_messages
    infrastructure_records = [
        record
        for record in caplog.records
        if record.name == websocket_auth.__name__
        and "authentication infrastructure failure" in record.getMessage()
    ]
    assert len(infrastructure_records) == 1
    log_record = infrastructure_records[0]
    assert "transport=websocket" in log_record.getMessage()
    assert f"route={route_template}" in log_record.getMessage()
    rendered_log_data = f"{log_record.getMessage()} {log_record.args!r}"
    assert token not in rendered_log_data
    assert query_secret not in rendered_log_data


@pytest.mark.asyncio
async def test_operational_auth_failure_without_extension_accepts_then_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket, messages = _asgi_websocket(denial_extension=False)
    websocket.scope["extensions"] = None

    async def timeout(_operation: object) -> None:
        raise TimeoutError("database pool token=secret")

    monkeypatch.setattr(websocket_auth, "run_db_io_cancellation_safe", timeout)

    with pytest.raises(websocket_auth._WebSocketAuthenticationTerminated):
        await websocket_auth.get_authenticated_user(websocket, "signed-token")

    assert messages == [
        {"type": "websocket.accept", "subprotocol": None, "headers": []},
        {
            "type": "websocket.close",
            "code": 1011,
            "reason": "Internal server error",
        },
    ]
    serialized_messages = json.dumps(messages, default=lambda value: value.decode())
    assert "signed-token" not in serialized_messages
    assert "database pool" not in serialized_messages


@pytest.mark.asyncio
async def test_connected_auth_failure_closes_from_websocket_application_state() -> None:
    websocket = MagicMock()
    websocket.application_state = WebSocketState.CONNECTED
    websocket.scope = {"extensions": {"websocket.http.response": {}}}
    websocket.close = AsyncMock()
    websocket.accept = AsyncMock()
    websocket.send_denial_response = AsyncMock()

    await websocket_auth.send_websocket_authentication_infrastructure_failure(
        websocket,
        TimeoutError("database pool"),
    )

    websocket.close.assert_awaited_once_with(
        code=1011,
        reason="Internal server error",
    )


@pytest.mark.asyncio
async def test_denial_send_failure_is_terminal_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = MagicMock()
    websocket.scope = {"extensions": {"websocket.http.response": {}}}
    websocket.send_denial_response = AsyncMock(side_effect=ConnectionError("closed"))
    websocket.accept = AsyncMock()
    websocket.close = AsyncMock()

    async def timeout(_operation: object) -> None:
        raise TimeoutError("database pool")

    monkeypatch.setattr(websocket_auth, "run_db_io_cancellation_safe", timeout)

    with pytest.raises(websocket_auth._WebSocketAuthenticationTerminated):
        await websocket_auth.get_authenticated_user(websocket, "signed-token")

    websocket.send_denial_response.assert_awaited_once()
    websocket.accept.assert_not_awaited()
    websocket.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_extensionless_accept_failure_is_terminal_without_close_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = MagicMock()
    websocket.scope = {"extensions": {}}
    websocket.accept = AsyncMock(side_effect=ConnectionError("closed"))
    websocket.close = AsyncMock()
    websocket.send_denial_response = AsyncMock()

    async def timeout(_operation: object) -> None:
        raise TimeoutError("database pool")

    monkeypatch.setattr(websocket_auth, "run_db_io_cancellation_safe", timeout)

    with pytest.raises(websocket_auth._WebSocketAuthenticationTerminated):
        await websocket_auth.get_authenticated_user(websocket, "signed-token")

    websocket.accept.assert_awaited_once()
    websocket.close.assert_not_awaited()
    websocket.send_denial_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_extensionless_close_failure_is_terminal_without_second_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = MagicMock()
    websocket.scope = {"extensions": {}}
    websocket.accept = AsyncMock()
    websocket.close = AsyncMock(side_effect=ConnectionError("closed"))
    websocket.send_denial_response = AsyncMock()

    async def timeout(_operation: object) -> None:
        raise TimeoutError("database pool")

    monkeypatch.setattr(websocket_auth, "run_db_io_cancellation_safe", timeout)

    with pytest.raises(websocket_auth._WebSocketAuthenticationTerminated):
        await websocket_auth.get_authenticated_user(websocket, "signed-token")

    websocket.accept.assert_awaited_once()
    websocket.close.assert_awaited_once_with(code=1011, reason="Internal server error")
    websocket.send_denial_response.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "control_signal", [asyncio.CancelledError, KeyboardInterrupt, SystemExit]
)
@pytest.mark.parametrize("terminal_operation", ["denial", "accept", "close"])
async def test_terminal_send_process_control_propagates_without_marker_or_retry(
    monkeypatch: pytest.MonkeyPatch,
    control_signal: type[BaseException],
    terminal_operation: str,
) -> None:
    signal = control_signal()
    websocket = MagicMock()
    websocket.send_denial_response = AsyncMock()
    websocket.accept = AsyncMock()
    websocket.close = AsyncMock()

    if terminal_operation == "denial":
        websocket.scope = {"extensions": {"websocket.http.response": {}}}
        websocket.send_denial_response.side_effect = signal
    elif terminal_operation == "accept":
        websocket.scope = {"extensions": {}}
        websocket.accept.side_effect = signal
    else:
        websocket.scope = {"extensions": {}}
        websocket.close.side_effect = signal

    async def timeout(_operation: object) -> None:
        raise TimeoutError("database pool")

    monkeypatch.setattr(websocket_auth, "run_db_io_cancellation_safe", timeout)

    with pytest.raises(control_signal) as raised:
        await websocket_auth.get_authenticated_user(websocket, "signed-token")

    assert raised.value is signal
    assert not isinstance(
        raised.value, websocket_auth._WebSocketAuthenticationTerminated
    )
    if terminal_operation == "denial":
        websocket.send_denial_response.assert_awaited_once()
        websocket.accept.assert_not_awaited()
        websocket.close.assert_not_awaited()
    elif terminal_operation == "accept":
        websocket.send_denial_response.assert_not_awaited()
        websocket.accept.assert_awaited_once()
        websocket.close.assert_not_awaited()
    else:
        websocket.send_denial_response.assert_not_awaited()
        websocket.accept.assert_awaited_once()
        websocket.close.assert_awaited_once_with(
            code=1011, reason="Internal server error"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("control_signal", [KeyboardInterrupt, SystemExit])
async def test_process_control_signals_are_not_translated(
    monkeypatch: pytest.MonkeyPatch,
    control_signal: type[BaseException],
) -> None:
    websocket = MagicMock()
    websocket.send_denial_response = AsyncMock()

    async def interrupted(_operation: object) -> None:
        raise control_signal()

    monkeypatch.setattr(websocket_auth, "run_db_io_cancellation_safe", interrupted)

    with pytest.raises(control_signal):
        await websocket_auth.get_authenticated_user(websocket, "signed-token")

    websocket.send_denial_response.assert_not_awaited()


async def _authentication_terminated(*_args: object, **_kwargs: object) -> None:
    raise websocket_auth._WebSocketAuthenticationTerminated()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "endpoint_args"),
    [
        (websocket_api.websocket_chat_endpoint, (MagicMock(), 1, "signed-token")),
        (websocket_api.websocket_builder_chat_endpoint, (MagicMock(), "signed-token")),
        (websocket_api.websocket_build_preview_endpoint, (MagicMock(), "signed-token")),
    ],
)
async def test_main_endpoints_return_after_shared_terminal_authentication(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: object,
    endpoint_args: tuple[MagicMock, object, str],
) -> None:
    websocket = endpoint_args[0]
    websocket.close = AsyncMock()
    websocket.accept = AsyncMock()
    manager = MagicMock()
    manager.connect = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", manager)
    monkeypatch.setattr(
        websocket_api, "get_authenticated_user", _authentication_terminated
    )

    await endpoint(*endpoint_args)  # type: ignore[operator]

    websocket.close.assert_not_awaited()
    websocket.accept.assert_not_awaited()
    manager.connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_progress_returns_after_shared_terminal_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = MagicMock()
    websocket.close = AsyncMock()
    websocket.accept = AsyncMock()
    broadcaster = MagicMock()
    broadcaster.connect = AsyncMock()
    authenticated_user = AsyncMock(side_effect=_authentication_terminated)
    monkeypatch.setattr(progress_ws, "get_authenticated_user", authenticated_user)
    monkeypatch.setattr(progress_ws, "progress_broadcaster", broadcaster)

    await progress_ws.progress_websocket_endpoint(websocket, "task", "signed-token")

    authenticated_user.assert_awaited_once_with(websocket, "signed-token")
    websocket.close.assert_not_awaited()
    websocket.accept.assert_not_awaited()
    broadcaster.connect.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "endpoint_args"),
    [
        (websocket_api.websocket_chat_endpoint, (MagicMock(), 1, "invalid")),
        (websocket_api.websocket_builder_chat_endpoint, (MagicMock(), "invalid")),
        (websocket_api.websocket_build_preview_endpoint, (MagicMock(), "invalid")),
    ],
)
async def test_main_endpoints_retain_invalid_credential_close_codes(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: object,
    endpoint_args: tuple[MagicMock, object, str],
) -> None:
    websocket = endpoint_args[0]
    websocket.close = AsyncMock()
    websocket.accept = AsyncMock(side_effect=AssertionError("auth bypass"))
    manager = MagicMock()
    manager.connect = AsyncMock(side_effect=AssertionError("auth bypass"))
    session = _BoundSQLiteSession()
    monkeypatch.setattr(websocket_api, "manager", manager)
    monkeypatch.setattr(websocket_auth, "get_session_local", lambda: lambda: session)

    await endpoint(*endpoint_args[:-1], WRONG_TYPE_ACCESS_TOKEN_CASE.build_token())  # type: ignore[operator]

    websocket.close.assert_awaited_once_with(
        code=4001, reason="Authentication required"
    )
    websocket.accept.assert_not_awaited()
    manager.connect.assert_not_awaited()
    assert session.enter_count == 1
    assert session.exit_count == 1


@pytest.mark.asyncio
async def test_progress_retains_invalid_credential_close_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = MagicMock()
    websocket.close = AsyncMock()
    websocket.accept = AsyncMock(side_effect=AssertionError("auth bypass"))
    broadcaster = MagicMock()
    broadcaster.connect = AsyncMock(side_effect=AssertionError("auth bypass"))
    session = _BoundSQLiteSession()
    monkeypatch.setattr(progress_ws, "progress_broadcaster", broadcaster)
    monkeypatch.setattr(websocket_auth, "get_session_local", lambda: lambda: session)

    await progress_ws.progress_websocket_endpoint(
        websocket, "task", WRONG_TYPE_ACCESS_TOKEN_CASE.build_token()
    )

    websocket.close.assert_awaited_once_with(code=1008)
    websocket.accept.assert_not_awaited()
    broadcaster.connect.assert_not_awaited()
    assert session.enter_count == 1
    assert session.exit_count == 1
