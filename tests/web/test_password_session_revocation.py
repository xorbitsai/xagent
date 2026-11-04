"""Real password changes must invalidate live credentials, not just refresh."""

import asyncio
import hashlib
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests.web import test_auth_api as auth_tests
from tests.web.test_auth_api import test_db as test_db
from xagent.web.api import auth, files, oidc_google, progress_ws
from xagent.web.api import websocket as websocket_api
from xagent.web.api import websocket_auth
from xagent.web.auth_dependencies import (
    get_user_from_token,
    get_user_from_websocket_token,
)
from xagent.web.models.user import User

pytestmark = pytest.mark.usefixtures("test_db")


@pytest.mark.parametrize(
    "path,missing_token_code",
    [
        ("/ws/chat/1", 4001),
        ("/ws/build/chat", 4001),
        ("/ws/build/preview", 4001),
        ("/ws/progress/task", 1008),
    ],
)
@pytest.mark.parametrize("authenticated", [False, True])
def test_routed_websocket_requires_token_and_revokes_idle_session(
    monkeypatch, path, missing_token_code, authenticated
):
    """Exercise real FastAPI handshake admission and live revocation."""
    auth_tests.setup_first_admin(password="old-password-123")
    tokens = auth._login_in_worker(
        auth.LoginRequest(username="administrator", password="old-password-123"),
        auth_tests.TestingSessionLocal,
    )
    monkeypatch.setattr(websocket_auth, "_SESSION_RECHECK_SECONDS", 0.01)
    manager = MagicMock()
    monkeypatch.setattr(websocket_api, "manager", manager)
    monkeypatch.setattr(
        websocket_api, "_private_websocket_task_access_sync", lambda **_: "authorized"
    )
    monkeypatch.setattr(websocket_api, "handle_status_request", AsyncMock())
    progress_manager = MagicMock()
    progress_manager.get_task_progress.return_value = None
    monkeypatch.setattr(progress_ws, "get_progress_manager", lambda: progress_manager)
    broadcaster = MagicMock()
    broadcaster.connect = AsyncMock()
    broadcaster.disconnect = AsyncMock()
    monkeypatch.setattr(progress_ws, "progress_broadcaster", broadcaster)
    app = FastAPI()
    app.include_router(websocket_api.ws_router)
    app.include_router(progress_ws.progress_ws_router)
    with TestClient(app) as client:
        if not authenticated:
            with pytest.raises(WebSocketDisconnect) as error:
                with client.websocket_connect(path):
                    pytest.fail("A missing token must never establish a session")
            assert error.value.code == missing_token_code
            manager.register_connection.assert_not_called()
            broadcaster.connect.assert_not_awaited()
            return
        with client.websocket_connect(
            path,
            subprotocols=[
                "xagent-auth-v1",
                f"xagent-auth-token.{tokens['access_token']}",
            ],
        ) as socket:
            assert socket.accepted_subprotocol == "xagent-auth-v1"
            _change_password_in_separate_session("change")
            # Starlette's public receive() has no timeout. Bound its underlying
            # ASGI receive so a missing monitor fails instead of hanging CI.
            message = socket.portal.call(asyncio.wait_for, socket._send_rx.receive(), 5)
            assert message == {
                "type": "websocket.close",
                "code": 4001,
                "reason": "Session expired",
            }


@pytest.mark.parametrize("operation", ["change", "reset"])
def test_password_change_revokes_access_refresh_and_file_tickets(operation):
    auth_tests.setup_first_admin(password="old-password-123")
    client = auth_tests.client
    before = client.post(
        "/api/auth/login",
        json={
            "username": "administrator",
            "password": "old-password-123",
        },
    ).json()
    headers = {"Authorization": f"Bearer {before['access_token']}"}
    with auth_tests.TestingSessionLocal() as session:
        user = session.query(User).filter_by(username="administrator").one()
        ticket = asyncio.run(files.issue_preview_stream_ticket("example-file", user))
        raw_ticket = ticket["path"].split("ticket=", 1)[1]
        assert get_user_from_token(before["access_token"], session) is not None
        if operation == "reset":
            user.password_reset_token_hash = auth.hash_auth_token("reset-secret")
            user.password_reset_expires_at = datetime.now(timezone.utc) + timedelta(
                minutes=5
            )
            session.commit()
    if operation == "change":
        result = client.post(
            "/api/auth/change-password",
            headers=headers,
            json={
                "current_password": "old-password-123",
                "new_password": "new-password-456",
            },
        )
    else:
        result = client.post(
            "/api/auth/reset-password",
            json={
                "token": "reset-secret",
                "new_password": "new-password-456",
            },
        )
    assert result.status_code == 200
    assert result.json()["success"]
    assert client.get("/api/auth/me", headers=headers).status_code == 401
    assert (
        client.post(
            "/api/auth/refresh",
            json={
                "refresh_token": before["refresh_token"],
            },
        ).status_code
        == 401
    )
    with auth_tests.TestingSessionLocal() as session:
        assert get_user_from_token(before["access_token"], session) is None
        assert get_user_from_websocket_token(before["access_token"], session) is None
        with pytest.raises(HTTPException) as error:
            files._user_from_stream_ticket(session, raw_ticket, "example-file")
        assert error.value.status_code == 401
    after = client.post(
        "/api/auth/login",
        json={
            "username": "administrator",
            "password": "new-password-456",
        },
    ).json()
    assert (
        client.get(
            "/api/auth/me",
            headers={
                "Authorization": f"Bearer {after['access_token']}",
            },
        ).status_code
        == 200
    )
    refreshed = client.post(
        "/api/auth/refresh",
        json={
            "refresh_token": after["refresh_token"],
        },
    )
    assert refreshed.status_code == 200
    assert (
        client.get(
            "/api/auth/me",
            headers={
                "Authorization": f"Bearer {refreshed.json()['access_token']}",
            },
        ).status_code
        == 200
    )


def test_reset_token_cannot_be_reused():
    auth_tests.setup_first_admin(password="old-password-123")
    with auth_tests.TestingSessionLocal() as session:
        user = session.query(User).first()
        user.password_reset_token_hash = auth.hash_auth_token("one-use")
        user.password_reset_expires_at = datetime.now(timezone.utc) + timedelta(
            minutes=5
        )
        session.commit()
    request = {"token": "one-use", "new_password": "new-password-456"}
    assert auth_tests.client.post("/api/auth/reset-password", json=request).json()[
        "success"
    ]
    assert not auth_tests.client.post("/api/auth/reset-password", json=request).json()[
        "success"
    ]


@pytest.mark.parametrize(
    "conflict", ["password-changed", "reset-reissued", "reset-expired"]
)
def test_password_update_loses_conditional_write_without_rotating_epoch(
    monkeypatch, conflict
):
    auth_tests.setup_first_admin(password="old-password-123")
    token = auth_tests.login_and_get_token("administrator", "old-password-123")
    with auth_tests.TestingSessionLocal() as session:
        user = session.query(User).one()
        user.password_reset_token_hash = auth.hash_auth_token("race-reset")
        user.password_reset_expires_at = datetime.now(timezone.utc) + timedelta(
            minutes=5
        )
        session.commit()
    original_hash = auth.hash_password_async
    winner_hash = auth.hash_password("winning-password-789")

    async def hash_after_conflict(password):
        result = await original_hash(password)
        with auth_tests.TestingSessionLocal() as session:
            user = session.query(User).one()
            if conflict == "password-changed":
                user.password_hash = winner_hash
            elif conflict == "reset-reissued":
                user.password_reset_token_hash = auth.hash_auth_token("replacement")
            else:
                user.password_reset_expires_at = datetime.now(timezone.utc) - timedelta(
                    seconds=1
                )
            session.commit()
        return result

    monkeypatch.setattr(auth, "hash_password_async", hash_after_conflict)
    if conflict == "password-changed":
        response = auth_tests.client.post(
            "/api/auth/change-password",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "current_password": "old-password-123",
                "new_password": "new-password-456",
            },
        )
    else:
        response = auth_tests.client.post(
            "/api/auth/reset-password",
            json={"token": "race-reset", "new_password": "new-password-456"},
        )
    assert response.status_code == 200
    assert not response.json()["success"]
    with auth_tests.TestingSessionLocal() as session:
        user = session.query(User).one()
        assert user.session_epoch is None
        assert user.refresh_token is not None
        assert auth.verify_password(
            "winning-password-789"
            if conflict == "password-changed"
            else "old-password-123",
            user.password_hash,
        )


def test_refresh_checks_epoch_even_if_refresh_token_is_unchanged(monkeypatch):
    auth_tests.setup_first_admin(password="old-password-123")
    tokens = auth._login_in_worker(
        auth.LoginRequest(username="administrator", password="old-password-123"),
        auth_tests.TestingSessionLocal,
    )
    original = auth._prepare_refresh_response

    def prepare_then_rotate(*args):
        response = original(*args)
        with auth_tests.TestingSessionLocal() as session:
            session.query(User).update({User.session_epoch: "new-generation"})
            session.commit()
        return response

    monkeypatch.setattr(auth, "_prepare_refresh_response", prepare_then_rotate)
    with pytest.raises(HTTPException) as error:
        auth._refresh_in_worker(
            auth.RefreshTokenRequest(refresh_token=tokens["refresh_token"]),
            tokens["user_id"],
            auth_tests.TestingSessionLocal,
        )
    assert error.value.status_code == 401
    with auth_tests.TestingSessionLocal() as session:
        user = session.query(User).one()
        assert user.session_epoch == "new-generation"
        assert user.refresh_token == tokens["refresh_token"]


def test_session_epoch_generator_fits_the_database_column():
    from xagent.web.auth_sessions import new_session_epoch

    first, second = new_session_epoch(), new_session_epoch()
    assert len(first) == User.__table__.c.session_epoch.type.length == 32
    assert first.isascii()
    assert first != second


@pytest.mark.parametrize("late_publication", [False, True])
def test_admin_cli_reset_revokes_all_credentials(monkeypatch, late_publication):
    from xagent.web import reset_admin_password as admin_reset

    auth_tests.setup_first_admin(password="old-password-123")
    tokens = {}

    def publish_credentials():
        tokens.update(
            auth._login_in_worker(
                auth.LoginRequest(
                    username="administrator", password="old-password-123"
                ),
                auth_tests.TestingSessionLocal,
            )
        )
        with auth_tests.TestingSessionLocal() as session:
            user = session.query(User).one()
            tokens["ticket"] = files.create_access_token(
                {
                    **auth.user_session_claims(user),
                    "type": files.FILE_STREAM_TICKET_TYPE,
                    "file_id": "example-file",
                }
            )
            user.password_reset_token_hash = auth.hash_auth_token("old-reset")
            user.password_reset_expires_at = datetime.now(timezone.utc) + timedelta(
                minutes=5
            )
            session.commit()

    if late_publication:
        original_hash = admin_reset.hash_password

        def publish_after_admin_read(password):
            publish_credentials()
            return original_hash(password)

        monkeypatch.setattr(admin_reset, "hash_password", publish_after_admin_read)
    else:
        publish_credentials()

    admin_reset.reset_admin_password("administrator", "new-password-456")
    with auth_tests.TestingSessionLocal() as session:
        user = session.query(User).one()
        assert auth.verify_password("new-password-456", user.password_hash)
        assert user.session_epoch is not None
        assert user.refresh_token is None
        assert user.refresh_token_expires_at is None
        assert user.password_reset_token_hash is None
        assert user.password_reset_expires_at is None
        assert get_user_from_token(tokens["access_token"], session) is None
        assert get_user_from_websocket_token(tokens["access_token"], session) is None
        with pytest.raises(HTTPException) as rejected:
            files._user_from_stream_ticket(session, tokens["ticket"], "example-file")
        assert rejected.value.status_code == 401
    assert (
        auth_tests.client.post(
            "/api/auth/refresh", json={"refresh_token": tokens["refresh_token"]}
        ).status_code
        == 401
    )
    assert not auth_tests.client.post(
        "/api/auth/reset-password",
        json={"token": "old-reset", "new_password": "another-password-789"},
    ).json()["success"]


def _change_password_in_separate_session(operation):
    with auth_tests.TestingSessionLocal() as session:
        user = session.query(User).one()
        if operation == "reset":
            result = asyncio.run(
                auth.reset_password(
                    auth.ResetPasswordRequest(
                        token="race-reset", new_password="new-password-456"
                    ),
                    session,
                )
            )
        else:
            result = asyncio.run(
                auth.change_password(
                    auth.ChangePasswordRequest(
                        current_password="old-password-123",
                        new_password="new-password-456",
                    ),
                    session,
                    user,
                )
            )
        assert result.success


@pytest.mark.parametrize("operation", ["change", "reset"])
@pytest.mark.parametrize("issuer", ["login", "refresh", "oidc"])
def test_password_change_wins_against_in_flight_token_publication(
    monkeypatch, operation, issuer
):
    """Interleave real handlers at the old credential publication race window."""
    auth_tests.setup_first_admin(password="old-password-123")
    before = auth._login_in_worker(
        auth.LoginRequest(username="administrator", password="old-password-123"),
        auth_tests.TestingSessionLocal,
    )
    with auth_tests.TestingSessionLocal() as session:
        user = session.query(User).one()
        user_id = user.id
        user.password_reset_token_hash = auth.hash_auth_token("race-reset")
        user.password_reset_expires_at = datetime.now(timezone.utc) + timedelta(
            minutes=5
        )
        session.commit()

    captured = {}
    if issuer in {"login", "refresh"}:
        name = (
            "_prepare_login_response"
            if issuer == "login"
            else "_prepare_refresh_response"
        )
        original = getattr(auth, name)

        def prepare_then_revoke(*args):
            response = original(*args)
            captured["refresh"] = (
                response["refresh_token"]
                if issuer == "login"
                else response.refresh_token
            )
            _change_password_in_separate_session(operation)
            return response

        monkeypatch.setattr(auth, name, prepare_then_revoke)
    else:
        original = oidc_google.create_refresh_token

        def create_then_revoke(*args, **kwargs):
            token = original(*args, **kwargs)
            captured["refresh"] = token
            _change_password_in_separate_session(operation)
            return token

        monkeypatch.setattr(oidc_google, "create_refresh_token", create_then_revoke)

    with pytest.raises(HTTPException) as error:
        if issuer == "login":
            auth._login_in_worker(
                auth.LoginRequest(
                    username="administrator", password="old-password-123"
                ),
                auth_tests.TestingSessionLocal,
            )
        elif issuer == "refresh":
            auth._refresh_in_worker(
                auth.RefreshTokenRequest(refresh_token=before["refresh_token"]),
                user_id,
                auth_tests.TestingSessionLocal,
            )
        else:
            with auth_tests.TestingSessionLocal() as session:
                oidc_google._issue_auth_payload(session, session.get(User, user_id))
    assert error.value.status_code == 401
    with auth_tests.TestingSessionLocal() as session:
        user = session.get(User, user_id)
        assert auth.verify_password("new-password-456", user.password_hash)
        assert user.session_epoch is not None
        assert user.refresh_token is None
        assert get_user_from_token(before["access_token"], session) is None
    for token in (before["refresh_token"], captured["refresh"]):
        with pytest.raises(HTTPException) as rejected:
            auth._refresh_in_worker(
                auth.RefreshTokenRequest(refresh_token=token),
                user_id,
                auth_tests.TestingSessionLocal,
            )
        assert rejected.value.status_code == 401


def test_password_change_revokes_pending_oidc_exchange_code():
    auth_tests.setup_first_admin(password="old-password-123")
    with auth_tests.TestingSessionLocal() as session:
        user = session.query(User).one()
        code = oidc_google._create_exchange_code(user.id, user.session_epoch)
    _change_password_in_separate_session("change")
    with auth_tests.TestingSessionLocal() as session:
        with pytest.raises(HTTPException) as error:
            oidc_google.google_oidc_exchange(
                oidc_google.OidcExchangeRequest(code=code), session
            )
        assert error.value.status_code == 400
    with auth_tests.TestingSessionLocal() as session:
        user = session.query(User).one()
        current_code = oidc_google._create_exchange_code(user.id, user.session_epoch)
        response = oidc_google.google_oidc_exchange(
            oidc_google.OidcExchangeRequest(code=current_code), session
        )
        assert get_user_from_token(response["access_token"], session) is not None


def test_login_preserves_atomic_legacy_password_upgrade():
    auth_tests.setup_first_admin(password="old-password-123")
    with auth_tests.TestingSessionLocal() as session:
        user = session.query(User).one()
        user.password_hash = hashlib.sha256(b"old-password-123").hexdigest()
        session.commit()
    response = auth._login_in_worker(
        auth.LoginRequest(username="administrator", password="old-password-123"),
        auth_tests.TestingSessionLocal,
    )
    with auth_tests.TestingSessionLocal() as session:
        user = session.query(User).one()
        assert user.password_hash.startswith("$2")
        assert auth.verify_password("old-password-123", user.password_hash)
        assert user.refresh_token == response["refresh_token"]


def test_login_checks_session_version_even_when_password_hash_is_unchanged(monkeypatch):
    auth_tests.setup_first_admin(password="old-password-123")
    original = auth._prepare_login_response

    def prepare_then_rotate(*args):
        response = original(*args)
        with auth_tests.TestingSessionLocal() as session:
            session.query(User).update(
                {User.session_epoch: "new-generation", User.refresh_token: None}
            )
            session.commit()
        return response

    monkeypatch.setattr(auth, "_prepare_login_response", prepare_then_rotate)
    with pytest.raises(HTTPException) as error:
        auth._login_in_worker(
            auth.LoginRequest(username="administrator", password="old-password-123"),
            auth_tests.TestingSessionLocal,
        )
    assert error.value.status_code == 401
    with auth_tests.TestingSessionLocal() as session:
        user = session.query(User).one()
        assert user.session_epoch == "new-generation"
        assert user.refresh_token is None


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["change", "reset"])
@pytest.mark.parametrize("endpoint_name", ["chat", "builder", "preview", "progress"])
@pytest.mark.parametrize("mode", ["idle", "command"])
async def test_open_websocket_is_revoked_and_owned_work_is_drained(
    monkeypatch, operation, endpoint_name, mode
):
    auth_tests.setup_first_admin(password="old-password-123")
    tokens = auth._login_in_worker(
        auth.LoginRequest(username="administrator", password="old-password-123"),
        auth_tests.TestingSessionLocal,
    )
    with auth_tests.TestingSessionLocal() as session:
        user = session.query(User).one()
        user.password_reset_token_hash = auth.hash_auth_token("race-reset")
        user.password_reset_expires_at = datetime.now(timezone.utc) + timedelta(
            minutes=5
        )
        session.commit()
    incoming = asyncio.Queue()
    await incoming.put({"type": "websocket.connect"})
    accepted = asyncio.Event()
    outgoing = []

    async def send(message):
        outgoing.append(message)
        if message["type"] == "websocket.accept":
            accepted.set()

    websocket = WebSocket(
        {"type": "websocket", "path": "/ws/test", "headers": [], "query_string": b""},
        incoming.get,
        send,
    )
    monkeypatch.setattr(
        websocket_auth, "_SESSION_RECHECK_SECONDS", 0.01 if mode == "idle" else 3600
    )
    manager = MagicMock()
    monkeypatch.setattr(websocket_api, "manager", manager)
    monkeypatch.setattr(
        websocket_api, "_private_websocket_task_access_sync", lambda **_: "authorized"
    )
    monkeypatch.setattr(websocket_api, "handle_status_request", AsyncMock())
    chat_command = AsyncMock()
    preview_command = AsyncMock()
    monkeypatch.setattr(websocket_api, "handle_chat_message", chat_command)
    monkeypatch.setattr(
        websocket_api, "handle_build_preview_execution", preview_command
    )
    progress_manager = MagicMock()
    progress_manager.get_task_progress.return_value = None
    monkeypatch.setattr(progress_ws, "get_progress_manager", lambda: progress_manager)
    broadcaster = MagicMock()
    broadcaster.connect = AsyncMock()
    broadcaster.disconnect = AsyncMock()
    monkeypatch.setattr(progress_ws, "progress_broadcaster", broadcaster)
    streaming = asyncio.Event()
    drained = asyncio.Event()

    async def stream_until_cancelled(*_):
        streaming.set()
        try:
            await asyncio.Event().wait()
        finally:
            drained.set()

    monkeypatch.setattr(websocket_api, "handle_builder_chat", stream_until_cancelled)
    endpoints = {
        "chat": lambda: websocket_api.websocket_chat_endpoint(
            websocket, 1, tokens["access_token"]
        ),
        "builder": lambda: websocket_api.websocket_builder_chat_endpoint(
            websocket, tokens["access_token"]
        ),
        "preview": lambda: websocket_api.websocket_build_preview_endpoint(
            websocket, tokens["access_token"]
        ),
        "progress": lambda: progress_ws.progress_websocket_endpoint(
            websocket, "task", tokens["access_token"]
        ),
    }
    handler = asyncio.create_task(endpoints[endpoint_name]())
    try:
        await asyncio.wait_for(accepted.wait(), timeout=5)
        if endpoint_name == "builder" and mode == "idle":
            await incoming.put(
                {"type": "websocket.receive", "text": '{"message":"hello"}'}
            )
            await asyncio.wait_for(streaming.wait(), timeout=5)
        await asyncio.to_thread(_change_password_in_separate_session, operation)
        if mode == "command":
            await incoming.put(
                {
                    "type": "websocket.receive",
                    "text": '{"type":"preview"}'
                    if endpoint_name == "preview"
                    else '{"type":"chat","message":"hello"}',
                }
            )
        await asyncio.wait_for(handler, timeout=5)
    finally:
        if not handler.done():
            handler.cancel()
        await asyncio.gather(handler, return_exceptions=True)
    assert outgoing[-1] == {
        "type": "websocket.close",
        "code": 4001,
        "reason": "Session expired",
    }
    if endpoint_name == "progress":
        broadcaster.disconnect.assert_awaited_once()
    elif endpoint_name == "builder":
        assert drained.is_set() if mode == "idle" else not streaming.is_set()
        assert websocket.state.chat_task is None
    else:
        manager.disconnect.assert_called_once_with(websocket)
    chat_command.assert_not_awaited()
    preview_command.assert_not_awaited()
    assert not any(
        task.get_name() == "websocket-session-monitor" for task in asyncio.all_tasks()
    )
