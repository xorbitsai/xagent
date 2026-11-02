"""Header-only login admission, independent of public widget capabilities."""

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from xagent.web.api import progress_ws, speech
from xagent.web.api import websocket as websocket_api
from xagent.web.api import websocket_auth
from xagent.web.api.websocket_auth import revalidate_websocket_session
from xagent.web.websocket_credentials import (
    InvalidWebSocketCredentials,
    websocket_header_token,
)


def _scope(headers):
    return {
        "type": "websocket",
        "headers": [(k.encode(), v.encode()) for k, v in headers],
    }


@pytest.mark.parametrize(
    "path",
    [
        "/ws/chat/1",
        "/ws/build/chat",
        "/ws/build/preview",
        "/ws/progress/1",
        "/api/models/speech/asr/stream",
    ],
)
@pytest.mark.parametrize("legacy", [False, True])
def test_every_private_route_uses_the_shared_header_guard(monkeypatch, path, legacy):
    authenticate = AsyncMock(return_value=None)
    monkeypatch.setattr(websocket_api, "get_authenticated_user", authenticate)
    monkeypatch.setattr(progress_ws, "get_authenticated_user", authenticate)
    monkeypatch.setattr(speech, "get_authenticated_user", authenticate)
    app = FastAPI()
    app.include_router(websocket_api.ws_router)
    app.include_router(progress_ws.progress_ws_router)
    app.include_router(speech.speech_router)
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                path + ("?token=legacy-secret" if legacy else ""),
                subprotocols=["xagent-auth-v1", "xagent-auth-token.signed"],
            ):
                pass
    if legacy:
        authenticate.assert_not_awaited()
    else:
        assert authenticate.await_count == 1
        assert authenticate.await_args.args[1] == "signed"


@pytest.mark.parametrize(
    "headers, expected",
    [
        ([], None),
        ([("authorization", "Basic gateway")], None),
        ([("authorization", "")], None),
        ([("authorization", "Bearer abc.def.ghi")], "abc.def.ghi"),
        ([("authorization", "bearer abc.def.ghi")], "abc.def.ghi"),
        (
            [
                ("authorization", "Bearer abc.def.ghi"),
                ("sec-websocket-protocol", "unrelated"),
            ],
            "abc.def.ghi",
        ),
        (
            [
                ("sec-websocket-protocol", "xagent-auth-v1"),
                ("sec-websocket-protocol", "xagent-auth-token.abc.def.ghi"),
            ],
            "abc.def.ghi",
        ),
        (
            [
                ("authorization", "Basic gateway"),
                (
                    "sec-websocket-protocol",
                    "xagent-auth-v1, xagent-auth-token.abc.def.ghi",
                ),
            ],
            "abc.def.ghi",
        ),
        (
            [
                ("authorization", ""),
                (
                    "sec-websocket-protocol",
                    "xagent-auth-v1, xagent-auth-token.abc.def.ghi",
                ),
            ],
            "abc.def.ghi",
        ),
        (
            [
                (
                    "sec-websocket-protocol",
                    "xagent-auth-v1, xagent-auth-token.abc.def.ghi",
                )
            ],
            "abc.def.ghi",
        ),
        (
            [
                (
                    "sec-websocket-protocol",
                    "xagent-session-v1, xagent-session-token.guest",
                )
            ],
            None,
        ),
    ],
)
def test_header_credential_parsing(headers, expected):
    assert websocket_header_token(_scope(headers)) == expected


@pytest.mark.parametrize(
    "headers",
    [
        [("authorization", "Bearer")],
        [("authorization", "Bearer ")],
        [("authorization", "Bearer a b")],
        [("authorization", "Bearer one"), ("authorization", "Bearer two")],
        [("sec-websocket-protocol", "xagent-auth-v1")],
        [("sec-websocket-protocol", "xagent-auth-token.secret")],
        [("sec-websocket-protocol", "xagent-auth-v1, xagent-auth-token.")],
        [("sec-websocket-protocol", "xagent-auth-v1, xagent-auth-token.a b")],
        [
            (
                "sec-websocket-protocol",
                "xagent-auth-v1, xagent-auth-token.one, xagent-auth-token.two",
            )
        ],
        [
            (
                "sec-websocket-protocol",
                "xagent-auth-v1, xagent-auth-v1, xagent-auth-token.secret",
            )
        ],
        [("sec-websocket-protocol", "xagent-auth-v1, xagent-auth-token." + "a" * 8193)],
        [
            ("authorization", "Bearer one"),
            ("sec-websocket-protocol", "xagent-auth-v1, xagent-auth-token.two"),
        ],
    ],
)
def test_malformed_or_ambiguous_credentials_fail_without_echo(headers):
    with pytest.raises(InvalidWebSocketCredentials) as error:
        websocket_header_token(_scope(headers))
    assert str(error.value) in {
        "Invalid WebSocket credentials",
        "Ambiguous WebSocket credentials",
    }


@pytest.fixture
def guarded_client(monkeypatch):
    async def load(callback):
        return callback()

    monkeypatch.setattr(websocket_auth, "run_db_io_cancellation_safe", load)
    loaded = []

    def principal(token, **kwargs):
        loaded.append(token)
        return websocket_auth.WebSocketPrincipal(id=1, is_admin=False)

    monkeypatch.setattr(websocket_auth, "_load_websocket_principal_sync", principal)
    app = FastAPI()

    @app.websocket("/ws/private")
    @revalidate_websocket_session
    async def endpoint(websocket: WebSocket, token: str | None = None):
        user = await websocket_auth.get_authenticated_user(websocket, token)
        if user is None:
            await websocket.close(code=4001)
            return
        await websocket_auth.accept_authenticated_websocket(websocket)
        await websocket.send_json({"user_id": user.id})
        await websocket.receive_text()

    with TestClient(app) as client:
        yield client, loaded


@pytest.mark.parametrize("transport", ["authorization", "subprotocol"])
def test_guard_injects_header_credentials_without_echoing_them(
    guarded_client, transport
):
    client, loaded = guarded_client
    options = (
        {"headers": {"authorization": "Bearer signed"}}
        if transport == "authorization"
        else {"subprotocols": ["xagent-auth-v1", "xagent-auth-token.signed"]}
    )
    with client.websocket_connect("/ws/private", **options) as ws:
        assert ws.accepted_subprotocol == (
            "xagent-auth-v1" if transport == "subprotocol" else None
        )
        assert ws.receive_json() == {"user_id": 1}
        ws.send_text("done")
    assert loaded == ["signed"]


@pytest.mark.parametrize(
    "query",
    [
        "token=secret",
        "token",
        "token=",
        "access_token=secret",
        "t%6fken=secret",
        "token=a&token=b",
    ],
)
def test_guard_rejects_legacy_uri_credentials_even_with_valid_header(
    guarded_client, query
):
    client, loaded = guarded_client
    with pytest.raises(WebSocketDisconnect) as error:
        with client.websocket_connect(
            "/ws/private?" + query, headers={"authorization": "Bearer signed"}
        ):
            pass
    assert error.value.code == 4001
    assert loaded == []


def test_header_authenticated_socket_keeps_idle_session_revocation_monitor(
    guarded_client, monkeypatch
):
    client, _ = guarded_client
    calls = []

    def principal(token, **kwargs):
        calls.append((token, kwargs))
        return (
            websocket_auth.WebSocketPrincipal(id=1, is_admin=False)
            if len(calls) == 1
            else None
        )

    monkeypatch.setattr(websocket_auth, "_load_websocket_principal_sync", principal)
    monkeypatch.setattr(websocket_auth, "_SESSION_RECHECK_SECONDS", 0.01)
    with client.websocket_connect(
        "/ws/private", subprotocols=["xagent-auth-v1", "xagent-auth-token.signed"]
    ) as ws:
        assert ws.receive_json() == {"user_id": 1}
        with pytest.raises(WebSocketDisconnect) as error:
            ws.receive_json()
        assert error.value.code == 4001
    assert calls[-1] == ("signed", {"verify_exp": False})


@pytest.fixture
def public_chat_client(monkeypatch):
    """Share/widget chat routes with authorization stubbed to record the token.

    The stub denies with a sentinel reason, so every test ends in a 4003 and
    the recorded token shows exactly what reached access validation.
    """
    from fastapi import HTTPException

    from xagent.web.api import public_chat_access
    from xagent.web.api.share import share_router
    from xagent.web.api.widget import widget_router
    from xagent.web.services.share_rate_limit import reset_share_rate_limiter

    reset_share_rate_limiter()
    seen: list[str] = []

    async def authorize(*, token, **kwargs):
        seen.append(token)
        raise HTTPException(status_code=403, detail="authorized-stub")

    monkeypatch.setattr(
        public_chat_access, "_authorize_share_chat_websocket", authorize
    )
    monkeypatch.setattr(
        public_chat_access, "_authorize_public_chat_websocket", authorize
    )
    app = FastAPI()
    app.include_router(share_router)
    app.include_router(widget_router)
    with TestClient(app) as client:
        yield client, seen
    reset_share_rate_limiter()


_PUBLIC_CHAT_PATHS = ["/api/share/chat/ws/1", "/api/widget/chat/ws/1"]


@pytest.mark.parametrize("path", _PUBLIC_CHAT_PATHS)
@pytest.mark.parametrize("transport", ["subprotocol", "authorization"])
def test_public_chat_socket_reads_guest_token_from_handshake_headers(
    public_chat_client, path, transport
):
    client, seen = public_chat_client
    options = (
        {"subprotocols": ["xagent-auth-v1", "xagent-auth-token.guest.jwt"]}
        if transport == "subprotocol"
        else {"headers": {"authorization": "Bearer guest.jwt"}}
    )
    with client.websocket_connect(path, **options) as ws:
        # Only the fixed protocol is negotiated, never the credential.
        assert ws.accepted_subprotocol == (
            "xagent-auth-v1" if transport == "subprotocol" else None
        )
        with pytest.raises(WebSocketDisconnect) as error:
            ws.receive_text()
    assert (error.value.code, error.value.reason) == (4003, "authorized-stub")
    assert seen == ["guest.jwt"]


@pytest.mark.parametrize("path", _PUBLIC_CHAT_PATHS)
@pytest.mark.parametrize("with_header", [False, True])
@pytest.mark.parametrize("query", ["token=guest.jwt", "access_token=guest.jwt"])
def test_public_chat_socket_refuses_query_string_guest_tokens(
    public_chat_client, path, with_header, query
):
    client, seen = public_chat_client
    options = (
        {"subprotocols": ["xagent-auth-v1", "xagent-auth-token.guest.jwt"]}
        if with_header
        else {}
    )
    with client.websocket_connect(f"{path}?{query}", **options) as ws:
        with pytest.raises(WebSocketDisconnect) as error:
            ws.receive_text()
    assert (error.value.code, error.value.reason) == (4003, "Authentication required")
    assert seen == []


@pytest.mark.parametrize("path", _PUBLIC_CHAT_PATHS)
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"subprotocols": ["xagent-auth-v1"]},
        {"subprotocols": ["xagent-auth-v1", "xagent-auth-token.a b"]},
    ],
)
def test_public_chat_socket_refuses_missing_or_malformed_credentials(
    public_chat_client, path, options
):
    client, seen = public_chat_client
    with client.websocket_connect(path, **options) as ws:
        with pytest.raises(WebSocketDisconnect) as error:
            ws.receive_text()
    assert (error.value.code, error.value.reason) == (4003, "Authentication required")
    assert seen == []
