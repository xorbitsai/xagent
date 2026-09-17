from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.utils.encryption import encrypt_value
from xagent.web.api import auth as auth_api
from xagent.web.api.auth import create_access_token, generic_oauth_callback
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.tools import config as tool_config


class MockResponse:
    def __init__(self, json_data=None, status_code: int = 200, text: str = ""):
        self._json_data = json_data or {}
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._json_data


@pytest.fixture()
def db_session(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = SessionLocal()

    user = User(username="alice", password_hash="x", is_admin=False)
    db.add(user)
    db.add(
        PublicMCPApp(
            app_id="zoom",
            name="Zoom",
            description="Zoom connector",
            transport="oauth",
            provider_name="zoom",
            category="Scheduling",
            oauth_scopes=["meeting:read:meeting", "user:read:user"],
            is_visible_in_connector=True,
            launch_config={
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.zoom"],
                "env_mapping": {"ZOOM_ACCESS_TOKEN": "access_token"},
            },
        )
    )
    db.commit()
    db.refresh(user)

    yield db, user
    db.close()
    engine.dispose()


def _zoom_provider() -> SimpleNamespace:
    return SimpleNamespace(
        provider_name="zoom",
        client_id=encrypt_value("zoom-client-id"),
        client_secret=encrypt_value("zoom-client-secret"),
        token_url="https://zoom.us/oauth/token",
        redirect_uri="https://app.example.com/api/auth/zoom/callback",
        userinfo_url="https://api.zoom.us/v2/users/me",
        user_id_path="id",
        email_path="email",
        default_scopes=["meeting:read:meeting", "user:read:user"],
    )


def test_zoom_callback_exchanges_code_with_http_basic_auth(db_session, monkeypatch):
    """Zoom's token endpoint rejects client_secret in the POST body — it requires
    HTTP Basic Auth. Assert the exchange sends Basic auth and omits the secret
    (and client_id) from the form body, not just that the call "worked"."""
    db, user = db_session
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "zoom",
            "app_id": "zoom",
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "zoom-code", "state": state})

    post = Mock(
        return_value=MockResponse(
            {
                "access_token": "zoom-token",
                "refresh_token": "zoom-refresh",
                "token_type": "bearer",
                "expires_in": 3600,
                "scope": "meeting:read:meeting user:read:user",
            }
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", post)
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(
            return_value=MockResponse({"id": "zoom-user-1", "email": "alice@zoom.us"})
        ),
    )

    response = generic_oauth_callback("zoom", request, db, _zoom_provider())

    assert response.status_code == 200
    assert post.call_args.kwargs["auth"] == ("zoom-client-id", "zoom-client-secret")
    assert "client_id" not in post.call_args.kwargs["data"]
    assert "client_secret" not in post.call_args.kwargs["data"]
    assert post.call_args.kwargs["data"]["grant_type"] == "authorization_code"

    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "zoom")
        .one()
    )
    assert oauth_account.access_token == "zoom-token"
    assert oauth_account.refresh_token == "zoom-refresh"
    assert oauth_account.provider_user_id == "zoom-user-1"
    assert oauth_account.email == "alice@zoom.us"

    server = db.query(MCPServer).filter(MCPServer.name == "Zoom").one()
    assert server.transport == "oauth"
    user_mcp = (
        db.query(UserMCPServer)
        .filter(
            UserMCPServer.user_id == user.id,
            UserMCPServer.mcpserver_id == server.id,
        )
        .one()
    )
    assert user_mcp.is_active is True


def test_non_zoom_callback_still_sends_client_secret_in_body(db_session, monkeypatch):
    """Guard the branch condition: a non-zoom provider must be unaffected by the
    Zoom-specific Basic-Auth carve-out."""
    db, user = db_session
    db.add(
        PublicMCPApp(
            app_id="gmail",
            name="Gmail",
            transport="oauth",
            provider_name="google",
        )
    )
    db.commit()

    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "google",
            "app_id": "gmail",
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "code", "state": state})
    post = Mock(
        return_value=MockResponse(
            {
                "access_token": "tok",
                "token_type": "Bearer",
                "scope": "",
                "expires_in": 3600,
            }
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", post)
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(return_value=MockResponse({"sub": "u1", "email": "alice@gmail.com"})),
    )

    google_provider = SimpleNamespace(
        provider_name="google",
        client_id=encrypt_value("google-client-id"),
        client_secret=encrypt_value("google-client-secret"),
        token_url="https://oauth2.googleapis.com/token",
        redirect_uri="https://app.example.com/api/auth/google/callback",
        userinfo_url="https://openidconnect.googleapis.com/v1/userinfo",
        user_id_path="sub",
        email_path="email",
        default_scopes=["https://www.googleapis.com/auth/gmail.modify"],
    )

    response = generic_oauth_callback("google", request, db, google_provider)

    assert response.status_code == 200
    assert post.call_args.kwargs.get("auth") is None
    assert post.call_args.kwargs["data"]["client_id"] == "google-client-id"
    assert post.call_args.kwargs["data"]["client_secret"] == "google-client-secret"


@pytest.mark.asyncio
async def test_zoom_expired_token_refresh_uses_http_basic_auth(db_session, monkeypatch):
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="zoom",
            name="Zoom",
            client_id=encrypt_value("zoom-client-id"),
            client_secret=encrypt_value("zoom-client-secret"),
            auth_url="https://zoom.us/oauth/authorize",
            token_url="https://zoom.us/oauth/token",
            redirect_uri="https://app.example.com/api/auth/zoom/callback",
            userinfo_url="https://api.zoom.us/v2/users/me",
            user_id_path="id",
            email_path="email",
            default_scopes=["meeting:read:meeting"],
        )
    )
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="zoom",
        access_token="old-token",
        refresh_token="old-refresh",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_user_id="zoom-user-1",
    )
    db.add(oauth_account)
    db.commit()

    captured_requests = []

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            captured_requests.append((url, kwargs))
            return MockResponse(
                {
                    "access_token": "new-token",
                    "refresh_token": "new-refresh",
                    "token_type": "bearer",
                    "expires_in": 3600,
                },
                status_code=200,
            )

    monkeypatch.setattr(tool_config.httpx, "AsyncClient", FakeAsyncClient)

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "zoom")
        is True
    )

    assert oauth_account.access_token == "new-token"
    assert oauth_account.refresh_token == "new-refresh"
    assert len(captured_requests) == 1
    url, kwargs = captured_requests[0]
    assert url == "https://zoom.us/oauth/token"
    assert kwargs["data"] == {
        "grant_type": "refresh_token",
        "refresh_token": "old-refresh",
    }
    auth = kwargs["auth"]
    assert isinstance(auth, tool_config.httpx.BasicAuth)
    expected_credential = base64.b64encode(b"zoom-client-id:zoom-client-secret").decode(
        "ascii"
    )
    assert auth._auth_header == f"Basic {expected_credential}"


@pytest.mark.asyncio
async def test_zoom_expired_token_refresh_uses_http_basic_auth_with_capitalized_name(
    db_session, monkeypatch
):
    """An admin-created provider row named "Zoom" (capitalized, e.g. via
    POST /admin/mcp/providers) must take the same Basic-Auth branch as the
    lowercase "zoom" — the branch check normalizes case exactly once rather
    than comparing provider_name verbatim."""
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="Zoom",
            name="Zoom",
            client_id=encrypt_value("zoom-client-id"),
            client_secret=encrypt_value("zoom-client-secret"),
            auth_url="https://zoom.us/oauth/authorize",
            token_url="https://zoom.us/oauth/token",
            redirect_uri="https://app.example.com/api/auth/zoom/callback",
            userinfo_url="https://api.zoom.us/v2/users/me",
            user_id_path="id",
            email_path="email",
            default_scopes=["meeting:read:meeting"],
        )
    )
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="Zoom",
        access_token="old-token",
        refresh_token="old-refresh",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_user_id="zoom-user-1",
    )
    db.add(oauth_account)
    db.commit()

    captured_requests = []

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            captured_requests.append((url, kwargs))
            return MockResponse(
                {
                    "access_token": "new-token",
                    "token_type": "bearer",
                    "expires_in": 3600,
                },
                status_code=200,
            )

    monkeypatch.setattr(tool_config.httpx, "AsyncClient", FakeAsyncClient)

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "Zoom")
        is True
    )

    assert len(captured_requests) == 1
    _, kwargs = captured_requests[0]
    assert isinstance(kwargs["auth"], tool_config.httpx.BasicAuth)
    assert "client_id" not in kwargs["data"]
    assert "client_secret" not in kwargs["data"]


@pytest.mark.asyncio
async def test_generic_provider_refresh_sends_client_id_and_secret_in_body(
    db_session, monkeypatch
):
    """The non-special-cased branch of refresh_oauth_token_if_needed (no
    meta/zoom/linkedin carve-out) must send client_id/client_secret in the
    POST body on the success path — the existing coverage for this branch
    only asserted that secrets aren't logged on *failure*."""
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="google",
            name="Google",
            client_id=encrypt_value("google-client-id"),
            client_secret=encrypt_value("google-client-secret"),
            auth_url="https://accounts.google.com/o/oauth2/auth",
            token_url="https://oauth2.googleapis.com/token",
            redirect_uri="https://app.example.com/api/auth/google/callback",
            userinfo_url="https://openidconnect.googleapis.com/v1/userinfo",
            user_id_path="sub",
            email_path="email",
            default_scopes=["https://www.googleapis.com/auth/gmail.modify"],
        )
    )
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="google",
        access_token="old-token",
        refresh_token="old-refresh",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_user_id="google-user-1",
    )
    db.add(oauth_account)
    db.commit()

    captured_requests = []

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            captured_requests.append((url, kwargs))
            return MockResponse(
                {
                    "access_token": "new-token",
                    "refresh_token": "new-refresh",
                    "token_type": "bearer",
                    "expires_in": 3600,
                },
                status_code=200,
            )

    monkeypatch.setattr(tool_config.httpx, "AsyncClient", FakeAsyncClient)

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "google")
        is True
    )

    assert oauth_account.access_token == "new-token"
    assert oauth_account.refresh_token == "new-refresh"
    assert len(captured_requests) == 1
    url, kwargs = captured_requests[0]
    assert url == "https://oauth2.googleapis.com/token"
    assert "auth" not in kwargs
    assert kwargs["data"] == {
        "grant_type": "refresh_token",
        "refresh_token": "old-refresh",
        "client_id": "google-client-id",
        "client_secret": "google-client-secret",
    }
