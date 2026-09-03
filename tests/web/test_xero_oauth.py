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
            app_id="xero",
            name="Xero",
            description="Xero connector",
            transport="oauth",
            provider_name="xero",
            category="Accounting",
            oauth_scopes=["offline_access", "accounting.contacts"],
            # True here regardless of the real catalog row's current value
            # (which ships hidden as a release gate, see
            # builtin_mcp_registry.py's own comment on it): a hidden app's
            # OAuth callback 404s unconditionally (auth.py's
            # _reject_hidden_catalog_app, enforced server-side on all three
            # connect paths) -- this fixture exists to isolate and verify
            # the Basic-Auth token-exchange mechanics below, which is a
            # separate concern from that visibility gate and shouldn't be
            # coupled to whichever value the real row happens to have.
            is_visible_in_connector=True,
            launch_config={
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.xero"],
                "env_mapping": {"XERO_ACCESS_TOKEN": "access_token"},
            },
        )
    )
    db.commit()
    db.refresh(user)

    yield db, user
    db.close()
    engine.dispose()


def _xero_provider() -> SimpleNamespace:
    return SimpleNamespace(
        provider_name="xero",
        client_id=encrypt_value("xero-client-id"),
        client_secret=encrypt_value("xero-client-secret"),
        token_url="https://identity.xero.com/connect/token",
        redirect_uri="https://app.example.com/api/auth/xero/callback",
        userinfo_url="https://identity.xero.com/connect/userinfo",
        user_id_path="sub",
        email_path="email",
        default_scopes=["openid", "profile", "email"],
    )


def test_xero_callback_exchanges_code_with_http_basic_auth(db_session, monkeypatch):
    """Xero's token endpoint (per its own official SDK, XeroAPI/xero-node's
    tokenRequest()) authenticates via HTTP Basic Auth, not a client_secret in
    the POST body -- assert the exchange sends Basic auth and omits the
    secret (and client_id) from the form body, not just that the call
    "worked"."""
    db, user = db_session
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "xero",
            "app_id": "xero",
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "xero-code", "state": state})

    post = Mock(
        return_value=MockResponse(
            {
                "access_token": "xero-token",
                "refresh_token": "xero-refresh",
                "token_type": "bearer",
                "expires_in": 1800,
                "scope": "openid profile email offline_access accounting.contacts",
            }
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", post)
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(
            return_value=MockResponse({"sub": "xero-user-1", "email": "alice@xero.com"})
        ),
    )

    response = generic_oauth_callback("xero", request, db, _xero_provider())

    assert response.status_code == 200
    assert post.call_args.kwargs["auth"] == ("xero-client-id", "xero-client-secret")
    assert "client_id" not in post.call_args.kwargs["data"]
    assert "client_secret" not in post.call_args.kwargs["data"]
    assert post.call_args.kwargs["data"]["grant_type"] == "authorization_code"

    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "xero")
        .one()
    )
    assert oauth_account.access_token == "xero-token"
    assert oauth_account.refresh_token == "xero-refresh"
    assert oauth_account.provider_user_id == "xero-user-1"
    assert oauth_account.email == "alice@xero.com"

    server = db.query(MCPServer).filter(MCPServer.name == "Xero").one()
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


@pytest.mark.asyncio
async def test_xero_expired_token_refresh_uses_http_basic_auth(db_session, monkeypatch):
    """Xero access tokens last only 30 minutes, so a missed Basic-Auth branch
    on the refresh leg would surface as a broken connection within the hour
    even if the initial code exchange happened to work."""
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="xero",
            name="Xero",
            client_id=encrypt_value("xero-client-id"),
            client_secret=encrypt_value("xero-client-secret"),
            auth_url="https://login.xero.com/identity/connect/authorize",
            token_url="https://identity.xero.com/connect/token",
            redirect_uri="https://app.example.com/api/auth/xero/callback",
            userinfo_url="https://identity.xero.com/connect/userinfo",
            user_id_path="sub",
            email_path="email",
            default_scopes=["openid", "profile", "email"],
        )
    )
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="xero",
        access_token="old-token",
        refresh_token="old-refresh",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_user_id="xero-user-1",
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
                    "expires_in": 1800,
                },
                status_code=200,
            )

    monkeypatch.setattr(tool_config.httpx, "AsyncClient", FakeAsyncClient)

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "xero")
        is True
    )

    assert oauth_account.access_token == "new-token"
    assert oauth_account.refresh_token == "new-refresh"
    assert len(captured_requests) == 1
    url, kwargs = captured_requests[0]
    assert url == "https://identity.xero.com/connect/token"
    assert kwargs["data"] == {
        "grant_type": "refresh_token",
        "refresh_token": "old-refresh",
    }
    auth = kwargs["auth"]
    assert isinstance(auth, tool_config.httpx.BasicAuth)
    expected_credential = base64.b64encode(b"xero-client-id:xero-client-secret").decode(
        "ascii"
    )
    assert auth._auth_header == f"Basic {expected_credential}"
