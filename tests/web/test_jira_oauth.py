from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import parse_qs, urlparse

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
            app_id="jira",
            name="Jira",
            description="Jira connector",
            transport="oauth",
            provider_name="jira",
            category="Productivity",
            oauth_scopes=["read:jira-work", "write:jira-work", "offline_access"],
            is_visible_in_connector=True,
            launch_config={
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.jira"],
                "env_mapping": {"JIRA_ACCESS_TOKEN": "access_token"},
            },
        )
    )
    db.commit()
    db.refresh(user)

    yield db, user
    db.close()
    engine.dispose()


def _jira_provider() -> SimpleNamespace:
    return SimpleNamespace(
        provider_name="jira",
        client_id=encrypt_value("jira-client-id"),
        client_secret=encrypt_value("jira-client-secret"),
        auth_url="https://auth.atlassian.com/authorize",
        token_url="https://auth.atlassian.com/oauth/token",
        redirect_uri="https://app.example.com/api/auth/jira/callback",
        userinfo_url="https://api.atlassian.com/me",
        user_id_path="account_id",
        email_path="email",
        default_scopes=["read:jira-work", "write:jira-work", "offline_access"],
    )


def test_login_sends_audience_and_prompt_consent(db_session):
    db, user = db_session
    token = create_access_token(
        data={"sub": user.username, "type": "access"},
        expires_delta=timedelta(minutes=5),
    )

    response = auth_api.generic_oauth_login(
        "jira", token=token, app_id="jira", db=db, db_provider=_jira_provider()
    )

    assert response.status_code == 307
    query = parse_qs(urlparse(response.headers["location"]).query)
    assert query["audience"] == ["api.atlassian.com"]
    assert query["prompt"] == ["consent"]


def test_jira_callback_exchanges_code_with_json_body(db_session, monkeypatch):
    """Atlassian's token endpoint requires a JSON body -- a form-urlencoded
    POST (every other provider here) gets rejected."""
    db, user = db_session
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "jira",
            "app_id": "jira",
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "jira-code", "state": state})

    post = Mock(
        return_value=MockResponse(
            {
                "access_token": "jira-token",
                "refresh_token": "jira-refresh",
                "token_type": "bearer",
                "expires_in": 3600,
                "scope": "read:jira-work write:jira-work offline_access",
            }
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", post)
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(
            return_value=MockResponse(
                {"account_id": "jira-user-1", "email": "alice@example.com"}
            )
        ),
    )

    response = generic_oauth_callback("jira", request, db, _jira_provider())

    assert response.status_code == 200
    assert post.call_args.kwargs["headers"]["Content-Type"] == "application/json"
    assert post.call_args.kwargs["json"] == {
        "grant_type": "authorization_code",
        "code": "jira-code",
        "redirect_uri": "https://app.example.com/api/auth/jira/callback",
        "client_id": "jira-client-id",
        "client_secret": "jira-client-secret",
    }
    assert "data" not in post.call_args.kwargs

    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "jira")
        .one()
    )
    assert oauth_account.access_token == "jira-token"
    assert oauth_account.refresh_token == "jira-refresh"
    assert oauth_account.provider_user_id == "jira-user-1"
    assert oauth_account.email == "alice@example.com"

    server = db.query(MCPServer).filter(MCPServer.name == "Jira").one()
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


def test_non_jira_callback_still_sends_form_urlencoded_body(db_session, monkeypatch):
    """Guard the branch condition: a non-jira provider must be unaffected by
    the Atlassian-specific JSON-body carve-out."""
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
    assert post.call_args.kwargs["headers"]["Content-Type"] == (
        "application/x-www-form-urlencoded"
    )
    assert post.call_args.kwargs["data"]["client_id"] == "google-client-id"
    assert "json" not in post.call_args.kwargs


@pytest.mark.asyncio
async def test_jira_expired_token_refresh_uses_json_body(db_session, monkeypatch):
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="jira",
            name="Jira",
            client_id=encrypt_value("jira-client-id"),
            client_secret=encrypt_value("jira-client-secret"),
            auth_url="https://auth.atlassian.com/authorize",
            token_url="https://auth.atlassian.com/oauth/token",
            redirect_uri="https://app.example.com/api/auth/jira/callback",
            userinfo_url="https://api.atlassian.com/me",
            user_id_path="account_id",
            email_path="email",
            default_scopes=["read:jira-work", "offline_access"],
        )
    )
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="jira",
        access_token="old-token",
        refresh_token="old-refresh",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_user_id="jira-user-1",
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
                    # Atlassian rotates refresh tokens on every use.
                    "refresh_token": "new-refresh",
                    "token_type": "bearer",
                    "expires_in": 3600,
                },
                status_code=200,
            )

    monkeypatch.setattr(tool_config.httpx, "AsyncClient", FakeAsyncClient)

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "jira")
        is True
    )

    assert oauth_account.access_token == "new-token"
    assert oauth_account.refresh_token == "new-refresh"
    assert len(captured_requests) == 1
    url, kwargs = captured_requests[0]
    assert url == "https://auth.atlassian.com/oauth/token"
    assert kwargs["headers"]["Content-Type"] == "application/json"
    assert kwargs["json"] == {
        "grant_type": "refresh_token",
        "refresh_token": "old-refresh",
        "client_id": "jira-client-id",
        "client_secret": "jira-client-secret",
    }
    assert "data" not in kwargs
