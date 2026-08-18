from __future__ import annotations

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
            app_id="github",
            name="GitHub",
            description="GitHub connector",
            transport="oauth",
            provider_name="github",
            category="Development",
            # read:org was dropped from the canonical registry (no tool
            # calls an organization endpoint) -- kept in sync here too,
            # even though get_app_by_id overlays the canonical registry's
            # oauth_scopes for this builtin app_id regardless of this row's
            # value (see test_generic_oauth_login.py's
            # test_github_login_requests_exact_canonical_scope).
            oauth_scopes=["repo", "user:email"],
            is_visible_in_connector=True,
            launch_config={
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.github"],
                "env_mapping": {"GITHUB_ACCESS_TOKEN": "access_token"},
            },
        )
    )
    db.commit()
    db.refresh(user)

    yield db, user
    db.close()
    engine.dispose()


def _github_provider() -> SimpleNamespace:
    return SimpleNamespace(
        provider_name="github",
        client_id=encrypt_value("github-client-id"),
        client_secret=encrypt_value("github-client-secret"),
        token_url="https://github.com/login/oauth/access_token",
        redirect_uri="https://app.example.com/api/auth/github/callback",
        userinfo_url="https://api.github.com/user",
        user_id_path="id",
        email_path="login",
        default_scopes=["read:user"],
    )


def test_github_callback_requests_json_and_sends_secret_in_body(
    db_session, monkeypatch
):
    """GitHub's token endpoint answers with a form-urlencoded body unless the
    request explicitly asks for JSON via Accept -- assert the exchange sets
    that header (unlike zoom, GitHub still takes client_id/secret in the body,
    not HTTP Basic Auth)."""
    db, user = db_session
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "github",
            "app_id": "github",
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "github-code", "state": state})

    post = Mock(
        return_value=MockResponse(
            {
                "access_token": "github-token",
                "token_type": "bearer",
                "scope": "repo,user:email",
            }
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", post)
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(return_value=MockResponse({"id": 42, "login": "octocat"})),
    )

    response = generic_oauth_callback("github", request, db, _github_provider())

    assert response.status_code == 200
    assert post.call_args.kwargs["headers"]["Accept"] == "application/json"
    assert post.call_args.kwargs["auth"] is None
    assert post.call_args.kwargs["data"]["client_id"] == "github-client-id"
    assert post.call_args.kwargs["data"]["client_secret"] == "github-client-secret"

    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "github")
        .one()
    )
    assert oauth_account.access_token == "github-token"
    assert oauth_account.provider_user_id == "42"
    assert oauth_account.email == "octocat"

    server = db.query(MCPServer).filter(MCPServer.name == "GitHub").one()
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


def test_non_github_callback_omits_accept_json_header(db_session, monkeypatch):
    """Guard the branch condition: a non-github provider must be unaffected by
    the GitHub-specific Accept header."""
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
            {"access_token": "tok", "token_type": "Bearer", "scope": ""}
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
    assert "Accept" not in post.call_args.kwargs["headers"]


@pytest.mark.asyncio
async def test_github_expired_token_refresh_sends_accept_json_header(
    db_session, monkeypatch
):
    """Same form-urlencoded-response quirk as the initial code exchange
    applies to GitHub's refresh_token grant too -- without the Accept header,
    a successful refresh's response.json() call would raise instead of
    parsing the new access token."""
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="github",
            name="GitHub",
            client_id=encrypt_value("github-client-id"),
            client_secret=encrypt_value("github-client-secret"),
            auth_url="https://github.com/login/oauth/authorize",
            token_url="https://github.com/login/oauth/access_token",
            redirect_uri="https://app.example.com/api/auth/github/callback",
            userinfo_url="https://api.github.com/user",
            user_id_path="id",
            email_path="login",
            default_scopes=["read:user"],
        )
    )
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="github",
        access_token="old-token",
        refresh_token="old-refresh",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_user_id="42",
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
                    "expires_in": 28800,
                },
                status_code=200,
            )

    monkeypatch.setattr(tool_config.httpx, "AsyncClient", FakeAsyncClient)

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "github")
        is True
    )

    assert oauth_account.access_token == "new-token"
    assert oauth_account.refresh_token == "new-refresh"
    assert len(captured_requests) == 1
    url, kwargs = captured_requests[0]
    assert url == "https://github.com/login/oauth/access_token"
    assert kwargs["headers"]["Accept"] == "application/json"
    assert kwargs["data"] == {
        "grant_type": "refresh_token",
        "refresh_token": "old-refresh",
        "client_id": "github-client-id",
        "client_secret": "github-client-secret",
    }


@pytest.mark.asyncio
async def test_github_refresh_surfaces_failure_instead_of_raising_on_form_body(
    db_session, monkeypatch
):
    """Guard against a regression of the bug this fix addresses: if GitHub
    ever answers form-urlencoded despite the Accept header, response.json()
    must fail cleanly (refresh reported as failed) rather than propagating an
    unhandled exception out of refresh_oauth_token_if_needed."""
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="github",
            name="GitHub",
            client_id=encrypt_value("github-client-id"),
            client_secret=encrypt_value("github-client-secret"),
            auth_url="https://github.com/login/oauth/authorize",
            token_url="https://github.com/login/oauth/access_token",
            redirect_uri="https://app.example.com/api/auth/github/callback",
            userinfo_url="https://api.github.com/user",
            user_id_path="id",
            email_path="login",
            default_scopes=["read:user"],
        )
    )
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="github",
        access_token="old-token",
        refresh_token="old-refresh",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_user_id="42",
    )
    db.add(oauth_account)
    db.commit()

    class FormBodyResponse:
        status_code = 200

        def json(self):
            raise ValueError("Expecting value: line 1 column 1 (char 0)")

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            return FormBodyResponse()

    monkeypatch.setattr(tool_config.httpx, "AsyncClient", FakeAsyncClient)

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "github")
        is False
    )
    assert oauth_account.access_token == "old-token"
