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
            app_id="salesforce",
            name="Salesforce",
            transport="oauth",
            provider_name="salesforce",
        )
    )
    db.commit()
    db.refresh(user)

    yield db, user
    db.close()
    engine.dispose()


def _salesforce_provider() -> SimpleNamespace:
    return SimpleNamespace(
        provider_name="salesforce",
        client_id=encrypt_value("salesforce-client-id"),
        client_secret=encrypt_value("salesforce-client-secret"),
        auth_url="https://login.salesforce.com/services/oauth2/authorize",
        token_url="https://login.salesforce.com/services/oauth2/token",
        redirect_uri="https://app.example.com/api/auth/salesforce/callback",
        userinfo_url="",
        user_id_path="user_id",
        email_path="email",
        default_scopes=["api", "refresh_token", "openid"],
    )


def _callback_request(db, user) -> SimpleNamespace:
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "salesforce",
            "app_id": "salesforce",
        },
        expires_delta=timedelta(minutes=10),
    )
    return SimpleNamespace(query_params={"code": "sf-code", "state": state})


def test_callback_persists_instance_url_and_skips_userinfo_lookup(
    db_session, monkeypatch
):
    """With userinfo_url left empty, the callback must skip the identity
    fetch entirely (no request attempted) while still persisting
    instance_url -- the per-org host every subsequent API call needs."""
    db, user = db_session
    mock_post = Mock(
        return_value=MockResponse(
            {
                "access_token": "sf-token",
                "refresh_token": "sf-refresh",
                "instance_url": "https://acme.my.salesforce.com",
                "token_type": "Bearer",
                "scope": "api refresh_token openid",
                "id": "https://login.salesforce.com/id/00D.../005...",
            }
        )
    )
    mock_get = Mock()
    monkeypatch.setattr(auth_api.requests, "post", mock_post)
    monkeypatch.setattr(auth_api.requests, "get", mock_get)

    response = generic_oauth_callback(
        "salesforce", _callback_request(db, user), db, _salesforce_provider()
    )

    assert response.status_code == 200
    mock_get.assert_not_called()
    # Standard form-urlencoded exchange -- Salesforce needs none of the
    # per-provider quirks Zoom/GitHub/Jira require.
    assert mock_post.call_args.kwargs["headers"]["Content-Type"] == (
        "application/x-www-form-urlencoded"
    )
    assert mock_post.call_args.kwargs["data"]["client_id"] == "salesforce-client-id"
    assert mock_post.call_args.kwargs["data"]["client_secret"] == (
        "salesforce-client-secret"
    )

    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "salesforce")
        .first()
    )
    assert oauth_account is not None
    assert oauth_account.access_token == "sf-token"
    assert oauth_account.refresh_token == "sf-refresh"
    assert oauth_account.instance_url == "https://acme.my.salesforce.com"
    assert oauth_account.email is None

    server = db.query(MCPServer).filter(MCPServer.name == "Salesforce").one()
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


def test_non_salesforce_callback_does_not_persist_instance_url(db_session, monkeypatch):
    """Guard the branch condition: a provider whose token response has no
    instance_url key must leave that column unset, not crash or coerce it."""
    db, user = db_session

    # No app_id: generic_oauth_callback persists UserOAuth.provider as
    # (app_id or provider), so a bare connect keeps it queryable as "google"
    # directly rather than under an app_id like "gmail".
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "google",
            "app_id": None,
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "code", "state": state})
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {"access_token": "tok", "token_type": "Bearer", "expires_in": 3600}
            )
        ),
    )
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
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "google")
        .one()
    )
    assert oauth_account.instance_url is None


@pytest.mark.asyncio
async def test_salesforce_refresh_updates_instance_url(db_session, monkeypatch):
    """Salesforce can return a different instance_url on refresh (e.g.
    after an org migration) -- this must be re-persisted, not just captured
    at initial connect."""
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="salesforce",
            name="Salesforce",
            client_id=encrypt_value("salesforce-client-id"),
            client_secret=encrypt_value("salesforce-client-secret"),
            auth_url="https://login.salesforce.com/services/oauth2/authorize",
            token_url="https://login.salesforce.com/services/oauth2/token",
            redirect_uri="https://app.example.com/api/auth/salesforce/callback",
            userinfo_url="",
            user_id_path="user_id",
            email_path="email",
            default_scopes=["api", "refresh_token", "openid"],
        )
    )
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="salesforce",
        access_token="old-token",
        refresh_token="old-refresh",
        instance_url="https://acme.my.salesforce.com",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_user_id="005...",
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
                    "instance_url": "https://acme2.my.salesforce.com",
                    "token_type": "Bearer",
                },
                status_code=200,
            )

    monkeypatch.setattr(tool_config.httpx, "AsyncClient", FakeAsyncClient)

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "salesforce")
        is True
    )

    assert oauth_account.access_token == "new-token"
    assert oauth_account.instance_url == "https://acme2.my.salesforce.com"
    assert len(captured_requests) == 1
    _, kwargs = captured_requests[0]
    assert kwargs["data"] == {
        "grant_type": "refresh_token",
        "refresh_token": "old-refresh",
        "client_id": "salesforce-client-id",
        "client_secret": "salesforce-client-secret",
    }
