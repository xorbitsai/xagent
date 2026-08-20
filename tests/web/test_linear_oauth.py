from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.utils.encryption import encrypt_value
from xagent.web.api.auth import (
    _merge_oauth_scopes,
    _oauth_scope_separator,
    create_access_token,
    generic_oauth_callback,
    generic_oauth_login,
)
from xagent.web.models.database import Base
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
            app_id="linear",
            name="Linear",
            transport="oauth",
            provider_name="linear",
        )
    )
    db.commit()
    db.refresh(user)

    yield db, user
    db.close()
    engine.dispose()


def _linear_provider() -> SimpleNamespace:
    return SimpleNamespace(
        provider_name="linear",
        client_id=encrypt_value("linear-client-id"),
        client_secret=encrypt_value("linear-client-secret"),
        auth_url="https://linear.app/oauth/authorize",
        token_url="https://api.linear.app/oauth/token",
        redirect_uri="https://app.example.com/api/auth/linear/callback",
        # Deliberately empty, matching the registry row: Linear's API is
        # GraphQL-only, so there is no compatible flat REST userinfo endpoint.
        userinfo_url="",
        user_id_path="id",
        email_path="email",
        # NOT the same as the registry provider row's default_scopes
        # (["read"] there -- the "write" scope lives on the app row and is
        # merged in via app_id at authorize time). This fixture bypasses
        # that merge (db_provider is passed in directly, app_id lookup is
        # separate), so it uses both scopes directly to exercise the ","
        # separator regardless of the merge path.
        default_scopes=["read", "write"],
    )


def _callback_request(db, user) -> SimpleNamespace:
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "linear",
            "app_id": "linear",
        },
        expires_delta=timedelta(minutes=10),
    )
    return SimpleNamespace(query_params={"code": "linear-code", "state": state})


def test_scope_separator_joins_linear_scopes_with_commas():
    """Linear's authorize endpoint documents scope as a comma-separated
    list, unlike most providers here — a refactor of _oauth_scope_separator
    must not silently change this."""
    assert _oauth_scope_separator("linear") == ","
    provider = _linear_provider()
    joined = _oauth_scope_separator("linear").join(
        _merge_oauth_scopes(provider.default_scopes, None)
    )
    assert joined == "read,write"


def test_login_sends_comma_separated_scope_param(db_session):
    db, user = db_session
    token = create_access_token(
        data={"sub": user.username, "type": "access"},
        expires_delta=timedelta(minutes=5),
    )

    response = generic_oauth_login(
        "linear", token=token, app_id="linear", db=db, db_provider=_linear_provider()
    )

    assert response.status_code == 307
    assert "scope=read%2Cwrite" in response.headers["location"]


def test_callback_persists_token_without_userinfo_lookup(db_session, monkeypatch):
    """With userinfo_url left empty, the callback must skip the identity
    fetch entirely (no request attempted) and still persist the token."""
    db, user = db_session
    mock_post = Mock(
        return_value=MockResponse(
            {
                "access_token": "linear-token",
                "refresh_token": "linear-refresh",
                "token_type": "Bearer",
                "scope": "read,write",
                # Linear access tokens expire in 24 hours.
                "expires_in": 86400,
            }
        )
    )
    mock_get = Mock()
    monkeypatch.setattr("xagent.web.api.auth.requests.post", mock_post)
    monkeypatch.setattr("xagent.web.api.auth.requests.get", mock_get)

    response = generic_oauth_callback(
        "linear", _callback_request(db, user), db, _linear_provider()
    )

    assert response.status_code == 200
    mock_get.assert_not_called()

    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "linear")
        .first()
    )
    assert oauth_account is not None
    assert oauth_account.access_token == "linear-token"
    assert oauth_account.refresh_token == "linear-refresh"
    assert oauth_account.expires_at is not None
    assert oauth_account.email is None
    assert oauth_account.provider_user_id is None


def test_callback_normalizes_legacy_array_scope_to_string(db_session, monkeypatch):
    """Linear OAuth applications created before December 1, 2023 return
    "scope" as a list of strings rather than a joined string. UserOAuth.scope
    is a plain String column, so persisting the list as-is would raise at
    flush time instead of saving a valid connection."""
    db, user = db_session
    mock_post = Mock(
        return_value=MockResponse(
            {
                "access_token": "legacy-linear-token",
                "refresh_token": "legacy-linear-refresh",
                "token_type": "Bearer",
                "scope": ["read", "write"],
                "expires_in": 86400,
            }
        )
    )
    monkeypatch.setattr("xagent.web.api.auth.requests.post", mock_post)
    monkeypatch.setattr("xagent.web.api.auth.requests.get", Mock())

    response = generic_oauth_callback(
        "linear", _callback_request(db, user), db, _linear_provider()
    )

    assert response.status_code == 200
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "linear")
        .one()
    )
    assert oauth_account.scope == "read,write"


async def test_linear_expired_token_refresh_uses_generic_form_body(
    db_session, monkeypatch
):
    """Linear access tokens expire in 24 hours and have no provider-specific
    branch in refresh_oauth_token_if_needed, so this exercises the fully
    generic refresh path: form-encoded body, grant_type=refresh_token, and
    client credentials sent in the body (not Basic Auth, not JSON)."""
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="linear",
            name="Linear",
            client_id=encrypt_value("linear-client-id"),
            client_secret=encrypt_value("linear-client-secret"),
            auth_url="https://linear.app/oauth/authorize",
            token_url="https://api.linear.app/oauth/token",
            redirect_uri="https://app.example.com/api/auth/linear/callback",
            userinfo_url="",
            user_id_path="id",
            email_path="email",
            default_scopes=["read"],
        )
    )
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="linear",
        access_token="old-token",
        refresh_token="old-refresh",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_user_id="linear-user-1",
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
                    "token_type": "Bearer",
                    "expires_in": 86400,
                },
                status_code=200,
            )

    monkeypatch.setattr(tool_config.httpx, "AsyncClient", FakeAsyncClient)

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "linear")
        is True
    )

    assert oauth_account.access_token == "new-token"
    assert oauth_account.refresh_token == "new-refresh"
    assert oauth_account.expires_at > datetime.now(timezone.utc) + timedelta(hours=1)
    assert len(captured_requests) == 1
    url, kwargs = captured_requests[0]
    assert url == "https://api.linear.app/oauth/token"
    assert kwargs["data"] == {
        "grant_type": "refresh_token",
        "refresh_token": "old-refresh",
        "client_id": "linear-client-id",
        "client_secret": "linear-client-secret",
    }
    assert "json" not in kwargs
    assert "auth" not in kwargs
    assert kwargs["headers"] == {}
