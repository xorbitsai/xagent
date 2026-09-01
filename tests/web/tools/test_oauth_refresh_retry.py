from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# refresh_oauth_token_if_needed lazily imports xagent.web.api.auth (which
# pulls in authlib's httpx-subclassing OAuth2 client) on first call. Forcing
# that import here, before any test monkeypatches httpx.AsyncClient, avoids
# a metaclass conflict when this file is the first to trigger it in the
# pytest session (harmless once xagent.web.api.auth is already imported
# elsewhere, which every sibling *_oauth.py test file already does).
import xagent.web.api.auth  # noqa: F401
from xagent.core.utils.encryption import encrypt_value
from xagent.web.models.database import Base
from xagent.web.models.oauth_provider import OAuthProvider
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
            email_path="email",
            default_scopes=["repo"],
        )
    )
    db.commit()
    db.refresh(user)

    yield db, user
    db.close()
    engine.dispose()


def _github_oauth_account(user) -> UserOAuth:
    return UserOAuth(
        user_id=user.id,
        provider="github",
        access_token="old-token",
        refresh_token="old-refresh",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_user_id="42",
    )


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """The retry backoff would otherwise add real wall-clock delay to
    every test in this file."""

    async def instant_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(tool_config.asyncio, "sleep", instant_sleep)


class _FakeAsyncClient:
    """Minimal async-context-manager stand-in for httpx.AsyncClient whose
    `post` is supplied by the test."""

    def __init__(self, post):
        self._post = post

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, *args, **kwargs):
        return await self._post(*args, **kwargs)


@pytest.mark.asyncio
async def test_refresh_retries_transient_network_error_then_succeeds(
    db_session, monkeypatch
):
    db, user = db_session
    oauth_account = _github_oauth_account(user)
    db.add(oauth_account)
    db.commit()

    calls: list[int] = []

    async def post(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectTimeout("connection timed out")
        return MockResponse({"access_token": "new-token", "expires_in": 3600})

    monkeypatch.setattr(
        tool_config.httpx, "AsyncClient", lambda: _FakeAsyncClient(post)
    )

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "github")
        is True
    )
    assert len(calls) == 2
    assert oauth_account.access_token == "new-token"


@pytest.mark.asyncio
async def test_refresh_retries_5xx_then_succeeds(db_session, monkeypatch):
    db, user = db_session
    oauth_account = _github_oauth_account(user)
    db.add(oauth_account)
    db.commit()

    calls: list[int] = []

    async def post(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return MockResponse({"error": "server_error"}, status_code=503)
        return MockResponse({"access_token": "new-token", "expires_in": 3600})

    monkeypatch.setattr(
        tool_config.httpx, "AsyncClient", lambda: _FakeAsyncClient(post)
    )

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "github")
        is True
    )
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_refresh_does_not_retry_on_definitive_4xx(db_session, monkeypatch):
    """A 4xx is the provider's definitive answer about this token -- a
    retry can't change it, so it should not spend one."""
    db, user = db_session
    oauth_account = _github_oauth_account(user)
    db.add(oauth_account)
    db.commit()

    calls: list[int] = []

    async def post(*args, **kwargs):
        calls.append(1)
        return MockResponse({"error": "invalid_grant"}, status_code=400)

    monkeypatch.setattr(
        tool_config.httpx, "AsyncClient", lambda: _FakeAsyncClient(post)
    )

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "github")
        is False
    )
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_refresh_gives_up_after_max_attempts_on_persistent_network_error(
    db_session, monkeypatch
):
    db, user = db_session
    oauth_account = _github_oauth_account(user)
    db.add(oauth_account)
    db.commit()

    calls: list[int] = []

    async def post(*args, **kwargs):
        calls.append(1)
        raise httpx.ConnectTimeout("connection timed out")

    monkeypatch.setattr(
        tool_config.httpx, "AsyncClient", lambda: _FakeAsyncClient(post)
    )

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "github")
        is False
    )
    assert len(calls) == tool_config.OAUTH_REFRESH_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_refresh_does_not_retry_on_ambiguous_read_timeout(
    db_session, monkeypatch
):
    """A ReadTimeout means the request was already sent -- the provider may
    have processed the grant (and, for a rotating-refresh-token provider,
    already issued a new refresh_token) before the response was lost.
    Blindly resending the now-stale refresh_token isn't safe, so this must
    fail without retrying, unlike a connect-phase failure."""
    db, user = db_session
    oauth_account = _github_oauth_account(user)
    db.add(oauth_account)
    db.commit()

    calls: list[int] = []

    async def post(*args, **kwargs):
        calls.append(1)
        raise httpx.ReadTimeout("timed out waiting for response")

    monkeypatch.setattr(
        tool_config.httpx, "AsyncClient", lambda: _FakeAsyncClient(post)
    )

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "github")
        is False
    )
    assert len(calls) == 1
