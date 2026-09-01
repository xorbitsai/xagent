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


def _meta_oauth_account(user) -> UserOAuth:
    return UserOAuth(
        user_id=user.id,
        provider="facebook",
        access_token="old-long-token",
        refresh_token=None,
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_user_id="meta-user-1",
    )


@pytest.fixture(autouse=True)
def sleep_delays(monkeypatch):
    """Replace the retry backoff's real wall-clock sleep with one that
    records the delay it was asked for, so tests can assert on the
    backoff/jitter formula instead of just the resulting call count."""
    delays: list[float] = []

    async def instant_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(tool_config.asyncio, "sleep", instant_sleep)
    return delays


class _FakeAsyncClient:
    """Minimal async-context-manager stand-in for httpx.AsyncClient whose
    `post` and/or `get` are supplied by the test. Meta's refresh path uses
    `get` (fb_exchange_token); every other provider uses `post`."""

    def __init__(self, post=None, get=None):
        self._post = post
        self._get = get

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, *args, **kwargs):
        if self._post is None:
            raise NotImplementedError("post not mocked for this test")
        return await self._post(*args, **kwargs)

    async def get(self, *args, **kwargs):
        if self._get is None:
            raise NotImplementedError("get not mocked for this test")
        return await self._get(*args, **kwargs)


@pytest.mark.asyncio
async def test_refresh_retries_transient_network_error_then_succeeds(
    db_session, monkeypatch, sleep_delays
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
    # base delay (0.5s) plus up to a full base delay of jitter -- see
    # OAUTH_REFRESH_RETRY_BASE_DELAY_SECONDS and the backoff formula in
    # _request_oauth_refresh_with_retries.
    assert len(sleep_delays) == 1
    assert 0.5 <= sleep_delays[0] < 1.0


@pytest.mark.asyncio
async def test_refresh_retries_connect_error_then_succeeds(db_session, monkeypatch):
    """ConnectError (e.g. connection refused, DNS failure) is a distinct
    exception from ConnectTimeout but carries the same "request never
    transmitted" guarantee -- make sure it's retried too, not just
    ConnectTimeout."""
    db, user = db_session
    oauth_account = _github_oauth_account(user)
    db.add(oauth_account)
    db.commit()

    calls: list[int] = []

    async def post(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("connection refused")
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
async def test_refresh_succeeds_on_first_try_without_retry(db_session, monkeypatch):
    """The plain happy path: no failure, no retry spent."""
    db, user = db_session
    oauth_account = _github_oauth_account(user)
    db.add(oauth_account)
    db.commit()

    calls: list[int] = []

    async def post(*args, **kwargs):
        calls.append(1)
        return MockResponse({"access_token": "new-token", "expires_in": 3600})

    monkeypatch.setattr(
        tool_config.httpx, "AsyncClient", lambda: _FakeAsyncClient(post)
    )

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "github")
        is True
    )
    assert len(calls) == 1
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
    # Asserts the literal attempt count (2), not tool_config.OAUTH_REFRESH_
    # MAX_ATTEMPTS -- reading the same constant the code under test uses
    # would make this self-referential and unable to catch a regression in
    # the intended attempt count.
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_refresh_gives_up_after_max_attempts_on_persistent_5xx(
    db_session, monkeypatch
):
    """The exception-exhaustion case above has a response-based sibling:
    every attempt returning 5xx (never an exception) must also give up
    after OAUTH_REFRESH_MAX_ATTEMPTS and return False, not retry forever."""
    db, user = db_session
    oauth_account = _github_oauth_account(user)
    db.add(oauth_account)
    db.commit()

    calls: list[int] = []

    async def post(*args, **kwargs):
        calls.append(1)
        return MockResponse({"error": "server_error"}, status_code=503)

    monkeypatch.setattr(
        tool_config.httpx, "AsyncClient", lambda: _FakeAsyncClient(post)
    )

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "github")
        is False
    )
    assert len(calls) == 2


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


@pytest.mark.asyncio
async def test_refresh_retries_transient_network_error_for_meta_get_path(
    db_session, monkeypatch
):
    """Meta's refresh path uses client.get (fb_exchange_token), not POST
    like every other provider -- make sure the retry wrapper covers it too,
    not just the POST path the other tests in this file exercise."""
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="meta",
            name="Meta",
            client_id=encrypt_value("meta-client-id"),
            client_secret=encrypt_value("meta-client-secret"),
            auth_url="https://www.facebook.com/v25.0/dialog/oauth",
            token_url="https://graph.facebook.com/v25.0/oauth/access_token",
            redirect_uri="https://app.example.com/api/auth/meta/callback",
            userinfo_url="https://graph.facebook.com/v25.0/me?fields=id,email",
            user_id_path="id",
            email_path="email",
            default_scopes=["public_profile"],
        )
    )
    oauth_account = _meta_oauth_account(user)
    db.add(oauth_account)
    db.commit()

    calls: list[int] = []

    async def get(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectTimeout("connection timed out")
        return MockResponse({"access_token": "new-long-token", "expires_in": 5184000})

    monkeypatch.setattr(
        tool_config.httpx, "AsyncClient", lambda: _FakeAsyncClient(get=get)
    )

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "meta")
        is True
    )
    assert len(calls) == 2
    assert oauth_account.access_token == "new-long-token"
