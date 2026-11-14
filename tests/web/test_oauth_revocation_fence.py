"""The revocation fence between a Google disconnect and a concurrent callback.

Google revokes a whole grant per Google account and OAuth client. When Alice
disconnects while Bob connects the same Google account, Bob's code exchange
may be issued tokens of the very grant Alice's revoke then kills. These tests
interleave the two for real -- Alice's revoke runs on its own session while
Bob's callback is inside its token exchange -- with only provider HTTP mocked.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from tests.shared.oauth_browser_fixture import (
    create_access_token,
    generic_oauth_callback,
)
from xagent.core.utils.encryption import encrypt_value
from xagent.web.api import auth as auth_api
from xagent.web.models.database import Base
from xagent.web.models.oauth_revocation_fence import OAuthRevocationFence
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services import oauth_revocation_fences

TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
SUBJECT = "google-subject"


class MockResponse:
    def __init__(self, json_data=None, status_code: int = 200, text: str = ""):
        self._json_data = json_data or {}
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._json_data


@pytest.fixture()
def sessions(tmp_path):
    """Two sessions on one file database: Bob's callback and Alice's revoke."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'fence.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    bob_db, alice_db = factory(), factory()
    yield bob_db, alice_db
    bob_db.close()
    alice_db.close()
    engine.dispose()


def _google_provider() -> SimpleNamespace:
    return SimpleNamespace(
        provider_name="google",
        client_id=encrypt_value("google-client-id"),
        client_secret=encrypt_value("google-client-secret"),
        auth_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url=TOKEN_URL,
        redirect_uri="https://app.example.com/api/auth/google/callback",
        userinfo_url="https://openidconnect.googleapis.com/v1/userinfo",
        user_id_path="sub",
        email_path="email",
        default_scopes=["openid"],
    )


def _bob(db: Session) -> User:
    bob = User(username="bob", password_hash="x", is_admin=False)
    db.add(bob)
    db.commit()
    db.refresh(bob)
    return bob


def _connect_google(db: Session, user: User):
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": int(user.id),
            "provider": "google",
            "app_id": None,
        },
        expires_delta=timedelta(minutes=10),
    )
    return generic_oauth_callback(
        "google",
        SimpleNamespace(query_params={"code": "code", "state": state}),
        db,
        _google_provider(),
    )


def _revoke_alices_grant(db: Session) -> None:
    """Alice's disconnect of the same Google account, after its commit."""
    auth_api.revoke_builtin_oauth_grant(
        db,
        provider="google",
        access_token="alice-access",
        provider_user_id=SUBJECT,
        refresh_token="alice-refresh",
    )


def _provider_http(monkeypatch, *, during_exchange=None) -> list[str]:
    """Mock Google; returns the tokens sent to its revoke endpoint."""
    revoked: list[str] = []

    def post(url, **kwargs):
        if url == REVOKE_URL:
            revoked.append(kwargs["data"]["token"])
            return MockResponse(status_code=200)
        assert url == TOKEN_URL
        if during_exchange is not None:
            during_exchange()
        return MockResponse(
            {
                "access_token": "bob-access",
                "refresh_token": "bob-refresh",
                "token_type": "Bearer",
                "scope": "openid",
            }
        )

    monkeypatch.setattr(auth_api.requests, "post", post)
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(return_value=MockResponse({"sub": SUBJECT, "email": "bob@example.com"})),
    )
    return revoked


def test_a_callback_whose_tokens_a_revoke_may_have_killed_is_refused(
    sessions, monkeypatch
):
    """Alice's revoke reaches Google while Bob's exchange is in flight: her
    reference check cannot see Bob's uncommitted row, so Bob must not store
    the now-dead tokens and report success."""
    bob_db, alice_db = sessions
    bob = _bob(bob_db)
    revoked = _provider_http(
        monkeypatch, during_exchange=lambda: _revoke_alices_grant(alice_db)
    )

    response = _connect_google(bob_db, bob)

    assert revoked == ["alice-refresh"]
    assert response.status_code == 409
    assert "connect again" in response.body.decode()
    assert bob_db.query(UserOAuth).count() == 0


def _fence_deadline(db: Session) -> datetime:
    """The fence's stored deadline, read fresh and tz-aware."""
    db.expire_all()
    fence = db.get(OAuthRevocationFence, ("google", SUBJECT))
    assert fence is not None and fence.revoking_until is not None
    return oauth_revocation_fences._utc(fence.revoking_until)


def test_a_callback_that_starts_after_the_revoke_finished_is_accepted(
    sessions, monkeypatch
):
    """A reconnect right after a disconnect gets a fresh grant."""
    bob_db, alice_db = sessions
    bob = _bob(bob_db)
    revoked = _provider_http(monkeypatch)
    # The close margin only absorbs clock skew between hosts; this callback
    # starts in the same instant on the same clock.
    monkeypatch.setattr(
        oauth_revocation_fences, "OAUTH_REVOCATION_CLOSE_MARGIN", timedelta(0)
    )
    _revoke_alices_grant(alice_db)

    response = _connect_google(bob_db, bob)

    assert revoked == ["alice-refresh"]
    assert response.status_code == 200
    assert bob_db.query(UserOAuth).filter_by(provider_user_id=SUBJECT).count() == 1


def test_a_revoke_after_the_callback_committed_is_skipped(sessions, monkeypatch):
    bob_db, alice_db = sessions
    bob = _bob(bob_db)
    revoked = _provider_http(monkeypatch)

    assert _connect_google(bob_db, bob).status_code == 200
    _revoke_alices_grant(alice_db)

    assert revoked == []


def test_a_revoker_that_never_closed_its_fence_refuses_until_it_lapses(
    sessions, monkeypatch
):
    """A revoker that died mid-call leaves its deadline in place; callbacks
    are refused only until it lapses."""
    bob_db, alice_db = sessions
    bob = _bob(bob_db)
    _provider_http(monkeypatch)
    fence = oauth_revocation_fences.lock_oauth_revocation_fence(
        alice_db, provider="google", provider_user_id=SUBJECT
    )
    oauth_revocation_fences.open_oauth_revocation_fence(fence)
    alice_db.commit()

    assert _connect_google(bob_db, bob).status_code == 409

    stored = alice_db.get(OAuthRevocationFence, ("google", SUBJECT))
    stored.revoking_until -= oauth_revocation_fences.OAUTH_REVOCATION_FENCE_WINDOW
    alice_db.commit()

    assert _connect_google(bob_db, bob).status_code == 200


def test_a_revoker_delayed_past_its_fence_skips_the_call(sessions, monkeypatch):
    _bob_db, alice_db = sessions
    revoked = _provider_http(monkeypatch)
    monkeypatch.setattr(
        oauth_revocation_fences,
        "OAUTH_REVOCATION_CALL_MARGIN",
        oauth_revocation_fences.OAUTH_REVOCATION_FENCE_WINDOW,
    )

    _revoke_alices_grant(alice_db)

    assert revoked == []


def test_a_closed_fence_keeps_refusing_for_the_clock_skew_margin(sessions, monkeypatch):
    """A callback host whose clock runs ahead of the revoker's records an
    exchange start after the close although the exchange began before the
    call ended; the close margin covers that."""
    bob_db, alice_db = sessions
    bob = _bob(bob_db)
    _provider_http(monkeypatch)
    _revoke_alices_grant(alice_db)

    margin = oauth_revocation_fences.OAUTH_REVOCATION_CLOSE_MARGIN
    assert _fence_deadline(alice_db) > datetime.now(timezone.utc)
    assert _fence_deadline(alice_db) <= datetime.now(timezone.utc) + margin
    assert _connect_google(bob_db, bob).status_code == 409


def test_a_second_revoker_cannot_close_the_fence_under_one_in_flight(
    sessions, monkeypatch
):
    """Two disconnects of the same account pass their reference check. While
    the first one's provider call is in flight the second runs to the end: it
    must neither revoke nor close the fence, or a reconnect right after it
    would be accepted and then killed by the first call."""
    bob_db, alice_db = sessions
    carol_db = sessionmaker(bind=alice_db.get_bind(), autoflush=False)()
    bob = _bob(bob_db)
    revoked = _provider_http(monkeypatch)
    provider_post = auth_api.requests.post
    reconnect_status: list[int] = []

    def post(url, **kwargs):
        if url == REVOKE_URL and not reconnect_status:
            reconnect_status.append(0)  # re-entrancy guard
            auth_api.revoke_builtin_oauth_grant(
                carol_db,
                provider="google",
                access_token="carol-access",
                provider_user_id=SUBJECT,
                refresh_token="carol-refresh",
            )
            reconnect_status[0] = _connect_google(bob_db, bob).status_code
        return provider_post(url, **kwargs)

    monkeypatch.setattr(auth_api.requests, "post", post)

    try:
        _revoke_alices_grant(alice_db)
    finally:
        carol_db.close()

    assert revoked == ["alice-refresh"]
    assert reconnect_status == [409]


def _alices_revocation() -> auth_api.BuiltinOAuthRevocation:
    return auth_api.BuiltinOAuthRevocation(
        provider="google",
        access_token="alice-access",
        client_id="",
        client_secret="",
        provider_user_id=SUBJECT,
        refresh_token="alice-refresh",
    )


@pytest.mark.asyncio
async def test_a_cancelled_revoke_closes_its_fence_once_the_call_ended(
    sessions, monkeypatch
):
    """Cancelling the request while the provider call is in flight must not
    leave the fence open until its deadline, nor close it while the call's
    thread is still running."""
    bob_db, alice_db = sessions
    in_call, finish_call = threading.Event(), threading.Event()
    finished: list[bool] = []

    def blocked_revoke(revocation) -> None:
        in_call.set()
        finish_call.wait(timeout=10)
        finished.append(True)

    monkeypatch.setattr(auth_api, "revoke_resolved_builtin_oauth_grant", blocked_revoke)
    task = asyncio.create_task(
        auth_api.revoke_builtin_oauth_grants(
            alice_db, [_alices_revocation()], context="test"
        )
    )
    assert await asyncio.to_thread(in_call.wait, 10)

    task.cancel()
    await asyncio.sleep(0.05)
    # Cancelled, but the call is still running: the fence stays open.
    assert not task.done()
    window = oauth_revocation_fences.OAUTH_REVOCATION_FENCE_WINDOW
    assert _fence_deadline(bob_db) > datetime.now(timezone.utc) + window / 2

    finish_call.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert finished == [True]
    margin = oauth_revocation_fences.OAUTH_REVOCATION_CLOSE_MARGIN
    assert _fence_deadline(bob_db) <= datetime.now(timezone.utc) + margin
