"""Server-enforced idle expiry for interactive browser logins (issue #3771).

State lives in the real SQLite schema: separate Sessions stand in for separate
API workers, and an aged ``browser_sessions.last_activity_at`` stands in for
wall-clock idle time.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from tests.web import test_auth_api as auth_tests
from tests.web.test_auth_api import test_db as test_db
from xagent.web import browser_idle
from xagent.web import jwt_codec as jwt
from xagent.web.api import auth, files, oidc_google, progress_ws
from xagent.web.api import websocket as websocket_api
from xagent.web.api import websocket_auth
from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY, validate_auth_config
from xagent.web.auth_sessions import hash_refresh_token, user_session_claims
from xagent.web.models.browser_session import BrowserIdleEnforcement, BrowserSession
from xagent.web.models.user import User

pytestmark = pytest.mark.usefixtures("test_db")

PASSWORD = "idle-password-123"
ENV_ENABLED = "XAGENT_BROWSER_IDLE_TIMEOUT_ENABLED"
ENV_MINUTES = "XAGENT_BROWSER_IDLE_TIMEOUT_MINUTES"
ENV_ADMIN_MINUTES = "XAGENT_BROWSER_IDLE_TIMEOUT_ADMIN_MINUTES"


@pytest.fixture(autouse=True)
def _default_policy(monkeypatch):
    for name in (ENV_ENABLED, ENV_MINUTES, ENV_ADMIN_MINUTES):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def enforced(monkeypatch):
    monkeypatch.setenv(ENV_ENABLED, "true")


def _switch(monkeypatch, on: bool) -> None:
    monkeypatch.setenv(ENV_ENABLED, "true" if on else "false")


def _users() -> None:
    """One platform admin and one ordinary user."""
    auth_tests.setup_first_admin(password=PASSWORD)
    with auth_tests.TestingSessionLocal() as session:
        session.add(
            User(
                username="member",
                email="member@example.com",
                password_hash=auth.hash_password(PASSWORD),
                is_admin=False,
            )
        )
        session.commit()


def _login(username: str = "member") -> dict[str, Any]:
    response = auth_tests.client.post(
        "/api/auth/login", json={"username": username, "password": PASSWORD}
    )
    assert response.status_code == 200, response.text
    return response.json()


def _bearer(tokens: dict[str, Any]) -> dict[str, str]:
    return {"Authorization": f"Bearer {tokens['access_token']}"}


def _verify(tokens: dict[str, Any]):
    return auth_tests.client.get("/api/auth/verify", headers=_bearer(tokens))


def _refresh(tokens: dict[str, Any]):
    return auth_tests.client.post(
        "/api/auth/refresh", json={"refresh_token": tokens["refresh_token"]}
    )


def _claims(token: str) -> dict[str, Any]:
    return jwt.decode(
        token,
        JWT_SECRET_KEY,
        algorithms=[JWT_ALGORITHM],
        options={"verify_sub": False, "verify_exp": False},
    )


def _sid(tokens: dict[str, Any]) -> str:
    return _claims(tokens["access_token"])["sid"]


def _user(username: str = "member") -> User:
    with auth_tests.TestingSessionLocal() as session:
        user = session.query(User).filter_by(username=username).one()
        session.expunge(user)
        return user


def _row(sid: str) -> BrowserSession | None:
    with auth_tests.TestingSessionLocal() as session:
        row = session.get(BrowserSession, sid)
        if row is not None:
            session.expunge(row)
        return row


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _set_activity(sid: str, at: datetime) -> None:
    with auth_tests.TestingSessionLocal() as session:
        session.get(BrowserSession, sid).last_activity_at = at
        session.commit()


def _idle_for(tokens: dict[str, Any], minutes: float) -> datetime:
    at = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    _set_activity(_sid(tokens), at)
    return at


def _assert_idle_rejection(response) -> None:
    assert response.status_code == 401, response.text
    assert response.headers.get("Error-Type") == "SessionIdleExpired"


# --- Configuration ------------------------------------------------------------


def test_disabled_policy_keeps_long_idle_logins_working():
    _users()
    tokens = _login()
    _idle_for(tokens, 24 * 60)

    assert _verify(tokens).status_code == 200
    refreshed = _refresh(tokens)
    assert refreshed.status_code == 200
    idle = refreshed.json()["idle_session"]
    assert idle["enforced"] is False
    assert idle["sid"] == _sid(tokens)


@pytest.mark.parametrize(
    "env",
    [
        {ENV_ENABLED: "true", ENV_MINUTES: "0"},
        {ENV_ENABLED: "true", ENV_ADMIN_MINUTES: "-5"},
        {ENV_ENABLED: "true", ENV_MINUTES: "thirty"},
        {ENV_ENABLED: "true", ENV_MINUTES: str(366 * 24 * 60)},
        {ENV_ENABLED: "maybe"},
    ],
)
def test_invalid_enabled_policy_refuses_to_start(monkeypatch, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match="XAGENT_BROWSER_IDLE_TIMEOUT"):
        validate_auth_config()


def test_invalid_thresholds_do_not_block_startup_while_disabled(monkeypatch):
    monkeypatch.setenv(ENV_ENABLED, "false")
    monkeypatch.setenv(ENV_MINUTES, "thirty")
    validate_auth_config()


# --- Login identity and response contract --------------------------------------


def test_each_login_gets_its_own_browser_session(enforced):
    _users()
    first = _login()
    second = _login()

    sid = _sid(second)
    assert sid == _claims(second["refresh_token"])["sid"]
    assert sid != _sid(first)
    assert _row(sid) is not None and _row(_sid(first)) is not None
    # The single stored refresh token belongs to the newest login.
    assert _user().browser_session_id == sid
    idle = second["idle_session"]
    assert idle["sid"] == sid
    assert idle["enforced"] is True
    assert idle["idle_timeout_seconds"] == 30 * 60
    assert idle["absolute_expires_in"] is None


def test_oidc_login_starts_a_browser_session(enforced):
    _users()
    with auth_tests.TestingSessionLocal() as session:
        user = session.query(User).filter_by(username="member").one()
        payload = oidc_google._issue_auth_payload(session, user)
    sid = _sid(payload)
    assert _claims(payload["refresh_token"])["sid"] == sid
    assert payload["idle_session"]["sid"] == sid
    assert _row(sid) is not None
    assert _user().browser_session_id == sid


# --- Enforcement ----------------------------------------------------------------


def test_idle_login_is_rejected_before_its_jwt_expires(enforced):
    _users()
    tokens = _login()
    _idle_for(tokens, 31)

    _assert_idle_rejection(_verify(tokens))
    _assert_idle_rejection(
        auth_tests.client.get("/api/auth/me", headers=_bearer(tokens))
    )
    _assert_idle_rejection(_refresh(tokens))
    assert _row(_sid(tokens)).revoked_at is not None


def test_expired_login_stays_dead_after_the_switch_is_turned_off(monkeypatch):
    """Idle expiry -> switch off -> old access and refresh tokens still fail."""
    _switch(monkeypatch, True)
    _users()
    old = _login()
    _idle_for(old, 31)
    _assert_idle_rejection(_verify(old))  # Seen while the switch is on.

    _switch(monkeypatch, False)
    _assert_idle_rejection(_verify(old))
    _assert_idle_rejection(_refresh(old))
    newer = _login()
    assert _verify(newer).status_code == 200
    assert _refresh(newer).status_code == 200


def test_sweep_revokes_logins_no_request_ever_saw_expire(monkeypatch):
    """A closed browser's login is revoked by the sweep, durably."""
    _switch(monkeypatch, True)
    _users()
    closed = _login()
    active = _login("administrator")
    _idle_for(closed, 31)
    with auth_tests.TestingSessionLocal() as session:
        revoked, _ = browser_idle.sweep_browser_sessions(session)
    assert revoked == 1

    _switch(monkeypatch, False)
    _assert_idle_rejection(_verify(closed))
    _assert_idle_rejection(_refresh(closed))
    assert _verify(active).status_code == 200


def _lease_until(at: datetime, ordinary_minutes: int = 30) -> None:
    """As left by enforcing replicas: one threshold set and its lease end."""
    with auth_tests.TestingSessionLocal() as session:
        session.merge(
            BrowserIdleEnforcement(
                ordinary_minutes=ordinary_minutes,
                admin_minutes=15,
                enforced_until=at,
            )
        )
        session.commit()


def test_expiry_no_request_saw_stays_dead_after_switch_off(monkeypatch):
    """Expiry after the last sweep, no request, switch off, then access."""
    _switch(monkeypatch, True)
    _users()
    tokens = _login()
    with auth_tests.TestingSessionLocal() as session:
        browser_idle.renew_enforcement_lease(session)  # The enabled heartbeat.
        session.commit()
    _idle_for(tokens, 31)  # Expired while enforced; no request, no sweep.

    _switch(monkeypatch, False)
    _assert_idle_rejection(_verify(tokens))
    _assert_idle_rejection(_refresh(tokens))
    assert _row(_sid(tokens)).revoked_at is not None


def test_a_login_alive_at_switch_off_is_free_at_once(monkeypatch):
    """The policy went off 40 minutes ago; this login was alive then."""
    _switch(monkeypatch, False)
    _users()
    tokens = _login()
    now = datetime.now(timezone.utc)
    _lease_until(now - timedelta(minutes=40))
    _idle_for(tokens, 35)  # Deadline 5 minutes ago, after the lease ended.

    assert _verify(tokens).status_code == 200
    assert _refresh(tokens).status_code == 200


@pytest.mark.parametrize("offset_seconds,rejected", [(-1, True), (1, False)])
def test_switch_off_decision_at_the_lease_boundary(
    monkeypatch, offset_seconds, rejected
):
    _switch(monkeypatch, False)
    _users()
    tokens = _login()
    lease_end = datetime.now(timezone.utc) - timedelta(minutes=10)
    _lease_until(lease_end)
    deadline = lease_end + timedelta(seconds=offset_seconds)
    _set_activity(_sid(tokens), deadline - timedelta(minutes=30))

    response = _verify(tokens)
    if rejected:
        _assert_idle_rejection(response)
    else:
        assert response.status_code == 200


def test_legacy_refresh_tokens_stay_rejected_after_an_enforced_period(monkeypatch):
    """Rollout enables the policy only after pre-release access tokens expired;
    a pre-release refresh token the policy refused is not migrated later."""
    _switch(monkeypatch, False)
    _users()
    legacy = _legacy_tokens()
    _lease_until(datetime.now(timezone.utc) - timedelta(minutes=5))
    _assert_idle_rejection(_refresh(legacy))


def test_switch_off_judges_expiry_with_the_thresholds_then_in_force(monkeypatch):
    """Raising the threshold together with switch-off revives nothing."""
    _switch(monkeypatch, False)
    monkeypatch.setenv(ENV_MINUTES, "120")
    _users()
    tokens = _login()
    _lease_until(datetime.now(timezone.utc) - timedelta(minutes=5))
    _idle_for(tokens, 40)  # Expired under 30 minutes before the lease ended.
    _assert_idle_rejection(_verify(tokens))


def test_a_disabled_replicas_activity_write_cannot_revive(monkeypatch):
    """Enforcing replicas still run (lease ahead); this one is switched off."""
    _switch(monkeypatch, False)
    _users()
    tokens = _login()
    _lease_until(datetime.now(timezone.utc) + timedelta(seconds=10))
    idle_since = _idle_for(tokens, 31)
    # The read-side check passed on this replica; the write must still refuse.
    monkeypatch.setattr(browser_idle, "session_rejected", lambda *_a, **_k: False)
    _assert_idle_rejection(
        auth_tests.client.post("/api/auth/activity", headers=_bearer(tokens))
    )
    assert _as_utc(_row(_sid(tokens)).last_activity_at) == idle_since


def test_a_demotion_after_switch_off_revives_nothing(monkeypatch):
    """An admin login expired at 15 minutes; demotion would allow 30."""
    _switch(monkeypatch, False)
    _users()
    admin = _login("administrator")
    _lease_until(datetime.now(timezone.utc) - timedelta(minutes=2))
    _idle_for(admin, 20)
    with auth_tests.TestingSessionLocal() as session:
        session.query(User).filter_by(username="administrator").one().is_admin = False
        session.commit()
    _assert_idle_rejection(_verify(admin))


def test_a_promotion_while_enforced_is_recorded_without_activity(monkeypatch):
    """Promoted while on, idle, expired at 15 minutes, demoted after switch-off."""
    _switch(monkeypatch, True)
    _users()
    member = _login("member")
    with auth_tests.TestingSessionLocal() as session:
        session.query(User).filter_by(username="member").one().is_admin = True
        session.commit()
        browser_idle.renew_enforcement_lease(session)  # The enabled heartbeat.
        browser_idle.record_current_admin_roles(session)
        session.commit()
    _idle_for(member, 20)

    _switch(monkeypatch, False)
    with auth_tests.TestingSessionLocal() as session:
        session.query(User).filter_by(username="member").one().is_admin = False
        session.commit()
    _assert_idle_rejection(_verify(member))


def test_activity_records_the_current_role(enforced):
    _users()
    member = _login("member")
    _idle_for(member, 5)
    with auth_tests.TestingSessionLocal() as session:
        session.query(User).filter_by(username="member").one().is_admin = True
        session.commit()
    assert _row(_sid(member)).recorded_admin is False
    auth_tests.client.post("/api/auth/activity", headers=_bearer(member))
    assert _row(_sid(member)).recorded_admin is True


def test_each_threshold_set_applies_after_a_rolling_change(monkeypatch):
    """Replicas renewed 30-minute and 60-minute leases until switch-off."""
    _switch(monkeypatch, False)
    _users()
    expired_under_30 = _login()
    alive_under_both = _login("administrator")
    ended = datetime.now(timezone.utc) - timedelta(minutes=2)
    _lease_until(ended, ordinary_minutes=30)
    _lease_until(ended, ordinary_minutes=60)
    _idle_for(expired_under_30, 40)
    _idle_for(alive_under_both, 10)  # Admin threshold 15: alive at switch-off.

    _assert_idle_rejection(_verify(expired_under_30))
    assert _verify(alive_under_both).status_code == 200


def test_renewal_keeps_one_row_per_threshold_set(monkeypatch):
    _users()
    now = datetime.now(timezone.utc)
    with auth_tests.TestingSessionLocal() as session:
        browser_idle.renew_enforcement_lease(session, now=now)
        monkeypatch.setenv(ENV_MINUTES, "60")
        browser_idle.renew_enforcement_lease(session, now=now + timedelta(seconds=10))
        session.commit()
        leases = browser_idle.enforcement_leases(session)
    ends = {lease.ordinary_minutes: lease.until for lease in leases}
    lease = browser_idle.ENFORCEMENT_LEASE
    # The 60-minute replica's renewal does not extend the 30-minute set.
    assert ends == {30: now + lease, 60: now + timedelta(seconds=10) + lease}


def test_adapters_can_judge_switch_off_without_a_core_user():
    now = datetime.now(timezone.utc)
    lease = browser_idle.EnforcementLease(
        until=now - timedelta(minutes=2), ordinary_minutes=30, admin_minutes=15
    )
    judge = browser_idle.expired_while_enforced
    assert judge(
        [lease],
        last_activity_at=now - timedelta(minutes=40),
        roles=(False,),
        now=now,
    )
    assert not judge(
        [lease],
        last_activity_at=now - timedelta(minutes=20),
        roles=(False,),
        now=now,
    )


def test_the_lease_runs_ahead_of_now_and_never_moves_back():
    """Ahead, so it covers the moment an enforcing replica dies."""
    _users()
    now = datetime.now(timezone.utc)
    with auth_tests.TestingSessionLocal() as session:
        browser_idle.renew_enforcement_lease(session, now=now)
        browser_idle.renew_enforcement_lease(session, now=now - timedelta(hours=1))
        session.commit()
        until = browser_idle.enforced_until(session)
    assert until == now + browser_idle.ENFORCEMENT_LEASE
    assert browser_idle.ENFORCEMENT_LEASE > timedelta(
        seconds=browser_idle.ENFORCEMENT_HEARTBEAT_SECONDS
    )


@pytest.mark.parametrize("on", [True, False])
def test_the_heartbeat_renews_the_lease_only_while_enabled(monkeypatch, on):
    _switch(monkeypatch, on)
    _users()
    member = _login("member")
    with auth_tests.TestingSessionLocal() as session:
        session.query(User).filter_by(username="member").one().is_admin = True
        session.commit()

    async def one_tick():
        stop = asyncio.Event()
        task = asyncio.create_task(
            browser_idle.run_browser_session_sweep_loop(
                auth_tests.TestingSessionLocal, stop_event=stop
            )
        )
        await asyncio.sleep(0.2)
        stop.set()
        await task

    asyncio.run(one_tick())
    with auth_tests.TestingSessionLocal() as session:
        assert (browser_idle.enforced_until(session) is not None) is on
    # While enforced, the heartbeat also records the promotion.
    assert _row(_sid(member)).recorded_admin is on


def test_sweep_uses_each_users_server_side_threshold(enforced):
    _users()
    member = _login("member")
    admin = _login("administrator")
    _idle_for(member, 20)  # Within the ordinary 30 minutes.
    _idle_for(admin, 16)  # Past the platform-admin 15 minutes.
    with auth_tests.TestingSessionLocal() as session:
        revoked, _ = browser_idle.sweep_browser_sessions(session)
    assert revoked == 1
    assert _row(_sid(member)).revoked_at is None
    assert _row(_sid(admin)).revoked_at is not None


def test_sweep_never_revokes_while_disabled_and_prunes_unused_rows(monkeypatch):
    _switch(monkeypatch, False)
    _users()
    idle = _login()
    _idle_for(idle, 24 * 60)
    kept = _login("administrator")
    stale = _login("administrator")
    now = datetime.now(timezone.utc)
    with auth_tests.TestingSessionLocal() as session:
        # Pruned only once no token of the login can still be valid.
        session.get(BrowserSession, _sid(kept)).last_used_at = (
            now - browser_idle._RETENTION + timedelta(hours=1)
        )
        session.get(BrowserSession, _sid(stale)).last_used_at = (
            now - browser_idle._RETENTION - timedelta(hours=1)
        )
        session.commit()
        revoked, deleted = browser_idle.sweep_browser_sessions(session, now=now)
    assert (revoked, deleted) == (0, 1)
    assert _row(_sid(idle)).revoked_at is None
    assert _row(_sid(kept)) is not None
    assert _row(_sid(stale)) is None


def test_platform_admin_threshold_comes_from_the_server_role(enforced):
    _users()
    admin = _login("administrator")
    member = _login("member")
    _idle_for(admin, 16)
    _idle_for(member, 16)

    _assert_idle_rejection(_verify(admin))
    assert _verify(member).status_code == 200

    # Promotion is read from the database on the next request, not from claims.
    with auth_tests.TestingSessionLocal() as session:
        session.query(User).filter_by(username="member").one().is_admin = True
        session.commit()
    _assert_idle_rejection(_verify(member))


def test_configured_thresholds_apply(monkeypatch, enforced):
    monkeypatch.setenv(ENV_MINUTES, "5")
    _users()
    tokens = _login()
    _idle_for(tokens, 6)
    _assert_idle_rejection(_verify(tokens))


def test_a_second_login_keeps_the_first_device_on_its_own_deadline(enforced):
    _users()
    first = _login()
    second = _login()
    assert _verify(first).status_code == 200

    # The second device's activity does not keep the first one alive.
    _idle_for(first, 31)
    _assert_idle_rejection(_verify(first))
    assert _verify(second).status_code == 200


# --- Activity -----------------------------------------------------------------


def test_activity_extends_only_the_idle_deadline(enforced):
    _users()
    tokens = _login()
    _idle_for(tokens, 20)
    refresh_expiry_before = _user().refresh_token_expires_at

    response = auth_tests.client.post("/api/auth/activity", headers=_bearer(tokens))
    assert response.status_code == 200
    last = _as_utc(_row(_sid(tokens)).last_activity_at)
    assert datetime.now(timezone.utc) - last < timedelta(seconds=10)
    assert _user().refresh_token_expires_at == refresh_expiry_before
    assert response.json()["idle_session"]["idle_expires_in"] > 29 * 60

    _idle_for(tokens, 25)  # 25 minutes after the report: still within 30.
    assert _verify(tokens).status_code == 200


def test_activity_reports_are_coalesced(enforced):
    _users()
    tokens = _login()
    recent = datetime.now(timezone.utc) - timedelta(seconds=30)
    _set_activity(_sid(tokens), recent)

    response = auth_tests.client.post("/api/auth/activity", headers=_bearer(tokens))
    assert response.status_code == 200
    assert _as_utc(_row(_sid(tokens)).last_activity_at) == recent


def test_refresh_and_verify_are_not_activity(enforced):
    _users()
    tokens = _login()
    idle_since = _idle_for(tokens, 20)

    refreshed = _refresh(tokens)
    assert refreshed.status_code == 200
    assert refreshed.json()["idle_session"]["idle_expires_in"] <= 10 * 60
    assert _verify(tokens).status_code == 200
    assert _as_utc(_row(_sid(tokens)).last_activity_at) == idle_since


def test_late_activity_cannot_revive_an_expired_login(enforced):
    _users()
    tokens = _login()
    idle_since = _idle_for(tokens, 31)

    _assert_idle_rejection(
        auth_tests.client.post("/api/auth/activity", headers=_bearer(tokens))
    )
    assert _as_utc(_row(_sid(tokens)).last_activity_at) == idle_since


def test_activity_write_is_fenced_against_concurrent_expiry(monkeypatch, enforced):
    """The read-side check passed, then the deadline passed before the write."""
    _users()
    tokens = _login()
    idle_since = _idle_for(tokens, 31)
    monkeypatch.setattr(browser_idle, "session_rejected", lambda *_a, **_k: False)

    _assert_idle_rejection(
        auth_tests.client.post("/api/auth/activity", headers=_bearer(tokens))
    )
    assert _as_utc(_row(_sid(tokens)).last_activity_at) == idle_since


# The refresh UPDATE's own fence is exercised with a second worker on SQLite
# and PostgreSQL in test_sync_auth_worker.py.


# --- Logout ---------------------------------------------------------------------


def test_logout_revokes_only_its_own_login(enforced):
    _users()
    first = _login()
    second = _login()
    response = auth_tests.client.post(
        "/api/auth/logout",
        headers=_bearer(first),
        json={"refresh_token": first["refresh_token"]},
    )
    assert response.status_code == 200
    _assert_idle_rejection(_verify(first))
    assert _verify(second).status_code == 200
    assert _refresh(second).status_code == 200


def test_logout_from_a_stale_tab_still_revokes_its_own_login(enforced):
    """The first device's refresh token was replaced; its logout still counts."""
    _users()
    first = _login()
    second = _login()
    response = auth_tests.client.post(
        "/api/auth/logout",
        headers=_bearer(first),
        json={"refresh_token": first["refresh_token"]},
    )
    assert response.status_code == 200
    assert _row(_sid(first)).revoked_at is not None
    assert _refresh(second).status_code == 200


def _logout(tokens: dict[str, Any], body: dict[str, Any] | None = None):
    return auth_tests.client.post(
        "/api/auth/logout", headers=_bearer(tokens), json=body
    )


@pytest.mark.parametrize("on", [True, False])
def test_an_old_logins_logout_never_clears_a_newer_refresh_token(monkeypatch, on):
    """A signs in, B signs in, A logs out without a refresh token."""
    _switch(monkeypatch, on)
    _users()
    first = _login()
    second = _login()
    assert _logout(first).status_code == 200
    assert _refresh(second).status_code == 200


def test_logout_without_a_refresh_token_clears_the_logins_own():
    _users()
    tokens = _login()
    assert _logout(tokens).status_code == 200
    assert _user().refresh_token is None


def test_a_legacy_token_logout_keeps_revoking_all():
    """Compatibility exception: a token without sid cannot name its login."""
    _users()
    legacy = _legacy_tokens()
    assert _logout(legacy).status_code == 200
    assert _user().refresh_token is None


@pytest.mark.parametrize("presents_refresh_token", [False, True])
def test_a_newer_login_committed_during_logout_keeps_its_token(
    presents_refresh_token,
):
    """The ownership test runs in the UPDATE, not on the loaded user."""
    _users()
    first = _login()
    with auth_tests.TestingSessionLocal() as session:
        stale = session.query(User).filter_by(username="member").one()
        stale.refresh_token  # Loaded while the first login holds the token.
        second = _login()  # Another worker commits a newer login meanwhile.
        auth._clear_refresh_token_at_logout(
            session,
            stale,
            sid=_sid(first),
            presented=first["refresh_token"] if presents_refresh_token else None,
        )
        session.commit()
    assert _refresh(second).status_code == 200


def test_logout_is_unchanged_while_disabled():
    _users()
    tokens = _login()
    response = auth_tests.client.post(
        "/api/auth/logout",
        headers=_bearer(tokens),
        json={"refresh_token": tokens["refresh_token"]},
    )
    assert response.status_code == 200
    assert _row(_sid(tokens)).revoked_at is None
    assert _verify(tokens).status_code == 200  # As before: until JWT expiry.


# --- Activation and legacy credentials -------------------------------------------


def _legacy_tokens() -> dict[str, str]:
    """Credentials issued by a release that predates browser sessions."""
    with auth_tests.TestingSessionLocal() as session:
        user = session.query(User).filter_by(username="member").one()
        access = auth.create_access_token(user_session_claims(user))
        refresh = auth.create_refresh_token(user_session_claims(user))
        user.refresh_token = hash_refresh_token(refresh)
        user.refresh_token_expires_at = datetime.now(timezone.utc) + timedelta(days=1)
        user.browser_session_id = None
        session.commit()
    return {"access_token": access, "refresh_token": refresh}


def test_legacy_credentials_are_rejected_once_enforced(enforced):
    _users()
    legacy = _legacy_tokens()
    _assert_idle_rejection(_verify(legacy))
    _assert_idle_rejection(_refresh(legacy))


def test_refresh_moves_a_legacy_login_onto_a_browser_session_while_disabled():
    _users()
    legacy = _legacy_tokens()
    assert _user().browser_session_id is None  # Counted as not yet migrated.

    refreshed = _refresh(legacy)
    assert refreshed.status_code == 200
    sid = _sid(refreshed.json())
    assert _user().browser_session_id == sid
    assert _row(sid) is not None


def test_activity_with_a_legacy_token_is_harmless_while_disabled():
    _users()
    legacy = _legacy_tokens()
    response = auth_tests.client.post("/api/auth/activity", headers=_bearer(legacy))
    assert response.status_code == 200
    assert response.json()["idle_session"]["sid"] is None


def test_a_rejected_login_keeps_its_rejection_reason(monkeypatch):
    """Every later request of a rejected sid says SessionIdleExpired."""
    _switch(monkeypatch, True)
    _users()
    tokens = _login()
    _idle_for(tokens, 31)
    for _ in range(2):
        _assert_idle_rejection(_verify(tokens))
        _assert_idle_rejection(_refresh(tokens))
        _assert_idle_rejection(
            auth_tests.client.post("/api/auth/activity", headers=_bearer(tokens))
        )


def test_adapter_helpers_need_no_core_user(enforced):
    now = datetime.now(timezone.utc)
    idle = browser_idle.build_idle_session(
        is_platform_admin=False,
        sid="adapter-login",
        last_activity_at=now - timedelta(minutes=10),
        now=now,
        absolute_expires_at=now + timedelta(hours=2),
    )
    assert idle["idle_timeout_seconds"] == browser_idle.idle_timeout_seconds(False)
    assert idle["idle_expires_in"] == 20 * 60
    assert idle["absolute_expires_in"] == 2 * 60 * 60
    error = browser_idle.rejection_http_exception()
    assert error.status_code == 401
    assert error.headers["Error-Type"] == "SessionIdleExpired"


def test_adapter_refresh_moves_a_legacy_login_while_disabled():
    """An adapter's own refresh path migrates like core refresh does."""
    _users()
    _legacy_tokens()
    with auth_tests.TestingSessionLocal() as session:
        user = session.query(User).filter_by(username="member").one()
        sid, last_activity = browser_idle.check_browser_session(session, user, None)
        assert last_activity is None  # A new login starts now.
        browser_idle.begin_browser_session(session, int(user.id), sid=sid)
        session.commit()
    assert _user().browser_session_id == sid
    assert _row(sid) is not None


def test_adapter_refresh_of_an_unknown_sid_while_disabled_starts_a_login():
    _users()
    with auth_tests.TestingSessionLocal() as session:
        user = session.query(User).filter_by(username="member").one()
        sid, last_activity = browser_idle.check_browser_session(
            session, user, "pruned-or-unknown-sid"
        )
    assert last_activity is None and sid != "pruned-or-unknown-sid"


@pytest.mark.parametrize("presented_sid", [None, "unknown-sid"])
def test_adapter_refresh_without_a_known_sid_is_rejected_once_enforced(
    enforced, presented_sid
):
    _users()
    with auth_tests.TestingSessionLocal() as session:
        user = session.query(User).filter_by(username="member").one()
        with pytest.raises(browser_idle.BrowserSessionRejected):
            browser_idle.check_browser_session(session, user, presented_sid)


def test_revoking_an_old_login_does_not_touch_a_newer_one(enforced):
    _users()
    old = _login()
    new = _login()
    with auth_tests.TestingSessionLocal() as session:
        browser_idle.revoke_browser_session(session, _sid(old), reason="idle")
        session.commit()
    assert _row(_sid(new)).revoked_at is None
    assert _verify(new).status_code == 200


# --- Other browser credentials ---------------------------------------------------


def _ticket_for(tokens: dict[str, Any]) -> str:
    with auth_tests.TestingSessionLocal() as session:
        user = session.query(User).filter_by(username="member").one()
        # As token resolution does for the request that mints the ticket.
        browser_idle.remember_request_session(user, _sid(tokens))
        ticket = asyncio.run(files.issue_preview_stream_ticket("example-file", user))
    return ticket["path"].split("ticket=", 1)[1]


def test_file_stream_ticket_is_rejected_after_idle_expiry(enforced):
    _users()
    tokens = _login()
    raw_ticket = _ticket_for(tokens)
    assert _claims(raw_ticket)["sid"] == _sid(tokens)

    with auth_tests.TestingSessionLocal() as session:
        assert files._user_from_stream_ticket(session, raw_ticket, "example-file")
    _idle_for(tokens, 31)
    with auth_tests.TestingSessionLocal() as session:
        with pytest.raises(HTTPException) as error:
            files._user_from_stream_ticket(session, raw_ticket, "example-file")
    assert error.value.status_code == 401


def test_connected_websocket_closes_with_the_idle_code(monkeypatch, enforced):
    _users()
    tokens = _login()
    monkeypatch.setattr(websocket_auth, "_SESSION_RECHECK_SECONDS", 0.01)
    progress_manager = MagicMock()
    progress_manager.get_task_progress.return_value = None
    monkeypatch.setattr(progress_ws, "get_progress_manager", lambda: progress_manager)
    broadcaster = MagicMock()
    broadcaster.connect = AsyncMock()
    broadcaster.disconnect = AsyncMock()
    monkeypatch.setattr(progress_ws, "progress_broadcaster", broadcaster)
    monkeypatch.setattr(websocket_api, "manager", MagicMock())
    app = FastAPI()
    app.include_router(progress_ws.progress_ws_router)
    with TestClient(app) as client:
        with client.websocket_connect(
            "/ws/progress/task",
            subprotocols=[
                "xagent-auth-v1",
                f"xagent-auth-token.{tokens['access_token']}",
            ],
        ) as socket:
            _idle_for(tokens, 31)
            message = socket.portal.call(asyncio.wait_for, socket._send_rx.receive(), 5)
            assert message["type"] == "websocket.close"
            assert message["code"] == 4010
