"""Server-enforced idle expiry for interactive browser logins.

Every login gets a random ``sid`` and a ``browser_sessions`` row. Tokens and
tickets issued for the login carry the ``sid``. The row's ``last_activity_at``
moves only on login and on an explicit ``POST /api/auth/activity`` report;
refresh, polling, keepalives and other requests never move it.

With the policy enabled, a credential is rejected once its login is past
``last activity + threshold``, has no row, or carries no ``sid`` (issued before
this release). An expired login is revoked on its row, and a revoked row is
rejected whether or not the policy is currently enabled, so switching the
policy off does not revive it. Revocations are written only while the policy
is enabled. Rows live in the database, so all workers see the same state and
it survives restarts; writes are conditional UPDATEs that repeat the deadline
test, so a report or refresh racing with expiry cannot revive a login.

Refresh-token storage is unchanged: one per user, recorded against the login
that holds it in ``users.browser_session_id``. See docs/browser-idle-timeout.md.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, cast

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .auth_config import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS,
    BrowserIdlePolicy,
    get_browser_idle_policy,
)
from .models.browser_session import BrowserSession
from .models.user import User
from .services.writer_workers import run_tracked

logger = logging.getLogger(__name__)

SESSION_ID_CLAIM = "sid"
IDLE_EXPIRED_ERROR_TYPE = "SessionIdleExpired"
IDLE_WEBSOCKET_CLOSE_CODE = 4010
# Activity writes per login are bounded to one per interval; the deadline is
# therefore accurate to within it.
ACTIVITY_WRITE_INTERVAL = timedelta(seconds=60)
SWEEP_INTERVAL_SECONDS = 60.0
# A row unused for longer than this cannot back any valid token: its refresh
# token was last rotated no later than last_used_at, and every access token
# expires within its own lifetime.
_RETENTION = (
    timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    + timedelta(days=1)
)
# The authenticated request's sid, set on the User by token resolution.
_REQUEST_SID_ATTR = "_xagent_browser_session_id"


class BrowserSessionRejected(Exception):
    """The login is idle-expired, revoked, unknown or predates browser sessions."""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    # SQLite returns naive values for DateTime(timezone=True); they are UTC.
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def new_browser_session_id() -> str:
    """Exactly 32 URL-safe characters for browser_sessions.sid."""
    return secrets.token_urlsafe(24)


def browser_session_claims(sid: str) -> dict[str, str]:
    """Claims that bind a token or ticket to one login."""
    return {SESSION_ID_CLAIM: sid}


def claimed_session_id(claims: Mapping[str, Any]) -> str | None:
    value = claims.get(SESSION_ID_CLAIM)
    if type(value) is str and value and value.isascii() and len(value) <= 32:
        return value
    return None


def remember_request_session(user: User, sid: str | None) -> None:
    """Record which login authenticated this request on the loaded User."""
    setattr(user, _REQUEST_SID_ATTR, sid)


def request_session_id(user: User) -> str | None:
    """The sid of the access token that authenticated ``user`` in this request."""
    return cast(str | None, getattr(user, _REQUEST_SID_ATTR, None))


def request_session_claims(user: User) -> dict[str, str]:
    """Claims for a ticket minted on behalf of the current request's login."""
    sid = request_session_id(user)
    return browser_session_claims(sid) if sid else {}


def idle_timeout(user: User, policy: BrowserIdlePolicy) -> timedelta:
    """The server-side role decides the threshold, never a client claim."""
    return timedelta(seconds=policy.timeout_seconds(bool(user.is_admin)))


def _load(db: Session, user: User, sid: str | None) -> BrowserSession | None:
    if sid is None:
        return None
    row = db.get(BrowserSession, sid)
    if row is None or row.user_id != user.id:
        return None
    return row


def rejection_http_exception() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Session expired due to inactivity",
        headers={"WWW-Authenticate": "Bearer", "Error-Type": IDLE_EXPIRED_ERROR_TYPE},
    )


def revoke_browser_session(
    db: Session,
    sid: str,
    *,
    reason: str,
    user_id: int | None = None,
    now: datetime | None = None,
) -> bool:
    """Mark one login revoked; the caller commits. Never touches other logins."""
    query = db.query(BrowserSession).filter(
        BrowserSession.sid == sid, BrowserSession.revoked_at.is_(None)
    )
    if user_id is not None:
        query = query.filter(BrowserSession.user_id == user_id)
    changed = query.update(
        {
            BrowserSession.revoked_at: now or utcnow(),
            BrowserSession.revoked_reason: reason,
        },
        synchronize_session=False,
    )
    return changed == 1


def _revoke_in_new_transaction(db: Session, sid: str, reason: str) -> None:
    """Persist a revocation seen by a read-only request.

    Uses its own Session on the same engine so the caller's transaction is
    neither committed nor rolled back. Best effort: the request is rejected
    either way, and the sweep revokes the login if this write fails. Two
    limits: it takes a second pooled connection while the request holds one,
    and if the caller's Session is bound to a Connection rather than an
    Engine, the write joins that connection's transaction and is lost if the
    caller rolls back.
    """
    try:
        with Session(bind=db.get_bind()) as revoking:
            if revoke_browser_session(revoking, sid, reason=reason):
                revoking.commit()
    except Exception:
        logger.warning("Could not revoke an idle-expired browser login", exc_info=True)


def _rejects(
    db: Session,
    user: User,
    sid: str | None,
    row: BrowserSession | None,
    *,
    now: datetime | None,
    policy: BrowserIdlePolicy,
) -> bool:
    if row is not None and row.revoked_at is not None:
        return True
    if not policy.enabled:
        return False
    if row is None:
        return True
    deadline = _as_utc(cast(datetime, row.last_activity_at)) + idle_timeout(
        user, policy
    )
    if deadline <= (now or utcnow()):
        assert sid is not None
        _revoke_in_new_transaction(db, sid, "idle")
        return True
    return False


def session_rejected(
    db: Session,
    user: User,
    claims: Mapping[str, Any],
    *,
    now: datetime | None = None,
    policy: BrowserIdlePolicy | None = None,
) -> bool:
    """Whether verified ``claims`` may no longer act for ``user``.

    A revoked login is rejected even while the policy is disabled; otherwise a
    disabled policy rejects nothing, which keeps existing behaviour.
    """
    sid = claimed_session_id(claims)
    return _rejects(
        db,
        user,
        sid,
        _load(db, user, sid),
        now=now,
        policy=policy or get_browser_idle_policy(),
    )


def add_browser_session(
    db: Session, user_id: int, sid: str, *, now: datetime | None = None
) -> None:
    """Insert the row for a new login; the caller commits."""
    now = now or utcnow()
    db.add(
        BrowserSession(
            sid=sid,
            user_id=user_id,
            created_at=now,
            last_activity_at=now,
            last_used_at=now,
        )
    )


def begin_browser_session(
    db: Session,
    user_id: int,
    *,
    sid: str | None = None,
    now: datetime | None = None,
) -> str:
    """Start a login for an adapter that signs its own tokens.

    Inserts the row and records it as the holder of the user's refresh token.
    The caller commits in the same transaction that stores that refresh token,
    and puts ``browser_session_claims(sid)`` into every token it issues. Pass
    the ``sid`` that ``check_browser_session`` returned with no activity time to
    move a pre-release refresh token onto a login.
    """
    sid = sid or new_browser_session_id()
    add_browser_session(db, user_id, sid, now=now)
    db.query(User).filter(User.id == user_id).update(
        {User.browser_session_id: sid}, synchronize_session=False
    )
    return sid


def check_browser_session(
    db: Session,
    user: User,
    presented_sid: str | None,
    *,
    policy: BrowserIdlePolicy | None = None,
) -> tuple[str, datetime | None]:
    """Check a login before refresh: its sid and last activity.

    Raises BrowserSessionRejected for a rejected login; refresh never extends
    the idle deadline. A ``None`` activity time means a new login starts now:
    while the policy is disabled, a refresh token without a known sid starts
    one, so logins already carry it when the policy is switched on.
    """
    row = _load(db, user, presented_sid)
    if _rejects(
        db,
        user,
        presented_sid,
        row,
        now=None,
        policy=policy or get_browser_idle_policy(),
    ):
        raise BrowserSessionRejected()
    if row is not None:
        assert presented_sid is not None
        return presented_sid, _as_utc(cast(datetime, row.last_activity_at))
    return new_browser_session_id(), None


def fence_browser_session_for_refresh(
    db: Session,
    user: User,
    sid: str,
    *,
    now: datetime | None = None,
    policy: BrowserIdlePolicy | None = None,
) -> bool:
    """Confirm, inside the refresh transaction, that the login is still valid.

    A conditional UPDATE on the login's row, run before the ``users`` UPDATE so
    the lock order matches logout. Deleting a user locks ``users`` first and
    then cascades to its rows, so on PostgreSQL a refresh racing that delete
    can deadlock; the database aborts one of them, which is retry-safe.
    Records the use for retention; never moves the idle deadline.
    """
    policy = policy or get_browser_idle_policy()
    now = now or utcnow()
    conditions = [
        BrowserSession.sid == sid,
        BrowserSession.user_id == user.id,
        BrowserSession.revoked_at.is_(None),
    ]
    if policy.enabled:
        conditions.append(
            BrowserSession.last_activity_at > now - idle_timeout(user, policy)
        )
    changed = (
        db.query(BrowserSession)
        .filter(*conditions)
        .update({BrowserSession.last_used_at: now}, synchronize_session=False)
    )
    return changed == 1


def record_activity(
    db: Session,
    user: User,
    sid: str | None,
    *,
    now: datetime | None = None,
    policy: BrowserIdlePolicy | None = None,
) -> datetime | None:
    """Record a user-initiated activity report; returns the login's last activity.

    Raises BrowserSessionRejected when the login expired before the write could
    land: the UPDATE repeats the deadline test, so a report racing expiry
    cannot revive the login. Reports inside ``ACTIVITY_WRITE_INTERVAL`` are
    coalesced. Returns ``None`` for a pre-release login while disabled.
    """
    policy = policy or get_browser_idle_policy()
    now = now or utcnow()
    row = _load(db, user, sid)
    if row is None or row.revoked_at is not None:
        if policy.enabled or row is not None:
            raise BrowserSessionRejected()
        return None
    last = _as_utc(cast(datetime, row.last_activity_at))
    if now - last < ACTIVITY_WRITE_INTERVAL:
        return last
    conditions = [BrowserSession.sid == sid, BrowserSession.revoked_at.is_(None)]
    if policy.enabled:
        conditions.append(
            BrowserSession.last_activity_at > now - idle_timeout(user, policy)
        )
    changed = (
        db.query(BrowserSession)
        .filter(*conditions)
        .update(
            {BrowserSession.last_activity_at: now, BrowserSession.last_used_at: now},
            synchronize_session=False,
        )
    )
    if changed != 1:
        db.rollback()
        raise BrowserSessionRejected()
    db.commit()
    return now


def last_activity(db: Session, user: User, sid: str | None) -> datetime | None:
    row = _load(db, user, sid)
    if row is None:
        return None
    return _as_utc(cast(datetime, row.last_activity_at))


def idle_timeout_seconds(
    is_platform_admin: bool, *, policy: BrowserIdlePolicy | None = None
) -> int:
    """The threshold for a role; for adapters that have no core User."""
    return (policy or get_browser_idle_policy()).timeout_seconds(is_platform_admin)


def build_idle_session(
    *,
    is_platform_admin: bool,
    sid: str | None,
    last_activity_at: datetime | None,
    now: datetime | None = None,
    absolute_expires_at: datetime | None = None,
    policy: BrowserIdlePolicy | None = None,
) -> dict[str, Any]:
    """The ``idle_session`` response object; needs no core User."""
    policy = policy or get_browser_idle_policy()
    now = now or utcnow()
    timeout = timedelta(seconds=policy.timeout_seconds(is_platform_admin))
    if last_activity_at is None:
        expires_in = 0
    else:
        remaining = _as_utc(last_activity_at) + timeout - now
        expires_in = max(0, int(remaining.total_seconds()))
    absolute_expires_in = None
    if absolute_expires_at is not None:
        absolute_expires_in = max(
            0, int((_as_utc(absolute_expires_at) - now).total_seconds())
        )
    return {
        "enforced": policy.enabled,
        "sid": sid,
        "idle_timeout_seconds": int(timeout.total_seconds()),
        "idle_expires_in": expires_in,
        "server_time": now.isoformat().replace("+00:00", "Z"),
        "absolute_expires_in": absolute_expires_in,
        "activity_min_interval_seconds": int(ACTIVITY_WRITE_INTERVAL.total_seconds()),
    }


def idle_session_fields(
    user: User,
    sid: str | None,
    *,
    last_activity_at: datetime | None,
    now: datetime | None = None,
    absolute_expires_at: datetime | None = None,
    policy: BrowserIdlePolicy | None = None,
) -> dict[str, Any]:
    """``build_idle_session`` for a core User; the role comes from the database."""
    return build_idle_session(
        is_platform_admin=bool(user.is_admin),
        sid=sid,
        last_activity_at=last_activity_at,
        now=now,
        absolute_expires_at=absolute_expires_at,
        policy=policy,
    )


def sweep_browser_sessions(
    db: Session,
    *,
    now: datetime | None = None,
    policy: BrowserIdlePolicy | None = None,
) -> tuple[int, int]:
    """Revoke idle-expired logins (policy enabled only) and prune unused rows.

    Idempotent and safe to run on every replica at once. Covers logins whose
    browser never sends another request. Returns (revoked, deleted).
    """
    policy = policy or get_browser_idle_policy()
    now = now or utcnow()
    revoked = 0
    if policy.enabled:
        for is_admin in (True, False):
            cutoff = now - timedelta(seconds=policy.timeout_seconds(is_admin))
            role_users = select(User.id).where(User.is_admin.is_(is_admin))
            revoked += (
                db.query(BrowserSession)
                .filter(
                    BrowserSession.revoked_at.is_(None),
                    BrowserSession.last_activity_at <= cutoff,
                    BrowserSession.user_id.in_(role_users),
                )
                .update(
                    {
                        BrowserSession.revoked_at: now,
                        BrowserSession.revoked_reason: "idle",
                    },
                    synchronize_session=False,
                )
            )
    deleted = (
        db.query(BrowserSession)
        .filter(BrowserSession.last_used_at < now - _RETENTION)
        .delete(synchronize_session=False)
    )
    db.commit()
    return revoked, deleted


async def run_browser_session_sweep_loop(
    session_factory: sessionmaker[Session],
    *,
    stop_event: asyncio.Event,
    interval_seconds: float = SWEEP_INTERVAL_SECONDS,
) -> None:
    """Sweep every interval until stopped; a failed sweep is logged and retried.

    Each sweep writes, so it runs through ``run_tracked`` and shutdown's
    ``retire_writers`` waits for an in-flight sweep.
    """

    def sweep() -> None:
        with session_factory() as db:
            revoked, deleted = sweep_browser_sessions(db)
        if revoked or deleted:
            logger.info("Browser session sweep revoked=%d deleted=%d", revoked, deleted)

    while not stop_event.is_set():
        try:
            await run_tracked(sweep)
        except Exception:
            logger.exception("Browser session sweep failed")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            pass
