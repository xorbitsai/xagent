"""Server-enforced idle expiry for interactive browser logins.

Every login gets a random ``sid`` and a ``browser_sessions`` row. Tokens and
tickets issued for the login carry the ``sid``. The row's ``last_activity_at``
moves only on login and on an explicit ``POST /api/auth/activity`` report;
refresh, polling, keepalives and other requests never move it.

With the policy enabled, a credential is rejected once its login is past
``last activity + threshold``, has no row, or carries no ``sid`` (issued before
this release). A login that reached an enforced deadline stays expired: every
decision also checks the recorded threshold sets and roles, so a later threshold
increase, demotion or switch-off cannot re-admit it. Requests only reject; the
sweep and logout revoke a login's row, and a revoked row is rejected whether or
not the policy is enabled.

Switching the policy off must not revive a login that expired while it was
enforced, including one no request saw expire. Replicas with the policy
enabled keep renewing an enforcement lease (``browser_idle_enforcement``). With
the policy disabled, a login whose deadline passed before the lease ended is
still rejected; a login alive at switch-off is free at once.
Deployments that never enable the policy have no lease and behave as before.

Rows live in the database, so all workers see the same state and it survives
restarts; writes are conditional UPDATEs that repeat the deadline test, so a
report or refresh racing with expiry cannot revive a login. Refresh-token
storage is unchanged: one per user, recorded against the login that holds it in
``users.browser_session_id``. See docs/browser-idle-timeout.md.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, cast

from fastapi import HTTPException, status
from sqlalchemy import and_, case, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from .auth_config import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS,
    BrowserIdlePolicy,
    get_browser_idle_policy,
)
from .models.browser_session import BrowserIdleEnforcement, BrowserSession
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
# Replicas with the policy enabled renew the enforcement lease every interval,
# each time to now + lease. The lease outlasts the interval, so the recorded
# end still covers the moment an enforcing replica stops, even by crashing;
# the margin also absorbs small clock differences between replicas.
ENFORCEMENT_HEARTBEAT_SECONDS = 5.0
ENFORCEMENT_LEASE = timedelta(seconds=15)
_MIN_THRESHOLD = timedelta(minutes=1)
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


@dataclass(frozen=True)
class EnforcementLease:
    """One threshold set and how long the policy was enforced with it."""

    until: datetime
    ordinary_minutes: int
    admin_minutes: int

    def timeout(self, is_platform_admin: bool) -> timedelta:
        minutes = self.admin_minutes if is_platform_admin else self.ordinary_minutes
        return timedelta(minutes=minutes)

    def strictest_timeout(self, *roles: bool) -> timedelta:
        return min(self.timeout(role) for role in roles)


def enforcement_leases(db: Session) -> list[EnforcementLease]:
    """Every recorded threshold set; empty if the policy was never enforced.

    Read on every decision that needs it (no cache): a cached value would lag
    the recorded time. One small query over a handful of rows.
    """
    return [
        EnforcementLease(
            until=_as_utc(cast(datetime, row.enforced_until)),
            ordinary_minutes=int(row.ordinary_minutes),
            admin_minutes=int(row.admin_minutes),
        )
        for row in db.query(BrowserIdleEnforcement).all()
    ]


def _active_leases(db: Session, now: datetime) -> list[EnforcementLease]:
    """Threshold sets some replica is still enforcing (lease not yet ended)."""
    return [lease for lease in enforcement_leases(db) if lease.until > now]


def enforced_until(db: Session) -> datetime | None:
    """The latest lease end over all threshold sets, or None if never enforced."""
    leases = enforcement_leases(db)
    return max((lease.until for lease in leases), default=None)


def renew_enforcement_lease(
    db: Session,
    *,
    now: datetime | None = None,
    policy: BrowserIdlePolicy | None = None,
) -> None:
    """Record that this replica's thresholds are enforced until now + lease.

    Called by every replica with the policy enabled; each threshold set has its
    own row, so replicas running different thresholds during a rolling change
    do not overwrite each other. Never moves a row back. The caller commits.
    """
    policy = policy or get_browser_idle_policy()
    until = (now or utcnow()) + ENFORCEMENT_LEASE
    key = (policy.ordinary_minutes, policy.admin_minutes)
    changed = (
        db.query(BrowserIdleEnforcement)
        .filter(
            BrowserIdleEnforcement.ordinary_minutes == key[0],
            BrowserIdleEnforcement.admin_minutes == key[1],
            BrowserIdleEnforcement.enforced_until < until,
        )
        .update(
            {BrowserIdleEnforcement.enforced_until: until}, synchronize_session=False
        )
    )
    if changed == 0 and db.get(BrowserIdleEnforcement, key) is None:
        nested = db.begin_nested()
        try:
            db.add(
                BrowserIdleEnforcement(
                    ordinary_minutes=key[0], admin_minutes=key[1], enforced_until=until
                )
            )
            nested.commit()
        except IntegrityError:
            # Another replica created the row first; its lease is current.
            nested.rollback()


def record_current_admin_roles(db: Session) -> int:
    """Mark live logins of current platform admins as admin; the caller commits.

    Run by enabled replicas with each lease renewal, so a promotion while the
    policy is enforced is recorded even if the login sees no further activity;
    a demotion after switch-off then cannot loosen the threshold.
    """
    return (
        db.query(BrowserSession)
        .filter(
            BrowserSession.revoked_at.is_(None),
            BrowserSession.recorded_admin.is_(False),
            BrowserSession.user_id.in_(select(User.id).where(User.is_admin.is_(True))),
        )
        .update({BrowserSession.recorded_admin: True}, synchronize_session=False)
    )


def expired_while_enforced(
    leases: list[EnforcementLease],
    *,
    last_activity_at: datetime,
    roles: tuple[bool, ...],
    now: datetime,
) -> bool:
    """Whether a login expired while the policy was enforced (policy now off).

    True if, under any recorded threshold set and the strictest of ``roles``,
    its deadline has passed and fell before that set's lease ended. Needs no
    core User, so adapters that keep their own session state can use it with
    ``enforcement_leases(db)``.
    """
    last = _as_utc(last_activity_at)
    for lease in leases:
        deadline = last + lease.strictest_timeout(*roles)
        if deadline <= now and deadline < lease.until:
            return True
    return False


def _lease_conditions(
    leases: list[EnforcementLease], *, current: bool, now: datetime
) -> list[Any]:
    """SQL: not expired under any recorded set while it was enforced.

    Uses the stricter of the row's recorded role and the current role.
    """
    conditions = []
    for lease in leases:
        cutoff_end = min(now, lease.until)
        conditions.append(
            BrowserSession.last_activity_at
            > case(
                (
                    BrowserSession.recorded_admin.is_(True),
                    cutoff_end - lease.strictest_timeout(True, current),
                ),
                else_=cutoff_end - lease.strictest_timeout(False, current),
            )
        )
    return conditions


def _alive_condition(
    db: Session, user: User, *, now: datetime, policy: BrowserIdlePolicy
) -> Any | None:
    """SQL condition that the login has not expired, for conditional writes.

    A login that reached an enforced deadline stays expired: besides this
    replica's current threshold (when enabled), it must not have expired under
    any recorded threshold set while that set was enforced, judged with the
    stricter of the recorded and current role. So no write can revive it after a
    threshold increase, a demotion, a rolling change or switch-off. Never
    enforced and disabled: no condition.
    """
    current = bool(user.is_admin)
    conditions = _lease_conditions(enforcement_leases(db), current=current, now=now)
    if policy.enabled:
        conditions.append(
            BrowserSession.last_activity_at > now - idle_timeout(user, policy)
        )
    return and_(*conditions) if conditions else None


def _expired(
    db: Session,
    user: User,
    last: datetime,
    recorded_admin: bool,
    *,
    now: datetime,
    policy: BrowserIdlePolicy,
) -> bool:
    """Whether the login has reached an enforced idle deadline.

    Enabled: past this replica's current threshold. In any state: expired
    under a recorded threshold set while that set was still enforced, with the
    stricter of the recorded and current role. The second test keeps a login
    expired after a later threshold increase, a demotion, a rolling change or
    switch-off, without any request-time write. No threshold is below a
    minute, so a login active within the last minute skips the lease read.
    """
    if policy.enabled and last + idle_timeout(user, policy) <= now:
        return True
    if last > now - _MIN_THRESHOLD:
        return False
    return expired_while_enforced(
        enforcement_leases(db),
        last_activity_at=last,
        roles=(recorded_admin, bool(user.is_admin)),
        now=now,
    )


def _rejects(
    db: Session,
    user: User,
    sid: str | None,
    row: BrowserSession | None,
    *,
    now: datetime | None,
    policy: BrowserIdlePolicy,
) -> bool:
    """Whether the login is rejected; never writes.

    Expiry is not revoked here: durability comes from the enforcement lease
    and the sweep, whose conditional UPDATE decides and writes in one
    statement. A request reads no extra connection, and a stale row cannot
    revoke a login another request has just kept alive.
    """
    if row is not None and row.revoked_at is not None:
        return True
    if row is None:
        # No row (no sid, unknown or pruned): rejected while enforced; with
        # the policy off it is a pre-release login. Rollout enables the policy
        # only after every pre-release access token has expired; refresh
        # tokens are handled in check_browser_session.
        return policy.enabled
    now = now or utcnow()
    last = _as_utc(cast(datetime, row.last_activity_at))
    if not _expired(db, user, last, bool(row.recorded_admin), now=now, policy=policy):
        return False
    # Re-read the committed state before rejecting: the loaded row may predate
    # an activity report that kept the login alive.
    fresh = db.execute(
        select(
            BrowserSession.last_activity_at,
            BrowserSession.revoked_at,
            BrowserSession.recorded_admin,
        ).where(BrowserSession.sid == row.sid)
    ).one_or_none()
    if fresh is None:
        return policy.enabled
    if fresh.revoked_at is not None:
        return True
    return _expired(
        db,
        user,
        _as_utc(fresh.last_activity_at),
        bool(fresh.recorded_admin),
        now=now,
        policy=policy,
    )


def session_rejected(
    db: Session,
    user: User,
    claims: Mapping[str, Any],
    *,
    now: datetime | None = None,
    policy: BrowserIdlePolicy | None = None,
) -> bool:
    """Whether verified ``claims`` may no longer act for ``user``.

    While the policy is disabled, only revoked logins and logins that expired
    while it was still enforced are rejected; with no enforcement lease
    recorded, nothing is, which keeps existing behaviour.
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
    """Insert the row for a new login, recording the user's current role.

    One INSERT (the role comes from a subquery); the caller commits.
    """
    now = now or utcnow()
    db.execute(
        insert(BrowserSession).values(
            sid=sid,
            user_id=user_id,
            created_at=now,
            last_activity_at=now,
            last_used_at=now,
            recorded_admin=select(User.is_admin)
            .where(User.id == user_id)
            .scalar_subquery(),
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
    policy = policy or get_browser_idle_policy()
    row = _load(db, user, presented_sid)
    if _rejects(db, user, presented_sid, row, now=None, policy=policy):
        raise BrowserSessionRejected()
    if row is None and enforcement_leases(db):
        # A refresh token the policy refused while enforced is not migrated
        # after it is switched off.
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
    alive = _alive_condition(db, user, now=now, policy=policy)
    if alive is not None:
        conditions.append(alive)
    changed = (
        db.query(BrowserSession)
        .filter(*conditions)
        .update(
            {
                BrowserSession.last_used_at: now,
                BrowserSession.recorded_admin: bool(user.is_admin),
            },
            synchronize_session=False,
        )
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
    conditions = [
        BrowserSession.sid == sid,
        BrowserSession.revoked_at.is_(None),
        # Coalescing is part of the atomic predicate, so concurrent reports
        # write at most once per interval and never move the time backward.
        BrowserSession.last_activity_at <= now - ACTIVITY_WRITE_INTERVAL,
    ]
    alive = _alive_condition(db, user, now=now, policy=policy)
    if alive is not None:
        conditions.append(alive)
    changed = (
        db.query(BrowserSession)
        .filter(*conditions)
        .update(
            {
                BrowserSession.last_activity_at: now,
                BrowserSession.last_used_at: now,
                BrowserSession.recorded_admin: bool(user.is_admin),
            },
            synchronize_session=False,
        )
    )
    if changed == 1:
        db.commit()
        return now
    db.rollback()
    # Nothing written: another report got there first (coalesced), or the
    # login expired or was revoked. Decide from the committed state.
    current = _load(db, user, sid)
    if current is None or _rejects(db, user, sid, current, now=now, policy=policy):
        raise BrowserSessionRejected()
    return _as_utc(cast(datetime, current.last_activity_at))


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
    timeout: timedelta | None = None,
) -> dict[str, Any]:
    """The ``idle_session`` response object; needs no core User.

    ``timeout`` overrides the policy threshold, e.g. with the effective one
    during a rolling threshold change.
    """
    policy = policy or get_browser_idle_policy()
    now = now or utcnow()
    if timeout is None:
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


def effective_idle_timeout(
    db: Session,
    *,
    is_platform_admin: bool,
    now: datetime | None = None,
    policy: BrowserIdlePolicy | None = None,
) -> timedelta:
    """The threshold admission enforces for a role; needs no core User.

    With the policy enabled, the stricter of this replica's threshold and every
    set still being enforced (a rolling change). Pass it to
    ``build_idle_session(timeout=...)``.
    """
    policy = policy or get_browser_idle_policy()
    own = timedelta(seconds=policy.timeout_seconds(is_platform_admin))
    if not policy.enabled:
        return own
    leases = _active_leases(db, now or utcnow())
    return min([own] + [lease.timeout(is_platform_admin) for lease in leases])


def idle_session_fields(
    user: User,
    sid: str | None,
    *,
    last_activity_at: datetime | None,
    now: datetime | None = None,
    absolute_expires_at: datetime | None = None,
    policy: BrowserIdlePolicy | None = None,
    db: Session | None = None,
) -> dict[str, Any]:
    """``build_idle_session`` for a core User; the role comes from the database.

    With ``db``, the reported threshold is the one admission enforces: the
    stricter of this replica's and every threshold set still being enforced,
    and of the login's recorded and the user's current role.
    """
    policy = policy or get_browser_idle_policy()
    now = now or utcnow()
    timeout = None
    if db is not None and policy.enabled:
        roles = {bool(user.is_admin)}
        row = _load(db, user, sid)
        if row is not None:
            roles.add(bool(row.recorded_admin))
        timeout = min(
            effective_idle_timeout(db, is_platform_admin=role, now=now, policy=policy)
            for role in roles
        )
    return build_idle_session(
        is_platform_admin=bool(user.is_admin),
        sid=sid,
        last_activity_at=last_activity_at,
        now=now,
        absolute_expires_at=absolute_expires_at,
        policy=policy,
        timeout=timeout,
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
    heartbeat_seconds: float = ENFORCEMENT_HEARTBEAT_SECONDS,
    sweep_seconds: float = SWEEP_INTERVAL_SECONDS,
) -> None:
    """Renew the enforcement lease and sweep until stopped.

    With the policy enabled the lease is renewed every ``heartbeat_seconds``,
    starting immediately; the sweep runs every ``sweep_seconds``. Failures are
    logged and retried; while enabled, a replica enforces without the lease,
    so a failed renewal never weakens enforcement. Each tick writes, so it runs
    through ``run_tracked`` and shutdown's ``retire_writers`` waits for it.
    """

    def heartbeat() -> None:
        if not get_browser_idle_policy().enabled:
            return
        with session_factory() as db:
            renew_enforcement_lease(db)
            record_current_admin_roles(db)
            db.commit()

    def sweep() -> None:
        with session_factory() as db:
            revoked, deleted = sweep_browser_sessions(db)
        if revoked or deleted:
            logger.info("Browser session sweep revoked=%d deleted=%d", revoked, deleted)

    async def every(seconds: float, tick: Callable[[], None], what: str) -> None:
        # Separate loops, so a slow sweep never delays a lease renewal.
        while not stop_event.is_set():
            try:
                await run_tracked(tick)
            except Exception:
                logger.exception("Browser idle %s failed", what)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=seconds)
            except asyncio.TimeoutError:
                pass

    await asyncio.gather(
        every(heartbeat_seconds, heartbeat, "enforcement lease renewal"),
        every(sweep_seconds, sweep, "session sweep"),
    )
