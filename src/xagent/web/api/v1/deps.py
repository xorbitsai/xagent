"""Auth dependency for ``/v1/*`` endpoints.

``get_agent_from_api_key`` resolves the ``Authorization: Bearer
xag_<prefix>_<secret>`` header to detached agent/key snapshots.
``get_workforce_from_api_key`` is the workforce-bound counterpart, and
``get_principal_from_api_key`` accepts either owner type for endpoints
shared by both (the ``/v1/chat/tasks/*`` family). All database work runs
inside a worker-owned short session; bcrypt starts only after that
session has closed, so neither pool checkout nor password hashing blocks
the asyncio event loop or pins a connection.

Failure paths intentionally share the same response code and burn the
same ~100ms of bcrypt work (via :func:`verify_dummy`) regardless of
which check failed:

  - Missing / malformed header
  - Prefix not in DB
  - Secret doesn't match the stored bcrypt hash
  - Bound owner row missing (orphan key, shouldn't happen but defended)

Without this symmetry, an attacker could enumerate live prefixes by
timing the response (slow = bcrypt ran = prefix exists; fast = index
miss = prefix doesn't exist). See SDK design doc §7 (key format and
auth flow) and §10 (security considerations).
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import cast

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import case
from sqlalchemy.orm import joinedload

from ....core.utils.api_key import (
    ApiKeyKind,
    parse_api_key,
    verify_api_key,
    verify_dummy,
)
from ...models.agent import is_workforce_generated_manager_agent
from ...models.agent_api_key import AgentApiKey
from ...models.database import get_session_local
from ...models.user_api_key import UserApiKey
from ...services.db_runtime import run_db_io_cancellation_safe
from ...utils.db_timezone import normalize_datetime_from_db
from .errors import V1ApiError, V1ErrorCode

# ``auto_error=False`` so we can raise our own V1ApiError envelope
# instead of FastAPI's default 403 ``{"detail": "Not authenticated"}``
# when the header is missing -- the SDK contract is one error shape.
_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class RuntimeApiKeySnapshot:
    """Detached public-safe identity of a runtime API key."""

    key_prefix: str


@dataclass(frozen=True)
class AgentPrincipalSnapshot:
    """Detached agent fields needed after runtime-key authentication."""

    id: int
    user_id: int
    execution_mode: str
    status: str
    origin: str


@dataclass(frozen=True)
class WorkforcePrincipalSnapshot:
    """Detached workforce fields needed after runtime-key authentication."""

    id: int
    owner_user_id: int


@dataclass(frozen=True)
class UserPrincipalSnapshot:
    """Detached user fields needed after personal-key authentication."""

    id: int
    username: str
    email: str | None
    is_admin: bool


@dataclass(frozen=True)
class PersonalApiKeySnapshot:
    """Detached public-safe identity of a personal API key."""

    key_prefix: str


@dataclass(frozen=True)
class ApiKeyPrincipal:
    """Detached owner of a presented SDK API key.

    Exactly one of ``agent`` / ``workforce`` is set, mirroring the
    exactly-one-FK invariant on :class:`AgentApiKey`. No Session or
    attached ORM row crosses the authentication worker boundary.
    """

    key: RuntimeApiKeySnapshot
    agent: AgentPrincipalSnapshot | None = None
    workforce: WorkforcePrincipalSnapshot | None = None

    @property
    def owner_user_id(self) -> int:
        """The user identity SDK calls act as: the owner of the bound
        agent or workforce."""
        if self.agent is not None:
            return int(self.agent.user_id)
        assert self.workforce is not None
        return int(self.workforce.owner_user_id)


@dataclass(frozen=True)
class _RuntimeKeyRecord:
    """Private credential material loaded inside one short DB Session."""

    key: RuntimeApiKeySnapshot
    key_hash: str
    agent_id: int | None
    workforce_id: int | None
    agent: AgentPrincipalSnapshot | None
    workforce: WorkforcePrincipalSnapshot | None


@dataclass(frozen=True)
class _PersonalKeyRecord:
    """Private personal-key material loaded inside one short DB Session."""

    key: PersonalApiKeySnapshot
    key_hash: str
    expires_at: datetime | None
    user: UserPrincipalSnapshot | None


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))


def _load_runtime_key_record(prefix: str) -> _RuntimeKeyRecord | None:
    """Load and detach runtime-key state before bcrypt verification."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        key_row = (
            db.query(AgentApiKey)
            .options(
                joinedload(AgentApiKey.agent),
                joinedload(AgentApiKey.workforce),
            )
            .filter(
                AgentApiKey.key_prefix == prefix,
                AgentApiKey.revoked_at.is_(None),
                AgentApiKey.paused_at.is_(None),
            )
            .first()
        )
        if key_row is None:
            return None

        agent = key_row.agent
        agent_snapshot = (
            AgentPrincipalSnapshot(
                id=int(agent.id),
                user_id=int(agent.user_id),
                execution_mode=_enum_value(agent.execution_mode),
                status=_enum_value(agent.status),
                origin=_enum_value(agent.origin),
            )
            if agent is not None
            else None
        )
        workforce = key_row.workforce
        workforce_snapshot = (
            WorkforcePrincipalSnapshot(
                id=int(workforce.id),
                owner_user_id=int(workforce.owner_user_id),
            )
            if workforce is not None
            else None
        )
        return _RuntimeKeyRecord(
            key=RuntimeApiKeySnapshot(key_prefix=str(key_row.key_prefix)),
            key_hash=str(key_row.key_hash),
            agent_id=(int(key_row.agent_id) if key_row.agent_id is not None else None),
            workforce_id=(
                int(key_row.workforce_id) if key_row.workforce_id is not None else None
            ),
            agent=agent_snapshot,
            workforce=workforce_snapshot,
        )


def _resolve_principal_from_credentials(
    credentials: HTTPAuthorizationCredentials | None,
) -> ApiKeyPrincipal:
    """Resolve one runtime key entirely inside a synchronous DB worker.

    All failure paths spend ~100ms of bcrypt work and return the same
    ``invalid_api_key`` code, so an attacker cannot enumerate live
    prefixes by either error code or response timing. The DB Session is
    closed before bcrypt runs, so the expensive hash check never pins a
    connection-pool slot.
    """
    # Missing header. Still burn bcrypt time so a curl-with-no-header
    # response can't be timed apart from a curl-with-bad-secret one.
    if credentials is None:
        verify_dummy()
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    raw = credentials.credentials

    # Format check. parse_api_key returns None for anything not shaped
    # like ``xag_<6 alnum>_<32 alnum>``. Workforce keys share the AGENT
    # wire format -- the owner type is a DB fact, not a key-shape fact.
    parsed = parse_api_key(raw)
    if parsed is None:
        verify_dummy()
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    if parsed.kind != ApiKeyKind.AGENT:
        verify_dummy()
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    prefix = parsed.prefix

    # The short Session exists only inside _load_runtime_key_record and
    # returns detached scalar snapshots. A paused key is treated exactly
    # like a missing/revoked key.
    record = _load_runtime_key_record(prefix)

    # Prefix missing, revoked, or paused. verify_dummy to keep timing
    # indistinguishable from the "secret wrong" branch below.
    if record is None:
        verify_dummy()
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    # Real bcrypt check on the full key. Constant time within the cost
    # factor; ``verify_api_key`` returns False (not raise) on malformed
    # hash inputs.
    if not verify_api_key(raw, record.key_hash):
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    # Workforce-bound key: resolves to the workforce itself, never to
    # its generated manager agent (which stays 401 below for
    # agent-bound keys).
    if record.workforce_id is not None:
        workforce = record.workforce
        if workforce is None:
            raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)
        return ApiKeyPrincipal(key=record.key, workforce=workforce)

    # Bound agent. Should always exist (CASCADE on agents -> api_keys),
    # but defend against an out-of-band DELETE that somehow leaves a
    # dangling key row. Treat as
    # invalid_api_key rather than 404 so the client retries with a
    # fresh key instead of investigating an imaginary agent.
    agent = record.agent
    if agent is None or is_workforce_generated_manager_agent(agent):
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    return ApiKeyPrincipal(key=record.key, agent=agent)


async def get_principal_from_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> ApiKeyPrincipal:
    """Resolve an SDK API key to its owner, agent- or workforce-bound.

    Used by endpoints shared by both owner types (the
    ``/v1/chat/tasks/*`` family); owner-specific endpoints use
    :func:`get_agent_from_api_key` / :func:`get_workforce_from_api_key`.
    """
    return await run_db_io_cancellation_safe(
        lambda: _resolve_principal_from_credentials(credentials)
    )


async def get_agent_from_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> tuple[AgentPrincipalSnapshot, RuntimeApiKeySnapshot]:
    """Resolve an SDK API key to detached agent/key snapshots.

    Workforce-bound keys are rejected with the same opaque 401 --
    endpoints using this dependency are agent-only surfaces.

    Returns:
        Detached snapshots carrying only the fields downstream routes need.

    Raises:
        V1ApiError(INVALID_API_KEY, 401): on any auth failure -- one
            opaque code so caller cannot distinguish *which* check
            failed.
    """
    principal = await get_principal_from_api_key(credentials)
    if principal.agent is None:
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)
    return principal.agent, principal.key


async def get_workforce_from_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> tuple[WorkforcePrincipalSnapshot, RuntimeApiKeySnapshot]:
    """Resolve an SDK API key to detached workforce/key snapshots.

    Agent-bound keys are rejected with the same opaque 401 --
    endpoints using this dependency are workforce-only surfaces.
    """
    principal = await get_principal_from_api_key(credentials)
    if principal.workforce is None:
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)
    return principal.workforce, principal.key


def _record_key_usage_sync(key_prefix: str) -> None:
    """Persist one usage increment in a worker-owned short Session."""

    now = datetime.now(timezone.utc)
    current_month = now.strftime("%Y-%m")
    session_local = get_session_local()
    local_db = session_local()
    try:
        local_db.query(AgentApiKey).filter(
            AgentApiKey.key_prefix == key_prefix,
            # Belt-and-suspenders: today all call sites are gated by runtime
            # key authentication, which already excludes revoked/paused keys.
            AgentApiKey.revoked_at.is_(None),
            AgentApiKey.paused_at.is_(None),
        ).update(
            {
                AgentApiKey.last_used_at: now,
                AgentApiKey.usage_month: current_month,
                AgentApiKey.usage_month_calls: case(
                    (
                        AgentApiKey.usage_month == current_month,
                        AgentApiKey.usage_month_calls + 1,
                    ),
                    else_=1,
                ),
            },
            synchronize_session=False,
        )
        local_db.commit()
    except Exception:
        local_db.rollback()
        logging.getLogger(__name__).warning(
            "Failed to record API key usage for key_prefix=%s", key_prefix
        )
    finally:
        local_db.close()


async def record_key_usage(key_prefix: str) -> None:
    """Best-effort, non-blocking usage tracking for API-key stat cards.

    Deliberately NOT called from :func:`get_agent_from_api_key` itself --
    that dependency backs every ``/v1/chat/tasks/*`` route including the
    read-only status/steps polling endpoints SDK clients hit repeatedly,
    and recording a write there would (a) put a DB write+commit on the
    busiest possible request path and (b) conflate polling with real
    invocations in the "calls this month" stat. Callers should invoke
    this only from endpoints that represent an actual SDK-visible call
    (e.g. creating a task or appending a message), not from polling GETs.

    The atomic UPDATE, commit, and any QueuePool checkout wait run in a
    worker-owned short Session. No request Session crosses this boundary,
    and a saturated pool cannot block the asyncio event loop. Persistence
    remains best-effort: the synchronous owner logs and swallows failures.
    """
    await run_db_io_cancellation_safe(lambda: _record_key_usage_sync(key_prefix))


def _load_personal_key_record(prefix: str) -> _PersonalKeyRecord | None:
    """Load and detach personal-key state inside one short DB Session."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        key_row = (
            db.query(UserApiKey)
            .options(joinedload(UserApiKey.user))
            .filter(
                UserApiKey.key_prefix == prefix,
                UserApiKey.revoked_at.is_(None),
            )
            .first()
        )
        if key_row is None:
            return None
        user = key_row.user
        user_snapshot = (
            UserPrincipalSnapshot(
                id=int(user.id),
                username=str(user.username),
                email=str(user.email) if user.email is not None else None,
                is_admin=bool(user.is_admin),
            )
            if user is not None
            else None
        )
        return _PersonalKeyRecord(
            key=PersonalApiKeySnapshot(key_prefix=str(key_row.key_prefix)),
            key_hash=str(key_row.key_hash),
            expires_at=cast(datetime | None, key_row.expires_at),
            user=user_snapshot,
        )


def _resolve_user_from_personal_credentials(
    credentials: HTTPAuthorizationCredentials | None,
) -> tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot]:
    """Resolve a personal key inside a DB worker without retaining ORM state."""

    if credentials is None:
        verify_dummy()
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    raw = credentials.credentials
    parsed = parse_api_key(raw)
    if parsed is None or parsed.kind != ApiKeyKind.PERSONAL:
        verify_dummy()
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    record = _load_personal_key_record(parsed.prefix)
    if record is None:
        verify_dummy()
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    now = datetime.now(timezone.utc)
    expires_at = record.expires_at
    # ``DateTime(timezone=True)`` reads back naive on SQLite; normalize
    # to aware UTC before comparing so an expired key yields 401, not a
    # 500 from comparing naive vs aware datetimes.
    if expires_at is not None and normalize_datetime_from_db(expires_at) <= now:
        verify_dummy()
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    # The loading Session is already closed before this bcrypt check runs.
    if not verify_api_key(raw, record.key_hash):
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    user = record.user
    if user is None:
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    return user, record.key


async def get_user_from_personal_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot]:
    """Resolve a personal key to detached user/key snapshots off-loop."""

    return await run_db_io_cancellation_safe(
        lambda: _resolve_user_from_personal_credentials(credentials)
    )
