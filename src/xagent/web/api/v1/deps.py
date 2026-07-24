"""Auth dependency for ``/v1/*`` endpoints.

``get_agent_from_api_key`` resolves the ``Authorization: Bearer
xag_<prefix>_<secret>`` header to the bound :class:`Agent` row,
returning ``(Agent, AgentApiKey)`` on success and raising
:class:`V1ApiError` ``invalid_api_key`` (HTTP 401) on every failure
path. ``get_workforce_from_api_key`` is the workforce-bound
counterpart, and ``get_principal_from_api_key`` accepts either owner
type for endpoints shared by both (the ``/v1/chat/tasks/*`` family).

Failure paths intentionally share the same response code and burn the
same ~100ms of bcrypt work (via :func:`verify_dummy`) regardless of
which check failed:

  - Missing / malformed header
  - Prefix not in DB
  - Secret doesn't match the stored bcrypt hash
  - Bound agent row missing (orphan key, shouldn't happen but defended)

Without this symmetry, an attacker could enumerate live prefixes by
timing the response (slow = bcrypt ran = prefix exists; fast = index
miss = prefix doesn't exist). See SDK design doc §7 (key format and
auth flow) and §10 (security considerations).
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import case
from sqlalchemy.orm import Session, joinedload

from ....core.utils.api_key import (
    ApiKeyKind,
    parse_api_key,
    verify_api_key,
    verify_dummy,
)
from ...models.agent import Agent, is_workforce_generated_manager_agent
from ...models.agent_api_key import AgentApiKey
from ...models.database import get_db, get_session_local
from ...models.user import User
from ...models.user_api_key import UserApiKey
from ...models.workforce import Workforce
from ...utils.db_timezone import normalize_datetime_from_db
from .errors import V1ApiError, V1ErrorCode

# ``auto_error=False`` so we can raise our own V1ApiError envelope
# instead of FastAPI's default 403 ``{"detail": "Not authenticated"}``
# when the header is missing -- the SDK contract is one error shape.
_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class ApiKeyPrincipal:
    """Resolved owner of a presented SDK API key.

    Exactly one of ``agent`` / ``workforce`` is set, mirroring the
    exactly-one-FK invariant on :class:`AgentApiKey`.
    """

    key_row: AgentApiKey
    agent: Optional[Agent] = None
    workforce: Optional[Workforce] = None

    @property
    def owner_user_id(self) -> int:
        """The user identity SDK calls act as: the owner of the bound
        agent or workforce."""
        if self.agent is not None:
            return int(self.agent.user_id)
        assert self.workforce is not None
        return int(self.workforce.owner_user_id)


def _resolve_principal_from_credentials(
    credentials: HTTPAuthorizationCredentials | None, db: Session
) -> ApiKeyPrincipal:
    """Shared key-resolution core for the runtime-key dependencies.

    All failure paths spend ~100ms of bcrypt work and return the same
    ``invalid_api_key`` code, so an attacker cannot enumerate live
    prefixes by either error code or response timing.
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

    # Index lookup is O(1) on ix_agent_api_keys_key_prefix; revoked (and,
    # below, paused) rows are excluded by the filter, not by any DB
    # constraint -- an owner can hold multiple simultaneously-active
    # keys, so uniqueness is no longer enforced at the schema level.
    # ``joinedload`` pulls the bound owner rows in the same SELECT so we
    # don't pay a second round-trip on the success path (the relationships
    # default to lazy='select', which would otherwise emit a separate
    # query when we access ``key_row.agent`` / ``key_row.workforce``).
    key_row = (
        db.query(AgentApiKey)
        .options(
            joinedload(AgentApiKey.agent),
            joinedload(AgentApiKey.workforce),
        )
        .filter(
            AgentApiKey.key_prefix == prefix,
            AgentApiKey.revoked_at.is_(None),
            # A paused key is treated identically to a missing/revoked one
            # -- same opaque 401, same verify_dummy timing -- so a caller
            # can't distinguish "paused" from "never existed" below.
            AgentApiKey.paused_at.is_(None),
        )
        .first()
    )

    # Prefix missing, revoked, or paused. verify_dummy to keep timing
    # indistinguishable from the "secret wrong" branch below.
    if key_row is None:
        verify_dummy()
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    # Real bcrypt check on the full key. Constant time within the cost
    # factor; ``verify_api_key`` returns False (not raise) on malformed
    # hash inputs.
    if not verify_api_key(raw, key_row.key_hash):  # type: ignore[arg-type]
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    # Workforce-bound key: resolves to the workforce itself, never to
    # its generated manager agent (which stays 401 below for
    # agent-bound keys).
    if key_row.workforce_id is not None:
        workforce = key_row.workforce
        if workforce is None:
            raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)
        return ApiKeyPrincipal(key_row=key_row, workforce=workforce)

    # Bound agent. Should always exist (CASCADE on agents -> api_keys),
    # but defend against an out-of-band DELETE that somehow leaves a
    # dangling key row. The relationship was eagerly loaded above, so
    # this is a memory access, not another query. Treat as
    # invalid_api_key rather than 404 so the client retries with a
    # fresh key instead of investigating an imaginary agent.
    agent = key_row.agent
    if agent is None or is_workforce_generated_manager_agent(agent):
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    return ApiKeyPrincipal(key_row=key_row, agent=agent)


async def get_principal_from_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> ApiKeyPrincipal:
    """Resolve an SDK API key to its owner, agent- or workforce-bound.

    Used by endpoints shared by both owner types (the
    ``/v1/chat/tasks/*`` family); owner-specific endpoints use
    :func:`get_agent_from_api_key` / :func:`get_workforce_from_api_key`.
    """
    return _resolve_principal_from_credentials(credentials, db)


async def get_agent_from_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> Tuple[Agent, AgentApiKey]:
    """Resolve an SDK API key to ``(Agent, AgentApiKey)``.

    Workforce-bound keys are rejected with the same opaque 401 --
    endpoints using this dependency are agent-only surfaces.

    Returns:
        Tuple ``(agent, key_row)`` where ``agent`` is the bound Agent
        ORM row and ``key_row`` is the matching active AgentApiKey row.
        Use the key_row for downstream operations that need to know
        which prefix was presented (e.g. ``/v1/me``).

    Raises:
        V1ApiError(INVALID_API_KEY, 401): on any auth failure -- one
            opaque code so caller cannot distinguish *which* check
            failed.
    """
    principal = _resolve_principal_from_credentials(credentials, db)
    if principal.agent is None:
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)
    return principal.agent, principal.key_row


async def get_workforce_from_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> Tuple[Workforce, AgentApiKey]:
    """Resolve an SDK API key to ``(Workforce, AgentApiKey)``.

    Agent-bound keys are rejected with the same opaque 401 --
    endpoints using this dependency are workforce-only surfaces.
    """
    principal = _resolve_principal_from_credentials(credentials, db)
    if principal.workforce is None:
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)
    return principal.workforce, principal.key_row


def record_key_usage(key_prefix: str) -> None:
    """Best-effort usage tracking for the API Keys page's stat cards.

    Deliberately NOT called from :func:`get_agent_from_api_key` itself --
    that dependency backs every ``/v1/chat/tasks/*`` route including the
    read-only status/steps polling endpoints SDK clients hit repeatedly,
    and recording a write there would (a) put a DB write+commit on the
    busiest possible request path and (b) conflate polling with real
    invocations in the "calls this month" stat. Callers should invoke
    this only from endpoints that represent an actual SDK-visible call
    (e.g. creating a task or appending a message), not from polling GETs.

    Runs on its own DB session -- deliberately isolated from the
    request-scoped ``db`` session used for auth -- and issues a single
    atomic UPDATE (month rollover included via ``case()``) rather than a
    Python-side read-modify-write. This avoids three problems a shared
    session would have: committing here would prematurely flush/commit
    whatever the caller's endpoint later stages on the same session;
    rolling back on failure would poison that session for the rest of
    the request; and incrementing ``usage_month_calls`` in Python is
    subject to lost updates under concurrent calls with the same key.
    Never allowed to fail the request -- errors are logged and
    swallowed, not raised.
    """
    now = datetime.now(timezone.utc)
    current_month = now.strftime("%Y-%m")
    session_local = get_session_local()
    local_db = session_local()
    try:
        local_db.query(AgentApiKey).filter(
            AgentApiKey.key_prefix == key_prefix,
            # Belt-and-suspenders: today both call sites are gated by
            # get_agent_from_api_key, which already excludes revoked/paused
            # keys before the endpoint body runs. Repeating the guard here
            # means a future caller that forgets that gate can't silently
            # bump usage on a dead key.
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


async def get_user_from_personal_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> Tuple[User, UserApiKey]:
    """Resolve a personal management API key to ``(User, UserApiKey)``."""
    if credentials is None:
        verify_dummy()
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    raw = credentials.credentials
    parsed = parse_api_key(raw)
    if parsed is None or parsed.kind != ApiKeyKind.PERSONAL:
        verify_dummy()
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    key_row = (
        db.query(UserApiKey)
        .options(joinedload(UserApiKey.user))
        .filter(
            UserApiKey.key_prefix == parsed.prefix,
            UserApiKey.revoked_at.is_(None),
        )
        .first()
    )
    if key_row is None:
        verify_dummy()
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    now = datetime.now(timezone.utc)
    expires_at = key_row.expires_at
    # ``DateTime(timezone=True)`` reads back naive on SQLite; normalize
    # to aware UTC before comparing so an expired key yields 401, not a
    # 500 from comparing naive vs aware datetimes.
    if (
        expires_at is not None and normalize_datetime_from_db(expires_at) <= now  # type: ignore[arg-type]
    ):
        verify_dummy()
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    if not verify_api_key(raw, key_row.key_hash):  # type: ignore[arg-type]
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    user = key_row.user
    if user is None:
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    return user, key_row
