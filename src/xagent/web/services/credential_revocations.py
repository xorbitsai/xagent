"""Revoke every OAuth credential that left the database with its provider.

The invariant
-------------
Deleting a native ``user_oauth`` row, deleting an ``mcp_oauth_grants`` row or
marking one ``revoked`` writes a ``credential_revocations`` record in the same
transaction, through database triggers (``models.credential_revocation``). So
a credential can no longer leave unrevoked because some code path deleted it
without first reading its tokens: whatever removes the row -- an ORM delete, a
bulk delete, a foreign-key cascade, raw SQL -- the record holds the tokens the
row held at that moment, and a rollback discards both together.

This module is the other half: the worker that works those records off, and
the only code that revokes a stored credential with its provider. Deletion
paths only delete. That holds for native GitHub and Google grants, for MCP
grants (RFC 7009), and for the token pair an MCP refresh brings back for a
grant that was deleted while the refresh was in flight
(:func:`enqueue_mcp_oauth_token_revocation`).

One record, one attempt at a time
---------------------------------
:func:`run_credential_revocation_batch` claims due records, revokes each and
writes its outcome:

* **Claim.** A due ``pending`` record is leased by pushing ``next_attempt_at``
  out by :data:`CLAIM_LEASE` and counting the attempt, in one short
  transaction (``FOR UPDATE SKIP LOCKED`` on PostgreSQL, so replicas never
  wait on each other; SQLite has one writer and the claim's compare-and-set
  update is enough). A process that dies holding a lease loses it once the
  lease lapses, and its attempt still counts, so a record that crashes every
  attempt still reaches ``dead``.
* **One grant at a time.** Only the oldest open record of a ``grant_key`` is
  ever claimable, so two records for the same upstream grant never run
  concurrently, and they run in the order their credentials left.
* **Reference check, at execution time.** Google revokes a whole grant and
  GitHub deletes one, per upstream account and OAuth client; an MCP
  authorization server may end the whole authorization. So a record is
  ``skipped`` while a live credential still references the same grant: that
  credential's own deletion enqueues a record later, and revoking it then
  revokes this one's grant too. Native records check (and open) the
  account's revocation fence (``services.oauth_revocation_fences``) exactly
  as disconnects used to, so a concurrent reconnect is either seen by the
  check or refused by its callback.
* **Outcome.** ``revoked`` when the provider revoked it or answered that the
  token was already invalid (revocation is idempotent); ``skipped`` or
  ``failed`` when it must not or cannot be revoked (an unknown issuer is
  never guessed); otherwise it is retried with exponential backoff and
  becomes ``dead``, logged at error level, after :data:`MAX_ATTEMPTS`.
  Every terminal outcome erases the record's token material.

Every outcome write is fenced on the record's ``(id, attempts)``, so a worker
whose lease lapsed cannot overwrite the outcome of the one that took over.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy import exists, select, update
from sqlalchemy.orm import Session, aliased

from ...core.utils.encryption import (
    EncryptionDecodeError,
    decrypt_value_strict,
    encrypt_value,
)
from ..models.credential_revocation import (
    CredentialRevocation,
    CredentialRevocationKind,
    CredentialRevocationStatus,
)
from .db_runtime import run_db_io_cancellation_safe
from .writer_workers import run_tracked

logger = logging.getLogger(__name__)

#: How long a claimed record is left alone: far longer than one revocation
#: (two provider requests bounded by their 10 s timeouts) takes.
CLAIM_LEASE = timedelta(minutes=10)
#: Delay before the second attempt; each later one doubles, up to the cap.
BASE_RETRY_BACKOFF = timedelta(minutes=1)
MAX_RETRY_BACKOFF = timedelta(hours=6)
#: Attempts before a record is ``dead``: with the backoff above about a day
#: of retrying, which rides out a provider outage.
MAX_ATTEMPTS = 12
#: How often the loop looks for due records when the last batch was not full.
POLL_INTERVAL_SECONDS = 15.0
BATCH_SIZE = 20
#: ``last_error`` is for an operator, never a sink for a provider's body.
MAX_ERROR_LENGTH = 500

_TERMINAL_ERASURE = {"access_token": None, "refresh_token": None, "client_secret": None}
_RFC7009_AUTH_METHODS = frozenset({"none", "client_secret_post", "client_secret_basic"})
#: RFC 7009 2.2.1: the token is already invalid, or the server cannot revoke
#: its type. Either way nothing is left to revoke with it.
_RFC7009_SETTLED_ERRORS = frozenset({"invalid_token", "unsupported_token_type"})


class ProviderRevocationError(Exception):
    """A provider revocation request that did not succeed and may on retry.

    Its message goes into ``last_error``: it names the provider's answer
    (an HTTP status, an exception type), never a token or a secret.
    """


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ClaimedRevocation:
    """A detached copy of one claimed record, as the worker reads it."""

    id: int
    kind: CredentialRevocationKind
    attempts: int
    source_id: int
    user_id: int
    issuer: str | None
    provider_user_id: str | None
    access_token: str | None = field(repr=False)
    refresh_token: str | None = field(repr=False)
    mcp_server_id: int | None
    mcp_oauth_client_id: int | None
    resource_owner_key: str | None
    resource: str | None
    client_id: str | None
    client_secret: str | None = field(repr=False)
    token_endpoint_auth_method: str | None
    revocation_endpoint: str | None

    @classmethod
    def of(cls, row: CredentialRevocation) -> ClaimedRevocation:
        return cls(
            id=row.id,
            kind=CredentialRevocationKind(row.kind),
            attempts=row.attempts,
            source_id=row.source_id,
            user_id=row.user_id,
            issuer=row.issuer,
            provider_user_id=row.provider_user_id,
            access_token=row.access_token,
            refresh_token=row.refresh_token,
            mcp_server_id=row.mcp_server_id,
            mcp_oauth_client_id=row.mcp_oauth_client_id,
            resource_owner_key=row.resource_owner_key,
            resource=row.resource,
            client_id=row.client_id,
            client_secret=row.client_secret,
            token_endpoint_auth_method=row.token_endpoint_auth_method,
            revocation_endpoint=row.revocation_endpoint,
        )


@dataclass(frozen=True)
class RevocationOutcome:
    """What one attempt decided.

    ``status`` is terminal, or ``PENDING`` for a retry. ``defer_until`` marks
    a record that was not attempted at all (another revocation of the same
    account is in flight): it is retried then and its attempt is not counted.
    """

    status: CredentialRevocationStatus
    detail: str | None = None
    defer_until: datetime | None = None

    @classmethod
    def revoked(cls, detail: str | None = None) -> RevocationOutcome:
        return cls(CredentialRevocationStatus.REVOKED, detail)

    @classmethod
    def skipped(cls, detail: str) -> RevocationOutcome:
        return cls(CredentialRevocationStatus.SKIPPED, detail)

    @classmethod
    def failed(cls, detail: str) -> RevocationOutcome:
        return cls(CredentialRevocationStatus.FAILED, detail)

    @classmethod
    def retry(cls, detail: str) -> RevocationOutcome:
        return cls(CredentialRevocationStatus.PENDING, detail)

    @classmethod
    def deferred(cls, until: datetime, detail: str) -> RevocationOutcome:
        return cls(CredentialRevocationStatus.PENDING, detail, defer_until=until)


@dataclass(frozen=True)
class CredentialRevocationReport:
    """Per-status counts of one batch, for the log and for tests."""

    claimed: int = 0
    outcomes: dict[str, int] = field(default_factory=dict)


# --- enqueueing outside the triggers ------------------------------------------


def enqueue_mcp_oauth_token_revocation(  # noqa: PLR0913
    db: Session,
    *,
    grant_id: int,
    lookup_hash: str,
    user_id: int,
    mcp_server_id: int,
    mcp_oauth_client_id: int,
    resource_owner_key: str,
    issuer: str,
    resource: str,
    client: Any,
    token_data: dict[str, Any],
) -> None:
    """Record a token pair that no grant row holds, for the worker to revoke.

    For the pair a refresh brought back after its grant was deleted or
    revoked while the request was in flight: it is persisted nowhere, so no
    trigger can see it. The record is shaped exactly like the one a deletion
    of that grant writes -- same ``grant_key``, same reference identity,
    tokens encrypted the way the grant stores them -- so the worker treats it
    the same way, including skipping it if a reconnect has meanwhile stored
    an active successor. ``client`` carries the OAuth client's ``client_id``,
    encrypted ``client_secret``, ``token_endpoint_auth_method`` and
    ``metadata_json``, read before the request in case the client row is
    deleted meanwhile. The caller commits.
    """
    metadata = client.metadata_json if isinstance(client.metadata_json, dict) else {}
    endpoint = metadata.get("revocation_endpoint")
    refresh_token = token_data.get("refresh_token")
    db.add(
        CredentialRevocation(
            kind=CredentialRevocationKind.MCP_OAUTH_GRANT.value,
            source_id=grant_id,
            user_id=user_id,
            grant_key=f"mcp_oauth_grant:{lookup_hash}",
            issuer=issuer,
            access_token=encrypt_value(str(token_data["access_token"])),
            refresh_token=encrypt_value(str(refresh_token)) if refresh_token else None,
            mcp_server_id=mcp_server_id,
            mcp_oauth_client_id=mcp_oauth_client_id,
            resource_owner_key=resource_owner_key,
            resource=resource,
            client_id=str(client.client_id),
            client_secret=str(client.client_secret) if client.client_secret else None,
            token_endpoint_auth_method=str(client.token_endpoint_auth_method or "none"),
            revocation_endpoint=endpoint if isinstance(endpoint, str) else None,
        )
    )


# --- provider calls -----------------------------------------------------------


async def revoke_rfc7009_tokens(
    *,
    revocation_endpoint: str,
    client_id: str,
    client_secret: str,
    token_endpoint_auth_method: str,
    tokens: Sequence[tuple[str, str]],
) -> None:
    """Revoke each ``(token, token_type_hint)`` at an RFC 7009 endpoint.

    The client authenticates the way it does at the token endpoint. Returns
    once every token is revoked or reported already invalid (a 2xx, or a 400
    ``invalid_token`` / ``unsupported_token_type``); raises
    :class:`ProviderRevocationError` otherwise, so the caller retries --
    revoking the same tokens again is harmless. ``ValueError`` for a client
    authentication method this cannot speak, which no retry fixes.
    """
    from .mcp_oauth import (
        MCP_OAUTH_HTTP_TIMEOUT_SECONDS,
        create_mcp_oauth_http_client,
        oauth_post,
    )

    if token_endpoint_auth_method not in _RFC7009_AUTH_METHODS:
        raise ValueError("unsupported client authentication method")
    auth: httpx.Auth | None = None
    base_data = {"client_id": client_id}
    if token_endpoint_auth_method == "client_secret_post" and client_secret:
        base_data["client_secret"] = client_secret
    elif token_endpoint_auth_method == "client_secret_basic" and client_secret:
        auth = httpx.BasicAuth(client_id, client_secret)
    try:
        async with create_mcp_oauth_http_client(
            timeout=MCP_OAUTH_HTTP_TIMEOUT_SECONDS
        ) as http_client:
            for token, token_type_hint in tokens:
                request_kwargs: dict[str, Any] = {
                    "data": {
                        **base_data,
                        "token": token,
                        "token_type_hint": token_type_hint,
                    },
                    "headers": {"Content-Type": "application/x-www-form-urlencoded"},
                }
                if auth is not None:
                    request_kwargs["auth"] = auth
                response = await oauth_post(
                    revocation_endpoint, client=http_client, **request_kwargs
                )
                if response.is_success or _rfc7009_settled(response):
                    continue
                raise ProviderRevocationError(
                    f"revocation endpoint answered HTTP {response.status_code} "
                    f"for the {token_type_hint}"
                )
    except ProviderRevocationError:
        raise
    except Exception as exc:
        # Only the type: an httpx error's text can carry the endpoint URL.
        raise ProviderRevocationError(
            f"revocation request failed ({type(exc).__name__})"
        ) from None


def _rfc7009_settled(response: httpx.Response) -> bool:
    if response.status_code != 400:
        return False
    try:
        payload = response.json()
    except ValueError:
        return False
    return isinstance(payload, dict) and payload.get("error") in _RFC7009_SETTLED_ERRORS


# --- attempts -----------------------------------------------------------------


def _decrypt(value: str | None) -> str | None:
    return decrypt_value_strict(value) if value else None


def _resolve_native_sync(
    session_factory: Callable[[], Session],
    record: ClaimedRevocation,
    access_token: str,
    refresh_token: str | None,
) -> Any:
    from ..api.auth import resolve_builtin_oauth_revocation

    with session_factory() as db:
        return resolve_builtin_oauth_revocation(
            db,
            oauth_provider=record.issuer,
            access_token=access_token,
            provider_user_id=record.provider_user_id,
            refresh_token=refresh_token,
        )


@dataclass(frozen=True)
class _FenceClaim:
    """A native revocation cleared to call its provider.

    ``deadline`` is the fence deadline its call must start well before;
    ``None`` for a row with no recorded upstream account, which has neither a
    reference check nor a fence.
    """

    deadline: datetime | None


@dataclass(frozen=True)
class _FenceRefusal:
    outcome: RevocationOutcome


def _claim_native_fence_sync(
    session_factory: Callable[[], Session], revocation: Any
) -> _FenceClaim | _FenceRefusal:
    """Reference-check one native revocation under its account's fence.

    The fence row stays locked from before the reference check until the
    fence is opened and committed, and an OAuth callback for the same
    account takes the same lock before storing its credentials. So a
    callback either committed first and the check sees its row, or it finds
    the opened fence and is refused instead of storing tokens this revoke
    may kill. A fence another revoker still holds defers the record to its
    deadline: only one call per account may be in flight, or a second
    holder closing the fence early could let a reconnect through.
    """
    from ..api.auth import builtin_oauth_references
    from .oauth_revocation_fences import (
        lock_oauth_revocation_fence,
        oauth_revocation_fence_is_open,
        open_oauth_revocation_fence,
    )

    if revocation.provider_user_id is None:
        return _FenceClaim(deadline=None)
    with session_factory() as db:
        fence = lock_oauth_revocation_fence(
            db,
            provider=revocation.provider,
            provider_user_id=revocation.provider_user_id,
        )
        references = builtin_oauth_references(
            db,
            provider=revocation.provider,
            provider_user_id=revocation.provider_user_id,
        )
        if references:
            only_unidentified = all(row.provider_user_id is None for row in references)
            db.commit()
            return _FenceRefusal(
                RevocationOutcome.skipped(
                    "a legacy connection with no recorded account id may share "
                    "this grant"
                    if only_unidentified
                    else "another local connection still references this account"
                )
            )
        if oauth_revocation_fence_is_open(fence):
            until = fence.revoking_until
            db.commit()
            return _FenceRefusal(
                RevocationOutcome.deferred(
                    _aware(until) if until is not None else _utcnow(),
                    "another revocation of this account is in flight",
                )
            )
        deadline = open_oauth_revocation_fence(fence)
        db.commit()
        return _FenceClaim(deadline=deadline)


def _release_native_fence_sync(
    session_factory: Callable[[], Session], revocation: Any, claim: _FenceClaim
) -> None:
    """Close the fence once the provider call is over. Never raises: a
    failed release only delays reconnects until the deadline lapses."""
    from .oauth_revocation_fences import close_oauth_revocation_fence

    if claim.deadline is None or revocation.provider_user_id is None:
        return
    try:
        with session_factory() as db:
            close_oauth_revocation_fence(
                db,
                provider=revocation.provider,
                provider_user_id=revocation.provider_user_id,
                deadline=claim.deadline,
            )
            db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "OAuth revocation fence release failed (exception_type=%s)",
            type(exc).__name__,
        )


def _aware(value: datetime) -> datetime:
    """SQLite returns timezone-aware columns naive; they were stored as UTC."""
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


async def _revoke_user_oauth(
    record: ClaimedRevocation, session_factory: Callable[[], Session]
) -> RevocationOutcome:
    from ..api.auth import (
        BUILTIN_OAUTH_REVOCATION_PROVIDERS,
        revoke_resolved_builtin_oauth_grant,
    )
    from .oauth_revocation_fences import oauth_revocation_call_allowed

    if record.issuer is None:
        return RevocationOutcome.failed(
            "unknown issuer: the credential recorded no issuing provider, and "
            "revoking it would mean guessing one"
        )
    if record.issuer not in BUILTIN_OAUTH_REVOCATION_PROVIDERS:
        return RevocationOutcome.skipped(
            f"provider {record.issuer!r} has no provider-side revocation"
        )
    try:
        access_token = _decrypt(record.access_token)
        refresh_token = _decrypt(record.refresh_token)
    except EncryptionDecodeError:
        return RevocationOutcome.failed(
            "the stored token cannot be decrypted with the configured key"
        )
    if not (access_token or refresh_token):
        return RevocationOutcome.skipped("the credential held no token")
    revocation = await run_tracked(
        _resolve_native_sync, session_factory, record, access_token or "", refresh_token
    )
    if revocation is None:
        if record.issuer == "github":
            # The provider row or its client credentials are missing: an
            # operator can fix that, so keep retrying until the budget ends.
            return RevocationOutcome.retry(
                "GitHub OAuth client credentials are not configured"
            )
        return RevocationOutcome.skipped(
            "no Google account id was recorded, so nothing proves the grant "
            "is unshared"
        )
    claim = await run_tracked(_claim_native_fence_sync, session_factory, revocation)
    if isinstance(claim, _FenceRefusal):
        return claim.outcome
    try:
        if claim.deadline is not None and not oauth_revocation_call_allowed(
            claim.deadline
        ):
            return RevocationOutcome.retry("delayed past its revocation fence")
        try:
            # Drained, not abandoned, when this task is cancelled: the fence
            # may close only once the provider call has really ended, never
            # while its thread is still in flight.
            detail = await run_db_io_cancellation_safe(
                partial(revoke_resolved_builtin_oauth_grant, revocation)
            )
        except ProviderRevocationError as exc:
            return RevocationOutcome.retry(str(exc))
        return RevocationOutcome.revoked(detail)
    finally:
        await run_tracked(
            _release_native_fence_sync, session_factory, revocation, claim
        )


def _mcp_grant_still_referenced_sync(
    session_factory: Callable[[], Session], record: ClaimedRevocation
) -> bool:
    """Whether an active grant holds the same authorization as ``record``.

    The identity a reconnect keeps (``_replaced_by_an_active_grant`` and
    ``_delete_superseded_mcp_oauth_grants`` match on it): same server, user,
    owner, client, issuer and resource. A provider may end the whole
    authorization when one of its tokens is revoked, which would take that
    live grant down too.
    """
    from ..models.mcp_oauth import MCPOAuthGrant

    with session_factory() as db:
        return (
            db.query(MCPOAuthGrant.id)
            .filter(
                MCPOAuthGrant.mcp_server_id == record.mcp_server_id,
                MCPOAuthGrant.user_id == record.user_id,
                MCPOAuthGrant.resource_owner_key == record.resource_owner_key,
                MCPOAuthGrant.mcp_oauth_client_id == record.mcp_oauth_client_id,
                MCPOAuthGrant.issuer == record.issuer,
                MCPOAuthGrant.resource == record.resource,
                MCPOAuthGrant.status == "active",
            )
            .first()
            is not None
        )


async def _revoke_mcp_oauth_grant(
    record: ClaimedRevocation, session_factory: Callable[[], Session]
) -> RevocationOutcome:
    if record.client_id is None:
        return RevocationOutcome.failed(
            "the grant's OAuth client registration was gone when it was recorded"
        )
    endpoint = record.revocation_endpoint
    if not endpoint:
        return RevocationOutcome.skipped(
            "the authorization server advertises no revocation endpoint"
        )
    if await run_tracked(_mcp_grant_still_referenced_sync, session_factory, record):
        return RevocationOutcome.skipped(
            "an active grant still holds the same authorization"
        )
    try:
        client_secret = _decrypt(record.client_secret) or ""
        tokens = [
            (token, hint)
            for token, hint in (
                (_decrypt(record.access_token), "access_token"),
                (_decrypt(record.refresh_token), "refresh_token"),
            )
            if token
        ]
    except EncryptionDecodeError:
        return RevocationOutcome.failed(
            "a stored token or client secret cannot be decrypted with the "
            "configured key"
        )
    if not tokens:
        return RevocationOutcome.skipped("the grant held no token")
    try:
        await revoke_rfc7009_tokens(
            revocation_endpoint=endpoint,
            client_id=record.client_id,
            client_secret=client_secret,
            token_endpoint_auth_method=record.token_endpoint_auth_method or "none",
            tokens=tokens,
        )
    except ValueError as exc:
        return RevocationOutcome.failed(str(exc))
    except ProviderRevocationError as exc:
        return RevocationOutcome.retry(str(exc))
    return RevocationOutcome.revoked()


async def _attempt(
    record: ClaimedRevocation, session_factory: Callable[[], Session]
) -> RevocationOutcome:
    match record.kind:
        case CredentialRevocationKind.USER_OAUTH:
            return await _revoke_user_oauth(record, session_factory)
        case CredentialRevocationKind.MCP_OAUTH_GRANT:
            return await _revoke_mcp_oauth_grant(record, session_factory)


# --- claim and outcome --------------------------------------------------------


def _retry_backoff(attempts: int) -> timedelta:
    """1, 2, 4, ... minutes after attempt 1, 2, 3, ..., capped."""
    return min(BASE_RETRY_BACKOFF * (2 ** max(attempts - 1, 0)), MAX_RETRY_BACKOFF)


def _truncate(text: str | None) -> str | None:
    return text[:MAX_ERROR_LENGTH] if text else text


def _claim_due_sync(
    session_factory: Callable[[], Session],
    *,
    now: datetime,
    limit: int,
    max_attempts: int,
) -> list[ClaimedRevocation]:
    """Lease up to ``limit`` due records, oldest first, one per ``grant_key``.

    A due record whose budget is already spent -- its leases lapsed without
    an outcome that many times, so its attempt hangs or crashes the process
    -- is moved to ``dead`` here instead, since no outcome will ever do it.
    """
    record = CredentialRevocation
    older = aliased(CredentialRevocation)
    pending = CredentialRevocationStatus.PENDING.value
    older_open = exists().where(
        older.grant_key == record.grant_key,
        older.status == pending,
        older.id < record.id,
    )
    claimed: list[ClaimedRevocation] = []
    with session_factory() as db:
        rows = (
            db.execute(
                select(record)
                .where(
                    record.status == pending,
                    record.next_attempt_at <= now,
                    ~older_open,
                )
                .order_by(record.id)
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
            .scalars()
            .all()
        )
        for row in rows:
            snapshot = ClaimedRevocation.of(row)
            exhausted = snapshot.attempts >= max_attempts
            values: dict[str, Any] = (
                {
                    "status": CredentialRevocationStatus.DEAD.value,
                    "last_error": "every attempt ended without an outcome",
                    **_TERMINAL_ERASURE,
                }
                if exhausted
                else {
                    "attempts": snapshot.attempts + 1,
                    "next_attempt_at": now + CLAIM_LEASE,
                }
            )
            # Compare-and-set on top of the row lock: SQLite takes no lock
            # on the select, and two processes may share one file.
            won = db.execute(
                update(record)
                .where(
                    record.id == snapshot.id,
                    record.attempts == snapshot.attempts,
                    record.status == pending,
                )
                .values(updated_at=now, **values)
            ).rowcount
            if won != 1:
                continue
            if exhausted:
                logger.error(
                    "Credential revocation %s (%s %s) is dead: every attempt "
                    "ended without an outcome",
                    snapshot.id,
                    snapshot.kind.value,
                    snapshot.source_id,
                )
                continue
            claimed.append(replace(snapshot, attempts=snapshot.attempts + 1))
        db.commit()
    return claimed


def _settle_sync(
    session_factory: Callable[[], Session],
    record: ClaimedRevocation,
    outcome: RevocationOutcome,
    *,
    now: datetime,
    max_attempts: int,
) -> CredentialRevocationStatus | None:
    """Write one attempt's outcome; ``None`` if the lease was lost meanwhile."""
    status = outcome.status
    values: dict[str, Any] = {"last_error": _truncate(outcome.detail), "updated_at": now}
    if outcome.defer_until is not None:
        # Not attempted: hand the counted attempt back.
        values |= {
            "attempts": record.attempts - 1,
            "next_attempt_at": max(outcome.defer_until, now),
        }
    elif status is CredentialRevocationStatus.PENDING:
        if record.attempts >= max_attempts:
            status = CredentialRevocationStatus.DEAD
            values |= {"status": status.value, **_TERMINAL_ERASURE}
        else:
            values["next_attempt_at"] = now + _retry_backoff(record.attempts)
    else:
        values |= {"status": status.value, **_TERMINAL_ERASURE}
    with session_factory() as db:
        won = db.execute(
            update(CredentialRevocation)
            .where(
                CredentialRevocation.id == record.id,
                CredentialRevocation.attempts == record.attempts,
                CredentialRevocation.status == CredentialRevocationStatus.PENDING.value,
            )
            .values(**values)
        ).rowcount
        db.commit()
    if won != 1:
        return None
    _log_outcome(record, status, outcome.detail)
    return status


def _log_outcome(
    record: ClaimedRevocation,
    status: CredentialRevocationStatus,
    detail: str | None,
) -> None:
    label = f"Credential revocation {record.id} ({record.kind.value} {record.source_id})"
    match status:
        case CredentialRevocationStatus.REVOKED:
            logger.info("%s revoked%s", label, f": {detail}" if detail else "")
        case CredentialRevocationStatus.SKIPPED:
            logger.info("%s skipped: %s", label, detail)
        case CredentialRevocationStatus.FAILED:
            logger.warning("%s failed: %s", label, detail)
        case CredentialRevocationStatus.DEAD:
            logger.error(
                "%s is dead after %d attempts: %s", label, record.attempts, detail
            )
        case CredentialRevocationStatus.PENDING:
            logger.warning(
                "%s attempt %d failed, will retry: %s", label, record.attempts, detail
            )


async def run_credential_revocation_batch(
    session_factory: Callable[[], Session],
    *,
    now: Callable[[], datetime] = _utcnow,
    limit: int = BATCH_SIZE,
    max_attempts: int = MAX_ATTEMPTS,
) -> CredentialRevocationReport:
    """Claim due records and attempt each once. ``now`` is a clock for tests."""
    claimed = await run_tracked(
        _claim_due_sync,
        session_factory,
        now=now(),
        limit=limit,
        max_attempts=max_attempts,
    )
    outcomes: dict[str, int] = {}
    for record in claimed:
        try:
            outcome = await _attempt(record, session_factory)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Credential revocation %s attempt raised unexpectedly", record.id
            )
            outcome = RevocationOutcome.retry(
                f"attempt raised unexpectedly ({type(exc).__name__})"
            )
        try:
            status = await run_tracked(
                _settle_sync,
                session_factory,
                record,
                outcome,
                now=now(),
                max_attempts=max_attempts,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Credential revocation %s outcome could not be written; its "
                "lease will lapse and it will be retried",
                record.id,
                exc_info=True,
            )
            continue
        if status is not None:
            outcomes[status.value] = outcomes.get(status.value, 0) + 1
    return CredentialRevocationReport(claimed=len(claimed), outcomes=outcomes)


async def run_credential_revocation_loop(
    session_factory: Callable[[], Session],
    *,
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
    batch_size: int = BATCH_SIZE,
) -> None:
    """Work the outbox off until cancelled.

    Runs in every web process: replicas split the due records between them
    (the claim skips rows another replica holds). A full batch means more
    may be due, so the next one follows at once; a failed batch is logged
    and retried after the interval. Cancellation is the stop signal; an
    attempt cut short keeps its lease, which lapses and is retried.
    """
    while True:
        full = False
        try:
            report = await run_credential_revocation_batch(
                session_factory, limit=batch_size
            )
            full = report.claimed >= batch_size
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("Credential revocation batch failed")
        await asyncio.sleep(0 if full else poll_interval_seconds)
