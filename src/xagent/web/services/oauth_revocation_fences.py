"""Fence builtin OAuth grant revocations against concurrent callbacks.

GitHub and Google revoke a whole grant per upstream account and OAuth client.
A disconnect revokes only once no local row references that account any more
(``has_other_builtin_oauth_reference``), but a concurrent callback for the
same account may already have exchanged its code -- its tokens belong to the
very grant about to be revoked -- without having committed its row yet. The
revoke would then succeed and the callback would store dead credentials and
report success.

Both sides therefore lock one ``OAuthRevocationFence`` row per (provider,
upstream account) and hold it until their transaction commits:

* the revoker locks it, runs the reference check, and -- only when the
  revoke goes ahead -- records ``revoking_until``, a deadline by which its
  provider call has either started or been skipped; after the call it pulls
  the deadline back to the moment the call ended;
* the callback locks it after its token exchange and refuses (asking for a
  reconnect) when the exchange started before ``revoking_until``.

Whichever commits first wins. A callback committed first is seen by the
reference check, so nothing is revoked; a revoker committed first leaves a
deadline every overlapping callback sees. The exchange-start time and the
deadline come from different hosts' clocks, so the window keeps a margin far
larger than ordinary clock skew.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from ..models.oauth_revocation_fence import OAuthRevocationFence

# How long an opened fence refuses callbacks if its revoker never closes it.
OAUTH_REVOCATION_FENCE_WINDOW = timedelta(minutes=2)
# The provider call must start at least this long before the fence lapses,
# covering its own timeouts; a revoker delayed past that skips the call.
OAUTH_REVOCATION_CALL_MARGIN = timedelta(minutes=1)


def _utc(value: datetime) -> datetime:
    """SQLite returns timezone-aware columns naive; they were stored as UTC."""
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def lock_oauth_revocation_fence(
    db: Session, *, provider: str, provider_user_id: str
) -> OAuthRevocationFence:
    """Create the account's fence row if needed and lock it until commit.

    The insert happens first on both sides so the lock never depends on a
    row that may not exist yet: PostgreSQL makes a concurrent insert of the
    same key wait for the first one's commit, and SQLite's single writer
    serializes the write statements outright.
    """
    dialect = db.get_bind().dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    elif dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert  # type: ignore[assignment]
    else:
        raise RuntimeError("OAuth revocation fences require PostgreSQL or SQLite")
    db.execute(
        insert(OAuthRevocationFence)
        .values(provider=provider, provider_user_id=provider_user_id)
        .on_conflict_do_nothing(index_elements=["provider", "provider_user_id"])
    )
    fence: OAuthRevocationFence = (
        db.query(OAuthRevocationFence)
        .filter(
            OAuthRevocationFence.provider == provider,
            OAuthRevocationFence.provider_user_id == provider_user_id,
        )
        .with_for_update()
        .populate_existing()
        .one()
    )
    return fence


def open_oauth_revocation_fence(fence: OAuthRevocationFence) -> datetime:
    """Record that a revocation is about to reach the provider; returns its deadline.

    Never moves an existing later deadline back: another revoker of the same
    account may still be in flight.
    """
    deadline = datetime.now(timezone.utc) + OAUTH_REVOCATION_FENCE_WINDOW
    current = fence.revoking_until
    if current is None or _utc(current) < deadline:
        fence.revoking_until = deadline
    return deadline


def oauth_revocation_call_allowed(deadline: datetime) -> bool:
    """Whether a revoker may still start its provider call before ``deadline``."""
    return datetime.now(timezone.utc) < deadline - OAUTH_REVOCATION_CALL_MARGIN


def close_oauth_revocation_fence(
    db: Session, *, provider: str, provider_user_id: str, deadline: datetime
) -> None:
    """Pull the deadline back to now once this revoker's provider call is over.

    Only the revoker that set the current deadline may shorten it; the
    caller commits.
    """
    fence = lock_oauth_revocation_fence(
        db, provider=provider, provider_user_id=provider_user_id
    )
    if fence.revoking_until is not None and _utc(fence.revoking_until) == deadline:
        fence.revoking_until = datetime.now(timezone.utc)


def oauth_revocation_fence_refuses(
    fence: OAuthRevocationFence, *, exchange_started_at: datetime
) -> bool:
    """Whether a revocation may have reached the provider after this exchange
    started, killing the tokens it was issued."""
    return fence.revoking_until is not None and exchange_started_at <= _utc(
        fence.revoking_until
    )
