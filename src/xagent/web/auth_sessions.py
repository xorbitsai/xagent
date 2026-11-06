"""Shared session-generation checks for HTTP, OAuth and file credentials."""

import hashlib
import hmac
import re
import secrets
from datetime import datetime
from typing import Any, cast

from fastapi import HTTPException
from sqlalchemy.orm import Session

from .models.user import User


def new_session_epoch() -> str:
    """Generate exactly 32 URL-safe characters for User.session_epoch."""
    return secrets.token_urlsafe(24)


_REFRESH_TOKEN_HASH_RE = re.compile(r"[0-9a-f]{64}")


def hash_refresh_token(token: str) -> str:
    """SHA-256 hex digest stored in User.refresh_token in place of the JWT.

    The token is a signed, high-entropy JWT, so a plain unsalted digest is enough
    to keep a copy of the database from yielding usable credentials.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def refresh_token_lookup_values(token: str) -> list[str]:
    """Stored values that identify ``token``, for use in a SQL ``IN`` filter.

    The digest, plus the raw token for a row written before the hashing
    migration ran (a deploy that overlaps the old release can still write one).
    The caller must have verified the JWT signature first: a bare digest taken
    from a leaked row is not a valid JWT and must not reach this filter.

    TODO: remove the plaintext fallback here and in ``refresh_token_matches``
    once no release that stores the raw token is running. Until then a row an
    old instance wrote after the migration scan stays redeemable at rest until
    its owner next signs in or refreshes, or the token expires; re-run the
    conversion (migration ``20261106_hash_user_refresh_tokens``) in the release
    that removes the fallback.
    """
    return [hash_refresh_token(token), token]


def refresh_token_matches(stored: str | None, presented: str) -> bool:
    """Compare a presented refresh token with the value stored for the user."""
    # Logout passes the client's token unverified. A JWT and a digest are ASCII,
    # so anything else cannot match, and compare_digest rejects non-ASCII str.
    if not stored or not presented.isascii():
        return False
    if _REFRESH_TOKEN_HASH_RE.fullmatch(stored):
        return hmac.compare_digest(stored, hash_refresh_token(presented))
    # Legacy plaintext row; see refresh_token_lookup_values.
    return hmac.compare_digest(stored, presented)


def user_session_claims(user: User) -> dict[str, Any]:
    return {
        "sub": user.username,
        "user_id": user.id,
        "session_epoch": user.session_epoch,
    }


def matches_user_session(user: User, claims: dict[str, Any]) -> bool:
    return epoch_matches(
        cast(str | None, user.session_epoch), claims.get("session_epoch")
    )


def epoch_matches(epoch: str | None, supplied: object) -> bool:
    # Pre-upgrade JWTs remain valid until the first credential reset. The
    # generation is durable and shared by all API processes.
    if epoch is None:
        return supplied is None
    # Generations are URL-safe ASCII; compare_digest rejects non-ASCII strings.
    # Malformed claims must be credential rejections, not infrastructure errors.
    return (
        type(supplied) is str
        and supplied.isascii()
        and secrets.compare_digest(str(epoch), supplied)
    )


def lock_current_session_epoch(db: Session, user_id: int, epoch: object) -> bool:
    """Fence an OAuth credential write against concurrent password revocation.

    A no-op conditional UPDATE takes the user-row write lock until the caller's
    credential transaction commits. Do not hold this lock during provider I/O.
    """
    if epoch is not None and (type(epoch) is not str or not epoch.isascii()):
        return False
    return (
        db.query(User)
        .filter(User.id == user_id, User.session_epoch == epoch)
        .update(
            {User.session_epoch: epoch, User.updated_at: User.updated_at},
            synchronize_session=False,
        )
        == 1
    )


def commit_refresh_credentials(
    db: Session,
    *,
    user_id: int,
    password_hash: str,
    session_epoch: str | None,
    refresh_token: str,
    expires_at: datetime,
    upgraded_password_hash: str | None = None,
) -> None:
    """Publish credentials only if their authenticated snapshot is still current.

    ``refresh_token`` is the issued JWT; only its digest is stored.

    Discard pending ORM writes and the read transaction before the conditional
    write (also avoiding SQLite read-to-write snapshot upgrades). A password
    change/reset must never be followed by a stale login restoring refresh.
    """
    db.rollback()
    values: dict[Any, Any] = {
        User.refresh_token: hash_refresh_token(refresh_token),
        User.refresh_token_expires_at: expires_at,
    }
    if upgraded_password_hash is not None:
        values[User.password_hash] = upgraded_password_hash
    changed = (
        db.query(User)
        .filter(
            User.id == user_id,
            User.password_hash == password_hash,
            User.session_epoch == session_epoch,
        )
        .update(values, synchronize_session=False)
    )
    if changed != 1:
        db.rollback()
        raise HTTPException(
            status_code=401, detail="Credentials changed; sign in again"
        )
    db.commit()
