"""Shared session-generation checks for HTTP, OAuth and file credentials."""

import secrets
from datetime import datetime
from typing import Any

from fastapi import HTTPException
from sqlalchemy.orm import Session

from .models.user import User


def user_session_claims(user: User) -> dict[str, Any]:
    return {
        "sub": user.username,
        "user_id": user.id,
        "session_epoch": user.session_epoch,
    }


def matches_user_session(user: User, claims: dict[str, Any]) -> bool:
    epoch = user.session_epoch
    supplied = claims.get("session_epoch")
    # Pre-upgrade JWTs remain valid until the first credential reset. The
    # generation is durable and shared by all API processes.
    if epoch is None:
        return supplied is None
    return type(supplied) is str and secrets.compare_digest(str(epoch), supplied)


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

    Discard pending ORM writes and the read transaction before the conditional
    write (also avoiding SQLite read-to-write snapshot upgrades). A password
    change/reset must never be followed by a stale login restoring refresh.
    """
    db.rollback()
    values: dict[Any, Any] = {
        User.refresh_token: refresh_token,
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
