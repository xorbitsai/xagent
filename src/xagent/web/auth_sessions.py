"""Shared session-generation checks for HTTP, OAuth and file credentials."""

import secrets
from typing import Any

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
