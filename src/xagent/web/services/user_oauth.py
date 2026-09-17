"""Owner-scoped access to builtin OAuth credentials.

All ordinary callers pass ``resource_owner_key=None``. Trusted actor callers
pass an exact server-derived key. Centralizing the predicate prevents a direct
ID lookup or provider list from accidentally widening into another namespace.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import and_
from sqlalchemy.orm import Query, Session
from sqlalchemy.sql.elements import ColumnElement

from ..models.user_oauth import (
    USER_OAUTH_RESOURCE_OWNER_KEY_MAX_LENGTH,
    UserOAuth,
)

GMAIL_OAUTH_PROVIDER = "gmail"


def normalize_user_oauth_resource_owner_key(value: Any) -> str | None:
    """Return one valid owner key, preserving ``None`` as ordinary ownership."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("resource_owner_key must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError("resource_owner_key must not be blank")
    if len(normalized) > USER_OAUTH_RESOURCE_OWNER_KEY_MAX_LENGTH:
        raise ValueError(
            "resource_owner_key exceeds "
            f"{USER_OAUTH_RESOURCE_OWNER_KEY_MAX_LENGTH} characters"
        )
    return normalized


def user_oauth_owner_clause(
    resource_owner_key: str | None,
) -> ColumnElement[bool]:
    """Build the exact SQL predicate for one credential owner namespace."""
    owner_key = normalize_user_oauth_resource_owner_key(resource_owner_key)
    if owner_key is None:
        return UserOAuth.resource_owner_key.is_(None)
    return UserOAuth.resource_owner_key == owner_key


def ordinary_gmail_clause() -> ColumnElement[bool]:
    """Build the ordinary-Gmail conditions for a direct ``UserOAuth`` query.

    Queries based on ``scoped_user_oauth_query(..., resource_owner_key=None)``
    must also filter ``UserOAuth.provider`` to Gmail.
    """
    return and_(
        UserOAuth.provider == GMAIL_OAUTH_PROVIDER,
        user_oauth_owner_clause(None),
    )


def is_ordinary_gmail(account: UserOAuth) -> bool:
    """Return whether a loaded credential is an ordinary Gmail credential."""
    return bool(
        account.provider == GMAIL_OAUTH_PROVIDER and account.resource_owner_key is None
    )


def scoped_user_oauth_query(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str | None,
) -> Query[UserOAuth]:
    """Build a query restricted to one xagent user and one owner namespace."""
    return db.query(UserOAuth).filter(
        UserOAuth.user_id == user_id,
        user_oauth_owner_clause(resource_owner_key),
    )


def list_scoped_user_oauth_accounts(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str | None,
) -> list[UserOAuth]:
    """List one owner's credentials in stable creation order."""
    return (
        scoped_user_oauth_query(
            db,
            user_id=user_id,
            resource_owner_key=resource_owner_key,
        )
        .order_by(UserOAuth.id)
        .all()
    )


def get_scoped_user_oauth_account(
    db: Session,
    *,
    user_id: int,
    account_id: int,
    resource_owner_key: str | None,
) -> UserOAuth | None:
    """Get a credential by ID only when the expected owner also matches."""
    return (
        scoped_user_oauth_query(
            db,
            user_id=user_id,
            resource_owner_key=resource_owner_key,
        )
        .filter(UserOAuth.id == int(account_id))
        .populate_existing()
        .first()
    )


def get_user_oauth_account_by_id(
    db: Session,
    *,
    account_id: int,
    resource_owner_key: str | None,
) -> UserOAuth | None:
    """Get a foreign-key-addressed credential in one owner namespace.

    Internal Gmail watch rows already bind a globally unique OAuth primary key
    and may not have an independently trusted account id at every reload site.
    This shape preserves that existing lookup while centralizing its owner
    namespace predicate.
    """
    return (
        db.query(UserOAuth)
        .filter(
            UserOAuth.id == int(account_id),
            user_oauth_owner_clause(resource_owner_key),
        )
        .populate_existing()
        .first()
    )


def delete_scoped_user_oauth_accounts(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str | None,
    providers: Sequence[str],
) -> int:
    """Delete selected local credentials without committing the session.

    An empty sequence deletes nothing. Requiring explicit provider names keeps
    a missing or falsy filter from becoming a namespace-wide deletion.
    Transaction ownership remains with the caller; this function never
    commits. Callers must not retain matching identity-mapped rows because the
    bulk delete does not synchronize those in-memory objects.
    """
    if isinstance(providers, str):
        raise TypeError("providers must be a sequence, not a string")
    provider_keys = tuple(dict.fromkeys(str(provider) for provider in providers))
    if not provider_keys:
        return 0
    query = scoped_user_oauth_query(
        db,
        user_id=user_id,
        resource_owner_key=resource_owner_key,
    ).filter(UserOAuth.provider.in_(provider_keys))
    return int(query.delete(synchronize_session=False))
