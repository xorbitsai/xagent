"""One Shopify grant per store (#2810).

Shopify keeps one current expiring offline token per app and store. A new
grant (the authorization code exchange) retires every other refresh token the
app holds for the store, and a grant and a refresh running at the same time
retire each other's result
(https://shopify.dev/docs/apps/build/authentication-authorization/access-tokens#offline-access-tokens).
Grants are owned per user or actor, so two owners connecting one store would
keep logging each other out.

So a store with a live grant belongs to that grant's owner: a callback for
anyone else is refused before its code exchange, which leaves the owner's
token untouched. The callback's exchange and every refresh for the store run
under :func:`lock_shopify_store`, so two of them never overlap.

The check covers one database, so one region. Two regions sharing one Shopify
app can still retire each other's tokens.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from sqlalchemy import func
from sqlalchemy.orm import Session

from ..models.shopify_store_lock import ShopifyStoreLock
from ..models.user_oauth import UserOAuth
from ..shopify_oauth import is_shopify_provider
from .time_utils import coerce_utc

# Shopify issues each refresh token for at most 90 days, and a refresh can
# only shorten that. A grant whose access token expired longer ago than that
# has no usable refresh token left, whether or not anyone tried it.
SHOPIFY_REFRESH_TOKEN_LIFETIME = timedelta(days=90)


def lock_shopify_store(db: Session, *, provider: str, store: str) -> None:
    """Create the store's lock row if needed and lock it until commit.

    ``provider`` is the OAuth provider row (one Shopify app) and ``store`` the
    store label (``acme`` for acme.myshopify.com). As with the OAuth
    revocation fences, the insert comes first so the lock never depends on a
    row that may not exist yet: PostgreSQL makes a concurrent insert of the
    same key wait for the first one's commit, and SQLite's single writer
    serializes the write statements outright. On SQLite that means every
    write waits while a store is locked, as it already does during an actor
    refresh.
    """
    dialect = db.get_bind().dialect.name
    insert: Any
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    elif dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    else:
        raise RuntimeError("Shopify store locks require PostgreSQL or SQLite")
    key = {"provider": provider.lower(), "store": store.lower()}
    db.execute(
        insert(ShopifyStoreLock)
        .values(**key)
        .on_conflict_do_nothing(index_elements=["provider", "store"])
    )
    db.query(ShopifyStoreLock).filter_by(**key).with_for_update().one()


def shopify_store_holder(
    db: Session,
    *,
    provider: str,
    store: str,
    user_id: int | None,
    resource_owner_key: str | None,
    connector_key: str,
) -> Literal["another_owner", "another_connection"] | None:
    """Who else holds a live grant of ``provider`` for ``store``, if anyone.

    An owner is a user's own connection (``resource_owner_key`` is NULL) or
    one actor of that user, the same split ``UserOAuth`` stores grants by.
    The same owner's grant under another connector (``UserOAuth.provider``)
    is another connection: a reconnect replaces only the same connector's
    grant, so both would be live and retire each other. A grant whose access
    token was cleared (a dead refresh token) or cannot be decrypted no longer
    holds the store: its owner has to reconnect anyway. Nor does one past
    Shopify's refresh token lifetime: an owner who stopped using the store
    never triggers the failed refresh that would clear it. Call it under
    :func:`lock_shopify_store` so the answer still holds when the grant is
    stored.
    """
    grants = db.query(UserOAuth).filter(
        func.lower(UserOAuth.oauth_provider) == provider.lower(),
        func.lower(UserOAuth.instance_url) == store.lower(),
    )
    now = datetime.now(timezone.utc)
    holder: Literal["another_owner", "another_connection"] | None = None
    for grant in grants:
        if not grant.access_token or _past_refresh_window(grant.expires_at, now):
            continue
        if (grant.user_id, grant.resource_owner_key) != (user_id, resource_owner_key):
            return "another_owner"
        if grant.provider != connector_key:
            holder = "another_connection"
    return holder


def _past_refresh_window(expires_at: Any, now: datetime) -> bool:
    """Whether a grant's refresh token has certainly expired.

    The refresh token was issued with the access token, no later than its
    expiry. A grant with no expiry is a non-expiring token and never lapses.
    """
    expires_at = coerce_utc(expires_at)
    if expires_at is None:
        return False
    return bool(expires_at + SHOPIFY_REFRESH_TOKEN_LIFETIME < now)


def shopify_store_of(provider_name: object, instance_url: object) -> str | None:
    """The store a grant of ``provider_name`` refreshes against, if Shopify's.

    A Shopify grant stores its store label in ``instance_url``.
    """
    if (
        isinstance(provider_name, str)
        and is_shopify_provider(provider_name)
        and isinstance(instance_url, str)
        and instance_url
    ):
        return instance_url
    return None
