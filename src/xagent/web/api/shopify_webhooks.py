"""Shopify app webhooks (#2810).

A public Shopify app must handle ``app/uninstalled`` and the three mandatory
compliance topics (``customers/data_request``, ``customers/redact`` and
``shop/redact``) before it can be listed. Shopify posts all four to
``POST /api/shopify/webhooks``:

- ``app/uninstalled``: Shopify has already revoked the store's token, so the
  tokens of every grant of this deployment's Shopify app for the store are
  cleared and the row is kept as a reconnect tombstone, the same as a grant
  whose refresh failed for good (tools/config.py): its owner sees the
  connector ask them to reconnect.
- ``shop/redact`` (about 48 hours after an uninstall): the store's grants are
  deleted, tombstones included. A grant (store label, shop id, owner) is the
  only record this module keeps per store.
- ``customers/redact`` and ``customers/data_request``: no per-customer records
  are kept, so these are acknowledged and logged without the customer's
  details. Customer data that connector tools returned into a conversation is
  the user's own task history and out of scope here.

A grant that holds a token and was written after the uninstall is kept: the uninstall revoked the
store's token, so only a new install writes one, and Shopify retries for hours
without keeping order, so a late delivery must not delete the grant of a
merchant who has since reconnected. The uninstall is taken to have happened
at ``X-Shopify-Triggered-At`` for ``app/uninstalled``, and, going by
Shopify's documented 48-hour delay, ``SHOP_REDACT_DELAY`` before it for
``shop/redact``. Without a usable time
every grant for the store is deleted.

The store is matched by its label (``<label>.myshopify.com`` in the body) or
by the shop's id, which the callback stores as the grant's identity
(``gid://shopify/Shop/<id>``). Grants of an administrator-created
``shopify-*`` provider row belong to another Shopify app, signed with another
secret, and are not touched.

Shopify signs the raw body, not the headers (``verify_shopify_webhook``), so
the store comes from the signed body, and a topic that deletes data must carry
its own body shape: a signed customer request relabelled ``shop/redact``
deletes nothing. Every topic is idempotent, so retries and repeated deliveries
are harmless.

A multi-region deployment (xagent-saas) reuses
:func:`receive_verified_shopify_webhook` to apply one delivery on every region, since
Shopify sends it to a single address and no region knows where a store's
grants live.
"""

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, cast

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy import func, or_
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from ..models.database import get_db
from ..models.oauth_provider import OAuthProvider
from ..models.user_oauth import UserOAuth
from ..shopify_oauth import (
    _SHOP_GID_PREFIX,
    SHOPIFY_PROVIDER,
    shopify_store_label,
    verify_shopify_webhook,
)
from .auth import _resolve_oauth_secret

logger = logging.getLogger(__name__)

SHOPIFY_WEBHOOK_PATH = "/api/shopify/webhooks"

TOPIC_APP_UNINSTALLED = "app/uninstalled"
TOPIC_SHOP_REDACT = "shop/redact"
TOPIC_CUSTOMERS_REDACT = "customers/redact"
TOPIC_CUSTOMERS_DATA_REQUEST = "customers/data_request"
# Topics that clear or delete the store's grants.
_STORE_DELETING_TOPICS = frozenset({TOPIC_APP_UNINSTALLED, TOPIC_SHOP_REDACT})
_CUSTOMER_TOPICS = frozenset({TOPIC_CUSTOMERS_REDACT, TOPIC_CUSTOMERS_DATA_REQUEST})
# Shopify sends shop/redact this long after the uninstall.
SHOP_REDACT_DELAY = timedelta(hours=48)
# Header values are unsigned; only a plain id is worth putting in a log line.
_LOGGABLE_ID = re.compile(r"[A-Za-z0-9-]{1,64}")

router = APIRouter(tags=["shopify"])


class ShopifyWebhookRefused(Exception):
    """A delivery that must not be applied, with the status to answer it."""

    def __init__(self, status_code: int, reason: str) -> None:
        super().__init__(reason)
        self.status_code = status_code
        self.reason = reason


@dataclass(frozen=True)
class ShopifyWebhook:
    """A verified delivery.

    ``store`` and ``shop_gid`` identify the store when the body names it;
    ``triggered_at`` is when Shopify says the event happened, if known.
    """

    topic: str
    store: str | None
    shop_gid: str | None
    triggered_at: datetime | None
    webhook_id: str | None

    @property
    def applies(self) -> bool:
        """Whether this deployment acts on (and passes on) the topic."""
        return self.topic in _STORE_DELETING_TOPICS or self.topic in _CUSTOMER_TOPICS


def shopify_webhook_secret(db: Session) -> str:
    """The Shopify app's client secret, or ``""`` when it is not configured.

    Resolved as the connector's OAuth client is: the provider row first, then
    ``SHOPIFY_CLIENT_SECRET``.
    """
    provider = (
        db.query(OAuthProvider)
        .filter(OAuthProvider.provider_name == SHOPIFY_PROVIDER)
        .first()
    )
    return _resolve_oauth_secret(
        SHOPIFY_PROVIDER,
        cast(str | None, provider.client_secret) if provider is not None else None,
        "CLIENT_SECRET",
    )


def _shop_gid(shop_id: object) -> str | None:
    """The grant identity of a shop id from the body, or ``None``."""
    if isinstance(shop_id, int) and not isinstance(shop_id, bool) and shop_id > 0:
        return f"{_SHOP_GID_PREFIX}{shop_id}"
    if isinstance(shop_id, str) and shop_id.isascii() and shop_id.isdigit():
        return f"{_SHOP_GID_PREFIX}{int(shop_id)}"
    return None


def _store_of(topic: str, payload: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """The store label and shop gid a payload names, once it fits ``topic``.

    ``app/uninstalled`` carries the shop resource (``id``,
    ``myshopify_domain``); the compliance topics carry ``shop_id`` and
    ``shop_domain``, and only the customer topics a ``customer``. Requiring the
    customer field to match the topic keeps a signed customer request from
    being replayed under the unsigned ``shop/redact`` header.
    """
    if topic == TOPIC_APP_UNINSTALLED:
        return shopify_store_label(payload.get("myshopify_domain")), _shop_gid(
            payload.get("id")
        )
    is_customer_topic = topic in _CUSTOMER_TOPICS
    if is_customer_topic != isinstance(payload.get("customer"), dict):
        raise ShopifyWebhookRefused(400, "payload_mismatch")
    return shopify_store_label(payload.get("shop_domain")), _shop_gid(
        payload.get("shop_id")
    )


def _triggered_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if moment.tzinfo is None:
        return None
    return moment


def read_shopify_webhook(
    body: bytes, headers: Mapping[str, str], *, client_secret: str
) -> ShopifyWebhook:
    """Verify a delivery and read the store it is about.

    ``headers`` are matched case-insensitively. Raises :class:`ShopifyWebhookRefused`: 401 when the signature does not
    verify (Shopify's own check expects 401), 400 when a signed body does not
    fit its topic.
    """
    headers = {name.lower(): value for name, value in headers.items()}
    if not verify_shopify_webhook(
        body, headers.get("x-shopify-hmac-sha256"), client_secret=client_secret
    ):
        raise ShopifyWebhookRefused(401, "hmac_invalid")
    topic = headers.get("x-shopify-topic", "").strip().lower()
    raw_id = headers.get("x-shopify-webhook-id", "").strip()
    webhook = ShopifyWebhook(
        topic=topic,
        store=None,
        shop_gid=None,
        triggered_at=_triggered_at(headers.get("x-shopify-triggered-at")),
        webhook_id=raw_id if _LOGGABLE_ID.fullmatch(raw_id) else None,
    )
    if not webhook.applies:
        return webhook
    try:
        payload = json.loads(body)
    except ValueError:
        raise ShopifyWebhookRefused(400, "payload_invalid") from None
    if not isinstance(payload, dict):
        raise ShopifyWebhookRefused(400, "payload_invalid")
    store, shop_gid = _store_of(topic, payload)
    return replace(webhook, store=store, shop_gid=shop_gid)


def _written_at(grant: UserOAuth) -> datetime | None:
    moment = grant.updated_at or grant.created_at
    if not isinstance(moment, datetime):
        return None
    # SQLite hands timestamps back naive; they are UTC (func.now()).
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def _clear_store_grants(db: Session, webhook: ShopifyWebhook) -> tuple[int, int]:
    """Tombstone (uninstall) or delete (redact) the store's grants.

    Returns how many grants were changed and how many were kept as written
    after the uninstall. A kept grant must still hold a token: a tombstone is
    always the uninstalled grant, even when clearing it moved ``updated_at``
    past the uninstall.
    """
    identity = []
    if webhook.store is not None:
        identity.append(func.lower(UserOAuth.instance_url) == webhook.store)
    if webhook.shop_gid is not None:
        identity.append(UserOAuth.provider_user_id == webhook.shop_gid)
    if not identity:
        return 0, 0
    grants = (
        db.query(UserOAuth)
        .filter(
            func.lower(UserOAuth.oauth_provider) == SHOPIFY_PROVIDER,
            or_(*identity),
        )
        .all()
    )
    uninstalled_at = webhook.triggered_at
    if uninstalled_at is not None and webhook.topic == TOPIC_SHOP_REDACT:
        uninstalled_at -= SHOP_REDACT_DELAY
    changed = kept = 0
    for grant in grants:
        written_at = _written_at(grant)
        if (
            grant.access_token
            and uninstalled_at is not None
            and written_at is not None
            and written_at > uninstalled_at
        ):
            kept += 1
        elif webhook.topic == TOPIC_SHOP_REDACT:
            db.delete(grant)
            changed += 1
        elif grant.access_token or grant.refresh_token:
            setattr(grant, "access_token", "")
            setattr(grant, "refresh_token", None)
            setattr(grant, "expires_at", None)
            changed += 1
    db.commit()
    return changed, kept


def apply_shopify_webhook(db: Session, webhook: ShopifyWebhook) -> None:
    """Apply a verified delivery to this deployment's database and commit."""
    if not webhook.applies:
        logger.info(
            "Ignoring Shopify webhook topic %r (webhook %s)",
            webhook.topic[:64],
            webhook.webhook_id,
        )
        return
    if webhook.topic in _STORE_DELETING_TOPICS:
        if webhook.store is None and webhook.shop_gid is None:
            # Signed by Shopify but naming no store we can look up (Shopify's
            # sample payloads carry placeholders). Refusing it would only make
            # Shopify retry the same body.
            logger.warning(
                "Shopify %s named no store (webhook %s); nothing changed",
                webhook.topic,
                webhook.webhook_id,
            )
            return
        changed, kept = _clear_store_grants(db, webhook)
        logger.info(
            "Shopify %s for store %s / %s (webhook %s): %s %d grant(s), "
            "kept %d written after the uninstall%s",
            webhook.topic,
            webhook.store,
            webhook.shop_gid,
            webhook.webhook_id,
            "deleted" if webhook.topic == TOPIC_SHOP_REDACT else "cleared",
            changed,
            kept,
            "" if webhook.triggered_at is not None else " (no event time)",
        )
        return
    # No per-customer records are kept; never log the customer's details.
    logger.info(
        "Shopify %s for store %s (webhook %s): no per-customer records are kept",
        webhook.topic,
        webhook.store,
        webhook.webhook_id,
    )


async def receive_verified_shopify_webhook(
    request: Request, db: Session
) -> tuple[bytes, ShopifyWebhook] | Response:
    """Read, verify and apply a delivery on this deployment.

    Returns the raw body and the delivery once applied, or the response to
    answer a refused delivery with. It ends the transaction of ``db`` (a
    rollback after reading the secret, then the apply's own commit), so the
    caller must not have pending changes on it.
    """
    body = await request.body()
    client_secret = await run_in_threadpool(shopify_webhook_secret, db)
    # End the read's transaction: a multi-region caller keeps this session
    # open while it passes the delivery on.
    await run_in_threadpool(db.rollback)
    if not client_secret:
        logger.warning("Shopify webhook received with no Shopify client configured")
        return Response(status_code=503)
    try:
        webhook = read_shopify_webhook(
            body, request.headers, client_secret=client_secret
        )
    except ShopifyWebhookRefused as exc:
        logger.warning("Refused a Shopify webhook: %s", exc.reason)
        return Response(status_code=exc.status_code)
    await run_in_threadpool(apply_shopify_webhook, db, webhook)
    return body, webhook


@router.post(SHOPIFY_WEBHOOK_PATH)
async def receive_shopify_webhook(
    request: Request, db: Session = Depends(get_db)
) -> Response:
    """Shopify's endpoint for ``app/uninstalled`` and the compliance topics."""
    result = await receive_verified_shopify_webhook(request, db)
    if isinstance(result, Response):
        return result
    return Response(status_code=200)
