"""Shopify app webhooks (#2810).

A public Shopify app must handle ``app/uninstalled`` and the three mandatory
compliance topics (``customers/data_request``, ``customers/redact`` and
``shop/redact``) before it can be listed. Shopify posts all four to
``POST /api/shopify/webhooks``:

- ``app/uninstalled``: Shopify has already revoked the store's token, so every
  grant of this deployment's Shopify app for the store is deleted; its owners
  see the connector ask them to reconnect.
- ``shop/redact`` (about 48 hours after an uninstall): the same deletion, again,
  in case anything was reconnected or left behind. The grant is everything
  this deployment keeps about a store.
- ``customers/redact`` and ``customers/data_request``: no per-customer records
  are kept, so these are acknowledged and logged without the customer's
  details.

Shopify signs the raw body, not the headers (``verify_shopify_webhook``), so
the store comes from the signed body, and a topic that deletes data must carry
its own body shape: a signed customer request relabelled ``shop/redact``
deletes nothing. Every topic is idempotent, so retries and repeated deliveries
are harmless.

A multi-region deployment (xagent-saas) reuses :func:`read_shopify_webhook`
and :func:`apply_shopify_webhook` to apply one delivery on every region, since
Shopify sends it to a single address and no region knows where a store's
grants live.
"""

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy import func
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from ..models.database import get_db
from ..models.oauth_provider import OAuthProvider
from ..models.user_oauth import UserOAuth
from ..oauth_account_hosts import OAuthAccountError, normalize_oauth_account
from ..shopify_oauth import (
    SHOPIFY_HOST_SUFFIX,
    SHOPIFY_PROVIDER,
    verify_shopify_webhook,
)
from .auth import _resolve_oauth_secret

logger = logging.getLogger(__name__)

SHOPIFY_WEBHOOK_PATH = "/api/shopify/webhooks"

TOPIC_APP_UNINSTALLED = "app/uninstalled"
TOPIC_SHOP_REDACT = "shop/redact"
TOPIC_CUSTOMERS_REDACT = "customers/redact"
TOPIC_CUSTOMERS_DATA_REQUEST = "customers/data_request"
# Topics that delete the store's grants.
_STORE_DELETING_TOPICS = frozenset({TOPIC_APP_UNINSTALLED, TOPIC_SHOP_REDACT})
_CUSTOMER_TOPICS = frozenset({TOPIC_CUSTOMERS_REDACT, TOPIC_CUSTOMERS_DATA_REQUEST})
# Header values are unsigned; only a plain id is worth putting in a log line.
_LOGGABLE_ID = re.compile(r"[A-Za-z0-9-]{1,64}")

router = APIRouter(prefix="/api/shopify", tags=["shopify"])


class ShopifyWebhookRefused(Exception):
    """A delivery that must not be applied, with the status to answer it."""

    def __init__(self, status_code: int, reason: str) -> None:
        super().__init__(reason)
        self.status_code = status_code
        self.reason = reason


@dataclass(frozen=True)
class ShopifyWebhook:
    """A verified delivery. ``store`` is ``None`` for a topic we ignore."""

    topic: str
    store: str | None
    webhook_id: str | None


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


def _store_label(domain: object) -> str:
    """The store label of ``<label>.myshopify.com``; refuse anything else."""
    if not isinstance(domain, str):
        raise ShopifyWebhookRefused(400, "shop_invalid")
    label, _, suffix = domain.strip().lower().partition(".")
    try:
        account = normalize_oauth_account(label)
    except OAuthAccountError:
        account = None
    if account is None or suffix != SHOPIFY_HOST_SUFFIX:
        raise ShopifyWebhookRefused(400, "shop_invalid")
    return account


def _store_of(topic: str, payload: Mapping[str, Any]) -> str:
    """The store a payload names, once the payload fits ``topic``.

    ``app/uninstalled`` carries the shop resource (``myshopify_domain``); the
    compliance topics carry ``shop_domain``, and only the customer topics a
    ``customer``. Requiring each topic's own fields keeps a signed body of one
    topic from being replayed under the unsigned header of another.
    """
    if topic == TOPIC_APP_UNINSTALLED:
        return _store_label(payload.get("myshopify_domain"))
    is_customer_topic = topic in _CUSTOMER_TOPICS
    if is_customer_topic != isinstance(payload.get("customer"), dict):
        raise ShopifyWebhookRefused(400, "payload_mismatch")
    return _store_label(payload.get("shop_domain"))


def read_shopify_webhook(
    body: bytes, headers: Mapping[str, str], *, client_secret: str
) -> ShopifyWebhook:
    """Verify a delivery and read the store it is about.

    Raises :class:`ShopifyWebhookRefused`: 401 when the signature does not
    verify (Shopify's own check expects 401), 400 when a signed body does not
    fit its topic.
    """
    if not verify_shopify_webhook(
        body, headers.get("x-shopify-hmac-sha256"), client_secret=client_secret
    ):
        raise ShopifyWebhookRefused(401, "hmac_invalid")
    topic = headers.get("x-shopify-topic", "").strip().lower()
    raw_id = headers.get("x-shopify-webhook-id", "").strip()
    webhook_id = raw_id if _LOGGABLE_ID.fullmatch(raw_id) else None
    if topic not in _STORE_DELETING_TOPICS and topic not in _CUSTOMER_TOPICS:
        return ShopifyWebhook(topic=topic, store=None, webhook_id=webhook_id)
    try:
        payload = json.loads(body)
    except ValueError:
        raise ShopifyWebhookRefused(400, "payload_invalid") from None
    if not isinstance(payload, dict):
        raise ShopifyWebhookRefused(400, "payload_invalid")
    return ShopifyWebhook(
        topic=topic, store=_store_of(topic, payload), webhook_id=webhook_id
    )


def apply_shopify_webhook(db: Session, webhook: ShopifyWebhook) -> None:
    """Apply a verified delivery to this deployment's database and commit."""
    if webhook.store is None:
        logger.info(
            "Ignoring Shopify webhook topic %r (webhook %s)",
            webhook.topic[:64],
            webhook.webhook_id,
        )
        return
    if webhook.topic in _STORE_DELETING_TOPICS:
        deleted = (
            db.query(UserOAuth)
            .filter(
                func.lower(UserOAuth.oauth_provider) == SHOPIFY_PROVIDER,
                func.lower(UserOAuth.instance_url) == webhook.store,
            )
            .delete(synchronize_session=False)
        )
        db.commit()
        logger.info(
            "Shopify %s for store %s (webhook %s): deleted %d grant(s)",
            webhook.topic,
            webhook.store,
            webhook.webhook_id,
            deleted,
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
    answer a refused delivery with.
    """
    body = await request.body()
    client_secret = await run_in_threadpool(shopify_webhook_secret, db)
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


@router.post("/webhooks")
async def receive_shopify_webhook(
    request: Request, db: Session = Depends(get_db)
) -> Response:
    """Shopify's endpoint for ``app/uninstalled`` and the compliance topics."""
    result = await receive_verified_shopify_webhook(request, db)
    if isinstance(result, Response):
        return result
    return Response(status_code=200)
