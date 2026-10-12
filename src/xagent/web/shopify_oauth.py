"""Shopify's own steps in the per-account OAuth flow (#2810).

oauth_account_hosts.py derives Shopify's authorize and token endpoints from
the store label bound in the signed OAuth state. generic_oauth_callback calls
the three checks here, which are specific to Shopify:

- ``verify_shopify_callback``: Shopify signs every callback query with the
  app's client secret (``hmac``) and names the store that approved it
  (``shop``). Both are checked before the code is exchanged, so a forged
  callback, or a code approved on another store, never reaches the token
  endpoint.
- ``require_shopify_scopes``: a merchant can change the requested scopes on
  the consent screen, so the granted ``scope`` is checked before anything is
  stored.
- ``fetch_shopify_shop_identity``: Shopify has no userinfo endpoint for an
  offline token, and the Admin API authenticates with
  ``X-Shopify-Access-Token`` rather than a Bearer header, so the shop's
  GraphQL id becomes the grant's identity.
"""

import hashlib
import hmac
import re
from collections.abc import Iterable

import requests

from .oauth_provider_quirks import matches_provider_family

SHOPIFY_PROVIDER = "shopify"
SHOPIFY_HOST_SUFFIX = "myshopify.com"
# The same values as tools/mcp/shopify.py's SHOPIFY_REQUIRED_ADMIN_SCOPES and
# SHOPIFY_API_VERSION. That module starts an MCP server when imported, so the
# values are repeated here, and a test keeps the two in step.
SHOPIFY_REQUIRED_SCOPES = frozenset(
    {"write_products", "write_orders", "read_customers"}
)
SHOPIFY_API_VERSION = "2026-07"

_SHOP_GID_PREFIX = "gid://shopify/Shop/"
_SCOPE_SEPARATOR = re.compile(r"[\s,]+")
_IDENTITY_QUERY = "{ shop { id myshopifyDomain } }"


class ShopifyOAuthError(ValueError):
    """A Shopify callback step failed.

    ``reason`` is a code-shaped value for the callback rejection log;
    ``message`` is safe to show the merchant and never contains a token.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


def is_shopify_provider(provider: str) -> bool:
    return matches_provider_family(provider, SHOPIFY_PROVIDER)


def _store_host(account: str) -> str:
    return f"{account}.{SHOPIFY_HOST_SUFFIX}"


def verify_shopify_callback(
    query_items: Iterable[tuple[str, str]], *, client_secret: str, bound_account: str
) -> None:
    """Check the callback's signature, then that it names the bound store.

    Shopify signs the decoded query parameters other than ``hmac``, sorted by
    key and joined as ``key=value`` pairs with ``&``, using HMAC-SHA256 with
    the client secret, as a hex digest. A repeated parameter has no single
    signed value, so it fails verification.
    """
    params: dict[str, str] = {}
    repeated = False
    for key, value in query_items:
        repeated = repeated or key in params
        params[key] = value
    received = params.pop("hmac", "").strip().lower()
    if not received:
        raise ShopifyOAuthError(
            "hmac_missing", "Shopify's callback was not signed. Please try again."
        )
    message = "&".join(f"{key}={value}" for key, value in sorted(params.items()))
    expected = hmac.new(
        client_secret.encode(), message.encode(), hashlib.sha256
    ).hexdigest()
    if repeated or not hmac.compare_digest(expected.encode(), received.encode()):
        raise ShopifyOAuthError(
            "hmac_invalid",
            "Shopify's callback signature did not verify. Please try again.",
        )
    # Trusted only now that the signature covers it.
    shop = params.get("shop", "").strip().lower()
    if shop != _store_host(bound_account):
        raise ShopifyOAuthError(
            "shop_mismatch",
            "Shopify approved a different store than the one you entered. "
            "Connect again and approve access on that store.",
        )


def require_shopify_scopes(scope: object) -> str:
    """Check the token response's granted ``scope`` against the required set.

    Returns the granted scopes space-separated, the format the stored scope
    column's readers split on; Shopify sends them comma-separated.
    """
    granted = _parse_scopes(scope)
    missing = _missing_scopes(granted)
    if missing:
        raise ShopifyOAuthError(
            "scope_missing",
            "Shopify did not grant the access this connector needs: "
            f"{', '.join(missing)}. Connect again and approve every requested "
            "permission.",
        )
    return " ".join(sorted(granted))


def _parse_scopes(scope: object) -> frozenset[str]:
    """Return the granted scopes from Shopify's comma-separated ``scope``."""
    if isinstance(scope, str):
        parts: Iterable[object] = _SCOPE_SEPARATOR.split(scope)
    elif isinstance(scope, list):
        parts = scope
    else:
        return frozenset()
    return frozenset(
        stripped
        for part in parts
        if isinstance(part, str) and (stripped := part.strip())
    )


def _missing_scopes(granted: frozenset[str]) -> list[str]:
    """Return the required scopes ``granted`` lacks, sorted.

    Shopify grants ``read_x`` along with ``write_x`` without listing it, so a
    granted write scope satisfies its read scope.
    """
    return sorted(
        scope
        for scope in SHOPIFY_REQUIRED_SCOPES
        if scope not in granted
        and not (
            scope.startswith("read_")
            and f"write_{scope.removeprefix('read_')}" in granted
        )
    )


def fetch_shopify_shop_identity(account: str, access_token: str) -> str:
    """Return the GraphQL id of the shop ``access_token`` belongs to.

    ``account`` is the store label already validated and resolved for this
    callback by resolve_oauth_account_endpoints. The token must belong to that
    store; a token for any other shop is rejected.
    """
    store_host = _store_host(account)
    identity_unverified = ShopifyOAuthError(
        "identity_failed", "Shopify did not confirm which store was connected."
    )
    try:
        response = requests.post(
            f"https://{store_host}/admin/api/{SHOPIFY_API_VERSION}/graphql.json",
            json={"query": _IDENTITY_QUERY},
            headers={
                "X-Shopify-Access-Token": access_token,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=10.0,
            # A redirect would skip the store host's private-network check.
            allow_redirects=False,
        )
    except requests.RequestException:
        # The exception text is not echoed: it can carry request details.
        raise ShopifyOAuthError(
            "identity_unreachable",
            "Could not reach Shopify to confirm the connected store.",
        ) from None
    if response.status_code != 200:
        raise identity_unverified
    try:
        body = response.json()
    except ValueError:
        raise identity_unverified from None
    if not isinstance(body, dict) or body.get("errors"):
        raise identity_unverified
    data = body.get("data")
    shop = data.get("shop") if isinstance(data, dict) else None
    if not isinstance(shop, dict):
        raise identity_unverified
    shop_id = shop.get("id")
    if not isinstance(shop_id, str) or not shop_id.startswith(_SHOP_GID_PREFIX):
        raise identity_unverified
    domain = shop.get("myshopifyDomain")
    if not isinstance(domain, str) or domain.strip().lower() != store_host:
        raise ShopifyOAuthError(
            "shop_mismatch",
            "Shopify issued access for a different store than the one you entered.",
        )
    return shop_id
