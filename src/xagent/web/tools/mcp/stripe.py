import hashlib
import json
import logging
import os
import time
from typing import Any
from urllib.parse import quote

import requests
from mcp.server.fastmcp import FastMCP

from ...utils.graphql_errors import truncate_error_text
from .utils import setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stripe-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("stripe-mcp")

BASE_URL = "https://api.stripe.com/v1"
DEFAULT_TIMEOUT_SECONDS = 30
MAX_LIMIT = 100
# Stripe is rate-limited; on a 429 with a small Retry-After we wait once and
# retry rather than failing outright, mirroring jira.py's/intercom.py's/
# slack.py's bounded-retry policy for the same REST-shaped 429 signal.
MAX_RETRY_AFTER_SECONDS = 30


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _headers(idempotency_key: str | None = None) -> dict[str, str]:
    api_key = os.environ.get("STRIPE_API_KEY")
    if not api_key:
        raise ValueError("STRIPE_API_KEY environment variable is missing")
    headers = {"Authorization": f"Bearer {api_key}"}
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    return headers


def _clamp_limit(limit: int) -> int:
    return max(1, min(int(limit), MAX_LIMIT))


def _path_segment(value: str) -> str:
    """Percent-encode a value for safe interpolation into a URL path segment
    (e.g. a customer/charge/invoice id), matching jira.py's _path_segment /
    zoom.py's _encode_meeting_id. Percent-encoding - not a blocklist of "/",
    "?", "#" - is what actually prevents a value like "cus_1/../account"
    from escaping its intended path segment.
    """
    return quote(str(value), safe="")


def _idempotency_key(method: str, path: str, form_data: dict[str, Any] | None) -> str:
    """Derive a stable Idempotency-Key for a mutating request from its exact
    arguments, so an agent retry of the identical tool call (e.g. after a
    timeout or dropped connection) is deduped by Stripe instead of creating a
    second real refund/customer, while a call with genuinely different
    arguments still gets a different key.
    """
    canonical = json.dumps(
        {"method": method, "path": path, "form_data": form_data or {}},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _flatten_form_params(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten nested dict/list params into Stripe's bracket-notation form
    encoding, e.g. {"metadata": {"order_id": "6735"}} ->
    [("metadata[order_id]", "6735")].

    Stripe's API takes application/x-www-form-urlencoded bodies, not JSON --
    `requests` does not flatten nested dict/list values on its own, so
    passing a dict straight through as `data=` would serialize it as an
    unusable Python repr string instead of the bracket-keyed pairs Stripe
    expects.
    """
    items: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, sub_value in value.items():
            new_prefix = f"{prefix}[{key}]" if prefix else str(key)
            items.extend(_flatten_form_params(sub_value, new_prefix))
    elif isinstance(value, list):
        for index, sub_value in enumerate(value):
            items.extend(_flatten_form_params(sub_value, f"{prefix}[{index}]"))
    elif isinstance(value, bool):
        items.append((prefix, "true" if value else "false"))
    elif value is not None:
        items.append((prefix, value))
    return items


def _extract_error_detail(response: requests.Response) -> str | None:
    """Pull the human-readable message out of a Stripe error body.

    Stripe error responses are {"error": {"type", "code", "message",
    "param"}} -- the "message" field alone is more useful to the LLM than
    the raw envelope. Returns None if the body isn't in the expected shape,
    so the caller can fall back to the raw response text.
    """
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    message = error.get("message")
    return message if isinstance(message, str) and message else None


def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    form_data: dict[str, Any] | None = None,
) -> Any:
    idempotency_key = (
        _idempotency_key(method, path, form_data) if method == "POST" else None
    )
    for attempt in (0, 1):
        response = requests.request(
            method=method,
            url=f"{BASE_URL}{path}",
            headers=_headers(idempotency_key),
            params=params,
            data=_flatten_form_params(form_data) if form_data else None,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        if response.status_code == 429 and attempt == 0:
            try:
                retry_after = int(response.headers.get("Retry-After", "0"))
            except ValueError:
                retry_after = 0
            if 0 < retry_after <= MAX_RETRY_AFTER_SECONDS:
                time.sleep(retry_after)
                continue
        break

    if response.status_code >= 400:
        detail = _extract_error_detail(response)
        if detail is None:
            detail = truncate_error_text(response.text.strip())
        raise RuntimeError(
            f"Stripe API error (status {response.status_code}): {detail}"
        )

    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


def _paginated_results(payload: dict[str, Any], limit: int) -> tuple[list[Any], bool]:
    data = payload.get("data") or []
    truncated = bool(payload.get("has_more")) or len(data) > limit
    return data[:limit], truncated


@mcp.tool()
def stripe_get_account_info() -> str:
    """
    Get the Stripe account this connector's API key belongs to (id,
    business name, email, country, default currency). Use this for "my
    account" / "who am I" requests instead of asking the user for their
    Stripe account id.
    """
    try:
        result = _request("GET", "/account")
        return _success(
            account={
                "id": result.get("id"),
                "business_name": (result.get("business_profile") or {}).get("name"),
                "email": result.get("email"),
                "country": result.get("country"),
                "default_currency": result.get("default_currency"),
            }
        )
    except Exception as e:
        logger.error(f"Error fetching authenticated Stripe account: {e}")
        return _error(str(e))


@mcp.tool()
def stripe_get_balance() -> str:
    """
    Get the current Stripe balance (available and pending amounts per
    currency) for the connected account.
    """
    try:
        result = _request("GET", "/balance")
        return _success(
            available=result.get("available"), pending=result.get("pending")
        )
    except Exception as e:
        logger.error(f"Error fetching Stripe balance: {e}")
        return _error(str(e))


@mcp.tool()
def stripe_list_customers(
    email: str = "", limit: int = 10, starting_after: str = ""
) -> str:
    """
    List customers, most recently created first.
    email: optional exact-match filter on the customer's email.
    starting_after: a customer id from a previous page, to page forward.
    """
    try:
        max_results = _clamp_limit(limit)
        params: dict[str, Any] = {"limit": max_results}
        if email:
            params["email"] = email
        if starting_after:
            params["starting_after"] = starting_after
        result = _request("GET", "/customers", params=params)
        customers, truncated = _paginated_results(result, max_results)
        return _success(customers=customers, truncated=truncated)
    except Exception as e:
        logger.error(f"Error listing Stripe customers: {e}")
        return _error(str(e))


@mcp.tool()
def stripe_get_customer(customer_id: str) -> str:
    """
    Get one customer's full details by id.
    customer_id: a Stripe customer id, e.g. "cus_ABC123".
    """
    try:
        result = _request("GET", f"/customers/{_path_segment(customer_id)}")
        return _success(customer=result)
    except Exception as e:
        logger.error(f"Error fetching Stripe customer {customer_id}: {e}")
        return _error(str(e))


@mcp.tool()
def stripe_create_customer(
    name: str = "",
    email: str = "",
    description: str = "",
    metadata: dict[str, str] | None = None,
) -> str:
    """
    Create a new customer.
    name/email/description: optional customer profile fields.
    metadata: optional string key/value pairs to attach for your own
    bookkeeping, e.g. {"internal_id": "6735"}.
    """
    try:
        form_data: dict[str, Any] = {}
        if name:
            form_data["name"] = name
        if email:
            form_data["email"] = email
        if description:
            form_data["description"] = description
        if metadata:
            form_data["metadata"] = metadata
        result = _request("POST", "/customers", form_data=form_data)
        return _success(customer=result)
    except Exception as e:
        logger.error(f"Error creating Stripe customer: {e}")
        return _error(str(e))


@mcp.tool()
def stripe_list_charges(
    customer_id: str = "", limit: int = 10, starting_after: str = ""
) -> str:
    """
    List charges, most recently created first.
    customer_id: optional Stripe customer id to filter to one customer's
    charges.
    starting_after: a charge id from a previous page, to page forward.
    """
    try:
        max_results = _clamp_limit(limit)
        params: dict[str, Any] = {"limit": max_results}
        if customer_id:
            params["customer"] = customer_id
        if starting_after:
            params["starting_after"] = starting_after
        result = _request("GET", "/charges", params=params)
        charges, truncated = _paginated_results(result, max_results)
        return _success(charges=charges, truncated=truncated)
    except Exception as e:
        logger.error(f"Error listing Stripe charges: {e}")
        return _error(str(e))


@mcp.tool()
def stripe_get_charge(charge_id: str) -> str:
    """
    Get one charge's full details by id.
    charge_id: a Stripe charge id, e.g. "ch_ABC123".
    """
    try:
        result = _request("GET", f"/charges/{_path_segment(charge_id)}")
        return _success(charge=result)
    except Exception as e:
        logger.error(f"Error fetching Stripe charge {charge_id}: {e}")
        return _error(str(e))


@mcp.tool()
def stripe_create_refund(
    charge_id: str = "",
    payment_intent_id: str = "",
    amount: int | None = None,
    reason: str = "",
) -> str:
    """
    Refund a charge, in full or in part.
    charge_id: a Stripe charge id, e.g. "ch_ABC123". Provide this or
    payment_intent_id (at least one is required).
    payment_intent_id: a Stripe payment intent id, e.g. "pi_ABC123", as an
    alternative way to identify the payment to refund.
    amount: optional amount to refund in the currency's smallest unit (e.g.
    cents for USD); omit to refund the full remaining amount.
    reason: optional one of "duplicate", "fraudulent", or
    "requested_by_customer".
    """
    try:
        if not charge_id and not payment_intent_id:
            return _error("Either charge_id or payment_intent_id is required")
        form_data: dict[str, Any] = {}
        if charge_id:
            form_data["charge"] = charge_id
        if payment_intent_id:
            form_data["payment_intent"] = payment_intent_id
        if amount is not None:
            form_data["amount"] = amount
        if reason:
            form_data["reason"] = reason
        result = _request("POST", "/refunds", form_data=form_data)
        return _success(refund=result)
    except Exception as e:
        logger.error(f"Error creating Stripe refund: {e}")
        return _error(str(e))


@mcp.tool()
def stripe_list_payment_intents(
    customer_id: str = "", limit: int = 10, starting_after: str = ""
) -> str:
    """
    List payment intents, most recently created first.
    customer_id: optional Stripe customer id to filter to one customer's
    payment intents.
    starting_after: a payment intent id from a previous page, to page
    forward.
    """
    try:
        max_results = _clamp_limit(limit)
        params: dict[str, Any] = {"limit": max_results}
        if customer_id:
            params["customer"] = customer_id
        if starting_after:
            params["starting_after"] = starting_after
        result = _request("GET", "/payment_intents", params=params)
        payment_intents, truncated = _paginated_results(result, max_results)
        return _success(payment_intents=payment_intents, truncated=truncated)
    except Exception as e:
        logger.error(f"Error listing Stripe payment intents: {e}")
        return _error(str(e))


@mcp.tool()
def stripe_list_invoices(
    customer_id: str = "",
    status: str = "",
    limit: int = 10,
    starting_after: str = "",
) -> str:
    """
    List invoices, most recently created first.
    customer_id: optional Stripe customer id to filter to one customer's
    invoices.
    status: optional one of "draft", "open", "paid", "uncollectible", or
    "void".
    starting_after: an invoice id from a previous page, to page forward.
    """
    try:
        max_results = _clamp_limit(limit)
        params: dict[str, Any] = {"limit": max_results}
        if customer_id:
            params["customer"] = customer_id
        if status:
            params["status"] = status
        if starting_after:
            params["starting_after"] = starting_after
        result = _request("GET", "/invoices", params=params)
        invoices, truncated = _paginated_results(result, max_results)
        return _success(invoices=invoices, truncated=truncated)
    except Exception as e:
        logger.error(f"Error listing Stripe invoices: {e}")
        return _error(str(e))


@mcp.tool()
def stripe_get_invoice(invoice_id: str) -> str:
    """
    Get one invoice's full details by id.
    invoice_id: a Stripe invoice id, e.g. "in_ABC123".
    """
    try:
        result = _request("GET", f"/invoices/{_path_segment(invoice_id)}")
        return _success(invoice=result)
    except Exception as e:
        logger.error(f"Error fetching Stripe invoice {invoice_id}: {e}")
        return _error(str(e))


@mcp.tool()
def stripe_list_subscriptions(
    customer_id: str = "",
    status: str = "",
    limit: int = 10,
    starting_after: str = "",
) -> str:
    """
    List subscriptions, most recently created first.
    customer_id: optional Stripe customer id to filter to one customer's
    subscriptions.
    status: optional one of "active", "past_due", "unpaid", "canceled",
    "incomplete", "incomplete_expired", "trialing", "paused", or "all"
    (if omitted, Stripe returns every non-canceled status, not just
    "active" -- pass "active" explicitly to filter to active-only).
    starting_after: a subscription id from a previous page, to page forward.
    """
    try:
        max_results = _clamp_limit(limit)
        params: dict[str, Any] = {"limit": max_results}
        if customer_id:
            params["customer"] = customer_id
        if status:
            params["status"] = status
        if starting_after:
            params["starting_after"] = starting_after
        result = _request("GET", "/subscriptions", params=params)
        subscriptions, truncated = _paginated_results(result, max_results)
        return _success(subscriptions=subscriptions, truncated=truncated)
    except Exception as e:
        logger.error(f"Error listing Stripe subscriptions: {e}")
        return _error(str(e))


@mcp.tool()
def stripe_list_products(
    active: bool | None = None, limit: int = 10, starting_after: str = ""
) -> str:
    """
    List products, most recently created first.
    active: optional filter to only active (true) or inactive (false)
    products; omit to return both.
    starting_after: a product id from a previous page, to page forward.
    """
    try:
        max_results = _clamp_limit(limit)
        params: dict[str, Any] = {"limit": max_results}
        if active is not None:
            # `requests` serializes a bare bool as "True"/"False" in query
            # strings, but Stripe's API only accepts lowercase "true"/"false".
            params["active"] = "true" if active else "false"
        if starting_after:
            params["starting_after"] = starting_after
        result = _request("GET", "/products", params=params)
        products, truncated = _paginated_results(result, max_results)
        return _success(products=products, truncated=truncated)
    except Exception as e:
        logger.error(f"Error listing Stripe products: {e}")
        return _error(str(e))


@mcp.tool()
def stripe_list_prices(
    product_id: str = "",
    active: bool | None = None,
    limit: int = 10,
    starting_after: str = "",
) -> str:
    """
    List prices, most recently created first.
    product_id: optional Stripe product id to filter to one product's
    prices.
    active: optional filter to only active (true) or inactive (false)
    prices; omit to return both.
    starting_after: a price id from a previous page, to page forward.
    """
    try:
        max_results = _clamp_limit(limit)
        params: dict[str, Any] = {"limit": max_results}
        if product_id:
            params["product"] = product_id
        if active is not None:
            # `requests` serializes a bare bool as "True"/"False" in query
            # strings, but Stripe's API only accepts lowercase "true"/"false".
            params["active"] = "true" if active else "false"
        if starting_after:
            params["starting_after"] = starting_after
        result = _request("GET", "/prices", params=params)
        prices, truncated = _paginated_results(result, max_results)
        return _success(prices=prices, truncated=truncated)
    except Exception as e:
        logger.error(f"Error listing Stripe prices: {e}")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
