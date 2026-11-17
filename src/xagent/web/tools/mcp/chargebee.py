"""Chargebee MCP connector, backed by the Chargebee REST API v2.

Read-only for now: customers, subscriptions, invoices and transactions. These
four resources and their list filters are the same on Product Catalog 1.0 and
2.0, so the tools work on either. Catalog tools (plans/addons or items/item
prices) and any write operation wait for the customer to confirm which ones
they need (xorbitsai/xagent-saas#2466).

Why REST rather than Chargebee's hosted MCP servers: those are served per site
and per region (``https://<site>.mcp[.eu|.au].chargebee.com/<server>``) with a
static Bearer key, which the catalog has no per-user shape for, and their write
tools need the Latest Product Catalog. Going through REST keeps this connector
on the existing stdio/api_key path, where the site and key are ordinary
per-user encrypted env.

Configuration: ``CHARGEBEE_SITE`` (the ``acme`` in ``acme.chargebee.com``) and
``CHARGEBEE_API_KEY``. Requests use HTTP Basic auth with the key as the user
name and an empty password.

Scope note: written against the published API v2 contract and covered by
mocked tests; it has not been exercised against a live site. The API reference
shows ``https://{site}.chargebee.com/api/v2`` for every data center, and that
has not been checked against an EU or AU site either.
"""

import json
import logging
import os
import re
import socket
import time
from datetime import date, datetime, timezone
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from ....config import get_tool_max_output_length
from ....core.utils import public_requests as requests
from ....core.utils.security import (
    PrivateNetworkHostError,
    redact_sensitive_text,
    reject_private_network_host,
)
from ...utils.graphql_errors import truncate_error_text
from .utils import clamp_limit, success_with_capped_dict, url_path_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chargebee-mcp")

mcp = FastMCP("chargebee-mcp")

READ_ONLY = ToolAnnotations(readOnlyHint=True)

# The public_requests facade's Session pins every connection to the address it
# validated as public, honours only the operator-trusted egress proxy, ignores
# ambient proxies and .netrc, and keeps the CA-bundle environment variables.
_session = requests.Session()

CHARGEBEE_DOMAIN = "chargebee.com"

# One DNS label, ASCII, internal hyphens only. Anchored with \A/\Z because $
# also matches before a trailing newline, which would reach the Host header.
_LABEL_PATTERN = re.compile(r"\A[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_BARE_DATE_PATTERN = re.compile(r"\A[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")
# The query string of a URL path: a transport error embeds the request URL,
# whose filters carry customer data (an email, a company name).
_URL_QUERY_PATTERN = re.compile(r"(/[^\s?'\"()<>]*)\?[^\s'\"()<>]+")

DEFAULT_TIMEOUT_SECONDS = 30
# Chargebee's list contract: limit 1-100, default 10.
MAX_LIMIT = 100
DEFAULT_LIMIT = 10
# A 429 is retried once, only when Chargebee names a short wait.
MAX_RETRY_AFTER_SECONDS = 30


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _env(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise ValueError(f"{name} environment variable is missing or empty")
    return value


def _site() -> str:
    """Return the validated site label.

    Accepts ``acme``, ``acme.chargebee.com`` or ``https://acme.chargebee.com``
    (case and surrounding whitespace are normalized), since users copy the
    address bar. Anything that could change *which host* is contacted --
    another domain, extra labels, a path, port, userinfo or a plain-http
    scheme -- is rejected.
    """
    raw = _env("CHARGEBEE_SITE").lower()
    raw = raw.removeprefix("https://").removesuffix("/")
    raw = raw.removesuffix(f".{CHARGEBEE_DOMAIN}")
    if not _LABEL_PATTERN.match(raw):
        raise ValueError(
            "CHARGEBEE_SITE must be your Chargebee site name -- the 'acme' in "
            f"acme.{CHARGEBEE_DOMAIN} (or that full hostname or https URL), "
            "with no path, port or other host"
        )
    return raw


def _base_url() -> str:
    """Compose the API root, refusing a site whose name resolves privately.

    ``<site>.chargebee.com`` is in a zone only Chargebee controls; this check
    turns a misconfiguration into an early, clear error. The public_requests
    session still resolves again and connects only to the address it checked.
    """
    hostname = f"{_site()}.{CHARGEBEE_DOMAIN}"
    try:
        for *_, sockaddr in socket.getaddrinfo(
            hostname, 443, socket.AF_UNSPEC, socket.SOCK_STREAM
        ):
            reject_private_network_host(str(sockaddr[0]))
    except PrivateNetworkHostError as exc:
        raise ValueError(f"CHARGEBEE_SITE is not allowed: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"CHARGEBEE_SITE host could not be resolved: {exc}") from exc
    return f"https://{hostname}/api/v2"


def _api_key() -> str:
    return _env("CHARGEBEE_API_KEY")


def _safe_text(value: Any) -> str:
    """Text safe to log and return to the model: redacted, key-free,
    query-free and bounded.

    redact_sensitive_text matches credential-shaped text only, and a
    Chargebee key has no fixed shape, so the configured key is also removed
    literally. URL query strings are dropped because list filters carry
    customer data.
    """
    text = redact_sensitive_text(str(value))
    text = _URL_QUERY_PATTERN.sub(r"\1?<query redacted>", text)
    key = (os.environ.get("CHARGEBEE_API_KEY") or "").strip()
    if key:
        text = text.replace(key, "[redacted]")
    return truncate_error_text(text)


def _retry_after(response: requests.Response) -> int | None:
    """The wait Chargebee asked for, when it is short enough to honour."""
    value = (response.headers.get("Retry-After") or "").strip()
    if not value.isdigit():
        return None
    seconds = int(value)
    return seconds if 1 <= seconds <= MAX_RETRY_AFTER_SECONDS else None


def _error_message(response: requests.Response) -> str:
    """An actionable, key-free message for a Chargebee error response.

    Chargebee's error body is flat: ``message``, ``api_error_code``,
    optional ``type`` and ``param``.
    """
    status = response.status_code
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        code = str(body.get("api_error_code") or "")
        detail = str(body.get("message") or "")
    else:
        code, detail = "", (response.text or "").strip()
    detail = _safe_text(detail) if detail else ""
    label = f"HTTP {status}, {code}" if code else f"HTTP {status}"

    if status == 401:
        hint = (
            "check that CHARGEBEE_API_KEY is a current key for this site "
            "(test and live sites have different keys)"
        )
    elif status == 403:
        hint = (
            "the API key does not have permission for this data; a "
            "restricted read-only key covers only the resource groups chosen "
            "when it was created"
        )
    elif status == 404 and code == "site_not_found":
        hint = "check that CHARGEBEE_SITE names an existing Chargebee site"
    elif status == 429 and code == "lock_timeout":
        hint = "another request is updating the same record; try again shortly"
    elif status == 429:
        hint = "Chargebee rate limited the request; wait before trying again"
    elif status == 503 and code == "site_read_only_mode":
        hint = "the site is temporarily in read-only mode"
    else:
        hint = ""
    parts = [part for part in (hint, detail) if part]
    return f"Chargebee API error ({label})" + (
        f": {' -- '.join(parts)}" if parts else ""
    )


def _send(path: str, params: dict[str, Any]) -> requests.Response:
    try:
        response = _session.request(
            method="GET",
            url=f"{_base_url()}{path}",
            # Never follow a redirect with the Basic credential attached.
            allow_redirects=False,
            auth=(_api_key(), ""),
            headers={"Accept": "application/json"},
            params=params,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        # The text can embed a proxy URL with credentials; rebuild it.
        raise RuntimeError(f"Chargebee request failed: {_safe_text(exc)}") from None
    if 300 <= response.status_code < 400:
        raise RuntimeError(
            f"Chargebee returned an unexpected redirect (HTTP {response.status_code}); "
            "refusing to follow it with credentials attached"
        )
    return response


def _get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """GET one API path and return its JSON object.

    Unset (None or "") parameters are dropped, so a caller passing "" to mean
    "no filter" does not send an empty filter. A 429 is retried once when
    Chargebee names a short wait; a GET is safe to repeat.
    """
    query = {k: v for k, v in (params or {}).items() if v is not None and v != ""}
    response = _send(path, query)
    if response.status_code == 429:
        wait = _retry_after(response)
        if wait is not None:
            time.sleep(wait)
            response = _send(path, query)
    if response.status_code >= 400:
        raise RuntimeError(_error_message(response))
    try:
        body = response.json()
    except ValueError:
        raise RuntimeError(
            f"Chargebee returned a non-JSON response for {path}"
        ) from None
    if not isinstance(body, dict):
        raise RuntimeError(f"Chargebee returned an unexpected response for {path}")
    return body


def _timestamp(value: str, field_name: str) -> int:
    """Unix seconds for a date (midnight UTC) or a datetime with an offset.

    Chargebee filters timestamps in Unix seconds. A datetime without an
    offset is refused rather than guessed: the site's time zone is unknown.
    """
    text = value.strip() if isinstance(value, str) else ""
    try:
        if _BARE_DATE_PATTERN.match(text):
            parsed = datetime.combine(date.fromisoformat(text), datetime.min.time())
            return int(parsed.replace(tzinfo=timezone.utc).timestamp())
        if "T" in text:
            parsed = datetime.fromisoformat(text)
            if parsed.tzinfo is not None:
                return int(parsed.timestamp())
    except ValueError:
        pass
    raise ValueError(
        f"{field_name} must be a date such as 2026-11-01 (midnight UTC) or a "
        "datetime with an offset such as 2026-11-01T09:00:00+08:00"
    )


def _range(
    field: str, after: str | None, before: str | None, names: tuple[str, str]
) -> dict[str, Any]:
    return {
        f"{field}[after]": _timestamp(after, names[0]) if after else None,
        f"{field}[before]": _timestamp(before, names[1]) if before else None,
    }


def _success_with_capped_list(field: str, payload: dict[str, Any]) -> str:
    """A success payload, halving ``payload[field]`` until it fits the output
    limit.

    Halved entries belong to the page already fetched, so they are not on the
    next page. Their ids are returned, so a caller can still fetch each one
    with the matching get tool -- even when a single record is larger than the
    limit and no smaller page would show it.
    """
    max_output_length = get_tool_max_output_length()
    items = payload[field]

    def _build(kept: list[Any]) -> str:
        if len(kept) == len(items):
            return _success(**payload, truncated=False)
        omitted = [item.get("id") for item in items[len(kept) :]]
        return _success(
            **{**payload, field: kept},
            truncated=True,
            omitted_ids=omitted,
            message=(
                f"Returned {len(kept)} of {len(items)} {field}; the rest did "
                "not fit the output size limit and are not on the next page. "
                "Their ids are in omitted_ids: fetch each with the get tool, "
                "or re-run with a smaller limit."
            ),
        )

    kept = items
    response = _build(kept)
    while len(response) > max_output_length and kept:
        kept = kept[: len(kept) // 2]
        response = _build(kept)
    return response


def _list(
    path: str,
    resource: str,
    field: str,
    *,
    limit: int,
    offset: str | None,
    include_deleted: bool,
    filters: dict[str, Any],
) -> str:
    """One page of a list endpoint, with each entry unwrapped.

    Chargebee wraps every entry by resource name (``{"customer": {...}}``,
    sometimes with related objects beside it) and pages by an opaque
    ``next_offset``, absent on the last page.
    """
    body = _get(
        path,
        {
            **filters,
            # Lowercase: Python's str(True) is "True".
            "include_deleted": "true" if include_deleted else None,
            "limit": clamp_limit(limit, max_limit=MAX_LIMIT),
            "offset": offset,
        },
    )
    entries = body.get("list")
    if not isinstance(entries, list):
        raise RuntimeError(f"Chargebee returned an unexpected response for {path}")
    items = []
    for entry in entries:
        item = entry.get(resource) if isinstance(entry, dict) else None
        if not isinstance(item, dict):
            raise RuntimeError(
                f"Chargebee returned an unexpected list entry for {path}"
            )
        items.append(item)
    next_offset = body.get("next_offset")
    next_offset = next_offset if isinstance(next_offset, str) and next_offset else None
    return _success_with_capped_list(
        field,
        {field: items, "next_offset": next_offset, "has_more": next_offset is not None},
    )


def _retrieve(collection: str, resource: str, resource_id: str, arg: str) -> str:
    body = _get(f"/{collection}/{url_path_id(resource_id, arg)}")
    item = body.get(resource)
    if not isinstance(item, dict):
        raise RuntimeError(f"Chargebee returned an unexpected {resource} response")
    return success_with_capped_dict(resource, item)


def _run(name: str, call: Callable[[], str]) -> str:
    try:
        return call()
    except Exception as exc:
        message = _safe_text(exc)
        logger.error("Error in %s: %s", name, message)
        return _error(message)


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------


@mcp.tool(annotations=READ_ONLY)
def chargebee_list_customers(
    email: str | None = None,
    company: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
    include_deleted: bool = False,
    limit: int = DEFAULT_LIMIT,
    offset: str | None = None,
) -> str:
    """
    List Chargebee customers.

    email, company: exact matches.
    created_after, created_before: a date such as 2026-11-01 (midnight UTC) or
        a datetime with an offset such as 2026-11-01T09:00:00+08:00.
    limit: entries per page, 1-100 (default 10).
    offset: the next_offset from the previous page, passed back unchanged;
        omit for the first page. has_more is false on the last page.
    include_deleted: also return deleted records (default false).

    If truncated is true, the page was trimmed to fit the output limit and
    the dropped entries are NOT on the next page; re-run with a smaller limit.
    """
    return _run(
        "chargebee_list_customers",
        lambda: _list(
            "/customers",
            "customer",
            "customers",
            limit=limit,
            offset=offset,
            include_deleted=include_deleted,
            filters={
                "email[is]": email,
                "company[is]": company,
                **_range(
                    "created_at",
                    created_after,
                    created_before,
                    ("created_after", "created_before"),
                ),
            },
        ),
    )


@mcp.tool(annotations=READ_ONLY)
def chargebee_get_customer(customer_id: str) -> str:
    """Fetch one Chargebee customer by id."""
    return _run(
        "chargebee_get_customer",
        lambda: _retrieve("customers", "customer", customer_id, "customer_id"),
    )


# ---------------------------------------------------------------------------
# Subscriptions
# ---------------------------------------------------------------------------


@mcp.tool(annotations=READ_ONLY)
def chargebee_list_subscriptions(
    customer_id: str | None = None,
    status: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
    include_deleted: bool = False,
    limit: int = DEFAULT_LIMIT,
    offset: str | None = None,
) -> str:
    """
    List Chargebee subscriptions.

    customer_id: only this customer's subscriptions.
    status: one of future, in_trial, active, non_renewing, paused, cancelled.
    created_after, created_before: a date such as 2026-11-01 (midnight UTC) or
        a datetime with an offset such as 2026-11-01T09:00:00+08:00.
    limit: entries per page, 1-100 (default 10).
    offset: the next_offset from the previous page, passed back unchanged;
        omit for the first page. has_more is false on the last page.
    include_deleted: also return deleted records (default false).

    If truncated is true, the page was trimmed to fit the output limit and
    the dropped entries are NOT on the next page; re-run with a smaller limit.
    """
    return _run(
        "chargebee_list_subscriptions",
        lambda: _list(
            "/subscriptions",
            "subscription",
            "subscriptions",
            limit=limit,
            offset=offset,
            include_deleted=include_deleted,
            filters={
                "customer_id[is]": customer_id,
                "status[is]": status,
                **_range(
                    "created_at",
                    created_after,
                    created_before,
                    ("created_after", "created_before"),
                ),
            },
        ),
    )


@mcp.tool(annotations=READ_ONLY)
def chargebee_get_subscription(subscription_id: str) -> str:
    """Fetch one Chargebee subscription by id."""
    return _run(
        "chargebee_get_subscription",
        lambda: _retrieve(
            "subscriptions", "subscription", subscription_id, "subscription_id"
        ),
    )


# ---------------------------------------------------------------------------
# Invoices
# ---------------------------------------------------------------------------


@mcp.tool(annotations=READ_ONLY)
def chargebee_list_invoices(
    customer_id: str | None = None,
    subscription_id: str | None = None,
    status: str | None = None,
    date_after: str | None = None,
    date_before: str | None = None,
    include_deleted: bool = False,
    limit: int = DEFAULT_LIMIT,
    offset: str | None = None,
) -> str:
    """
    List Chargebee invoices.

    customer_id, subscription_id: only invoices of this customer or
        subscription. Chargebee does not accept subscription_id when the site
        uses consolidated invoicing; filter by customer_id there.
    status: one of paid, posted, payment_due, not_paid, voided, pending.
        Overdue invoices are payment_due or not_paid.
    date_after, date_before: the invoice date, as a date such as 2026-11-01
        (midnight UTC) or a datetime with an offset such as
        2026-11-01T09:00:00+08:00.
    limit: entries per page, 1-100 (default 10).
    offset: the next_offset from the previous page, passed back unchanged;
        omit for the first page. has_more is false on the last page.
    include_deleted: also return deleted records (default false).

    If truncated is true, the page was trimmed to fit the output limit and
    the dropped entries are NOT on the next page; re-run with a smaller limit.
    """
    return _run(
        "chargebee_list_invoices",
        lambda: _list(
            "/invoices",
            "invoice",
            "invoices",
            limit=limit,
            offset=offset,
            include_deleted=include_deleted,
            filters={
                "customer_id[is]": customer_id,
                "subscription_id[is]": subscription_id,
                "status[is]": status,
                **_range(
                    "date", date_after, date_before, ("date_after", "date_before")
                ),
            },
        ),
    )


@mcp.tool(annotations=READ_ONLY)
def chargebee_get_invoice(invoice_id: str) -> str:
    """Fetch one Chargebee invoice by id, with its line items."""
    return _run(
        "chargebee_get_invoice",
        lambda: _retrieve("invoices", "invoice", invoice_id, "invoice_id"),
    )


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------


@mcp.tool(annotations=READ_ONLY)
def chargebee_list_transactions(
    customer_id: str | None = None,
    subscription_id: str | None = None,
    status: str | None = None,
    date_after: str | None = None,
    date_before: str | None = None,
    include_deleted: bool = False,
    limit: int = DEFAULT_LIMIT,
    offset: str | None = None,
) -> str:
    """
    List Chargebee transactions (payments, refunds and authorizations).

    customer_id, subscription_id: only transactions of this customer or
        subscription.
    status: one of in_progress, success, voided, failure, timeout,
        needs_attention.
    date_after, date_before: the transaction date, as a date such as
        2026-11-01 (midnight UTC) or a datetime with an offset such as
        2026-11-01T09:00:00+08:00.
    limit: entries per page, 1-100 (default 10).
    offset: the next_offset from the previous page, passed back unchanged;
        omit for the first page. has_more is false on the last page.
    include_deleted: also return deleted records (default false).

    If truncated is true, the page was trimmed to fit the output limit and
    the dropped entries are NOT on the next page; re-run with a smaller limit.
    """
    return _run(
        "chargebee_list_transactions",
        lambda: _list(
            "/transactions",
            "transaction",
            "transactions",
            limit=limit,
            offset=offset,
            include_deleted=include_deleted,
            filters={
                "customer_id[is]": customer_id,
                "subscription_id[is]": subscription_id,
                "status[is]": status,
                **_range(
                    "date", date_after, date_before, ("date_after", "date_before")
                ),
            },
        ),
    )


@mcp.tool(annotations=READ_ONLY)
def chargebee_get_transaction(transaction_id: str) -> str:
    """Fetch one Chargebee transaction by id."""
    return _run(
        "chargebee_get_transaction",
        lambda: _retrieve(
            "transactions", "transaction", transaction_id, "transaction_id"
        ),
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
