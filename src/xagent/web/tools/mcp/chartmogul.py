import json
import logging
import os
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Any, Literal, get_args

from mcp.server.fastmcp import FastMCP

from ....config import get_tool_max_output_length
from ....core.utils import public_requests as requests
from ....core.utils.security import redact_sensitive_text
from ...utils.graphql_errors import truncate_error_text
from .utils import clamp_limit, success_with_capped_dict, url_path_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chartmogul-mcp")


mcp = FastMCP("chartmogul-mcp")

# ChartMogul's REST API (dev.chartmogul.com) authenticates every request with
# HTTP Basic Auth: the account's API key as the username, empty password --
# confirmed against the official chartmogul-python SDK's Config class
# (chartmogul/api/config.py: `self.auth = (api_key, "")`), not guessed. There
# is no OAuth flow; the key is generated per-user from
# Profile -> API keys in their own ChartMogul account, same self-serve bar
# as an OAuth app. Unlike Stripe, ChartMogul's docs describe only this one
# key type per account -- no separate restricted/scoped key tier to enforce
# the way stripe.py checks for an "rk_"-prefixed key.
API_BASE_URL = "https://api.chartmogul.com/v1"
DEFAULT_TIMEOUT_SECONDS = 30
# ChartMogul's own page-size cap (dev.chartmogul.com/reference/list-customers:
# per_page defaults to, and maxes out at, 200); passing a larger value is
# rejected server-side rather than clamped.
MAX_PER_PAGE = 200
DEFAULT_PER_PAGE = 20

MetricName = Literal[
    "all",
    "mrr",
    "arr",
    "arpa",
    "asp",
    "customer-count",
    "customer-churn-rate",
    "mrr-churn-rate",
    "ltv",
]
MetricInterval = Literal["day", "week", "month", "quarter", "year"]

_METRIC_NAMES = get_args(MetricName)
_METRIC_INTERVALS = get_args(MetricInterval)
_METRIC_INTERVALS_BY_NAME = {
    metric: frozenset(_METRIC_INTERVALS) for metric in _METRIC_NAMES
}
_METRIC_INTERVALS_BY_NAME["asp"] = frozenset({"month", "quarter", "year"})
_METRIC_INTERVALS_BY_NAME["ltv"] = frozenset({"month"})
_METRIC_MONEY_FIELDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "all": (
        ("mrr", "arr", "ltv", "asp", "arpa"),
        (
            "current-mrr",
            "previous-mrr",
            "current-arr",
            "previous-arr",
            "current-ltv",
            "previous-ltv",
            "current-asp",
            "previous-asp",
            "current-arpa",
            "previous-arpa",
        ),
    ),
    "mrr": (
        (
            "mrr",
            "mrr-new-business",
            "mrr-expansion",
            "mrr-contraction",
            "mrr-churn",
            "mrr-reactivation",
        ),
        ("current", "previous"),
    ),
    "arr": (("arr",), ("current", "previous")),
    "arpa": (("arpa",), ("current", "previous")),
    "asp": (("asp",), ("current", "previous")),
    "ltv": (("ltv",), ("current", "previous")),
}


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _major_units(cents: Any, field: str) -> str:
    """Render an integer cents field as an exact two-decimal major-unit value."""
    if isinstance(cents, bool) or not isinstance(cents, int):
        raise TypeError(
            f"ChartMogul returned a non-integer value for {field}: "
            f"{type(cents).__name__}"
        )
    return f"{Decimal(cents) / Decimal(100):.2f}"


def _with_major_units(
    record: dict[str, Any],
    money_fields: tuple[str, ...],
    *,
    allow_fractional: bool = False,
) -> dict[str, Any]:
    """Normalize known money fields while retaining their source cents."""
    normalized = dict(record)
    for field in money_fields:
        if field not in record or record[field] is None:
            continue
        cents = record[field]
        normalized[f"{field.replace('-', '_')}_cents"] = cents
        if not allow_fractional:
            normalized[field] = _major_units(cents, field)
            continue
        if isinstance(cents, bool) or not isinstance(cents, (int, float)):
            raise TypeError(
                f"ChartMogul returned a non-numeric value for {field}: "
                f"{type(cents).__name__}"
            )
        decimal_cents = Decimal(str(cents))
        if not decimal_cents.is_finite():
            raise TypeError(
                f"ChartMogul returned a non-finite value for {field}: {cents}"
            )
        normalized[field] = f"{decimal_cents / Decimal(100):.2f}"
    return normalized


def _customer_with_major_units(customer: dict[str, Any]) -> dict[str, Any]:
    """Make ChartMogul customer MRR/ARR safe for user-facing consumption."""
    return _with_major_units(customer, ("mrr", "arr"))


def _opportunity_with_major_units(opportunity: dict[str, Any]) -> dict[str, Any]:
    """Add an exact user-facing amount while preserving the explicit raw field."""
    normalized = dict(opportunity)
    cents = opportunity.get("amount_in_cents")
    if cents is not None:
        normalized["amount"] = _major_units(cents, "amount_in_cents")
    return normalized


def _entries_with_major_units(
    result: dict[str, Any],
    endpoint: str,
    normalizer: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Normalize monetary fields on one validated customers/opportunities page."""
    entries: list[dict[str, Any]] = []
    for entry in result["entries"]:
        if not isinstance(entry, dict):
            raise TypeError(
                f"ChartMogul returned an unexpected response for {endpoint}"
            )
        entries.append(normalizer(entry))
    return {**result, "entries": entries}


def _metrics_with_major_units(
    result: Any, metric: str, endpoint: str
) -> dict[str, Any]:
    """Validate a metrics response and normalize only its monetary values."""
    if not isinstance(result, dict):
        raise TypeError(f"ChartMogul returned an unexpected response for {endpoint}")
    entries = result.get("entries")
    summary = result.get("summary")
    if not isinstance(entries, list) or not isinstance(summary, dict):
        raise TypeError(f"ChartMogul returned an unexpected response for {endpoint}")
    if any(not isinstance(entry, dict) for entry in entries):
        raise TypeError(f"ChartMogul returned an unexpected response for {endpoint}")

    entry_money_fields, summary_money_fields = _METRIC_MONEY_FIELDS.get(
        metric, ((), ())
    )

    return {
        **result,
        "entries": [
            _with_major_units(entry, entry_money_fields, allow_fractional=True)
            for entry in entries
        ],
        "summary": _with_major_units(
            summary, summary_money_fields, allow_fractional=True
        ),
    }


def _iso_date(value: str, field: str) -> str:
    """Require a canonical YYYY-MM-DD calendar date."""
    normalized = value.strip()
    try:
        parsed = date.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{field} must be a valid date in YYYY-MM-DD format") from exc
    if parsed.isoformat() != normalized:
        raise ValueError(f"{field} must be a valid date in YYYY-MM-DD format")
    return normalized


def _success_with_capped_list(list_field: str, payload: dict[str, Any]) -> str:
    """Build a success payload from a ChartMogul cursor-paginated list
    response, halving ``payload[list_field]`` until it fits the platform's
    output limit.

    Mirrors deputy.py's ``_success_with_capped_list``, including its
    explicit "data lost" message when halving actually ran: ChartMogul's
    ``cursor``/``has_more`` describe the *already-fetched* page from the one
    API call this made, so a halved-away item is gone for this call, not
    recoverable by resuming from that same cursor (which fetches the *next*
    page, not the rest of this one) -- there's no cursor a caller could
    retry with to get the dropped items back.
    """
    max_output_length = get_tool_max_output_length()
    items = payload.get(list_field) or []

    def _build(items: list[Any], truncated: bool, halved: bool) -> str:
        extra: dict[str, Any] = {}
        if halved:
            extra["message"] = (
                f"Returned {len(items)} {list_field} out of the full page; the "
                "rest did not fit the output size limit and cannot be recovered "
                "via this tool call (a smaller per_page avoids this)."
            )
        return _success(
            **{**payload, list_field: items, "truncated": truncated, **extra}
        )

    halved = False
    response = _build(items, False, halved)
    while len(response) > max_output_length and items:
        items = items[: len(items) // 2]
        halved = True
        response = _build(items, True, halved)
    if halved and len(response) > max_output_length:
        # Even an empty item list didn't buy back enough room (an unusually
        # large cursor/other payload field, or a very small configured
        # limit) -- drop the added "message" text itself as a last resort,
        # matching deputy.py's identical fallback.
        response = _build(items, True, False)
    return response


def _api_key() -> str:
    # Stripped, not just a bare os.environ.get(): a stray leading/trailing
    # newline or space in the injected key (e.g. from a copy-pasted env
    # value in a manual launch_config override) would otherwise silently
    # produce a malformed Basic Auth header rather than the clear "missing"
    # error below.
    api_key = (os.environ.get("CHARTMOGUL_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("CHARTMOGUL_API_KEY environment variable is missing or empty")
    return api_key


def _extract_error_detail(response: requests.Response) -> str | None:
    """Pull a human-readable message out of a ChartMogul error body, trying
    a few plausible key names rather than assuming one fixed shape. Returns
    None if the body isn't in any of the expected shapes, so the caller
    falls back to the raw response text.
    """
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    for key in ("message", "error", "errors"):
        detail = payload.get(key)
        if isinstance(detail, str) and detail:
            return detail
        # "errors" in particular can be a list of per-field messages (e.g.
        # {"errors": ["external_id can't be blank"]}), not just a dict.
        if isinstance(detail, (dict, list)) and detail:
            return json.dumps(detail, ensure_ascii=False)
    return None


def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_data: Any = None,
) -> Any:
    try:
        response = requests.request(
            method=method,
            url=f"{API_BASE_URL}{path}",
            auth=(_api_key(), ""),
            # Drop "" as well as None: every list filter below is optional,
            # and an LLM tool-call passing e.g. email="" to mean "no filter"
            # would otherwise become a real `?email=` query param instead of
            # being omitted.
            params={
                k: v for k, v in (params or {}).items() if v is not None and v != ""
            },
            json=json_data,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        # A connection/timeout/proxy failure's message can itself embed
        # sensitive data -- e.g. a ProxyError echoing the ambient
        # explicitly trusted proxy URL, which may embed user:pass@ credentials --
        # matches stripe.py's/posthog.py's identical fix for the same
        # shared proxy-env exposure surface.
        raise RuntimeError(
            f"ChartMogul request failed: {truncate_error_text(redact_sensitive_text(str(exc)))}"
        ) from exc
    if response.status_code >= 400:
        detail = _extract_error_detail(response)
        if detail is None:
            detail = response.text.strip()
        # The response body is host-controlled content, not something this
        # module wrote -- if it happens to echo request headers (e.g. a
        # misconfigured proxy/WAF/gateway error page), redact the Basic
        # Auth credential before it reaches logs or the LLM's context,
        # matching stripe.py's/posthog.py's identical treatment.
        detail = truncate_error_text(redact_sensitive_text(detail))
        raise RuntimeError(
            f"ChartMogul API error (status {response.status_code})"
            + (f": {detail}" if detail else "")
        )
    if response.status_code == 204 or not response.content:
        return {}
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"ChartMogul returned a 2xx response with a non-JSON body: {exc}"
        ) from exc


def _clamp_per_page(per_page: int) -> int:
    return clamp_limit(per_page, max_limit=MAX_PER_PAGE)


def _validated_dict(result: Any, endpoint: str) -> dict[str, Any]:
    """Return ``result`` if it's a dict, else raise.

    Every caller is inside a ``try/except Exception -> _error(str(e))``
    block, so raising here (rather than returning a ``dict | str`` union
    every call site would have to narrow) reaches the caller's own error
    envelope unchanged.
    """
    if not isinstance(result, dict):
        raise RuntimeError(f"ChartMogul returned an unexpected response for {endpoint}")
    return result


def _validated_list_result(result: Any, endpoint: str) -> dict[str, Any]:
    """Return ``result`` with ``entries`` normalized to a list, else raise.

    A present-but-``null`` ``entries`` field is treated the same as an
    empty list rather than an error, in case ChartMogul ever returns
    ``null`` instead of ``[]`` for a zero-result page. A *missing*
    ``entries`` key is still rejected as an unexpected shape -- unlike an
    explicit ``null``, its absence isn't evidence of "empty", just of a
    response that isn't a page of this resource at all.
    """
    if not isinstance(result, dict) or "entries" not in result:
        raise RuntimeError(f"ChartMogul returned an unexpected response for {endpoint}")
    entries = result["entries"]
    if entries is None:
        entries = []
    elif not isinstance(entries, list):
        raise RuntimeError(f"ChartMogul returned an unexpected response for {endpoint}")
    return {**result, "entries": entries}


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------


@mcp.tool()
def chartmogul_list_customers(
    data_source_uuid: str | None = None,
    external_id: str | None = None,
    email: str | None = None,
    status: str | None = None,
    system: str | None = None,
    cursor: str | None = None,
    per_page: int = DEFAULT_PER_PAGE,
) -> str:
    """
    List customers (GET /customers). Returns a page of entries plus a
    ``cursor``/``has_more`` pair -- pass the returned ``cursor`` back in to
    fetch the next page.

    Returned ``mrr`` and ``arr`` values are exact two-decimal strings in each
    customer's ``currency``. Source integers use the explicit names
    ``mrr_cents`` and ``arr_cents``.

    data_source_uuid: optional, restrict to one connected data source.
    external_id: optional, the customer's id in the connected billing
    system (e.g. a Stripe customer id).
    email: optional, filter by the customer's email address.
    status: optional, one of "New_Lead", "Working_Lead", "Qualified_Lead",
    "Unqualified_Lead", "Active", "Past_Due", "Cancelled".
    system: optional, filter by billing system name (e.g. "Stripe",
    "Recurly", "Custom") -- case-sensitive.
    cursor: optional, from a previous call's response, to fetch the next
    page.
    per_page: page size, 1-200 (ChartMogul's own server-side cap).
    """
    try:
        result = _request(
            "GET",
            "/customers",
            params={
                "data_source_uuid": data_source_uuid,
                "external_id": external_id,
                "email": email,
                "status": status,
                "system": system,
                "cursor": cursor,
                "per_page": _clamp_per_page(per_page),
            },
        )
        result = _validated_list_result(result, "/customers")
        result = _entries_with_major_units(
            result, "/customers", _customer_with_major_units
        )
        return _success_with_capped_list("entries", result)
    except Exception as e:
        logger.error(f"Error listing ChartMogul customers: {e}", exc_info=True)
        return _error(str(e))


@mcp.tool()
def chartmogul_create_customer(data: dict[str, Any]) -> str:
    """
    Create a customer (POST /customers).

    data: a ChartMogul customer object. Must include "data_source_uuid" and
    "external_id" (the id from your billing/CRM system) at minimum; commonly
    also "name", "email", "company", "country", "city", "state", "zip".
    Returned ``mrr``/``arr`` values use exact two-decimal major units; source
    integers use ``mrr_cents``/``arr_cents``.
    """
    try:
        result = _request("POST", "/customers", json_data=data)
        result = _validated_dict(result, "/customers")
        result = _customer_with_major_units(result)
        return success_with_capped_dict("customer", result)
    except Exception as e:
        logger.error(f"Error creating ChartMogul customer: {e}", exc_info=True)
        return _error(str(e))


@mcp.tool()
def chartmogul_get_customer(uuid: str) -> str:
    """
    Get one customer by uuid (GET /customers/{uuid}).

    Returned ``mrr`` and ``arr`` values are exact two-decimal strings in the
    customer's ``currency``; source integers use ``mrr_cents``/``arr_cents``.

    uuid: the customer's ChartMogul uuid (e.g. "cus_00000000-0000-0000-0000-000000000000").
    """
    try:
        safe_uuid = url_path_id(uuid, "uuid")
        result = _request("GET", f"/customers/{safe_uuid}")
        result = _validated_dict(result, "/customers")
        result = _customer_with_major_units(result)
        return success_with_capped_dict("customer", result)
    except Exception as e:
        logger.error(f"Error fetching ChartMogul customer {uuid}: {e}", exc_info=True)
        return _error(str(e))


@mcp.tool()
def chartmogul_update_customer(uuid: str, data: dict[str, Any]) -> str:
    """
    Update a customer (PATCH /customers/{uuid}).

    uuid: the customer's ChartMogul uuid.
    data: fields to change, e.g. {"company": "Acme Inc.", "country": "US"}.
    Returned ``mrr``/``arr`` values use exact two-decimal major units; source
    integers use ``mrr_cents``/``arr_cents``.
    """
    try:
        safe_uuid = url_path_id(uuid, "uuid")
        result = _request("PATCH", f"/customers/{safe_uuid}", json_data=data)
        result = _validated_dict(result, "/customers")
        result = _customer_with_major_units(result)
        return success_with_capped_dict("customer", result)
    except Exception as e:
        logger.error(f"Error updating ChartMogul customer {uuid}: {e}", exc_info=True)
        return _error(str(e))


@mcp.tool()
def chartmogul_delete_customer(uuid: str) -> str:
    """
    Permanently delete a customer, including all its associated data
    (subscriptions, invoices, activities) (DELETE /customers/{uuid}). This
    cannot be undone -- confirm with the user before calling this.

    uuid: the customer's ChartMogul uuid.
    """
    try:
        safe_uuid = url_path_id(uuid, "uuid")
        _request("DELETE", f"/customers/{safe_uuid}")
        return _success(uuid=uuid)
    except Exception as e:
        logger.error(f"Error deleting ChartMogul customer {uuid}: {e}", exc_info=True)
        return _error(str(e))


# ---------------------------------------------------------------------------
# Contacts
# ---------------------------------------------------------------------------


@mcp.tool()
def chartmogul_list_contacts(
    email: str | None = None,
    customer_uuid: str | None = None,
    customer_external_id: str | None = None,
    data_source_uuid: str | None = None,
    cursor: str | None = None,
    per_page: int = DEFAULT_PER_PAGE,
) -> str:
    """
    List contacts (GET /contacts). Returns a page of entries plus a
    ``cursor``/``has_more`` pair -- pass the returned ``cursor`` back in to
    fetch the next page.

    email: optional, filter by exact email address.
    customer_uuid: optional, restrict to one customer's contacts.
    customer_external_id: optional, restrict by the customer's external id
    instead of its ChartMogul uuid.
    data_source_uuid: optional, restrict to one connected data source.
    cursor: optional, from a previous call's response, to fetch the next
    page.
    per_page: page size, 1-200 (ChartMogul's own server-side cap).
    """
    try:
        result = _request(
            "GET",
            "/contacts",
            params={
                "email": email,
                "customer_uuid": customer_uuid,
                "customer_external_id": customer_external_id,
                "data_source_uuid": data_source_uuid,
                "cursor": cursor,
                "per_page": _clamp_per_page(per_page),
            },
        )
        result = _validated_list_result(result, "/contacts")
        return _success_with_capped_list("entries", result)
    except Exception as e:
        logger.error(f"Error listing ChartMogul contacts: {e}", exc_info=True)
        return _error(str(e))


@mcp.tool()
def chartmogul_create_contact(customer_uuid: str, data: dict[str, Any]) -> str:
    """
    Create a contact under a customer (POST /customers/{customer_uuid}/contacts).

    Unlike customers and opportunities, contact creation is nested under its
    customer -- confirmed against ChartMogul's official docs
    (dev.chartmogul.com/reference/customers/add-contact): "customer_uuid" is
    a path parameter, not a body field.

    customer_uuid: the ChartMogul customer this contact belongs to.
    data: a ChartMogul contact object. Must include "data_source_uuid" (the
    customer's data source; an error is returned if the customer isn't
    associated with it); optionally "first_name", "last_name", "email",
    "title", "phone", "position", "external_id", "linked_in", "twitter",
    "notes", "custom".
    """
    try:
        safe_customer_uuid = url_path_id(customer_uuid, "customer_uuid")
        result = _request(
            "POST", f"/customers/{safe_customer_uuid}/contacts", json_data=data
        )
        result = _validated_dict(result, f"/customers/{safe_customer_uuid}/contacts")
        return success_with_capped_dict("contact", result)
    except Exception as e:
        logger.error(
            f"Error creating ChartMogul contact for customer {customer_uuid}: {e}",
            exc_info=True,
        )
        return _error(str(e))


@mcp.tool()
def chartmogul_get_contact(uuid: str) -> str:
    """
    Get one contact by uuid (GET /contacts/{uuid}).

    uuid: the contact's ChartMogul uuid.
    """
    try:
        safe_uuid = url_path_id(uuid, "uuid")
        result = _request("GET", f"/contacts/{safe_uuid}")
        result = _validated_dict(result, "/contacts")
        return success_with_capped_dict("contact", result)
    except Exception as e:
        logger.error(f"Error fetching ChartMogul contact {uuid}: {e}", exc_info=True)
        return _error(str(e))


@mcp.tool()
def chartmogul_update_contact(uuid: str, data: dict[str, Any]) -> str:
    """
    Update a contact (PATCH /contacts/{uuid}).

    uuid: the contact's ChartMogul uuid.
    data: fields to change, e.g. {"title": "VP Engineering", "phone": "+1..."}.
    """
    try:
        safe_uuid = url_path_id(uuid, "uuid")
        result = _request("PATCH", f"/contacts/{safe_uuid}", json_data=data)
        result = _validated_dict(result, "/contacts")
        return success_with_capped_dict("contact", result)
    except Exception as e:
        logger.error(f"Error updating ChartMogul contact {uuid}: {e}", exc_info=True)
        return _error(str(e))


@mcp.tool()
def chartmogul_delete_contact(uuid: str) -> str:
    """
    Permanently delete a contact (DELETE /contacts/{uuid}). This cannot be
    undone -- confirm with the user before calling this.

    uuid: the contact's ChartMogul uuid.
    """
    try:
        safe_uuid = url_path_id(uuid, "uuid")
        _request("DELETE", f"/contacts/{safe_uuid}")
        return _success(uuid=uuid)
    except Exception as e:
        logger.error(f"Error deleting ChartMogul contact {uuid}: {e}", exc_info=True)
        return _error(str(e))


# ---------------------------------------------------------------------------
# Opportunities
# ---------------------------------------------------------------------------


@mcp.tool()
def chartmogul_list_opportunities(
    customer_uuid: str | None = None,
    owner: str | None = None,
    pipeline: str | None = None,
    pipeline_stage: str | None = None,
    estimated_close_date_on_or_after: str | None = None,
    estimated_close_date_on_or_before: str | None = None,
    cursor: str | None = None,
    per_page: int = DEFAULT_PER_PAGE,
) -> str:
    """
    List sales opportunities (GET /opportunities). Returns a page of
    entries plus a ``cursor``/``has_more`` pair -- pass the returned
    ``cursor`` back in to fetch the next page.

    Each entry with ``amount_in_cents`` also includes ``amount`` as an exact
    two-decimal string in the entry's ``currency``. Use ``amount`` for
    user-facing values.

    customer_uuid: optional, restrict to one customer's opportunities.
    owner: optional, filter by the owner's email address.
    pipeline: optional, filter by pipeline name.
    pipeline_stage: optional, filter by pipeline stage name.
    estimated_close_date_on_or_after: optional, ISO 8601 date, only
    opportunities closing on or after this date.
    estimated_close_date_on_or_before: optional, ISO 8601 date, only
    opportunities closing on or before this date.
    cursor: optional, from a previous call's response, to fetch the next
    page.
    per_page: page size, 1-200 (ChartMogul's own server-side cap).
    """
    try:
        result = _request(
            "GET",
            "/opportunities",
            params={
                "customer_uuid": customer_uuid,
                "owner": owner,
                "pipeline": pipeline,
                "pipeline_stage": pipeline_stage,
                "estimated_close_date_on_or_after": estimated_close_date_on_or_after,
                "estimated_close_date_on_or_before": estimated_close_date_on_or_before,
                "cursor": cursor,
                "per_page": _clamp_per_page(per_page),
            },
        )
        result = _validated_list_result(result, "/opportunities")
        result = _entries_with_major_units(
            result, "/opportunities", _opportunity_with_major_units
        )
        return _success_with_capped_list("entries", result)
    except Exception as e:
        logger.error(f"Error listing ChartMogul opportunities: {e}", exc_info=True)
        return _error(str(e))


@mcp.tool()
def chartmogul_create_opportunity(data: dict[str, Any]) -> str:
    """
    Create a sales opportunity (POST /opportunities).

    data: a ChartMogul opportunity object. Must include "customer_uuid",
    "owner", "pipeline", "pipeline_stage", "estimated_close_date",
    "currency", and "amount_in_cents"; optionally "type",
    "forecast_category", "win_likelihood".
    The request remains integer cents. The returned opportunity also includes
    exact two-decimal ``amount`` in ``currency``.
    """
    try:
        result = _request("POST", "/opportunities", json_data=data)
        result = _validated_dict(result, "/opportunities")
        result = _opportunity_with_major_units(result)
        return success_with_capped_dict("opportunity", result)
    except Exception as e:
        logger.error(f"Error creating ChartMogul opportunity: {e}", exc_info=True)
        return _error(str(e))


@mcp.tool()
def chartmogul_get_opportunity(uuid: str) -> str:
    """
    Get one sales opportunity by uuid (GET /opportunities/{uuid}).

    When present, ``amount`` is an exact two-decimal string in ``currency``;
    ``amount_in_cents`` remains the explicit source value.

    uuid: the opportunity's ChartMogul uuid.
    """
    try:
        safe_uuid = url_path_id(uuid, "uuid")
        result = _request("GET", f"/opportunities/{safe_uuid}")
        result = _validated_dict(result, "/opportunities")
        result = _opportunity_with_major_units(result)
        return success_with_capped_dict("opportunity", result)
    except Exception as e:
        logger.error(
            f"Error fetching ChartMogul opportunity {uuid}: {e}", exc_info=True
        )
        return _error(str(e))


@mcp.tool()
def chartmogul_update_opportunity(uuid: str, data: dict[str, Any]) -> str:
    """
    Update a sales opportunity (PATCH /opportunities/{uuid}).

    uuid: the opportunity's ChartMogul uuid.
    data: fields to change, e.g. {"pipeline_stage": "Won", "win_likelihood": 100}.
    Write monetary changes as integer ``amount_in_cents``. The returned
    opportunity also includes exact two-decimal ``amount`` when available.
    """
    try:
        safe_uuid = url_path_id(uuid, "uuid")
        result = _request("PATCH", f"/opportunities/{safe_uuid}", json_data=data)
        result = _validated_dict(result, "/opportunities")
        result = _opportunity_with_major_units(result)
        return success_with_capped_dict("opportunity", result)
    except Exception as e:
        logger.error(
            f"Error updating ChartMogul opportunity {uuid}: {e}", exc_info=True
        )
        return _error(str(e))


@mcp.tool()
def chartmogul_delete_opportunity(uuid: str) -> str:
    """
    Permanently delete a sales opportunity (DELETE /opportunities/{uuid}).
    This cannot be undone -- confirm with the user before calling this.

    uuid: the opportunity's ChartMogul uuid.
    """
    try:
        safe_uuid = url_path_id(uuid, "uuid")
        _request("DELETE", f"/opportunities/{safe_uuid}")
        return _success(uuid=uuid)
    except Exception as e:
        logger.error(
            f"Error deleting ChartMogul opportunity {uuid}: {e}", exc_info=True
        )
        return _error(str(e))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@mcp.tool()
def chartmogul_get_metrics(
    metric: MetricName,
    start_date: str,
    end_date: str,
    interval: MetricInterval = "month",
    geo: str | None = None,
    plans: str | None = None,
) -> str:
    """
    Retrieve an official ChartMogul metric time series (GET /metrics/*).
    Use this tool for every MRR, ARR, ARPA, ASP, customer-count, churn-rate,
    or LTV question. Never derive account MRR by summing customer records:
    ChartMogul's metrics API applies the product's own subscription,
    expansion, contraction, reactivation, and churn calculations.

    Monetary values are two-decimal strings in the ChartMogul account's
    selected currency. Fractional metric cents are rounded to the nearest
    cent; exact source values are retained under explicit ``*_cents`` fields.
    Counts, rates, and percentage-change fields remain unchanged.

    metric: one of "all", "mrr", "arr", "arpa", "asp", "customer-count",
      "customer-churn-rate", "mrr-churn-rate", or "ltv". Use "all" to
      retrieve every key metric in one request.
    start_date: inclusive start date in YYYY-MM-DD format.
    end_date: inclusive end date in YYYY-MM-DD format; must be on or after
      start_date.
    interval: one of "day", "week", "month", "quarter", or "year" for most
      metrics. ASP supports only "month", "quarter", and "year". Standalone
      LTV is monthly-only, so it accepts only "month" and does not send an
      interval parameter to ChartMogul. For long ranges, prefer a coarser
      interval over "day" to avoid output truncation.
    geo: optional comma-separated ISO 3166-1 alpha-2 country codes.
    plans: optional comma-separated plan UUIDs, external IDs, or names.
    """
    try:
        if metric not in _METRIC_NAMES:
            raise ValueError(
                "metric must be one of: " + ", ".join(sorted(_METRIC_NAMES))
            )
        allowed_intervals = _METRIC_INTERVALS_BY_NAME[metric]
        if interval not in allowed_intervals:
            raise ValueError(
                f"interval for {metric} must be one of: "
                + ", ".join(sorted(allowed_intervals))
            )
        endpoint = f"/metrics/{metric}"
        normalized_start = _iso_date(start_date, "start_date")
        normalized_end = _iso_date(end_date, "end_date")
        if date.fromisoformat(normalized_start) > date.fromisoformat(normalized_end):
            raise ValueError("end_date must be on or after start_date")

        result = _request(
            "GET",
            endpoint,
            params={
                "start-date": normalized_start,
                "end-date": normalized_end,
                "interval": None if metric == "ltv" else interval,
                "geo": geo,
                "plans": plans,
            },
        )
        result = _metrics_with_major_units(result, metric, endpoint)
        return success_with_capped_dict(
            "metrics",
            result,
            extra_fields={"metric": metric},
            keep_list_tail=True,
        )
    except Exception as e:
        logger.exception(f"Error fetching ChartMogul {metric} metrics")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
