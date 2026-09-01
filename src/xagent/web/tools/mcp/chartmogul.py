import json
import logging
import os
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

from ....config import get_tool_max_output_length
from ....core.utils.security import redact_sensitive_text
from ...utils.graphql_errors import truncate_error_text
from .utils import setup_proxy_env, success_with_capped_dict, url_path_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chartmogul-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

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


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


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
        if isinstance(detail, dict) and detail:
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
            params={k: v for k, v in (params or {}).items() if v is not None},
            json=json_data,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        # A connection/timeout/proxy failure's message can itself embed
        # sensitive data -- e.g. a ProxyError echoing the ambient
        # HTTPS_PROXY URL, which may carry embedded user:pass@ credentials
        # (setup_proxy_env() exports whatever the OS has configured) --
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
    return response.json()


def _clamp_per_page(per_page: int) -> int:
    return max(1, min(int(per_page), MAX_PER_PAGE))


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
        if not isinstance(result, dict) or not isinstance(result.get("entries"), list):
            return _error("ChartMogul returned an unexpected response for /customers")
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
    """
    try:
        result = _request("POST", "/customers", json_data=data)
        if not isinstance(result, dict):
            return _error("ChartMogul returned an unexpected response for /customers")
        return success_with_capped_dict("customer", result)
    except Exception as e:
        logger.error(f"Error creating ChartMogul customer: {e}", exc_info=True)
        return _error(str(e))


@mcp.tool()
def chartmogul_get_customer(uuid: str) -> str:
    """
    Get one customer by uuid (GET /customers/{uuid}).

    uuid: the customer's ChartMogul uuid (e.g. "cus_00000000-0000-0000-0000-000000000000").
    """
    try:
        safe_uuid = url_path_id(uuid, "uuid")
        result = _request("GET", f"/customers/{safe_uuid}")
        if not isinstance(result, dict):
            return _error("ChartMogul returned an unexpected response for /customers")
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
    """
    try:
        safe_uuid = url_path_id(uuid, "uuid")
        result = _request("PATCH", f"/customers/{safe_uuid}", json_data=data)
        if not isinstance(result, dict):
            return _error("ChartMogul returned an unexpected response for /customers")
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
        if not isinstance(result, dict) or not isinstance(result.get("entries"), list):
            return _error("ChartMogul returned an unexpected response for /contacts")
        return _success_with_capped_list("entries", result)
    except Exception as e:
        logger.error(f"Error listing ChartMogul contacts: {e}", exc_info=True)
        return _error(str(e))


@mcp.tool()
def chartmogul_create_contact(data: dict[str, Any]) -> str:
    """
    Create a contact (POST /contacts).

    data: a ChartMogul contact object. Should include "customer_uuid" (or
    "customer_external_id" together with "data_source_uuid") to associate
    it with a customer; commonly also "first_name", "last_name", "email",
    "title", "phone".
    """
    try:
        result = _request("POST", "/contacts", json_data=data)
        if not isinstance(result, dict):
            return _error("ChartMogul returned an unexpected response for /contacts")
        return success_with_capped_dict("contact", result)
    except Exception as e:
        logger.error(f"Error creating ChartMogul contact: {e}", exc_info=True)
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
        if not isinstance(result, dict):
            return _error("ChartMogul returned an unexpected response for /contacts")
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
        if not isinstance(result, dict):
            return _error("ChartMogul returned an unexpected response for /contacts")
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
        if not isinstance(result, dict) or not isinstance(result.get("entries"), list):
            return _error(
                "ChartMogul returned an unexpected response for /opportunities"
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
    """
    try:
        result = _request("POST", "/opportunities", json_data=data)
        if not isinstance(result, dict):
            return _error(
                "ChartMogul returned an unexpected response for /opportunities"
            )
        return success_with_capped_dict("opportunity", result)
    except Exception as e:
        logger.error(f"Error creating ChartMogul opportunity: {e}", exc_info=True)
        return _error(str(e))


@mcp.tool()
def chartmogul_get_opportunity(uuid: str) -> str:
    """
    Get one sales opportunity by uuid (GET /opportunities/{uuid}).

    uuid: the opportunity's ChartMogul uuid.
    """
    try:
        safe_uuid = url_path_id(uuid, "uuid")
        result = _request("GET", f"/opportunities/{safe_uuid}")
        if not isinstance(result, dict):
            return _error(
                "ChartMogul returned an unexpected response for /opportunities"
            )
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
    """
    try:
        safe_uuid = url_path_id(uuid, "uuid")
        result = _request("PATCH", f"/opportunities/{safe_uuid}", json_data=data)
        if not isinstance(result, dict):
            return _error(
                "ChartMogul returned an unexpected response for /opportunities"
            )
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


if __name__ == "__main__":
    mcp.run()
