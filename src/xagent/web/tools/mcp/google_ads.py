import json
import logging
import os
import re
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

from .utils import setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("google-ads-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("google-ads-mcp")

GOOGLE_ADS_BASE_URL = "https://googleads.googleapis.com/v23"
DEFAULT_TIMEOUT_SECONDS = 30

_CUSTOMER_ID_PATTERN = re.compile(r"^[0-9][0-9-]*\Z")


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _normalize_customer_id(customer_id: str, *, field_name: str) -> str:
    """Validate a customer id is digits/dashes (starting with a digit), then
    strip the dashes.

    Rejects anything else (rather than silently sanitizing it) since this
    value is interpolated directly into an HTTP header and a URL path —
    a malformed value could otherwise inject headers or redirect the request
    to an unintended API path. Requiring a leading digit also rules out an
    all-dash input (e.g. "---"), which would otherwise strip down to an
    empty string and silently produce a malformed header/URL instead of
    failing validation.
    """
    if not _CUSTOMER_ID_PATTERN.match(str(customer_id)):
        raise ValueError(f"{field_name} must contain only digits and dashes")
    return str(customer_id).replace("-", "")


def _headers(login_customer_id: str | None = None) -> dict[str, str]:
    access_token = os.environ.get("GOOGLE_ACCESS_TOKEN")
    if not access_token:
        raise ValueError("GOOGLE_ACCESS_TOKEN environment variable is missing")

    developer_token = os.environ.get("GOOGLE_ADS_DEVELOPER_TOKEN")
    if not developer_token:
        raise ValueError("GOOGLE_ADS_DEVELOPER_TOKEN environment variable is missing")

    headers = {
        "Authorization": f"Bearer {access_token}",
        "developer-token": developer_token,
        "Content-Type": "application/json",
    }
    if login_customer_id:
        headers["login-customer-id"] = _normalize_customer_id(
            login_customer_id, field_name="login_customer_id"
        )
    return headers


def _extract_error_detail(response: requests.Response) -> str | None:
    """Pull the human-readable message(s) out of a Google Ads error body.

    Google Ads error responses are a large structured JSON payload
    (``{"error": {"message": ..., "details": [{"errors": [...]}]}}``);
    returning that whole blob to the LLM wastes tokens and is harder for it
    to act on than the plain message(s) it actually needs to fix a bad GAQL
    query. Returns None if the body isn't in the expected shape, so the
    caller can fall back to the raw response text.
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

    messages: list[str] = []
    top_message = error.get("message")
    if isinstance(top_message, str) and top_message:
        messages.append(top_message)

    for detail in error.get("details") or []:
        if not isinstance(detail, dict):
            continue
        for sub_error in detail.get("errors") or []:
            if not isinstance(sub_error, dict):
                continue
            sub_message = sub_error.get("message")
            if isinstance(sub_message, str) and sub_message:
                messages.append(sub_message)

    if not messages:
        return None
    return "; ".join(dict.fromkeys(messages))


def _require_dict_result(result: Any) -> dict[str, Any]:
    """Guard against a non-dict payload (e.g. a list or string) before
    calling dict methods on it. A dict that's merely empty (zero
    accessible customers, zero GAQL rows) is a normal, valid response and
    must not be rejected here."""
    if not isinstance(result, dict):
        raise ValueError("Unexpected response format from Google Ads API")
    return result


def _request(
    method: str,
    path: str,
    *,
    login_customer_id: str | None = None,
    body: dict[str, Any] | None = None,
) -> Any:
    response = requests.request(
        method=method,
        url=f"{GOOGLE_ADS_BASE_URL}{path}",
        headers=_headers(login_customer_id),
        json=body,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        message = str(exc)
        detail = _extract_error_detail(response)
        if detail is None:
            detail = response.text.strip()
        if detail:
            message = f"{message} - {detail}"
        raise RuntimeError(message) from exc

    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


@mcp.tool()
def google_ads_list_accessible_customers() -> str:
    """
    List the Google Ads customer IDs accessible to the connected account.
    Use this first to discover which customer_id values are available for google_ads_search.
    """
    try:
        result = _require_dict_result(
            _request("GET", "/customers:listAccessibleCustomers")
        )
        resource_names = result.get("resourceNames", [])
        customer_ids = [
            name.rsplit("/", 1)[-1] for name in resource_names if isinstance(name, str)
        ]
        return _success(customer_ids=customer_ids)
    except Exception as e:
        logger.error(f"Error listing accessible customers: {e}")
        return _error(str(e))


@mcp.tool()
def google_ads_search(
    customer_id: str, query: str, login_customer_id: str | None = None
) -> str:
    """
    Run a Google Ads Query Language (GAQL) query against one customer account,
    e.g. to list campaigns, ad groups, or performance metrics.
    customer_id is the account to query (digits only, no dashes).
    login_customer_id is required when customer_id is a client account managed
    under a manager (MCC) account, and should be the manager's customer id.
    """
    try:
        normalized_customer_id = _normalize_customer_id(
            customer_id, field_name="customer_id"
        )
        result = _require_dict_result(
            _request(
                "POST",
                f"/customers/{normalized_customer_id}/googleAds:search",
                login_customer_id=login_customer_id,
                body={"query": query},
            )
        )
        return _success(results=result.get("results", []))
    except Exception as e:
        logger.error(f"Error running Google Ads search: {e}")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
