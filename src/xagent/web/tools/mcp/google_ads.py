import json
import logging
import os
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


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


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
        headers["login-customer-id"] = login_customer_id.replace("-", "")
    return headers


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
        response_text = response.text.strip()
        message = str(exc)
        if response_text:
            message = f"{message} - {response_text}"
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
        result = _request("GET", "/customers:listAccessibleCustomers")
        resource_names = result.get("resourceNames", [])
        customer_ids = [name.rsplit("/", 1)[-1] for name in resource_names]
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
        result = _request(
            "POST",
            f"/customers/{customer_id.replace('-', '')}/googleAds:search",
            login_customer_id=login_customer_id,
            body={"query": query},
        )
        return _success(results=result.get("results", []))
    except Exception as e:
        logger.error(f"Error running Google Ads search: {e}")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
