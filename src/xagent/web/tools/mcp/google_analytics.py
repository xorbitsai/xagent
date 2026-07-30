import json
import logging
import os
import re
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

from .utils import setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("google-analytics-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("google-analytics-mcp")

DATA_API_BASE_URL = "https://analyticsdata.googleapis.com/v1beta"
ADMIN_API_BASE_URL = "https://analyticsadmin.googleapis.com/v1beta"
DEFAULT_TIMEOUT_SECONDS = 30
MAX_ACCOUNT_SUMMARY_PAGES = 20

_PROPERTY_ID_PATTERN = re.compile(r"^[0-9]+\Z")


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _normalize_property_id(property_id: str) -> str:
    """Accept either a bare numeric id ("123456") or a resource name
    ("properties/123456") and return the bare numeric id.

    Rejects anything else rather than silently sanitizing it, since this
    value is interpolated directly into a URL path.
    """
    raw = str(property_id)
    if raw.startswith("properties/"):
        raw = raw[len("properties/") :]
    if not _PROPERTY_ID_PATTERN.match(raw):
        raise ValueError(
            "property_id must be numeric (optionally prefixed 'properties/')"
        )
    return raw


def _headers() -> dict[str, str]:
    access_token = os.environ.get("GOOGLE_ACCESS_TOKEN")
    if not access_token:
        raise ValueError("GOOGLE_ACCESS_TOKEN environment variable is missing")
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }


def _extract_error_detail(response: requests.Response) -> str | None:
    """Pull the human-readable message out of a Google API error body
    (``{"error": {"code": ..., "message": ..., "status": ...}}``).
    Returns None if the body isn't in the expected shape, so the caller can
    fall back to the raw response text.
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
    url: str,
    *,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
) -> Any:
    response = requests.request(
        method=method,
        url=url,
        headers=_headers(),
        params=params,
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


def _require_dict_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("Unexpected response format from Google Analytics API")
    return result


def _date_range_body(date_range: dict[str, Any]) -> dict[str, Any]:
    # FastMCP's signature validation guarantees a dict, but not its keys — an
    # empty dict or misnamed keys (e.g. camelCase "startDate") would otherwise
    # sail through and reach the API as two nulls. Fail here with a message
    # the LLM can self-correct from.
    if not date_range.get("start_date") or not date_range.get("end_date"):
        raise ValueError(
            'Each date range needs "start_date" and "end_date" (YYYY-MM-DD or '
            'GA4 relative terms like "7daysAgo"/"today")'
        )
    body = {
        "startDate": date_range["start_date"],
        "endDate": date_range["end_date"],
    }
    if date_range.get("name"):
        body["name"] = date_range["name"]
    return body


@mcp.tool()
def google_analytics_list_properties() -> str:
    """
    List the GA4 accounts and properties accessible to the connected account.
    Use this first to discover which property_id values are available for
    google_analytics_run_report / google_analytics_get_metadata.
    """
    try:
        properties: list[dict[str, Any]] = []
        page_token: str | None = None
        for _ in range(MAX_ACCOUNT_SUMMARY_PAGES):
            params: dict[str, Any] = {"pageSize": 200}
            if page_token:
                params["pageToken"] = page_token
            result = _require_dict_result(
                _request(
                    "GET",
                    f"{ADMIN_API_BASE_URL}/accountSummaries",
                    params=params,
                )
            )
            for account in result.get("accountSummaries") or []:
                if not isinstance(account, dict):
                    continue
                account_name = account.get("displayName")
                for prop in account.get("propertySummaries") or []:
                    if not isinstance(prop, dict):
                        continue
                    resource_name = prop.get("property")
                    if not isinstance(resource_name, str) or not resource_name:
                        # No usable resource name -> no property_id to offer;
                        # skip rather than emitting a garbage empty id.
                        continue
                    properties.append(
                        {
                            "account_name": account_name,
                            "property_id": resource_name.rsplit("/", 1)[-1],
                            "property_display_name": prop.get("displayName"),
                        }
                    )
            page_token = result.get("nextPageToken")
            if not page_token:
                break
        return _success(properties=properties)
    except Exception as e:
        logger.error(f"Error listing Google Analytics properties: {e}")
        return _error(str(e))


@mcp.tool()
def google_analytics_get_metadata(property_id: str) -> str:
    """
    List the dimensions and metrics available to query for a GA4 property.
    Use this before google_analytics_run_report if you're unsure which
    dimension/metric names are valid for this property.
    """
    try:
        normalized_property_id = _normalize_property_id(property_id)
        result = _require_dict_result(
            _request(
                "GET",
                f"{DATA_API_BASE_URL}/properties/{normalized_property_id}/metadata",
            )
        )
        return _success(
            dimensions=result.get("dimensions") or [],
            metrics=result.get("metrics") or [],
        )
    except Exception as e:
        logger.error(f"Error getting Google Analytics metadata for {property_id}: {e}")
        return _error(str(e))


@mcp.tool()
def google_analytics_run_report(
    property_id: str,
    metrics: list[str],
    date_ranges: list[dict[str, str]],
    dimensions: list[str] | None = None,
    dimension_filter: dict[str, Any] | None = None,
    order_bys: list[dict[str, Any]] | None = None,
    limit: int | None = None,
) -> str:
    """
    Run a GA4 report: quantify performance by any combination of dimensions
    (e.g. channel, campaign, landing page, segment) over one or more date
    ranges.

    metrics: metric names, e.g. ["sessions", "conversions", "totalRevenue"].
    date_ranges: one dict per period to compare, each with "start_date" and
      "end_date" (YYYY-MM-DD, or GA4 relative terms like "7daysAgo"/"today"),
      and an optional "name" to label the period (e.g. "current", "previous")
      — pass two date_ranges to compare a period against a prior one in a
      single call; the response's rows are tagged with the matching name.
    dimensions: dimension names, e.g. ["sessionDefaultChannelGroup",
      "landingPagePlusQueryString"]. Use google_analytics_get_metadata to
      discover valid names for this property.
    dimension_filter: a raw GA4 FilterExpression dict, for narrowing results
      (e.g. to one campaign or landing page). Optional.
    order_bys: a raw GA4 OrderBy list, e.g. to sort by a metric descending.
    limit: max rows to return.
    """
    try:
        normalized_property_id = _normalize_property_id(property_id)
        body: dict[str, Any] = {
            "metrics": [{"name": m} for m in metrics],
            "dateRanges": [_date_range_body(dr) for dr in date_ranges],
        }
        if dimensions:
            body["dimensions"] = [{"name": d} for d in dimensions]
        if dimension_filter:
            body["dimensionFilter"] = dimension_filter
        if order_bys:
            body["orderBys"] = order_bys
        if limit is not None:
            body["limit"] = str(limit)

        result = _require_dict_result(
            _request(
                "POST",
                f"{DATA_API_BASE_URL}/properties/{normalized_property_id}:runReport",
                body=body,
            )
        )
        return _success(
            dimension_headers=result.get("dimensionHeaders") or [],
            metric_headers=result.get("metricHeaders") or [],
            rows=result.get("rows") or [],
            row_count=result.get("rowCount") or 0,
        )
    except Exception as e:
        logger.error(f"Error running Google Analytics report for {property_id}: {e}")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
