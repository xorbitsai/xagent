import datetime
import json
import logging
import os
import re
from typing import Any
from urllib.parse import quote

import requests
from mcp.server.fastmcp import FastMCP

from ....config import get_tool_max_output_length
from .utils import setup_proxy_env

logger = logging.getLogger("google-search-console-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("google-search-console-mcp")

WEBMASTERS_API_BASE_URL = "https://www.googleapis.com/webmasters/v3"
SEARCH_CONSOLE_API_BASE_URL = "https://searchconsole.googleapis.com/v1"
DEFAULT_TIMEOUT_SECONDS = 30

VALID_DIMENSIONS = {"query", "page", "country", "device", "date", "searchAppearance"}
VALID_SEARCH_TYPES = {"web", "image", "video", "news", "discover", "googleNews"}
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}\Z")

# The Search Analytics API itself allows rowLimit up to 25000. That's capped
# tighter here to bound the serialized MCP response; query_search_analytics
# still guards the actual serialized size and trims rows if needed, same as
# google_analytics_run_report does.
QUERY_DEFAULT_ROW_LIMIT = 100
QUERY_MAX_ROW_LIMIT = 1000


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


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
            # Cap the raw-body fallback: an upstream HTML error page can be
            # hundreds of KB and would otherwise be embedded whole into the
            # error message.
            detail = response.text.strip()[:1000]
        if detail:
            message = f"{message} - {detail}"
        raise RuntimeError(message) from exc

    if response.status_code == 204 or not response.content:
        return {}
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Failed to parse JSON response from Google Search Console API: {response.text[:200]}"
        ) from exc


def _require_dict_result(result: Any) -> dict[str, Any]:
    """Guard against a non-dict payload (e.g. a list or string) before
    calling dict methods on it. A dict that's merely empty (zero sites, zero
    report rows) is a normal, valid response and must not be rejected here.
    """
    if not isinstance(result, dict):
        raise ValueError("Unexpected response format from Google Search Console API")
    return result


def _encoded_site_url(site_url: str) -> str:
    """Search Console site identifiers are either a URL-prefix property
    (e.g. "https://example.com/") or a domain property
    (e.g. "sc-domain:example.com"); both must be percent-encoded before use
    in a URL path segment.
    """
    if not isinstance(site_url, str) or not site_url:
        raise ValueError(
            'site_url must be a non-empty string, e.g. "https://example.com/" '
            'or "sc-domain:example.com" (see google_search_console_list_sites)'
        )
    return quote(site_url, safe="")


def _validate_date(label: str, value: str) -> None:
    if not isinstance(value, str) or not _DATE_PATTERN.match(value):
        raise ValueError(f"{label} must be a date string in YYYY-MM-DD format")
    try:
        datetime.date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{label} must be a valid calendar date") from None


@mcp.tool()
def google_search_console_list_sites() -> str:
    """
    List the sites (URL-prefix or domain properties) the connected account
    has Search Console access to, along with the permission level. Use this
    first to discover which site_url values are available for the other
    google_search_console_* tools.
    """
    try:
        result = _require_dict_result(
            _request("GET", f"{WEBMASTERS_API_BASE_URL}/sites")
        )
        sites = [
            {
                "site_url": entry.get("siteUrl"),
                "permission_level": entry.get("permissionLevel"),
            }
            for entry in (result.get("siteEntry") or [])
            if isinstance(entry, dict) and entry.get("siteUrl")
        ]
        return _success(sites=sites)
    except Exception as e:
        logger.error(f"Error listing Google Search Console sites: {e}")
        return _error(str(e))


@mcp.tool()
def google_search_console_list_sitemaps(site_url: str) -> str:
    """
    List the sitemaps submitted for a site, including their status and
    warning/error counts.

    site_url: a value from google_search_console_list_sites, e.g.
      "https://example.com/" or "sc-domain:example.com".
    """
    try:
        encoded_site_url = _encoded_site_url(site_url)
        result = _require_dict_result(
            _request(
                "GET",
                f"{WEBMASTERS_API_BASE_URL}/sites/{encoded_site_url}/sitemaps",
            )
        )
        return _success(sitemaps=result.get("sitemap") or [])
    except Exception as e:
        logger.error(f"Error listing sitemaps for {site_url!r}: {e}")
        return _error(str(e))


@mcp.tool()
def google_search_console_query_search_analytics(
    site_url: str,
    start_date: str,
    end_date: str,
    dimensions: list[str] | None = None,
    search_type: str = "web",
    dimension_filter_groups: list[dict[str, Any]] | None = None,
    row_limit: int = QUERY_DEFAULT_ROW_LIMIT,
    start_row: int = 0,
) -> str:
    """
    Query Search Analytics: clicks, impressions, CTR, and average position,
    broken down by any combination of dimensions, for a date range.

    site_url: a value from google_search_console_list_sites.
    start_date, end_date: YYYY-MM-DD (inclusive). Search Console data is
      typically delayed 1-3 days, so end_date should not be today.
    dimensions: any of "query", "page", "country", "device", "date",
      "searchAppearance". Omit for a single aggregate row.
    search_type: one of "web", "image", "video", "news", "discover",
      "googleNews" (default "web").
    dimension_filter_groups: raw Search Analytics API filter groups, e.g.
      [{"filters": [{"dimension": "country", "operator": "equals",
      "expression": "usa"}]}]. Optional.
    row_limit: max rows to return (default 100, max 1000 — keeps a wide
      query's serialized response under the MCP output size limit).
    start_row: row offset for paging — if the response returns exactly
      row_limit rows, call again with start_row += row_limit to fetch more.
    """
    try:
        encoded_site_url = _encoded_site_url(site_url)
        _validate_date("start_date", start_date)
        _validate_date("end_date", end_date)
        if start_date > end_date:
            raise ValueError("start_date must not be after end_date")
        if search_type not in VALID_SEARCH_TYPES:
            raise ValueError(f"search_type must be one of {sorted(VALID_SEARCH_TYPES)}")
        if row_limit < 1 or row_limit > QUERY_MAX_ROW_LIMIT:
            raise ValueError(f"row_limit must be between 1 and {QUERY_MAX_ROW_LIMIT}")
        if start_row < 0:
            raise ValueError("start_row must be >= 0")
        if dimensions:
            invalid = sorted(set(dimensions) - VALID_DIMENSIONS, key=str)
            if invalid:
                raise ValueError(
                    f"invalid dimensions {invalid}; must be from {sorted(VALID_DIMENSIONS)}"
                )

        body: dict[str, Any] = {
            "startDate": start_date,
            "endDate": end_date,
            "type": search_type,
            "rowLimit": row_limit,
            "startRow": start_row,
        }
        if dimensions:
            body["dimensions"] = dimensions
        if dimension_filter_groups:
            body["dimensionFilterGroups"] = dimension_filter_groups

        result = _require_dict_result(
            _request(
                "POST",
                f"{WEBMASTERS_API_BASE_URL}/sites/{encoded_site_url}/searchAnalytics/query",
                body=body,
            )
        )
        rows = result.get("rows")
        if not isinstance(rows, list):
            rows = []

        # row_limit bounds row *count*, not bytes: several "query"/"page"
        # dimension values (often full URLs) at once can still cross the
        # platform's output truncation threshold. Halve the returned rows
        # until the serialized response fits, mirroring
        # google_analytics_run_report's trimming approach.
        max_output_length = get_tool_max_output_length()
        original_row_count = len(rows)
        response = _success(rows=rows, row_count=len(rows), truncated=False)
        while len(response) > max_output_length and rows:
            rows = rows[: len(rows) // 2]
            response = _success(rows=rows, row_count=len(rows), truncated=True)
        if len(rows) < original_row_count:
            logger.warning(
                f"Google Search Console query_search_analytics response trimmed "
                f"from {original_row_count} to {len(rows)} rows to stay under "
                f"the {max_output_length}-char output limit"
            )
        return response
    except Exception as e:
        logger.error(f"Error querying search analytics for {site_url!r}: {e}")
        return _error(str(e))


@mcp.tool()
def google_search_console_inspect_url(
    site_url: str, inspection_url: str, language_code: str = "en-US"
) -> str:
    """
    Inspect a URL's indexing status: whether it's indexed, canonical URL,
    coverage state, mobile usability, and rich result eligibility.

    site_url: the property the URL belongs to, from
      google_search_console_list_sites.
    inspection_url: the fully-qualified URL to inspect (must belong to
      site_url).
    language_code: BCP-47 language for the human-readable result fields
      (default "en-US").
    """
    try:
        if not isinstance(site_url, str) or not site_url:
            raise ValueError("site_url must be a non-empty string")
        if not isinstance(inspection_url, str) or not inspection_url:
            raise ValueError("inspection_url must be a non-empty string")
        body = {
            "inspectionUrl": inspection_url,
            "siteUrl": site_url,
            "languageCode": language_code,
        }
        result = _require_dict_result(
            _request(
                "POST",
                f"{SEARCH_CONSOLE_API_BASE_URL}/urlInspection/index:inspect",
                body=body,
            )
        )
        return _success(inspection_result=result.get("inspectionResult") or {})
    except Exception as e:
        logger.error(f"Error inspecting URL {inspection_url!r}: {e}")
        return _error(str(e))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run()
