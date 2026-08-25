import datetime
import json
import logging
import os
import re
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

from ....config import get_tool_max_output_length
from .utils import require_clean_identifier, setup_proxy_env, url_path_id

logger = logging.getLogger("google-search-console-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("google-search-console-mcp")

WEBMASTERS_API_BASE_URL = "https://www.googleapis.com/webmasters/v3"
SEARCH_CONSOLE_API_BASE_URL = "https://searchconsole.googleapis.com/v1"
DEFAULT_TIMEOUT_SECONDS = 30

VALID_DIMENSIONS = {"query", "page", "country", "device", "date", "searchAppearance"}
VALID_SEARCH_TYPES = {"web", "image", "video", "news", "discover", "googleNews"}
# Google does not disclose search terms for Discover/Google News surfaces,
# so the Search Analytics API rejects the "query" dimension for these two
# search types with a 400. Checked locally so the calling LLM gets an
# actionable message instead of an upstream API error.
_SEARCH_TYPES_WITHOUT_QUERY_DIMENSION = {"discover", "googleNews"}
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
    in a URL path segment. Delegates to the shared url_path_id helper (used
    by the other stdio MCP connectors) for that encoding plus its "."/".."
    guard, after our own check for the more actionable message on the
    empty/wrong-type case.
    """
    if not isinstance(site_url, str) or not site_url:
        raise ValueError(
            'site_url must be a non-empty string, e.g. "https://example.com/" '
            'or "sc-domain:example.com" (see google_search_console_list_sites)'
        )
    return url_path_id(site_url, "site_url")


def _dimension_filter_groups_reference_query(
    dimension_filter_groups: list[dict[str, Any]],
) -> bool:
    """Whether any filter inside dimension_filter_groups filters on the
    "query" dimension. Tolerates a malformed/unexpected shape (a non-dict
    group, a non-list "filters", a non-dict filter entry) by treating it as
    not referencing "query" rather than raising — this check runs before
    the request body is built, and a shape the caller got wrong here will
    still surface as a clear upstream 400 once sent."""
    for group in dimension_filter_groups:
        if not isinstance(group, dict):
            continue
        filters = group.get("filters")
        if not isinstance(filters, list):
            continue
        for filter_entry in filters:
            if (
                isinstance(filter_entry, dict)
                and filter_entry.get("dimension") == "query"
            ):
                return True
    return False


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
        site_entries = result.get("siteEntry")
        if not isinstance(site_entries, list):
            if site_entries is not None:
                logger.warning(
                    f"Google Search Console list_sites returned a non-list "
                    f"'siteEntry' field ({type(site_entries).__name__}); "
                    f"treating as empty"
                )
            site_entries = []
        sites = [
            {
                "site_url": entry.get("siteUrl"),
                "permission_level": entry.get("permissionLevel"),
            }
            for entry in site_entries
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
        sitemaps = result.get("sitemap")
        if not isinstance(sitemaps, list):
            sitemaps = []
        return _success(sitemaps=sitemaps)
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
      "searchAppearance". Omit for a single aggregate row. "query" is not
      available (as a dimension or as a dimension_filter_groups filter)
      when search_type is "discover" or "googleNews" (Google doesn't
      disclose search terms for those surfaces) — "position" values on
      rows from those two surfaces are also not meaningful ranking data,
      per Google's own documentation.
    search_type: one of "web", "image", "video", "news", "discover",
      "googleNews" (default "web").
    dimension_filter_groups: raw Search Analytics API filter groups, e.g.
      [{"filters": [{"dimension": "country", "operator": "equals",
      "expression": "usa"}]}]. Optional.
    row_limit: max rows to return (default 100, max 1000 — keeps a wide
      query's serialized response under the MCP output size limit).
    start_row: row offset for paging — if row_count equals row_limit, call
      again with start_row += row_limit to fetch more. row_count is the
      number of rows this single response returned (capped at row_limit),
      not a total match count — Google's own API docs note it isn't
      guaranteed to return literally every matching row even once you've
      fully paginated. It stays accurate even when "rows" is shortened
      below for output-size reasons — see "truncated".
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
        if search_type in _SEARCH_TYPES_WITHOUT_QUERY_DIMENSION:
            # Google rejects "query" for these two surfaces whether it's
            # requested as a grouping dimension or as a filter condition —
            # both reach the same API restriction, so both must be checked
            # here rather than just the dimensions list.
            uses_query = bool(dimensions and "query" in dimensions) or (
                dimension_filter_groups is not None
                and _dimension_filter_groups_reference_query(dimension_filter_groups)
            )
            if uses_query:
                raise ValueError(
                    f'the "query" dimension is not available for '
                    f"search_type={search_type!r} (Google does not disclose "
                    f"search terms for Discover/Google News surfaces), whether "
                    f"requested as a dimension or filtered on via "
                    f"dimension_filter_groups"
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
            if rows is not None:
                logger.warning(
                    f"Google Search Console query_search_analytics returned a "
                    f"non-list 'rows' field ({type(rows).__name__}); treating "
                    f"as empty"
                )
            rows = []

        # row_limit bounds row *count*, not bytes: several "query"/"page"
        # dimension values (often full URLs) at once can still cross the
        # platform's output truncation threshold. Halve the returned rows
        # until the serialized response fits, mirroring
        # google_analytics_run_report's trimming approach.
        #
        # row_count is pinned to the pre-trim count and never recomputed
        # from the (possibly shrunk) "rows" list: it's the pagination
        # signal the docstring tells callers to check against row_limit,
        # and trimming rows for output-size reasons must not be confused
        # with "this page had fewer matches than row_limit" — that would
        # make a caller stop paging early and silently miss rows, with no
        # way to resume since the dropped rows already consumed part of
        # this request's offset window.
        max_output_length = get_tool_max_output_length()
        original_row_count = len(rows)
        response = _success(rows=rows, row_count=original_row_count, truncated=False)
        while len(response) > max_output_length and rows:
            rows = rows[: len(rows) // 2]
            response = _success(rows=rows, row_count=original_row_count, truncated=True)
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
        site_url = require_clean_identifier(site_url, "site_url")
        inspection_url = require_clean_identifier(inspection_url, "inspection_url")
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
        inspection_result = result.get("inspectionResult")
        if not isinstance(inspection_result, dict):
            if inspection_result is not None:
                logger.warning(
                    f"Google Search Console inspect_url returned a non-dict "
                    f"'inspectionResult' field "
                    f"({type(inspection_result).__name__}); treating as empty"
                )
            inspection_result = {}
        return _success(inspection_result=inspection_result)
    except Exception as e:
        logger.error(f"Error inspecting URL {inspection_url!r}: {e}")
        return _error(str(e))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run()
