import json
import logging
import os
import re
from datetime import date
from typing import Any

from mcp.server.fastmcp import FastMCP

from ....config import get_tool_max_output_length
from ....core.utils import public_requests as requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("google-analytics-mcp")


mcp = FastMCP("google-analytics-mcp")

DATA_API_BASE_URL = "https://analyticsdata.googleapis.com/v1beta"
ADMIN_API_BASE_URL = "https://analyticsadmin.googleapis.com/v1beta"
DEFAULT_TIMEOUT_SECONDS = 30
MAX_ACCOUNT_SUMMARY_PAGES = 20
# The limit bounds row *count*, not bytes: a 5-dimension + 5-metric row with
# short (~11-char) values serializes to ~275 chars through _success(), which
# keeps RUN_REPORT_MAX_LIMIT rows comfortably under the platform's 51200-char
# MCP output truncation threshold — but several long-valued dimensions (e.g.
# full URLs) at once can still cross it. google_analytics_run_report guards
# the actual serialized size and trims rows if needed, same as
# google_analytics_list_properties does for its own response.
RUN_REPORT_DEFAULT_LIMIT = 100
RUN_REPORT_MAX_LIMIT = 150

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
            # Cap the raw-body fallback: an upstream HTML error page can be
            # hundreds of KB and would otherwise be embedded whole into the
            # error message.
            detail = response.text.strip()[:1000]
        if detail:
            message = f"{message} - {detail}"
        raise RuntimeError(message) from exc

    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


def _require_dict_result(result: Any) -> dict[str, Any]:
    """Guard against a non-dict payload (e.g. a list or string) before
    calling dict methods on it. A dict that's merely empty (zero properties,
    zero report rows) is a normal, valid response and must not be rejected
    here."""
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
            'GA4 relative terms like "7daysAgo"/"yesterday")'
        )
    body = {
        "startDate": date_range["start_date"],
        "endDate": date_range["end_date"],
    }
    if date_range.get("name"):
        body["name"] = date_range["name"]
    return body


def _ends_today(date_range: dict[str, Any]) -> bool:
    return str(date_range.get("end_date", "")).strip().lower() == "today"


_RELATIVE_DATE_PATTERN = re.compile(r"^(\d+)daysAgo$", re.IGNORECASE)


def _relative_day_offset(value: str) -> int | None:
    """Days before "today" a GA4 relative date term refers to (0 for
    "today", 1 for "yesterday", N for "NdaysAgo"), or None if `value` isn't
    one of those recognized relative terms (e.g. an explicit YYYY-MM-DD)."""
    normalized = value.strip().lower()
    if normalized == "today":
        return 0
    if normalized == "yesterday":
        return 1
    match = _RELATIVE_DATE_PATTERN.match(normalized)
    return int(match.group(1)) if match else None


def _date_range_span(
    date_range: dict[str, Any],
) -> tuple[int, int] | tuple[date, date] | None:
    """(start, end) in a comparable, ordered form: either two GA4 relative
    day offsets (both counted as days before "today", so start >= end) or
    two explicit `date` objects -- whichever bound type start_date/end_date
    both share. None if a bound is unrecognized, or one bound is a relative
    term and the other an explicit date -- comparable only by knowing
    "today"'s actual date, which this tool never resolves itself."""
    start = str(date_range.get("start_date", ""))
    end = str(date_range.get("end_date", ""))
    start_offset = _relative_day_offset(start)
    end_offset = _relative_day_offset(end)
    if start_offset is not None and end_offset is not None:
        return (start_offset, end_offset)
    if start_offset is None and end_offset is None:
        try:
            return (date.fromisoformat(start), date.fromisoformat(end))
        except ValueError:
            return None
    return None


def _date_range_day_count(date_range: dict[str, Any]) -> int | None:
    """Inclusive day span of a date_range (e.g. 2 for "yesterday"/"today",
    7 for "7daysAgo"/"yesterday"), or None if _date_range_span can't
    resolve it."""
    span = _date_range_span(date_range)
    if span is None:
        return None
    start, end = span
    # _date_range_span only ever returns same-type pairs, but its return
    # type (tuple[int, int] | tuple[date, date]) doesn't tell mypy that
    # start and end are correlated -- narrow both explicitly per branch.
    if isinstance(start, int) and isinstance(end, int):
        count = start - end + 1
    elif isinstance(start, date) and isinstance(end, date):
        count = (end - start).days + 1
    else:
        return None
    return count if count > 0 else None


def _date_ranges_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """True if two date_ranges share at least one calendar day. Only
    comparable when both resolve to the same bound kind via
    _date_range_span (both relative offsets, or both explicit dates);
    returns False rather than guessing otherwise."""
    span_a = _date_range_span(a)
    span_b = _date_range_span(b)
    if span_a is None or span_b is None:
        return False
    a_start, a_end = span_a
    b_start, b_end = span_b
    if (
        isinstance(a_start, int)
        and isinstance(a_end, int)
        and isinstance(b_start, int)
        and isinstance(b_end, int)
    ):
        # Relative offsets count "days before today", so they run
        # backwards -- negate onto a normal ascending axis before the
        # standard interval-overlap comparison.
        a_lo, a_hi = -a_start, -a_end
        b_lo, b_hi = -b_start, -b_end
        return a_lo <= b_hi and b_lo <= a_hi
    if (
        isinstance(a_start, date)
        and isinstance(a_end, date)
        and isinstance(b_start, date)
        and isinstance(b_end, date)
    ):
        return a_start <= b_end and b_start <= a_end
    return False


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
        # page_token still set here means MAX_ACCOUNT_SUMMARY_PAGES ran out
        # with more pages pending — tell the caller the list is incomplete.
        return _success(properties=properties, truncated=bool(page_token))
    except Exception as e:
        logger.error(f"Error listing Google Analytics properties: {e}")
        return _error(str(e))


def _project_metadata_entries(entries: list[Any], search: str) -> list[dict[str, Any]]:
    """Project metadata entries down to the fields the LLM needs to pick a
    name (apiName/uiName/category), optionally filtered by a case-insensitive
    substring. The full GA4 metadata document (200-350+ entries with
    descriptions) serializes to 50-140KB, which would be hard-truncated by
    the platform output filter into broken JSON."""
    needle = search.strip().lower()
    projected = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        api_name = entry.get("apiName")
        if not isinstance(api_name, str) or not api_name:
            # No usable apiName -> nothing for the LLM to pass back into
            # run_report; skip rather than emitting a garbage empty name.
            continue
        ui_name = str(entry.get("uiName") or "")
        if needle and needle not in api_name.lower() and needle not in ui_name.lower():
            continue
        projected.append(
            {
                "apiName": api_name,
                "uiName": ui_name,
                "category": entry.get("category"),
            }
        )
    return projected


@mcp.tool()
def google_analytics_get_metadata(property_id: str, search: str = "") -> str:
    """
    List the dimension and metric names available to query for a GA4 property
    (apiName, uiName, category). Use this before google_analytics_run_report
    if you're unsure which dimension/metric names are valid for this property.
    search: optional case-insensitive substring filter on the name, e.g.
    "session" or "conversion" — prefer it over listing everything.
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
            dimensions=_project_metadata_entries(
                result.get("dimensions") or [], search
            ),
            metrics=_project_metadata_entries(result.get("metrics") or [], search),
        )
    except Exception as e:
        logger.error(
            f"Error getting Google Analytics metadata for {property_id!r}: {e}"
        )
        return _error(str(e))


@mcp.tool()
def google_analytics_run_report(
    property_id: str,
    metrics: list[str],
    date_ranges: list[dict[str, str]],
    dimensions: list[str] | None = None,
    dimension_filter: dict[str, Any] | None = None,
    metric_filter: dict[str, Any] | None = None,
    order_bys: list[dict[str, Any]] | None = None,
    limit: int = RUN_REPORT_DEFAULT_LIMIT,
    offset: int = 0,
) -> str:
    """
    Run a GA4 report: quantify performance by any combination of dimensions
    (e.g. channel, campaign, landing page, segment) over one or more date
    ranges.

    metrics: metric names, e.g. ["sessions", "conversions", "totalRevenue"].
    date_ranges: one dict per period to compare (max 4), each with
      "start_date" and "end_date" (YYYY-MM-DD, or GA4 relative terms like
      "7daysAgo"/"yesterday"), and an optional "name" to label the period
      (e.g. "current", "previous") — pass two date_ranges to compare a
      period against a prior one in a single call; the response's rows are
      tagged with the matching name. Both bounds are inclusive, so use
      "yesterday" rather than "today" as the end_date to match the GA UI's
      "Last N days", and give every compared date_range the same number of
      non-overlapping days — the response's "note" field flags it if either
      isn't the case.
    dimensions: dimension names, e.g. ["sessionDefaultChannelGroup",
      "landingPagePlusQueryString"]. Use google_analytics_get_metadata to
      discover valid names for this property.
    dimension_filter: a raw GA4 FilterExpression dict, for narrowing results
      (e.g. to one campaign or landing page). Optional.
    metric_filter: a raw GA4 FilterExpression dict applied to metric values
      (e.g. "sessions > 100"). Optional.
    order_bys: a raw GA4 OrderBy list, e.g. to sort by sessions descending:
      [{"metric": {"metricName": "sessions"}, "desc": true}]. Note the field
      is "metricName" (camelCase); for a dimension it's
      [{"dimension": {"dimensionName": "date"}, "desc": true}].
    limit: max rows to return (default 100, max 150 — keeps a wide report's
      serialized response under the MCP output size limit).
    offset: row offset for paging. Follow next_offset only when it is present.

    The response always requests GA4's TOTAL metric aggregation. Use totals
    for headline totals; never replace them by summing rows. A metric can be
    non-additive across dimensions, so if the row sum differs from totals,
    report the difference and consult metadata for GA4-reported causes. With
    multiple date_ranges, totals contains one row per range, identified by
    the dateRange dimension value (the supplied name or date_range_N); do not
    interpret totals[0] as a grand total across ranges. The response echoes
    the normalized property ID and the other request inputs as supplied so
    values from concurrent calls cannot be attributed to the wrong query.

    Pagination fields distinguish an incomplete GA4 page from local output
    trimming. When output_truncated is true and next_offset is present,
    continue there because the retained rows are a prefix of the API page. If
    next_offset is absent, no row fit; retry the same offset with a narrower
    query.
    """
    try:
        normalized_property_id = _normalize_property_id(property_id)
        if not metrics:
            raise ValueError("metrics must contain at least one metric name")
        if not date_ranges:
            raise ValueError("date_ranges must contain at least one date range")
        if len(date_ranges) > 4:
            raise ValueError("GA4 allows at most 4 date_ranges per report")
        if limit < 1 or limit > RUN_REPORT_MAX_LIMIT:
            raise ValueError(f"limit must be between 1 and {RUN_REPORT_MAX_LIMIT}")
        if offset < 0:
            raise ValueError("offset must be >= 0")
        body: dict[str, Any] = {
            "metrics": [{"name": m} for m in metrics],
            "dateRanges": [_date_range_body(dr) for dr in date_ranges],
            # Keep the authoritative total in the same GA4 request as any
            # dimensional breakdown. This avoids a second request drifting
            # to a different property, range, or filter.
            "metricAggregations": ["TOTAL"],
            # Always sent explicitly: GA4's default would otherwise return up
            # to 10k rows in one response.
            "limit": str(limit),
        }
        if dimensions:
            body["dimensions"] = [{"name": d} for d in dimensions]
        if dimension_filter:
            body["dimensionFilter"] = dimension_filter
        if metric_filter:
            body["metricFilter"] = metric_filter
        if order_bys:
            body["orderBys"] = order_bys
        if offset > 0:
            body["offset"] = str(offset)

        result = _require_dict_result(
            _request(
                "POST",
                f"{DATA_API_BASE_URL}/properties/{normalized_property_id}:runReport",
                body=body,
            )
        )
        dimension_headers = result.get("dimensionHeaders") or []
        metric_headers = result.get("metricHeaders") or []
        totals = result.get("totals") or []
        metadata = result.get("metadata") or {}
        rows = list(result.get("rows") or [])
        row_count = int(result.get("rowCount") or 0)
        api_returned_count = len(rows)
        has_more = offset + api_returned_count < row_count

        query: dict[str, Any] = {
            "property_id": normalized_property_id,
            "metrics": list(metrics),
            "date_ranges": date_ranges,
            "limit": limit,
            "offset": offset,
        }
        if dimensions:
            query["dimensions"] = list(dimensions)
        if dimension_filter:
            query["dimension_filter"] = dimension_filter
        if metric_filter:
            query["metric_filter"] = metric_filter
        if order_bys:
            query["order_bys"] = order_bys

        # Surface two gotchas in the result itself, so the LLM catches them
        # even if it never reads the docstring.
        notes: list[str] = []
        if any(_ends_today(dr) for dr in date_ranges):
            # A date_range ending at "today" pulls in a partial,
            # still-processing day that the GA UI's equivalent "last N
            # days" report excludes.
            notes.append(
                'One or more date_ranges end at "today", which includes '
                "partial, still-processing data for the current day. This "
                "will not exactly match a GA UI report for the same nominal "
                "period, which uses only complete days. To match GA UI's "
                '"Last N days", use "yesterday" as the end_date instead.'
            )
        day_counts = [
            (dr.get("name") or f"range {i + 1}", _date_range_day_count(dr))
            for i, dr in enumerate(date_ranges)
        ]
        known_spans = {count for _, count in day_counts if count is not None}
        if len(known_spans) > 1:
            # Comparing date_ranges of different lengths (e.g. an 8-day
            # "this_week" against a 7-day "last_week") makes any
            # percentage change between them meaningless, independent of
            # whether either range touches "today".
            detail = ", ".join(
                f"{name}={count}d" for name, count in day_counts if count is not None
            )
            notes.append(
                f"date_ranges span different numbers of days ({detail}) -- "
                "a percentage change between periods of unequal length "
                "isn't meaningful. Make every compared date_range cover "
                "the same number of days."
            )
        overlapping_pairs = [
            (day_counts[i][0], day_counts[j][0])
            for i in range(len(date_ranges))
            for j in range(i + 1, len(date_ranges))
            if _date_ranges_overlap(date_ranges[i], date_ranges[j])
        ]
        if overlapping_pairs:
            # Equal-length, non-overlapping is necessary for a meaningful
            # comparison but not sufficient: e.g. "7daysAgo"/"today" and
            # "14daysAgo"/"7daysAgo" are both 8 days but share day -7.
            detail = ", ".join(f'"{a}" and "{b}"' for a, b in overlapping_pairs)
            notes.append(
                f"date_ranges {detail} overlap (share at least one day) -- "
                "a period comparison should use non-overlapping date_ranges."
            )
        extra_kwargs: dict[str, Any] = {}
        if notes:
            # Prepended (not appended) so it isn't the first thing an
            # output-length truncation cuts off a large report's response.
            extra_kwargs["note"] = " ".join(notes)
        # Multiple ranges add GA4's implicit dateRange dimension even when
        # the caller did not request any dimensions explicitly.
        if dimensions or len(date_ranges) > 1:
            if totals:
                extra_kwargs["aggregation_note"] = (
                    "Use totals for headline totals; do not replace totals by "
                    "summing rows. If the row sum differs, report the delta "
                    "and use metadata for GA4-reported thresholding, sampling, "
                    "or (other) row effects; do not invent another cause."
                )
            else:
                extra_kwargs["aggregation_note"] = (
                    "GA4 returned no totals even though TOTAL was requested. "
                    "Do not invent a total by summing rows."
                )
            if len(date_ranges) > 1 and totals:
                extra_kwargs["aggregation_note"] += (
                    " Multiple date_ranges produce one totals row per range; "
                    "match each total by its dateRange dimension value and do "
                    "not treat totals[0] as a grand total across ranges."
                )

        # RUN_REPORT_MAX_LIMIT bounds row count, not bytes: several long
        # dimension values (e.g. full URLs) at once can still cross the
        # platform's output truncation threshold. Halve the returned rows
        # until the serialized response fits, mirroring the truncated-flag
        # pattern in google_analytics_list_properties.
        max_output_length = get_tool_max_output_length()

        def _build_response(*, output_truncated: bool) -> str:
            next_offset = None
            pagination_hint = None
            if output_truncated:
                if rows:
                    next_offset = offset + len(rows)
                    pagination_hint = (
                        "Response rows were trimmed for output size. The rows "
                        "returned are the API page prefix; continue with "
                        f"offset={next_offset}."
                    )
                else:
                    pagination_hint = (
                        "Response rows were trimmed for output size. Retry the "
                        f"same offset={offset} with fewer metrics/dimensions or "
                        "narrower filters; do not advance the offset."
                    )
            elif has_more:
                next_offset = offset + api_returned_count
                pagination_hint = (
                    f"More GA4 rows are available; call again with "
                    f"offset={next_offset}."
                )

            return _success(
                **extra_kwargs,
                query=query,
                dimension_headers=dimension_headers,
                metric_headers=metric_headers,
                totals=totals,
                metadata=metadata,
                rows=rows,
                row_count=row_count,
                returned_count=len(rows),
                api_returned_count=api_returned_count,
                has_more=has_more,
                next_offset=next_offset,
                output_truncated=output_truncated,
                truncated=has_more or output_truncated,
                **(
                    {"pagination_hint": pagination_hint}
                    if pagination_hint is not None
                    else {}
                ),
            )

        response = _build_response(output_truncated=False)
        while len(response) > max_output_length and rows:
            rows = rows[: len(rows) // 2]
            response = _build_response(output_truncated=True)
        if len(rows) < api_returned_count:
            logger.warning(
                f"Google Analytics run_report response trimmed from "
                f"{api_returned_count} to {len(rows)} rows to stay under "
                f"the {max_output_length}-char output limit"
            )
        if len(response) > max_output_length:
            # Rows are the only optional part we can safely discard. The
            # echoed query identifies which property/range/filter produced
            # the totals, so truncating either query or totals would make the
            # result ambiguous or incorrect. Return a small, valid JSON error
            # instead of letting the platform cut an oversized success
            # response at an arbitrary character boundary.
            logger.warning(
                "Google Analytics run_report metadata and totals exceed the "
                f"{max_output_length}-char output limit after all rows were removed"
            )
            return _error(
                "Google Analytics report metadata and totals cannot fit the "
                "configured output limit even with all rows removed. Narrow "
                "the filters or request fewer metrics and dimensions."
            )
        return response
    except Exception as e:
        # !r escapes embedded newlines, keeping a crafted property_id from
        # injecting fake log lines.
        logger.error(f"Error running Google Analytics report for {property_id!r}: {e}")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
