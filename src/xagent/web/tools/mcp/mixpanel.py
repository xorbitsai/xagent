import json
import logging
import re
import time
from datetime import date, datetime
from os import environ
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

from ....core.utils.security import redact_sensitive_text
from ...utils.graphql_errors import truncate_error_text
from .utils import clamp_limit, setup_proxy_env, success_with_capped_dict, url_path_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mixpanel-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("mixpanel-mcp")

DEFAULT_TIMEOUT_SECONDS = 30
MAX_LIMIT = 100
MAX_RETRY_AFTER_SECONDS = 30
# mixpanel_export_events streams newline-delimited JSON that has no
# server-side row cap of its own (a wide date range can return millions of
# events) -- this is a client-side stop so one call can't pull an unbounded
# amount of data into the tool's own output.
MAX_EXPORT_EVENTS = 500

# Mixpanel splits its HTTP API across three host+path families per
# data-residency region, all sharing the same three regional hostnames
# (mixpanel.com / eu.mixpanel.com / in.mixpanel.com for US/EU/India) but
# different path prefixes and, for raw export, a different hostname
# altogether:
#   - Query API (segmentation/retention/funnels/engage/events/*):
#     https://{host}/api/query/...
#   - App API (annotations only): https://{host}/api/app/...
#   - Export API (raw event export): https://data[-eu|-in].mixpanel.com/api/2.0/export
# "query" below covers both the Query and App families (same hostname,
# callers pass the right path prefix); "export" is the separate data host.
# Unlike PostHog's connector, there is no user-supplied host to validate:
# MIXPANEL_REGION only ever selects one of these three known pairs, so
# there is no SSRF surface to guard here.
_REGION_HOSTS = {
    "us": {"query": "mixpanel.com", "export": "data.mixpanel.com"},
    "eu": {"query": "eu.mixpanel.com", "export": "data-eu.mixpanel.com"},
    "in": {"query": "in.mixpanel.com", "export": "data-in.mixpanel.com"},
}

_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}\Z")


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _auth() -> tuple[str, str]:
    # Stripped like MIXPANEL_REGION below: a value that's only whitespace
    # (e.g. a trailing newline from copy-pasting the credential into a
    # connect-flow form) is not a usable credential and should be treated
    # as missing here rather than sent to Mixpanel as a malformed Basic
    # Auth username/password.
    username = environ.get("MIXPANEL_SERVICE_ACCOUNT_USERNAME", "").strip()
    secret = environ.get("MIXPANEL_SERVICE_ACCOUNT_SECRET", "").strip()
    if not username:
        raise ValueError(
            "MIXPANEL_SERVICE_ACCOUNT_USERNAME environment variable is missing"
        )
    if not secret:
        raise ValueError(
            "MIXPANEL_SERVICE_ACCOUNT_SECRET environment variable is missing"
        )
    return (username, secret)


def _project_id() -> str:
    project_id = environ.get("MIXPANEL_PROJECT_ID", "").strip()
    if not project_id:
        raise ValueError("MIXPANEL_PROJECT_ID environment variable is missing")
    return project_id


def _region_hosts() -> dict[str, str]:
    region = environ.get("MIXPANEL_REGION", "us").strip().lower() or "us"
    hosts = _REGION_HOSTS.get(region)
    if hosts is None:
        raise ValueError(
            f"MIXPANEL_REGION must be one of {sorted(_REGION_HOSTS)}, got {region!r}"
        )
    return hosts


def _clamp_limit(limit: int) -> int:
    return clamp_limit(limit, max_limit=MAX_LIMIT)


def _validate_date(value: str, field_name: str) -> str:
    """Validate a Mixpanel date param (YYYY-MM-DD).

    Mixpanel's Query API silently accepts a malformed date and returns an
    opaque empty/zeroed result rather than a 4xx, so this catches the
    common mistake (wrong format, swapped month/day) with a clear error
    instead of a confusing "no data" response. Anchored with \\Z rather
    than $ -- $ matches immediately before a trailing newline as well as at
    the true end of the string, which would let e.g. "2026-01-15\\n" slip
    through as "valid". The regex alone accepts a shape-correct but
    calendar-invalid date like "2026-02-30"; date.fromisoformat() rejects
    that, matching google_search_console.py's identical validator.
    """
    if not isinstance(value, str) or not _DATE_PATTERN.match(value):
        raise ValueError(
            f"{field_name} must be a YYYY-MM-DD date string, got {value!r}"
        )
    try:
        date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{field_name} must be a valid calendar date") from None
    return value


def _validate_annotation_datetime(value: str, field_name: str) -> str:
    """Validate mixpanel_create_annotation's `date` param (YYYY-MM-DD
    HH:MM:SS) -- distinct from _validate_date above, since an annotation is
    anchored to a specific time, not just a day. Unvalidated, a malformed
    value here would reach Mixpanel's API as-is and likely produce a
    confusing error rather than a clear local one.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must be a YYYY-MM-DD HH:MM:SS string, got {value!r}"
        )
    try:
        datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        raise ValueError(
            f"{field_name} must be a YYYY-MM-DD HH:MM:SS string, got {value!r}"
        ) from None
    return value


def _extract_error_detail(response: requests.Response) -> str | None:
    """Pull the human-readable message out of a Mixpanel error body.

    Mixpanel's Query/App/Export APIs are an older, less consistent surface
    than its newer JSON APIs: most errors come back as {"error": "message"},
    but some (e.g. an auth failure) come back as plain text with no JSON
    envelope at all. Returns None so the caller falls back to the raw
    response text in that case.
    """
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    return error if isinstance(error, str) and error else None


def _request(
    method: str,
    host: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    form_data: dict[str, Any] | None = None,
    json_data: dict[str, Any] | None = None,
    include_project_id: bool = True,
    stream: bool = False,
) -> Any:
    """Call a Mixpanel API host and return the parsed response.

    Every call is authenticated with the Service Account's HTTP Basic
    credentials. include_project_id controls whether MIXPANEL_PROJECT_ID is
    injected as the `project_id` query param -- true for every Query/Export
    API endpoint, which are project-scoped via that param, but false for
    the App API's annotations endpoints, which instead take the project id
    as a URL path segment (the caller builds that into `path` itself).

    stream=True returns the raw (still-open) response for a 2xx result
    instead of buffering it into `.text`/`.json()` here -- used only by
    mixpanel_export_events, whose NDJSON body has no server-side row cap of
    its own and can be arbitrarily large; the caller is responsible for
    iterating and closing it (a `with` block). Every other outcome closes
    the response here regardless of `stream` before returning/raising --
    a redirect or a discarded 429 retry response never reaches the caller,
    and an error (>=400) response is read (error bodies are small) via
    `_extract_error_detail`/`.text`, which drains and effectively closes it
    -- so `stream=True` only ever hands back a connection the caller must
    manage on the single success path.
    """
    url = f"https://{host}{path}"
    request_params = dict(params or {})
    if include_project_id:
        request_params["project_id"] = _project_id()
    try:
        for attempt in (0, 1):
            response = requests.request(
                method=method,
                url=url,
                auth=_auth(),
                params=request_params,
                data=form_data,
                json=json_data,
                timeout=DEFAULT_TIMEOUT_SECONDS,
                # A redirect response is never followed with Basic Auth
                # credentials still attached: Mixpanel's documented API
                # doesn't redirect, so a 3xx here is either a
                # misconfiguration or a host trying to relay the
                # credentials elsewhere.
                allow_redirects=False,
                stream=stream,
            )
            if response.status_code == 429 and attempt == 0:
                try:
                    retry_after = int(response.headers.get("Retry-After", "0"))
                except ValueError:
                    retry_after = 0
                if 0 < retry_after <= MAX_RETRY_AFTER_SECONDS:
                    # With stream=True this response's connection was never
                    # read (a non-streamed 429 body is tiny and already
                    # drained by `requests` itself, so this is a no-op
                    # there) -- close it explicitly before discarding the
                    # reference, or the socket for this attempt is held
                    # open, unreleased, for the entire sleep below.
                    response.close()
                    time.sleep(retry_after)
                    continue
            break
    except requests.RequestException as exc:
        # A connection/timeout/proxy failure's message can itself embed
        # sensitive data -- e.g. a ProxyError echoing the ambient
        # HTTPS_PROXY URL, which may carry embedded user:pass@ credentials
        # (setup_proxy_env() exports whatever the OS has configured).
        raise RuntimeError(redact_sensitive_text(str(exc))) from exc

    if 300 <= response.status_code < 400:
        # Same as the 429-retry path above: closes a still-open stream=True
        # connection before this response is discarded (a no-op for the
        # non-streamed case, where the body is already drained).
        response.close()
        raise RuntimeError(
            f"Mixpanel returned an unexpected redirect (HTTP {response.status_code}); "
            "refusing to follow it with credentials attached"
        )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        message = str(exc)
        detail = _extract_error_detail(response)
        if detail is None:
            detail = truncate_error_text(response.text.strip())
        if detail:
            # The response body is attacker/host-controlled content, not
            # something this module wrote; if it happens to echo request
            # headers, this redacts Bearer tokens, key=/secret=-style
            # assignments, and a couple of named API-key headers -- NOT an
            # echoed "Authorization: Basic <base64>" (this connector's own
            # auth scheme), which redact_sensitive_text has no pattern for
            # today. Narrower coverage than the comment in sibling
            # Bearer-auth connectors this was adapted from implies for a
            # Basic-auth one; tracked as a follow-up to add a Basic-auth
            # pattern to core/utils/security.py rather than fixed here,
            # since that module is shared by every connector, not owned by
            # this file.
            message = f"{message} - {redact_sensitive_text(detail)}"
        raise RuntimeError(message) from exc

    if stream:
        return response
    if not response.content:
        return {}
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Mixpanel returned a 2xx response with a non-JSON body: {exc}"
        ) from exc


@mcp.tool()
def mixpanel_list_event_names(event_type: str = "general", limit: int = 50) -> str:
    """
    List event names tracked in this project, ordered by volume.
    event_type: "general" (top events), "unique" (events by unique users),
    or "average" (events by average count per user) -- Mixpanel's own
    categorization of how "top" is ranked.
    """
    try:
        max_results = _clamp_limit(limit)
        result = _request(
            "GET",
            _region_hosts()["query"],
            "/api/query/events/names",
            params={"type": event_type, "limit": max_results},
        )
        return success_with_capped_dict("event_names", {"event_names": result})
    except Exception as e:
        logger.error(f"Error listing Mixpanel event names: {e}")
        return _error(str(e))


@mcp.tool()
def mixpanel_get_top_events(event_type: str = "general", limit: int = 50) -> str:
    """
    Get the top events in this project with their volume and change vs. the
    previous period.
    event_type: "general", "unique", or "average" -- see
    mixpanel_list_event_names for what each means.
    """
    try:
        max_results = _clamp_limit(limit)
        result = _request(
            "GET",
            _region_hosts()["query"],
            "/api/query/events/top",
            params={"type": event_type, "limit": max_results},
        )
        return success_with_capped_dict("top_events", result)
    except Exception as e:
        logger.error(f"Error fetching Mixpanel top events: {e}")
        return _error(str(e))


@mcp.tool()
def mixpanel_get_event_properties(event: str, limit: int = 10) -> str:
    """
    Get the top property names tracked on a given event, with their volume.
    event: an event name from mixpanel_list_event_names.
    limit: max property names to return (Mixpanel's own default is 10).
    """
    try:
        max_results = _clamp_limit(limit)
        result = _request(
            "GET",
            _region_hosts()["query"],
            "/api/query/events/properties/top",
            params={"event": event, "limit": max_results},
        )
        return success_with_capped_dict("event_properties", result)
    except Exception as e:
        logger.error(f"Error fetching Mixpanel properties for event {event}: {e}")
        return _error(str(e))


@mcp.tool()
def mixpanel_query_segmentation(
    event: str,
    from_date: str,
    to_date: str,
    unit: str = "day",
    on: str = "",
    where: str = "",
) -> str:
    """
    Get segmented event counts over a date range -- the core "how many times
    did X happen, broken down by Y" analytics query.
    event: an event name from mixpanel_list_event_names.
    from_date, to_date: inclusive YYYY-MM-DD dates.
    unit: "minute", "hour", "day", or "month" -- the time bucket to group
    results by.
    on: an optional Mixpanel expression to segment/break down by, e.g.
    "properties[\\"$browser\\"]".
    where: an optional Mixpanel expression to filter events by, e.g.
    "properties[\\"plan\\"] == \\"pro\\"".
    """
    try:
        _validate_date(from_date, "from_date")
        _validate_date(to_date, "to_date")
        params: dict[str, Any] = {
            "event": event,
            "from_date": from_date,
            "to_date": to_date,
            "unit": unit,
        }
        if on:
            params["on"] = on
        if where:
            params["where"] = where
        result = _request(
            "GET", _region_hosts()["query"], "/api/query/segmentation", params=params
        )
        return success_with_capped_dict("segmentation", result)
    except Exception as e:
        logger.error(f"Error running Mixpanel segmentation for event {event}: {e}")
        return _error(str(e))


@mcp.tool()
def mixpanel_query_retention(
    from_date: str,
    to_date: str,
    retention_type: str = "birth",
    born_event: str = "",
    event: str = "",
    born_where: str = "",
    where: str = "",
    unit: str = "day",
) -> str:
    """
    Get a retention report: of the users who did a "born" event in a given
    period, what fraction came back and did a (or the same) event later.
    from_date, to_date: inclusive YYYY-MM-DD dates for the born-event window.
    retention_type: "birth" (first time someone does born_event) or
    "compounded" (any time someone does born_event).
    born_event: the event that starts a user's retention cohort --
    required when retention_type is "birth" (Mixpanel has no "any event"
    fallback for that case, unlike "compounded"); optional otherwise.
    event: the event that counts as a user "returning"; defaults to
    born_event if omitted (classic retention).
    born_where, where: optional Mixpanel expressions filtering the born and
    return events respectively.
    unit: "day", "week", or "month" -- the retention bucket size.
    """
    try:
        _validate_date(from_date, "from_date")
        _validate_date(to_date, "to_date")
        if retention_type == "birth" and not born_event:
            raise ValueError('born_event is required when retention_type is "birth"')
        params: dict[str, Any] = {
            "from_date": from_date,
            "to_date": to_date,
            "retention_type": retention_type,
            "unit": unit,
        }
        if born_event:
            params["born_event"] = born_event
        if event:
            params["event"] = event
        if born_where:
            params["born_where"] = born_where
        if where:
            params["where"] = where
        result = _request(
            "GET", _region_hosts()["query"], "/api/query/retention", params=params
        )
        return success_with_capped_dict("retention", result)
    except Exception as e:
        logger.error(f"Error running Mixpanel retention query: {e}")
        return _error(str(e))


@mcp.tool()
def mixpanel_list_funnels() -> str:
    """
    List saved funnels in this project -- funnel_id and name. Use the
    returned funnel_id with mixpanel_query_funnel.
    """
    try:
        result = _request("GET", _region_hosts()["query"], "/api/query/funnels/list")
        return success_with_capped_dict("funnels", {"funnels": result})
    except Exception as e:
        logger.error(f"Error listing Mixpanel funnels: {e}")
        return _error(str(e))


@mcp.tool()
def mixpanel_query_funnel(
    funnel_id: int, from_date: str, to_date: str, unit: str = "day"
) -> str:
    """
    Get conversion data for one saved funnel over a date range.
    funnel_id: a funnel id from mixpanel_list_funnels.
    from_date, to_date: inclusive YYYY-MM-DD dates.
    unit: "day", "week", or "month" -- the time bucket to group results by.
    """
    try:
        _validate_date(from_date, "from_date")
        _validate_date(to_date, "to_date")
        result = _request(
            "GET",
            _region_hosts()["query"],
            "/api/query/funnels",
            params={
                "funnel_id": funnel_id,
                "from_date": from_date,
                "to_date": to_date,
                "unit": unit,
            },
        )
        return success_with_capped_dict("funnel", result)
    except Exception as e:
        logger.error(f"Error fetching Mixpanel funnel {funnel_id}: {e}")
        return _error(str(e))


@mcp.tool()
def mixpanel_query_engage(
    where: str = "", output_properties: str = "", session_id: str = "", page: int = 0
) -> str:
    """
    Query user profiles (Mixpanel's "Engage" store) -- distinct_id and
    profile properties, optionally filtered. Mixpanel paginates results
    server-side in fixed-size pages; there is no client-specified limit.
    where: an optional Mixpanel expression to filter profiles by, e.g.
    "properties[\\"$email\\"] is set".
    output_properties: optional comma-separated property names to return,
    e.g. "$email,$last_name" -- restricting this can speed up large queries.
    session_id, page: pass the previous call's own session_id and page + 1
    to fetch the next page -- Mixpanel's engage results are cursor-paginated
    server-side, not by a client-specified offset. page is only meaningful
    together with session_id; passing page without session_id is rejected
    rather than silently ignored.
    """
    try:
        if page and not session_id:
            return _error("page requires session_id from a previous call's response")
        form: dict[str, Any] = {}
        if where:
            form["where"] = where
        if output_properties:
            properties = [p.strip() for p in output_properties.split(",") if p.strip()]
            if properties:
                form["output_properties"] = json.dumps(properties)
        if session_id:
            form["session_id"] = session_id
            form["page"] = page
        result = _request(
            "POST", _region_hosts()["query"], "/api/query/engage", form_data=form
        )
        return success_with_capped_dict("engage", result)
    except Exception as e:
        logger.error(f"Error querying Mixpanel engage profiles: {e}")
        return _error(str(e))


@mcp.tool()
def mixpanel_list_annotations(from_date: str, to_date: str) -> str:
    """
    List annotations (notes marking a point in time, e.g. a deploy or
    incident) in a date range.
    from_date, to_date: inclusive YYYY-MM-DD dates.
    """
    try:
        _validate_date(from_date, "from_date")
        _validate_date(to_date, "to_date")
        result = _request(
            "GET",
            _region_hosts()["query"],
            f"/api/app/projects/{url_path_id(_project_id(), 'project_id')}/annotations",
            params={"fromDate": from_date, "toDate": to_date},
            include_project_id=False,
        )
        annotations = (
            (result.get("results") or []) if isinstance(result, dict) else result
        )
        return success_with_capped_dict("annotations", {"annotations": annotations})
    except Exception as e:
        logger.error(f"Error listing Mixpanel annotations: {e}")
        return _error(str(e))


@mcp.tool()
def mixpanel_create_annotation(date: str, description: str) -> str:
    """
    Create an annotation -- a note marking a point in time on Mixpanel's
    graphs, e.g. "Deployed v2.3" or "Started incident".
    date: the "YYYY-MM-DD HH:MM:SS" timestamp the annotation is anchored to.
    description: the annotation's text.
    """
    try:
        _validate_annotation_datetime(date, "date")
        result = _request(
            "POST",
            _region_hosts()["query"],
            f"/api/app/projects/{url_path_id(_project_id(), 'project_id')}/annotations",
            json_data={"date": date, "description": description},
            include_project_id=False,
        )
        return _success(annotation=result)
    except Exception as e:
        logger.error(f"Error creating Mixpanel annotation: {e}")
        return _error(str(e))


@mcp.tool()
def mixpanel_export_events(
    from_date: str, to_date: str, event: str = "", where: str = ""
) -> str:
    """
    Export raw event data over a date range -- individual events with their
    full property payload, not an aggregate report.
    from_date, to_date: inclusive YYYY-MM-DD dates. Mixpanel's raw export
    has no server-side row limit, so a wide date range can be extremely
    large; this tool asks Mixpanel to cap the response at MAX_EXPORT_EVENTS
    events itself (via the API's own `limit` param) and additionally
    enforces that cap client-side, reporting row_limit_reached=true if more
    were available -- narrow the date range or add event/where filters to
    see the rest. stream_error=true instead means the connection broke or
    returned a malformed line partway through -- the returned events are
    only what was successfully read before that point, not a complete
    answer for the requested range even though row_limit_reached is false;
    retry the same call to get the rest.
    event: an optional single event name to filter by.
    where: an optional Mixpanel expression to filter events by.
    """
    try:
        _validate_date(from_date, "from_date")
        _validate_date(to_date, "to_date")
        params: dict[str, Any] = {
            "from_date": from_date,
            "to_date": to_date,
            "limit": MAX_EXPORT_EVENTS,
        }
        if event:
            params["event"] = json.dumps([event])
        if where:
            params["where"] = where
        response = _request(
            "GET",
            _region_hosts()["export"],
            "/api/2.0/export",
            params=params,
            stream=True,
        )
        events: list[Any] = []
        row_limit_reached = False
        stream_error = False
        # Streamed rather than buffered whole: the `limit` param above asks
        # Mixpanel to cap the export server-side, but this is the actual
        # enforcement -- iterating line-by-line and stopping (closing the
        # connection via the `with` block) at MAX_EXPORT_EVENTS means a
        # wide date range can never pull more than that many events'
        # worth of data over the wire, regardless of whether Mixpanel
        # honors `limit` on every account/plan.
        with response:
            for line in response.iter_lines():
                if not line:
                    continue
                if len(events) >= MAX_EXPORT_EVENTS:
                    row_limit_reached = True
                    break
                try:
                    events.append(json.loads(line))
                except ValueError:
                    # A truncated/malformed line -- realistic on a
                    # mid-stream connection reset (requests.iter_lines()
                    # can split a chunk boundary mid-record) -- is treated
                    # as the effective end of the stream rather than
                    # discarding every event already parsed by letting this
                    # propagate to the outer except below. Tracked as a
                    # distinct signal from row_limit_reached: this stopped
                    # because the stream broke, not because the requested
                    # range was fully covered up to the row cap, and a
                    # caller conflating the two would wrongly read a
                    # truncated result as "the complete answer."
                    stream_error = True
                    break
        # No "count" field: success_with_capped_dict can still halve
        # `events` further if the JSON payload built here exceeds the
        # platform's own output-size cap, which would leave a
        # precomputed count out of sync with the array actually returned.
        # Its own "truncated" flag (size-driven), row_limit_reached (this
        # function's own MAX_EXPORT_EVENTS cap), and stream_error (an
        # abnormal mid-stream stop) are the signals that stay accurate;
        # len(events) is trivial for a caller to derive from the array
        # itself.
        return success_with_capped_dict(
            "events",
            {
                "events": events,
                "row_limit_reached": row_limit_reached,
                "stream_error": stream_error,
            },
        )
    except Exception as e:
        logger.error(f"Error exporting Mixpanel events: {e}")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
