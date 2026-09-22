import json
import logging
import os
import time
from typing import Any
from urllib.parse import quote

import requests
from mcp.server.fastmcp import FastMCP

from ....config import get_tool_max_output_length
from ....core.utils.security import redact_sensitive_text
from ...utils.graphql_errors import truncate_error_text
from .utils import (
    clamp_limit,
    clamp_offset,
    require_clean_text,
    setup_proxy_env,
    success_with_capped_dict,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jira-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("jira-mcp")

ACCESSIBLE_RESOURCES_URL = "https://api.atlassian.com/oauth/token/accessible-resources"
ME_URL = "https://api.atlassian.com/me"
JIRA_API_BASE = "https://api.atlassian.com/ex/jira"
DEFAULT_TIMEOUT_SECONDS = 30
MAX_LIMIT = 100
# Jira endpoints are rate-limited; on a 429 with a small Retry-After we wait
# once and retry rather than failing outright, mirroring the same bounded-
# retry policy as the Slack/Intercom sibling modules.
MAX_RETRY_AFTER_SECONDS = 30


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _safe_text(value: Any) -> str:
    """Redact credential-shaped substrings (Bearer tokens, key=/secret=
    -style assignments, URL credentials) out of arbitrary text before it
    reaches a log line or a tool's returned error message.

    str(exc) is often built from data this module doesn't control --
    proxy/gateway error bodies, low-level connection-error reprs that
    can embed request details -- so it isn't guaranteed free of the
    Authorization header _headers() sets on every request. Matches
    shopify.py's/mixpanel.py's _safe_text convention for the same risk.

    Callers must apply this exactly once, at the point a message is
    first captured from an exception -- redact_sensitive_text is NOT
    idempotent (its masking can itself look credential-shaped enough to
    get masked again), so _error() does not call this itself: a binary
    search that re-invokes _error() on many different slices of the
    same message (jira_search_issues' _bounded_search_error does this)
    would have a second redaction pass per slice keep shrinking the
    masked portion on every call, breaking the "longer slice never
    produces shorter output" assumption that kind of search's
    correctness depends on.
    """
    return redact_sensitive_text(str(value))


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _headers() -> dict[str, str]:
    access_token = os.environ.get("JIRA_ACCESS_TOKEN")
    if not access_token:
        raise ValueError("JIRA_ACCESS_TOKEN environment variable is missing")
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _clean_text_error(value: str, field_name: str, *, hint: str = "") -> str | None:
    """Return an _error(...) payload if `value` fails require_clean_text,
    else None -- callers do `if (err := _clean_text_error(...)): return err`.

    A caller-input validation failure here is a routine, expected rejection
    (an agent will retry with a fixed value), not a connector/API error, so
    it's handled the same way for every field: caught locally and returned
    directly, never left to propagate into the surrounding try/except's
    `logger.error` -- which is reserved for genuine request failures.
    """
    try:
        require_clean_text(value, field_name)
    except ValueError as exc:
        return _error(f"{exc}{hint}")
    return None


def _path_segment(value: str) -> str:
    """Percent-encode a value for safe interpolation into a URL path
    segment (e.g. an issue key or cloud id), matching hubspot.py's
    _url_path_id / intercom.py's inline quote() calls. Percent-encoding -
    not a blocklist of "/", "?", "#" - is what actually prevents a value
    like "ENG-1/../other" from escaping its intended path segment.

    "." and ".." are the one exception that survives encoding unchanged
    (always-unreserved per RFC 3986, so quote() never touches them), and
    requests/urllib3 normalize dot-segments out of the final URL before
    sending -- collapsing e.g. ".../issue/.." to ".../issue" (or further),
    a different endpoint than the one requested. Rejected explicitly,
    matching utils.py's url_path_id, since encoding can't close this off.

    Also rejects a blank or whitespace-padded value (url_path_id's
    require_clean_identifier half) -- an empty issue_key (e.g. an
    unresolved templated variable from an LLM caller) would otherwise
    silently build /rest/api/2/issue/, hitting the issue-collection
    endpoint instead of a clear local error naming the actual mistake.
    Rejects None explicitly for the same reason, ahead of the str()
    coercion below: str(None) is the non-blank, non-padded string
    "None", which would otherwise sail through both checks above and
    silently become a literal path segment instead of raising.
    """
    if value is None:
        raise ValueError("path segment must not be None")
    text = str(value)
    if text in (".", ".."):
        raise ValueError(f"invalid path segment: {text!r}")
    if not text or text.strip() != text:
        raise ValueError(f"path segment must not be blank or padded: {text!r}")
    return quote(text, safe="")


def _issue_path(issue_key: str, suffix: str = "") -> str:
    """Build an issue-scoped REST path with the issue key percent-encoded.

    Single choke point for every /rest/api/2/issue/{key}... path so a new
    issue-scoped tool can't forget _path_segment and reopen the
    path-escape the encoding exists to prevent.
    """
    return f"/rest/api/2/issue/{_path_segment(issue_key)}{suffix}"


def _extract_error_detail(response: requests.Response) -> str | None:
    """Pull the human-readable message out of a Jira error body.

    Jira error responses are typically {"errorMessages": [...], "errors":
    {field: message}}; joining both is more useful to the LLM than the raw
    envelope. Returns None if the body isn't in the expected shape, so the
    caller can fall back to the raw response text.
    """
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    raw_messages = payload.get("errorMessages")
    if isinstance(raw_messages, list):
        messages = [str(m) for m in raw_messages]
    elif isinstance(raw_messages, str):
        messages = [raw_messages]
    else:
        messages = []
    field_errors = payload.get("errors")
    if isinstance(field_errors, dict):
        messages.extend(f"{field}: {msg}" for field, msg in field_errors.items())
    if not messages:
        return None
    return truncate_error_text("; ".join(messages))


def _request_absolute(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    json_data: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    allow_retry: bool = True,
) -> Any:
    for attempt in (0, 1):
        response = requests.request(
            method=method,
            url=url,
            headers=_headers(),
            params=params,
            json=json_data,
            timeout=timeout,
        )
        if allow_retry and response.status_code == 429 and attempt == 0:
            try:
                retry_after = int(response.headers.get("Retry-After", "0"))
            except ValueError:
                retry_after = 0
            if 0 < retry_after <= MAX_RETRY_AFTER_SECONDS:
                time.sleep(retry_after)
                continue
        break

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        message = str(exc)
        detail = _extract_error_detail(response)
        if detail is None:
            detail = truncate_error_text(response.text.strip())
        if detail:
            message = f"{message} - {detail}"
        raise RuntimeError(message) from exc

    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


def _accessible_resources() -> list[dict[str, Any]]:
    result = _request_absolute("GET", ACCESSIBLE_RESOURCES_URL)
    if not isinstance(result, list):
        raise ValueError(
            "Unexpected response format from Jira accessible-resources API"
        )
    return result


def _resolve_cloud_id(cloud_id: str) -> str:
    """Resolve cloud_id, auto-detecting it when there's exactly one
    accessible Jira site (the common case) -- Jira has no magic "current
    site" path segment, so this is done by actually listing sites.
    """
    if cloud_id:
        return cloud_id
    sites = _accessible_resources()
    if not sites:
        raise ValueError("No accessible Jira sites found for this account")
    if len(sites) == 1:
        site_id = sites[0].get("id") if isinstance(sites[0], dict) else None
        # Presence, not truthiness: an id is opaque and never contractually
        # excludes a falsy-but-valid value like 0 or "0".
        if site_id is None:
            raise ValueError("The single accessible Jira site is missing a valid 'id'")
        return str(site_id)
    site_list = (
        ", ".join(
            f"{s.get('name') or 'Unknown'} ({s.get('id') or 'No ID'})"
            for s in sites
            if isinstance(s, dict)
        )
        or "details unavailable -- response entries were not in the expected shape"
    )
    raise ValueError(
        f"Multiple Jira sites are accessible ({site_list}) -- call "
        "jira_list_accessible_sites and pass cloud_id explicitly"
    )


def _request(
    method: str,
    cloud_id: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_data: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    allow_retry: bool = True,
) -> Any:
    resolved_cloud_id = _resolve_cloud_id(cloud_id)
    return _request_absolute(
        method,
        f"{JIRA_API_BASE}/{_path_segment(resolved_cloud_id)}{path}",
        params=params,
        json_data=json_data,
        timeout=timeout,
        allow_retry=allow_retry,
    )


@mcp.tool()
def jira_list_accessible_sites() -> str:
    """
    List the Atlassian sites (Jira Cloud instances) this account's OAuth
    grant can access -- id (cloud_id), name, url, and granted scopes.
    Every other tool here takes an optional cloud_id; pass one from here
    when the account has more than one accessible site (auto-resolved
    without asking when there's only one).
    """
    try:
        sites = _accessible_resources()
        return _success(sites=sites)
    except Exception as e:
        safe_message = _safe_text(e)
        logger.error(f"Error listing accessible Jira sites: {safe_message}")
        return _error(safe_message)


@mcp.tool()
def jira_get_current_user() -> str:
    """
    Get the profile of the Atlassian account this connector is
    authenticated as (account_id, email, name). Use this for "my account" /
    "who am I" requests instead of asking the user for their Jira account id.
    """
    try:
        result = _request_absolute("GET", ME_URL)
        if not isinstance(result, dict):
            return _error("Unexpected response format from Atlassian profile API")
        return _success(
            user={
                "account_id": result.get("account_id"),
                "email": result.get("email"),
                "name": result.get("name"),
                "picture": result.get("picture"),
            }
        )
    except Exception as e:
        safe_message = _safe_text(e)
        logger.error(f"Error fetching authenticated Jira user: {safe_message}")
        return _error(safe_message)


def _summarize_project(project: dict[str, Any]) -> dict[str, Any]:
    """Drop a project's avatarUrls (an id/emoji/size ladder of icon
    links) -- the id/key/name/type are all a caller ever needs to pick a
    project for jira_create_issue/jira_search_issues.
    """
    return {
        "id": project.get("id"),
        "key": project.get("key"),
        "name": project.get("name"),
        "project_type_key": project.get("projectTypeKey"),
    }


def _fit_raw_list_page(
    key: str,
    raw_items: list[Any],
    offset: int,
    base_truncated: bool,
    base_next_start_at: int | None,
    max_output_length: int,
) -> str | None:
    """Build {key: kept[:count], truncated, next_start_at} at the
    largest count (halving from len(kept)) that fits max_output_length,
    for a raw (unsummarized) list page -- the summarized path's items
    are small enough per-entry that this has never been a practical
    concern, but a raw one (avatarUrls, permissions, and other metadata
    the summary drops) can be large enough to need the same
    halve-until-it-fits treatment _fit_comments_page uses.

    raw_items is the UNFILTERED raw page (may contain non-dict entries
    from a malformed response) -- filtered here, pairing each kept item
    with its RAW index, rather than by the caller pre-filtering: if the
    count has to shrink, next_start_at is computed from the actual raw
    position of the last item kept (mirroring _fit_comments_page's
    `kept` pattern), not from `offset + count`, which is only correct
    when zero entries were filtered out ahead of the cut -- otherwise a
    caller resuming from it would either skip an already-filtered-past
    entry or re-fetch (duplicate) one already delivered. At count=0,
    this points one past the first kept item's raw index, not at
    `offset` itself -- unlike a naive `offset + count`, which would
    equal the caller's own start_at and make a caller mechanically
    following next_start_at loop on this same page forever.
    """
    kept: list[tuple[int, dict[str, Any]]] = [
        (index, item) for index, item in enumerate(raw_items) if isinstance(item, dict)
    ]

    def next_start_at_for(count: int) -> int | None:
        if count == len(kept):
            return base_next_start_at
        return offset + kept[max(count - 1, 0)][0] + 1

    def response_for(count: int) -> str:
        truncated = base_truncated if count == len(kept) else True
        items = [item for _, item in kept[:count]]
        return _success(
            **{key: items}, truncated=truncated, next_start_at=next_start_at_for(count)
        )

    count = len(kept)
    response = response_for(count)
    while len(response) > max_output_length and count > 0:
        count //= 2
        response = response_for(count)
    return response if len(response) <= max_output_length else None


@mcp.tool()
def jira_list_projects(
    cloud_id: str = "", limit: int = 50, start_at: int = 0, raw_fields: bool = False
) -> str:
    """
    List projects on a Jira site -- id, key (e.g. "ENG"), name, and
    project_type_key. Use the returned key with jira_create_issue and
    jira_search_issues. Other project metadata (lead, category, archived/
    private flags) is not returned unless raw_fields is set.
    cloud_id: optional site id from jira_list_accessible_sites; omit when
    the account has only one accessible site.
    start_at: offset into the full project list -- pass the previous
    response's next_start_at to fetch the next page (0 to start over).
    raw_fields: return each project as Jira's own object (avatarUrls,
    lead, category, archived/private flags, and everything else this
    tool normally drops) instead of the summary above -- for an
    existing integration written against the raw shape this tool
    returned before summarization was added. Bigger per-project payload,
    so fewer projects may fit a page than the same request would at the
    default setting, and a page can shrink below what `limit` asked for
    to stay within the output budget -- always check `truncated` and
    resume from next_start_at rather than assuming one page is
    everything.
    """
    try:
        max_results = clamp_limit(limit, max_limit=MAX_LIMIT)
        offset = clamp_offset(start_at)
        result = _request(
            "GET",
            cloud_id,
            "/rest/api/2/project/search",
            params={"maxResults": max_results, "startAt": offset},
        )
        if not isinstance(result, dict):
            return _error("Unexpected response format from Jira projects API")
        raw_projects = result.get("values") or []
        # bool(raw_projects) guards against a server that signals more
        # pages while returning an empty page: without it next_start_at
        # would equal start_at and a caller following it would loop
        # forever. Counting the RAW page (not the filtered list below) --
        # same pattern as jira_search_users below -- matters because
        # Jira's startAt is positional over the raw page: undercounting
        # by however many entries got filtered out would make the next
        # request re-fetch (duplicate) entries already consumed here.
        truncated = bool(raw_projects) and not result.get("isLast", True)
        next_start_at = (offset + len(raw_projects)) if truncated else None
        if raw_fields:
            response = _fit_raw_list_page(
                "projects",
                raw_projects,
                offset,
                truncated,
                next_start_at,
                get_tool_max_output_length(),
            )
            if response is None:
                return _error(
                    "A Jira projects page exceeds the tool output limit even "
                    "when empty; fetch it individually or narrow the request"
                )
            return response
        projects = [_summarize_project(p) for p in raw_projects if isinstance(p, dict)]
        return _success(
            projects=projects, truncated=truncated, next_start_at=next_start_at
        )
    except Exception as e:
        safe_message = _safe_text(e)
        logger.error(f"Error listing Jira projects: {safe_message}")
        return _error(safe_message)


def _as_dict(value: Any) -> dict[str, Any]:
    """Return value if it's a dict, else {}.

    A single choke point for the "unwrap an optional Jira object field"
    pattern used throughout this module, so every such unwrap gets the
    same isinstance guard the array-shaped fields (components,
    fixVersions, issuelinks, subtasks) already get -- a non-dict truthy
    value here (e.g. a malformed gateway response serializing an object
    field as a bare string, or a misconfigured custom field) would
    otherwise raise AttributeError on the next `.get(...)` and fail the
    whole tool call.
    """
    return value if isinstance(value, dict) else {}


def _summarize_issue(issue: dict[str, Any]) -> dict[str, Any]:
    """Slim a raw Jira issue down to the handful of fields a search/triage
    view needs.

    A raw issue carries `expand`/`self`/`avatarUrls`/`iconUrl` on every
    nested object (status, priority, issuetype, project, ...), which runs
    ~3-4 KB per issue. jira_search_issues can return up to MAX_LIMIT (100)
    of them in one call, so an unslimmed page routinely exceeds the
    per-tool-output-string cap (see OutputFilteredToolWrapper) and gets cut
    off mid-JSON -- including the `truncated`/`next_page_token` fields at
    the tail, so the caller can't even tell pagination was in play.
    """
    # _as_dict guards every nested unwrap: a non-dict truthy value on
    # any of these (e.g. a misconfigured custom field, or a proxy that
    # reshapes one field on one malformed issue in an otherwise-good
    # page) would otherwise raise AttributeError on the next .get(...)
    # from inside the list comprehension in _fetch_and_summarize_page,
    # failing the ENTIRE page over a single bad issue instead of just
    # that issue degrading gracefully.
    fields = _as_dict(issue.get("fields"))
    assignee = _as_dict(fields.get("assignee"))
    status = _as_dict(fields.get("status"))
    priority = _as_dict(fields.get("priority"))
    issuetype = _as_dict(fields.get("issuetype"))
    project = _as_dict(fields.get("project"))
    parent = _as_dict(fields.get("parent"))
    resolution = _as_dict(fields.get("resolution"))
    return {
        "key": issue.get("key"),
        "summary": fields.get("summary"),
        "status": status.get("name"),
        "status_category": _as_dict(status.get("statusCategory")).get("name"),
        "assignee": (
            {
                "account_id": assignee.get("accountId"),
                "display_name": assignee.get("displayName"),
            }
            if assignee
            else None
        ),
        "priority": priority.get("name"),
        "issue_type": issuetype.get("name"),
        "project_key": project.get("key"),
        "labels": (
            fields.get("labels") if isinstance(fields.get("labels"), list) else []
        ),
        "parent_key": parent.get("key"),
        "resolution": resolution.get("name"),
        "created": fields.get("created"),
        "updated": fields.get("updated"),
    }


#: A short, no-retry budget for the advisory approximate-count call. The
#: primary search has already succeeded by the time this runs, so a slow
#: or rate-limited count endpoint must not hold the whole tool call for
#: up to DEFAULT_TIMEOUT_SECONDS (or longer, via the normal 429
#: retry-sleep) just for a number that is allowed to come back None.
_APPROXIMATE_COUNT_TIMEOUT_SECONDS = 5

#: A conservative page size to fall back to if the summarized page still
#: doesn't fit the output budget at the caller's requested `limit` (e.g.
#: very long summaries/labels at limit=100, or a lower configured
#: XAGENT_TOOL_MAX_OUTPUT_LENGTH). Even a worst-case issue (255-char
#: summary, several labels) runs under 1 KB summarized, so 10 of them
#: comfortably fits any realistic budget.
_RETRY_PAGE_SIZE = 10

#: Timeout/retry budget for the SECOND-and-later attempts in
#: jira_search_issues' page-size fallback loop (never the first, which
#: keeps the normal DEFAULT_TIMEOUT_SECONDS/allow_retry=True so the
#: caller's actual requested page respects Jira's real backpressure).
#: Without this, up to 3 attempts could each independently hit a 429
#: and sleep up to MAX_RETRY_AFTER_SECONDS before retrying -- a single
#: tool call stacking towards ~4.5 minutes of worst-case latency purely
#: to recover from an output-budget overflow, the same "advisory/
#: recovery work must not hold the primary call hostage" principle
#: already applied to _approximate_count, just for a different reason
#: (bounding a self-inflicted retry instead of skipping optional data).
_FALLBACK_ATTEMPT_TIMEOUT_SECONDS = 10

#: Rough upper bound on how many extra characters replacing
#: "approximate_total_count": null with an actual number can add (the
#: field's key, comma, and quoting are already present in the two-null
#: page, so the only real delta is null (4 chars) -> up to a many-digit
#: integer). Checked before calling _approximate_count so a page already
#: close to max_output_length skips the network round trip for a count
#: that's about to be discarded anyway, rather than fetching it and
#: finding out after the fact.
_TOTAL_COUNT_RESERVE_CHARS = 20

#: Fields requested for jira_search_issues. Kept in one place so every
#: page-size attempt in jira_search_issues' fallback loop below requests
#: the same fields as the first.
_SEARCH_ISSUE_FIELDS = (
    "summary,status,assignee,priority,issuetype,project,"
    "updated,created,resolution,labels,parent"
)


def _page_size_candidates(max_results: int) -> tuple[int, ...]:
    """Descending, deduplicated page sizes to try in jira_search_issues:
    the caller's requested size, then _RETRY_PAGE_SIZE, then 1 -- each
    only kept if it's smaller than the one before and no larger than
    what the caller asked for.

    Without the final floor of 1, a caller-requested `limit` at or below
    _RETRY_PAGE_SIZE (e.g. limit=3) would skip every fallback and go
    straight to a bounded error that claims "even at a minimal page
    size" without ever actually trying one.
    """
    candidates: list[int] = []
    for size in (max_results, _RETRY_PAGE_SIZE, 1):
        if size <= max_results and (not candidates or size < candidates[-1]):
            candidates.append(size)
    return tuple(candidates)


def _log_metadata_only(level_fn: Any, action: str, exc: Exception | None) -> None:
    """Log `action` with, if this is an exception path, only the
    exception's type -- never the JQL itself or the exception's message.
    Jira's own JQL-syntax-error text routinely echoes back the literal
    clause that failed to parse (which can carry whatever sensitive text
    the caller searched for), so both are kept out of every log line
    touching a search. No query-derived identifier (e.g. a hash of the
    JQL) is logged either: JQL is caller-controlled and often low-
    entropy/structured (e.g. `project = ENG`, `text ~ "login bug"`), so
    an unsalted digest would let anyone with log access confirm a
    specific query via a small offline dictionary of likely predicates --
    not the secrecy boundary logging usually assumes a hash provides.
    Shared by every log site touching a JQL search so the no-raw-content
    policy has one place to read and one place to change, instead of
    being restated at each call site.
    """
    suffix = f": {type(exc).__name__}" if exc is not None else ""
    level_fn(f"{action}{suffix}")


def _approximate_count(cloud_id: str, jql: str) -> int | None:
    """Best-effort match count via /search/approximate-count, so a caller
    can tell "you're seeing all N matches" apart from "there are more".

    Returns None (never raises) on any failure -- the JQL search itself
    already succeeded by the time this is called, so a count endpoint
    that rejects an unbounded query or errors out must not fail the
    whole tool call over a number that is advisory anyway.
    """
    try:
        result = _request(
            "POST",
            cloud_id,
            "/rest/api/3/search/approximate-count",
            json_data={"jql": jql},
            timeout=_APPROXIMATE_COUNT_TIMEOUT_SECONDS,
            allow_retry=False,
        )
        if isinstance(result, dict):
            count = result.get("count")
            # bool is a subclass of int in Python -- exclude it
            # explicitly, or a `"count": true` shape anomaly would
            # silently return the bool as if it were a real count
            # instead of hitting the non-integer warning below.
            if isinstance(count, int) and not isinstance(count, bool):
                return count
            if count is not None:
                # Distinguish "the endpoint answered but with a shape we
                # don't recognize" (e.g. a proxy re-serializing the count
                # as a float or numeric string) from a genuine failure --
                # both currently return None, but only this one has no
                # exception to explain itself otherwise.
                logger.warning(
                    "Approximate count endpoint returned a non-integer "
                    f"count: {type(count).__name__}"
                )
        else:
            # Same "leave a breadcrumb" reasoning as the non-integer-count
            # case above: a proxy/gateway reshaping the whole body (not
            # just the count field) to a non-dict must not degrade to
            # total_count=None with zero log signal, indistinguishable
            # from "endpoint healthy, count genuinely unavailable".
            logger.warning(
                "Approximate count endpoint returned an unexpected "
                f"response shape: {type(result).__name__}"
            )
    except Exception as e:
        _log_metadata_only(logger.warning, "Could not get an approximate count", e)
    return None


def _fetch_and_summarize_page(
    cloud_id: str,
    jql: str,
    max_results: int,
    next_page_token: str,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    allow_retry: bool = True,
    raw_fields: bool = False,
) -> tuple[list[dict[str, Any]], str | None]:
    """One search/jql call, projected through _summarize_issue unless
    raw_fields is set (see jira_search_issues' raw_fields param).

    Raises on any request/shape failure; callers run this inside their
    own try/except (jira_search_issues' outer one already does).
    """
    # /rest/api/2/search and /rest/api/3/search are deprecated (removed
    # by Atlassian on Jira Cloud); /rest/api/3/search/jql is the
    # replacement and pages via nextPageToken instead of startAt/total.
    params: dict[str, Any] = {
        "jql": jql,
        "maxResults": max_results,
        "fields": _SEARCH_ISSUE_FIELDS,
    }
    if next_page_token:
        params["nextPageToken"] = next_page_token
    result = _request(
        "GET",
        cloud_id,
        "/rest/api/3/search/jql",
        params=params,
        timeout=timeout,
        allow_retry=allow_retry,
    )
    if not isinstance(result, dict):
        raise RuntimeError("Unexpected response format from Jira search API")
    raw_issues = result.get("issues")
    if raw_issues is None:
        raw_issues = []
    elif not isinstance(raw_issues, list):
        # `or []` alone only replaces a falsy value -- a non-list truthy
        # "issues" (e.g. a proxy/gateway rewriting it to a string or
        # dict) would otherwise iterate silently and produce an empty
        # page with no error, contradicting this function's own "raises
        # on any shape failure" contract.
        raise RuntimeError("Unexpected 'issues' shape in Jira search response")
    issues = (
        [issue for issue in raw_issues if isinstance(issue, dict)]
        if raw_fields
        else [
            _summarize_issue(issue) for issue in raw_issues if isinstance(issue, dict)
        ]
    )
    # The enhanced-search endpoint signals "more pages" by including
    # nextPageToken; isLast is not guaranteed to be present, so the
    # token's presence is the reliable pagination signal in both
    # directions (no token => last page, token => more pages).
    return issues, result.get("nextPageToken")


def _build_search_response(
    issues: list[dict[str, Any]],
    *,
    total_count: int | None = None,
    approximate_total_count: int | None = None,
    next_token: str | None,
) -> str:
    # Field order matters here: the count fields/returned_count/truncated/
    # next_page_token come before the (much larger) issues list so that
    # if this string is ever truncated downstream anyway, the caller
    # still sees the pagination signal, not just a partial issues array
    # with no idea more pages exist.
    #
    # total_count and approximate_total_count are deliberately separate,
    # mutually-exclusive fields rather than one field whose precision
    # varies by call: total_count is only ever set from len(issues) (a
    # final page's exact count), approximate_total_count only ever from
    # Jira's own approximate-count endpoint, which Atlassian's API docs
    # explicitly document as an estimate. A single shared field would
    # let a consumer read a wire value as exact on one call and as an
    # estimate on another with no way to tell which case it's looking
    # at. total_count keeps its original always-present-as-null
    # behavior (existing callers already treat a null there as "not
    # computed"); approximate_total_count is omitted entirely rather
    # than sent as null on every call that never computes it -- this is
    # the common case (most pages don't have both a continuation token
    # AND output-budget headroom for the advisory count call), and this
    # tool's whole purpose is staying under an output-size budget, so
    # a boilerplate null field on every response works against that.
    payload: dict[str, Any] = {"total_count": total_count}
    if approximate_total_count is not None:
        payload["approximate_total_count"] = approximate_total_count
    payload.update(
        returned_count=len(issues),
        truncated=bool(next_token),
        next_page_token=next_token or None,
        issues=issues,
    )
    return _success(**payload)


def _bounded_search_error(message: str, max_output_length: int) -> str:
    """Build an _error(...) response for jira_search_issues that's
    guaranteed to fit max_output_length whenever that's structurally
    possible, trying progressively shorter text -- and preferring a
    candidate that still has a "message" key over one that doesn't,
    since every _error(...) response elsewhere in this file (and a
    generic `if result["status"] == "error": show(result["message"])`
    caller) can otherwise assume that key is always present.

    XAGENT_TOOL_MAX_OUTPUT_LENGTH has no enforced minimum (see
    config.get_tool_max_output_length), so a deployment could configure
    a cap small enough that even this function's own bounded-error
    message would itself get sliced mid-JSON by the downstream
    OutputValueFilter -- exactly the invalid-output failure mode this
    whole fallback path exists to avoid. The last candidate ({"status":
    "error"}, no message key at all) is the smallest valid envelope this
    module can produce; below that there's no budget left to signal
    failure in valid JSON at all, which is a systemic gap in the cap
    itself, not something a single tool's error path can close.
    """
    full = _error(message)
    if len(full) <= max_output_length:
        return full
    # Trim the message text itself (not _truncate's fixed 1000-char
    # cap, which does nothing for a configured budget smaller than
    # that) down to whatever's left after the envelope's own overhead.
    # budget=0 degrades this to _error("") -- a "message" key that's
    # present but empty -- which is as close to preserving that key as
    # any cap under len(_error(message)) can get; below len(_error(""))
    # (34 chars) there is no valid JSON this can produce that both fits
    # and still has a "message" key, so the bare envelope below is the
    # true floor, not a choice this function is making.
    #
    # message is often str(exception) -- e.g. Jira's own JQL-syntax-
    # error text, which can quote the offending clause -- so it isn't
    # guaranteed plain ASCII with no JSON-escapable characters. A `"`,
    # `\`, or control character costs MORE than one output character
    # per source character (up to 6x, for a \u00XX-escaped control
    # char), so a budget-sized slice can still overflow. Shrinking by
    # exactly the *output* overshoot assumes a 1:1 source-to-output
    # ratio that only holds for plain characters -- against an
    # escape-heavy prefix it overshoots the true fitting size and can
    # land on budget<=0 (discarding the message entirely) even when a
    # smaller, still-fitting slice exists. `len(_error(message[:n]))`
    # is monotonically non-decreasing in n (a longer prefix never
    # produces fewer output characters, since JSON escaping only ever
    # adds characters), so binary search over n finds the exact
    # largest fitting slice in O(log budget) steps regardless of how
    # escape-heavy the message is.
    budget = max_output_length - len(_error(""))
    best: str | None = None
    lo, hi = 0, budget
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = _error(message[:mid])
        if len(candidate) <= max_output_length:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    if best is not None:
        return best
    return json.dumps({"status": "error"}, ensure_ascii=False)


@mcp.tool()
def jira_search_issues(
    jql: str,
    cloud_id: str = "",
    limit: int = 50,
    next_page_token: str = "",
    raw_fields: bool = True,
) -> str:
    """
    Search issues with JQL (Jira Query Language) -- the recommended way to
    find issues by project, assignee, status, text, etc. By default each
    result is Jira's own nested object (under "fields", e.g.
    issue["fields"]["status"]["id"]), restricted to a fixed field subset
    (summary, status, assignee, priority, issuetype, project, updated,
    created, resolution, labels, parent) -- not the full issue. Use
    jira_get_issue for a description, dependencies (issue_links/subtasks),
    or any other field not in that list.
    jql: a JQL query, e.g. 'project = ENG AND status = "In Progress"
    ORDER BY updated DESC' or 'text ~ "login bug"'.
    limit: max issues to return per page (default 50, capped at 100).
    next_page_token: pass the previous response's next_page_token to fetch
    the next page -- always check `truncated` and re-call with it instead
    of assuming one page is everything.
    raw_fields: defaults to True (Jira's own nested shape, matching what
    this tool has always returned -- callers that don't pass this
    parameter keep getting the same field layout they always have,
    whichever version they were written against). Pass False to opt into
    a smaller, flattened per-issue projection (key, summary, status,
    status_category, assignee, priority, issue_type, project_key, labels,
    parent_key, resolution, created, updated) recommended for new
    integrations -- roughly 10x smaller per issue, so a page is far less
    likely to need a fallback retry at a smaller size to fit the output
    budget. Either way, only the same fixed field subset above is
    fetched; this changes how those fields are shaped, not which ones --
    use jira_get_issue for a field not in that list either way.
    A search error (invalid JQL, a page too big even at minimal size, a
    stuck pagination cursor) always has a "message" key describing it,
    except under an extremely small configured output cap where even
    that can't fit -- there, the response is the bare
    {"status": "error"} envelope with no message key at all.
    Returns two mutually-exclusive count fields for the total matching
    the JQL (independent of pagination; neither is populated when this
    isn't the first page, the count endpoint failed, or including a
    count would have pushed an otherwise-fitting page over the output
    limit -- these cases all look the same to the caller). Only ever
    computed for the first page of a search (when next_page_token is
    empty) -- it's a per-JQL constant, not tied to any one page, so
    carry it forward yourself for later pages of the same search
    instead of expecting it again.
    - total_count: the EXACT total, always present (null when not
      computed) -- only set when this first page is also the last (no
      more results beyond it), since len(issues) is already exact in
      that case with no extra call needed.
    - approximate_total_count: an ESTIMATE from Jira's own
      approximate-count endpoint, present ONLY when there ARE more
      results beyond this first page and it was actually computed
      (Atlassian's API explicitly documents this endpoint's result as
      an estimate, not an exact count) -- absent, not null, otherwise.
    Compare either one against returned_count to know whether more
    issues exist beyond what truncated/next_page_token alone would
    tell you.
    """
    try:
        resolved_cloud_id = _resolve_cloud_id(cloud_id)
        max_results = clamp_limit(limit, max_limit=MAX_LIMIT)
        max_output_length = get_tool_max_output_length()
    except Exception as e:
        # None of this setup touches jql, so the full exception detail
        # is safe to log here -- unlike the try block below, which does
        # touch jql and routes through _log_metadata_only instead. Still
        # redacted for credential-shaped content (_safe_text), since a
        # connection-level error from _resolve_cloud_id's own request
        # can embed request/header details this module doesn't control.
        safe_message = _safe_text(e)
        logger.error(f"Error searching Jira issues: {safe_message}")
        return _bounded_search_error(safe_message, get_tool_max_output_length())

    try:
        issues: list[dict[str, Any]] = []
        next_token: str | None = None
        response = ""
        fit = False
        for attempt, size in enumerate(_page_size_candidates(max_results)):
            # Every attempt re-issues the search FROM THE SAME
            # next_page_token the caller passed in (never a
            # previous attempt's own nextPageToken): Jira's
            # nextPageToken is positional over the whole maxResults it
            # was asked for, so slicing an already-fetched oversized page
            # locally while keeping its token, or chaining a smaller
            # retry off of it, would silently skip rows on the caller's
            # next call. Re-fetching at a smaller size from the original
            # starting point is what makes the returned token correctly
            # correspond to exactly what's returned here.
            is_first_attempt = attempt == 0
            issues, next_token = _fetch_and_summarize_page(
                resolved_cloud_id,
                jql,
                size,
                next_page_token,
                timeout=(
                    DEFAULT_TIMEOUT_SECONDS
                    if is_first_attempt
                    else _FALLBACK_ATTEMPT_TIMEOUT_SECONDS
                ),
                allow_retry=is_first_attempt,
                raw_fields=raw_fields,
            )
            response = _build_search_response(issues, next_token=next_token)
            if len(response) <= max_output_length:
                fit = True
                break

        if not fit:
            # Even a minimal page doesn't fit -- a single issue's summary
            # and/or labels alone exceed the budget. Fail safely rather
            # than return invalid (cut-mid-JSON) output or a token that
            # would skip rows.
            logger.warning(
                "jira_search_issues: minimal page still exceeds the output limit"
            )
            return _bounded_search_error(
                "A Jira search result page exceeds the tool output limit "
                "even at a minimal page size; narrow the JQL query",
                max_output_length,
            )

        if next_token and next_token == next_page_token:
            # A cursor that didn't advance would make a caller
            # mechanically following truncated/next_page_token loop on
            # the same page forever -- fail instead of handing back a
            # token that goes nowhere (mirrors shopify.py's
            # _success_paginated cursor-safety check).
            logger.warning("jira_search_issues: pagination cursor did not advance")
            return _bounded_search_error(
                "Jira returned a pagination cursor that did not advance; "
                "pagination stopped to prevent an infinite loop",
                max_output_length,
            )

        # The count is a per-JQL constant, independent of pagination --
        # only fetch it once, for the caller's first page, and only when
        # there's actually more beyond this page: an exact count is
        # already known for free (len(issues)) when this page is both
        # the first and the last, so calling the advisory endpoint there
        # would just be a redundant network round trip for a number
        # that's already exact. Never fetched on either early return
        # above (size-overflow or stuck-cursor) either, for the same
        # "don't do wasted work" reason. exact_total_count and
        # approximate_total_count are mutually exclusive: at most one
        # of them is ever set below (see _build_search_response for why
        # they're kept as two separate fields instead of one whose
        # precision silently varies by call).
        exact_total_count: int | None = None
        approximate_total_count: int | None = None
        if next_page_token:
            pass
        elif not next_token:
            exact_total_count = len(issues)
        elif len(response) + _TOTAL_COUNT_RESERVE_CHARS > max_output_length:
            # `response` is the already-fitting page the loop above just
            # built -- if there's not even enough headroom left for a
            # plausible count value, skip the network round trip (and
            # the count endpoint's own rate-limit budget) for a result
            # that's about to be discarded as soon as it comes back
            # anyway.
            pass
        else:
            approximate_total_count = _approximate_count(resolved_cloud_id, jql)

        if exact_total_count is not None or approximate_total_count is not None:
            with_count = _build_search_response(
                issues,
                total_count=exact_total_count,
                approximate_total_count=approximate_total_count,
                next_token=next_token,
            )
            if len(with_count) <= max_output_length:
                response = with_count
            else:
                # Both counts are additive/advisory; if adding either is
                # ever what tips an already-fitting page over budget,
                # drop it rather than fail a search that otherwise fit
                # -- reuse `response` (the loop's already-fitting,
                # already-built page) instead of paying for another
                # full serialization of `issues` just to reproduce the
                # same page again.
                exact_total_count = None
                approximate_total_count = None

        logger.info(
            f"jira_search_issues: returned={len(issues)} "
            f"exact_total={exact_total_count} "
            f"approximate_total={approximate_total_count} "
            f"has_next_page={bool(next_token)}"
        )
        return response
    except Exception as e:
        _log_metadata_only(logger.error, "Error searching Jira issues", e)
        return _bounded_search_error(_safe_text(e), max_output_length)


#: Cap on ADF recursion depth, mirroring the same trace-depth guard used
#: in core.agent.trace._MAX_TRACE_DEPTH. Without this, a pathologically
#: nested document (e.g. hundreds of levels of nested lists/blockquotes,
#: from a pasted document or a buggy upstream integration) would drive
#: Python's call stack past its default recursion limit and raise
#: RecursionError instead of degrading gracefully.
_ADF_MAX_DEPTH = 50
_ADF_NESTED_TOO_DEEP_MESSAGE = "[... nested too deep ...]"

#: Proactive per-field caps for the two known hot-spot text fields that
#: can otherwise make jira_get_issue/jira_list_comments' whole
#: serialized response exceed the output budget on their own: a Jira
#: description (single object, so there is no "smaller page" fallback
#: the way a list of issues/comments has) and a single comment body.
#: Both are well under a typical 50 KiB budget on their own, leaving
#: room for the rest of the response, and both are capped BEFORE the
#: response is ever measured, so the caller gets an explicit
#: truncation signal and predictable partial content instead of
#: whatever OutputValueFilter happens to slice off mid-JSON.
_ISSUE_DESCRIPTION_MAX_CHARS = 30_000
_COMMENT_BODY_MAX_CHARS = 4_000


def _cap_text_field(payload: dict[str, Any], field: str, max_chars: int) -> bool:
    """Truncate payload[field] in place if it's a str longer than
    max_chars, marking the cut. Returns whether it was truncated.
    """
    value = payload.get(field)
    if isinstance(value, str) and len(value) > max_chars:
        payload[field] = truncate_error_text(value, max_chars)
        return True
    return False


def _flatten_adf(value: Any) -> str | None:
    """Render a Jira ADF (Atlassian Document Format) rich-text field --
    used by an issue's description and a comment's body -- down to
    plain text.

    Falls back to returning a plain string unchanged, so this is safe
    to call unconditionally on either shape. That fallback isn't just a
    Server/Data Center accommodation: per Atlassian's own migration
    docs, ADF is a v3-API-only representation -- REST API v2 (which
    every request in this file uses; see _issue_path) returns
    description/comment body as a plain wiki-markup STRING even on
    Jira Cloud. The ADF-object branches below exist for forward
    compatibility (a future move to v3, or a Jira change to v2's
    default) rather than because v2 is expected to send one today; the
    plain-string fallback is what today's requests actually exercise.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if not isinstance(value, dict):
        return None

    # Appending to a shared list and joining once at the end is O(N) in
    # the document's total text length; returning and re-joining a
    # string at every ancestor level (the previous approach) is O(N*D)
    # for nesting depth D, since each character gets copied once per
    # level on the way back up the tree.
    out: list[str] = []

    def _walk(node: Any, depth: int) -> None:
        if not isinstance(node, dict):
            return
        if depth > _ADF_MAX_DEPTH:
            out.append(_ADF_NESTED_TOO_DEEP_MESSAGE)
            return
        node_type = node.get("type")
        attrs = _as_dict(node.get("attrs"))
        if node_type == "text":
            out.append(str(node.get("text") or ""))
            return
        if node_type == "hardBreak":
            out.append("\n")
            return
        # mention/emoji/inlineCard are leaf inline nodes -- no "text" and
        # no "content" to recurse into -- so without explicit handling
        # they'd silently render as "", dropping the @-mention, emoji,
        # or link entirely.
        if node_type == "mention":
            # `text` (a "@Display Name" label) is documented as optional
            # on a mention node -- only `id` is required -- so falling
            # back to the raw account id still surfaces *something*
            # instead of the mention vanishing when text is omitted.
            mention_id = attrs.get("id")
            out.append(
                str(
                    attrs.get("text")
                    or (f"@{mention_id}" if mention_id is not None else "")
                )
            )
            return
        if node_type == "emoji":
            out.append(str(attrs.get("text") or attrs.get("shortName") or ""))
            return
        if node_type == "inlineCard":
            # Atlassian's docs: an inlineCard's attrs carries EITHER
            # `data` (a JSON-LD resource) OR `url`, never both -- so a
            # `data`-only smart link still needs a fallback or it
            # silently renders as "" exactly like the bug this function
            # exists to fix.
            url = attrs.get("url")
            if not url:
                data = _as_dict(attrs.get("data"))
                url = data.get("url") or data.get("name") or data.get("title")
            out.append(str(url or ""))
            return
        for child in node.get("content") or []:
            _walk(child, depth + 1)
        # listItem/blockquote wrap block children (typically a paragraph)
        # that already append their own trailing "\n" -- appending a
        # second one here would double it into a blank line per list
        # item / at the end of every blockquote.
        if node_type in ("paragraph", "heading", "codeBlock"):
            out.append("\n")

    for block in value.get("content") or []:
        _walk(block, 1)

    text = "".join(out)
    return text.strip() or None


def _summarize_mini_issue(entry: dict[str, Any]) -> dict[str, Any]:
    """Extract key/summary/status from a Jira "mini issue" shape --
    {key, fields: {summary, status: {name}}} -- the same shape Jira uses
    both for a subtasks entry and for issuelinks' linked-issue object.
    Called directly for subtasks and via _summarize_issue_link for
    issuelinks, so the two don't drift if a field is added to one and
    not the other.
    """
    fields = _as_dict(entry.get("fields"))
    return {
        "issue_key": entry.get("key"),
        "summary": fields.get("summary"),
        "status": _as_dict(fields.get("status")).get("name"),
    }


def _summarize_issue_link(link: dict[str, Any]) -> dict[str, Any]:
    """Slim one issuelinks entry to the relationship plus the linked
    issue's key/summary/status.

    issuelinks/subtasks are only ever present on jira_get_issue's raw
    payload -- jira_search_issues has no such field to request -- so this
    (and jira_get_issue below) is the only place a caller can learn an
    issue's dependencies at all.
    """
    # Presence, not truthiness: Jira can send a present-but-redacted
    # "outwardIssue": {} for a permission-restricted linked issue, which
    # must still be reported as an outward link rather than silently
    # falling through to inwardIssue.
    is_outward = "outwardIssue" in link
    linked = link.get("outwardIssue") if is_outward else link.get("inwardIssue")
    link_type = _as_dict(link.get("type"))
    return {
        "relationship": link_type.get("outward" if is_outward else "inward"),
        **_summarize_mini_issue(_as_dict(linked)),
    }


def _summarize_person(person: Any) -> dict[str, Any] | None:
    """Presence, not truthiness: Jira can send a present-but-redacted
    "{}" for a permission-restricted assignee/reporter/creator/author,
    which must still be reported as "someone, details unavailable"
    rather than silently collapsed into the same None used for a
    genuinely unassigned field.
    """
    if person is None:
        return None
    person = _as_dict(person)
    return {
        "account_id": person.get("accountId"),
        "display_name": person.get("displayName"),
    }


def _summarize_full_issue(issue: dict[str, Any]) -> dict[str, Any]:
    """Slim a full raw issue (jira_get_issue's payload) down to what a
    detail/dependency view needs, flattening ADF rich text and dropping
    the url/avatar/icon/changelog clutter that let a single issue run up
    to ~180 KB and get cut mid-JSON by the output filter.
    """
    fields = _as_dict(issue.get("fields"))
    status = _as_dict(fields.get("status"))
    priority = _as_dict(fields.get("priority"))
    issuetype = _as_dict(fields.get("issuetype"))
    project = _as_dict(fields.get("project"))
    parent = _as_dict(fields.get("parent"))
    resolution = _as_dict(fields.get("resolution"))
    return {
        "id": issue.get("id"),
        "key": issue.get("key"),
        "summary": fields.get("summary"),
        "description": _flatten_adf(fields.get("description")),
        "status": status.get("name"),
        "status_category": _as_dict(status.get("statusCategory")).get("name"),
        "assignee": _summarize_person(fields.get("assignee")),
        "reporter": _summarize_person(fields.get("reporter")),
        "creator": _summarize_person(fields.get("creator")),
        "priority": priority.get("name"),
        "issue_type": issuetype.get("name"),
        "project_key": project.get("key"),
        "labels": fields.get("labels") or [],
        "parent_key": parent.get("key"),
        "resolution": resolution.get("name"),
        "components": [
            c.get("name") for c in fields.get("components") or [] if isinstance(c, dict)
        ],
        "fix_versions": [
            v.get("name")
            for v in fields.get("fixVersions") or []
            if isinstance(v, dict)
        ],
        "due_date": fields.get("duedate"),
        "created": fields.get("created"),
        "updated": fields.get("updated"),
        "issue_links": [
            _summarize_issue_link(link)
            for link in fields.get("issuelinks") or []
            if isinstance(link, dict)
        ],
        "subtasks": [
            _summarize_mini_issue(subtask)
            for subtask in fields.get("subtasks") or []
            if isinstance(subtask, dict)
        ],
    }


#: Fields requested for jira_get_issue. Keep this in lockstep with every
#: `fields.get(...)` call inside _summarize_full_issue below: Jira's API
#: returns ONLY this subset once any `fields` param is passed, so a key
#: _summarize_full_issue reads but this list omits will silently and
#: permanently come back as None. test_get_issue_requests_exact_field_list
#: in test_jira_mcp.py pins this constant's value so a change here shows
#: up in review, but it does not by itself verify the two stay matched --
#: check _summarize_full_issue's fields.get(...) calls by hand when
#: editing either.
_GET_ISSUE_FIELDS = (
    "summary,description,status,assignee,reporter,creator,"
    "priority,issuetype,project,parent,resolution,labels,"
    "components,fixVersions,duedate,created,updated,"
    "issuelinks,subtasks"
)

#: Parsed once for jira_get_issue's extra_fields dedup: a caller-named
#: field already in the default set would otherwise be fetched, ADF-
#: flattened, and capped a second time under extra_field_values, wasting
#: both work and output budget on data already visible in `issue`.
_GET_ISSUE_FIELD_NAMES = frozenset(_GET_ISSUE_FIELDS.split(","))


def _cap_extra_field_value(value: Any, max_chars: int) -> tuple[Any, bool]:
    """Bound one extra_fields raw value the same way description/comment
    bodies are bounded, regardless of whether Jira returns it as a plain
    string, an ADF rich-text dict (a custom "Rich Text" field uses the
    same ADF shape description does), or some other JSON-shaped value
    (a multi-value picker's list, for instance). Without this, only the
    str case was capped -- a dict/list value bypassed the cap entirely,
    and, being the sole key of extra_field_values, could also make that
    whole field collapse to {} in one step if success_with_capped_dict's
    halving fallback ever had to shrink it (halving a single-key dict's
    keys drops to nothing, unlike halving a list). Returns (possibly-
    capped value, whether it was truncated).
    """
    if isinstance(value, dict) and value.get("type") == "doc":
        value = _flatten_adf(value)
    if isinstance(value, str):
        if len(value) > max_chars:
            return truncate_error_text(value, max_chars), True
        return value, False
    if value is None:
        return None, False
    serialized = json.dumps(value, ensure_ascii=False)
    if len(serialized) > max_chars:
        return truncate_error_text(serialized, max_chars), True
    return value, False


@mcp.tool()
def jira_get_issue(
    issue_key: str,
    cloud_id: str = "",
    extra_fields: str = "",
    raw_fields: bool = False,
) -> str:
    """
    Get one issue's details -- description, status, assignee,
    reporter/creator, priority, components, fix versions, due date, and
    its dependencies (issue_links -- e.g. blocks/is blocked by/relates to
    -- and subtasks). Only the fields listed here are fetched by default;
    other Jira fields (attachments, worklog, votes, watchers, time
    tracking, affected versions, security level, sprint/epic link,
    custom fields) are not requested and will not appear in the result
    unless named via extra_fields.
    issue_key: an issue key (e.g. "ENG-123") or its numeric id.
    extra_fields: optional comma-separated Jira field ids not in the
    default set above -- standard (e.g. "votes") or custom (e.g.
    "customfield_10010", found via your Jira admin's field
    configuration); a name already in the default set is ignored, since
    it's already visible under its own summarized key. Returned as
    {field_id: raw_value} under a top-level extra_field_values (a
    sibling of "issue", not nested inside it), so you can reach any
    field this tool doesn't summarize by name. An ADF-shaped (rich
    text) value is flattened to plain text first, matching description;
    a value (of any type) still too large after that is turned into a
    truncated JSON-text string instead -- so a list/dict-valued custom
    field can come back as a string once it's cut, not its original
    shape. extra_field_values_truncated is set to true if any entry was
    cut, including one entirely omitted because too many fields were
    requested to fit even a fair share of the output budget.
    Use this, not jira_search_issues, when you need an issue's
    dependencies or full description: issue_links/subtasks/description
    are only returned here. A description longer than
    _ISSUE_DESCRIPTION_MAX_CHARS is truncated with description_truncated
    set to true, so the result is always parseable JSON regardless of
    how large the real description is.
    raw_fields: return each fetched field (the default set, plus any
    extra_fields) as Jira's own nested object under "fields" (e.g.
    issue["fields"]["status"]["id"]) instead of the flattened summary
    above -- for an existing integration written against the raw shape
    this tool returned before summarization was added, or to reach a
    nested sub-field (e.g. status.id, not just status.name) the summary
    doesn't expose. extra_fields still controls which fields beyond the
    default set are fetched; extra_field_values/description_truncated
    are not produced in this mode since the raw "fields" object already
    contains everything requested. Bigger per-issue payload than the
    default projection, so a raw response can shrink further under a
    small output budget (see success_with_capped_dict) than the same
    request would at the default setting.
    """
    try:
        # Jira's `fields` query param treats a leading "-" as "exclude
        # this field" and "*" as a wildcard selector (e.g. "*all"), not
        # just a literal field id -- passed through unfiltered, either
        # could override the tool's own guaranteed default field set
        # (e.g. extra_fields="-description" would suppress the very
        # field this tool's docstring promises is always returned).
        # Rejected the same way an already-default name is: silently
        # dropped rather than erroring, since a caller only ever loses
        # an invalid/malicious token, never a legitimate field id.
        requested_extra = [
            stripped
            for name in extra_fields.split(",")
            if (stripped := name.strip())
            and stripped not in _GET_ISSUE_FIELD_NAMES
            and not stripped.startswith(("-", "*"))
        ]
        fields_param = _GET_ISSUE_FIELDS
        if requested_extra:
            fields_param = f"{fields_param},{','.join(requested_extra)}"
        result = _request(
            "GET",
            cloud_id,
            _issue_path(issue_key),
            params={"fields": fields_param},
        )
        if not isinstance(result, dict):
            return _error("Unexpected response format from Jira issue API")
        if raw_fields:
            # The raw "fields" object already contains every field this
            # request fetched (default set plus any requested_extra), so
            # extra_field_values/description_truncated -- which only
            # exist to work around the summary's flattening -- don't
            # apply here. success_with_capped_dict still bounds the
            # whole thing the same way the summarized path is bounded,
            # just without a protected top-level extra to carve out.
            return success_with_capped_dict("issue", result)
        issue = _summarize_full_issue(result)
        description_truncated = _cap_text_field(
            issue, "description", _ISSUE_DESCRIPTION_MAX_CHARS
        )
        # Fixed top-level fields alongside "issue", not nested inside
        # it: success_with_capped_dict's generic shrink pass only
        # touches `issue` itself, so a flag (or extra_field_values
        # itself) nested inside it could be dropped along with the data
        # it describes by the very shrink pass it exists to signal --
        # exactly when the signal matters most. This also means nothing
        # else ever shrinks extra_field_values gradually, so it needs
        # its own aggregate cap below: each value is capped
        # individually by _cap_extra_field_value, but that alone doesn't
        # bound the total -- two ordinary custom fields near that
        # per-field cap can already exceed a default-sized output
        # budget on their own, before `issue` is even considered.
        top_level_extra: dict[str, Any] = {
            "description_truncated": description_truncated
        }
        if requested_extra:
            response_fields = _as_dict(result.get("fields"))
            extra_field_values: dict[str, Any] = {}
            extra_field_values_truncated = False
            # Half the configured budget, reserved for the combined
            # extra_field_values dict -- this leaves the other half for
            # `issue` and the rest of the envelope. This must scale
            # DOWN with a small configured max_output_length, not just
            # up with a large one: an earlier version floored this at
            # _ISSUE_DESCRIPTION_MAX_CHARS (30_000) "so a single
            # requested field still gets a reasonable amount of room",
            # but that floor made the aggregate budget bigger than the
            # ENTIRE configured output cap whenever max_output_length
            # was under 60_000 (the current default, 50 KiB, included) --
            # success_with_capped_dict's `issue` shrinking can't touch
            # extra_field_values at all (it's a protected top-level
            # extra), so an oversized aggregate here forced the
            # function's last-resort with_extras=False fallback, which
            # drops extra_field_values -- and description_truncated
            # alongside it -- entirely, reporting a plain "success" with
            # no sign any of it was ever there. Each individual field is
            # still separately capped at _ISSUE_DESCRIPTION_MAX_CHARS by
            # the min(...) below, and remaining_budget<=0 below already
            # degrades gracefully to an empty, explicitly-flagged
            # extra_field_values instead of silently vanishing -- that
            # graceful path only works if this budget is never inflated
            # past what success_with_capped_dict can actually deliver.
            remaining_budget = get_tool_max_output_length() // 2
            for name in requested_extra:
                if remaining_budget <= 0:
                    extra_field_values_truncated = True
                    break
                value, was_truncated = _cap_extra_field_value(
                    response_fields.get(name),
                    min(_ISSUE_DESCRIPTION_MAX_CHARS, remaining_budget),
                )
                extra_field_values[name] = value
                extra_field_values_truncated = (
                    extra_field_values_truncated or was_truncated
                )
                remaining_budget -= len(json.dumps(value, ensure_ascii=False))
            top_level_extra["extra_field_values"] = extra_field_values
            top_level_extra["extra_field_values_truncated"] = (
                extra_field_values_truncated
            )
        # success_with_capped_dict already tries the full response first
        # and only shrinks `issue` if that doesn't fit -- no need to
        # separately build and measure a plain response here first.
        return success_with_capped_dict("issue", issue, extra_fields=top_level_extra)
    except Exception as e:
        safe_message = _safe_text(e)
        logger.error(f"Error fetching Jira issue {issue_key}: {safe_message}")
        return _error(safe_message)


@mcp.tool()
def jira_create_issue(
    project_key: str,
    summary: str,
    cloud_id: str = "",
    description: str = "",
    issue_type: str = "Task",
    assignee_account_id: str = "",
    priority: str = "",
) -> str:
    """
    Create a new issue.
    project_key: the target project's key (e.g. "ENG"), from jira_list_projects.
    summary: the issue title.
    description: optional body (plain text).
    issue_type: the issue type's name (e.g. "Task", "Bug", "Story") --
    must be one of the target project's configured issue types.
    assignee_account_id: optional account id from jira_search_users.
    priority: optional priority name (e.g. "High", "Medium", "Low") -- must
    be one of the site's configured priorities.
    """
    try:
        if err := _clean_text_error(summary, "summary"):
            return err
        fields: dict[str, Any] = {
            "project": {"key": project_key},
            "summary": summary,
            "issuetype": {"name": issue_type},
        }
        if description:
            fields["description"] = description
        if assignee_account_id:
            fields["assignee"] = {"accountId": assignee_account_id}
        if priority:
            fields["priority"] = {"name": priority}

        result = _request(
            "POST", cloud_id, "/rest/api/2/issue", json_data={"fields": fields}
        )
        return _success(issue=result)
    except Exception as e:
        safe_message = _safe_text(e)
        logger.error(
            f"Error creating Jira issue in project {project_key}: {safe_message}"
        )
        return _error(safe_message)


@mcp.tool()
def jira_update_issue(
    issue_key: str,
    cloud_id: str = "",
    summary: str | None = None,
    description: str | None = None,
    assignee_account_id: str | None = None,
    priority: str | None = None,
) -> str:
    """
    Update an existing issue. Only the fields explicitly provided are
    changed; leave a parameter unset (None) to leave that field untouched.
    issue_key: an issue key (e.g. "ENG-123") or its numeric id.
    assignee_account_id: an account id from jira_search_users, to reassign
    the issue -- pass an empty string to unassign it.
    priority: a priority name (e.g. "High") -- must be one of the site's
    configured priorities.
    """
    try:
        fields: dict[str, Any] = {}
        if summary is not None:
            if err := _clean_text_error(summary, "summary"):
                return err
            fields["summary"] = summary
        if description is not None:
            fields["description"] = description
        if assignee_account_id is not None:
            fields["assignee"] = (
                {"accountId": assignee_account_id} if assignee_account_id else None
            )
        if priority is not None:
            if err := _clean_text_error(
                priority,
                "priority",
                hint=(
                    " -- Jira also has no way to clear priority through this "
                    "field; omit the parameter instead"
                ),
            ):
                return err
            fields["priority"] = {"name": priority}
        if not fields:
            return _error("No fields provided to update")

        _request(
            "PUT",
            cloud_id,
            _issue_path(issue_key),
            json_data={"fields": fields},
        )
        return _success(issue_key=issue_key)
    except Exception as e:
        safe_message = _safe_text(e)
        logger.error(f"Error updating Jira issue {issue_key}: {safe_message}")
        return _error(safe_message)


@mcp.tool()
def jira_list_transitions(issue_key: str, cloud_id: str = "") -> str:
    """
    List an issue's available workflow transitions (e.g. "Start Progress",
    "Done") -- id and name. Resolve a transition name here before passing
    it to jira_transition_issue, or pass the name directly since
    jira_transition_issue resolves it internally too.
    """
    try:
        result = _request(
            "GET",
            cloud_id,
            _issue_path(issue_key, "/transitions"),
        )
        if not isinstance(result, dict):
            return _error("Unexpected response format from Jira transitions API")
        transitions = [
            {"id": t.get("id"), "name": t.get("name")}
            for t in result.get("transitions") or []
            if isinstance(t, dict)
        ]
        return _success(transitions=transitions)
    except Exception as e:
        safe_message = _safe_text(e)
        logger.error(
            f"Error listing transitions for Jira issue {issue_key}: {safe_message}"
        )
        return _error(safe_message)


@mcp.tool()
def jira_transition_issue(
    issue_key: str, transition_name: str, cloud_id: str = ""
) -> str:
    """
    Move an issue through its workflow (e.g. to "Done", "In Progress").
    transition_name: a transition's name, case-insensitive (see
    jira_list_transitions for the exact set available on this issue --
    available transitions depend on the issue's current status).
    """
    try:
        resolved_cloud_id = _resolve_cloud_id(cloud_id)
        result = _request(
            "GET",
            resolved_cloud_id,
            _issue_path(issue_key, "/transitions"),
        )
        if not isinstance(result, dict):
            return _error("Unexpected response format from Jira transitions API")
        needle = transition_name.strip().lower()
        match = next(
            (
                t
                for t in result.get("transitions") or []
                if isinstance(t, dict) and str(t.get("name") or "").lower() == needle
            ),
            None,
        )
        if not match:
            available = ", ".join(
                str(t.get("name"))
                for t in result.get("transitions") or []
                if isinstance(t, dict)
            )
            return _error(
                f"Transition '{transition_name}' is not available for {issue_key}. "
                f"Available transitions: {available}"
            )
        transition_id = match.get("id")
        # Presence, not truthiness: an id is opaque and never contractually
        # excludes a falsy-but-valid value like 0 or "0".
        if transition_id is None:
            return _error(
                f"Matched transition '{transition_name}' is missing a valid 'id'"
            )

        _request(
            "POST",
            resolved_cloud_id,
            _issue_path(issue_key, "/transitions"),
            json_data={"transition": {"id": transition_id}},
        )
        return _success(issue_key=issue_key, transitioned_to=match.get("name"))
    except Exception as e:
        safe_message = _safe_text(e)
        logger.error(
            f"Error transitioning Jira issue {issue_key} to '{transition_name}': "
            f"{safe_message}"
        )
        return _error(safe_message)


def _summarize_comment(
    comment: dict[str, Any], *, body_max_chars: int = _COMMENT_BODY_MAX_CHARS
) -> dict[str, Any]:
    """Flatten a comment's ADF body to plain text and drop the author's
    avatarUrls -- a long comment thread otherwise repeats that avatar
    ladder once per comment and is a routine way jira_list_comments'
    output ran past the per-tool-output-string cap.

    body_max_chars is only ever overridden by _fit_comments_page's
    single-comment shrink search (a smaller cap than the fixed default
    for the one comment that doesn't fit any other way); every other
    caller gets the fixed default.
    """
    raw_visibility = comment.get("visibility")
    visibility = _as_dict(raw_visibility)
    jsd_public = comment.get("jsdPublic")
    summarized = {
        "id": comment.get("id"),
        "author": _summarize_person(comment.get("author")),
        "body": _flatten_adf(comment.get("body")),
        # A comment can be restricted to a role/group; surfacing that
        # (rather than dropping it) matters because a caller relaying
        # comments elsewhere (e.g. into a customer-facing channel) needs
        # to know a comment was never meant to be public. Checked by
        # presence, not truthiness: a present-but-empty {} visibility
        # object must still be reported as restricted, not silently
        # treated the same as "no visibility field at all".
        "visibility": (
            {"type": visibility.get("type"), "value": visibility.get("value")}
            if raw_visibility is not None
            else None
        ),
        # jsdPublic is a SEPARATE mechanism from visibility, injected by
        # Jira Service Management on a service-desk request's comments:
        # a JSM agent can mark a comment internal-only (jsdPublic=false)
        # with no role/group visibility restriction set at all, so
        # visibility alone can silently report a JSM-internal comment as
        # public. None (not False) when the issue isn't a JSM request,
        # since this key is simply absent there -- not itself a public/
        # internal signal.
        "jsd_public": jsd_public if isinstance(jsd_public, bool) else None,
        "created": comment.get("created"),
        "updated": comment.get("updated"),
    }
    # Capping ONE comment's body can't by itself make an oversized PAGE
    # fit (see _fit_comments_page for that), but it does guarantee a
    # single pathologically large comment is never why a page can't be
    # built at all.
    summarized["body_truncated"] = _cap_text_field(summarized, "body", body_max_chars)
    return summarized


def _build_comments_response(
    comments: list[dict[str, Any]], total: int, next_start_at: int | None
) -> str:
    return _success(
        total=total,
        returned_count=len(comments),
        truncated=next_start_at is not None,
        next_start_at=next_start_at,
        comments=comments,
    )


def _fit_comments_page(
    raw_comments: list[Any],
    offset: int,
    total: int,
    has_more_raw: bool,
    max_output_length: int,
    *,
    raw_fields: bool = False,
) -> str | None:
    """Build a whole-comment prefix (each comment already body-capped by
    _summarize_comment, unless raw_fields is set) that fits
    max_output_length, trying every raw entry first and halving the
    count until it fits if that overflows. Halving can under-deliver
    relative to the true largest fitting prefix (e.g. 8 kept comments
    where the full page and half (4) both overflow but 6 would fit
    tries 8/4/2/1/0, never 6) -- accepted for consistency with the same
    halving strategy every sibling connector uses for this "shrink a
    list until the response fits" problem, rather than this being the
    one bespoke exact-fit search in the file.

    When even a single SUMMARIZED comment doesn't fit at its default
    per-comment body cap, this retries that SAME comment with a
    progressively smaller body cap (binary search) rather than skipping
    it outright -- shrinking is far more likely to make one comment fit
    than the fixed default assumed, and preserves its content instead
    of losing it. raw_fields skips this refinement (a raw comment's
    body is Jira's own ADF object, not the plain string _cap_text_field
    shrinks) and falls straight to the empty-page fallback below when a
    single raw comment doesn't fit.

    If not even a body-length-zero single comment fits (or raw_fields,
    which never attempts that retry), this falls back to a genuinely
    empty page (0 comments, next_start_at skipping past the one that
    couldn't fit) -- an empty envelope is smaller than even a
    body-length-zero single comment's id/author/visibility/jsd_public/
    timestamps overhead, so there's a real budget range where this
    still succeeds. Returns None only if not even that empty page fits
    -- the caller should treat that as a bounded error, the same way
    jira_search_issues does when its minimal page still doesn't fit.

    next_start_at is computed from the RAW position of (or immediately
    after) the last comment actually kept -- not from the filtered/kept
    count -- so a caller resuming from it neither skips nor re-fetches a
    comment already accounted for here (same reasoning
    jira_list_projects/jira_search_users use for their own
    next_start_at, extended to also cover a size-triggered cut, not
    just a filtered-out malformed entry).
    """
    # (raw index, summarized-or-raw comment) pairs -- kept together
    # instead of two parallel lists so the two can't drift out of
    # alignment under a future edit to the filter below.
    kept: list[tuple[int, dict[str, Any]]] = [
        (index, raw if raw_fields else _summarize_comment(raw))
        for index, raw in enumerate(raw_comments)
        if isinstance(raw, dict)
    ]

    def next_start_at_for(count: int) -> int | None:
        if count == len(kept):
            # Every raw comment on this page is accounted for -- the
            # existing "is there more beyond this raw page" signal is
            # exactly right, size aside.
            return (offset + len(raw_comments)) if has_more_raw else None
        # One past the last comment actually kept -- or, when count is 0
        # (nothing kept, the one comment this page has doesn't fit even
        # alone), one past that comment instead. Using max(count-1, 0)
        # rather than a separate count==0 branch means there's only one
        # formula to verify, not two that must independently agree: the
        # negative-index wraparound `kept[count-1]` at count=0 would
        # otherwise silently read kept[-1] (the LAST kept comment) if a
        # future edit merged the branches without noticing the trap.
        # Skipping past the unfittable entry (not resuming AT it) matters
        # because resuming at it would equal the caller's own start_at
        # whenever that comment is first on its raw page -- a caller
        # mechanically following next_start_at would re-fetch this exact
        # position forever, the same "must never equal start_at"
        # invariant jira_list_comments already checks for its own
        # raw-page case. A page this tool can never actually deliver
        # isn't recoverable by retrying, so forward progress matters more
        # than not dropping an unfittable entry.
        return offset + kept[max(count - 1, 0)][0] + 1

    def response_for(count: int) -> str:
        comments = [comment for _, comment in kept[:count]]
        return _build_comments_response(comments, total, next_start_at_for(count))

    # Try every kept comment first, then halve the count until the
    # response fits -- the same halving strategy utils.py's
    # success_with_capped_dict/_halve_largest_list_fields_until_bounded
    # use for the same "shrink a list field until the response fits"
    # problem, rather than a bespoke bisection: MAX_LIMIT bounds a page
    # to 100 comments, so the extra precision an exact-largest-fit search
    # would buy isn't worth this being the one mechanism in the file that
    # doesn't match its siblings. Stops at count=1 (not 0): a lone
    # comment that still doesn't fit at its default body cap gets a
    # dedicated shrink search below instead of being dropped straight
    # to an empty page.
    count = len(kept)
    response = response_for(count)
    while len(response) > max_output_length and count > 1:
        count //= 2
        response = response_for(count)
    if len(response) <= max_output_length:
        return response

    if kept and not raw_fields:
        # A single comment doesn't fit even at _COMMENT_BODY_MAX_CHARS.
        # Retry the SAME comment (not a different one -- next_start_at_for
        # must stay anchored to this raw index either way) with a smaller
        # body cap. Binary search is safe here even though response length
        # isn't STRICTLY monotonic in the cap: truncate_error_text's slice
        # is a prefix of the original body, so length increases with the
        # cap right up to cap == len(body) -- where the "... [truncated]"
        # marker disappears entirely, briefly making the response SHORTER
        # than at cap == len(body) - 1. That one dip can't make the search
        # return an over-budget "best": every candidate is fit-checked
        # directly before being recorded, and response length is constant
        # (not decreasing) for every cap >= len(body), so the search's
        # right-biased walk (lo = mid + 1 on a fit) still reaches the
        # untruncated body whenever it fits, rather than getting stuck on
        # a smaller, needlessly-truncated candidate.
        #
        # Flattening the ADF body is the expensive part of
        # _summarize_comment (a full tree walk), and is identical across
        # every candidate cap tried below -- computed once here rather
        # than inside response_at_cap, which would otherwise re-walk the
        # same ADF tree on every one of the search's O(log
        # _COMMENT_BODY_MAX_CHARS) iterations. `template` supplies the
        # other, cap-independent fields (author/visibility/jsd_public/
        # timestamps) without recomputing them either.
        raw_comment = raw_comments[kept[0][0]]
        flattened_body = _flatten_adf(raw_comment.get("body"))
        template = kept[0][1]

        def response_at_cap(body_max_chars: int) -> str:
            comment = dict(template)
            comment["body"] = flattened_body
            comment["body_truncated"] = _cap_text_field(comment, "body", body_max_chars)
            return _build_comments_response([comment], total, next_start_at_for(1))

        lo, hi = 0, _COMMENT_BODY_MAX_CHARS
        best: str | None = None
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = response_at_cap(mid)
            if len(candidate) <= max_output_length:
                best = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        if best is not None:
            return best

    # Not even a body-length-zero single comment fits (or raw_fields,
    # which has no smaller-cap retry to attempt at all -- a raw
    # comment's body is Jira's own ADF object, not the plain string
    # _cap_text_field/_summarize_comment's body_max_chars shrinks).
    # Fall back to a genuinely empty page (0 comments, next_start_at
    # skipping past the one that couldn't fit) before giving up
    # entirely -- an empty envelope is smaller than even a body-length-
    # zero single comment (which still carries id/author/visibility/
    # jsd_public/timestamps), so there's a real budget range where this
    # succeeds and a hard error would otherwise have been wrong.
    empty_response = response_for(0)
    if len(empty_response) <= max_output_length:
        return empty_response
    return None


@mcp.tool()
def jira_list_comments(
    issue_key: str,
    cloud_id: str = "",
    limit: int = 50,
    start_at: int = 0,
    raw_fields: bool = False,
) -> str:
    """
    List comments on an issue (id, body, author, visibility, jsd_public,
    timestamps). visibility is set (role/group) when a comment is
    restricted; jsd_public is set (true/false) on a Jira Service
    Management request's comments independent of visibility -- false
    means an agent marked it internal-only, not visible to the
    customer. Neither should be assumed public just because the other
    is null.
    start_at: offset into the full comment list -- pass the previous
    response's next_start_at to fetch the next page (0 to start over).
    A comment body longer than _COMMENT_BODY_MAX_CHARS is truncated
    with body_truncated set to true on that comment; if the whole page
    still doesn't fit the tool output limit even after that, fewer
    comments than requested are returned (down to one) with
    next_start_at pointing at the first one left out, so pagination
    stays valid. In this default (non-raw_fields) mode, a single
    comment that still doesn't fit at that point is shrunk further (a
    smaller body cap, same body_truncated signal) rather than dropped
    -- next_start_at only ever skips past a comment when not even an
    empty-bodied version of it fits in the configured output limit at
    all. raw_fields mode has a lower bar for that skip -- see below.
    raw_fields: return each comment as Jira's own object (body as its
    original ADF rich-text structure, not flattened plain text; every
    other field this endpoint returns by default) instead of the
    summary above -- for an existing integration written against the
    raw shape this tool returned before summarization was added. Pages
    still shrink (down to one comment) to fit the output budget the
    same way the summarized path does; a single raw comment that
    doesn't fit even alone is not shrunk further (unlike the summarized
    path, its body isn't a plain string _cap_text_field can trim), so
    that case skips straight to the same empty-page fallback described
    above as soon as the raw comment alone doesn't fit -- a lower bar
    than the summarized path's "not even empty-bodied" threshold, since
    there's no smaller-cap retry in between.
    """
    try:
        max_results = clamp_limit(limit, max_limit=MAX_LIMIT)
        offset = clamp_offset(start_at)
        result = _request(
            "GET",
            cloud_id,
            _issue_path(issue_key, "/comment"),
            params={"maxResults": max_results, "startAt": offset},
        )
        if not isinstance(result, dict):
            return _error("Unexpected response format from Jira comments API")
        raw_comments = result.get("comments") or []
        raw_total = result.get("total")
        total = raw_total if isinstance(raw_total, int) else offset + len(raw_comments)
        # bool(raw_comments) guards the no-progress case (empty page
        # while total still exceeds offset): next_start_at must never
        # equal start_at. Counted from the RAW page (not any filtered/
        # size-cut list) -- same reasoning as jira_list_projects/
        # jira_search_users: Jira's startAt is positional over the raw
        # page, so undercounting here would make the next request
        # re-fetch (duplicate) comments already consumed.
        has_more_raw = bool(raw_comments) and offset + len(raw_comments) < total
        response = _fit_comments_page(
            raw_comments,
            offset,
            total,
            has_more_raw,
            get_tool_max_output_length(),
            raw_fields=raw_fields,
        )
        if response is None:
            return _error(
                "A Jira comments page exceeds the tool output limit even "
                "when empty; fetch it individually or narrow the request"
            )
        return response
    except Exception as e:
        safe_message = _safe_text(e)
        logger.error(
            f"Error listing comments for Jira issue {issue_key}: {safe_message}"
        )
        return _error(safe_message)


@mcp.tool()
def jira_add_comment(issue_key: str, body: str, cloud_id: str = "") -> str:
    """
    Add a comment to an issue.
    body: the comment text (plain text).
    """
    try:
        result = _request(
            "POST",
            cloud_id,
            _issue_path(issue_key, "/comment"),
            json_data={"body": body},
        )
        return _success(comment=result)
    except Exception as e:
        safe_message = _safe_text(e)
        logger.error(f"Error adding comment to Jira issue {issue_key}: {safe_message}")
        return _error(safe_message)


@mcp.tool()
def jira_search_users(
    query: str, cloud_id: str = "", limit: int = 20, start_at: int = 0
) -> str:
    """
    Search site users by name or email -- accountId, displayName, email.
    Resolve a person to an accountId here before passing
    assignee_account_id to jira_create_issue or jira_update_issue.
    start_at: offset into the full user list -- pass the previous
    response's next_start_at to fetch the next page (0 to start over).
    """
    try:
        max_results = clamp_limit(limit, max_limit=MAX_LIMIT)
        offset = clamp_offset(start_at)
        result = _request(
            "GET",
            cloud_id,
            "/rest/api/2/user/search",
            params={"query": query, "maxResults": max_results, "startAt": offset},
        )
        if not isinstance(result, list):
            return _error("Unexpected response format from Jira user search API")
        users = [
            {
                "account_id": u.get("accountId"),
                "display_name": u.get("displayName"),
                "email": u.get("emailAddress"),
            }
            for u in result
            if isinstance(u, dict)
        ]
        # This endpoint returns a plain array with no total/isLast -- a full
        # page is the only signal a caller has that more results may exist.
        # Count the raw page (not the filtered rows) so a malformed element
        # can't stall pagination or skew the next offset.
        truncated = len(result) == max_results
        return _success(
            users=users,
            truncated=truncated,
            next_start_at=(offset + len(result)) if truncated else None,
        )
    except Exception as e:
        safe_message = _safe_text(e)
        logger.error(f"Error searching Jira users for '{query}': {safe_message}")
        return _error(safe_message)


if __name__ == "__main__":
    mcp.run()
