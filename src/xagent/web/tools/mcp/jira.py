import json
import logging
import os
import time
from typing import Any
from urllib.parse import quote

import requests
from mcp.server.fastmcp import FastMCP

from ....config import get_tool_max_output_length
from .utils import setup_proxy_env

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
# Matches zoom.py's convention: an error body that isn't the expected
# {"errorMessages": [...]} shape (e.g. an HTML gateway error page) must not
# be forwarded to the LLM/logs verbatim and unbounded.
MAX_ERROR_RESPONSE_TEXT_CHARS = 1000
# Jira endpoints are rate-limited; on a 429 with a small Retry-After we wait
# once and retry rather than failing outright, mirroring the same bounded-
# retry policy as the Slack/Intercom sibling modules.
MAX_RETRY_AFTER_SECONDS = 30


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


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


def _clamp_limit(limit: int) -> int:
    return max(1, min(int(limit), MAX_LIMIT))


def _truncate(text: str) -> str:
    """Bound error text to MAX_ERROR_RESPONSE_TEXT_CHARS, marking the cut."""
    if len(text) > MAX_ERROR_RESPONSE_TEXT_CHARS:
        return text[:MAX_ERROR_RESPONSE_TEXT_CHARS] + "... [truncated]"
    return text


def _path_segment(value: str) -> str:
    """Percent-encode a value for safe interpolation into a URL path
    segment (e.g. an issue key or cloud id), matching mcp/utils.py's
    url_path_id. Percent-encoding - not a blocklist of "/", "?", "#" -
    is what actually prevents a value like "ENG-1/../other" from
    escaping its intended path segment.

    "." and ".." are the one case percent-encoding alone can't close:
    both are always-unreserved per RFC 3986, so quote() never touches
    them, and requests/urllib3 normalize dot-segments out of the final
    URL before sending it (verified directly: requests.Request("GET",
    ".../issue/%2E%2E/secrets").prepare().url collapses right back to
    ".../issue/../secrets" -- a completely different, still-valid
    endpoint). Rejecting the value outright, exactly as url_path_id
    does, is what actually closes this off; encoding alone can't.

    Also rejects a blank or whitespace-padded value (url_path_id's
    require_clean_identifier half, which this file's own copy had been
    missing) -- an empty issue_key (e.g. an unresolved templated
    variable from an LLM caller) would otherwise silently build
    /rest/api/2/issue/, hitting the issue-collection endpoint instead
    of a clear local error naming the actual mistake.
    """
    value = str(value)
    if value in (".", ".."):
        raise ValueError(f"path segment must not be '.' or '..': {value!r}")
    if not value or value.strip() != value:
        raise ValueError(f"path segment must not be blank or padded: {value!r}")
    return quote(value, safe="")


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
    return _truncate("; ".join(messages))


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
            detail = _truncate(response.text.strip())
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
        if not site_id:
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
        logger.error(f"Error listing accessible Jira sites: {e}")
        return _error(str(e))


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
        logger.error(f"Error fetching authenticated Jira user: {e}")
        return _error(str(e))


@mcp.tool()
def jira_list_projects(cloud_id: str = "", limit: int = 50, start_at: int = 0) -> str:
    """
    List projects on a Jira site -- id, key (e.g. "ENG"), and name. Use the
    returned key with jira_create_issue and jira_search_issues.
    cloud_id: optional site id from jira_list_accessible_sites; omit when
    the account has only one accessible site.
    start_at: offset into the full project list -- pass the previous
    response's next_start_at to fetch the next page (0 to start over).
    """
    try:
        max_results = _clamp_limit(limit)
        offset = max(0, int(start_at))
        result = _request(
            "GET",
            cloud_id,
            "/rest/api/2/project/search",
            params={"maxResults": max_results, "startAt": offset},
        )
        if not isinstance(result, dict):
            return _error("Unexpected response format from Jira projects API")
        projects = result.get("values") or []
        # bool(projects) guards against a server that signals more pages
        # while returning an empty page: without it next_start_at would
        # equal start_at and a caller following it would loop forever.
        truncated = bool(projects) and not result.get("isLast", True)
        return _success(
            projects=projects,
            truncated=truncated,
            next_start_at=(offset + len(projects)) if truncated else None,
        )
    except Exception as e:
        logger.error(f"Error listing Jira projects: {e}")
        return _error(str(e))


def _as_dict(value: Any) -> dict[str, Any]:
    """Return value if it's a dict, else {}.

    A single choke point for the "unwrap an optional Jira object field"
    pattern, so a non-dict truthy value (e.g. a misconfigured custom
    field, or a proxy that reshapes one field) degrades to {} instead of
    raising AttributeError on the next .get(...) call.
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

#: Rough upper bound on how many extra characters replacing a page's
#: "total_count": null with an actual number can add (the field's key,
#: comma, and quoting are already present in the null-count page, so the
#: only real delta is null (4 chars) -> up to a many-digit integer).
#: Checked before calling _approximate_count so a page already close to
#: max_output_length skips the network round trip for a count that's
#: about to be discarded anyway, rather than fetching it and finding out
#: after the fact.
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
    issues: list[dict[str, Any]], total_count: int | None, next_token: str | None
) -> str:
    # Field order matters here: total_count/returned_count/truncated/
    # next_page_token come before the (much larger) issues list so that
    # if this string is ever truncated downstream anyway, the caller
    # still sees the pagination signal, not just a partial issues array
    # with no idea more pages exist.
    return _success(
        total_count=total_count,
        returned_count=len(issues),
        truncated=bool(next_token),
        next_page_token=next_token or None,
        issues=issues,
    )


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
    # per source character, so a single budget-sized slice can still
    # overflow; shrink by exactly the overshoot and retry rather than
    # giving up on any message content the moment one slice attempt
    # doesn't fit. Bounded to a handful of iterations since each retry
    # strictly reduces the remaining budget.
    budget = max_output_length - len(_error(""))
    for _ in range(50):
        if budget <= 0:
            break
        short = _error(message[:budget])
        overflow = len(short) - max_output_length
        if overflow <= 0:
            return short
        budget -= overflow
    return json.dumps({"status": "error"}, ensure_ascii=False)


@mcp.tool()
def jira_search_issues(
    jql: str,
    cloud_id: str = "",
    limit: int = 50,
    next_page_token: str = "",
    raw_fields: bool = False,
) -> str:
    """
    Search issues with JQL (Jira Query Language) -- the recommended way to
    find issues by project, assignee, status, text, etc. Each result is a
    compact projection (key, summary, status, status_category, assignee,
    priority, issue_type, project_key, labels, parent_key, resolution,
    created, updated) -- not the full issue. Use jira_get_issue for a
    description, dependencies (issue_links/subtasks), or any other field
    not in that list.
    jql: a JQL query, e.g. 'project = ENG AND status = "In Progress"
    ORDER BY updated DESC' or 'text ~ "login bug"'.
    limit: max issues to return per page (default 50, capped at 100).
    next_page_token: pass the previous response's next_page_token to fetch
    the next page -- always check `truncated` and re-call with it instead
    of assuming one page is everything.
    raw_fields: return each issue as Jira's own nested object (under
    "fields", e.g. issue["fields"]["status"]["id"]) instead of the
    compact projection -- for an existing integration written against
    the raw shape this tool returned before compact projection was
    added. Still only the same field subset the compact projection
    covers (see above); it changes how those fields are shaped, not
    which ones are fetched -- use jira_get_issue for a field not in
    that list either way. Bigger per-issue payload, so a raw page can
    need more/smaller fallback pages to fit the output budget than the
    same query would at the default setting; prefer the default
    projection for new integrations.
    A search error (invalid JQL, a page too big even at minimal size, a
    stuck pagination cursor) always has a "message" key describing it,
    except under an extremely small configured output cap where even
    that can't fit -- there, the response is the bare
    {"status": "error"} envelope with no message key at all.
    Returns total_count: an approximate count of ALL issues matching the
    JQL (independent of pagination; null if unavailable, e.g. the count
    endpoint failed, OR because including it would have pushed an
    otherwise-fitting page over the output limit -- both cases look the
    same to the caller). Only computed for the first page of a search
    (when next_page_token is empty) -- it's a per-JQL constant, not tied
    to any one page, so carry it forward yourself for later pages of the
    same search instead of expecting it again. Compare it against
    returned_count to know whether more issues exist beyond what
    truncated/next_page_token alone would tell you.
    """
    try:
        resolved_cloud_id = _resolve_cloud_id(cloud_id)
        max_results = _clamp_limit(limit)
        max_output_length = get_tool_max_output_length()

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
            response = _build_search_response(issues, None, next_token)
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

        # total_count is a per-JQL constant, independent of pagination --
        # only fetch it once, for the caller's first page, and only when
        # there's actually more beyond this page: an exact count is
        # already known for free (len(issues)) when this page is both
        # the first and the last, so calling the advisory endpoint there
        # would just be a redundant network round trip for a number
        # that's already exact. Never fetched on either early return
        # above (size-overflow or stuck-cursor) either, for the same
        # "don't do wasted work" reason.
        if next_page_token:
            total_count = None
        elif not next_token:
            total_count = len(issues)
        elif len(response) + _TOTAL_COUNT_RESERVE_CHARS > max_output_length:
            # `response` is the already-fitting page the loop above just
            # built -- if there's not even enough headroom left for a
            # plausible total_count value, skip the network round trip
            # (and the count endpoint's own rate-limit budget) for a
            # result that's about to be discarded as soon as it comes
            # back anyway.
            total_count = None
        else:
            total_count = _approximate_count(resolved_cloud_id, jql)

        if total_count is not None:
            with_count = _build_search_response(issues, total_count, next_token)
            if len(with_count) <= max_output_length:
                response = with_count
            else:
                # total_count is advisory; if adding it is ever what
                # tips an already-fitting page over budget, drop it
                # rather than fail a search that otherwise fit -- reuse
                # `response` (the loop's already-fitting, already-built
                # page) instead of paying for another full serialization
                # of `issues` just to reproduce the same page again.
                total_count = None

        logger.info(
            f"jira_search_issues: returned={len(issues)} "
            f"approx_total={total_count} has_next_page={bool(next_token)}"
        )
        return response
    except Exception as e:
        _log_metadata_only(logger.error, "Error searching Jira issues", e)
        return _bounded_search_error(str(e), get_tool_max_output_length())


@mcp.tool()
def jira_get_issue(issue_key: str, cloud_id: str = "") -> str:
    """
    Get one issue's full details, including description, status, assignee,
    and priority.
    issue_key: an issue key (e.g. "ENG-123") or its numeric id.
    """
    try:
        result = _request("GET", cloud_id, _issue_path(issue_key))
        return _success(issue=result)
    except Exception as e:
        logger.error(f"Error fetching Jira issue {issue_key}: {e}")
        return _error(str(e))


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
        if not summary:
            return _error("summary cannot be empty")
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
        logger.error(f"Error creating Jira issue in project {project_key}: {e}")
        return _error(str(e))


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
            if not summary:
                return _error("summary cannot be empty")
            fields["summary"] = summary
        if description is not None:
            fields["description"] = description
        if assignee_account_id is not None:
            fields["assignee"] = (
                {"accountId": assignee_account_id} if assignee_account_id else None
            )
        if priority is not None:
            if not priority:
                return _error(
                    "priority cannot be empty -- Jira has no way to clear "
                    "priority through this field; omit the parameter instead"
                )
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
        logger.error(f"Error updating Jira issue {issue_key}: {e}")
        return _error(str(e))


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
        logger.error(f"Error listing transitions for Jira issue {issue_key}: {e}")
        return _error(str(e))


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
        if not transition_id:
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
        logger.error(
            f"Error transitioning Jira issue {issue_key} to '{transition_name}': {e}"
        )
        return _error(str(e))


@mcp.tool()
def jira_list_comments(
    issue_key: str, cloud_id: str = "", limit: int = 50, start_at: int = 0
) -> str:
    """
    List comments on an issue (body, author, timestamp).
    start_at: offset into the full comment list -- pass the previous
    response's next_start_at to fetch the next page (0 to start over).
    """
    try:
        max_results = _clamp_limit(limit)
        offset = max(0, int(start_at))
        result = _request(
            "GET",
            cloud_id,
            _issue_path(issue_key, "/comment"),
            params={"maxResults": max_results, "startAt": offset},
        )
        if not isinstance(result, dict):
            return _error("Unexpected response format from Jira comments API")
        comments = result.get("comments") or []
        raw_total = result.get("total")
        total = raw_total if isinstance(raw_total, int) else offset + len(comments)
        # bool(comments) guards the no-progress case (empty page while total
        # still exceeds offset): next_start_at must never equal start_at.
        truncated = bool(comments) and offset + len(comments) < total
        return _success(
            comments=comments,
            truncated=truncated,
            next_start_at=(offset + len(comments)) if truncated else None,
        )
    except Exception as e:
        logger.error(f"Error listing comments for Jira issue {issue_key}: {e}")
        return _error(str(e))


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
        logger.error(f"Error adding comment to Jira issue {issue_key}: {e}")
        return _error(str(e))


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
        max_results = _clamp_limit(limit)
        offset = max(0, int(start_at))
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
        logger.error(f"Error searching Jira users for '{query}': {e}")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
