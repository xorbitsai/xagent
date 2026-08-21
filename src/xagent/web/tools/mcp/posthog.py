import json
import logging
import os
import socket
import time
from typing import Any
from urllib.parse import quote, urlsplit

import requests
from mcp.server.fastmcp import FastMCP

from ....core.utils.security import (
    PrivateNetworkHostError,
    redact_sensitive_text,
    reject_private_network_host,
)
from .utils import setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("posthog-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("posthog-mcp")

DEFAULT_TIMEOUT_SECONDS = 30
MAX_LIMIT = 100
# Matches zoom.py's/linear.py's convention: an error body that isn't the
# expected {"detail": ...} shape (e.g. an HTML gateway error page) must not
# be forwarded to the LLM/logs verbatim and unbounded.
MAX_ERROR_RESPONSE_TEXT_CHARS = 1000
# PostHog rate-limits aggressively, especially the query endpoint; on a 429
# with a small Retry-After we wait once and retry rather than failing
# outright, mirroring the same bounded-retry policy already used by the
# Jira/Slack/Intercom sibling modules.
MAX_RETRY_AFTER_SECONDS = 30


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _headers() -> dict[str, str]:
    api_key = os.environ.get("POSTHOG_API_KEY")
    if not api_key:
        raise ValueError("POSTHOG_API_KEY environment variable is missing")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


# PostHog Cloud has exactly two regional hosts; both always serve over
# https on the default port. An exact-match allowlist -- rather than a
# ".posthog.com" suffix check -- is both simpler and strictly safer: it
# can't be fooled by a malformed hostname (e.g. an empty label like
# ".posthog.com", which a suffix check would accept but which crashes
# socket.getaddrinfo with a UnicodeEncodeError) or by a hypothetical
# posthog.com subdomain that isn't actually one of PostHog's own API
# hosts, and it makes "what port should this connect to" unambiguous.
ALLOWED_POSTHOG_HOSTS = frozenset({"us.posthog.com", "eu.posthog.com"})


def _base_url() -> str:
    """Return the region host this connector's Personal API key belongs to.

    PostHog's US and EU clouds are separate deployments (unlike, say,
    Intercom's single auto-routing host) -- a key created on one region's
    host is not valid against the other, so which host to call is a
    connect-time configuration choice, not something this module can infer.

    POSTHOG_HOST is user-supplied and gets an Authorization: Bearer header
    attached to every request built from it, so it is validated in layers:
    a bare HTTPS origin (no embedded credentials, no path/query/fragment
    that could redirect the request elsewhere), restricted to exactly the
    two supported PostHog Cloud hosts -- self-hosted PostHog instances are
    not a documented target of this connector, so there is no reason to
    send this key anywhere else -- and, as defense in depth against DNS
    for one of those two hosts being rebound to a private/internal
    address, checked against every address it actually resolves to
    (reusing the same reject_private_network_host guard
    src/xagent/web/services/mcp_oauth.py uses for a user-configured MCP
    host, though that module goes further and pins its actual connection
    to the validated address -- this one only validates at call time and
    lets `requests` re-resolve independently, a narrower residual gap
    than not checking DNS at all, not full parity with that module).
    """
    host = os.environ.get("POSTHOG_HOST", "").strip()
    if not host:
        raise ValueError("POSTHOG_HOST environment variable is missing")
    if "://" not in host:
        host = f"https://{host}"

    parsed = urlsplit(host)
    if parsed.scheme.lower() != "https":
        raise ValueError("POSTHOG_HOST must be an https:// URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("POSTHOG_HOST must not contain embedded credentials")
    if parsed.path.strip("/") or parsed.query or parsed.fragment:
        raise ValueError(
            "POSTHOG_HOST must be a bare host, not a URL with a path, "
            "query, or fragment"
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"POSTHOG_HOST has an invalid port: {exc}") from exc
    # 443 is the port every request already goes to below, so spelling it
    # out is a harmless no-op, not a real customization -- only reject a
    # port that would actually change where this connects.
    if port is not None and port != 443:
        raise ValueError(
            "POSTHOG_HOST must not include a port other than the default 443"
        )
    # A trailing "." denotes the DNS root and is semantically equivalent to
    # the same name without it (e.g. "us.posthog.com." == "us.posthog.com");
    # drop it before the allowlist comparison, or a syntactically valid
    # FQDN gets wrongly rejected.
    hostname = (parsed.hostname or "").rstrip(".")
    if hostname not in ALLOWED_POSTHOG_HOSTS:
        raise ValueError(
            "POSTHOG_HOST must be us.posthog.com or eu.posthog.com; "
            "self-hosted PostHog instances are not supported"
        )

    try:
        resolved = socket.getaddrinfo(
            hostname, 443, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
        for *_, sockaddr in resolved:
            reject_private_network_host(str(sockaddr[0]))
    except PrivateNetworkHostError as exc:
        raise ValueError(f"POSTHOG_HOST is not allowed: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"POSTHOG_HOST could not be resolved: {exc}") from exc

    return f"https://{hostname}"


def _clamp_limit(limit: int) -> int:
    return max(1, min(int(limit), MAX_LIMIT))


def _clamp_offset(offset: int) -> int:
    return max(0, int(offset))


def _path_segment(value: str, field_name: str = "id") -> str:
    """Validate then percent-encode a value for safe interpolation into a
    URL path segment (e.g. an organization/project/person/insight id),
    matching jira.py's _path_segment / hubspot.py's _url_path_id.

    An empty value is rejected outright rather than silently building a
    malformed path like ".../persons//" -- every id parameter in this file
    (including project_id, now required rather than defaulting to
    "@current" for the one write tool) goes through this single choke
    point, so a caller passing an empty id gets one clear error here
    instead of a confusing 404 from PostHog.

    Percent-encoding - not a blocklist of "/", "?", "#" - is what actually
    prevents a value like "1/../2" from escaping its intended path segment.
    "@" is left unescaped (RFC 3986 already permits it literally in a path
    segment) purely so the "@current"/"@me" sentinel values every read tool
    here defaults to stay readable in URLs and logs; it plays no role in
    the escape this helper prevents.
    """
    if value is None or not str(value):
        raise ValueError(f"{field_name} must not be empty")
    return quote(str(value), safe="@")


def _extract_error_detail(response: requests.Response) -> str | None:
    """Pull the human-readable message out of a PostHog error body.

    PostHog error responses are {"type", "code", "detail", "attr"}; the
    "detail" field alone is more useful to the LLM than the raw envelope.
    Returns None if the body isn't in the expected shape, so the caller can
    fall back to the raw response text.
    """
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    detail = payload.get("detail")
    return detail if isinstance(detail, str) and detail else None


def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_data: dict[str, Any] | None = None,
) -> Any:
    url = f"{_base_url()}{path}"
    try:
        for attempt in (0, 1):
            response = requests.request(
                method=method,
                url=url,
                headers=_headers(),
                params=params,
                json=json_data,
                timeout=DEFAULT_TIMEOUT_SECONDS,
                # A redirect response is never followed with the Bearer
                # header still attached: PostHog's documented API doesn't
                # redirect, so a 3xx here is either a misconfiguration or
                # exactly the "redirect to an internal host and carry the
                # credential along" SSRF vector _base_url()'s host
                # validation guards against on the way in.
                allow_redirects=False,
            )
            if response.status_code == 429 and attempt == 0:
                try:
                    retry_after = int(response.headers.get("Retry-After", "0"))
                except ValueError:
                    retry_after = 0
                if 0 < retry_after <= MAX_RETRY_AFTER_SECONDS:
                    time.sleep(retry_after)
                    continue
            break
    except requests.RequestException as exc:
        # A connection/timeout/proxy failure's message can itself embed
        # sensitive data -- e.g. a ProxyError echoing the ambient
        # HTTPS_PROXY URL, which may carry embedded user:pass@ credentials
        # (setup_proxy_env() exports whatever the OS has configured) -- so
        # this gets the same redaction the HTTPError response-body path
        # below already has, not just that one case.
        raise RuntimeError(redact_sensitive_text(str(exc))) from exc

    if 300 <= response.status_code < 400:
        raise RuntimeError(
            f"PostHog returned an unexpected redirect (HTTP {response.status_code}); "
            "refusing to follow it with credentials attached"
        )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        message = str(exc)
        detail = _extract_error_detail(response)
        if detail is None:
            detail = response.text.strip()
            if len(detail) > MAX_ERROR_RESPONSE_TEXT_CHARS:
                detail = detail[:MAX_ERROR_RESPONSE_TEXT_CHARS] + "... [truncated]"
        if detail:
            # The response body is attacker/host-controlled content, not
            # something this module wrote; if it happens to echo request
            # headers (e.g. a misconfigured gateway's error page), redact
            # the Bearer token before it reaches logs or the LLM's context.
            message = f"{message} - {redact_sensitive_text(detail)}"
        raise RuntimeError(message) from exc

    if response.status_code == 204 or not response.content:
        return {}
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"PostHog returned a 2xx response with a non-JSON body: {exc}"
        ) from exc


def _paginated_results(
    payload: dict[str, Any], limit: int, offset: int
) -> tuple[list[Any], bool, int | None]:
    """Slice one page of results and compute the offset for the next page.

    PostHog's own "next" field is a full URL; per PostHog's docs that URL is
    a signal that more results exist, not something safe to fetch as-is (it
    would let a compromised/malicious response redirect this connector to
    an arbitrary host on the next call). Callers instead resurface
    next_offset — reusable as this same tool's own bounded offset param —
    so pagination never means following a server-supplied URL.
    """
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a JSON object from PostHog, got {type(payload).__name__}"
        )
    results = payload.get("results") or []
    if not isinstance(results, list):
        raise ValueError(
            f'Expected PostHog\'s "results" field to be a list, got {type(results).__name__}'
        )
    page = results[:limit]
    truncated = bool(payload.get("next")) or len(results) > limit
    # A truncated-but-empty page (a self-contradictory but not-impossible
    # server response) would otherwise yield next_offset == offset, and a
    # caller that mechanically retries with next_offset would loop forever
    # on the exact same request.
    next_offset = offset + len(page) if truncated and page else None
    return page, truncated, next_offset


@mcp.tool()
def posthog_get_current_user() -> str:
    """
    Get the profile of the PostHog account this connector's Personal API key
    belongs to (uuid, email, first_name, last_name). Use this for "my
    account" / "who am I" requests instead of asking the user for their
    PostHog user id.
    """
    try:
        result = _request("GET", "/api/users/@me/")
        return _success(
            user={
                "uuid": result.get("uuid"),
                "email": result.get("email"),
                "first_name": result.get("first_name"),
                "last_name": result.get("last_name"),
            }
        )
    except Exception as e:
        logger.error(f"Error fetching authenticated PostHog user: {e}")
        return _error(str(e))


@mcp.tool()
def posthog_list_organizations(limit: int = 50, offset: int = 0) -> str:
    """
    List organizations this account can see — id, name.
    Use the returned id with posthog_list_projects.
    offset: 0-based result offset for pagination; pass the previous call's
    next_offset to fetch the next page (null when truncated is False).
    """
    try:
        max_results = _clamp_limit(limit)
        page_offset = _clamp_offset(offset)
        result = _request(
            "GET",
            "/api/organizations/",
            params={"limit": max_results, "offset": page_offset},
        )
        organizations, truncated, next_offset = _paginated_results(
            result, max_results, page_offset
        )
        return _success(
            organizations=organizations, truncated=truncated, next_offset=next_offset
        )
    except Exception as e:
        logger.error(f"Error listing PostHog organizations: {e}")
        return _error(str(e))


@mcp.tool()
def posthog_list_projects(
    organization_id: str = "@current", limit: int = 50, offset: int = 0
) -> str:
    """
    List projects in an organization — id, name, and timezone.
    organization_id: an organization id from posthog_list_organizations;
    defaults to the API key owner's most recently active organization.
    Use the returned id (project_id) with every other tool here.
    offset: 0-based result offset for pagination; pass the previous call's
    next_offset to fetch the next page (null when truncated is False).
    """
    try:
        max_results = _clamp_limit(limit)
        page_offset = _clamp_offset(offset)
        result = _request(
            "GET",
            f"/api/organizations/{_path_segment(organization_id, 'organization_id')}"
            "/projects/",
            params={"limit": max_results, "offset": page_offset},
        )
        projects, truncated, next_offset = _paginated_results(
            result, max_results, page_offset
        )
        return _success(projects=projects, truncated=truncated, next_offset=next_offset)
    except Exception as e:
        logger.error(f"Error listing PostHog projects for org {organization_id}: {e}")
        return _error(str(e))


@mcp.tool()
def posthog_query(
    hogql_query: str, project_id: str = "@current", name: str = "", limit: int = 50
) -> str:
    """
    Run a read-only HogQL (PostHog's SQL dialect) query against a project's
    data — the recommended way to query events, persons, or any other
    analytics table (PostHog's own docs deprecate the plain events-listing
    endpoint in favor of this one).
    hogql_query: a SELECT query, e.g. "select event, count() from events
    where timestamp > now() - interval 7 day group by event order by
    count() desc limit 20".
    project_id: a project id from posthog_list_projects; defaults to the
    API key owner's most recently active project.
    name: optional descriptive label for the query (shown in PostHog's own
    query log; purely for the user's/PostHog's bookkeeping).
    limit: caps rows returned here (default 50, hard cap 100 -- see
    MAX_LIMIT), independent of any LIMIT already in hogql_query -- HogQL
    is arbitrary caller-supplied text, so unlike the tools above this one
    has no query-side limit to lean on, and a query result table can be
    large enough to get truncated mid-JSON by this server's own
    tool-output size cap. This is a client-side cap on the response
    already returned, not a row count passed to PostHog, and there is no
    offset param to page past it -- a caller that needs more than 100
    rows, or a specific slice of a larger result, should add its own
    LIMIT/OFFSET to hogql_query instead.
    """
    try:
        body: dict[str, Any] = {"query": {"kind": "HogQLQuery", "query": hogql_query}}
        if name:
            body["name"] = name
        result = _request(
            "POST",
            f"/api/projects/{_path_segment(project_id, 'project_id')}/query/",
            json_data=body,
        )
        max_results = _clamp_limit(limit)
        # Reuses _paginated_results' payload-shape guards and slicing
        # rather than a second, divergent hand-rolled version: PostHog's
        # query endpoint has no "next"/offset concept, so the discarded
        # next_offset is always None here, but truncated is still exactly
        # "there were more rows than max_results".
        rows, truncated, _next_offset = _paginated_results(result, max_results, 0)
        return _success(
            columns=result.get("columns"),
            results=rows,
            hogql=result.get("hogql"),
            truncated=truncated,
        )
    except Exception as e:
        logger.error(f"Error running PostHog HogQL query in project {project_id}: {e}")
        return _error(str(e))


@mcp.tool()
def posthog_list_persons(
    project_id: str = "@current", search: str = "", limit: int = 50, offset: int = 0
) -> str:
    """
    Search/list persons (users tracked by PostHog) in a project.
    search: optional substring matched against distinct_id/email/name by
    PostHog's own search — for aggregate or property-filtered lookups,
    prefer posthog_query against the `persons` table instead.
    offset: 0-based result offset for pagination; pass the previous call's
    next_offset to fetch the next page (null when truncated is False).
    """
    try:
        max_results = _clamp_limit(limit)
        page_offset = _clamp_offset(offset)
        params: dict[str, Any] = {"limit": max_results, "offset": page_offset}
        if search:
            params["search"] = search
        result = _request(
            "GET",
            f"/api/projects/{_path_segment(project_id, 'project_id')}/persons/",
            params=params,
        )
        persons, truncated, next_offset = _paginated_results(
            result, max_results, page_offset
        )
        return _success(persons=persons, truncated=truncated, next_offset=next_offset)
    except Exception as e:
        logger.error(f"Error listing PostHog persons in project {project_id}: {e}")
        return _error(str(e))


@mcp.tool()
def posthog_get_person(person_id: str, project_id: str = "@current") -> str:
    """
    Get one person's full details (distinct_ids, properties, last seen).
    person_id: a person's numeric id or uuid, from posthog_list_persons or
    posthog_query.
    """
    try:
        result = _request(
            "GET",
            f"/api/projects/{_path_segment(project_id, 'project_id')}/persons/"
            f"{_path_segment(person_id, 'person_id')}/",
        )
        return _success(person=result)
    except Exception as e:
        logger.error(
            f"Error fetching PostHog person {person_id} in project {project_id}: {e}"
        )
        return _error(str(e))


@mcp.tool()
def posthog_list_insights(
    project_id: str = "@current", search: str = "", limit: int = 50, offset: int = 0
) -> str:
    """
    List saved insights (trends, funnels, retention, etc.) in a project —
    id, short_id, name, and the insight type.
    search: optional substring matched against the insight's name.
    offset: 0-based result offset for pagination; pass the previous call's
    next_offset to fetch the next page (null when truncated is False).
    """
    try:
        max_results = _clamp_limit(limit)
        page_offset = _clamp_offset(offset)
        params: dict[str, Any] = {
            "limit": max_results,
            "offset": page_offset,
            "basic": "true",
        }
        if search:
            params["search"] = search
        result = _request(
            "GET",
            f"/api/projects/{_path_segment(project_id, 'project_id')}/insights/",
            params=params,
        )
        insights, truncated, next_offset = _paginated_results(
            result, max_results, page_offset
        )
        return _success(insights=insights, truncated=truncated, next_offset=next_offset)
    except Exception as e:
        logger.error(f"Error listing PostHog insights in project {project_id}: {e}")
        return _error(str(e))


@mcp.tool()
def posthog_get_insight(insight_id: str, project_id: str = "@current") -> str:
    """
    Get one saved insight's full definition and cached result.
    insight_id: an insight's numeric id or short_id, from posthog_list_insights.
    """
    try:
        result = _request(
            "GET",
            f"/api/projects/{_path_segment(project_id, 'project_id')}/insights/"
            f"{_path_segment(insight_id, 'insight_id')}/",
        )
        return _success(insight=result)
    except Exception as e:
        logger.error(
            f"Error fetching PostHog insight {insight_id} in project {project_id}: {e}"
        )
        return _error(str(e))


@mcp.tool()
def posthog_list_feature_flags(
    project_id: str = "@current", limit: int = 50, offset: int = 0
) -> str:
    """
    List feature flags in a project — id, key, name, and whether it's active.
    offset: 0-based result offset for pagination; pass the previous call's
    next_offset to fetch the next page (null when truncated is False).
    """
    try:
        max_results = _clamp_limit(limit)
        page_offset = _clamp_offset(offset)
        result = _request(
            "GET",
            f"/api/projects/{_path_segment(project_id, 'project_id')}/feature_flags/",
            params={"limit": max_results, "offset": page_offset},
        )
        flags, truncated, next_offset = _paginated_results(
            result, max_results, page_offset
        )
        return _success(
            feature_flags=flags, truncated=truncated, next_offset=next_offset
        )
    except Exception as e:
        logger.error(
            f"Error listing PostHog feature flags in project {project_id}: {e}"
        )
        return _error(str(e))


@mcp.tool()
def posthog_list_dashboards(
    project_id: str = "@current", limit: int = 50, offset: int = 0
) -> str:
    """
    List dashboards in a project — id, name, and description.
    offset: 0-based result offset for pagination; pass the previous call's
    next_offset to fetch the next page (null when truncated is False).
    """
    try:
        max_results = _clamp_limit(limit)
        page_offset = _clamp_offset(offset)
        result = _request(
            "GET",
            f"/api/projects/{_path_segment(project_id, 'project_id')}/dashboards/",
            params={"limit": max_results, "offset": page_offset},
        )
        dashboards, truncated, next_offset = _paginated_results(
            result, max_results, page_offset
        )
        return _success(
            dashboards=dashboards, truncated=truncated, next_offset=next_offset
        )
    except Exception as e:
        logger.error(f"Error listing PostHog dashboards in project {project_id}: {e}")
        return _error(str(e))


@mcp.tool()
def posthog_list_annotations(
    project_id: str = "@current", search: str = "", limit: int = 50, offset: int = 0
) -> str:
    """
    List annotations (notes marking a point in time, e.g. a deploy or
    incident) in a project.
    search: optional substring matched against the annotation's content.
    offset: 0-based result offset for pagination; pass the previous call's
    next_offset to fetch the next page (null when truncated is False).
    """
    try:
        max_results = _clamp_limit(limit)
        page_offset = _clamp_offset(offset)
        params: dict[str, Any] = {"limit": max_results, "offset": page_offset}
        if search:
            params["search"] = search
        result = _request(
            "GET",
            f"/api/projects/{_path_segment(project_id, 'project_id')}/annotations/",
            params=params,
        )
        annotations, truncated, next_offset = _paginated_results(
            result, max_results, page_offset
        )
        return _success(
            annotations=annotations, truncated=truncated, next_offset=next_offset
        )
    except Exception as e:
        logger.error(f"Error listing PostHog annotations in project {project_id}: {e}")
        return _error(str(e))


@mcp.tool()
def posthog_create_annotation(
    content: str, project_id: str, date_marker: str = ""
) -> str:
    """
    Create an annotation — a note marking a point in time on PostHog's
    graphs, e.g. "Deployed v2.3" or "Started incident".
    content: the annotation's text.
    project_id: a project id from posthog_list_projects. Required — unlike
    the read-only tools here, this has no "@current" default: creating an
    annotation is a write with no delete/undo tool in this connector, and a
    multi-project key's "current" project (PostHog resolves "@current" from
    the account's mutable active-team setting) can silently differ from the
    project the caller intended.
    date_marker: optional ISO 8601 timestamp the annotation should be
    anchored to; defaults to now if omitted.
    """
    try:
        body: dict[str, Any] = {"content": content, "scope": "project"}
        if date_marker:
            body["date_marker"] = date_marker
        result = _request(
            "POST",
            f"/api/projects/{_path_segment(project_id, 'project_id')}/annotations/",
            json_data=body,
        )
        return _success(annotation=result)
    except Exception as e:
        logger.error(f"Error creating PostHog annotation in project {project_id}: {e}")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
