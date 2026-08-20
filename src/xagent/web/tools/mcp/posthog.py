import json
import logging
import os
from typing import Any
from urllib.parse import quote, urlsplit

import requests
from mcp.server.fastmcp import FastMCP

from ....core.utils.security import PrivateNetworkHostError, reject_private_network_host
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


def _base_url() -> str:
    """Return the region host this connector's Personal API key belongs to.

    PostHog's US and EU clouds are separate deployments (unlike, say,
    Intercom's single auto-routing host) -- a key created on one region's
    host is not valid against the other, so which host to call is a
    connect-time configuration choice, not something this module can infer.

    POSTHOG_HOST is user-supplied and gets an Authorization: Bearer header
    attached to every request built from it, so it is validated as a bare
    HTTPS origin (matching mcp_oauth.py's use of the same
    reject_private_network_host guard for a user-configured MCP host):
    no embedded credentials, no path/query/fragment that could redirect the
    request elsewhere, and no localhost/private/link-local target.
    """
    host = os.environ.get("POSTHOG_HOST", "").strip()
    if not host:
        raise ValueError("POSTHOG_HOST environment variable is missing")
    if not host.startswith(("http://", "https://")):
        host = f"https://{host}"

    parsed = urlsplit(host)
    if parsed.scheme != "https":
        raise ValueError("POSTHOG_HOST must be an https:// URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("POSTHOG_HOST must not contain embedded credentials")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError(
            "POSTHOG_HOST must be a bare host, not a URL with a path, "
            "query, or fragment"
        )
    if not parsed.hostname:
        raise ValueError("POSTHOG_HOST environment variable is missing")
    try:
        reject_private_network_host(parsed.hostname)
    except PrivateNetworkHostError as exc:
        raise ValueError(f"POSTHOG_HOST is not allowed: {exc}") from exc

    netloc = (
        parsed.hostname if parsed.port is None else f"{parsed.hostname}:{parsed.port}"
    )
    return f"https://{netloc}"


def _clamp_limit(limit: int) -> int:
    return max(1, min(int(limit), MAX_LIMIT))


def _clamp_offset(offset: int) -> int:
    return max(0, int(offset))


def _path_segment(value: str) -> str:
    """Percent-encode a value for safe interpolation into a URL path
    segment (e.g. an organization/project/person/insight id), matching
    jira.py's _path_segment / intercom.py's inline quote() calls.
    Percent-encoding - not a blocklist of "/", "?", "#" - is what actually
    prevents a value like "1/../2" from escaping its intended path segment.
    """
    return quote(str(value), safe="")


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
    response = requests.request(
        method=method,
        url=f"{_base_url()}{path}",
        headers=_headers(),
        params=params,
        json=json_data,
        timeout=DEFAULT_TIMEOUT_SECONDS,
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
            message = f"{message} - {detail}"
        raise RuntimeError(message) from exc

    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


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
    results = payload.get("results") or []
    page = results[:limit]
    truncated = bool(payload.get("next")) or len(results) > limit
    next_offset = offset + len(page) if truncated else None
    return page, truncated, next_offset


@mcp.tool()
def posthog_get_current_user() -> str:
    """
    Get the profile of the PostHog account this connector's Personal API key
    belongs to (id, email, name). Use this for "my account" / "who am I"
    requests instead of asking the user for their PostHog user id.
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
    next_offset to fetch the next page (only present while truncated=True).
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
    next_offset to fetch the next page (only present while truncated=True).
    """
    try:
        max_results = _clamp_limit(limit)
        page_offset = _clamp_offset(offset)
        result = _request(
            "GET",
            f"/api/organizations/{_path_segment(organization_id)}/projects/",
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
    hogql_query: str, project_id: str = "@current", name: str = ""
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
    """
    try:
        body: dict[str, Any] = {"query": {"kind": "HogQLQuery", "query": hogql_query}}
        if name:
            body["name"] = name
        result = _request(
            "POST", f"/api/projects/{_path_segment(project_id)}/query/", json_data=body
        )
        return _success(
            columns=result.get("columns"),
            results=result.get("results"),
            hogql=result.get("hogql"),
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
    next_offset to fetch the next page (only present while truncated=True).
    """
    try:
        max_results = _clamp_limit(limit)
        page_offset = _clamp_offset(offset)
        params: dict[str, Any] = {"limit": max_results, "offset": page_offset}
        if search:
            params["search"] = search
        result = _request(
            "GET", f"/api/projects/{_path_segment(project_id)}/persons/", params=params
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
            f"/api/projects/{_path_segment(project_id)}/persons/"
            f"{_path_segment(person_id)}/",
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
    next_offset to fetch the next page (only present while truncated=True).
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
            "GET", f"/api/projects/{_path_segment(project_id)}/insights/", params=params
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
            f"/api/projects/{_path_segment(project_id)}/insights/"
            f"{_path_segment(insight_id)}/",
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
    next_offset to fetch the next page (only present while truncated=True).
    """
    try:
        max_results = _clamp_limit(limit)
        page_offset = _clamp_offset(offset)
        result = _request(
            "GET",
            f"/api/projects/{_path_segment(project_id)}/feature_flags/",
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
    next_offset to fetch the next page (only present while truncated=True).
    """
    try:
        max_results = _clamp_limit(limit)
        page_offset = _clamp_offset(offset)
        result = _request(
            "GET",
            f"/api/projects/{_path_segment(project_id)}/dashboards/",
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
    next_offset to fetch the next page (only present while truncated=True).
    """
    try:
        max_results = _clamp_limit(limit)
        page_offset = _clamp_offset(offset)
        params: dict[str, Any] = {"limit": max_results, "offset": page_offset}
        if search:
            params["search"] = search
        result = _request(
            "GET",
            f"/api/projects/{_path_segment(project_id)}/annotations/",
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
            f"/api/projects/{_path_segment(project_id)}/annotations/",
            json_data=body,
        )
        return _success(annotation=result)
    except Exception as e:
        logger.error(f"Error creating PostHog annotation in project {project_id}: {e}")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
