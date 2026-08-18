import json
import logging
import os
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

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
    """
    host = os.environ.get("POSTHOG_HOST")
    if not host:
        raise ValueError("POSTHOG_HOST environment variable is missing")
    return host.rstrip("/")


def _clamp_limit(limit: int) -> int:
    return max(1, min(int(limit), MAX_LIMIT))


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


def _paginated_results(payload: dict[str, Any], limit: int) -> tuple[list[Any], bool]:
    results = payload.get("results") or []
    truncated = bool(payload.get("next")) or len(results) > limit
    return results[:limit], truncated


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
def posthog_list_organizations(limit: int = 50) -> str:
    """
    List organizations this account can see — id, name.
    Use the returned id with posthog_list_projects.
    """
    try:
        max_results = _clamp_limit(limit)
        result = _request("GET", "/api/organizations/", params={"limit": max_results})
        organizations, truncated = _paginated_results(result, max_results)
        return _success(organizations=organizations, truncated=truncated)
    except Exception as e:
        logger.error(f"Error listing PostHog organizations: {e}")
        return _error(str(e))


@mcp.tool()
def posthog_list_projects(organization_id: str = "@current", limit: int = 50) -> str:
    """
    List projects in an organization — id, name, and timezone.
    organization_id: an organization id from posthog_list_organizations;
    defaults to the API key owner's most recently active organization.
    Use the returned id (project_id) with every other tool here.
    """
    try:
        max_results = _clamp_limit(limit)
        result = _request(
            "GET",
            f"/api/organizations/{organization_id}/projects/",
            params={"limit": max_results},
        )
        projects, truncated = _paginated_results(result, max_results)
        return _success(projects=projects, truncated=truncated)
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
        result = _request("POST", f"/api/projects/{project_id}/query/", json_data=body)
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
    project_id: str = "@current", search: str = "", limit: int = 50
) -> str:
    """
    Search/list persons (users tracked by PostHog) in a project.
    search: optional substring matched against distinct_id/email/name by
    PostHog's own search — for aggregate or property-filtered lookups,
    prefer posthog_query against the `persons` table instead.
    """
    try:
        max_results = _clamp_limit(limit)
        params: dict[str, Any] = {"limit": max_results}
        if search:
            params["search"] = search
        result = _request("GET", f"/api/projects/{project_id}/persons/", params=params)
        persons, truncated = _paginated_results(result, max_results)
        return _success(persons=persons, truncated=truncated)
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
        result = _request("GET", f"/api/projects/{project_id}/persons/{person_id}/")
        return _success(person=result)
    except Exception as e:
        logger.error(
            f"Error fetching PostHog person {person_id} in project {project_id}: {e}"
        )
        return _error(str(e))


@mcp.tool()
def posthog_list_insights(
    project_id: str = "@current", search: str = "", limit: int = 50
) -> str:
    """
    List saved insights (trends, funnels, retention, etc.) in a project —
    id, short_id, name, and the insight type.
    search: optional substring matched against the insight's name.
    """
    try:
        max_results = _clamp_limit(limit)
        params: dict[str, Any] = {"limit": max_results, "basic": "true"}
        if search:
            params["search"] = search
        result = _request("GET", f"/api/projects/{project_id}/insights/", params=params)
        insights, truncated = _paginated_results(result, max_results)
        return _success(insights=insights, truncated=truncated)
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
        result = _request("GET", f"/api/projects/{project_id}/insights/{insight_id}/")
        return _success(insight=result)
    except Exception as e:
        logger.error(
            f"Error fetching PostHog insight {insight_id} in project {project_id}: {e}"
        )
        return _error(str(e))


@mcp.tool()
def posthog_list_feature_flags(project_id: str = "@current", limit: int = 50) -> str:
    """
    List feature flags in a project — id, key, name, and whether it's active.
    """
    try:
        max_results = _clamp_limit(limit)
        result = _request(
            "GET",
            f"/api/projects/{project_id}/feature_flags/",
            params={"limit": max_results},
        )
        flags, truncated = _paginated_results(result, max_results)
        return _success(feature_flags=flags, truncated=truncated)
    except Exception as e:
        logger.error(
            f"Error listing PostHog feature flags in project {project_id}: {e}"
        )
        return _error(str(e))


@mcp.tool()
def posthog_list_dashboards(project_id: str = "@current", limit: int = 50) -> str:
    """
    List dashboards in a project — id, name, and description.
    """
    try:
        max_results = _clamp_limit(limit)
        result = _request(
            "GET",
            f"/api/projects/{project_id}/dashboards/",
            params={"limit": max_results},
        )
        dashboards, truncated = _paginated_results(result, max_results)
        return _success(dashboards=dashboards, truncated=truncated)
    except Exception as e:
        logger.error(f"Error listing PostHog dashboards in project {project_id}: {e}")
        return _error(str(e))


@mcp.tool()
def posthog_list_annotations(
    project_id: str = "@current", search: str = "", limit: int = 50
) -> str:
    """
    List annotations (notes marking a point in time, e.g. a deploy or
    incident) in a project.
    search: optional substring matched against the annotation's content.
    """
    try:
        max_results = _clamp_limit(limit)
        params: dict[str, Any] = {"limit": max_results}
        if search:
            params["search"] = search
        result = _request(
            "GET", f"/api/projects/{project_id}/annotations/", params=params
        )
        annotations, truncated = _paginated_results(result, max_results)
        return _success(annotations=annotations, truncated=truncated)
    except Exception as e:
        logger.error(f"Error listing PostHog annotations in project {project_id}: {e}")
        return _error(str(e))


@mcp.tool()
def posthog_create_annotation(
    content: str, project_id: str = "@current", date_marker: str = ""
) -> str:
    """
    Create an annotation — a note marking a point in time on PostHog's
    graphs, e.g. "Deployed v2.3" or "Started incident".
    content: the annotation's text.
    date_marker: optional ISO 8601 timestamp the annotation should be
    anchored to; defaults to now if omitted.
    """
    try:
        body: dict[str, Any] = {"content": content, "scope": "project"}
        if date_marker:
            body["date_marker"] = date_marker
        result = _request(
            "POST", f"/api/projects/{project_id}/annotations/", json_data=body
        )
        return _success(annotation=result)
    except Exception as e:
        logger.error(f"Error creating PostHog annotation in project {project_id}: {e}")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
