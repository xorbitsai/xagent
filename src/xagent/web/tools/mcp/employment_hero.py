import json
import logging
import os
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

from .utils import clamp_limit, setup_proxy_env, success_with_capped_dict, url_path_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("employment-hero-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("employment-hero-mcp")

EMPLOYMENT_HERO_BASE_URL = "https://api.employmenthero.com/api/v1"
DEFAULT_TIMEOUT_SECONDS = 30
# Matches zoom.py/salesforce.py's convention: an error body that isn't the
# expected {"message": ...} shape (e.g. an HTML gateway error page) must not
# be forwarded to the LLM/logs verbatim and unbounded.
MAX_ERROR_RESPONSE_TEXT_CHARS = 1000
# Documented per-endpoint default/max for page_index/item_per_page across the
# Employment Hero API references (organisations, employees, teams,
# timesheet_entries all share this contract).
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _headers() -> dict[str, str]:
    access_token = os.environ.get("EMPLOYMENT_HERO_ACCESS_TOKEN")
    if not access_token:
        raise ValueError("EMPLOYMENT_HERO_ACCESS_TOKEN environment variable is missing")
    return {"Authorization": f"Bearer {access_token}"}


def _extract_error_detail(response: requests.Response) -> str | None:
    """Pull the human-readable message out of an Employment Hero error body.

    Returns None if the body isn't in the expected shape, so the caller can
    fall back to the raw response text.
    """
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    message = payload.get("message") or payload.get("error")
    return message if isinstance(message, str) and message else None


def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
) -> Any:
    response = requests.request(
        method=method,
        url=f"{EMPLOYMENT_HERO_BASE_URL}{path}",
        headers=_headers(),
        params=params,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    if response.status_code >= 400:
        detail = _extract_error_detail(response)
        if detail is None:
            detail = response.text.strip()
            if len(detail) > MAX_ERROR_RESPONSE_TEXT_CHARS:
                detail = detail[:MAX_ERROR_RESPONSE_TEXT_CHARS] + "... [truncated]"
        raise RuntimeError(
            f"Employment Hero API error (status {response.status_code}): {detail}"
        )

    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


def _unwrap_data(payload: Any) -> Any:
    """Every Employment Hero response wraps its real payload in a top-level
    "data" key -- unwrap it here once rather than in every tool below."""
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


def _pagination_params(page_index: int, item_per_page: int) -> dict[str, Any]:
    return {
        "page_index": max(1, int(page_index)),
        "item_per_page": clamp_limit(item_per_page, max_limit=MAX_PAGE_SIZE),
    }


@mcp.tool()
def employment_hero_list_organisations(
    page_index: int = 1, item_per_page: int = DEFAULT_PAGE_SIZE
) -> str:
    """
    List the organisations this connection can access, with each
    organisation's id, name, and country. Call this first -- every other
    tool in this connector needs an organisation_id, and this is how to find
    one.
    page_index: 1-based page number.
    item_per_page: results per page (max 100).
    """
    try:
        result = _request(
            "GET",
            "/organisations",
            params=_pagination_params(page_index, item_per_page),
        )
        return success_with_capped_dict("organisations", _unwrap_data(result))
    except Exception as e:
        logger.error(f"Error listing Employment Hero organisations: {e}")
        return _error(str(e))


@mcp.tool()
def employment_hero_list_employees(
    organisation_id: str,
    member_type: str = "",
    page_index: int = 1,
    item_per_page: int = DEFAULT_PAGE_SIZE,
) -> str:
    """
    List employees (and contractors) in an organisation.
    organisation_id: id from employment_hero_list_organisations.
    member_type: optional filter, e.g. "employee" or "contractor". Leave
    empty to return both.
    page_index: 1-based page number.
    item_per_page: results per page (max 100).
    """
    try:
        safe_org_id = url_path_id(organisation_id, "organisation_id")
        params = _pagination_params(page_index, item_per_page)
        if member_type:
            params["member_type"] = member_type
        result = _request(
            "GET", f"/organisations/{safe_org_id}/employees", params=params
        )
        return success_with_capped_dict("employees", _unwrap_data(result))
    except Exception as e:
        logger.error(f"Error listing Employment Hero employees: {e}")
        return _error(str(e))


@mcp.tool()
def employment_hero_get_employee(organisation_id: str, employee_id: str) -> str:
    """
    Get full details for one employee (name, job title, employment type,
    status, teams, primary manager, etc).
    organisation_id: id from employment_hero_list_organisations.
    employee_id: id from employment_hero_list_employees.
    """
    try:
        safe_org_id = url_path_id(organisation_id, "organisation_id")
        safe_employee_id = url_path_id(employee_id, "employee_id")
        result = _request(
            "GET", f"/organisations/{safe_org_id}/employees/{safe_employee_id}"
        )
        return success_with_capped_dict("employee", _unwrap_data(result))
    except Exception as e:
        logger.error(f"Error getting Employment Hero employee {employee_id}: {e}")
        return _error(str(e))


@mcp.tool()
def employment_hero_list_teams(
    organisation_id: str, page_index: int = 1, item_per_page: int = DEFAULT_PAGE_SIZE
) -> str:
    """
    List teams in an organisation.
    organisation_id: id from employment_hero_list_organisations.
    page_index: 1-based page number.
    item_per_page: results per page (max 100).
    """
    try:
        safe_org_id = url_path_id(organisation_id, "organisation_id")
        result = _request(
            "GET",
            f"/organisations/{safe_org_id}/teams",
            params=_pagination_params(page_index, item_per_page),
        )
        return success_with_capped_dict("teams", _unwrap_data(result))
    except Exception as e:
        logger.error(f"Error listing Employment Hero teams: {e}")
        return _error(str(e))


@mcp.tool()
def employment_hero_list_team_employees(
    organisation_id: str,
    team_id: str,
    page_index: int = 1,
    item_per_page: int = DEFAULT_PAGE_SIZE,
) -> str:
    """
    List the employees who belong to one team.
    organisation_id: id from employment_hero_list_organisations.
    team_id: id from employment_hero_list_teams.
    page_index: 1-based page number.
    item_per_page: results per page (max 100).
    """
    try:
        safe_org_id = url_path_id(organisation_id, "organisation_id")
        safe_team_id = url_path_id(team_id, "team_id")
        result = _request(
            "GET",
            f"/organisations/{safe_org_id}/teams/{safe_team_id}/employees",
            params=_pagination_params(page_index, item_per_page),
        )
        return success_with_capped_dict("employees", _unwrap_data(result))
    except Exception as e:
        logger.error(f"Error listing Employment Hero team employees: {e}")
        return _error(str(e))


@mcp.tool()
def employment_hero_list_timesheet_entries(
    organisation_id: str,
    employee_id: str,
    start_date: str = "",
    end_date: str = "",
    page_index: int = 1,
    item_per_page: int = DEFAULT_PAGE_SIZE,
) -> str:
    """
    List one employee's timesheet entries (date, start/end time, breaks,
    status, units worked).
    organisation_id: id from employment_hero_list_organisations.
    employee_id: id from employment_hero_list_employees.
    start_date, end_date: optional date range filter in dd/mm/yyyy format
    (Employment Hero's documented format for this endpoint). Leave both
    empty to return every entry.
    page_index: 1-based page number.
    item_per_page: results per page (max 100).
    """
    try:
        safe_org_id = url_path_id(organisation_id, "organisation_id")
        safe_employee_id = url_path_id(employee_id, "employee_id")
        params = _pagination_params(page_index, item_per_page)
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        result = _request(
            "GET",
            f"/organisations/{safe_org_id}/employees/{safe_employee_id}/timesheet_entries",
            params=params,
        )
        return success_with_capped_dict("timesheet_entries", _unwrap_data(result))
    except Exception as e:
        logger.error(f"Error listing Employment Hero timesheet entries: {e}")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
