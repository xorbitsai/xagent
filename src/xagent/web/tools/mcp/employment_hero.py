import json
import logging
import math
import os
from datetime import date
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from ....core.utils import public_requests as requests
from .utils import clamp_limit, success_with_capped_dict, url_path_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("employment-hero-mcp")


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

# Employment Hero's documented member_type filter values for the employees
# endpoint (see employment_hero_list_employees) -- validated up front rather
# than forwarded verbatim, so a typo'd/guessed value (e.g. "staff") gets a
# clear error instead of an unpredictable API-side response.
MEMBER_TYPES = ("employee", "contractor")
# Documented request-filter values for both the organisation manager view and
# the employee leave-request endpoint. Response payloads use title case, but
# the request contract uses these lowercase values.
LEAVE_REQUEST_STATUSES = ("pending", "approved", "declined")

READ_ONLY = ToolAnnotations(readOnlyHint=True)
CREATE = ToolAnnotations(destructiveHint=False, idempotentHint=False)
UPDATE = ToolAnnotations(destructiveHint=True, idempotentHint=True)

# Employment Hero's public endpoint reference does not publish its OAuth
# scope catalogue. These identifiers follow the connector's existing
# resource:action contract and must also be enabled on the vendor-side OAuth
# application before rollout. Keep the hints next to the calls they protect so
# a 403 tells a user which permission must be added before they reconnect.
LEAVE_CATEGORIES_LIST_SCOPE = "leave_categories:list"
LEAVE_BALANCES_LIST_SCOPE = "leave_balances:list"
LEAVE_REQUESTS_LIST_SCOPE = "leave_requests:list"
LEAVE_REQUESTS_CREATE_SCOPE = "leave_requests:create"
LEAVE_REQUESTS_APPROVE_SCOPE = "leave_requests:approve"


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _headers() -> dict[str, str]:
    access_token = os.environ.get("EMPLOYMENT_HERO_ACCESS_TOKEN")
    if not access_token:
        raise ValueError("EMPLOYMENT_HERO_ACCESS_TOKEN environment variable is missing")
    return {"Authorization": f"Bearer {access_token}"}


def _extract_error_detail(response: requests.Response) -> str | None:
    """Pull the human-readable message out of an Employment Hero error body.

    Employment Hero's API reference documents `error` as a top-level message
    container and `errors` as an array of per-field validation details (see
    https://developer.employmenthero.com/api-references/introduction) --
    a validation failure can carry only `errors`, with no `message`/`error`
    string, so this also unpacks that shape rather than only the top-level
    fields. Returns None if the body isn't in any recognized shape, so the
    caller can fall back to the raw response text.
    """
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    message = payload.get("message") or payload.get("error")
    if isinstance(message, str) and message:
        return message
    errors = payload.get("errors")
    if isinstance(errors, list) and errors:
        first = errors[0]
        detail = first.get("message") if isinstance(first, dict) else first
        if isinstance(detail, str) and detail:
            return detail
    elif isinstance(errors, dict) and errors:
        field, detail = next(iter(errors.items()))
        if isinstance(detail, list) and detail:
            detail = detail[0]
        if isinstance(detail, str) and detail:
            return f"{field}: {detail}"
    return None


def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_data: dict[str, Any] | None = None,
    required_scope: str | None = None,
) -> Any:
    request_kwargs: dict[str, Any] = {
        "method": method,
        "url": f"{EMPLOYMENT_HERO_BASE_URL}{path}",
        "headers": _headers(),
        "params": params,
        "timeout": DEFAULT_TIMEOUT_SECONDS,
    }
    if json_data is not None:
        request_kwargs["json"] = json_data
    response = requests.request(
        **request_kwargs,
    )
    if response.status_code >= 400:
        detail = _extract_error_detail(response)
        if detail is None:
            detail = response.text.strip()
        # Applies to both branches -- a long or numerous structured error
        # array (_extract_error_detail's case) is just as unbounded as raw
        # response text, and was previously only capped in the latter.
        if len(detail) > MAX_ERROR_RESPONSE_TEXT_CHARS:
            detail = detail[:MAX_ERROR_RESPONSE_TEXT_CHARS] + "... [truncated]"
        if response.status_code == 403 and required_scope:
            detail = (
                f"{detail}. This operation requires the Employment Hero OAuth "
                f"scope {required_scope!r}. Ask an administrator to enable it "
                "for the OAuth application, then disconnect and reconnect "
                "Employment Hero."
            )
        raise RuntimeError(
            f"Employment Hero API error (status {response.status_code}): {detail}"
        )

    if response.status_code == 204 or not response.content:
        return {}
    try:
        return response.json()
    except ValueError:
        # A 200 with a non-JSON body (e.g. an HTML gateway/proxy page) is
        # rare but not impossible -- every @mcp.tool() caller already
        # wraps this in try/except Exception, so raising here still
        # returns a clean _error(...) envelope, just with a message that
        # names the actual problem instead of a raw JSONDecodeError.
        detail = response.text.strip()
        if len(detail) > MAX_ERROR_RESPONSE_TEXT_CHARS:
            detail = detail[:MAX_ERROR_RESPONSE_TEXT_CHARS] + "... [truncated]"
        raise RuntimeError(
            f"Employment Hero returned a non-JSON response: {detail}"
        ) from None


def _unwrap_data(payload: Any) -> Any:
    """Every Employment Hero response wraps its real payload in a top-level
    "data" key -- unwrap it here once rather than in every tool below."""
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


def _list_payload(result: Any, resource: str) -> dict[str, Any]:
    """Unwrap a list endpoint's `data` payload, raising if it isn't the
    {"items": [...], ...} shape Employment Hero's documented response
    envelope guarantees.

    success_with_capped_dict only applies its own output-size cap when its
    `data` argument is a dict -- silently returning an unbounded response
    otherwise -- so this guards every list tool against a live response
    that ever deviates from the documented shape (this connector has not
    been exercised against a live Employment Hero account) rather than
    letting a stray list/None sail through uncapped.
    """
    data = _unwrap_data(result)
    if not isinstance(data, dict):
        raise RuntimeError(
            f"Employment Hero returned an unexpected response for {resource}"
        )
    return data


def _pagination_params(page_index: int, item_per_page: int) -> dict[str, Any]:
    return {
        "page_index": max(1, int(page_index)),
        "item_per_page": clamp_limit(item_per_page, max_limit=MAX_PAGE_SIZE),
    }


def _record_payload(result: Any, resource: str) -> dict[str, Any]:
    """Unwrap and validate a single-record response."""
    data = _unwrap_data(result)
    if not isinstance(data, dict) or not data:
        raise RuntimeError(
            f"Employment Hero returned an unexpected response for {resource}"
        )
    return data


def _iso_date(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a valid YYYY-MM-DD date")
    normalized = value.strip()
    try:
        parsed = date.fromisoformat(normalized)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a valid YYYY-MM-DD date") from None
    if parsed.isoformat() != normalized:
        raise ValueError(f"{field_name} must be a valid YYYY-MM-DD date")
    return normalized


def _required_value(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _date_range(start_date: str, end_date: str) -> tuple[str, str]:
    start = _iso_date(start_date, "start_date")
    end = _iso_date(end_date, "end_date")
    if date.fromisoformat(end) < date.fromisoformat(start):
        raise ValueError("end_date must be on or after start_date")
    return start, end


def _optional_date_range(start_date: str, end_date: str) -> tuple[str, str]:
    start = _iso_date(start_date, "start_date") if start_date else ""
    end = _iso_date(end_date, "end_date") if end_date else ""
    if start and end and date.fromisoformat(end) < date.fromisoformat(start):
        raise ValueError("end_date must be on or after start_date")
    return start, end


def _hours_per_day_payload(
    hours_per_day: list[dict[str, Any]] | None,
    start_date: str,
    end_date: str,
) -> list[dict[str, Any]] | None:
    if hours_per_day is None:
        return None
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    normalized: list[dict[str, Any]] = []
    seen_dates: set[str] = set()
    for index, entry in enumerate(hours_per_day):
        if not isinstance(entry, dict):
            raise ValueError(f"hours_per_day[{index}] must be an object")
        if set(entry) != {"date", "hours"}:
            raise ValueError(
                f"hours_per_day[{index}] must contain exactly 'date' and 'hours'"
            )
        entry_date = _iso_date(entry["date"], f"hours_per_day[{index}].date")
        parsed_date = date.fromisoformat(entry_date)
        if not start <= parsed_date <= end:
            raise ValueError(
                f"hours_per_day[{index}].date must fall within the leave request"
            )
        if entry_date in seen_dates:
            raise ValueError(f"hours_per_day contains duplicate date {entry_date}")
        hours = entry["hours"]
        if (
            isinstance(hours, bool)
            or not isinstance(hours, (int, float))
            or not math.isfinite(hours)
            or hours < 0
        ):
            raise ValueError(
                f"hours_per_day[{index}].hours must be a non-negative number"
            )
        seen_dates.add(entry_date)
        normalized.append({"date": entry_date, "hours": hours})
    return normalized


@mcp.tool(annotations=READ_ONLY)
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
        return success_with_capped_dict(
            "organisations", _list_payload(result, "organisations")
        )
    except Exception as e:
        logger.error(f"Error listing Employment Hero organisations: {e}")
        return _error(str(e))


@mcp.tool(annotations=READ_ONLY)
def employment_hero_list_employees(
    organisation_id: str,
    member_type: str = "",
    page_index: int = 1,
    item_per_page: int = DEFAULT_PAGE_SIZE,
    query: str = "",
) -> str:
    """
    List employees (and contractors) in an organisation.
    organisation_id: id from employment_hero_list_organisations.
    member_type: optional filter. Must be exactly "employee" or
    "contractor" (lowercase) if given. Leave empty to return both.
    page_index: 1-based page number.
    item_per_page: results per page (max 100).
    query: optional case-insensitive search by name, preferred name, email,
    job title, or worker code. Use this instead of loading every employee
    when resolving a person by name; an unfiltered large response can be
    truncated by the platform output limit.
    """
    if member_type and member_type not in MEMBER_TYPES:
        return _error(
            f"Invalid member_type {member_type!r}; expected one of "
            f"{', '.join(MEMBER_TYPES)}, or empty for both"
        )
    try:
        safe_org_id = url_path_id(organisation_id, "organisation_id")
        params = _pagination_params(page_index, item_per_page)
        if member_type:
            params["member_type"] = member_type
        if query.strip():
            params["query"] = query.strip()
        result = _request(
            "GET", f"/organisations/{safe_org_id}/employees", params=params
        )
        return success_with_capped_dict("employees", _list_payload(result, "employees"))
    except Exception as e:
        logger.error(f"Error listing Employment Hero employees: {e}")
        return _error(str(e))


@mcp.tool(annotations=READ_ONLY)
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
        employee = _unwrap_data(result)
        if not isinstance(employee, dict):
            # A 200 with a null/non-object payload (e.g. a soft-deleted or
            # access-restricted record) is distinct from the 404 a missing
            # id would raise -- matching deputy_get_resource's convention of
            # surfacing this as a clear error rather than a "success" envelope
            # wrapping a null.
            return _error(
                f"Employment Hero returned an unexpected response for employee "
                f"{employee_id}"
            )
        return success_with_capped_dict("employee", employee)
    except Exception as e:
        logger.error(f"Error getting Employment Hero employee {employee_id}: {e}")
        return _error(str(e))


@mcp.tool(annotations=READ_ONLY)
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
        return success_with_capped_dict("teams", _list_payload(result, "teams"))
    except Exception as e:
        logger.error(f"Error listing Employment Hero teams: {e}")
        return _error(str(e))


@mcp.tool(annotations=READ_ONLY)
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
        return success_with_capped_dict("employees", _list_payload(result, "employees"))
    except Exception as e:
        logger.error(f"Error listing Employment Hero team employees: {e}")
        return _error(str(e))


@mcp.tool(annotations=READ_ONLY)
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
        return success_with_capped_dict(
            "timesheet_entries", _list_payload(result, "timesheet_entries")
        )
    except Exception as e:
        logger.error(f"Error listing Employment Hero timesheet entries: {e}")
        return _error(str(e))


@mcp.tool(annotations=READ_ONLY)
def employment_hero_list_leave_categories(
    organisation_id: str,
    page_index: int = 1,
    item_per_page: int = DEFAULT_PAGE_SIZE,
) -> str:
    """
    List the leave categories available in an organisation. Use the returned
    category id when creating or filtering leave requests.
    organisation_id: id from employment_hero_list_organisations.
    page_index: 1-based page number.
    item_per_page: results per page (max 100).
    """
    try:
        safe_org_id = url_path_id(organisation_id, "organisation_id")
        result = _request(
            "GET",
            f"/organisations/{safe_org_id}/leave_categories",
            params=_pagination_params(page_index, item_per_page),
            required_scope=LEAVE_CATEGORIES_LIST_SCOPE,
        )
        return success_with_capped_dict(
            "leave_categories", _list_payload(result, "leave categories")
        )
    except Exception as e:
        logger.error(f"Error listing Employment Hero leave categories: {e}")
        return _error(str(e))


@mcp.tool(annotations=READ_ONLY)
def employment_hero_get_leave_balances(
    organisation_id: str,
    employee_id: str,
    page_index: int = 1,
    item_per_page: int = DEFAULT_PAGE_SIZE,
) -> str:
    """
    Get one employee's leave balances, including accrued, approved, pending,
    and available amounts for each category. Employment Hero only supports
    this endpoint for organisations using HR Leave with Leave Balance data;
    payroll-connected organisations are not supported by the vendor.
    organisation_id: id from employment_hero_list_organisations.
    employee_id: resolve with employment_hero_list_employees(query=...).
    page_index: 1-based page number.
    item_per_page: results per page (max 100).
    """
    try:
        safe_org_id = url_path_id(organisation_id, "organisation_id")
        safe_employee_id = url_path_id(employee_id, "employee_id")
        result = _request(
            "GET",
            f"/organisations/{safe_org_id}/employees/{safe_employee_id}/leave_balances",
            params=_pagination_params(page_index, item_per_page),
            required_scope=LEAVE_BALANCES_LIST_SCOPE,
        )
        return success_with_capped_dict(
            "leave_balances", _list_payload(result, "leave balances")
        )
    except Exception as e:
        logger.error(f"Error getting Employment Hero leave balances: {e}")
        return _error(str(e))


@mcp.tool(annotations=READ_ONLY)
def employment_hero_list_leave_requests(
    organisation_id: str,
    employee_id: str = "",
    statuses: list[str] | None = None,
    start_date: str = "",
    end_date: str = "",
    leave_category_id: str = "",
    page_index: int = 1,
    item_per_page: int = DEFAULT_PAGE_SIZE,
) -> str:
    """
    List leave requests. Leave employee_id empty for an organisation-wide
    manager/approver view, or provide it for one employee's requests.
    organisation_id: id from employment_hero_list_organisations.
    employee_id: optional id from employment_hero_list_employees(query=...).
    statuses: optional request status filters: pending, approved, or declined.
    start_date, end_date: optional inclusive YYYY-MM-DD range boundaries.
    leave_category_id: optional id from employment_hero_list_leave_categories.
    page_index: 1-based page number.
    item_per_page: results per page (max 100).
    """
    try:
        safe_org_id = url_path_id(organisation_id, "organisation_id")
        normalized_start, normalized_end = _optional_date_range(start_date, end_date)
        params = _pagination_params(page_index, item_per_page)
        if statuses:
            normalized_statuses = [
                status.strip().lower() for status in statuses if status.strip()
            ]
            if not normalized_statuses:
                raise ValueError("statuses must contain at least one non-empty value")
            invalid_statuses = sorted(
                set(normalized_statuses).difference(LEAVE_REQUEST_STATUSES)
            )
            if invalid_statuses:
                raise ValueError(
                    "Invalid statuses "
                    f"{', '.join(invalid_statuses)}; expected only "
                    f"{', '.join(LEAVE_REQUEST_STATUSES)}"
                )
            params["statuses"] = normalized_statuses
        if normalized_start:
            params["start_date"] = normalized_start
        if normalized_end:
            params["end_date"] = normalized_end
        if leave_category_id:
            params["leave_category_id"] = _required_value(
                leave_category_id, "leave_category_id"
            )
        path = f"/organisations/{safe_org_id}/leave_requests"
        if employee_id:
            safe_employee_id = url_path_id(employee_id, "employee_id")
            path = (
                f"/organisations/{safe_org_id}/employees/{safe_employee_id}"
                "/leave_requests"
            )
        result = _request(
            "GET",
            path,
            params=params,
            required_scope=LEAVE_REQUESTS_LIST_SCOPE,
        )
        return success_with_capped_dict(
            "leave_requests", _list_payload(result, "leave requests")
        )
    except Exception as e:
        logger.error(f"Error listing Employment Hero leave requests: {e}")
        return _error(str(e))


@mcp.tool(annotations=CREATE)
def employment_hero_create_leave_request(
    organisation_id: str,
    employee_id: str,
    leave_category_id: str,
    start_date: str,
    end_date: str,
    comment: str = "",
    hours_per_day: list[dict[str, Any]] | None = None,
) -> str:
    """
    Create a leave request for an employee. This changes Employment Hero and
    is not idempotent: confirm the employee, category, dates, hours, and
    comment with the user before calling, and do not blindly retry after an
    ambiguous timeout.
    organisation_id: id from employment_hero_list_organisations.
    employee_id: resolve with employment_hero_list_employees(query=...).
    leave_category_id: id from employment_hero_list_leave_categories.
    start_date, end_date: inclusive YYYY-MM-DD dates.
    comment: optional comment for the request.
    hours_per_day: optional [{"date": "YYYY-MM-DD", "hours": 7.6}, ...].
    Dates omitted here use the employee's typical working-day hours.
    """
    try:
        safe_org_id = url_path_id(organisation_id, "organisation_id")
        safe_employee_id = url_path_id(employee_id, "employee_id")
        category_id = _required_value(leave_category_id, "leave_category_id")
        normalized_start, normalized_end = _date_range(start_date, end_date)
        normalized_hours = _hours_per_day_payload(
            hours_per_day, normalized_start, normalized_end
        )
        body: dict[str, Any] = {
            "leave_category_id": category_id,
            "start_date": normalized_start,
            "end_date": normalized_end,
        }
        if comment:
            body["comment"] = comment
        if normalized_hours is not None:
            body["hours_per_day"] = normalized_hours
        result = _request(
            "POST",
            f"/organisations/{safe_org_id}/employees/{safe_employee_id}/leave_requests",
            json_data=body,
            required_scope=LEAVE_REQUESTS_CREATE_SCOPE,
        )
        return success_with_capped_dict(
            "leave_request", _record_payload(result, "created leave request")
        )
    except Exception as e:
        logger.error(f"Error creating Employment Hero leave request: {e}")
        return _error(str(e))


@mcp.tool(annotations=UPDATE)
def employment_hero_approve_leave_request(
    organisation_id: str,
    employee_id: str,
    leave_request_id: str,
) -> str:
    """
    Approve an employee's existing leave request. This changes its workflow
    state in Employment Hero; confirm the employee, request id, dates, and
    requested amount with the user before calling.
    organisation_id: id from employment_hero_list_organisations.
    employee_id: employee id on the leave request.
    leave_request_id: id from employment_hero_list_leave_requests.
    """
    try:
        safe_org_id = url_path_id(organisation_id, "organisation_id")
        safe_employee_id = url_path_id(employee_id, "employee_id")
        safe_request_id = url_path_id(leave_request_id, "leave_request_id")
        result = _request(
            "POST",
            f"/organisations/{safe_org_id}/employees/{safe_employee_id}"
            f"/leave_requests/{safe_request_id}/approve",
            required_scope=LEAVE_REQUESTS_APPROVE_SCOPE,
        )
        return success_with_capped_dict(
            "leave_request", _record_payload(result, "approved leave request")
        )
    except Exception as e:
        logger.error(f"Error approving Employment Hero leave request: {e}")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
