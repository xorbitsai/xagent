import json
import logging
import os
from typing import Any
from urllib.parse import urlparse

import requests
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from ....config import get_tool_max_output_length
from .utils import setup_proxy_env, success_with_capped_dict, url_path_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deputy-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("deputy-mcp")

# This connector wraps Deputy's V1 "Resource API" (/api/v1/resource/*),
# which Deputy's own docs mark as "in maintenance mode -- new integrations
# should prefer the V2 API" (developer.deputy.com/reference/searchemployee-1
# and sibling pages). V2 (developer.deputy.com/docs/new-employee-api-beta)
# only covers Employee management today, with no Roster/Timesheet/Leave
# equivalent yet -- V1 is the only way to cover this connector's actual
# scope (rosters, timesheets, leave, and generic resource querying), so
# this is a deliberate choice, not an oversight. Revisit if V2 gains
# coverage for the object types this connector needs.
DEFAULT_TIMEOUT_SECONDS = 30
# Matches zoom.py's/salesforce.py's convention: an error body that isn't
# the expected shape (e.g. an HTML gateway error page) must not be
# forwarded to the LLM/logs verbatim and unbounded.
MAX_ERROR_RESPONSE_TEXT_CHARS = 1000

_INSTANCE_URL_HOST_SUFFIX = "deputy.com"


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _record_response(context: str, result: Any, *, field_name: str = "record") -> str:
    """Wrap a single dict-shaped Deputy API response, or an error if it
    isn't one. Shared by every tool that expects exactly one record back
    (deputy_get_current_user's "/me", deputy_get_resource's/
    deputy_create_resource's/deputy_update_resource's {resource}/
    deputy_add_employee's "Employee"), which otherwise repeat this same
    isinstance-check-then-cap shape verbatim.

    ``context`` is the resource name (or "/me") to name in the error
    message on a malformed response -- not necessarily the same string as
    ``field_name``, which is the JSON key the record is nested under on
    success.
    """
    if not isinstance(result, dict):
        return _error(f"Deputy returned an unexpected response for {context}")
    return success_with_capped_dict(field_name, result)


def _empty_create_response(subject: str) -> str:
    """Shared by every create tool (deputy_create_resource,
    deputy_add_employee): an empty ``{}`` (a 204, or a 200 with no body --
    both normalized by ``_request``) after an otherwise-successful POST is
    a genuine "we don't know" for a create -- unlike a get/list, an empty
    response here isn't a valid "no data" result, but it's also not
    necessarily a failure, since Deputy already returned a non-error
    status. A flat ``_error`` here would be misleading (it reads as
    "nothing happened, safe to retry"), which is exactly the wrong signal
    for a non-idempotent write -- so this stays a success, with an
    explicit warning instead of a silent, confident-looking blank record.
    """
    return _success(
        record={},
        warning=(
            f"Deputy returned no content for this {subject} create -- the "
            "record may or may not have been created, and its id is "
            "unknown. Use deputy_query_resource to check before retrying."
        ),
    )


def _success_with_capped_list(
    list_field: str, items: list[Any], *, truncated: bool = False, **extra: Any
) -> str:
    """Build a success payload, halving ``items`` until the response fits
    the platform's output limit.

    A resource list/query result can serialize past the output filter's
    fixed character threshold and get hard-truncated into broken JSON.
    Halving (rather than a fixed slice) adapts to whatever size a given
    install's records happen to have, and continues down to zero items: a
    single oversized item must still be capped, not returned whole because
    there's nothing left to halve away from it. Matches
    salesforce.py's _success_with_capped_list -- neither
    deputy_list_resource nor deputy_query_resource exposes a cursor/offset
    the caller could retry with, so items dropped here are gone for this
    call, not just this page; when halving actually ran, the message says
    so plainly instead of leaving a bare ``truncated: true`` to imply a
    retry would help.
    """

    def _build(items: list[Any], truncated: bool, halved: bool) -> str:
        payload = {list_field: items, "truncated": truncated, **extra}
        if halved:
            payload["message"] = (
                f"Returned {len(items)} {list_field} out of the full result; "
                "the rest did not fit the output size limit and cannot be "
                "recovered via this tool call."
            )
        return _success(**payload)

    max_output_length = get_tool_max_output_length()
    halved = False
    response = _build(items, truncated, halved)
    while len(response) > max_output_length and items:
        items = items[: len(items) // 2]
        truncated = True
        halved = True
        response = _build(items, truncated, halved)
    if halved and len(response) > max_output_length:
        response = _build(items, truncated, False)
    return response


def _instance_url() -> str:
    """Return the per-install API origin this connector's OAuth grant
    belongs to.

    Deputy returns this in the token response (as ``endpoint``, normalized
    to a full origin by api/auth.py) instead of using a fixed API domain --
    which install to call is a property of the connected account, not
    something this module can infer. Canonicalizes to exactly
    ``scheme://host[:port]`` rather than returning the input string as-is:
    the value is used as a raw prefix (``f"{_instance_url()}{path}"``), so
    a value like "https://acme.au.deputy.com/evil/path" would otherwise
    pass a shallower check and then silently carry its extra path
    component into every outbound request URL.
    """
    instance_url = os.environ.get("DEPUTY_INSTANCE_URL")
    if not instance_url:
        raise ValueError("DEPUTY_INSTANCE_URL environment variable is missing")
    invalid = ValueError(
        f"DEPUTY_INSTANCE_URL is not a valid Deputy host: {instance_url!r}"
    )
    try:
        # urlparse() itself, not just the .port access below, can raise
        # ValueError on malformed input (e.g. an IPv6-literal-like host:
        # urlparse("https://[::1].deputy.com") raises "Invalid IPv6 URL")
        # -- both calls share this one try/except and raise the same
        # `invalid` error, rather than urlparse's cryptic one (or, for
        # .port, "Port could not be cast to integer value") escaping
        # uncaught. `parsed` is only ever used below once this block has
        # completed without raising, so mypy needs no Optional handling
        # for it.
        parsed = urlparse(instance_url.rstrip("/"))
        port = f":{parsed.port}" if parsed.port else ""
    except ValueError:
        raise invalid from None
    # rstrip: a trailing-dot FQDN (e.g. "acme.au.deputy.com.") is a valid,
    # equivalent hostname that just wouldn't satisfy endswith below
    # otherwise -- Deputy's own token response never sends one, but
    # there's no reason to reject it if it ever did.
    hostname = (parsed.hostname or "").rstrip(".")
    if parsed.scheme != "https" or not (
        hostname == _INSTANCE_URL_HOST_SUFFIX
        or hostname.endswith(f".{_INSTANCE_URL_HOST_SUFFIX}")
    ):
        raise invalid
    return f"{parsed.scheme}://{hostname}{port}"


def _headers() -> dict[str, str]:
    # Stripped, not just a bare os.environ.get(): a stray leading/trailing
    # newline or space in the injected token (e.g. from a copy-pasted env
    # value in a manual/local launch_config override) would otherwise
    # silently produce a malformed Authorization header rather than the
    # clear "missing" error below.
    access_token = (os.environ.get("DEPUTY_ACCESS_TOKEN") or "").strip()
    if not access_token:
        raise ValueError("DEPUTY_ACCESS_TOKEN environment variable is missing or empty")
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }


def _extract_error_detail(response: requests.Response) -> str | None:
    """Pull a human-readable message out of a Deputy error body, trying a
    few plausible key names rather than assuming one fixed shape. Returns
    None if the body isn't in any of the expected shapes, so the caller
    falls back to the raw response text.
    """
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    for key in ("error_description", "error", "Message", "message"):
        detail = payload.get(key)
        if isinstance(detail, str) and detail:
            return detail
    return None


def _request(
    method: str,
    path: str,
    *,
    json_data: Any = None,
) -> Any:
    response = requests.request(
        method=method,
        url=f"{_instance_url()}/api/v1{path}",
        headers=_headers(),
        json=json_data,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    if response.status_code >= 400:
        detail = _extract_error_detail(response)
        if detail is None:
            detail = response.text.strip()
        if len(detail) > MAX_ERROR_RESPONSE_TEXT_CHARS:
            detail = detail[:MAX_ERROR_RESPONSE_TEXT_CHARS] + "... [truncated]"
        raise RuntimeError(
            f"Deputy API error (status {response.status_code})"
            + (f": {detail}" if detail else "")
        )
    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


@mcp.tool()
def deputy_get_current_user() -> str:
    """
    Get the profile of the Deputy employee this connector is authenticated
    as (via GET /me). Use this for "my account" / "who am I" requests
    instead of asking the user for their Deputy employee id.
    """
    try:
        result = _request("GET", "/me")
        return _record_response("/me", result, field_name="user")
    except Exception as e:
        logger.error(f"Error fetching authenticated Deputy user: {e}", exc_info=True)
        return _error(str(e))


@mcp.tool()
def deputy_list_resource(resource: str) -> str:
    """
    List every record of a Deputy resource type (GET /resource/{resource}).
    resource: a Deputy Resource API object name, e.g. "Employee", "Roster",
    "Timesheet", "Leave", "Company", or "OperationalUnit". Use
    deputy_query_resource instead when you need to filter, sort, or join --
    this tool returns the install's full unfiltered list, which Deputy caps
    server-side and this tool may additionally truncate if the response is
    too large to return in one call.
    """
    try:
        safe_resource = url_path_id(resource, "resource")
        result = _request("GET", f"/resource/{safe_resource}")
        if result is None or result == {}:
            # A JSON `null` body is a common REST idiom for "no records" (an
            # ASP.NET-style API serializing a null collection reference
            # rather than an empty array) -- treated the same as [], not as
            # an unexpected shape. `{}` is included too: _request() itself
            # normalizes a 204 or empty-content 200 response (Deputy's own
            # backend already serializes "no records" inconsistently, per
            # the null case above) to `{}`, not `[]` or `None` -- that
            # normalization is shared with deputy_get_resource/
            # deputy_get_current_user, where `{}` is instead a genuinely
            # valid dict result, so it can't be changed at the source
            # without breaking those two.
            result = []
        if not isinstance(result, list):
            # Distinct from "genuinely zero records" (an empty list, or
            # null, is a valid, common response) -- coercing any OTHER
            # unexpected shape to [] here would make a malformed Deputy
            # response indistinguishable from a real empty result, unlike
            # deputy_get_resource/deputy_get_current_user, which both
            # already error on an unexpected shape rather than guessing.
            return _error(f"Deputy returned an unexpected response for {resource}")
        return _success_with_capped_list("records", result)
    except Exception as e:
        logger.error(f"Error listing Deputy resource {resource}: {e}", exc_info=True)
        return _error(str(e))


@mcp.tool()
def deputy_get_resource(resource: str, resource_id: str) -> str:
    """
    Get one record by id (GET /resource/{resource}/{id}).
    resource: a Deputy Resource API object name, e.g. "Employee", "Roster",
    "Timesheet", or "Leave".
    resource_id: the record's numeric id, as a string (e.g. "123").
    """
    try:
        safe_resource = url_path_id(resource, "resource")
        safe_resource_id = url_path_id(resource_id, "resource_id")
        result = _request("GET", f"/resource/{safe_resource}/{safe_resource_id}")
        return _record_response(resource, result)
    except Exception as e:
        logger.error(
            f"Error fetching Deputy {resource} record {resource_id}: {e}",
            exc_info=True,
        )
        return _error(str(e))


@mcp.tool()
def deputy_resource_info(resource: str) -> str:
    """
    Get the field list, types, and associations Deputy defines for a
    resource type (GET /resource/{resource}/INFO). Use this before
    deputy_create_resource/deputy_update_resource to learn what fields
    Deputy expects -- especially for the first record of a type, when
    there's no existing record yet for deputy_get_resource/
    deputy_list_resource/deputy_query_resource to show you.
    resource: a Deputy Resource API object name, e.g. "Employee", "Roster",
    "Timesheet", or "Leave".
    """
    try:
        safe_resource = url_path_id(resource, "resource")
        result = _request("GET", f"/resource/{safe_resource}/INFO")
        return _record_response(resource, result, field_name="info")
    except Exception as e:
        logger.error(
            f"Error fetching Deputy resource info for {resource}: {e}", exc_info=True
        )
        return _error(str(e))


@mcp.tool()
def deputy_query_resource(
    resource: str,
    search: dict[str, Any] | None = None,
    sort: dict[str, Any] | None = None,
    join: list[str] | None = None,
) -> str:
    """
    Run a filtered query against a Deputy resource type
    (POST /resource/{resource}/QUERY) -- the primary way to look up rosters/
    shifts, timesheets, or leave within a date range, for a specific
    employee, or matching any other field. Deputy caps this endpoint at 500
    records per response server-side.

    resource: a Deputy Resource API object name, e.g. "Employee", "Roster",
    "Timesheet", or "Leave".
    search: Deputy's search filter object, keyed by arbitrary condition
    names (e.g. "s1", "s2", ...), each an object with "field", "data", and
    "type" (a comparison operator, e.g. "eq", "gt", "lt", "ge", "le", "in").
    Example: {"s1": {"field": "Date", "data": "2026-08-01", "type": "gt"}}.
    sort: optional field-name-to-direction map, e.g. {"Id": "asc"}.
    join: optional list of related object names to include in each result,
    e.g. ["TimesheetObject"].
    """
    try:
        safe_resource = url_path_id(resource, "resource")
        body: dict[str, Any] = {}
        if search:
            body["search"] = search
        if sort:
            body["sort"] = sort
        if join:
            body["join"] = join
        result = _request("POST", f"/resource/{safe_resource}/QUERY", json_data=body)
        if result is None or result == {}:
            # See deputy_list_resource's identical null/empty-to-empty-list
            # check (also covers _request()'s 204/empty-content -> {}
            # normalization).
            result = []
        if not isinstance(result, list):
            # See deputy_list_resource's identical check: distinct from
            # "genuinely zero records" (a valid, common response).
            return _error(f"Deputy returned an unexpected response for {resource}")
        return _success_with_capped_list("records", result)
    except Exception as e:
        logger.error(f"Error querying Deputy resource {resource}: {e}", exc_info=True)
        return _error(str(e))


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False, idempotentHint=False))
def deputy_create_resource(resource: str, data: dict[str, Any]) -> str:
    """
    Create a new record (POST /resource/{resource}).
    resource: a Deputy Resource API object name, e.g. "Roster", "Timesheet",
    "Leave", or "Contact" -- NOT "Employee": use deputy_add_employee for
    that instead, which this tool refuses (see its docstring for why).
    data: field name -> value pairs for the new record. Use
    deputy_resource_info first to learn Deputy's required/optional fields
    for a resource type -- deputy_get_resource needs an id, which isn't
    available yet when creating the first record of a type.
    This is not idempotent: retrying after a timeout or connection error
    can create a duplicate record. Use deputy_query_resource to check
    whether the record already exists before retrying a failed call.
    """
    try:
        # Case/whitespace-insensitive: Deputy's own resource-name routing
        # is not guaranteed to be case-sensitive, and an LLM caller
        # guessing "employee"/"EMPLOYEE" instead of the exact string
        # "Employee" must not bypass this guard and reproduce the exact
        # incident it exists to prevent.
        if resource.strip().casefold() == "employee":
            # A bare POST /resource/Employee inserts an Employee row with
            # no location/workplace membership -- Deputy's permission
            # model is scoped by location, so the resulting record is then
            # neither listable nor directly readable (a 403 "Access to
            # object denied", not a 404) even though Deputy accepted the
            # write and returned a plausible-looking record with an id.
            # That's indistinguishable from "nothing was created" to both
            # the caller and Deputy's own UI (root-caused from a
            # production incident on 2026-09-21, where this produced two
            # such orphaned records). deputy_add_employee wraps Deputy's
            # own recommended management/supervise endpoint instead, which
            # sets up that membership as part of the same call.
            return _error(
                "Employee creation isn't supported here -- use "
                "deputy_add_employee instead. A bare POST /resource/Employee "
                "creates a record with no location/workplace membership, "
                "which Deputy's permission model then hides from list/get "
                "calls even though the write itself succeeds."
            )
        # "Id" is server-assigned on create; dropping any caller-supplied
        # value rather than forwarding it removes any ambiguity about
        # whether Deputy would honor, ignore, or reject a client-chosen
        # id (undocumented), matching deputy_update_resource's own
        # protection of "Id" against caller override. Stripped before the
        # empty-data check so a data={"Id": ...}-only call is correctly
        # rejected as no real data provided, not silently posted empty.
        create_data = {k: v for k, v in data.items() if k != "Id"}
        if not create_data:
            return _error("No data provided to create record")
        safe_resource = url_path_id(resource, "resource")
        result = _request("POST", f"/resource/{safe_resource}", json_data=create_data)
        if isinstance(result, dict) and not result:
            return _empty_create_response(resource)
        return _record_response(resource, result)
    except Exception as e:
        logger.error(f"Error creating Deputy {resource} record: {e}", exc_info=True)
        return _error(str(e))


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False, idempotentHint=False))
def deputy_add_employee(
    first_name: str,
    last_name: str,
    company_id: int,
    *,
    email: str | None = None,
    mobile_phone: str | None = None,
    role_id: int | None = None,
    stress_profile_id: int | None = None,
    start_date: str | None = None,
    date_of_birth: str | None = None,
    gender: int | None = None,
    country_code: str | None = None,
    payroll_id: str | None = None,
    weekday_rate: float | None = None,
    send_invite: bool = False,
) -> str:
    """
    Add a new employee (POST /supervise/employee) -- Deputy's own docs
    recommend this management endpoint for adding employees, not the
    generic Resource API. deputy_create_resource refuses resource=
    "Employee" for exactly this reason: a bare POST /resource/Employee
    inserts a row with no location/workplace membership, and Deputy's
    permission model is scoped by location, so the record gets created but
    is then neither listable nor directly readable -- indistinguishable
    from "nothing was created" to both the caller and Deputy's own UI.

    first_name, last_name: the employee's name.
    company_id: the Deputy Company (location) id to add the employee to --
    look one up first (e.g. via deputy_list_resource("Company")) rather
    than guessing.
    email: primary email address, if any.
    mobile_phone: mobile phone number, if any.
    role_id: the EmployeeRole (access level) id to grant -- look one up
    via deputy_list_resource("EmployeeRole") if unsure.
    stress_profile_id: the working-hours/StressProfile id to assign --
    look one up via deputy_list_resource("StressProfile") if unsure.
    start_date, date_of_birth: "YYYY-MM-DD".
    gender: 0 = prefer not to say, 1 = male, 2 = female, 3 = non-binary.
    country_code: country code for the employee's address.
    payroll_id: external payroll id to set.
    weekday_rate: weekday pay rate.
    send_invite: if true, Deputy emails the employee an invitation to set
    up their own Deputy login. Defaults to false -- only set true once
    the caller has confirmed this employee should get app access.

    This endpoint has no field for AllowAppraisal (or any Employee field
    outside the ones listed above) -- to set those, follow up with
    deputy_update_resource("Employee", <new id>, {...}) after this call.

    Deputy's official field reference for this endpoint
    (developer.deputy.com/reference/addanemployee) does not document
    email/role/stress-profile/invite fields at all; the names used here
    (strEmail, intRoleId, intStressProfile, blnSendInvite) come from a
    separate example in Deputy's own docs that is not confirmed against a
    live account. If a create reports success but email/role/stress
    profile/invite don't show up on the resulting record, Deputy is
    silently ignoring that field -- verify with deputy_get_resource
    afterward rather than trusting this call's reported success alone.

    This is not idempotent: retrying after a timeout or connection error
    can create a duplicate record. Use deputy_query_resource("Employee",
    ...) to check whether the record already exists before retrying a
    failed call.
    """
    try:
        # Flattened, individually-named parameters rather than a generic
        # data: dict[str, Any] (unlike deputy_create_resource): Deputy's
        # supervise/employee field names (strFirstName, intCompanyId,
        # intStressProfile, ...) don't match the Resource API's Employee
        # field names (FirstName, Company, StressProfile), and the
        # incident this tool fixes was partly caused by the caller
        # guessing at a field name ("Stress Profile" with a space) that
        # silently didn't match what Deputy expected. Typed keyword
        # arguments make that whole class of mistake impossible.
        body: dict[str, Any] = {
            "strFirstName": first_name,
            "strLastName": last_name,
            "intCompanyId": company_id,
            # Sent explicitly either way, rather than omitted when False,
            # so this call's behavior doesn't depend on Deputy's own
            # undocumented default for an absent field.
            "blnSendInvite": 1 if send_invite else 0,
        }
        if email is not None:
            body["strEmail"] = email
        if mobile_phone is not None:
            body["strMobilePhone"] = mobile_phone
        if role_id is not None:
            body["intRoleId"] = role_id
        if stress_profile_id is not None:
            body["intStressProfile"] = stress_profile_id
        if start_date is not None:
            body["strStartDate"] = start_date
        if date_of_birth is not None:
            body["strDob"] = date_of_birth
        if gender is not None:
            body["intGender"] = gender
        if country_code is not None:
            body["strCountryCode"] = country_code
        if payroll_id is not None:
            body["strPayrollId"] = payroll_id
        if weekday_rate is not None:
            body["fltWeekDayRate"] = weekday_rate
        result = _request("POST", "/supervise/employee", json_data=body)
        if isinstance(result, dict) and not result:
            return _empty_create_response("employee")
        return _record_response("Employee", result)
    except Exception as e:
        logger.error(f"Error adding Deputy employee: {e}", exc_info=True)
        return _error(str(e))


@mcp.tool(annotations=ToolAnnotations(destructiveHint=True, idempotentHint=True))
def deputy_update_resource(
    resource: str, resource_id: str, data: dict[str, Any]
) -> str:
    """
    Update an existing record. Only the fields provided are changed --
    internally this reads the current record and writes the merged
    result back, since Deputy has no partial-update support here (see
    deputy_get_resource first if you need to know a field's current
    value). A concurrent edit to the same record landing while this
    runs (elsewhere, or another call) can be silently overwritten;
    there is no conflict detection or retry.
    resource: a Deputy Resource API object name, e.g. "Employee", "Roster",
    "Timesheet", or "Leave".
    resource_id: the record's numeric id, as a string (e.g. "123").
    data: field name -> value pairs to change, e.g. {"Active": False} to
    deactivate an Employee. For "Employee", fields like "Company",
    "Contact", "Role", and "StressProfile" are ids referencing other
    records, not literal values -- look up a valid id first (e.g. via
    deputy_list_resource) rather than guessing one.
    """
    try:
        # "Id" is identified by resource_id/the URL, not data: the URL's
        # resource_id is what actually identifies the record being
        # written, and letting a caller-supplied "Id" in data silently
        # ride along in the body would decouple what the body claims to
        # be from what the URL targets, for no legitimate reason a
        # partial update would ever need. Dropped unconditionally rather
        # than only re-asserted from the fetched record (which would
        # leave a caller-supplied Id unprotected whenever the fetched
        # record happens to lack an "Id" key) -- matches
        # deputy_create_resource's own protection. Stripped before the
        # empty-data check so a data={"Id": ...}-only call is correctly
        # rejected as no real change requested, not turned into a
        # needless no-op GET+POST round trip.
        update_data = {k: v for k, v in data.items() if k != "Id"}
        if not update_data:
            return _error("No data provided to update")
        safe_resource = url_path_id(resource, "resource")
        safe_resource_id = url_path_id(resource_id, "resource_id")
        # Deputy's V1 Resource API requires the complete object on
        # POST /resource/{resource}/{id} -- it has no partial-update
        # support (Deputy's own V2 employee endpoint exists specifically
        # to add that for Employee; V1 has no equivalent for other
        # resource types) -- so fetch the current record and merge
        # `update_data` into it before writing the full object back,
        # rather than sending `update_data` alone. This is a GET-then-POST
        # with no retry: a concurrent edit landing between the two is a
        # lost-update race this function doesn't detect or retry on.
        # Not unique to this connector (myob.py's own generic full-object
        # update has the same tradeoff), so left as a known limitation
        # rather than solved here.
        current = _request("GET", f"/resource/{safe_resource}/{safe_resource_id}")
        # Both checks needed, not just one: `not current` alone lets a
        # truthy non-dict (e.g. a bare list) through to the dict-spread
        # below; `isinstance` alone lets an empty {} through -- _request
        # returns {} on a 204/empty response, and {**{}, **update_data}
        # would otherwise silently POST only the caller's partial fields
        # as if they were the whole record, wiping every other field
        # Deputy has for it. Matches myob.py's _update_resource, which
        # guards the same way for the same reason.
        if not current or not isinstance(current, dict):
            return _error(
                f"Deputy returned no existing record to update for {resource}"
                f" {resource_id}"
            )
        merged = {**current, **update_data}
        result = _request(
            "POST",
            f"/resource/{safe_resource}/{safe_resource_id}",
            json_data=merged,
        )
        return _record_response(resource, result)
    except Exception as e:
        logger.error(
            f"Error updating Deputy {resource} record {resource_id}: {e}",
            exc_info=True,
        )
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
