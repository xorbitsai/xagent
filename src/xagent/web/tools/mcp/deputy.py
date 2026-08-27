import json
import logging
import os
from typing import Any
from urllib.parse import urlparse

import requests
from mcp.server.fastmcp import FastMCP

from ....config import get_tool_max_output_length
from .utils import setup_proxy_env, success_with_capped_dict, url_path_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deputy-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("deputy-mcp")

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
        if not isinstance(result, dict):
            return _error("Deputy returned an unexpected response for /me")
        return success_with_capped_dict("user", result)
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
        if result is None:
            # A JSON `null` body is a common REST idiom for "no records" (an
            # ASP.NET-style API serializing a null collection reference
            # rather than an empty array) -- treated the same as [], not as
            # an unexpected shape.
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
        if not isinstance(result, dict):
            return _error(f"Deputy returned an unexpected response for {resource}")
        return success_with_capped_dict("record", result)
    except Exception as e:
        logger.error(
            f"Error fetching Deputy {resource} record {resource_id}: {e}",
            exc_info=True,
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
        if result is None:
            # See deputy_list_resource's identical null-to-empty-list check.
            result = []
        if not isinstance(result, list):
            # See deputy_list_resource's identical check: distinct from
            # "genuinely zero records" (a valid, common response).
            return _error(f"Deputy returned an unexpected response for {resource}")
        return _success_with_capped_list("records", result)
    except Exception as e:
        logger.error(f"Error querying Deputy resource {resource}: {e}", exc_info=True)
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
