import json
import logging
import os
from typing import Any
from urllib.parse import urlparse

import requests
from mcp.server.fastmcp import FastMCP

from ....config import get_tool_max_output_length
from .utils import (
    clamp_limit,
    clamp_offset,
    setup_proxy_env,
    success_with_capped_dict,
    url_path_id,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("salesforce-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("salesforce-mcp")

# The OIDC userinfo endpoint is documented as this fixed host, not the
# per-org instance_url used by every other endpoint here -- Salesforce
# routes the request to the correct org internally based on the token.
USERINFO_URL = "https://login.salesforce.com/services/oauth2/userinfo"

# Pinned to a long-supported version rather than the latest release --
# Salesforce supports past API versions for many years (not indefinitely;
# old versions are eventually retired), and the SOQL/sobject CRUD surface
# this connector uses has been stable across versions for years.
API_VERSION = "v59.0"
DEFAULT_TIMEOUT_SECONDS = 30
# Matches zoom.py's/linear.py's convention: an error body that isn't the
# expected shape (e.g. an HTML gateway error page) must not be forwarded to
# the LLM/logs verbatim and unbounded.
MAX_ERROR_RESPONSE_TEXT_CHARS = 1000
# Default and max page size for salesforce_list_sobjects/
# salesforce_describe_sobject's offset/limit pagination -- neither endpoint
# supports server-side paging, so this is a client-side window over the full
# list/describe response. DEFAULT_PAGE_LIMIT is sized to comfortably fit a
# page of salesforce_list_sobjects' small, fixed-shape summaries under the
# default output limit without needing _success_with_capped_page's halving
# fallback -- salesforce_describe_sobject's per-field metadata (unbounded
# picklist_values in particular) can still be large enough per item to hit
# that fallback at this same default, which is fine: the fallback exists
# for exactly that case. MAX_PAGE_LIMIT bounds how large a page a caller can
# request in one call -- matches posthog.py's MAX_LIMIT convention for the
# same reason: an unbounded limit would let one call demand the entire
# result set back, defeating the point of paging at all.
DEFAULT_PAGE_LIMIT = 50
MAX_PAGE_LIMIT = 200


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _success_with_capped_list(
    list_field: str, items: list[Any], *, truncated: bool = False, **extra: Any
) -> str:
    """Build a success payload, halving ``items`` until the response fits
    the platform's output limit.

    A SOQL result page or search hit list can each serialize past the
    output filter's fixed character threshold and get hard-truncated into
    broken JSON -- the same failure mode hubspot.py's
    _paged_list/_success_with_capped_dict exist to avoid. Halving (rather
    than a fixed slice) adapts to whatever size a given org's records
    happen to have, and continues down to zero items: a single oversized
    item must still be capped, not returned whole because there's nothing
    left to halve away from it. ``truncated`` seeds from any
    upstream-reported truncation (e.g. Salesforce's own SOQL ``done``
    flag) and is OR'd with whatever local capping adds.

    Only salesforce_query/salesforce_search call this now, and neither
    exposes a cursor/offset the caller could retry with to recover items
    this halving drops (unlike Salesforce query's own separate,
    still-unimplemented nextRecordsUrl pagination, tracked in #1541) --
    items dropped here are gone for this call, not just this page. When
    halving actually ran, the message says so plainly instead of leaving a
    bare ``truncated: true`` to imply a retry would help.
    salesforce_list_sobjects/salesforce_describe_sobject use
    _success_with_capped_page below instead, which does expose one.
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
        # Rebuilding with halved=True (message included) from the first
        # halving iteration onward, not just once at the end, means the
        # size check on the *next* loop condition already accounts for the
        # message's own weight -- appending it only after the loop
        # converged would let it silently push an already-fitted response
        # back over the limit, re-creating the exact invalid-JSON risk this
        # function exists to prevent.
        response = _build(items, truncated, halved)
    if halved and len(response) > max_output_length:
        # Halving already emptied ``items`` and the message text itself is
        # what's still pushing the response over the limit (an operator can
        # configure XAGENT_TOOL_MAX_OUTPUT_LENGTH arbitrarily small) --
        # there's nothing left to halve away, so drop the message rather
        # than return a payload that violates the caller's own size
        # contract.
        response = _build(items, truncated, False)
    return response


def _success_with_capped_page(
    list_field: str, page: list[Any], *, offset: int, total_count: int, **extra: Any
) -> str:
    """Build a paginated success payload for an already-sliced ``page``,
    halving it further only as a last resort if it still doesn't fit the
    platform's output limit.

    Unlike _success_with_capped_list, nothing returned by this function is
    ever unrecoverable: ``next_offset`` always reflects how many items
    actually made it into the response, not how many the caller's
    offset/limit window requested. So whether the caller simply hasn't
    reached the end of the result yet, or this call's own halving had to
    shrink the page further to fit the output limit, the same
    has_more/next_offset pair tells it exactly where to resume -- there is
    no separate "cannot be recovered" case to report.

    One edge case still needs a smaller ``limit``, not a bigger ``offset``:
    if a single item is itself too large to fit (e.g. one field with an
    enormous picklist), halving can empty ``returned`` entirely, and
    ``next_offset`` then equals the same ``offset`` the caller just
    passed in. Retrying with that unchanged offset only reproduces the
    same result -- the caller must lower ``limit`` instead so this
    function has more than one oversized item's worth of room to shrink
    within. When that happens, an explicit ``message`` says so plainly,
    the same reasoning _success_with_capped_list applies to its own
    halved-to-empty case.

    If even an empty page's fixed envelope (offset/has_more/next_offset/
    **extra, all load-bearing for the pagination contract, so none of
    them can be dropped the way a message can) still doesn't fit,
    ``total_count`` -- informational only, not required for the caller to
    make progress -- is dropped as a last resort, then the explanatory
    ``message`` itself if the envelope still doesn't fit without it.
    """

    def _build(
        returned: list[Any],
        *,
        include_total_count: bool = True,
        include_message: bool = True,
    ) -> str:
        next_offset = offset + len(returned)
        has_more = next_offset < total_count
        payload: dict[str, Any] = {
            list_field: returned,
            "offset": offset,
            "has_more": has_more,
            **extra,
        }
        if include_total_count:
            payload["total_count"] = total_count
        if has_more:
            payload["next_offset"] = next_offset
        if include_message and has_more and not returned:
            payload["message"] = (
                f"A single {list_field} item did not fit the output size "
                f"limit on its own; retry with a smaller limit at the same "
                f"offset ({offset}) rather than following next_offset."
            )
        return _success(**payload)

    max_output_length = get_tool_max_output_length()
    returned = page
    response = _build(returned)
    while len(response) > max_output_length and returned:
        returned = returned[: len(returned) // 2]
        response = _build(returned)
    if len(response) > max_output_length:
        response = _build(returned, include_total_count=False)
    if len(response) > max_output_length:
        response = _build(returned, include_total_count=False, include_message=False)
    return response


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


_INSTANCE_URL_HOST_SUFFIXES = ("salesforce.com",)


def _instance_url() -> str:
    """Return the per-org API host this connector's OAuth grant belongs to.

    Salesforce returns this in the token response instead of using a fixed
    API domain -- which org to call is a property of the connected account,
    not something this module can infer. Salesforce is the only connector
    in this codebase where the entire outbound API origin, not just a path
    segment, comes from provider-persisted data rather than a hardcoded
    constant, so this validates scheme+host rather than only checking
    non-empty: every request this module makes is built by interpolating
    this value directly into a URL.

    Canonicalizes to exactly ``scheme://host[:port]`` rather than returning
    the input string as-is: the value is used as a raw prefix
    (``f"{_instance_url()}{path}"``), so a value like
    "https://acme.my.salesforce.com/evil/path" or
    "https://user:pw@acme.my.salesforce.com" would otherwise pass the
    scheme+host check and then silently carry its extra path/userinfo
    component into every outbound request URL. force.com (Salesforce Sites
    / Experience Cloud, which can serve customer-authored content) is
    deliberately not in the allowed suffixes -- the OAuth token endpoint's
    instance_url is always a *.salesforce.com host in practice, and
    force.com would only widen this beyond what Salesforce actually sends.
    """
    instance_url = os.environ.get("SALESFORCE_INSTANCE_URL")
    if not instance_url:
        raise ValueError("SALESFORCE_INSTANCE_URL environment variable is missing")
    parsed = urlparse(instance_url.rstrip("/"))
    # rstrip: a trailing-dot FQDN (e.g. "acme.my.salesforce.com.") is a
    # valid, equivalent hostname that just wouldn't satisfy endswith below
    # otherwise -- Salesforce's own token response never sends one, but
    # there's no reason to reject it if it ever did.
    hostname = (parsed.hostname or "").rstrip(".")
    try:
        # .port is a lazy property that raises ValueError for a
        # non-numeric port (e.g. "...salesforce.com:abc") -- accessed here,
        # before the scheme/host check below, so that case raises this
        # function's own clear message instead of urlparse's cryptic
        # "Port could not be cast to integer value" escaping uncaught.
        port = f":{parsed.port}" if parsed.port else ""
    except ValueError:
        port = None
    if (
        port is None
        or parsed.scheme != "https"
        or not any(
            hostname == suffix or hostname.endswith(f".{suffix}")
            for suffix in _INSTANCE_URL_HOST_SUFFIXES
        )
    ):
        raise ValueError(
            f"SALESFORCE_INSTANCE_URL is not a valid Salesforce host: {instance_url!r}"
        )
    return f"{parsed.scheme}://{hostname}{port}"


def _headers() -> dict[str, str]:
    access_token = os.environ.get("SALESFORCE_ACCESS_TOKEN")
    if not access_token:
        raise ValueError("SALESFORCE_ACCESS_TOKEN environment variable is missing")
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }


def _extract_error_detail(response: requests.Response) -> str | None:
    """Pull the human-readable message out of a Salesforce error body.

    Salesforce REST API errors are a top-level JSON *array* of
    {"message", "errorCode", ...} objects (unlike most APIs here, which wrap
    errors in a dict) -- joining every message is more useful to the LLM
    than the raw envelope. Returns None if the body isn't in the expected
    shape, so the caller can fall back to the raw response text.
    """
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, list) or not payload:
        return None

    # A falsy (missing or empty-string) message falls back to errorCode --
    # still readable text, unlike str(item)'s Python dict-repr, which is
    # only used as a last resort when neither key has anything useful.
    messages = [
        str(item.get("message") or item.get("errorCode") or item)
        if isinstance(item, dict)
        else str(item)
        for item in payload
    ]
    return "; ".join(messages)


def _request_absolute(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    json_data: dict[str, Any] | None = None,
) -> Any:
    response = requests.request(
        method=method,
        url=url,
        headers=_headers(),
        params=params,
        json=json_data,
        timeout=DEFAULT_TIMEOUT_SECONDS,
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
        raise RuntimeError(
            f"Salesforce API error (status {response.status_code}): {detail}"
        )

    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_data: dict[str, Any] | None = None,
) -> Any:
    return _request_absolute(
        method, f"{_instance_url()}{path}", params=params, json_data=json_data
    )


@mcp.tool()
def salesforce_get_current_user() -> str:
    """
    Get the profile of the Salesforce user this connector is authenticated
    as (user id, org id, name, email, username). Use this for "my account" /
    "who am I" requests instead of asking the user for their Salesforce
    user id.
    """
    try:
        result = _request_absolute("GET", USERINFO_URL)
        return _success(
            user={
                "user_id": result.get("user_id"),
                "organization_id": result.get("organization_id"),
                "name": result.get("name"),
                "email": result.get("email"),
                "preferred_username": result.get("preferred_username"),
            }
        )
    except Exception as e:
        logger.error(f"Error fetching authenticated Salesforce user: {e}")
        return _error(str(e))


@mcp.tool()
def salesforce_query(soql: str) -> str:
    """
    Run a SOQL (Salesforce Object Query Language) query -- the primary way
    to read records of any object type, standard or custom.
    soql: a SELECT query, e.g. "SELECT Id, Name, Industry FROM Account
    WHERE BillingCountry = 'USA' ORDER BY Name LIMIT 20".
    """
    try:
        result = _request(
            "GET", f"/services/data/{API_VERSION}/query", params={"q": soql}
        )
        return _success_with_capped_list(
            "records",
            result.get("records") or [],
            truncated=not result.get("done", True),
            total_size=result.get("totalSize"),
        )
    except Exception as e:
        logger.error(f"Error running Salesforce SOQL query: {e}")
        return _error(str(e))


@mcp.tool()
def salesforce_search(sosl: str) -> str:
    """
    Run a SOSL (Salesforce Object Search Language) search -- full-text
    search across multiple object types at once, unlike SOQL which queries
    one object at a time.
    sosl: a FIND query, e.g. "FIND {Acme} IN ALL FIELDS RETURNING
    Account(Id, Name), Contact(Id, Name, Email)".
    """
    try:
        result = _request(
            "GET", f"/services/data/{API_VERSION}/search", params={"q": sosl}
        )
        return _success_with_capped_list("results", result.get("searchRecords") or [])
    except Exception as e:
        logger.error(f"Error running Salesforce SOSL search: {e}")
        return _error(str(e))


@mcp.tool()
def salesforce_list_sobjects(
    name_contains: str = "",
    offset: int = 0,
    limit: int = DEFAULT_PAGE_LIMIT,
) -> str:
    """
    List the object types (standard, e.g. Account/Contact/Lead/Opportunity/
    Case, and custom, e.g. ending in "__c") this org exposes -- name, label,
    and whether it's queryable/creatable/updateable/deletable. Use the
    returned name with every other tool here.

    An org can expose hundreds of objects, more than fits one response.
    name_contains narrows the result before pagination for when you know
    roughly what you're looking for; offset/limit page through it either
    way -- check the response's has_more/next_offset to fetch the rest
    instead of assuming one call returns everything.

    name_contains: optional case-insensitive substring to match against
    each object's name or label (e.g. "invoice").
    offset: pagination offset into the (optionally filtered) result set;
    0 for the first page. Clamped to >= 0.
    limit: maximum objects to return in this call. Clamped to
    [1, 200].
    """
    try:
        offset = clamp_offset(offset)
        limit = clamp_limit(limit, max_limit=MAX_PAGE_LIMIT)
        result = _request("GET", f"/services/data/{API_VERSION}/sobjects")
        needle = name_contains.strip().lower()
        sobjects = [
            {
                "name": s.get("name"),
                "label": s.get("label"),
                "queryable": s.get("queryable"),
                "createable": s.get("createable"),
                "updateable": s.get("updateable"),
                "deletable": s.get("deletable"),
                "custom": s.get("custom"),
            }
            for s in result.get("sobjects") or []
            if not needle
            or needle in (s.get("name") or "").lower()
            or needle in (s.get("label") or "").lower()
        ]
        page = sobjects[offset : offset + limit]
        return _success_with_capped_page(
            "sobjects", page, offset=offset, total_count=len(sobjects)
        )
    except Exception as e:
        logger.error(f"Error listing Salesforce sobjects: {e}")
        return _error(str(e))


@mcp.tool()
def salesforce_describe_sobject(
    sobject_type: str,
    fields: list[str] | None = None,
    names_only: bool = False,
    offset: int = 0,
    limit: int = DEFAULT_PAGE_LIMIT,
) -> str:
    """
    Get an object type's field schema -- name, label, type, and whether
    each field is required/updateable/a picklist (with its valid values).
    Use this before salesforce_create_record/salesforce_update_record to
    learn which fields exist and what values they accept.

    A custom object can have far more fields than fit one response.
    names_only/fields narrow the result for when you know roughly what
    you're looking for; offset/limit page through it either way -- check
    the response's has_more/next_offset to fetch the rest instead of
    assuming one call returns everything.

    sobject_type: an object's API name, e.g. "Account" or "My_Object__c".
    fields: optional list of field API names to return full metadata for,
    skipping every other field. Takes precedence over names_only.
    names_only: if true and fields is not given, return just the list of
    field API names (no label/type/picklist metadata) instead of full
    per-field metadata -- a small, cheap response useful for discovering
    what to ask for next.
    offset: pagination offset into the (optionally filtered) field list;
    0 for the first page. Clamped to >= 0.
    limit: maximum fields to return in this call. Clamped to [1, 200].
    """
    try:
        offset = clamp_offset(offset)
        limit = clamp_limit(limit, max_limit=MAX_PAGE_LIMIT)
        safe_sobject_type = url_path_id(sobject_type, "sobject_type")
        result = _request(
            "GET",
            f"/services/data/{API_VERSION}/sobjects/{safe_sobject_type}/describe",
        )
        raw_fields = result.get("fields") or []

        if names_only and not fields:
            field_names = [f.get("name") for f in raw_fields]
            page = field_names[offset : offset + limit]
            return _success_with_capped_page(
                "fields",
                page,
                offset=offset,
                total_count=len(field_names),
                name=result.get("name"),
                label=result.get("label"),
            )

        wanted = set(fields) if fields else None
        described_fields = [
            {
                "name": f.get("name"),
                "label": f.get("label"),
                "type": f.get("type"),
                "nillable": f.get("nillable"),
                "createable": f.get("createable"),
                "updateable": f.get("updateable"),
                "picklist_values": (
                    [
                        pv.get("value")
                        for pv in f.get("picklistValues") or []
                        if pv.get("active")
                    ]
                    if f.get("type") == "picklist"
                    else None
                ),
            }
            for f in raw_fields
            if wanted is None or f.get("name") in wanted
        ]
        page = described_fields[offset : offset + limit]
        return _success_with_capped_page(
            "fields",
            page,
            offset=offset,
            total_count=len(described_fields),
            name=result.get("name"),
            label=result.get("label"),
        )
    except Exception as e:
        logger.error(f"Error describing Salesforce sobject {sobject_type}: {e}")
        return _error(str(e))


@mcp.tool()
def salesforce_get_record(sobject_type: str, record_id: str, fields: str = "") -> str:
    """
    Get one record's field values by id.
    sobject_type: an object's API name, e.g. "Account" or "My_Object__c".
    fields: optional comma-separated field API names to return (e.g.
    "Name,Industry,AnnualRevenue"); omit to return every field.
    """
    try:
        safe_sobject_type = url_path_id(sobject_type, "sobject_type")
        safe_record_id = url_path_id(record_id, "record_id")
        params: dict[str, Any] = {"fields": fields} if fields else {}
        result = _request(
            "GET",
            f"/services/data/{API_VERSION}/sobjects/{safe_sobject_type}/{safe_record_id}",
            params=params,
        )
        return success_with_capped_dict("record", result)
    except Exception as e:
        logger.error(
            f"Error fetching Salesforce {sobject_type} record {record_id}: {e}"
        )
        return _error(str(e))


@mcp.tool()
def salesforce_create_record(sobject_type: str, fields: dict[str, Any]) -> str:
    """
    Create a new record.
    sobject_type: an object's API name, e.g. "Account", "Contact",
    "Opportunity", or a custom object's name (e.g. "My_Object__c").
    fields: field API name -> value pairs, e.g. {"Name": "Acme Corp",
    "Industry": "Technology"}. Use salesforce_describe_sobject to learn
    which fields exist and are createable.
    """
    try:
        safe_sobject_type = url_path_id(sobject_type, "sobject_type")
        result = _request(
            "POST",
            f"/services/data/{API_VERSION}/sobjects/{safe_sobject_type}",
            json_data=fields,
        )
        # Only added when Salesforce actually returned something in it --
        # an unconditional `errors: []` on every plain success would give a
        # caller pattern-matching on "errors" key presence (a natural
        # Salesforce-API idiom; its own bulk/collections endpoints always
        # include one) a false partial-failure signal.
        extra = {"errors": result["errors"]} if result.get("errors") else {}
        return _success(id=result.get("id"), success=result.get("success"), **extra)
    except Exception as e:
        logger.error(f"Error creating Salesforce {sobject_type} record: {e}")
        return _error(str(e))


@mcp.tool()
def salesforce_update_record(
    sobject_type: str, record_id: str, fields: dict[str, Any]
) -> str:
    """
    Update an existing record. Only the fields provided are changed.
    sobject_type: an object's API name, e.g. "Account" or "My_Object__c".
    fields: field API name -> value pairs to change, e.g.
    {"Industry": "Finance"}.
    """
    try:
        if not fields:
            return _error("No fields provided to update")
        safe_sobject_type = url_path_id(sobject_type, "sobject_type")
        safe_record_id = url_path_id(record_id, "record_id")
        _request(
            "PATCH",
            f"/services/data/{API_VERSION}/sobjects/{safe_sobject_type}/{safe_record_id}",
            json_data=fields,
        )
        return _success(id=record_id)
    except Exception as e:
        logger.error(
            f"Error updating Salesforce {sobject_type} record {record_id}: {e}"
        )
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
