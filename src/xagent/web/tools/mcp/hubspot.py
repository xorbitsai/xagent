import json
import logging
import os
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Literal

import requests
from mcp.server.fastmcp import FastMCP

from ....config import get_tool_max_output_length
from .utils import clamp_limit as _clamp_limit
from .utils import require_clean_identifier as _require_clean_identifier
from .utils import setup_proxy_env
from .utils import success_with_capped_dict as _success_with_capped_dict
from .utils import url_path_id as _url_path_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hubspot-mcp")

setup_proxy_env()

mcp = FastMCP("hubspot-mcp")

HUBSPOT_BASE_URL = "https://api.hubapi.com"
DEFAULT_TIMEOUT_SECONDS = 30
# A single association-listing call is scoped to one contact and capped at
# max_results, so it isn't inherently slow on its own. Timeouts at the 30s
# default were reported, though, during a bulk pull looping this call across
# a large portal (~53k contacts); no latency beyond "exceeded 30s" was
# measured, so 90s is a deliberate margin rather than a tuned value. Scoped
# to this call path (rather than raising the shared default) so a slow or
# degraded HubSpot API doesn't also make every other tool call hang for the
# same longer window.
ASSOCIATION_LISTING_TIMEOUT_SECONDS = 90

DEFAULT_CONTACT_PROPERTIES = [
    "email",
    "firstname",
    "lastname",
    "company",
    "jobtitle",
    "phone",
    "lifecyclestage",
    "hs_lead_status",
]
DEFAULT_COMPANY_PROPERTIES = [
    "name",
    "domain",
    "industry",
    "numberofemployees",
    "city",
    "country",
    "lifecyclestage",
]
DEFAULT_DEAL_PROPERTIES = [
    "dealname",
    "dealstage",
    "pipeline",
    "amount",
    "closedate",
    "hs_lastmodifieddate",
    # dealstage/pipeline are opaque IDs on a custom pipeline, and closedate
    # is set on open deals too (and typically isn't cleared on reopen), so
    # neither reliably signals whether a deal is closed. hs_is_closed(_won|_lost)
    # are HubSpot's own computed closed-state flags - see
    # https://knowledge.hubspot.com/properties/hubspots-default-deal-properties
    "hs_is_closed",
    "hs_is_closed_won",
    "hs_is_closed_lost",
]
DEFAULT_CAMPAIGN_PROPERTIES = [
    "hs_name",
    "hs_campaign_status",
    "hs_start_date",
    "hs_end_date",
    "hs_notes",
    "hs_owner",
]
_MAX_EMAIL_IDS_PER_STATISTICS_REQUEST = 100

# HUBSPOT_DEFINED association type ids for notes.
_NOTE_ASSOCIATION_TYPE_IDS = {"contact": 202, "company": 190, "deal": 214}

# HUBSPOT_DEFINED association type id for deal -> contact.
_DEAL_TO_CONTACT_ASSOCIATION_TYPE_ID = 3

_ASSOCIATION_PAGE_SIZE = 100

# Per HubSpot's OpenAPI spec for this endpoint, time_period also accepts a
# "summarize/{period}" form - a portal-wide summary across all breakdown_by
# values rather than one row per value. The plain forms below stay listed
# separately (not built by string-formatting a period) so a typo in one
# doesn't silently produce a still-valid "summarize/<typo>" value.
_VALID_ANALYTICS_BREAKDOWNS = {
    "total",
    "daily",
    "weekly",
    "monthly",
    "summarize/daily",
    "summarize/weekly",
    "summarize/monthly",
}
# Union of the two documented /analytics/v2/reports variants: breakdowns
# (totals, sessions, sources, ...) and content object types (forms, pages,
# ...). Both share the same URL shape.
_VALID_ANALYTICS_REPORT_TYPES = {
    "totals",
    "sessions",
    "sources",
    "geolocation",
    "utm-campaigns",
    "utm-contents",
    "utm-mediums",
    "utm-sources",
    "utm-terms",
    "event-completions",
    "forms",
    "pages",
    "social-assists",
}


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _paged_list(
    path: str,
    list_field: str,
    *,
    params: dict[str, Any],
    after: str | None,
    project: Callable[[list[Any]], list[Any]] = lambda items: items,
) -> str:
    """Fetch one page from a HubSpot cursor-paginated list endpoint, project
    its items, and halve them until the response fits the platform's
    output limit.

    Unprojected HubSpot objects (full property maps) can serialize past the
    output filter's threshold and get hard-truncated into broken JSON, the
    same failure mode documented in google_analytics.py's list/report
    tools. Halving (rather than a fixed slice) keeps the cap adaptive to
    whatever property payload size a given portal's objects happen to
    have, and continues down to zero items - a single oversized item must
    still be capped, not silently returned whole because there's nothing
    left to halve away from it.

    On truncation, the returned ``after`` is deliberately this call's own
    input cursor, not the server's next-page cursor: the server already
    considers the full (untruncated) page consumed, so advancing to its
    next page would permanently skip whatever this page didn't return for
    space reasons. Retrying with the same ``after`` and a smaller ``limit``
    is what actually surfaces the trimmed entries.

    The explanatory "dead end" message added when halving collapses to
    zero items is itself re-checked against the size limit before being
    kept -- appending it unconditionally after the halving loop already
    converged could push an already-fitted response back over the limit,
    the exact failure mode this function exists to prevent.
    """
    result = _request("GET", path, params=params)
    next_after = ((result.get("paging") or {}).get("next") or {}).get("after")
    items = project(result.get("results", []))

    max_output_length = get_tool_max_output_length()
    truncated = False
    payload = {
        list_field: items,
        "truncated": truncated,
        "has_more": bool(next_after),
        "after": next_after,
    }
    response = _success(**payload)
    while len(response) > max_output_length and items:
        items = items[: len(items) // 2]
        truncated = True
        payload = {
            list_field: items,
            "truncated": truncated,
            "has_more": True,
            "after": after,
        }
        response = _success(**payload)
    if truncated and not items:
        # Collapsing all the way to zero means even the single largest
        # remaining record didn't fit alone - "retry with a smaller limit"
        # (the general truncated-page guidance) isn't guaranteed to help
        # here, so say so plainly instead of implying a fix that may not
        # exist.
        response_with_message = _success(
            **{
                **payload,
                "message": (
                    "Every record in this page was individually too large "
                    "to fit the output size limit, so none could be "
                    "returned. Retrying with a smaller `limit` may surface "
                    "different records if more exist, but cannot shrink an "
                    "individually oversized record."
                ),
            }
        )
        # Only use the enriched response if it still fits -- an operator can
        # configure XAGENT_TOOL_MAX_OUTPUT_LENGTH small enough that even this
        # already-empty payload plus the message text pushes back over the
        # limit, and appending it unconditionally would silently reintroduce
        # the hard-truncated-into-broken-JSON failure mode this whole
        # function exists to prevent. Falling back to the message-less
        # payload keeps the caller's own size contract intact instead.
        if len(response_with_message) <= max_output_length:
            response = response_with_message
    return response


def _require_date_format(
    value: str, strptime_format: str, field_name: str, label: str
) -> datetime:
    """Reject a value that isn't a real calendar date in the endpoint's
    expected format, and return the parsed date for a caller that also
    needs to check ordering.

    A shape-only regex (matching digit positions but not their range) would
    accept "20261332" or "2026-13-32" as syntactically fine and only fail
    once the request reaches HubSpot; strptime rejects both. But strptime
    alone is *more* lenient than a regex in the other direction: %m/%d
    accept non-zero-padded values, and %Y will happily consume fewer than 4
    digits if the remaining directives can still find something to match
    (datetime.strptime("202611", "%Y%m%d") silently parses as 2026-01-01,
    not a rejection) - so a genuinely malformed value could parse into the
    *wrong* date instead of failing. Reformatting the parsed result and
    comparing it back to the original string catches both failure modes.
    The three HubSpot APIs this connector calls use three different date
    formats (YYYYMMDD, YYYY-MM-DD, and YYYY-MM-DDTHH:MM:SSZ); an LLM caller
    mixing them up would otherwise get an opaque upstream 400 instead of a
    message naming the fix.
    """
    try:
        parsed = datetime.strptime(value, strptime_format)
    except ValueError:
        parsed = None
    if parsed is None or parsed.strftime(strptime_format) != value:
        raise ValueError(f"{field_name} must be a {label} date string")
    return parsed


def _require_date_range_order(
    start: datetime | None, end: datetime | None, start_field: str, end_field: str
) -> None:
    if start is not None and end is not None and start > end:
        raise ValueError(f"{start_field} must not be after {end_field}")


def _headers() -> dict[str, str]:
    token = os.environ.get("HUBSPOT_ACCESS_TOKEN")
    if not token:
        raise ValueError("HUBSPOT_ACCESS_TOKEN environment variable is missing")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
    timeout: float | tuple[float, float] | None = DEFAULT_TIMEOUT_SECONDS,
) -> Any:
    response = requests.request(
        method=method,
        url=f"{HUBSPOT_BASE_URL}{path}",
        headers=_headers(),
        params=params,
        json=body,
        timeout=timeout,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        response_text = response.text.strip()
        message = str(exc)
        if response_text:
            message = f"{message} - {response_text}"
        raise RuntimeError(message) from exc

    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


def _list_association_ids(
    path: str, max_results: int, after: str | None = None
) -> tuple[list[str], str | None]:
    """Collect associated object ids across pages, up to ``max_results``.

    Starts at ``after`` and follows HubSpot's ``paging.next.after`` cursor so
    results beyond the API's default page size are not silently dropped.
    Invalid ids are ignored and duplicate ids are collapsed while preserving
    order. Returns the collected ids and the cursor for the next unread page.
    """
    ids: list[str] = []
    seen_ids: set[str] = set()
    cursor = after
    while True:
        params: dict[str, Any] = {
            "limit": min(_ASSOCIATION_PAGE_SIZE, max_results - len(ids))
        }
        if cursor:
            params["after"] = cursor
        page = _request(
            "GET", path, params=params, timeout=ASSOCIATION_LISTING_TIMEOUT_SECONDS
        )
        results = page.get("results", [])
        for item in results:
            raw_id = item.get("id")
            if not raw_id:
                continue
            association_id = str(raw_id)
            if association_id not in seen_ids:
                ids.append(association_id)
                seen_ids.add(association_id)
        cursor = ((page.get("paging") or {}).get("next") or {}).get("after")
        if len(ids) >= max_results:
            return ids[:max_results], cursor
        if not cursor:
            return ids, None
        if not results:
            # A page with no results but a next cursor would loop forever.
            return ids, cursor


def _parse_properties(properties_json: str) -> dict[str, Any]:
    properties = json.loads(properties_json)
    if not isinstance(properties, dict):
        raise ValueError("properties_json must be a JSON object of property values")
    return properties


def _parse_filter_groups(filter_groups_json: str) -> list[dict[str, Any]]:
    try:
        filter_groups = json.loads(filter_groups_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"filter_groups_json is not valid JSON: {e}") from e
    if not isinstance(filter_groups, list) or not all(
        isinstance(group, dict)
        and isinstance(group.get("filters"), list)
        and group["filters"]
        for group in filter_groups
    ):
        raise ValueError(
            "filter_groups_json must be a JSON array of HubSpot filterGroups "
            'objects, each with a non-empty "filters" list, e.g. '
            '[{"filters": [{"propertyName": "dealstage", "operator": "EQ", '
            '"value": "appointmentscheduled"}]}] - not a flat list of '
            "conditions, and not a group with an empty or missing filters "
            "list (which HubSpot treats as matching everything, defeating "
            "the point of filtering)."
        )
    return filter_groups


def _project_id_and_properties(items: list[Any] | None) -> list[dict[str, Any]]:
    if not items:
        return []
    return [
        {"id": item.get("id"), "properties": item.get("properties") or {}}
        for item in items
        if isinstance(item, dict)
    ]


def _search(
    object_type: str,
    properties: list[str],
    limit: int,
    query: str | None = None,
    filter_groups_json: str | None = None,
) -> dict[str, Any]:
    """Free-text search, property filtering, or both, against one HubSpot
    CRM object type's /search endpoint.

    filter_groups_json, when given, is HubSpot's native filterGroups JSON -
    see _parse_filter_groups for the exact shape required. Requires at
    least one of `query` or a non-empty `filter_groups_json`: the search
    endpoint would otherwise silently accept neither and return an
    arbitrary, unfiltered page that looks like a real match - the
    `hubspot_list_{object_type}` tool exists specifically for that case and
    says so up front instead of a plausible-looking but meaningless result.
    """
    query = query.strip() if query else None
    filter_groups = (
        _parse_filter_groups(filter_groups_json) if filter_groups_json else None
    )
    if not query and not filter_groups:
        raise ValueError(
            f"hubspot_search_{object_type} needs a query or filter_groups_json "
            f"- to list every {object_type} with neither, use "
            f"hubspot_list_{object_type} instead."
        )
    body: dict[str, Any] = {
        "properties": properties,
        "limit": max(1, min(limit, 100)),
    }
    if query:
        body["query"] = query
    if filter_groups:
        body["filterGroups"] = filter_groups
    result = _request("POST", f"/crm/v3/objects/{object_type}/search", body=body)
    return {
        "total": result.get("total") or 0,
        "results": _project_id_and_properties(result.get("results", [])),
    }


@mcp.tool()
def hubspot_search_contacts(
    query: str | None = None,
    filter_groups_json: str | None = None,
    limit: int = 10,
) -> str:
    """
    Search HubSpot contacts by free-text query (matches name, email, phone,
    company), by property filters, or both. Always search before creating a
    contact to avoid duplicates.

    filter_groups_json, when given, is HubSpot's native filterGroups JSON
    array, e.g. '[{"filters": [{"propertyName": "createdate", "operator":
    "GTE", "value": "2026-09-01"}]}]' - filters within one group are ANDed,
    and groups are ORed. See HubSpot's CRM search API docs for the full set
    of operators (EQ, GTE, LTE, CONTAINS_TOKEN, ...). To list every contact
    with no query or filter at all, use hubspot_list_contacts instead - it's
    the more direct tool for that and doesn't need an empty search body.
    """
    try:
        found = _search(
            "contacts",
            DEFAULT_CONTACT_PROPERTIES,
            limit,
            query=query,
            filter_groups_json=filter_groups_json,
        )
        return _success(**found)
    except Exception as e:
        logger.error(f"Error searching contacts: {e}")
        return _error(str(e))


@mcp.tool()
def hubspot_get_contact(contact_id: str) -> str:
    """
    Get a HubSpot contact by id, including associated company and deal ids.
    """
    try:
        contact_id = _url_path_id(contact_id, "contact_id")
        contact = _request(
            "GET",
            f"/crm/v3/objects/contacts/{contact_id}",
            params={
                "properties": ",".join(DEFAULT_CONTACT_PROPERTIES),
                "associations": "companies,deals",
            },
        )
        return _success(contact=contact)
    except Exception as e:
        logger.error(f"Error getting contact: {e}")
        return _error(str(e))


@mcp.tool()
def hubspot_create_contact(properties_json: str) -> str:
    """
    Create a HubSpot contact. properties_json is a JSON object of HubSpot contact
    properties, e.g. {"email": "a@b.com", "firstname": "Ada", "company": "Acme"}.
    Search for the contact first to avoid creating duplicates.
    """
    try:
        contact = _request(
            "POST",
            "/crm/v3/objects/contacts",
            body={"properties": _parse_properties(properties_json)},
        )
        return _success(contact=contact)
    except Exception as e:
        logger.error(f"Error creating contact: {e}")
        return _error(str(e))


@mcp.tool()
def hubspot_update_contact(contact_id: str, properties_json: str) -> str:
    """
    Update properties on an existing HubSpot contact.
    properties_json is a JSON object of the properties to change.
    """
    try:
        contact_id = _url_path_id(contact_id, "contact_id")
        contact = _request(
            "PATCH",
            f"/crm/v3/objects/contacts/{contact_id}",
            body={"properties": _parse_properties(properties_json)},
        )
        return _success(contact=contact)
    except Exception as e:
        logger.error(f"Error updating contact: {e}")
        return _error(str(e))


@mcp.tool()
def hubspot_search_companies(
    query: str | None = None,
    filter_groups_json: str | None = None,
    limit: int = 10,
) -> str:
    """
    Search HubSpot companies by free-text query (matches name, domain), by
    property filters, or both.

    filter_groups_json, when given, is HubSpot's native filterGroups JSON
    array - see hubspot_search_contacts for the exact shape and operator
    semantics. To list every company with no query or filter at all, use
    hubspot_list_companies instead.
    """
    try:
        found = _search(
            "companies",
            DEFAULT_COMPANY_PROPERTIES,
            limit,
            query=query,
            filter_groups_json=filter_groups_json,
        )
        return _success(**found)
    except Exception as e:
        logger.error(f"Error searching companies: {e}")
        return _error(str(e))


@mcp.tool()
def hubspot_create_company(properties_json: str) -> str:
    """
    Create a HubSpot company. properties_json is a JSON object of HubSpot company
    properties, e.g. {"name": "Acme Inc", "domain": "acme.com"}.
    Search for the company first to avoid creating duplicates.
    """
    try:
        company = _request(
            "POST",
            "/crm/v3/objects/companies",
            body={"properties": _parse_properties(properties_json)},
        )
        return _success(company=company)
    except Exception as e:
        logger.error(f"Error creating company: {e}")
        return _error(str(e))


@mcp.tool()
def hubspot_update_company(company_id: str, properties_json: str) -> str:
    """
    Update properties on an existing HubSpot company.
    properties_json is a JSON object of the properties to change.
    """
    try:
        company_id = _url_path_id(company_id, "company_id")
        company = _request(
            "PATCH",
            f"/crm/v3/objects/companies/{company_id}",
            body={"properties": _parse_properties(properties_json)},
        )
        return _success(company=company)
    except Exception as e:
        logger.error(f"Error updating company: {e}")
        return _error(str(e))


def _get_associated_deals(
    object_type: Literal["contacts", "companies"],
    object_id: str,
    limit: int,
    after: str | None,
) -> dict[str, Any]:
    """Fetch deals associated with a HubSpot contact or company record.

    Halves the deal list (mirroring _paged_list - see its docstring for why)
    if the response would otherwise be hard-truncated into invalid JSON past
    the platform's output limit.
    """
    deal_ids, next_after = _list_association_ids(
        f"/crm/v3/objects/{object_type}/{object_id}/associations/deals",
        max(1, min(limit, 100)),
        after,
    )
    if not deal_ids:
        return {
            "deals": [],
            "has_more": bool(next_after),
            "after": next_after,
            "truncated": False,
            "missing_deal_ids": [],
        }

    batch = _request(
        "POST",
        "/crm/v3/objects/deals/batch/read",
        body={
            "properties": DEFAULT_DEAL_PROPERTIES,
            "inputs": [{"id": deal_id} for deal_id in deal_ids],
        },
    )
    results = batch.get("results", [])
    deals = [
        {"id": item.get("id"), "properties": item.get("properties", {})}
        for item in results
    ]
    # A partial batch-read (e.g. an archived deal dropped between the
    # association listing above and this call) must be surfaced, not
    # silently returned as if every requested id came back. Normalize ids to
    # strings on both sides because HubSpot record ids are string identifiers.
    returned_ids = {str(item["id"]) for item in results if item.get("id")}
    missing_deal_ids = sorted(set(deal_ids) - returned_ids)

    def _build(
        deals_slice: list[Any],
        missing_slice: list[Any],
        truncated: bool,
        deals_truncated: bool,
    ) -> dict[str, Any]:
        return {
            "deals": deals_slice,
            "has_more": bool(next_after) or deals_truncated,
            # When deal data was trimmed, retry this same association page
            # with a smaller limit. Otherwise advance to HubSpot's next page.
            "after": after if deals_truncated else next_after,
            "truncated": truncated,
            "missing_deal_ids": missing_slice,
        }

    max_output_length = get_tool_max_output_length()
    payload = _build(deals, missing_deal_ids, False, False)
    truncated = False
    deals_truncated = False
    # Shrink missing_deal_ids before deals: it's a diagnostic list of ids
    # HubSpot didn't return, strictly less valuable than the deals the
    # caller actually asked for, so a batch-read shortfall large enough to
    # need trimming on its own shouldn't cost the caller real deal data
    # that already fit.
    while len(_success(**payload)) > max_output_length and (deals or missing_deal_ids):
        truncated = True
        if missing_deal_ids:
            missing_deal_ids = missing_deal_ids[: len(missing_deal_ids) // 2]
        else:
            deals = deals[: len(deals) // 2]
            deals_truncated = True
        payload = _build(deals, missing_deal_ids, truncated, deals_truncated)
    if truncated and not deals and not missing_deal_ids:
        # Collapsing everything away means even the single largest
        # remaining entry didn't fit alone - "retry with a smaller `limit`"
        # isn't guaranteed to help here, so say so plainly instead of
        # returning an empty result indistinguishable from "nothing exists".
        payload_with_message = {
            **payload,
            "message": (
                "Every deal (and/or every id HubSpot's batch lookup didn't "
                "return) was individually too large to fit the output size "
                "limit, so none could be returned from this page. Retrying "
                "with a smaller `limit` cannot shrink an individually "
                "oversized record."
            ),
        }
        if len(_success(**payload_with_message)) <= max_output_length:
            payload = payload_with_message
    return payload


@mcp.tool()
def hubspot_get_contact_deals(
    contact_id: str, limit: int = 100, after: str | None = None
) -> str:
    """
    List every deal associated with a HubSpot contact - open and closed alike
    - including deal stage, pipeline, amount, close date, and the closed-state
    flags hs_is_closed/hs_is_closed_won/hs_is_closed_lost. Use those flags to
    tell open from closed: dealstage and pipeline are opaque IDs on a custom
    pipeline, and closedate is set on open deals too (and typically isn't
    cleared when a deal is reopened), so neither reliably signals closed on
    its own. Returns at most `limit` deals (max 100). `has_more` is true when
    there are more deals than returned, either because of `limit` or because
    the response was trimmed to fit the output size limit (`truncated` is
    then also true). Pass the returned `after` cursor to retrieve the next
    page; if `truncated` is true because deals were trimmed, retry that cursor
    with a smaller `limit`. `missing_deal_ids` lists any requested deal id HubSpot's
    own batch lookup silently dropped (empty when none were) - though when
    `truncated` is also true, this list may itself have been trimmed, so
    it's not a complete accounting in that case.

    A contact's deals are not necessarily the same as its company's deals: two
    contacts can share a name (e.g. the same person with a role at two
    different companies) while belonging to different HubSpot companies. When
    the question is about a company's deals rather than one specific
    contact's, use `hubspot_get_company_deals` instead of guessing which
    contact to look up by name - it associates deals with the company record
    directly instead of relying on a contact match.
    """
    try:
        contact_id = _url_path_id(contact_id, "contact_id")
        return _success(**_get_associated_deals("contacts", contact_id, limit, after))
    except Exception as e:
        logger.error(f"Error getting contact deals: {e}")
        return _error(str(e))


@mcp.tool()
def hubspot_get_company_deals(
    company_id: str, limit: int = 100, after: str | None = None
) -> str:
    """
    List every deal associated with a HubSpot company - open and closed alike
    - including deal stage, pipeline, amount, close date, and the closed-state
    flags hs_is_closed/hs_is_closed_won/hs_is_closed_lost; use those flags to
    tell open from closed rather than dealstage/pipeline/closedate (see
    `hubspot_get_contact_deals` for why those aren't reliable on their own).
    Look up a company_id with `hubspot_search_companies` first if you don't
    already have one. Returns at most `limit` deals (max 100);
    `has_more`/`truncated`/`missing_deal_ids` behave as in
    `hubspot_get_contact_deals`.

    This reads the company-to-deal association directly. Do not substitute a
    contact lookup for this (e.g. searching for a person's name and calling
    `hubspot_get_contact_deals` on the match): a contact match found by name
    is not proof that contact belongs to the company being asked about, and a
    deal returned that way must not be reported as the company's unless a
    call scoped to this company's id actually returned it.
    """
    try:
        company_id = _url_path_id(company_id, "company_id")
        return _success(**_get_associated_deals("companies", company_id, limit, after))
    except Exception as e:
        logger.error(f"Error getting company deals: {e}")
        return _error(str(e))


@mcp.tool()
def hubspot_create_deal(properties_json: str, contact_id: str | None = None) -> str:
    """
    Create a HubSpot deal. properties_json is a JSON object of HubSpot deal
    properties, e.g. {"dealname": "Acme - Onboarding", "amount": "1000",
    "pipeline": "default", "dealstage": "appointmentscheduled"}.
    If contact_id is given, the new deal is associated with that contact -
    there is no company_id param, so a deal created here won't be found by
    `hubspot_get_company_deals` unless HubSpot separately associates it
    with a company (e.g. via the contact's own company association).
    """
    try:
        body: dict[str, Any] = {"properties": _parse_properties(properties_json)}
        if contact_id:
            # contact_id goes into the JSON body below, not a URL path, so
            # only reject a malformed id (url_path_id's percent-encoding
            # would send HubSpot the encoded string instead of the real id).
            contact_id = _require_clean_identifier(contact_id, "contact_id")
            body["associations"] = [
                {
                    "to": {"id": contact_id},
                    "types": [
                        {
                            "associationCategory": "HUBSPOT_DEFINED",
                            "associationTypeId": _DEAL_TO_CONTACT_ASSOCIATION_TYPE_ID,
                        }
                    ],
                }
            ]
        deal = _request("POST", "/crm/v3/objects/deals", body=body)
        return _success(deal=deal)
    except Exception as e:
        logger.error(f"Error creating deal: {e}")
        return _error(str(e))


@mcp.tool()
def hubspot_update_deal(deal_id: str, properties_json: str) -> str:
    """
    Update properties on an existing HubSpot deal, e.g. to move it to a new
    pipeline stage: {"dealstage": "qualifiedtobuy"}.
    properties_json is a JSON object of the properties to change.
    """
    try:
        deal_id = _url_path_id(deal_id, "deal_id")
        deal = _request(
            "PATCH",
            f"/crm/v3/objects/deals/{deal_id}",
            body={"properties": _parse_properties(properties_json)},
        )
        return _success(deal=deal)
    except Exception as e:
        logger.error(f"Error updating deal: {e}")
        return _error(str(e))


@mcp.tool()
def hubspot_search_deals(
    query: str | None = None,
    filter_groups_json: str | None = None,
    limit: int = 10,
) -> str:
    """
    Search HubSpot deals by property filters (stage, pipeline, amount,
    close date, ...), by free-text query, or both - across the whole
    portal, not scoped to one contact or company. `query` only matches
    `dealname` (deals have no broader default-searchable property set the
    way contacts/companies do) - for anything else (owner, amount, a
    stage), use filter_groups_json instead of expecting query to find it.

    filter_groups_json, when given, is HubSpot's native filterGroups JSON
    array, e.g. '[{"filters": [{"propertyName": "dealstage", "operator":
    "EQ", "value": "appointmentscheduled"}]}]' - see hubspot_search_contacts
    for the exact shape and operator semantics.

    For every deal tied to one contact or company, hubspot_get_contact_deals
    / hubspot_get_company_deals are cheaper and don't need a filter. To list
    every deal in the portal with no query or filter, use hubspot_list_deals.
    """
    try:
        found = _search(
            "deals",
            DEFAULT_DEAL_PROPERTIES,
            limit,
            query=query,
            filter_groups_json=filter_groups_json,
        )
        return _success(**found)
    except Exception as e:
        logger.error(f"Error searching deals: {e}")
        return _error(str(e))


@mcp.tool()
def hubspot_list_deals(limit: int = 100, after: str | None = None) -> str:
    """
    List every deal in the portal - not scoped to any one contact or
    company - including deal stage, pipeline, amount, close date, and the
    closed-state flags hs_is_closed/hs_is_closed_won/hs_is_closed_lost; use
    those flags to tell open from closed rather than
    dealstage/pipeline/closedate (see hubspot_get_contact_deals for why
    those aren't reliable alone). Returns at most `limit` deals (max 100);
    `has_more` is true when the portal has more deals past this page
    (`after` is the cursor to fetch it), and `truncated` is true when this
    page itself was trimmed to fit the output size limit - retry with a
    smaller `limit` to see the trimmed entries.

    Use this for "list/show me all deals" requests. For deals tied to one
    contact or company, hubspot_get_contact_deals / hubspot_get_company_deals
    are cheaper and don't require paging the whole portal. For deals
    matching a filter (stage, amount, date range, ...), use
    hubspot_search_deals instead of listing everything and filtering
    yourself.
    """
    try:
        params: dict[str, Any] = {
            "limit": _clamp_limit(limit, max_limit=100),
            "properties": ",".join(DEFAULT_DEAL_PROPERTIES),
        }
        if after:
            params["after"] = after
        return _paged_list(
            "/crm/v3/objects/deals",
            "deals",
            params=params,
            after=after,
            project=_project_id_and_properties,
        )
    except Exception as e:
        logger.error(f"Error listing deals: {e}")
        return _error(str(e))


@mcp.tool()
def hubspot_list_companies(limit: int = 100, after: str | None = None) -> str:
    """
    List every company in the portal. Returns at most `limit` companies
    (max 100); `has_more` is true when the portal has more companies past
    this page (`after` is the cursor to fetch it), and `truncated` is true
    when this page itself was trimmed to fit the output size limit - retry
    with a smaller `limit` to see the trimmed entries.

    Use this for "list/show me all companies" requests. For a name/domain
    lookup, use hubspot_search_companies instead - it's cheaper and doesn't
    require paging the whole portal.
    """
    try:
        params: dict[str, Any] = {
            "limit": _clamp_limit(limit, max_limit=100),
            "properties": ",".join(DEFAULT_COMPANY_PROPERTIES),
        }
        if after:
            params["after"] = after
        return _paged_list(
            "/crm/v3/objects/companies",
            "companies",
            params=params,
            after=after,
            project=_project_id_and_properties,
        )
    except Exception as e:
        logger.error(f"Error listing companies: {e}")
        return _error(str(e))


@mcp.tool()
def hubspot_list_contacts(limit: int = 100, after: str | None = None) -> str:
    """
    List every contact in the portal. Returns at most `limit` contacts
    (max 100); `has_more` is true when the portal has more contacts past
    this page (`after` is the cursor to fetch it), and `truncated` is true
    when this page itself was trimmed to fit the output size limit - retry
    with a smaller `limit` to see the trimmed entries.

    Use this for "list/show me all contacts" requests. For a name/email/
    phone/company lookup, use hubspot_search_contacts instead - it's
    cheaper and doesn't require paging the whole portal.
    """
    try:
        params: dict[str, Any] = {
            "limit": _clamp_limit(limit, max_limit=100),
            "properties": ",".join(DEFAULT_CONTACT_PROPERTIES),
        }
        if after:
            params["after"] = after
        return _paged_list(
            "/crm/v3/objects/contacts",
            "contacts",
            params=params,
            after=after,
            project=_project_id_and_properties,
        )
    except Exception as e:
        logger.error(f"Error listing contacts: {e}")
        return _error(str(e))


@mcp.tool()
def hubspot_get_contact_notes(contact_id: str, limit: int = 20) -> str:
    """
    List the notes associated with a HubSpot contact (most recent interaction
    history), including note body and timestamp. Returns at most `limit` notes
    (max 100); `has_more` is true when the contact has additional notes.
    """
    try:
        contact_id = _url_path_id(contact_id, "contact_id")
        note_ids, next_after = _list_association_ids(
            f"/crm/v3/objects/contacts/{contact_id}/associations/notes",
            max(1, min(limit, 100)),
        )
        if not note_ids:
            return _success(notes=[], has_more=bool(next_after))

        notes = _request(
            "POST",
            "/crm/v3/objects/notes/batch/read",
            body={
                "properties": ["hs_note_body", "hs_timestamp"],
                "inputs": [{"id": note_id} for note_id in note_ids],
            },
        )
        return _success(
            notes=[
                {"id": item.get("id"), "properties": item.get("properties", {})}
                for item in notes.get("results", [])
            ],
            has_more=bool(next_after),
        )
    except Exception as e:
        logger.error(f"Error getting contact notes: {e}")
        return _error(str(e))


@mcp.tool()
def hubspot_create_note(
    note_body: str,
    contact_id: str | None = None,
    company_id: str | None = None,
    deal_id: str | None = None,
) -> str:
    """
    Create a note in HubSpot and associate it with a contact, company, and/or deal.
    Use this to log activity summaries, qualification outcomes, or next steps.
    At least one of contact_id, company_id, or deal_id must be provided.
    """
    try:
        # Truthiness (not `is not None`) keeps an empty string meaning
        # "not provided", matching the association filter below - only a
        # non-empty id gets validated for stray whitespace.
        targets = {
            "contact": _require_clean_identifier(contact_id, "contact_id")
            if contact_id
            else None,
            "company": _require_clean_identifier(company_id, "company_id")
            if company_id
            else None,
            "deal": _require_clean_identifier(deal_id, "deal_id") if deal_id else None,
        }
        associations = [
            {
                "to": {"id": object_id},
                "types": [
                    {
                        "associationCategory": "HUBSPOT_DEFINED",
                        "associationTypeId": _NOTE_ASSOCIATION_TYPE_IDS[object_name],
                    }
                ],
            }
            for object_name, object_id in targets.items()
            if object_id
        ]
        if not associations:
            raise ValueError(
                "At least one of contact_id, company_id, or deal_id is required"
            )

        note = _request(
            "POST",
            "/crm/v3/objects/notes",
            body={
                "properties": {
                    "hs_note_body": note_body,
                    "hs_timestamp": datetime.now(timezone.utc).isoformat(),
                },
                "associations": associations,
            },
        )
        return _success(note=note)
    except Exception as e:
        logger.error(f"Error creating note: {e}")
        return _error(str(e))


_FORM_SUMMARY_FIELDS = ("id", "name", "formType", "createdAt", "updatedAt", "archived")
_EMAIL_SUMMARY_FIELDS = (
    "id",
    "name",
    "subject",
    "state",
    "publishDate",
    "createdAt",
    "updatedAt",
)


def _project_fields(items: list[Any], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {field: item.get(field) for field in fields}
        for item in items
        if isinstance(item, dict)
    ]


@mcp.tool()
def hubspot_list_forms(limit: int = 20, after: str | None = None) -> str:
    """
    List HubSpot marketing forms: id, name, form type, created/updated
    timestamps, and archived status. Returns at most `limit` forms (max
    100); `has_more` is true when the portal has more forms past this
    page (`after` is the cursor to fetch it), and `truncated` is true
    when this page itself was trimmed to fit the output size limit -
    retry with a smaller `limit` to see the trimmed entries. Use
    hubspot_get_form_submissions to pull the submissions collected by a
    specific form.
    """
    try:
        params: dict[str, Any] = {"limit": max(1, min(limit, 100))}
        if after:
            params["after"] = after
        return _paged_list(
            "/marketing/v3/forms",
            "forms",
            params=params,
            after=after,
            project=lambda items: _project_fields(items, _FORM_SUMMARY_FIELDS),
        )
    except Exception as e:
        logger.error(f"Error listing forms: {e}")
        return _error(str(e))


@mcp.tool()
def hubspot_get_form_submissions(
    form_id: str, limit: int = 20, after: str | None = None
) -> str:
    """
    Get recent submissions for a HubSpot form, including submitted field
    values, submission timestamp, and page URL. Returns at most `limit`
    submissions (max 50); `has_more` is true when the form has more
    submissions past this page (`after` is the cursor to fetch it), and
    `truncated` is true when this page itself was trimmed to fit the
    output size limit - retry with a smaller `limit` to see the trimmed
    entries.
    """
    try:
        form_id = _url_path_id(form_id, "form_id")
        params: dict[str, Any] = {"limit": max(1, min(limit, 50))}
        if after:
            params["after"] = after
        return _paged_list(
            f"/form-integrations/v1/submissions/forms/{form_id}",
            "submissions",
            params=params,
            after=after,
        )
    except Exception as e:
        logger.error(f"Error getting form submissions: {e}")
        return _error(str(e))


@mcp.tool()
def hubspot_get_analytics_report(
    report_type: str,
    start_date: str,
    end_date: str,
    breakdown: str = "total",
) -> str:
    """
    Get a HubSpot traffic analytics report over [start_date, end_date].
    `report_type` selects the dimension or content object to report on:
    "totals", "sessions", "sources", "geolocation", "utm-campaigns",
    "utm-contents", "utm-mediums", "utm-sources", "utm-terms",
    "event-completions", "forms", "pages", or "social-assists".
    `breakdown` selects the time granularity within that window: "total"
    (one summed value), "daily", "weekly", "monthly", or a
    "summarize/daily", "summarize/weekly", "summarize/monthly" variant
    (a portal-wide summary across all `report_type` values instead of one
    row per value). `start_date` and `end_date` are required "YYYYMMDD"
    strings; a "daily" breakdown spans at most 500 days.

    Note: this covers HubSpot's traffic/analytics dimensions, not
    custom reports or dashboards built in the HubSpot report editor -
    HubSpot has no public API to read those. Requires a Marketing Hub
    Basic, Professional, or Enterprise account; Free/Starter accounts are
    not supported by this HubSpot API.
    """
    try:
        normalized_report_type = report_type.strip().lower()
        if normalized_report_type not in _VALID_ANALYTICS_REPORT_TYPES:
            raise ValueError(
                f"report_type must be one of {sorted(_VALID_ANALYTICS_REPORT_TYPES)}"
            )
        normalized_breakdown = breakdown.strip().lower()
        if normalized_breakdown not in _VALID_ANALYTICS_BREAKDOWNS:
            raise ValueError(
                f"breakdown must be one of {sorted(_VALID_ANALYTICS_BREAKDOWNS)}"
            )
        start_dt = _require_date_format(start_date, "%Y%m%d", "start_date", "YYYYMMDD")
        end_dt = _require_date_format(end_date, "%Y%m%d", "end_date", "YYYYMMDD")
        _require_date_range_order(start_dt, end_dt, "start_date", "end_date")
        if normalized_breakdown == "daily" and (end_dt - start_dt).days > 500:
            raise ValueError(
                "start_date and end_date must not span more than 500 days "
                'for a "daily" breakdown'
            )
        result = _request(
            "GET",
            f"/analytics/v2/reports/{normalized_report_type}/{normalized_breakdown}",
            params={"start": start_date, "end": end_date},
        )
        return _success_with_capped_dict("report", result)
    except Exception as e:
        logger.error(f"Error getting analytics report: {e}")
        return _error(str(e))


@mcp.tool()
def hubspot_list_marketing_emails(limit: int = 20, after: str | None = None) -> str:
    """
    List HubSpot marketing emails: id, name, subject, state, publish
    date, and created/updated timestamps. Returns at most `limit` emails
    (max 100); `has_more` is true when the portal has more emails past
    this page (`after` is the cursor to fetch it), and `truncated` is
    true when this page itself was trimmed to fit the output size limit
    - retry with a smaller `limit` to see the trimmed entries. Use
    hubspot_get_marketing_email_statistics for performance metrics on a
    specific email.

    The connector requests the required marketing-email scope as optional
    (it's gated on Marketing Hub Enterprise, or the transactional email
    add-on on lower tiers), so this call fails with a permissions error on
    portals where that scope wasn't granted - the connection itself still
    works for every other tool.
    """
    try:
        params: dict[str, Any] = {"limit": max(1, min(limit, 100))}
        if after:
            params["after"] = after
        return _paged_list(
            "/marketing/v3/emails",
            "emails",
            params=params,
            after=after,
            project=lambda items: _project_fields(items, _EMAIL_SUMMARY_FIELDS),
        )
    except Exception as e:
        logger.error(f"Error listing marketing emails: {e}")
        return _error(str(e))


@mcp.tool()
def hubspot_get_marketing_email_statistics(
    email_ids: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """
    Get send/open/click/bounce statistics for HubSpot marketing emails.
    `email_ids` is a single email id, or a comma-separated list of ids
    (max 100) to fetch statistics for multiple emails at once - each id
    must be non-empty with no surrounding whitespace. `start_date`/
    `end_date` are optional ISO 8601 timestamps ("2024-01-01T00:00:00Z")
    that limit the reporting window; omitting both returns all-time
    statistics.

    The connector requests the required marketing-email scope as optional
    (it's gated on Marketing Hub Enterprise, or the transactional email
    add-on on lower tiers), so this call fails with a permissions error on
    portals where that scope wasn't granted - the connection itself still
    works for every other tool.
    """
    try:
        raw_ids = email_ids.split(",")
        ids = [_require_clean_identifier(raw_id, "each email id") for raw_id in raw_ids]
        if len(ids) > _MAX_EMAIL_IDS_PER_STATISTICS_REQUEST:
            raise ValueError(
                f"email_ids must contain at most "
                f"{_MAX_EMAIL_IDS_PER_STATISTICS_REQUEST} ids per request"
            )
        params: dict[str, Any] = {
            # HubSpot's emailIds is an array param: requests serializes a
            # list value as repeated "emailIds=<id>" pairs, matching that
            # shape rather than a single comma-joined value.
            "emailIds": ids
        }
        start_dt = end_dt = None
        if start_date:
            start_dt = _require_date_format(
                start_date, "%Y-%m-%dT%H:%M:%SZ", "start_date", "YYYY-MM-DDTHH:MM:SSZ"
            )
            params["startTimestamp"] = start_date
        if end_date:
            end_dt = _require_date_format(
                end_date, "%Y-%m-%dT%H:%M:%SZ", "end_date", "YYYY-MM-DDTHH:MM:SSZ"
            )
            params["endTimestamp"] = end_date
        _require_date_range_order(start_dt, end_dt, "start_date", "end_date")
        result = _request("GET", "/marketing/v3/emails/statistics/list", params=params)
        return _success_with_capped_dict("statistics", result)
    except Exception as e:
        logger.error(f"Error getting marketing email statistics: {e}")
        return _error(str(e))


@mcp.tool()
def hubspot_list_campaigns(limit: int = 20, after: str | None = None) -> str:
    """
    List HubSpot marketing campaigns, including id and properties (name,
    status, start/end date, notes, owner). Returns at most `limit`
    campaigns (max 100); `has_more` is true when the portal has more
    campaigns past this page (`after` is the cursor to fetch it), and
    `truncated` is true when this page itself was trimmed to fit the
    output size limit - retry with a smaller `limit` to see the trimmed
    entries. Use hubspot_get_campaign_metrics for performance metrics on
    a specific campaign.

    The connector requests marketing.campaigns.read as optional (the
    Campaigns API requires Marketing Hub Professional or higher), so this
    call fails with a permissions error on portals below that tier - the
    connection itself still works for every other tool.
    """
    try:
        params: dict[str, Any] = {
            "limit": max(1, min(limit, 100)),
            "properties": ",".join(DEFAULT_CAMPAIGN_PROPERTIES),
        }
        if after:
            params["after"] = after
        return _paged_list(
            "/marketing/v3/campaigns",
            "campaigns",
            params=params,
            after=after,
            project=lambda items: [
                {"id": item.get("id"), "properties": item.get("properties", {})}
                for item in items
                if isinstance(item, dict)
            ],
        )
    except Exception as e:
        logger.error(f"Error listing campaigns: {e}")
        return _error(str(e))


@mcp.tool()
def hubspot_get_campaign_metrics(
    campaign_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """
    Get attribution metrics for a HubSpot marketing campaign, such as
    sessions, new contacts, and influenced contacts. `start_date`/
    `end_date` are optional "YYYY-MM-DD" strings (a different format
    from hubspot_get_analytics_report's "YYYYMMDD" - this is a separate,
    newer HubSpot API) that limit the reporting window.

    The connector requests marketing.campaigns.read as optional (the
    Campaigns API requires Marketing Hub Professional or higher), so this
    call fails with a permissions error on portals below that tier - the
    connection itself still works for every other tool.
    """
    try:
        campaign_id = _url_path_id(campaign_id, "campaign_id")
        params: dict[str, Any] = {}
        start_dt = end_dt = None
        if start_date:
            start_dt = _require_date_format(
                start_date, "%Y-%m-%d", "start_date", "YYYY-MM-DD"
            )
            params["startDate"] = start_date
        if end_date:
            end_dt = _require_date_format(
                end_date, "%Y-%m-%d", "end_date", "YYYY-MM-DD"
            )
            params["endDate"] = end_date
        _require_date_range_order(start_dt, end_dt, "start_date", "end_date")
        result = _request(
            "GET",
            f"/marketing/v3/campaigns/{campaign_id}/reports/metrics",
            params=params,
        )
        return _success_with_capped_dict("metrics", result)
    except Exception as e:
        logger.error(f"Error getting campaign metrics: {e}")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
