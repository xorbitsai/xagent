import json
import logging
import os
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import requests
from mcp.server.fastmcp import FastMCP

from ....config import get_tool_max_output_length
from .utils import setup_proxy_env

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
        payload["message"] = (
            "Every record in this page was individually too large to fit "
            "the output size limit, so none could be returned. Retrying "
            "with a smaller `limit` may surface different records if more "
            "exist, but cannot shrink an individually oversized record."
        )
        response = _success(**payload)
    return response


def _success_with_capped_dict(field_name: str, data: Any) -> str:
    """Build a success payload, trimming a dict until the response fits the
    platform's output limit.

    HubSpot's analytics/statistics/metrics responses are dicts keyed by
    date, dimension, or id depending on the endpoint and parameters (e.g.
    one key per day for a "daily" analytics breakdown over a wide date
    range), where most of the payload's size typically lives in one or two
    large nested list/dict values while the rest are small scalars (an
    "offset" or "total" field alongside a big "breakdowns" list). Dropping
    whole top-level keys to shrink such a dict can discard the entire
    useful payload on the very first step while leaving small, mostly
    empty scalar fields behind - and there's no cursor to retry with, so
    that data is gone for this call. Phase 1 instead repeatedly finds the
    largest list/dict-valued key and halves *its* contents (recursing one
    level, not further), so small scalar keys survive untouched as long as
    there is a bigger key left to shrink first. Phase 2 is a fallback for
    the residual case - a dict with no list/dict-valued keys at all (e.g.
    a handful of scalar keys with huge string values) - and drops whole
    keys, exactly as phase 1 replaces; it's guaranteed to terminate at {}.
    """
    max_output_length = get_tool_max_output_length()
    response = _success(**{field_name: data, "truncated": False})
    if not isinstance(data, dict) or len(response) <= max_output_length:
        return response

    working = dict(data)
    truncated = False
    while len(response) > max_output_length:
        collection_keys = [
            key
            for key, value in working.items()
            if isinstance(value, (list, dict)) and len(value) > 0
        ]
        if not collection_keys:
            break
        target_key = max(
            collection_keys,
            key=lambda key: len(json.dumps(working[key], ensure_ascii=False)),
        )
        target_value = working[target_key]
        if isinstance(target_value, list):
            working[target_key] = target_value[: len(target_value) // 2]
        else:
            sub_keys = list(target_value.keys())
            working[target_key] = {
                sub_key: target_value[sub_key]
                for sub_key in sub_keys[: len(sub_keys) // 2]
            }
        truncated = True
        response = _success(**{field_name: working, "truncated": truncated})

    keys = list(working.keys())
    while len(response) > max_output_length and keys:
        keys = keys[: len(keys) // 2]
        working = {key: working[key] for key in keys}
        truncated = True
        response = _success(**{field_name: working, "truncated": truncated})
    return response


def _require_clean_identifier(value: str, field_name: str) -> str:
    """Reject an empty or whitespace-padded id rather than silently fixing it.

    An id copy-pasted or concatenated by a caller with accidental whitespace
    is more likely a bug worth surfacing than a value to repair - repairing
    it would mask the bug and could send a query for a different object.
    Use this for ids that go into a JSON request body; for ids interpolated
    into a URL path, use _url_path_id instead - encoding (not just
    rejecting whitespace) is what actually closes path/query injection.
    """
    if not value or value.strip() != value:
        raise ValueError(
            f"{field_name} must be a non-empty id with no surrounding whitespace"
        )
    return value


def _url_path_id(value: str, field_name: str) -> str:
    """Validate then percent-encode an id for safe interpolation into a URL
    path segment.

    Percent-encoding - not a blocklist of "/", "?", "#", or ".." - is what
    actually prevents a value like "x?limit=1&foo=/reports/metrics" from
    escaping its intended path segment: any character that could do that
    gets encoded regardless of which one it is, rather than relying on an
    enumeration that could miss one.
    """
    _require_clean_identifier(value, field_name)
    return quote(value, safe="")


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


def _list_association_ids(path: str, max_results: int) -> tuple[list[Any], bool]:
    """Collect associated object ids across pages, up to ``max_results``.

    Follows the ``paging.next.after`` cursor so results beyond the API's
    default page size are not silently dropped. Returns the collected ids and
    whether more associations remain on the server.
    """
    ids: list[Any] = []
    after: str | None = None
    while True:
        params: dict[str, Any] = {
            "limit": min(_ASSOCIATION_PAGE_SIZE, max_results - len(ids))
        }
        if after:
            params["after"] = after
        page = _request(
            "GET", path, params=params, timeout=ASSOCIATION_LISTING_TIMEOUT_SECONDS
        )
        results = page.get("results", [])
        ids.extend(item.get("id") for item in results)
        after = ((page.get("paging") or {}).get("next") or {}).get("after")
        if len(ids) >= max_results:
            return ids[:max_results], bool(after) or len(ids) > max_results
        if not after:
            return ids, False
        if not results:
            # A page with no results but a next cursor would loop forever.
            return ids, True


def _parse_properties(properties_json: str) -> dict[str, Any]:
    properties = json.loads(properties_json)
    if not isinstance(properties, dict):
        raise ValueError("properties_json must be a JSON object of property values")
    return properties


def _search(
    object_type: str, query: str, properties: list[str], limit: int
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "query": query,
        "properties": properties,
        "limit": max(1, min(limit, 100)),
    }
    result = _request("POST", f"/crm/v3/objects/{object_type}/search", body=body)
    return {
        "total": result.get("total", 0),
        "results": [
            {"id": item.get("id"), "properties": item.get("properties", {})}
            for item in result.get("results", [])
        ],
    }


@mcp.tool()
def hubspot_search_contacts(query: str, limit: int = 10) -> str:
    """
    Search HubSpot contacts by free-text query (matches name, email, phone, company).
    Always search before creating a contact to avoid duplicates.
    """
    try:
        found = _search("contacts", query, DEFAULT_CONTACT_PROPERTIES, limit)
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
def hubspot_search_companies(query: str, limit: int = 10) -> str:
    """
    Search HubSpot companies by free-text query (matches name, domain).
    """
    try:
        found = _search("companies", query, DEFAULT_COMPANY_PROPERTIES, limit)
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


@mcp.tool()
def hubspot_get_contact_deals(contact_id: str, limit: int = 100) -> str:
    """
    List the deals associated with a HubSpot contact, including deal stage,
    pipeline, amount, and close date. Returns at most `limit` deals (max 100);
    `has_more` is true when the contact has additional deals beyond the result.
    """
    try:
        contact_id = _url_path_id(contact_id, "contact_id")
        deal_ids, has_more = _list_association_ids(
            f"/crm/v3/objects/contacts/{contact_id}/associations/deals",
            max(1, min(limit, 100)),
        )
        if not deal_ids:
            return _success(deals=[], has_more=has_more)

        deals = _request(
            "POST",
            "/crm/v3/objects/deals/batch/read",
            body={
                "properties": DEFAULT_DEAL_PROPERTIES,
                "inputs": [{"id": deal_id} for deal_id in deal_ids],
            },
        )
        return _success(
            deals=[
                {"id": item.get("id"), "properties": item.get("properties", {})}
                for item in deals.get("results", [])
            ],
            has_more=has_more,
        )
    except Exception as e:
        logger.error(f"Error getting contact deals: {e}")
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
        note_ids, has_more = _list_association_ids(
            f"/crm/v3/objects/contacts/{contact_id}/associations/notes",
            max(1, min(limit, 100)),
        )
        if not note_ids:
            return _success(notes=[], has_more=has_more)

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
            has_more=has_more,
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
