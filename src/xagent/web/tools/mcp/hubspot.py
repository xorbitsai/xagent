import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

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

# HUBSPOT_DEFINED association type ids for notes.
_NOTE_ASSOCIATION_TYPE_IDS = {"contact": 202, "company": 190, "deal": 214}

_ASSOCIATION_PAGE_SIZE = 100

_VALID_ANALYTICS_BREAKDOWNS = {"total", "day", "week", "month"}


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


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
        targets = {"contact": contact_id, "company": company_id, "deal": deal_id}
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


@mcp.tool()
def hubspot_list_forms(limit: int = 20) -> str:
    """
    List HubSpot marketing forms, including id and full form metadata
    (name, field groups, etc.). Returns at most `limit` forms (max 100);
    `has_more` is true when the portal has additional forms beyond the
    result. Use hubspot_get_form_submissions to pull the submissions
    collected by a specific form.
    """
    try:
        result = _request(
            "GET", "/marketing/v3/forms", params={"limit": max(1, min(limit, 100))}
        )
        has_more = bool((result.get("paging") or {}).get("next"))
        return _success(forms=result.get("results", []), has_more=has_more)
    except Exception as e:
        logger.error(f"Error listing forms: {e}")
        return _error(str(e))


@mcp.tool()
def hubspot_get_form_submissions(form_id: str, limit: int = 20) -> str:
    """
    Get recent submissions for a HubSpot form, including submitted field
    values, submission timestamp, and page URL. Returns at most `limit`
    submissions (max 50); `has_more` is true when the form has additional
    submissions beyond the result.
    """
    try:
        result = _request(
            "GET",
            f"/form-integrations/v1/submissions/forms/{form_id}",
            params={"limit": max(1, min(limit, 50))},
        )
        has_more = bool((result.get("paging") or {}).get("next"))
        return _success(submissions=result.get("results", []), has_more=has_more)
    except Exception as e:
        logger.error(f"Error getting form submissions: {e}")
        return _error(str(e))


@mcp.tool()
def hubspot_get_analytics_report(
    report_type: str,
    breakdown: str = "total",
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """
    Get a HubSpot traffic analytics report. `report_type` is a HubSpot
    analytics dimension such as "sources", "utm-campaigns", "landing-pages",
    "pages", or "geolocation". `breakdown` is "total", "day", "week", or
    "month". `start_date`/`end_date` are optional "YYYYMMDD" strings that
    limit the reporting window.

    Note: this covers HubSpot's traffic/analytics dimensions, not
    custom reports or dashboards built in the HubSpot report editor -
    HubSpot has no public API to read those. Requires a Marketing Hub
    Basic, Professional, or Enterprise account; Free/Starter accounts are
    not supported by this HubSpot API.
    """
    try:
        if breakdown not in _VALID_ANALYTICS_BREAKDOWNS:
            raise ValueError(
                f"breakdown must be one of {sorted(_VALID_ANALYTICS_BREAKDOWNS)}"
            )
        params: dict[str, Any] = {}
        if start_date:
            params["start"] = start_date
        if end_date:
            params["end"] = end_date
        result = _request(
            "GET", f"/analytics/v2/reports/{report_type}/{breakdown}", params=params
        )
        return _success(report=result)
    except Exception as e:
        logger.error(f"Error getting analytics report: {e}")
        return _error(str(e))


@mcp.tool()
def hubspot_list_marketing_emails(limit: int = 20) -> str:
    """
    List HubSpot marketing emails, including id and full email metadata
    (name, subject, state, publish date, etc.). Returns at most `limit`
    emails (max 100); `has_more` is true when the portal has additional
    emails beyond the result. Use hubspot_get_marketing_email_statistics
    for performance metrics on a specific email.
    """
    try:
        result = _request(
            "GET", "/marketing/v3/emails", params={"limit": max(1, min(limit, 100))}
        )
        has_more = bool((result.get("paging") or {}).get("next"))
        return _success(emails=result.get("results", []), has_more=has_more)
    except Exception as e:
        logger.error(f"Error listing marketing emails: {e}")
        return _error(str(e))


@mcp.tool()
def hubspot_get_marketing_email_statistics(email_ids: str) -> str:
    """
    Get send/open/click/bounce statistics for HubSpot marketing emails.
    `email_ids` is a single email id, or a comma-separated list of ids to
    fetch statistics for multiple emails at once.
    """
    try:
        ids = [
            email_id.strip() for email_id in email_ids.split(",") if email_id.strip()
        ]
        result = _request(
            "GET",
            "/marketing/v3/emails/statistics/list",
            # HubSpot's emailIds is an array param: requests serializes a
            # list value as repeated "emailIds=<id>" pairs, matching that
            # shape rather than a single comma-joined value.
            params={"emailIds": ids},
        )
        return _success(statistics=result.get("results", result))
    except Exception as e:
        logger.error(f"Error getting marketing email statistics: {e}")
        return _error(str(e))


@mcp.tool()
def hubspot_list_campaigns(limit: int = 20) -> str:
    """
    List HubSpot marketing campaigns, including id and full campaign
    properties (name, status, date range, etc.). Returns at most `limit`
    campaigns (max 100); `has_more` is true when the portal has additional
    campaigns beyond the result. Use hubspot_get_campaign_metrics for
    performance metrics on a specific campaign.
    """
    try:
        result = _request(
            "GET",
            "/marketing/v3/campaigns",
            params={"limit": max(1, min(limit, 100))},
        )
        has_more = bool((result.get("paging") or {}).get("next"))
        return _success(campaigns=result.get("results", []), has_more=has_more)
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
    `end_date` are optional "YYYY-MM-DD" strings that limit the reporting
    window.
    """
    try:
        params: dict[str, Any] = {}
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date
        result = _request(
            "GET",
            f"/marketing/v3/campaigns/{campaign_id}/reports/metrics",
            params=params,
        )
        return _success(metrics=result)
    except Exception as e:
        logger.error(f"Error getting campaign metrics: {e}")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
