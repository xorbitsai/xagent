import json
import logging
import os
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

from .utils import setup_proxy_env

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
# Salesforce keeps every past API version working indefinitely, and the
# SOQL/sobject CRUD surface this connector uses has been stable across
# versions for years.
API_VERSION = "v59.0"
DEFAULT_TIMEOUT_SECONDS = 30
# Matches zoom.py's/linear.py's convention: an error body that isn't the
# expected shape (e.g. an HTML gateway error page) must not be forwarded to
# the LLM/logs verbatim and unbounded.
MAX_ERROR_RESPONSE_TEXT_CHARS = 1000


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _instance_url() -> str:
    """Return the per-org API host this connector's OAuth grant belongs to.

    Salesforce returns this in the token response instead of using a fixed
    API domain -- which org to call is a property of the connected account,
    not something this module can infer.
    """
    instance_url = os.environ.get("SALESFORCE_INSTANCE_URL")
    if not instance_url:
        raise ValueError("SALESFORCE_INSTANCE_URL environment variable is missing")
    return instance_url.rstrip("/")


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
    messages = [
        str(item.get("message") or item) if isinstance(item, dict) else str(item)
        for item in payload
    ]
    return "; ".join(messages) if messages else None


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
        return _success(
            records=result.get("records") or [],
            total_size=result.get("totalSize"),
            truncated=not result.get("done", True),
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
        return _success(results=result.get("searchRecords") or [])
    except Exception as e:
        logger.error(f"Error running Salesforce SOSL search: {e}")
        return _error(str(e))


@mcp.tool()
def salesforce_list_sobjects() -> str:
    """
    List the object types (standard, e.g. Account/Contact/Lead/Opportunity/
    Case, and custom, e.g. ending in "__c") this org exposes -- name, label,
    and whether it's queryable/creatable/updateable/deletable. Use the
    returned name with every other tool here.
    """
    try:
        result = _request("GET", f"/services/data/{API_VERSION}/sobjects")
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
        ]
        return _success(sobjects=sobjects)
    except Exception as e:
        logger.error(f"Error listing Salesforce sobjects: {e}")
        return _error(str(e))


@mcp.tool()
def salesforce_describe_sobject(sobject_type: str) -> str:
    """
    Get an object type's field schema -- name, label, type, and whether
    each field is required/updateable/a picklist (with its valid values).
    Use this before salesforce_create_record/salesforce_update_record to
    learn which fields exist and what values they accept.
    sobject_type: an object's API name, e.g. "Account" or "My_Object__c".
    """
    try:
        result = _request(
            "GET", f"/services/data/{API_VERSION}/sobjects/{sobject_type}/describe"
        )
        fields = [
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
            for f in result.get("fields") or []
        ]
        return _success(
            name=result.get("name"), label=result.get("label"), fields=fields
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
        params: dict[str, Any] = {"fields": fields} if fields else {}
        result = _request(
            "GET",
            f"/services/data/{API_VERSION}/sobjects/{sobject_type}/{record_id}",
            params=params,
        )
        return _success(record=result)
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
        result = _request(
            "POST",
            f"/services/data/{API_VERSION}/sobjects/{sobject_type}",
            json_data=fields,
        )
        return _success(id=result.get("id"), success=result.get("success"))
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
        _request(
            "PATCH",
            f"/services/data/{API_VERSION}/sobjects/{sobject_type}/{record_id}",
            json_data=fields,
        )
        return _success(id=record_id)
    except Exception as e:
        logger.error(
            f"Error updating Salesforce {sobject_type} record {record_id}: {e}"
        )
        return _error(str(e))


@mcp.tool()
def salesforce_delete_record(sobject_type: str, record_id: str) -> str:
    """
    Delete a record by id.
    sobject_type: an object's API name, e.g. "Account" or "My_Object__c".
    """
    try:
        _request(
            "DELETE",
            f"/services/data/{API_VERSION}/sobjects/{sobject_type}/{record_id}",
        )
        return _success(id=record_id)
    except Exception as e:
        logger.error(
            f"Error deleting Salesforce {sobject_type} record {record_id}: {e}"
        )
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
