import json
import logging
import os
from typing import Any

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
logger = logging.getLogger("myob-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("myob-mcp")

MYOB_BASE_URL = "https://api.myob.com/accountright/"
DEFAULT_TIMEOUT_SECONDS = 30
# Matches zoom.py's/salesforce.py's convention: an error body that isn't the
# expected {"Errors": [...]} shape (e.g. an HTML gateway error page) must not
# be forwarded to the LLM/logs verbatim and unbounded.
MAX_ERROR_RESPONSE_TEXT_CHARS = 1000
# MYOB's own default page size (per the reference uptick/pymyob client) is
# 400 -- far larger than this connector's default, which is sized for the
# LLM's output budget rather than MYOB's own server-side ceiling.
DEFAULT_PAGE_LIMIT = 20
MAX_PAGE_LIMIT = 100


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _business_id() -> str:
    business_id = os.environ.get("MYOB_BUSINESS_ID")
    if not business_id:
        raise ValueError("MYOB_BUSINESS_ID environment variable is missing")
    # Interpolated directly into every request URL below (_base_url) --
    # validated/percent-encoded the same way every other uid in this file
    # is, for house consistency, even though this specific value already
    # passed GUID validation once at connect time
    # (_normalize_myob_business_id in api/auth.py).
    return url_path_id(business_id, "MYOB_BUSINESS_ID")


def _base_url() -> str:
    return f"{MYOB_BASE_URL}{_business_id()}/"


def _headers() -> dict[str, str]:
    access_token = os.environ.get("MYOB_ACCESS_TOKEN")
    if not access_token:
        raise ValueError("MYOB_ACCESS_TOKEN environment variable is missing")
    api_key = os.environ.get("MYOB_API_KEY")
    if not api_key:
        raise ValueError("MYOB_API_KEY environment variable is missing")
    return {
        "Authorization": f"Bearer {access_token}",
        # Identifies the calling application (this deployment's own OAuth
        # client id), not the end user -- x-myobapi-cftoken (company-file
        # username/password auth) is a dead concept as of the current MYOB
        # auth model and deliberately not sent here; the OAuth token above
        # is sufficient on its own.
        "x-myobapi-key": api_key,
        "x-myobapi-version": "v2",
    }


def _extract_error_detail(response: requests.Response) -> str | None:
    """Pull the human-readable message out of a MYOB error body.

    MYOB error responses are ``{"Errors": [{"Name", "Message",
    "AdditionalDetails"}, ...]}`` -- joining every entry's Name/Message is
    more useful to the LLM than the raw envelope. Returns None if the body
    isn't in that shape, so the caller can fall back to the raw response
    text.
    """
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    errors = payload.get("Errors")
    if not isinstance(errors, list) or not errors:
        return None
    parts = []
    for entry in errors:
        if not isinstance(entry, dict):
            parts.append(str(entry))
            continue
        name = entry.get("Name") or ""
        message = entry.get("Message") or ""
        # AdditionalDetails is usually a plain string, but at least one
        # documented MYOB error shape carries it as a list of per-field
        # messages -- joined into readable text here rather than left to
        # fall through to the generic str(part) below, which would embed a
        # raw Python list repr (e.g. "['TaxCode is required']") in the
        # message surfaced to the LLM.
        raw_details = entry.get("AdditionalDetails")
        details = (
            ", ".join(str(item) for item in raw_details if item)
            if isinstance(raw_details, list)
            else raw_details or ""
        )
        parts.append(" ".join(str(part) for part in (name, message, details) if part))
    return "; ".join(part for part in parts if part) or None


def _raise_for_status(response: requests.Response) -> None:
    if response.status_code < 400:
        return
    detail = _extract_error_detail(response)
    if detail is None:
        detail = response.text.strip()
    if detail and len(detail) > MAX_ERROR_RESPONSE_TEXT_CHARS:
        detail = detail[:MAX_ERROR_RESPONSE_TEXT_CHARS] + "... [truncated]"
    message = f"MYOB API error (status {response.status_code})"
    if detail:
        message = f"{message}: {detail}"
    raise RuntimeError(message)


def _raw_request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_data: dict[str, Any] | None = None,
) -> requests.Response:
    return requests.request(
        method=method,
        url=f"{_base_url()}{path}",
        headers=_headers(),
        params=params,
        json=json_data,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )


def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_data: dict[str, Any] | None = None,
) -> Any:
    response = _raw_request(method, path, params=params, json_data=json_data)
    _raise_for_status(response)
    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


def _list_items(result: Any) -> tuple[list[Any], int | None]:
    """Normalize a MYOB list response into (items, total_count).

    MYOB's documented list envelope is ``{"Count": N, "Items": [...],
    "NextPageLink": ...}``, but the reference pymyob client passes the raw
    response straight through without asserting that shape, so this also
    tolerates a bare JSON array in case a given endpoint (or a future MYOB
    API revision) returns one directly, rather than crashing on
    ``.get("Items")`` against a list.

    total_count is None for the bare-array shape, not len(result): that
    length is only the current page, not MYOB's true across-all-pages
    total the way ``Count`` is -- returning it as total_count would make
    _success_with_capped_list's has_more/next_skip pagination silently
    report "no more results" the moment this fallback fires, even when
    more genuinely exist server-side.
    """
    if isinstance(result, dict):
        items = result.get("Items")
        return (items if isinstance(items, list) else [], result.get("Count"))
    if isinstance(result, list):
        return result, None
    return [], None


def _success_with_capped_list(
    list_field: str,
    items: list[Any],
    *,
    total_count: int | None,
    skip: int,
    **extra: Any,
) -> str:
    """Build a success payload, halving ``items`` until the response fits
    the platform's output limit.

    ``next_skip``/``has_more`` reflect how many items actually made it into
    this response (post-halving), the same convention salesforce.py's own
    offset-based capped-page helper uses -- a caller resuming from
    ``next_skip`` picks up exactly where this response left off, including
    any items a halving pass dropped, rather than jumping straight to
    ``skip + top`` and silently skipping over them the way returning a bare
    ``truncated: true`` with no cursor would.
    """

    def _build(items: list[Any], truncated: bool) -> str:
        next_skip = skip + len(items)
        has_more = total_count is not None and next_skip < total_count
        payload: dict[str, Any] = {
            list_field: items,
            "truncated": truncated,
            "has_more": has_more,
            **extra,
        }
        if total_count is not None:
            payload["total_count"] = total_count
        if has_more:
            payload["next_skip"] = next_skip
        return _success(**payload)

    max_output_length = get_tool_max_output_length()
    response = _build(items, False)
    while len(response) > max_output_length and items:
        items = items[: len(items) // 2]
        response = _build(items, True)
    return response


def _list_params(
    *, filter_str: str, orderby: str, top: int, skip: int
) -> dict[str, Any]:
    top = clamp_limit(top, max_limit=MAX_PAGE_LIMIT)
    skip = clamp_offset(skip)
    params: dict[str, Any] = {"$top": top, "$skip": skip}
    if filter_str:
        params["$filter"] = filter_str
    if orderby:
        params["$orderby"] = orderby
    return params


def _create_resource(path: str, fields: dict[str, Any]) -> dict[str, Any]:
    """POST a new resource, returning its full body.

    MYOB's write endpoints only include the created/updated body in the
    response when ``returnBody=true`` is passed -- sent on every write here,
    matching the reference pymyob client's own build_request_kwargs (which
    adds it unconditionally for PUT/POST). Falls back to the ``Location``
    response header's trailing id segment on the rare chance a 200/201 comes
    back with no body despite that.
    """
    response = _raw_request(
        "POST", path, params={"returnBody": "true"}, json_data=fields
    )
    _raise_for_status(response)
    if response.content:
        body: dict[str, Any] = response.json()
        return body
    location = response.headers.get("Location", "").rstrip("/")
    uid = location.rsplit("/", 1)[-1] if location else None
    return {"UID": uid} if uid else {}


def _update_resource(path: str, uid: str, fields: dict[str, Any]) -> dict[str, Any]:
    """PUT changes onto an existing resource, returning its full body.

    MYOB's PUT replaces the entire resource and enforces optimistic
    concurrency via a ``RowVersion`` field embedded in the object body
    itself (not an HTTP header) -- there is no partial-update verb. Rather
    than requiring every caller to round-trip the object's other fields and
    RowVersion themselves, this fetches the current body first and merges
    the caller's changes on top of it, so a caller only has to name what's
    actually changing.

    This is still a GET-then-PUT with no retry: a concurrent edit landing
    between the two (another tool call, or a change made directly in MYOB)
    is a classic lost-update race -- MYOB's own RowVersion check makes the
    PUT fail cleanly with a 409 rather than silently overwriting, but this
    function doesn't re-fetch and retry on that, it just surfaces the
    error. Not unique to this connector (other connectors in this codebase
    PUT the same way), so left as a known tradeoff rather than solved here.
    """
    safe_uid = url_path_id(uid, "uid")
    current = _request("GET", f"{path}{safe_uid}/")
    # Both checks needed, not just one: `not current` alone lets a truthy
    # non-dict (e.g. a bare list) through, which would blow up as an opaque
    # TypeError at the dict-spread below instead of this clear message;
    # `isinstance` alone lets an empty {} through -- _request returns {} on
    # a 204/empty response, and {} would otherwise pass straight through to
    # `merged = {**{}, **fields}` below, silently wiping every field this
    # record has other than the ones the caller named (including
    # RowVersion, which MYOB needs back on the PUT for optimistic
    # concurrency).
    if not current or not isinstance(current, dict):
        raise RuntimeError(f"MYOB returned no existing record for uid {uid}")
    merged = {**current, **fields}
    response = _raw_request(
        "PUT", f"{path}{safe_uid}/", params={"returnBody": "true"}, json_data=merged
    )
    _raise_for_status(response)
    return response.json() if response.content else merged


@mcp.tool()
def myob_get_business_info() -> str:
    """
    Get the connected business's own details (name, product version, and
    other company-file-level fields).
    """
    try:
        result = _request("GET", "")
        business = result.get("CompanyFile") if isinstance(result, dict) else None
        if not isinstance(business, dict):
            return _error("MYOB returned no CompanyFile data")
        return success_with_capped_dict("business", business)
    except Exception as e:
        logger.error(f"Error getting MYOB business info: {e}")
        return _error(str(e))


@mcp.tool()
def myob_list_customers(
    filter: str = "", orderby: str = "", top: int = DEFAULT_PAGE_LIMIT, skip: int = 0
) -> str:
    """
    List customer contacts.
    filter: an optional raw OData $filter expression, e.g. "IsActive eq true"
    or "substringof('Acme', CompanyName)".
    orderby: an optional OData $orderby expression, e.g. "CompanyName" or
    "CompanyName desc".
    top/skip: server-side page size (max 100) and offset.
    """
    try:
        params = _list_params(filter_str=filter, orderby=orderby, top=top, skip=skip)
        result = _request("GET", "Contact/Customer/", params=params)
        items, total_count = _list_items(result)
        return _success_with_capped_list(
            "customers", items, total_count=total_count, skip=params["$skip"]
        )
    except Exception as e:
        logger.error(f"Error listing MYOB customers: {e}")
        return _error(str(e))


@mcp.tool()
def myob_get_customer(uid: str) -> str:
    """Get one customer contact by its UID."""
    try:
        safe_uid = url_path_id(uid, "uid")
        result = _request("GET", f"Contact/Customer/{safe_uid}/")
        return success_with_capped_dict("customer", result)
    except Exception as e:
        logger.error(f"Error getting MYOB customer {uid}: {e}")
        return _error(str(e))


@mcp.tool()
def myob_create_customer(fields: dict[str, Any]) -> str:
    """
    Create a new customer contact.
    fields: MYOB Contact/Customer field name -> value pairs, e.g.
    {"CompanyName": "Acme Pty Ltd", "IsIndividual": False,
    "Addresses": [{"Location": 1, "Email": "billing@acme.example"}]}.
    Use myob_get_customer on an existing record to see the full field shape
    MYOB expects.
    """
    try:
        if not fields:
            return _error("No fields provided to create the customer")
        result = _create_resource("Contact/Customer/", fields)
        return success_with_capped_dict("customer", result)
    except Exception as e:
        logger.error(f"Error creating MYOB customer: {e}")
        return _error(str(e))


@mcp.tool()
def myob_update_customer(uid: str, fields: dict[str, Any]) -> str:
    """
    Update an existing customer contact. Only the fields provided are
    changed; every other field keeps its current value.
    fields: MYOB Contact/Customer field name -> value pairs to change.
    """
    try:
        if not fields:
            return _error("No fields provided to update")
        result = _update_resource("Contact/Customer/", uid, fields)
        return success_with_capped_dict("customer", result)
    except Exception as e:
        logger.error(f"Error updating MYOB customer {uid}: {e}")
        return _error(str(e))


@mcp.tool()
def myob_list_suppliers(
    filter: str = "", orderby: str = "", top: int = DEFAULT_PAGE_LIMIT, skip: int = 0
) -> str:
    """
    List supplier contacts.
    filter: an optional raw OData $filter expression, e.g. "IsActive eq true"
    or "substringof('Acme', CompanyName)".
    orderby: an optional OData $orderby expression, e.g. "CompanyName".
    top/skip: server-side page size (max 100) and offset.
    """
    try:
        params = _list_params(filter_str=filter, orderby=orderby, top=top, skip=skip)
        result = _request("GET", "Contact/Supplier/", params=params)
        items, total_count = _list_items(result)
        return _success_with_capped_list(
            "suppliers", items, total_count=total_count, skip=params["$skip"]
        )
    except Exception as e:
        logger.error(f"Error listing MYOB suppliers: {e}")
        return _error(str(e))


@mcp.tool()
def myob_get_supplier(uid: str) -> str:
    """Get one supplier contact by its UID."""
    try:
        safe_uid = url_path_id(uid, "uid")
        result = _request("GET", f"Contact/Supplier/{safe_uid}/")
        return success_with_capped_dict("supplier", result)
    except Exception as e:
        logger.error(f"Error getting MYOB supplier {uid}: {e}")
        return _error(str(e))


@mcp.tool()
def myob_create_supplier(fields: dict[str, Any]) -> str:
    """
    Create a new supplier contact.
    fields: MYOB Contact/Supplier field name -> value pairs, e.g.
    {"CompanyName": "Acme Supplies Pty Ltd", "IsIndividual": False}.
    Use myob_get_supplier on an existing record to see the full field shape
    MYOB expects.
    """
    try:
        if not fields:
            return _error("No fields provided to create the supplier")
        result = _create_resource("Contact/Supplier/", fields)
        return success_with_capped_dict("supplier", result)
    except Exception as e:
        logger.error(f"Error creating MYOB supplier: {e}")
        return _error(str(e))


@mcp.tool()
def myob_update_supplier(uid: str, fields: dict[str, Any]) -> str:
    """
    Update an existing supplier contact. Only the fields provided are
    changed; every other field keeps its current value.
    fields: MYOB Contact/Supplier field name -> value pairs to change.
    """
    try:
        if not fields:
            return _error("No fields provided to update")
        result = _update_resource("Contact/Supplier/", uid, fields)
        return success_with_capped_dict("supplier", result)
    except Exception as e:
        logger.error(f"Error updating MYOB supplier {uid}: {e}")
        return _error(str(e))


@mcp.tool()
def myob_list_sales_invoices(
    filter: str = "", orderby: str = "", top: int = DEFAULT_PAGE_LIMIT, skip: int = 0
) -> str:
    """
    List item-type sale invoices (invoices that bill inventory items, MYOB's
    most common invoice layout). Service/professional/time-billing layout
    invoices are not covered by this connector.
    filter: an optional raw OData $filter expression, e.g. "Status eq
    'Open'" or "Customer/UID eq guid'<customer-uid>'".
    orderby: an optional OData $orderby expression, e.g. "Date desc".
    top/skip: server-side page size (max 100) and offset.
    """
    try:
        params = _list_params(filter_str=filter, orderby=orderby, top=top, skip=skip)
        result = _request("GET", "Sale/Invoice/Item/", params=params)
        items, total_count = _list_items(result)
        return _success_with_capped_list(
            "invoices", items, total_count=total_count, skip=params["$skip"]
        )
    except Exception as e:
        logger.error(f"Error listing MYOB sales invoices: {e}")
        return _error(str(e))


@mcp.tool()
def myob_get_sales_invoice(uid: str) -> str:
    """Get one item-type sale invoice by its UID."""
    try:
        safe_uid = url_path_id(uid, "uid")
        result = _request("GET", f"Sale/Invoice/Item/{safe_uid}/")
        return success_with_capped_dict("invoice", result)
    except Exception as e:
        logger.error(f"Error getting MYOB sales invoice {uid}: {e}")
        return _error(str(e))


@mcp.tool()
def myob_create_sales_invoice(fields: dict[str, Any]) -> str:
    """
    Create a new item-type sale invoice.
    fields: MYOB Sale/Invoice/Item field name -> value pairs, e.g.
    {"Customer": {"UID": "<customer-uid>"}, "Lines": [{"Type": "Transaction",
    "Item": {"UID": "<item-uid>"}, "ShipQuantity": 1, "UnitPrice": 100.0}]}.
    Use myob_get_sales_invoice on an existing record to see the full field
    shape MYOB expects.
    """
    try:
        if not fields:
            return _error("No fields provided to create the invoice")
        result = _create_resource("Sale/Invoice/Item/", fields)
        return success_with_capped_dict("invoice", result)
    except Exception as e:
        logger.error(f"Error creating MYOB sales invoice: {e}")
        return _error(str(e))


@mcp.tool()
def myob_update_sales_invoice(uid: str, fields: dict[str, Any]) -> str:
    """
    Update an existing item-type sale invoice. Only the fields provided are
    changed; every other field (including existing Lines) keeps its current
    value unless explicitly overwritten.
    fields: MYOB Sale/Invoice/Item field name -> value pairs to change.
    """
    try:
        if not fields:
            return _error("No fields provided to update")
        result = _update_resource("Sale/Invoice/Item/", uid, fields)
        return success_with_capped_dict("invoice", result)
    except Exception as e:
        logger.error(f"Error updating MYOB sales invoice {uid}: {e}")
        return _error(str(e))


@mcp.tool()
def myob_list_purchase_bills(
    filter: str = "", orderby: str = "", top: int = DEFAULT_PAGE_LIMIT, skip: int = 0
) -> str:
    """
    List item-type purchase bills (bills for inventory items, the most
    common bill layout). Service/miscellaneous layout bills are not covered
    by this connector.
    filter: an optional raw OData $filter expression, e.g. "Status eq
    'Open'" or "Supplier/UID eq guid'<supplier-uid>'".
    orderby: an optional OData $orderby expression, e.g. "Date desc".
    top/skip: server-side page size (max 100) and offset.
    """
    try:
        params = _list_params(filter_str=filter, orderby=orderby, top=top, skip=skip)
        result = _request("GET", "Purchase/Bill/Item/", params=params)
        items, total_count = _list_items(result)
        return _success_with_capped_list(
            "bills", items, total_count=total_count, skip=params["$skip"]
        )
    except Exception as e:
        logger.error(f"Error listing MYOB purchase bills: {e}")
        return _error(str(e))


@mcp.tool()
def myob_get_purchase_bill(uid: str) -> str:
    """Get one item-type purchase bill by its UID."""
    try:
        safe_uid = url_path_id(uid, "uid")
        result = _request("GET", f"Purchase/Bill/Item/{safe_uid}/")
        return success_with_capped_dict("bill", result)
    except Exception as e:
        logger.error(f"Error getting MYOB purchase bill {uid}: {e}")
        return _error(str(e))


@mcp.tool()
def myob_create_purchase_bill(fields: dict[str, Any]) -> str:
    """
    Create a new item-type purchase bill.
    fields: MYOB Purchase/Bill/Item field name -> value pairs, e.g.
    {"Supplier": {"UID": "<supplier-uid>"}, "Lines": [{"Type": "Transaction",
    "Item": {"UID": "<item-uid>"}, "Quantity": 1, "UnitPrice": 50.0}]}.
    Use myob_get_purchase_bill on an existing record to see the full field
    shape MYOB expects.
    """
    try:
        if not fields:
            return _error("No fields provided to create the bill")
        result = _create_resource("Purchase/Bill/Item/", fields)
        return success_with_capped_dict("bill", result)
    except Exception as e:
        logger.error(f"Error creating MYOB purchase bill: {e}")
        return _error(str(e))


@mcp.tool()
def myob_update_purchase_bill(uid: str, fields: dict[str, Any]) -> str:
    """
    Update an existing item-type purchase bill. Only the fields provided are
    changed; every other field (including existing Lines) keeps its current
    value unless explicitly overwritten.
    fields: MYOB Purchase/Bill/Item field name -> value pairs to change.
    """
    try:
        if not fields:
            return _error("No fields provided to update")
        result = _update_resource("Purchase/Bill/Item/", uid, fields)
        return success_with_capped_dict("bill", result)
    except Exception as e:
        logger.error(f"Error updating MYOB purchase bill {uid}: {e}")
        return _error(str(e))


@mcp.tool()
def myob_list_accounts(
    filter: str = "", orderby: str = "", top: int = DEFAULT_PAGE_LIMIT, skip: int = 0
) -> str:
    """
    List general ledger accounts (the chart of accounts).
    filter: an optional raw OData $filter expression, e.g. "Type eq
    'Expense'".
    orderby: an optional OData $orderby expression, e.g. "DisplayID".
    top/skip: server-side page size (max 100) and offset.
    """
    try:
        params = _list_params(filter_str=filter, orderby=orderby, top=top, skip=skip)
        result = _request("GET", "GeneralLedger/Account/", params=params)
        items, total_count = _list_items(result)
        return _success_with_capped_list(
            "accounts", items, total_count=total_count, skip=params["$skip"]
        )
    except Exception as e:
        logger.error(f"Error listing MYOB accounts: {e}")
        return _error(str(e))


@mcp.tool()
def myob_get_account(uid: str) -> str:
    """Get one general ledger account by its UID."""
    try:
        safe_uid = url_path_id(uid, "uid")
        result = _request("GET", f"GeneralLedger/Account/{safe_uid}/")
        return success_with_capped_dict("account", result)
    except Exception as e:
        logger.error(f"Error getting MYOB account {uid}: {e}")
        return _error(str(e))


@mcp.tool()
def myob_list_tax_codes(
    filter: str = "", orderby: str = "", top: int = DEFAULT_PAGE_LIMIT, skip: int = 0
) -> str:
    """
    List tax codes configured on the business (e.g. GST, FRE, N-T).
    filter: an optional raw OData $filter expression, e.g. "Type eq 'GST'".
    orderby: an optional OData $orderby expression, e.g. "Code".
    top/skip: server-side page size (max 100) and offset.
    """
    try:
        params = _list_params(filter_str=filter, orderby=orderby, top=top, skip=skip)
        result = _request("GET", "GeneralLedger/TaxCode/", params=params)
        items, total_count = _list_items(result)
        return _success_with_capped_list(
            "tax_codes", items, total_count=total_count, skip=params["$skip"]
        )
    except Exception as e:
        logger.error(f"Error listing MYOB tax codes: {e}")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
