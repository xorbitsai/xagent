import json
import logging
import os
import re
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

from ...utils.graphql_errors import truncate_error_text
from .utils import setup_proxy_env, success_with_capped_dict, url_path_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("xero-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("xero-mcp")

# The Connections API (which organisations this token can access) and the
# Accounting API live on different hosts -- confirmed against Xero's own
# published OpenAPI spec (github.com/XeroAPI/Xero-OpenAPI/xero_accounting.yaml).
XERO_CONNECTIONS_URL = "https://api.xero.com/connections"
XERO_ACCOUNTING_BASE_URL = "https://api.xero.com/api.xro/2.0"
DEFAULT_TIMEOUT_SECONDS = 30

_INVOICE_TYPES = frozenset({"ACCREC", "ACCPAY"})  # sales invoice / purchase bill
_INVOICE_STATUSES = frozenset(
    {"DRAFT", "SUBMITTED", "AUTHORISED", "PAID", "VOIDED", "DELETED"}
)
_GUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _headers(tenant_id: str = "") -> dict[str, str]:
    access_token = os.environ.get("XERO_ACCESS_TOKEN")
    if not access_token:
        raise ValueError("XERO_ACCESS_TOKEN environment variable is missing")
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if tenant_id:
        # One Xero OAuth connection can be authorized against several
        # organisations ("tenants") at once -- every Accounting API call
        # (everything except listing the connections themselves) must say
        # which one it means via this header. There is no "current"/default
        # tenant the way posthog.py's "@current" project sentinel works:
        # Xero has no such concept, so every tool below requires tenant_id
        # explicitly rather than defaulting it.
        headers["Xero-tenant-id"] = tenant_id
    return headers


def _require_non_blank(value: str, field_name: str) -> str:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value


def _validate_guid(value: str, field_name: str) -> str:
    """Validate a Xero id is a well-formed GUID before it goes into a
    hand-built `where=` filter expression (e.g. `Guid("...")`)  -- a GUID
    can never contain a `"` or otherwise break out of that literal, so this
    closes the filter-injection risk a raw f-string interpolation would
    otherwise have, while also rejecting a malformed id with a clear local
    error instead of an opaque Xero 400."""
    if not _GUID_PATTERN.match(value):
        raise ValueError(f"{field_name} must be a UUID, got {value!r}")
    return value


def _reject_quote(value: str, field_name: str) -> str:
    """Reject a value containing '"' or '\\' before it's interpolated into a
    Xero `where=` string literal -- none of the values this is applied to
    (a status keyword, an account type keyword) should ever legitimately
    contain one, so rejecting outright is simpler and safer than attempting
    to escape Xero's filter-expression quoting rules. '\\' is rejected too
    since Xero's where-clause strings support backslash-escaping a quote --
    without this, a trailing backslash could consume the literal's own
    closing quote from the template rather than the caller's value.
    """
    if '"' in value or "\\" in value:
        raise ValueError(f"{field_name} must not contain a '\"' or '\\' character")
    return value


def _extract_error_detail(response: requests.Response) -> str | None:
    """Pull the human-readable message out of a Xero error body.

    Xero's Error schema is {"ErrorNumber", "Type", "Message", "Elements":
    [{"ValidationErrors": [{"Message": ...}]}]} -- Message alone is often
    generic ("A validation exception occurred"), so per-field validation
    messages nested under Elements are folded in when present. Returns
    None so the caller falls back to the raw response text when the body
    isn't in this shape at all.
    """
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    message = payload.get("Message")
    if not isinstance(message, str):
        message = None
    validation_messages = []
    elements = payload.get("Elements")
    if isinstance(elements, list):
        for element in elements:
            if not isinstance(element, dict):
                continue
            validation_errors = element.get("ValidationErrors")
            if not isinstance(validation_errors, list):
                continue
            for validation_error in validation_errors:
                if isinstance(validation_error, dict):
                    detail = validation_error.get("Message")
                    if isinstance(detail, str) and detail:
                        validation_messages.append(detail)
    if validation_messages:
        joined = "; ".join(validation_messages)
        return f"{message}: {joined}" if message else joined
    return message


def _request(
    method: str,
    url: str,
    *,
    tenant_id: str = "",
    params: dict[str, Any] | None = None,
    json_data: dict[str, Any] | None = None,
) -> Any:
    try:
        response = requests.request(
            method=method,
            url=url,
            headers=_headers(tenant_id),
            params=params,
            json=json_data,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise RuntimeError(str(exc)) from exc

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        message = str(exc)
        detail = _extract_error_detail(response)
        if detail is None:
            detail = truncate_error_text(response.text.strip())
        if detail:
            message = f"{message} - {detail}"
        raise RuntimeError(message) from exc

    if response.status_code == 204 or not response.content:
        return {}
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Xero returned a 2xx response with a non-JSON body: {exc}"
        ) from exc


def _accounting_request(
    method: str,
    tenant_id: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_data: dict[str, Any] | None = None,
) -> Any:
    _require_non_blank(tenant_id, "tenant_id")
    return _request(
        method,
        f"{XERO_ACCOUNTING_BASE_URL}{path}",
        tenant_id=tenant_id,
        params=params,
        json_data=json_data,
    )


def _list_items(result: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(result, dict):
        items = result.get(key)
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def _first_item(result: Any, key: str, not_found_message: str) -> dict[str, Any]:
    """Unwrap Xero's plural response envelope ({"Contacts": [...]},
    {"Invoices": [...]}, ...) and return its first element -- shared by
    every single-object get/create/update tool below, which all hit this
    same "envelope with exactly one element" shape."""
    items = _list_items(result, key)
    if not items:
        raise ValueError(not_found_message)
    first: dict[str, Any] = items[0]
    return first


def _contact_summary(contact: dict[str, Any]) -> dict[str, Any]:
    return {
        "contact_id": contact.get("ContactID"),
        "name": contact.get("Name"),
        "first_name": contact.get("FirstName"),
        "last_name": contact.get("LastName"),
        "email": contact.get("EmailAddress"),
        "status": contact.get("ContactStatus"),
        "is_customer": contact.get("IsCustomer"),
        "is_supplier": contact.get("IsSupplier"),
        "updated_date_utc": contact.get("UpdatedDateUTC"),
    }


def _invoice_summary(invoice: dict[str, Any]) -> dict[str, Any]:
    contact = invoice.get("Contact")
    if not isinstance(contact, dict):
        contact = {}
    return {
        "invoice_id": invoice.get("InvoiceID"),
        "invoice_number": invoice.get("InvoiceNumber"),
        "type": invoice.get("Type"),
        "status": invoice.get("Status"),
        "contact_id": contact.get("ContactID"),
        "contact_name": contact.get("Name"),
        "date": invoice.get("DateString") or invoice.get("Date"),
        "due_date": invoice.get("DueDateString") or invoice.get("DueDate"),
        "sub_total": invoice.get("SubTotal"),
        "total_tax": invoice.get("TotalTax"),
        "total": invoice.get("Total"),
        "amount_due": invoice.get("AmountDue"),
        "amount_paid": invoice.get("AmountPaid"),
        "currency_code": invoice.get("CurrencyCode"),
    }


def _account_summary(account: dict[str, Any]) -> dict[str, Any]:
    return {
        "account_id": account.get("AccountID"),
        "code": account.get("Code"),
        "name": account.get("Name"),
        "type": account.get("Type"),
        "tax_type": account.get("TaxType"),
        "status": account.get("Status"),
        "class": account.get("Class"),
    }


def _payment_summary(payment: dict[str, Any]) -> dict[str, Any]:
    invoice = payment.get("Invoice")
    if not isinstance(invoice, dict):
        invoice = {}
    account = payment.get("Account")
    if not isinstance(account, dict):
        account = {}
    return {
        "payment_id": payment.get("PaymentID"),
        "amount": payment.get("Amount"),
        "date": payment.get("Date"),
        "status": payment.get("Status"),
        "invoice_id": invoice.get("InvoiceID"),
        "invoice_number": invoice.get("InvoiceNumber"),
        "account_name": account.get("Name"),
    }


def _build_line_items(line_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate caller-supplied line items (snake_case keys: description,
    quantity, unit_amount, account_code) into Xero's PascalCase LineItem
    shape. A line item with only a description is valid per Xero's own
    docs (used for a text-only line with no amount), so quantity/
    unit_amount/account_code are all optional here."""
    built = []
    for index, item in enumerate(line_items):
        if not isinstance(item, dict):
            raise ValueError(f"line_items[{index}] must be an object")
        description = item.get("description")
        if not description or not str(description).strip():
            raise ValueError(f"line_items[{index}].description must not be blank")
        xero_item: dict[str, Any] = {"Description": description}
        quantity = item.get("quantity")
        if quantity is not None:
            if not isinstance(quantity, (int, float)):
                raise ValueError(f"line_items[{index}].quantity must be a number")
            xero_item["Quantity"] = quantity
        unit_amount = item.get("unit_amount")
        if unit_amount is not None:
            if not isinstance(unit_amount, (int, float)):
                raise ValueError(f"line_items[{index}].unit_amount must be a number")
            xero_item["UnitAmount"] = unit_amount
        if item.get("account_code"):
            xero_item["AccountCode"] = item["account_code"]
        built.append(xero_item)
    return built


@mcp.tool()
def xero_list_organisations() -> str:
    """
    List the Xero organisations ("tenants") this connection is authorized
    for -- tenant_id and name. Use the returned tenant_id with every other
    tool here; Xero has no default organisation, so it must always be
    passed explicitly.
    """
    try:
        result = _request("GET", XERO_CONNECTIONS_URL)
        connections = result if isinstance(result, list) else []
        organisations = [
            {
                "tenant_id": c.get("tenantId"),
                "tenant_name": c.get("tenantName"),
                "tenant_type": c.get("tenantType"),
            }
            for c in connections
            if isinstance(c, dict)
        ]
        return _success(organisations=organisations)
    except Exception as e:
        logger.error(f"Error listing Xero organisations: {e}")
        return _error(str(e))


@mcp.tool()
def xero_get_organisation(tenant_id: str) -> str:
    """
    Get basic details (name, legal name, base currency, timezone) about one
    Xero organisation.
    tenant_id: from xero_list_organisations.
    """
    try:
        result = _accounting_request("GET", tenant_id, "/Organisation")
        organisation = _first_item(
            result, "Organisations", "Xero returned no organisation data"
        )
        return _success(organisation=organisation)
    except Exception as e:
        logger.error(f"Error fetching Xero organisation {tenant_id}: {e}")
        return _error(str(e))


@mcp.tool()
def xero_list_contacts(tenant_id: str, search_term: str = "", page: int = 1) -> str:
    """
    Search/list contacts (customers and suppliers), most recently updated
    first is not guaranteed -- pass search_term to filter.
    search_term: optional case-insensitive substring matched against name,
    first/last name, contact number, and email.
    page: 1-based page number; Xero returns up to 100 contacts per page.
    """
    try:
        params: dict[str, Any] = {"page": max(1, page)}
        if search_term:
            params["searchTerm"] = search_term
        result = _accounting_request("GET", tenant_id, "/Contacts", params=params)
        contacts = _list_items(result, "Contacts")
        pagination = result.get("pagination") if isinstance(result, dict) else None
        if not isinstance(pagination, dict):
            pagination = {}
        return success_with_capped_dict(
            "contacts",
            {
                "contacts": [_contact_summary(c) for c in contacts],
                "page": pagination.get("page"),
                "page_count": pagination.get("pageCount"),
                "item_count": pagination.get("itemCount"),
            },
        )
    except Exception as e:
        logger.error(f"Error listing Xero contacts: {e}")
        return _error(str(e))


@mcp.tool()
def xero_get_contact(tenant_id: str, contact_id: str) -> str:
    """
    Get a Xero contact by id.
    """
    try:
        encoded_id = url_path_id(contact_id, "contact_id")
        result = _accounting_request("GET", tenant_id, f"/Contacts/{encoded_id}")
        contact = _first_item(result, "Contacts", f"Contact '{contact_id}' not found")
        return _success(contact=_contact_summary(contact))
    except Exception as e:
        logger.error(f"Error fetching Xero contact {contact_id}: {e}")
        return _error(str(e))


@mcp.tool()
def xero_create_contact(
    tenant_id: str, name: str, email: str = "", phone: str = ""
) -> str:
    """
    Create a new contact.
    name: the contact's display name (required, must be unique in the org).
    email: optional email address.
    phone: optional default phone number.
    """
    try:
        _require_non_blank(name, "name")
        contact: dict[str, Any] = {"Name": name}
        if email:
            contact["EmailAddress"] = email
        if phone:
            contact["Phones"] = [{"PhoneType": "DEFAULT", "PhoneNumber": phone}]
        result = _accounting_request(
            "POST", tenant_id, "/Contacts", json_data={"Contacts": [contact]}
        )
        created = _first_item(
            result, "Contacts", "Xero did not return the created contact"
        )
        return _success(contact=_contact_summary(created))
    except Exception as e:
        logger.error(f"Error creating Xero contact: {e}")
        return _error(str(e))


@mcp.tool()
def xero_update_contact(
    tenant_id: str,
    contact_id: str,
    name: str | None = None,
    email: str | None = None,
    phone: str | None = None,
) -> str:
    """
    Update an existing contact. Only the fields explicitly provided (not
    None) are changed. phone replaces the contact's *entire* phone number
    list with this single default number (Xero has no per-type patch) --
    pass phone="" to remove all phone numbers, or leave phone unset to
    leave the existing numbers untouched.
    """
    try:
        encoded_id = url_path_id(contact_id, "contact_id")
        contact: dict[str, Any] = {}
        if name is not None:
            _require_non_blank(name, "name")
            contact["Name"] = name
        if email is not None:
            contact["EmailAddress"] = email
        if phone is not None:
            contact["Phones"] = (
                [{"PhoneType": "DEFAULT", "PhoneNumber": phone}] if phone else []
            )
        if not contact:
            return _error("at least one field to update must be provided")

        result = _accounting_request(
            "POST",
            tenant_id,
            f"/Contacts/{encoded_id}",
            json_data={"Contacts": [contact]},
        )
        updated = _first_item(
            result, "Contacts", "Xero did not return the updated contact"
        )
        return _success(contact=_contact_summary(updated))
    except Exception as e:
        logger.error(f"Error updating Xero contact {contact_id}: {e}")
        return _error(str(e))


@mcp.tool()
def xero_list_invoices(
    tenant_id: str, status: str = "", contact_id: str = "", page: int = 1
) -> str:
    """
    Search/list invoices (sales invoices and purchase bills).
    status: optional filter, one of "DRAFT", "SUBMITTED", "AUTHORISED",
    "PAID", "VOIDED", "DELETED".
    contact_id: optional contact id (a UUID) to filter by, from
    xero_list_contacts.
    page: 1-based page number; Xero returns up to 100 invoices per page.
    """
    try:
        max_page = max(1, page)
        params: dict[str, Any] = {"page": max_page}
        where_clauses = []
        if status:
            if status not in _INVOICE_STATUSES:
                return _error(
                    f"status must be one of {sorted(_INVOICE_STATUSES)}, got {status!r}"
                )
            where_clauses.append(f'Status=="{status}"')
        if contact_id:
            _validate_guid(contact_id, "contact_id")
            where_clauses.append(f'Contact.ContactID==Guid("{contact_id}")')
        if where_clauses:
            params["where"] = " AND ".join(where_clauses)
        result = _accounting_request("GET", tenant_id, "/Invoices", params=params)
        invoices = _list_items(result, "Invoices")
        return success_with_capped_dict(
            "invoices", {"invoices": [_invoice_summary(i) for i in invoices]}
        )
    except Exception as e:
        logger.error(f"Error listing Xero invoices: {e}")
        return _error(str(e))


@mcp.tool()
def xero_get_invoice(tenant_id: str, invoice_id: str) -> str:
    """
    Get a Xero invoice by id, including its line items.
    """
    try:
        encoded_id = url_path_id(invoice_id, "invoice_id")
        result = _accounting_request("GET", tenant_id, f"/Invoices/{encoded_id}")
        invoice = _first_item(result, "Invoices", f"Invoice '{invoice_id}' not found")
        summary = _invoice_summary(invoice)
        line_items = invoice.get("LineItems")
        summary["line_items"] = line_items if isinstance(line_items, list) else []
        return success_with_capped_dict("invoice", summary)
    except Exception as e:
        logger.error(f"Error fetching Xero invoice {invoice_id}: {e}")
        return _error(str(e))


@mcp.tool()
def xero_create_invoice(
    tenant_id: str,
    contact_id: str,
    line_items: list[dict[str, Any]],
    invoice_type: str = "ACCREC",
    date: str = "",
    due_date: str = "",
    status: str = "DRAFT",
) -> str:
    """
    Create a new invoice.
    contact_id: the customer's (or, for a bill, the supplier's) contact id,
    from xero_list_contacts.
    line_items: a list of objects, each with "description" (required),
    and optionally "quantity", "unit_amount", and "account_code" (from
    xero_list_accounts) -- e.g.
    [{"description": "Consulting", "quantity": 2, "unit_amount": 150.0,
    "account_code": "200"}].
    invoice_type: "ACCREC" (sales invoice, default) or "ACCPAY" (bill you
    owe a supplier).
    date, due_date: optional "YYYY-MM-DD" dates; Xero defaults both to
    today if omitted.
    status: "DRAFT" (default), "SUBMITTED", or "AUTHORISED" (AUTHORISED
    makes the invoice final and visible to the customer/supplier).
    """
    try:
        _require_non_blank(contact_id, "contact_id")
        if invoice_type not in _INVOICE_TYPES:
            return _error(f"invoice_type must be one of {sorted(_INVOICE_TYPES)}")
        if status not in _INVOICE_STATUSES:
            return _error(f"status must be one of {sorted(_INVOICE_STATUSES)}")
        if not line_items:
            return _error("line_items must contain at least one item")
        invoice: dict[str, Any] = {
            "Type": invoice_type,
            "Contact": {"ContactID": contact_id},
            "LineItems": _build_line_items(line_items),
            "Status": status,
        }
        if date:
            invoice["Date"] = date
        if due_date:
            invoice["DueDate"] = due_date
        result = _accounting_request(
            "POST", tenant_id, "/Invoices", json_data={"Invoices": [invoice]}
        )
        created = _first_item(
            result, "Invoices", "Xero did not return the created invoice"
        )
        return _success(invoice=_invoice_summary(created))
    except Exception as e:
        logger.error(f"Error creating Xero invoice: {e}")
        return _error(str(e))


@mcp.tool()
def xero_update_invoice_status(tenant_id: str, invoice_id: str, status: str) -> str:
    """
    Change an invoice's status -- e.g. move a DRAFT to AUTHORISED, or VOID
    an invoice that shouldn't have been raised.
    status: one of "AUTHORISED", "DELETED" (DRAFT/SUBMITTED invoices only),
    "VOIDED" (AUTHORISED invoices only), "SUBMITTED".
    """
    try:
        if status not in _INVOICE_STATUSES:
            return _error(f"status must be one of {sorted(_INVOICE_STATUSES)}")
        encoded_id = url_path_id(invoice_id, "invoice_id")
        result = _accounting_request(
            "POST",
            tenant_id,
            f"/Invoices/{encoded_id}",
            json_data={"Invoices": [{"Status": status}]},
        )
        updated = _first_item(
            result, "Invoices", "Xero did not return the updated invoice"
        )
        return _success(invoice=_invoice_summary(updated))
    except Exception as e:
        logger.error(f"Error updating Xero invoice {invoice_id}: {e}")
        return _error(str(e))


@mcp.tool()
def xero_list_accounts(tenant_id: str, account_type: str = "") -> str:
    """
    List the chart of accounts. Xero does not paginate this endpoint, so a
    very large chart of accounts may be truncated in the response (see the
    truncated flag) -- narrow with account_type if you hit that.
    account_type: optional filter, e.g. "BANK", "REVENUE", "EXPENSE",
    "CURRENT", "FIXED".
    """
    try:
        params: dict[str, Any] = {}
        if account_type:
            _reject_quote(account_type, "account_type")
            params["where"] = f'Type=="{account_type}"'
        result = _accounting_request("GET", tenant_id, "/Accounts", params=params)
        accounts = _list_items(result, "Accounts")
        return success_with_capped_dict(
            "accounts", {"accounts": [_account_summary(a) for a in accounts]}
        )
    except Exception as e:
        logger.error(f"Error listing Xero accounts: {e}")
        return _error(str(e))


@mcp.tool()
def xero_list_payments(tenant_id: str, page: int = 1) -> str:
    """
    List payments applied to invoices, most recently updated first.
    page: 1-based page number; Xero returns up to 100 payments per page.
    """
    try:
        result = _accounting_request(
            "GET", tenant_id, "/Payments", params={"page": max(1, page)}
        )
        payments = _list_items(result, "Payments")
        return success_with_capped_dict(
            "payments", {"payments": [_payment_summary(p) for p in payments]}
        )
    except Exception as e:
        logger.error(f"Error listing Xero payments: {e}")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
