import json
import logging
import os
import re
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable, Iterator, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel

from ....config import get_tool_max_output_length
from .utils import clamp_limit, clamp_offset, setup_proxy_env, url_path_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("xero-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("xero-mcp")

# This connector replaces the third-party ``@xeroapi/xero-mcp-server`` package
# the Xero app used to launch. It keeps that package's tool names and argument
# names (so prompts and memories that mention them keep working) for the
# accounting tools, and drops the Payroll tools: the app is not granted any
# payroll scope, so they could never succeed.
#
# What it changes on purpose:
# * Lists report an exact ``total_count`` / ``has_more`` / ``next_page``.
#   The upstream package fixed the page size at 10 and returned only
#   "Found N invoices", so an agent could not tell whether it had read a list.
# * ``list_overdue_invoices`` reads *every* overdue invoice server-side.
# * ``list_paid_invoices`` reads *every* paid invoice server-side before
#   selecting the requested settlement date.  UpdatedDateUTC order is not a
#   valid boundary for FullyPaidOnDate.
# * A failed call raises, which FastMCP turns into ``isError``. The upstream
#   package returned the failure as ordinary text, so it counted as a success.
# * Authentication failures retain Xero's detail; a 403 also names the OAuth
#   scope the tool needs instead of returning an unexplained generic error.
XERO_API_BASE = "https://api.xero.com/api.xro/2.0/"
XERO_CONNECTIONS_URL = "https://api.xero.com/connections"
DEFAULT_TIMEOUT_SECONDS = 30
# Matches zoom.py's/salesforce.py's convention: an error body that isn't the
# expected shape (e.g. an HTML gateway error page) must not be forwarded to the
# LLM/logs verbatim and unbounded.
MAX_ERROR_RESPONSE_TEXT_CHARS = 1000
# Keep interactive list responses modest even though the Accounting API now
# accepts up to 1000 records per page. Full-ledger scans use the larger value so
# they consume fewer calls and are less likely to hit the scan safety limit.
XERO_MAX_PAGE_SIZE = 100
XERO_BULK_PAGE_SIZE = 1000
DEFAULT_PAGE_SIZE = 20
# Full-ledger tools fail rather than publishing partial values as totals if this
# bound is reached. 50 pages x 1000 invoices covers a 50,000-invoice ledger.
OPEN_INVOICE_MAX_PAGES = 50
PAID_INVOICE_MAX_PAGES = 50
OVERDUE_DEFAULT_LIMIT = 50
OVERDUE_MAX_LIMIT = 200
PAID_INVOICE_DEFAULT_LIMIT = 100
PAID_INVOICE_MAX_LIMIT = 200
INVOICE_HISTORY_MAX_LIMIT = 200
OVERDUE_TOP_CONTACTS = 25
OPEN_INVOICE_EXCLUDES = ["credit_notes", "overpayments", "prepayments"]
DEFAULT_INVOICE_TERMS_DAYS = 30
DEFAULT_QUOTE_VALID_DAYS = 7

INVOICE_STATUSES = frozenset(
    {"DRAFT", "SUBMITTED", "AUTHORISED", "PAID", "VOIDED", "DELETED"}
)

SCOPE_HINTS = {
    "invoices": "accounting.invoices (or accounting.invoices.read)",
    "payments": "accounting.payments (or accounting.payments.read)",
    "banktransactions": "accounting.banktransactions (or accounting.banktransactions.read)",
    "manualjournals": "accounting.manualjournals (or accounting.manualjournals.read)",
    "contacts": "accounting.contacts",
    "settings": "accounting.settings",
    "aged": "accounting.reports.aged.read",
    "profitandloss": "accounting.reports.profitandloss.read",
    "balancesheet": "accounting.reports.balancesheet.read",
    "trialbalance": "accounting.reports.trialbalance.read",
}

READ_ONLY = ToolAnnotations(readOnlyHint=True)
CREATE = ToolAnnotations(destructiveHint=False, idempotentHint=False)
UPDATE = ToolAnnotations(destructiveHint=True, idempotentHint=True)

_GUID_PATTERN = re.compile(r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}")
_MS_DATE_PATTERN = re.compile(r"/Date\((-?\d+)(?:[+-]\d{4})?\)/")

_connection_cache: dict[str, str] = {}

_XERO_TIME_ZONES = {
    "AUSEASTERNSTANDARDTIME": "Australia/Sydney",
    "AUSCENTRALSTANDARDTIME": "Australia/Darwin",
    "AUSCENTRALWSTANDARDTIME": "Australia/Eucla",
    "CENAUSTRALIASTANDARDTIME": "Australia/Adelaide",
    "EAUSTRALIASTANDARDTIME": "Australia/Brisbane",
    "WAUSTRALIASTANDARDTIME": "Australia/Perth",
    "LORDHOWESTANDARDTIME": "Australia/Lord_Howe",
    "TASMANIASTANDARDTIME": "Australia/Hobart",
    "NEWZEALANDSTANDARDTIME": "Pacific/Auckland",
    "CHATHAMISLANDSSTANDARDTIME": "Pacific/Chatham",
    "UTC": "UTC",
}


class XeroAPIError(RuntimeError):
    def __init__(self, message: str, *, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class Tracking(BaseModel):
    name: str
    option: str
    trackingCategoryID: str


class LineItem(BaseModel):
    description: str
    quantity: float
    unitAmount: float
    accountCode: str
    taxType: str
    itemCode: str | None = None
    tracking: list[Tracking] | None = None


class ManualJournalLine(BaseModel):
    lineAmount: float
    accountCode: str
    description: str | None = None
    taxType: str | None = None


class ItemDetails(BaseModel):
    unitPrice: float
    taxType: str | None = None
    accountCode: str | None = None


class Address(BaseModel):
    addressLine1: str
    addressLine2: str | None = None
    city: str | None = None
    region: str | None = None
    postalCode: str | None = None
    country: str | None = None


class TrackingOptionUpdate(BaseModel):
    trackingOptionId: str
    name: str | None = None
    status: Literal["ACTIVE", "ARCHIVED"] | None = None


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Validation helpers. Everything a caller supplies that ends up in a URL path
# or in a Xero ``where`` expression is validated first.
# --------------------------------------------------------------------------- #


def _guid(value: str, field_name: str) -> str:
    cleaned = str(value).strip()
    if not _GUID_PATTERN.fullmatch(cleaned):
        raise ValueError(f"{field_name} must be a Xero GUID, got {value!r}")
    return cleaned


def _guid_list(values: list[str] | None, field_name: str) -> list[str]:
    return [_guid(value, field_name) for value in (values or [])]


def _iso_date(value: str, field_name: str) -> date:
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError:
        raise ValueError(f"{field_name} must be YYYY-MM-DD, got {value!r}") from None


def _where_text(value: str, field_name: str) -> str:
    text = str(value).strip()
    if not text or '"' in text or "\\" in text or any(ord(ch) < 32 for ch in text):
        raise ValueError(f"{field_name} contains characters Xero filters cannot take")
    return text


def _where_date(value: date) -> str:
    return f"DateTime({value.year},{value.month},{value.day})"


def _page_number(page: int) -> int:
    return max(1, int(page))


def _page_size(page_size: int) -> int:
    return clamp_limit(page_size, max_limit=XERO_MAX_PAGE_SIZE)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def _access_token() -> str:
    token = os.environ.get("XERO_CLIENT_BEARER_TOKEN")
    if not token:
        raise ValueError("XERO_CLIENT_BEARER_TOKEN environment variable is missing")
    return token


def _tenant_id() -> str:
    cached = _connection_cache.get("tenant_id")
    if cached:
        return cached
    response = requests.get(
        XERO_CONNECTIONS_URL,
        headers={
            "Authorization": f"Bearer {_access_token()}",
            "Content-Type": "application/json",
        },
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    _raise_for_status(response, scope=None)
    connections = response.json()
    if not isinstance(connections, list) or not connections:
        raise RuntimeError("No Xero organisation is connected to this account")
    tenant_id = connections[0].get("tenantId")
    if not tenant_id:
        raise RuntimeError("Xero returned a connection without a tenant id")
    # The first connection is the organisation every tool works against, the
    # same choice the upstream Xero MCP server made.
    _connection_cache["tenant_id"] = str(tenant_id)
    return str(tenant_id)


def _headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {_access_token()}",
        "xero-tenant-id": _tenant_id(),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if extra:
        headers.update(extra)
    return headers


def _error_detail(response: requests.Response) -> str | None:
    """Pull the human-readable message out of a Xero error body.

    Validation failures are ``{"Message", "Elements": [{"ValidationErrors":
    [{"Message"}]}]}``; auth and rate-limit failures carry ``Title``/``Detail``.
    """
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    messages: list[str] = []
    for element in payload.get("Elements") or []:
        if isinstance(element, dict):
            for problem in element.get("ValidationErrors") or []:
                if isinstance(problem, dict) and problem.get("Message"):
                    messages.append(str(problem["Message"]))
    if messages:
        return "; ".join(dict.fromkeys(messages))
    for key in ("Message", "Detail", "Title"):
        if payload.get(key):
            return str(payload[key])
    return None


def _raise_for_status(response: requests.Response, *, scope: str | None) -> None:
    if response.status_code < 400:
        return
    status = response.status_code
    detail = _error_detail(response)
    if detail is None:
        detail = response.text.strip()
    if detail and len(detail) > MAX_ERROR_RESPONSE_TEXT_CHARS:
        detail = detail[:MAX_ERROR_RESPONSE_TEXT_CHARS] + "... [truncated]"
    detail_text = f" Details: {detail}" if detail else ""
    if status == 401:
        raise XeroAPIError(
            "Xero rejected the access token (status 401). Reconnect Xero if "
            f"the connection has expired.{detail_text}",
            status_code=status,
        )
    if status == 403:
        hint = (
            f" The tool needs the OAuth scope {SCOPE_HINTS[scope]}; if it was "
            "added after Xero was connected, reconnect Xero to grant it."
            if scope in SCOPE_HINTS
            else " Check that this Xero user can access the connected organisation."
        )
        raise XeroAPIError(
            f"Xero denied the request (status 403).{hint}{detail_text}",
            status_code=status,
        )
    if status == 429:
        retry = response.headers.get("Retry-After")
        wait = f" Retry after {retry} seconds." if retry else ""
        raise XeroAPIError(
            f"Xero rate limit reached (status 429).{wait}", status_code=status
        )
    message = f"Xero API error (status {status})"
    raise XeroAPIError(
        f"{message}: {detail}" if detail else message,
        status_code=status,
    )


def _request(
    method: str,
    path: str,
    *,
    scope: str | None,
    params: dict[str, Any] | None = None,
    json_data: Any = None,
    headers: dict[str, str] | None = None,
) -> Any:
    response = requests.request(
        method=method,
        url=f"{XERO_API_BASE}{path}",
        headers=_headers(headers),
        params=params,
        json=json_data,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    _raise_for_status(response, scope=scope)
    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


# --------------------------------------------------------------------------- #
# Response shaping
# --------------------------------------------------------------------------- #


def _xero_date(value: Any) -> str | None:
    """Render a Xero date as ISO text.

    Xero's JSON carries dates as ``/Date(1700000000000+0000)/`` and, for the
    main document dates, an ISO ``...String`` twin.
    """
    if not value:
        return None
    text = str(value)
    match = _MS_DATE_PATTERN.fullmatch(text)
    if match:
        moment = datetime.fromtimestamp(int(match.group(1)) / 1000, tz=timezone.utc)
        return moment.isoformat().replace("+00:00", "Z")
    return text


def _doc_date(record: dict[str, Any], field_name: str) -> str | None:
    return _xero_date(record.get(f"{field_name}String") or record.get(field_name))


def _compact(mapping: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if value not in (None, "")}


def _contact_ref(record: dict[str, Any]) -> dict[str, Any] | None:
    contact = record.get("Contact")
    if not isinstance(contact, dict):
        return None
    return _compact({"id": contact.get("ContactID"), "name": contact.get("Name")})


def _line_items(record: dict[str, Any]) -> list[dict[str, Any]]:
    lines = []
    for line in record.get("LineItems") or []:
        lines.append(
            _compact(
                {
                    "description": line.get("Description"),
                    "quantity": line.get("Quantity"),
                    "unit_amount": line.get("UnitAmount"),
                    "line_amount": line.get("LineAmount"),
                    "account_code": line.get("AccountCode"),
                    "tax_type": line.get("TaxType"),
                    "tax_amount": line.get("TaxAmount"),
                    "item_code": line.get("ItemCode"),
                }
            )
        )
    return lines


def _invoice_view(
    record: dict[str, Any], *, with_lines: bool = False
) -> dict[str, Any]:
    view = _compact(
        {
            "invoice_id": record.get("InvoiceID"),
            "invoice_number": record.get("InvoiceNumber"),
            "reference": record.get("Reference"),
            "type": record.get("Type"),
            "status": record.get("Status"),
            "contact": _contact_ref(record),
            "date": _doc_date(record, "Date"),
            "due_date": _doc_date(record, "DueDate"),
            "expected_payment_date": _xero_date(record.get("ExpectedPaymentDate")),
            "planned_payment_date": _xero_date(record.get("PlannedPaymentDate")),
            "currency": record.get("CurrencyCode"),
            "sub_total": record.get("SubTotal"),
            "total_tax": record.get("TotalTax"),
            "total": record.get("Total"),
            "amount_due": record.get("AmountDue"),
            "amount_paid": record.get("AmountPaid"),
            "amount_credited": record.get("AmountCredited"),
            "fully_paid_on": _xero_date(record.get("FullyPaidOnDate")),
            "updated": _xero_date(record.get("UpdatedDateUTC")),
        }
    )
    if with_lines:
        view["line_items"] = _line_items(record)
    return view


def _history_record_view(record: dict[str, Any]) -> dict[str, Any]:
    return _compact(
        {
            "details": record.get("Details"),
            "changes": record.get("Changes"),
            "user": record.get("User"),
            "date_utc": _xero_date(
                record.get("DateUTCString") or record.get("DateUTC")
            ),
        }
    )


def _contact_view(record: dict[str, Any]) -> dict[str, Any]:
    phones = [
        phone.get("PhoneNumber")
        for phone in record.get("Phones") or []
        if isinstance(phone, dict) and phone.get("PhoneNumber")
    ]
    return _compact(
        {
            "contact_id": record.get("ContactID"),
            "name": record.get("Name"),
            "first_name": record.get("FirstName"),
            "last_name": record.get("LastName"),
            "email": record.get("EmailAddress"),
            "phones": phones,
            "status": record.get("ContactStatus"),
            "is_customer": record.get("IsCustomer"),
            "is_supplier": record.get("IsSupplier"),
            "updated": _xero_date(record.get("UpdatedDateUTC")),
        }
    )


def _credit_note_view(
    record: dict[str, Any], *, with_lines: bool = False
) -> dict[str, Any]:
    view = _compact(
        {
            "credit_note_id": record.get("CreditNoteID"),
            "credit_note_number": record.get("CreditNoteNumber"),
            "reference": record.get("Reference"),
            "type": record.get("Type"),
            "status": record.get("Status"),
            "contact": _contact_ref(record),
            "date": _doc_date(record, "Date"),
            "currency": record.get("CurrencyCode"),
            "total": record.get("Total"),
            "remaining_credit": record.get("RemainingCredit"),
            "updated": _xero_date(record.get("UpdatedDateUTC")),
        }
    )
    if with_lines:
        view["line_items"] = _line_items(record)
    return view


def _quote_view(record: dict[str, Any], *, with_lines: bool = False) -> dict[str, Any]:
    view = _compact(
        {
            "quote_id": record.get("QuoteID"),
            "quote_number": record.get("QuoteNumber"),
            "reference": record.get("Reference"),
            "title": record.get("Title"),
            "summary": record.get("Summary"),
            "terms": record.get("Terms"),
            "status": record.get("Status"),
            "contact": _contact_ref(record),
            "date": _doc_date(record, "Date"),
            "expiry_date": _doc_date(record, "ExpiryDate"),
            "currency": record.get("CurrencyCode"),
            "total": record.get("Total"),
            "updated": _xero_date(record.get("UpdatedDateUTC")),
        }
    )
    if with_lines:
        view["line_items"] = _line_items(record)
    return view


def _bank_transaction_view(
    record: dict[str, Any], *, with_lines: bool = False
) -> dict[str, Any]:
    account = record.get("BankAccount") or {}
    view = _compact(
        {
            "bank_transaction_id": record.get("BankTransactionID"),
            "type": record.get("Type"),
            "status": record.get("Status"),
            "reference": record.get("Reference"),
            "contact": _contact_ref(record),
            "bank_account": _compact(
                {"id": account.get("AccountID"), "name": account.get("Name")}
            ),
            "date": _doc_date(record, "Date"),
            "currency": record.get("CurrencyCode"),
            "total": record.get("Total"),
            "is_reconciled": record.get("IsReconciled"),
            "updated": _xero_date(record.get("UpdatedDateUTC")),
        }
    )
    if with_lines:
        view["line_items"] = _line_items(record)
    return view


def _payment_view(record: dict[str, Any]) -> dict[str, Any]:
    invoice = record.get("Invoice") or {}
    account = record.get("Account") or {}
    return _compact(
        {
            "payment_id": record.get("PaymentID"),
            "payment_type": record.get("PaymentType"),
            "status": record.get("Status"),
            "date": _doc_date(record, "Date"),
            "amount": record.get("Amount"),
            "reference": record.get("Reference"),
            "invoice_id": invoice.get("InvoiceID"),
            "invoice_number": invoice.get("InvoiceNumber"),
            "account": _compact(
                {"id": account.get("AccountID"), "code": account.get("Code")}
            ),
            "updated": _xero_date(record.get("UpdatedDateUTC")),
        }
    )


def _manual_journal_view(record: dict[str, Any]) -> dict[str, Any]:
    lines = [
        _compact(
            {
                "line_amount": line.get("LineAmount"),
                "account_code": line.get("AccountCode"),
                "description": line.get("Description"),
                "tax_type": line.get("TaxType"),
            }
        )
        for line in record.get("JournalLines") or []
    ]
    return _compact(
        {
            "manual_journal_id": record.get("ManualJournalID"),
            "narration": record.get("Narration"),
            "status": record.get("Status"),
            "date": _doc_date(record, "Date"),
            "line_amount_types": record.get("LineAmountTypes"),
            "url": record.get("Url"),
            "show_on_cash_basis_reports": record.get("ShowOnCashBasisReports"),
            "journal_lines": lines,
            "updated": _xero_date(record.get("UpdatedDateUTC")),
        }
    )


def _item_view(record: dict[str, Any]) -> dict[str, Any]:
    def details(block: Any) -> dict[str, Any] | None:
        if not isinstance(block, dict):
            return None
        return _compact(
            {
                "unit_price": block.get("UnitPrice"),
                "account_code": block.get("AccountCode"),
                "tax_type": block.get("TaxType"),
            }
        )

    return _compact(
        {
            "item_id": record.get("ItemID"),
            "code": record.get("Code"),
            "name": record.get("Name"),
            "description": record.get("Description"),
            "purchase_description": record.get("PurchaseDescription"),
            "is_sold": record.get("IsSold"),
            "is_purchased": record.get("IsPurchased"),
            "is_tracked_as_inventory": record.get("IsTrackedAsInventory"),
            "sales_details": details(record.get("SalesDetails")),
            "purchase_details": details(record.get("PurchaseDetails")),
        }
    )


def _tracking_category_view(record: dict[str, Any]) -> dict[str, Any]:
    options = [
        _compact(
            {
                "option_id": option.get("TrackingOptionID"),
                "name": option.get("Name"),
                "status": option.get("Status"),
            }
        )
        for option in record.get("Options") or []
    ]
    return _compact(
        {
            "tracking_category_id": record.get("TrackingCategoryID"),
            "name": record.get("Name"),
            "status": record.get("Status"),
            "options": options,
        }
    )


def _first(data: Any, key: str) -> dict[str, Any]:
    records = data.get(key) if isinstance(data, dict) else None
    if not records:
        raise RuntimeError(f"Xero returned no {key} record")
    record = records[0]
    if not isinstance(record, dict):
        raise RuntimeError(f"Xero returned an unexpected {key} record")
    return record


def _fit_list(
    build: Callable[[list[Any], bool], dict[str, Any]], items: list[Any]
) -> str:
    """Render ``build(items, truncated)``, halving ``items`` until it fits the
    platform's output limit. ``build`` must report how many items it holds."""
    max_output_length = get_tool_max_output_length()
    response = _success(**build(items, False))
    while len(response) > max_output_length and items:
        items = items[: len(items) // 2]
        response = _success(**build(items, True))
    return response


def _paged_response(
    list_field: str,
    items: list[Any],
    *,
    data: dict[str, Any],
    page: int,
    page_size: int,
    raw_count: int,
) -> str:
    """Render one page of a Xero list with exact paging facts.

    Xero's ``pagination`` block carries ``itemCount`` (the true total) and
    ``pageCount``. Endpoints without it fall back to "a full page means there
    may be another".
    """
    pagination = data.get("pagination") if isinstance(data, dict) else None
    total_count = page_count = None
    if isinstance(pagination, dict):
        total_count = pagination.get("itemCount")
        page_count = pagination.get("pageCount")
    if page_count is not None:
        has_more = page < int(page_count)
    else:
        has_more = raw_count >= page_size

    def build(shown: list[Any], truncated: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            list_field: shown,
            "page": page,
            "page_size": page_size,
            "returned": len(shown),
            "has_more": has_more,
        }
        if total_count is not None:
            payload["total_count"] = total_count
        if page_count is not None:
            payload["page_count"] = page_count
        if has_more and not truncated:
            payload["next_page"] = page + 1
        if truncated:
            payload["truncated"] = True
            payload["note"] = (
                "The page did not fit the output limit, so items were dropped. "
                "Call again with a smaller page_size to read them all."
            )
        return payload

    return _fit_list(build, items)


def _unpaged_response(list_field: str, items: list[Any]) -> str:
    def build(shown: list[Any], truncated: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            list_field: shown,
            "total_count": len(items),
            "returned": len(shown),
        }
        if truncated:
            payload["truncated"] = True
        return payload

    return _fit_list(build, items)


def _flatten_report(report: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    def walk(row_list: list[Any], section: str | None) -> None:
        for row in row_list or []:
            if not isinstance(row, dict):
                continue
            row_type = row.get("RowType")
            title = row.get("Title")
            if row_type == "Section":
                if title:
                    rows.append({"row_type": "Section", "title": title})
                walk(row.get("Rows") or [], title or section)
                continue
            cells = [cell.get("Value", "") for cell in row.get("Cells") or []]
            if any(cells):
                entry: dict[str, Any] = {"row_type": row_type, "cells": cells}
                if section:
                    entry["section"] = section
                rows.append(entry)

    walk(report.get("Rows") or [], None)
    return {
        "report_name": report.get("ReportName"),
        "report_titles": report.get("ReportTitles"),
        "report_date": report.get("ReportDate"),
        "updated": _xero_date(report.get("UpdatedDateUTC")),
        "rows": rows,
    }


def _report_response(data: dict[str, Any]) -> str:
    reports = data.get("Reports") if isinstance(data, dict) else None
    if not reports:
        raise RuntimeError("Xero returned no report")
    report = _flatten_report(reports[0])
    rows = report.pop("rows")

    def build(shown: list[Any], truncated: bool) -> dict[str, Any]:
        payload = {**report, "rows": shown, "total_rows": len(rows)}
        if truncated:
            payload["truncated"] = True
        return payload

    return _fit_list(build, rows)


# --------------------------------------------------------------------------- #
# Deep links
# --------------------------------------------------------------------------- #


def _org_short_code() -> str | None:
    cached = _connection_cache.get("short_code")
    if cached:
        return cached
    data = _request("GET", "Organisation", scope="settings")
    organisations = data.get("Organisations") if isinstance(data, dict) else None
    short_code = organisations[0].get("ShortCode") if organisations else None
    if short_code:
        _connection_cache["short_code"] = str(short_code)
    return str(short_code) if short_code else None


def _deep_link(kind: str, record_id: str | None) -> str | None:
    """Link to a record in Xero, or None.

    Resolved only after a write has already succeeded, so a failure here must
    never turn that success into an error.
    """
    if not record_id:
        return None
    try:
        short_code = _org_short_code()
    except Exception:
        logger.warning("Could not resolve the Xero organisation short code")
        return None
    if not short_code:
        return None
    login = f"https://go.xero.com/organisationlogin/default.aspx?shortcode={short_code}"
    links = {
        "invoice": f"https://go.xero.com/app/{short_code}/invoicing/view/{record_id}",
        "contact": f"https://go.xero.com/app/{short_code}/contacts/contact/{record_id}",
        "quote": f"https://go.xero.com/app/{short_code}/quotes/view/{record_id}",
        "bill": f"{login}&redirecturl=/AccountsPayable/Edit.aspx?InvoiceID={record_id}",
        "credit_note": (
            f"{login}&redirecturl=/AccountsPayable/ViewCreditNote.aspx"
            f"?creditNoteID={record_id}"
        ),
        "payment": (
            f"{login}&redirecturl=/Bank/ViewTransaction.aspx"
            f"?bankTransactionID={record_id}"
        ),
        "manual_journal": f"https://go.xero.com/Journal/View.aspx?invoiceID={record_id}",
    }
    return links.get(kind)


# --------------------------------------------------------------------------- #
# Request body builders
# --------------------------------------------------------------------------- #


def _line_item_body(line: LineItem) -> dict[str, Any]:
    body: dict[str, Any] = {
        "Description": line.description,
        "Quantity": line.quantity,
        "UnitAmount": line.unitAmount,
        "AccountCode": line.accountCode,
        "TaxType": line.taxType,
    }
    if line.itemCode:
        body["ItemCode"] = line.itemCode
    if line.tracking:
        body["Tracking"] = [
            {
                "Name": tracking.name,
                "Option": tracking.option,
                "TrackingCategoryID": tracking.trackingCategoryID,
            }
            for tracking in line.tracking
        ]
    return body


def _line_item_bodies(lines: list[LineItem]) -> list[dict[str, Any]]:
    return [_line_item_body(line) for line in lines]


def _journal_line_bodies(lines: list[ManualJournalLine]) -> list[dict[str, Any]]:
    if len(lines) < 2:
        raise ValueError("A manual journal needs at least two journal lines")
    total = sum((Decimal(str(line.lineAmount)) for line in lines), Decimal(0))
    if total != 0:
        raise ValueError(
            f"Manual journal lines must balance to zero (debits positive, credits "
            f"negative); they sum to {total}"
        )
    bodies = []
    for line in lines:
        body: dict[str, Any] = {
            "LineAmount": line.lineAmount,
            "AccountCode": line.accountCode,
        }
        if line.description:
            body["Description"] = line.description
        if line.taxType:
            body["TaxType"] = line.taxType
        bodies.append(body)
    return bodies


def _item_details_body(details: ItemDetails) -> dict[str, Any]:
    body: dict[str, Any] = {"UnitPrice": details.unitPrice}
    if details.taxType:
        body["TaxType"] = details.taxType
    if details.accountCode:
        body["AccountCode"] = details.accountCode
    return body


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _organisation_today() -> date:
    """Return today in the connected organisation's accounting time zone.

    Xero exposes Windows-style zone identifiers for AU/NZ organisations. If
    the organisation lookup or zone mapping fails, retain the previous UTC
    behaviour rather than preventing an otherwise valid invoice read.
    """

    zone_name = _connection_cache.get("time_zone")
    if not zone_name:
        try:
            data = _request("GET", "Organisation", scope="settings")
            record = _first(data, "Organisations")
            raw_zone = str(record.get("Timezone") or "").strip()
            zone_name = _XERO_TIME_ZONES.get(raw_zone.upper(), raw_zone)
            if zone_name:
                _connection_cache["time_zone"] = zone_name
        except Exception:
            logger.warning("Could not resolve the Xero organisation time zone")
            return _today()
    try:
        return datetime.now(ZoneInfo(zone_name)).date()
    except (TypeError, ValueError, ZoneInfoNotFoundError):
        logger.warning("Unknown Xero organisation time zone %r", zone_name)
        return _today()


def _get_one(path: str, key: str, *, scope: str) -> dict[str, Any]:
    return _first(_request("GET", path, scope=scope), key)


def _require_draft(record: dict[str, Any], label: str, allowed: set[str]) -> None:
    status = record.get("Status")
    if status not in allowed:
        raise ValueError(
            f"Only a {' or '.join(sorted(allowed)).lower()} {label} can be updated; "
            f"this one is {status}"
        )


# --------------------------------------------------------------------------- #
# Lists
# --------------------------------------------------------------------------- #


def _list_paged(
    path: str,
    key: str,
    list_field: str,
    view: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    scope: str,
    page: int,
    page_size: int,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> str:
    page = _page_number(page)
    page_size = _page_size(page_size)
    query = {**(params or {}), "page": page, "pageSize": page_size}
    data = _request("GET", path, scope=scope, params=query, headers=headers)
    records = data.get(key) or []
    return _paged_response(
        list_field,
        [view(record) for record in records],
        data=data,
        page=page,
        page_size=page_size,
        raw_count=len(records),
    )


@mcp.tool(annotations=READ_ONLY)
def list_invoices(
    page: int = 1,
    contactIds: list[str] | None = None,
    invoiceNumbers: list[str] | None = None,
    statuses: list[str] | None = None,
    type: Literal["ACCREC", "ACCPAY"] | None = None,
    dueDateBefore: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> str:
    """
    List invoices in Xero, newest-updated first. Includes Draft, Submitted,
    Authorised, Paid, Voided and Deleted invoices unless narrowed.

    The result reports total_count, page_count, has_more and next_page. If
    has_more is true the list is NOT finished: call again with next_page. To
    find overdue invoices do not page through this list; use
    list_overdue_invoices, which reads all of them. For invoices settled on a
    specific date use list_paid_invoices; UpdatedDateUTC ordering cannot prove
    a FullyPaidOnDate list is complete.

    contactIds: only invoices for these contacts (from list_contacts).
    invoiceNumbers: only these invoice numbers; line items are then included.
    statuses: any of DRAFT, SUBMITTED, AUTHORISED, PAID, VOIDED, DELETED.
    type: ACCREC for sales invoices, ACCPAY for bills.
    dueDateBefore: YYYY-MM-DD; only invoices due before that date.
    page_size: invoices per page, 1-100 (default 20).
    """
    params: dict[str, Any] = {"order": "UpdatedDateUTC DESC"}
    where: list[str] = []
    if type:
        where.append(f'Type=="{type}"')
    if dueDateBefore:
        where.append(
            f"DueDate<{_where_date(_iso_date(dueDateBefore, 'dueDateBefore'))}"
        )
    if where:
        params["where"] = " AND ".join(where)
    if contactIds:
        params["ContactIDs"] = ",".join(_guid_list(contactIds, "contactIds"))
    if invoiceNumbers:
        params["InvoiceNumbers"] = ",".join(
            _where_text(number, "invoiceNumbers") for number in invoiceNumbers
        )
    if statuses:
        cleaned = [str(status).strip().upper() for status in statuses]
        unknown = sorted(set(cleaned) - INVOICE_STATUSES)
        if unknown:
            raise ValueError(f"Unknown invoice status: {', '.join(unknown)}")
        params["Statuses"] = ",".join(cleaned)
    with_lines = bool(invoiceNumbers)
    return _list_paged(
        "Invoices",
        "Invoices",
        "invoices",
        lambda record: _invoice_view(record, with_lines=with_lines),
        scope="invoices",
        page=page,
        page_size=page_size,
        params=params,
    )


def _iter_complete_invoice_pages(
    params: dict[str, Any],
    *,
    max_pages: int,
    label: str,
    partial_result: str,
    narrow_hint: str,
) -> Iterator[list[dict[str, Any]]]:
    """Yield invoice pages until Xero confirms the filtered list is complete."""
    page = 1
    while True:
        data = _request(
            "GET", "Invoices", scope="invoices", params={**params, "page": page}
        )
        yield data.get("Invoices") or []

        pagination = data.get("pagination") or {}
        page_count = pagination.get("pageCount")
        if page_count is None:
            raise RuntimeError(
                f"Xero {label} scan returned no pagination metadata; refusing "
                f"to report {partial_result} as complete"
            )
        if page >= int(page_count):
            break
        if page >= max_pages:
            raise RuntimeError(
                f"Xero {label} scan reached the safety limit of {max_pages} pages "
                f"before the end of the list; narrow the request with {narrow_hint}"
            )
        page += 1


def _collect_paid_invoices(
    paid_on: date,
    invoice_type: str,
    contact_ids: list[str] | None,
) -> list[dict[str, Any]]:
    """Read the complete paid-invoice ledger and select one settlement date.

    Xero orders the generic invoice list by update time, not by
    ``FullyPaidOnDate``.  A page containing another payment date therefore
    cannot be used as an end marker.  Keep paging until Xero's own pagination
    metadata says the PAID list is complete, then filter locally.
    """
    params: dict[str, Any] = {
        "where": f'Type=="{invoice_type}"',
        "Statuses": "PAID",
        "pageSize": XERO_BULK_PAGE_SIZE,
        "summaryOnly": "true",
    }
    if contact_ids:
        params["ContactIDs"] = ",".join(_guid_list(contact_ids, "contactIds"))

    found: list[dict[str, Any]] = []
    seen_invoice_ids: set[str] = set()
    for records in _iter_complete_invoice_pages(
        params,
        max_pages=PAID_INVOICE_MAX_PAGES,
        label="paid-invoice",
        partial_result="a partial settlement-date list",
        narrow_hint="contactIds",
    ):
        for record in records:
            if record.get("Type") != invoice_type or record.get("Status") != "PAID":
                continue
            fully_paid_on = _xero_date(record.get("FullyPaidOnDate"))
            if not fully_paid_on:
                identity = record.get("InvoiceID") or record.get("InvoiceNumber")
                raise RuntimeError(
                    "Xero returned PAID invoice "
                    f"{identity or '<unknown>'} without FullyPaidOnDate; refusing "
                    "to report a settlement-date list as complete"
                )
            if date.fromisoformat(fully_paid_on[:10]) != paid_on:
                continue
            identity = str(
                record.get("InvoiceID")
                or record.get("InvoiceNumber")
                or f"{record.get('Contact')}:{fully_paid_on}:{record.get('Total')}"
            )
            if identity in seen_invoice_ids:
                continue
            seen_invoice_ids.add(identity)
            found.append(_invoice_view(record))
    return found


@mcp.tool(annotations=READ_ONLY)
def list_paid_invoices(
    paidOnDate: str,
    contactIds: list[str] | None = None,
    type: Literal["ACCREC", "ACCPAY"] = "ACCREC",
    skip: int = 0,
    limit: int = PAID_INVOICE_DEFAULT_LIMIT,
) -> str:
    """
    List ALL invoices fully paid on one date, regardless of update ordering.

    Use this tool for requests such as "customer invoices settled on 13 Oct".
    It scans every PAID invoice page inside the connector and filters by
    FullyPaidOnDate. Do not infer settlement-date completeness from
    list_invoices: that list is ordered by UpdatedDateUTC, so matching dates can
    occur on later pages after other dates.

    This reports invoices that became fully settled on the requested date. It
    is not a list of every payment received that day; partial payments belong
    in list_payments.

    Each skip/limit window re-scans the complete PAID ledger, up to 50 pages of
    1000 invoices. Prefer the largest practical limit (up to 200) to conserve
    Xero's tenant request quota. Large tenants can narrow the scan with
    contactIds; the tool fails rather than returning partial results if the
    safety limit is reached.

    paidOnDate: settlement date in YYYY-MM-DD format.
    contactIds: optionally restrict to these contacts (from list_contacts).
    type: ACCREC for customer invoices (default), ACCPAY for bills you paid.
    skip, limit: window into the complete result (limit 1-200, default 100).
    """
    paid_on = _iso_date(paidOnDate, "paidOnDate")
    skip = max(0, int(skip))
    limit = max(1, min(int(limit), PAID_INVOICE_MAX_LIMIT))
    paid = _collect_paid_invoices(paid_on, type, contactIds)
    paid.sort(
        key=lambda view: (view.get("invoice_number", ""), view.get("invoice_id", ""))
    )

    totals: dict[str, Decimal] = {}
    for view in paid:
        currency = view.get("currency") or "UNKNOWN"
        amount = Decimal(str(view.get("total") or 0))
        totals[currency] = totals.get(currency, Decimal(0)) + amount
    window = paid[skip : skip + limit]

    def build(shown: list[Any], truncated: bool) -> dict[str, Any]:
        next_skip = skip + len(shown)
        has_more = next_skip < len(paid)
        payload: dict[str, Any] = {
            "paid_on_date": paid_on.isoformat(),
            "invoice_type": type,
            "date_basis": "FullyPaidOnDate",
            "total_count": len(paid),
            "total_invoice_amount_by_currency": _money_totals(totals),
            "complete": True,
            "skip": skip,
            "returned": len(shown),
            "has_more": has_more,
            "invoices": shown,
        }
        if has_more:
            payload["next_skip"] = next_skip
        if truncated:
            payload["truncated"] = True
        return payload

    return _fit_list(build, window)


@mcp.tool(annotations=READ_ONLY)
def list_invoice_history(
    invoiceId: str,
    skip: int = 0,
    limit: int = 100,
) -> str:
    """
    List Xero History & Notes records for one invoice, oldest to newest.

    Use this when a user asks for notes, comments or change history they can
    see in an invoice's History & Notes panel. These records are separate from
    the invoice Reference and are not returned by list_invoices. When has_more
    is true, call again with next_skip to read the remaining history.

    invoiceId: Xero invoice GUID from list_invoices.
    skip, limit: window into the complete history (limit 1-200, default 100).
    """
    safe_id = _guid(invoiceId, "invoiceId")
    data = _request(
        "GET",
        f"Invoices/{safe_id}/History",
        scope="invoices",
    )
    records = data.get("HistoryRecords") or []
    history = [_history_record_view(record) for record in records]
    history.sort(
        key=lambda record: (
            "date_utc" not in record,
            record.get("date_utc", ""),
        )
    )
    skip = clamp_offset(skip)
    limit = clamp_limit(limit, max_limit=INVOICE_HISTORY_MAX_LIMIT)
    window = history[skip : skip + limit]

    def build(shown: list[Any], truncated: bool) -> dict[str, Any]:
        next_skip = skip + len(shown)
        has_more = next_skip < len(history)
        payload: dict[str, Any] = {
            "total_count": len(history),
            "complete": True,
            "skip": skip,
            "returned": len(shown),
            "has_more": has_more,
            "history": shown,
        }
        if has_more:
            payload["next_skip"] = next_skip
        if truncated:
            payload["truncated"] = True
        return payload

    return _fit_list(build, window)


def _money_totals(totals: dict[str, Decimal]) -> dict[str, str]:
    return {currency: f"{amount:.2f}" for currency, amount in sorted(totals.items())}


def _ageing_bucket(days_overdue: int) -> str:
    if days_overdue <= 30:
        return "1-30"
    if days_overdue <= 60:
        return "31-60"
    if days_overdue <= 90:
        return "61-90"
    return "90+"


def _collect_open_invoices(
    as_of: date,
    invoice_type: str,
    contact_ids: list[str] | None,
    *,
    min_days_overdue: int | None,
) -> list[dict[str, Any]]:
    """Read every Authorised invoice of ``invoice_type`` that still has an
    amount due, across all pages.

    With ``min_days_overdue`` only invoices at least that many days past due
    are kept; with None every open invoice is kept (including not-yet-due ones,
    whose ``days_overdue`` is zero or negative). A successful return is always
    complete; reaching the safety bound raises rather than exposing partial
    values as totals.
    """
    params: dict[str, Any] = {
        "where": (f'Type=="{invoice_type}" AND Date<={_where_date(as_of)}'),
        "Statuses": "AUTHORISED",
        "pageSize": XERO_BULK_PAGE_SIZE,
        "summaryOnly": "true",
    }
    if min_days_overdue is not None:
        # DueDate <= asOf - n  <=>  DueDate < asOf - n + 1 day
        cutoff = as_of - timedelta(days=min_days_overdue - 1)
        params["where"] += f" AND DueDate<{_where_date(cutoff)}"
    if contact_ids:
        params["ContactIDs"] = ",".join(_guid_list(contact_ids, "contactIds"))

    found: list[dict[str, Any]] = []
    seen_invoice_ids: set[str] = set()
    for records in _iter_complete_invoice_pages(
        params,
        max_pages=OPEN_INVOICE_MAX_PAGES,
        label="invoice",
        partial_result="partial ledger totals",
        narrow_hint="contactIds or a stricter date range",
    ):
        for record in records:
            amount_due = Decimal(str(record.get("AmountDue") or 0))
            due = _doc_date(record, "DueDate")
            issued = _doc_date(record, "Date")
            # The request asks Xero for Authorised invoices, but enforce the
            # same contract locally as well.  A stale or non-conforming API
            # response must never let a Paid invoice into an overdue/follow-up
            # list merely because it still carries a non-zero AmountDue.
            if (
                record.get("Type") != invoice_type
                or record.get("Status") != "AUTHORISED"
                or amount_due <= 0
                or not due
                or not issued
                or date.fromisoformat(issued[:10]) > as_of
            ):
                continue
            identity = str(
                record.get("InvoiceID")
                or record.get("InvoiceNumber")
                or f"{record.get('Contact')}:{issued}:{due}:{amount_due}"
            )
            if identity in seen_invoice_ids:
                continue
            seen_invoice_ids.add(identity)
            days = (as_of - date.fromisoformat(due[:10])).days
            if min_days_overdue is not None and days < min_days_overdue:
                continue
            view = _invoice_view(record)
            view["days_overdue"] = days
            found.append(view)
    return found


@mcp.tool(annotations=READ_ONLY)
def list_overdue_invoices(
    asOfDate: str | None = None,
    contactIds: list[str] | None = None,
    type: Literal["ACCREC", "ACCPAY"] = "ACCREC",
    minDaysOverdue: int = 1,
    skip: int = 0,
    limit: int = OVERDUE_DEFAULT_LIMIT,
) -> str:
    """
    List ALL currently open overdue invoices in Xero, oldest due date first.

    Overdue means an Authorised invoice with an amount still due whose due date
    is before asOfDate. Paid, draft, voided and not-yet-due invoices are never
    returned, and this tool reads every page from Xero itself, so there is no
    need to call list_invoices page by page.

    The result always carries the complete open-invoice picture in total_count,
    total_amount_due_by_currency, ageing_buckets (1-30, 31-60, 61-90, 90+ days)
    and by_contact (largest balances first), even when only part of the invoice
    list fits in one response. When has_more is true, call again with next_skip
    to read the remaining invoices. Each window re-scans the open ledger, so
    prefer the largest practical limit (up to 200) to conserve Xero's tenant
    request quota.

    Historical limitation: a past asOfDate excludes invoices issued later, but
    payments, credits and allocations made after that date are not rolled back.
    Credit notes, overpayments and prepayments are not netted into these gross
    open-invoice totals; the response declares these exclusions explicitly.

    asOfDate: YYYY-MM-DD (default: today in the Xero organisation's time zone).
    contactIds: only these contacts (from list_contacts).
    type: ACCREC for customer invoices (default), ACCPAY for bills you owe.
    minDaysOverdue: only invoices at least this many days past due (default 1).
    skip, limit: window into the invoice list (limit 1-200, default 50).
    """
    as_of = _iso_date(asOfDate, "asOfDate") if asOfDate else _organisation_today()
    min_days = max(1, int(minDaysOverdue))
    skip = max(0, int(skip))
    limit = max(1, min(int(limit), OVERDUE_MAX_LIMIT))
    overdue = _collect_open_invoices(as_of, type, contactIds, min_days_overdue=min_days)

    overdue.sort(key=lambda view: (view["due_date"], view.get("invoice_number", "")))
    totals: dict[str, Decimal] = {}
    buckets: dict[str, dict[str, Any]] = {
        name: {"count": 0, "totals": {}} for name in ("1-30", "31-60", "61-90", "90+")
    }
    contacts: dict[tuple[str, str], dict[str, Any]] = {}
    for view in overdue:
        currency = view.get("currency") or "UNKNOWN"
        amount = Decimal(str(view.get("amount_due") or 0))
        totals[currency] = totals.get(currency, Decimal(0)) + amount
        bucket = buckets[_ageing_bucket(view["days_overdue"])]
        bucket["count"] += 1
        bucket["totals"][currency] = bucket["totals"].get(currency, Decimal(0)) + amount
        contact = view.get("contact") or {}
        key = (contact.get("id") or contact.get("name") or "unknown", currency)
        entry = contacts.setdefault(
            key,
            {
                "contact_id": contact.get("id"),
                "contact_name": contact.get("name"),
                "currency": currency,
                "invoice_count": 0,
                "amount_due": Decimal(0),
                "oldest_due_date": view["due_date"],
            },
        )
        entry["invoice_count"] += 1
        entry["amount_due"] += amount

    ranked: list[dict[str, Any]] = []
    for currency in sorted({entry["currency"] for entry in contacts.values()}):
        ranked.extend(
            sorted(
                (entry for entry in contacts.values() if entry["currency"] == currency),
                key=lambda entry: (-entry["amount_due"], entry["contact_name"] or ""),
            )[:OVERDUE_TOP_CONTACTS]
        )
    by_contact = [
        _compact(
            {
                "contact_id": entry["contact_id"],
                "contact_name": entry["contact_name"],
                "currency": entry["currency"],
                "invoice_count": entry["invoice_count"],
                "amount_due": f"{entry['amount_due']:.2f}",
                "oldest_due_date": entry["oldest_due_date"],
            }
        )
        for entry in ranked
    ]
    ageing = {
        name: {
            "count": bucket["count"],
            "amount_due_by_currency": _money_totals(bucket["totals"]),
        }
        for name, bucket in buckets.items()
    }
    window = overdue[skip : skip + limit]

    def build(shown: list[Any], truncated: bool) -> dict[str, Any]:
        next_skip = skip + len(shown)
        has_more = next_skip < len(overdue)
        payload: dict[str, Any] = {
            "as_of_date": as_of.isoformat(),
            "invoice_type": type,
            "min_days_overdue": min_days,
            "balance_basis": "current_open_invoice_state",
            "historical_payments_reversed": False,
            "excludes": OPEN_INVOICE_EXCLUDES,
            "total_count": len(overdue),
            "total_amount_due_by_currency": _money_totals(totals),
            "ageing_buckets": ageing,
            "by_contact": by_contact,
            "by_contact_complete": len(ranked) == len(contacts),
            "complete": True,
            "skip": skip,
            "returned": len(shown),
            "has_more": has_more,
            "invoices": shown,
        }
        if has_more:
            payload["next_skip"] = next_skip
        if truncated:
            payload["truncated"] = True
        return payload

    return _fit_list(build, window)


RECEIVABLE_BUCKETS = ("Current", "1-15", "16-30", "31-45", "46-60", "61-75", "76+")
RECEIVABLES_DEFAULT_LIMIT = 100
RECEIVABLES_MAX_LIMIT = 500
RECEIVABLES_SPREADSHEET_COLUMNS = (
    {"header": "Company Name", "field": "contact_name"},
    {"header": "Currency", "field": "currency"},
    {"header": "Invoice Count", "field": "invoice_count"},
    *(
        {
            "header": bucket if bucket == "Current" else f"{bucket} Days",
            "field": f"buckets.{bucket}",
        }
        for bucket in RECEIVABLE_BUCKETS
    ),
    {"header": "Total Due", "field": "total_due"},
)


def _receivable_bucket(days_overdue: int) -> str:
    """Xero's aged-receivables buckets; due today or later is Current."""
    if days_overdue <= 0:
        return "Current"
    for upper, name in ((15, "1-15"), (30, "16-30"), (45, "31-45"), (60, "46-60")):
        if days_overdue <= upper:
            return name
    return "61-75" if days_overdue <= 75 else "76+"


@mcp.tool(annotations=READ_ONLY)
def list_receivables_ageing(
    asOfDate: str | None = None,
    contactIds: list[str] | None = None,
    type: Literal["ACCREC", "ACCPAY"] = "ACCREC",
    skip: int = 0,
    limit: int = RECEIVABLES_DEFAULT_LIMIT,
) -> str:
    """
    Gross open-invoice ageing for ALL contacts in one call: one row per
    contact and currency with the amount still due in each bucket (Current,
    1-15, 16-30, 31-45, 46-60, 61-75, 76+ days overdue) and a total. Rows are
    grouped by currency and largest balance first within each currency; every
    row carries the contact's name.

    Built from open invoices, so it does not depend on Xero's
    aged-receivables report. Current means not yet due (due today or later).
    Only Authorised invoices with an amount still due count; paid, draft and
    voided invoices are excluded. Credit notes, overpayments and prepayments
    are not netted, so this gross balance may not reconcile to Xero's native
    aged-receivables report. This tool reads every page from Xero itself.

    The result always carries the complete picture in totals_by_currency,
    even when only part of the rows fit in one response. When has_more is true,
    call again with next_skip to read the remaining rows. Each window re-scans
    the open ledger, so prefer the largest practical limit (up to 500).
    complete confirms the ledger scan and totals are complete; rows_complete
    confirms that pagination has reached the final contact-row page.

    When creating a spreadsheet or table, follow spreadsheet_export: retrieve
    every page, use rows as the row source, and include Company Name from
    contact_name as the first column. totals_by_currency is summary metadata;
    do not substitute it for the contact rows unless the user explicitly asks
    for a summary-only output.

    Historical limitation: a past asOfDate excludes invoices issued later, but
    payments, credits and allocations made after that date are not rolled back.

    asOfDate: YYYY-MM-DD (default: today in the Xero organisation's time zone).
    contactIds: only these contacts (from list_contacts).
    type: ACCREC for receivables (default), ACCPAY for payables. Both use Xero's
        Current/1-15/16-30/31-45/46-60/61-75/76+ report-style bucket scheme;
        list_overdue_invoices uses operational 1-30/31-60/61-90/90+ buckets.
    skip, limit: window into the contact rows (limit 1-500, default 100).
    """
    as_of = _iso_date(asOfDate, "asOfDate") if asOfDate else _organisation_today()
    skip = max(0, int(skip))
    limit = max(1, min(int(limit), RECEIVABLES_MAX_LIMIT))
    invoices = _collect_open_invoices(as_of, type, contactIds, min_days_overdue=None)

    def empty_buckets() -> dict[str, Decimal]:
        return {name: Decimal(0) for name in RECEIVABLE_BUCKETS}

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    totals: dict[str, dict[str, Any]] = {}
    for view in invoices:
        currency = view.get("currency") or "UNKNOWN"
        amount = Decimal(str(view.get("amount_due") or 0))
        bucket = _receivable_bucket(view["days_overdue"])
        contact = view.get("contact") or {}
        key = (contact.get("id") or contact.get("name") or "unknown", currency)
        row = rows.setdefault(
            key,
            {
                "contact_id": contact.get("id"),
                "contact_name": contact.get("name"),
                "currency": currency,
                "invoice_count": 0,
                "buckets": empty_buckets(),
            },
        )
        row["invoice_count"] += 1
        row["buckets"][bucket] += amount
        total = totals.setdefault(currency, {**empty_buckets(), "invoice_count": 0})
        total[bucket] += amount
        total["invoice_count"] += 1

    def render(buckets: dict[str, Decimal]) -> dict[str, str]:
        return {name: f"{buckets[name]:.2f}" for name in RECEIVABLE_BUCKETS}

    ordered = sorted(
        rows.values(),
        key=lambda row: (
            row["currency"],
            -sum(row["buckets"].values(), Decimal(0)),
            row["contact_name"] or "",
        ),
    )
    rendered_rows = [
        _compact(
            {
                "contact_id": row["contact_id"],
                "contact_name": (
                    row["contact_name"] or row["contact_id"] or "Unknown contact"
                ),
                "currency": row["currency"],
                "invoice_count": row["invoice_count"],
                "buckets": render(row["buckets"]),
                "total_due": f"{sum(row['buckets'].values(), Decimal(0)):.2f}",
            }
        )
        for row in ordered
    ]
    totals_by_currency = {
        currency: {
            "invoice_count": total["invoice_count"],
            "buckets": render(total),
            "total_due": f"{sum((total[n] for n in RECEIVABLE_BUCKETS), Decimal(0)):.2f}",
        }
        for currency, total in sorted(totals.items())
    }
    window = rendered_rows[skip : skip + limit]

    def build(shown: list[Any], truncated: bool) -> dict[str, Any]:
        next_skip = skip + len(shown)
        has_more = next_skip < len(rendered_rows)
        payload: dict[str, Any] = {
            "as_of_date": as_of.isoformat(),
            "invoice_type": type,
            "balance_basis": "current_open_invoice_state",
            "historical_payments_reversed": False,
            "excludes": OPEN_INVOICE_EXCLUDES,
            "bucket_names": list(RECEIVABLE_BUCKETS),
            "total_invoice_count": len(invoices),
            "total_rows": len(rendered_rows),
            "totals_by_currency": totals_by_currency,
            "spreadsheet_export": {
                "row_source": "rows",
                "columns": list(RECEIVABLES_SPREADSHEET_COLUMNS),
                "retrieve_all_pages": True,
            },
            "complete": True,
            "rows_complete": not has_more,
            "skip": skip,
            "returned": len(shown),
            "has_more": has_more,
            "rows": shown,
        }
        if has_more:
            payload["next_skip"] = next_skip
        if truncated:
            payload["truncated"] = True
        return payload

    return _fit_list(build, window)


@mcp.tool(annotations=READ_ONLY)
def list_contacts(
    page: int = 1,
    searchTerm: str | None = None,
    page_size: int = 50,
) -> str:
    """
    List contacts (customers and suppliers) in Xero. The result reports
    total_count, has_more and next_page; if has_more is true call again with
    next_page.

    searchTerm: case-insensitive search across name, first name, last name,
        contact number and email address.
    page_size: contacts per page, 1-100 (default 50).
    """
    params: dict[str, Any] = {}
    if searchTerm:
        params["searchTerm"] = searchTerm.strip()
    return _list_paged(
        "Contacts",
        "Contacts",
        "contacts",
        _contact_view,
        scope="contacts",
        page=page,
        page_size=page_size,
        params=params,
    )


@mcp.tool(annotations=READ_ONLY)
def list_credit_notes(
    page: int = 1,
    contactId: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> str:
    """
    List credit notes in Xero. The result reports total_count, has_more and
    next_page; if has_more is true call again with next_page.

    contactId: only credit notes for this contact.
    page_size: credit notes per page, 1-100 (default 20).
    """
    params: dict[str, Any] = {}
    if contactId:
        params["where"] = f'Contact.ContactID==Guid("{_guid(contactId, "contactId")}")'
    return _list_paged(
        "CreditNotes",
        "CreditNotes",
        "credit_notes",
        _credit_note_view,
        scope="invoices",
        page=page,
        page_size=page_size,
        params=params,
    )


@mcp.tool(annotations=READ_ONLY)
def list_bank_transactions(
    page: int = 1,
    bankAccountId: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> str:
    """
    List bank transactions in Xero. The result reports total_count, has_more
    and next_page; if has_more is true call again with next_page.

    bankAccountId: only transactions on this bank account (from list_accounts).
    page_size: transactions per page, 1-100 (default 20).
    """
    params: dict[str, Any] = {}
    if bankAccountId:
        params["where"] = (
            f'BankAccount.AccountID==Guid("{_guid(bankAccountId, "bankAccountId")}")'
        )
    return _list_paged(
        "BankTransactions",
        "BankTransactions",
        "bank_transactions",
        _bank_transaction_view,
        scope="banktransactions",
        page=page,
        page_size=page_size,
        params=params,
    )


@mcp.tool(annotations=READ_ONLY)
def list_payments(
    page: int = 1,
    invoiceNumber: str | None = None,
    invoiceId: str | None = None,
    paymentId: str | None = None,
    reference: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> str:
    """
    List payments made against invoices in Xero, with date, amount and the
    invoice they settle. The result reports total_count, has_more and
    next_page; if has_more is true call again with next_page.

    invoiceNumber / invoiceId / reference: narrow to one invoice or reference.
    paymentId: fetch one payment.
    page_size: payments per page, 1-100 (default 20).
    """
    if paymentId:
        record = _get_one(
            f"Payments/{url_path_id(paymentId, 'paymentId')}",
            "Payments",
            scope="payments",
        )
        return _success(payments=[_payment_view(record)], total_count=1, has_more=False)
    where: list[str] = []
    if invoiceNumber:
        where.append(
            f'Invoice.InvoiceNumber=="{_where_text(invoiceNumber, "invoiceNumber")}"'
        )
    if invoiceId:
        where.append(f'Invoice.InvoiceID==Guid("{_guid(invoiceId, "invoiceId")}")')
    if reference:
        where.append(f'Reference=="{_where_text(reference, "reference")}"')
    params = {"where": " AND ".join(where)} if where else {}
    return _list_paged(
        "Payments",
        "Payments",
        "payments",
        _payment_view,
        scope="payments",
        page=page,
        page_size=page_size,
        params=params,
    )


@mcp.tool(annotations=READ_ONLY)
def list_quotes(
    page: int = 1,
    contactId: str | None = None,
    quoteNumber: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> str:
    """
    List quotes in Xero (up to 100 per page, newest first). If has_more is true
    call again with next_page.

    contactId: only quotes for this contact.
    quoteNumber: only this quote number.
    page_size: quotes per page, 1-100 (default 20).
    """
    params: dict[str, Any] = {"order": "UpdatedDateUTC DESC"}
    if contactId:
        params["ContactID"] = _guid(contactId, "contactId")
    if quoteNumber:
        params["QuoteNumber"] = _where_text(quoteNumber, "quoteNumber")
    return _list_paged(
        "Quotes",
        "Quotes",
        "quotes",
        _quote_view,
        scope="invoices",
        page=page,
        page_size=page_size,
        params=params,
    )


@mcp.tool(annotations=READ_ONLY)
def list_manual_journals(
    manualJournalId: str | None = None,
    modifiedAfter: str | None = None,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> str:
    """
    List manual journals in Xero. The result reports total_count, has_more and
    next_page; if has_more is true call again with next_page.

    manualJournalId: fetch one journal.
    modifiedAfter: YYYY-MM-DD; only journals modified after this date.
    page_size: journals per page, 1-100 (default 20).
    """
    if manualJournalId:
        record = _get_one(
            f"ManualJournals/{url_path_id(manualJournalId, 'manualJournalId')}",
            "ManualJournals",
            scope="manualjournals",
        )
        return _success(
            manual_journals=[_manual_journal_view(record)],
            total_count=1,
            has_more=False,
        )
    headers = None
    if modifiedAfter:
        since = _iso_date(modifiedAfter, "modifiedAfter")
        headers = {"If-Modified-Since": f"{since.isoformat()}T00:00:00"}
    return _list_paged(
        "ManualJournals",
        "ManualJournals",
        "manual_journals",
        _manual_journal_view,
        scope="manualjournals",
        page=page,
        page_size=page_size,
        headers=headers,
    )


@mcp.tool(annotations=READ_ONLY)
def list_items() -> str:
    """List all items (inventory and service items) in Xero, with the item codes
    to use on invoice lines."""
    data = _request("GET", "Items", scope="settings")
    return _unpaged_response(
        "items", [_item_view(record) for record in data.get("Items") or []]
    )


@mcp.tool(annotations=READ_ONLY)
def list_accounts() -> str:
    """List all accounts in the Xero chart of accounts, with the account codes
    to use on invoice lines, journals and payments."""
    data = _request("GET", "Accounts", scope="settings")
    accounts = [
        _compact(
            {
                "account_id": record.get("AccountID"),
                "code": record.get("Code"),
                "name": record.get("Name"),
                "type": record.get("Type"),
                "class": record.get("Class"),
                "status": record.get("Status"),
                "tax_type": record.get("TaxType"),
                "bank_account_type": record.get("BankAccountType"),
                "currency": record.get("CurrencyCode"),
            }
        )
        for record in data.get("Accounts") or []
    ]
    return _unpaged_response("accounts", accounts)


@mcp.tool(annotations=READ_ONLY)
def list_tax_rates() -> str:
    """List all tax rates in Xero, with the tax types to use on invoice lines."""
    data = _request("GET", "TaxRates", scope="settings")
    rates = [
        _compact(
            {
                "name": record.get("Name"),
                "tax_type": record.get("TaxType"),
                "status": record.get("Status"),
                "effective_rate": record.get("EffectiveRate"),
                "display_tax_rate": record.get("DisplayTaxRate"),
            }
        )
        for record in data.get("TaxRates") or []
    ]
    return _unpaged_response("tax_rates", rates)


@mcp.tool(annotations=READ_ONLY)
def list_tracking_categories(includeArchived: bool = False) -> str:
    """List tracking categories in Xero with their tracking options.

    includeArchived: also return archived categories and options.
    """
    params = {"includeArchived": "true"} if includeArchived else None
    data = _request("GET", "TrackingCategories", scope="settings", params=params)
    return _unpaged_response(
        "tracking_categories",
        [_tracking_category_view(r) for r in data.get("TrackingCategories") or []],
    )


@mcp.tool(annotations=READ_ONLY)
def list_contact_groups(contactGroupId: str | None = None) -> str:
    """List contact groups in Xero.

    contactGroupId: fetch one group, including its contacts.
    """
    path = "ContactGroups"
    if contactGroupId:
        path = f"ContactGroups/{url_path_id(contactGroupId, 'contactGroupId')}"
    data = _request("GET", path, scope="contacts")
    groups = []
    for record in data.get("ContactGroups") or []:
        groups.append(
            _compact(
                {
                    "contact_group_id": record.get("ContactGroupID"),
                    "name": record.get("Name"),
                    "status": record.get("Status"),
                    "contacts": [
                        _contact_view(contact)
                        for contact in record.get("Contacts") or []
                    ],
                }
            )
        )
    return _unpaged_response("contact_groups", groups)


@mcp.tool(annotations=READ_ONLY)
def list_organisation_details() -> str:
    """Get details of the connected Xero organisation (name, country, currency,
    financial year end, short code)."""
    data = _request("GET", "Organisation", scope="settings")
    record = _first(data, "Organisations")
    return _success(
        organisation=_compact(
            {
                "name": record.get("Name"),
                "legal_name": record.get("LegalName"),
                "organisation_type": record.get("OrganisationType"),
                "country_code": record.get("CountryCode"),
                "base_currency": record.get("BaseCurrency"),
                "time_zone": record.get("Timezone"),
                "financial_year_end_day": record.get("FinancialYearEndDay"),
                "financial_year_end_month": record.get("FinancialYearEndMonth"),
                "tax_number": record.get("TaxNumber"),
                "short_code": record.get("ShortCode"),
                "organisation_id": record.get("OrganisationID"),
            }
        )
    )


def _aged_report(
    path: str,
    contact_id: str,
    report_date: str | None,
    from_date: str | None,
    to_date: str | None,
) -> str:
    params: dict[str, Any] = {"contactId": _guid(contact_id, "contactId")}
    for name, value, label in (
        ("date", report_date, "reportDate"),
        ("fromDate", from_date, "invoicesFromDate"),
        ("toDate", to_date, "invoicesToDate"),
    ):
        if value:
            params[name] = _iso_date(value, label).isoformat()
    return _report_response(_request("GET", path, scope="aged", params=params))


@mcp.tool(annotations=READ_ONLY)
def list_aged_receivables_by_contact(
    contactId: str,
    reportDate: str | None = None,
    invoicesFromDate: str | None = None,
    invoicesToDate: str | None = None,
) -> str:
    """
    Xero's aged receivables report for one contact up to a report date. For
    every contact at once use list_receivables_ageing, whose gross invoice
    balances use the same bucket boundaries but exclude credits, overpayments
    and prepayments. Use list_overdue_invoices for the overdue invoice rows.

    reportDate: YYYY-MM-DD (default: end of the current month).
    invoicesFromDate / invoicesToDate: YYYY-MM-DD window for the invoices.
    """
    return _aged_report(
        "Reports/AgedReceivablesByContact",
        contactId,
        reportDate,
        invoicesFromDate,
        invoicesToDate,
    )


@mcp.tool(annotations=READ_ONLY)
def list_aged_payables_by_contact(
    contactId: str,
    reportDate: str | None = None,
    invoicesFromDate: str | None = None,
    invoicesToDate: str | None = None,
) -> str:
    """
    Aged payables for one contact up to a report date.

    reportDate: YYYY-MM-DD (default: end of the current month).
    invoicesFromDate / invoicesToDate: YYYY-MM-DD window for the invoices.
    """
    return _aged_report(
        "Reports/AgedPayablesByContact",
        contactId,
        reportDate,
        invoicesFromDate,
        invoicesToDate,
    )


@mcp.tool(annotations=READ_ONLY)
def list_profit_and_loss(
    fromDate: str | None = None,
    toDate: str | None = None,
    periods: int | None = None,
    timeframe: Literal["MONTH", "QUARTER", "YEAR"] | None = None,
    standardLayout: bool | None = None,
    paymentsOnly: bool | None = None,
) -> str:
    """
    Profit and loss report: revenue, expenses and profit or loss over a period.

    fromDate / toDate: YYYY-MM-DD.
    periods: number of comparison periods; timeframe: MONTH, QUARTER or YEAR.
    standardLayout: use Xero's standard layout.
    paymentsOnly: only accounts with payments (cash basis).
    """
    params: dict[str, Any] = {}
    if fromDate:
        params["fromDate"] = _iso_date(fromDate, "fromDate").isoformat()
    if toDate:
        params["toDate"] = _iso_date(toDate, "toDate").isoformat()
    if periods is not None:
        params["periods"] = int(periods)
    if timeframe:
        params["timeframe"] = timeframe
    if standardLayout is not None:
        params["standardLayout"] = str(bool(standardLayout)).lower()
    if paymentsOnly is not None:
        params["paymentsOnly"] = str(bool(paymentsOnly)).lower()
    return _report_response(
        _request("GET", "Reports/ProfitAndLoss", scope="profitandloss", params=params)
    )


@mcp.tool(annotations=READ_ONLY)
def list_report_balance_sheet(
    date: str | None = None,
    periods: int | None = None,
    timeframe: Literal["MONTH", "QUARTER", "YEAR"] | None = None,
    trackingOptionID1: str | None = None,
    trackingOptionID2: str | None = None,
    standardLayout: bool | None = None,
    paymentsOnly: bool | None = None,
) -> str:
    """
    Balance sheet report.

    date: YYYY-MM-DD (default: today).
    periods: number of comparison periods; timeframe: MONTH, QUARTER or YEAR.
    trackingOptionID1 / trackingOptionID2: filter by tracking options.
    standardLayout: use Xero's standard layout.
    paymentsOnly: only accounts with payments (cash basis).
    """
    params: dict[str, Any] = {}
    if date:
        params["date"] = _iso_date(date, "date").isoformat()
    if periods is not None:
        params["periods"] = int(periods)
    if timeframe:
        params["timeframe"] = timeframe
    if trackingOptionID1:
        params["trackingOptionID1"] = _guid(trackingOptionID1, "trackingOptionID1")
    if trackingOptionID2:
        params["trackingOptionID2"] = _guid(trackingOptionID2, "trackingOptionID2")
    if standardLayout is not None:
        params["standardLayout"] = str(bool(standardLayout)).lower()
    if paymentsOnly is not None:
        params["paymentsOnly"] = str(bool(paymentsOnly)).lower()
    return _report_response(
        _request("GET", "Reports/BalanceSheet", scope="balancesheet", params=params)
    )


@mcp.tool(annotations=READ_ONLY)
def list_trial_balance(
    date: str | None = None, paymentsOnly: bool | None = None
) -> str:
    """
    Trial balance: debit and credit balances for each account.

    date: YYYY-MM-DD (default: today).
    paymentsOnly: only accounts with payments (cash basis).
    """
    params: dict[str, Any] = {}
    if date:
        params["date"] = _iso_date(date, "date").isoformat()
    if paymentsOnly is not None:
        params["paymentsOnly"] = str(bool(paymentsOnly)).lower()
    return _report_response(
        _request("GET", "Reports/TrialBalance", scope="trialbalance", params=params)
    )


# --------------------------------------------------------------------------- #
# Creates. Every create returns the created record and a deep link to it.
# --------------------------------------------------------------------------- #


@mcp.tool(annotations=CREATE)
def create_invoice(
    contactId: str,
    lineItems: list[LineItem],
    type: Literal["ACCREC", "ACCPAY"] = "ACCREC",
    reference: str | None = None,
    date: str | None = None,
) -> str:
    """
    Create a DRAFT invoice in Xero, due 30 days after today. Returns the
    invoice and a deep link to show the user.

    contactId: from list_contacts.
    lineItems: account codes come from list_accounts, tax types from
        list_tax_rates, item codes from list_items.
    type: ACCREC for a sales invoice (default), ACCPAY for a bill.
    reference: reference number (for ACCPAY it becomes the invoice number).
    date: YYYY-MM-DD invoice date (default: today).
    This is not idempotent: check list_invoices before retrying.
    """
    issued = _iso_date(date, "date") if date else _today()
    body: dict[str, Any] = {
        "Type": type,
        "Contact": {"ContactID": _guid(contactId, "contactId")},
        "LineItems": _line_item_bodies(lineItems),
        "Date": issued.isoformat(),
        "DueDate": (_today() + timedelta(days=DEFAULT_INVOICE_TERMS_DAYS)).isoformat(),
        "Status": "DRAFT",
    }
    if reference:
        body["InvoiceNumber" if type == "ACCPAY" else "Reference"] = reference
    data = _request(
        "PUT",
        "Invoices",
        scope="invoices",
        json_data={"Invoices": [body]},
    )
    record = _first(data, "Invoices")
    link = _deep_link(
        "bill" if record.get("Type") == "ACCPAY" else "invoice", record.get("InvoiceID")
    )
    return _success(invoice=_invoice_view(record, with_lines=True), deep_link=link)


@mcp.tool(annotations=CREATE)
def create_contact(
    name: str, email: str | None = None, phone: str | None = None
) -> str:
    """Create a contact in Xero. Returns the contact and a deep link to show the
    user. Not idempotent: check list_contacts (searchTerm) before retrying."""
    body: dict[str, Any] = {"Name": name}
    if email:
        body["EmailAddress"] = email
    if phone:
        body["Phones"] = [{"PhoneType": "MOBILE", "PhoneNumber": phone}]
    data = _request(
        "PUT",
        "Contacts",
        scope="contacts",
        json_data={"Contacts": [body]},
    )
    record = _first(data, "Contacts")
    return _success(
        contact=_contact_view(record),
        deep_link=_deep_link("contact", record.get("ContactID")),
    )


@mcp.tool(annotations=CREATE)
def create_credit_note(
    contactId: str, lineItems: list[LineItem], reference: str | None = None
) -> str:
    """Create a DRAFT sales credit note in Xero dated today. Returns the credit
    note and a deep link to show the user. Not idempotent."""
    body: dict[str, Any] = {
        "Type": "ACCRECCREDIT",
        "Contact": {"ContactID": _guid(contactId, "contactId")},
        "LineItems": _line_item_bodies(lineItems),
        "Date": _today().isoformat(),
        "Status": "DRAFT",
    }
    if reference:
        body["Reference"] = reference
    data = _request(
        "PUT",
        "CreditNotes",
        scope="invoices",
        json_data={"CreditNotes": [body]},
    )
    record = _first(data, "CreditNotes")
    return _success(
        credit_note=_credit_note_view(record, with_lines=True),
        deep_link=_deep_link("credit_note", record.get("CreditNoteID")),
    )


@mcp.tool(annotations=CREATE)
def create_quote(
    contactId: str,
    lineItems: list[LineItem],
    reference: str | None = None,
    quoteNumber: str | None = None,
    terms: str | None = None,
    title: str | None = None,
    summary: str | None = None,
) -> str:
    """Create a DRAFT quote in Xero dated today, expiring in 7 days. Returns the
    quote and a deep link to show the user. Not idempotent."""
    body: dict[str, Any] = {
        "Contact": {"ContactID": _guid(contactId, "contactId")},
        "LineItems": _line_item_bodies(lineItems),
        "Date": _today().isoformat(),
        "ExpiryDate": (_today() + timedelta(days=DEFAULT_QUOTE_VALID_DAYS)).isoformat(),
        "Status": "DRAFT",
    }
    for key, value in (
        ("QuoteNumber", quoteNumber),
        ("Reference", reference),
        ("Terms", terms),
        ("Title", title),
        ("Summary", summary),
    ):
        if value:
            body[key] = value
    data = _request(
        "PUT",
        "Quotes",
        scope="invoices",
        json_data={"Quotes": [body]},
    )
    record = _first(data, "Quotes")
    return _success(
        quote=_quote_view(record, with_lines=True),
        deep_link=_deep_link("quote", record.get("QuoteID")),
    )


@mcp.tool(annotations=CREATE)
def create_bank_transaction(
    type: Literal["RECEIVE", "SPEND"],
    bankAccountId: str,
    contactId: str,
    lineItems: list[LineItem],
    reference: str | None = None,
    date: str | None = None,
) -> str:
    """Create an AUTHORISED bank transaction in Xero. Returns the transaction.
    Not idempotent.

    bankAccountId: a bank account from list_accounts.
    date: YYYY-MM-DD (default: today).
    """
    issued = _iso_date(date, "date") if date else _today()
    body: dict[str, Any] = {
        "Type": type,
        "BankAccount": {"AccountID": _guid(bankAccountId, "bankAccountId")},
        "Contact": {"ContactID": _guid(contactId, "contactId")},
        "LineItems": _line_item_bodies(lineItems),
        "Date": issued.isoformat(),
        "Status": "AUTHORISED",
    }
    if reference:
        body["Reference"] = reference
    data = _request(
        "PUT",
        "BankTransactions",
        scope="banktransactions",
        json_data={"BankTransactions": [body]},
    )
    record = _first(data, "BankTransactions")
    return _success(bank_transaction=_bank_transaction_view(record, with_lines=True))


@mcp.tool(annotations=CREATE)
def create_payment(
    invoiceId: str,
    accountId: str,
    amount: float,
    date: str | None = None,
    reference: str | None = None,
) -> str:
    """
    Record a payment against an AUTHORISED invoice that is not fully paid. The
    amount must be positive and not exceed the amount still due. Returns the
    payment. Not idempotent: check list_payments before retrying.

    accountId: the account the payment is made from (list_accounts).
    date: YYYY-MM-DD (default: today).
    """
    if amount <= 0:
        raise ValueError("amount must be positive")
    paid_on = _iso_date(date, "date") if date else _today()
    body: dict[str, Any] = {
        "Invoice": {"InvoiceID": _guid(invoiceId, "invoiceId")},
        "Account": {"AccountID": _guid(accountId, "accountId")},
        "Amount": amount,
        "Date": paid_on.isoformat(),
    }
    if reference:
        body["Reference"] = reference
    data = _request("POST", "Payments", scope="payments", json_data=body)
    record = _first(data, "Payments")
    return _success(payment=_payment_view(record))


@mcp.tool(annotations=CREATE)
def create_item(
    code: str,
    name: str,
    description: str | None = None,
    purchaseDescription: str | None = None,
    purchaseDetails: ItemDetails | None = None,
    salesDetails: ItemDetails | None = None,
    isTrackedAsInventory: bool | None = None,
    inventoryAssetAccountCode: str | None = None,
) -> str:
    """Create an item in Xero. Giving purchaseDetails marks it as purchased and
    salesDetails as sold. Not idempotent."""
    body: dict[str, Any] = {
        "Code": code,
        "Name": name,
        "IsPurchased": purchaseDetails is not None,
        "IsSold": salesDetails is not None,
    }
    if description:
        body["Description"] = description
    if purchaseDescription:
        body["PurchaseDescription"] = purchaseDescription
    if purchaseDetails:
        body["PurchaseDetails"] = _item_details_body(purchaseDetails)
    if salesDetails:
        body["SalesDetails"] = _item_details_body(salesDetails)
    if isTrackedAsInventory is not None:
        body["IsTrackedAsInventory"] = isTrackedAsInventory
    if inventoryAssetAccountCode:
        body["InventoryAssetAccountCode"] = inventoryAssetAccountCode
    data = _request(
        "PUT",
        "Items",
        scope="settings",
        json_data={"Items": [body]},
    )
    return _success(item=_item_view(_first(data, "Items")))


def _journal_body(
    narration: str,
    lines: list[ManualJournalLine],
    date_value: str | None,
    line_amount_types: str | None,
    status: str | None,
    url: str | None,
    show_on_cash_basis_reports: bool | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "Narration": narration,
        "JournalLines": _journal_line_bodies(lines),
    }
    if date_value:
        body["Date"] = _iso_date(date_value, "date").isoformat()
    if line_amount_types:
        body["LineAmountTypes"] = line_amount_types
    if status:
        body["Status"] = status
    if url:
        body["Url"] = url
    if show_on_cash_basis_reports is not None:
        body["ShowOnCashBasisReports"] = show_on_cash_basis_reports
    return body


@mcp.tool(annotations=CREATE)
def create_manual_journal(
    narration: str,
    manualJournalLines: list[ManualJournalLine],
    date: str | None = None,
    lineAmountTypes: Literal["EXCLUSIVE", "INCLUSIVE", "NO_TAX"] | None = None,
    status: Literal["DRAFT", "POSTED", "DELETED", "VOIDED", "ARCHIVED"] | None = None,
    url: str | None = None,
    showOnCashBasisReports: bool | None = None,
) -> str:
    """
    Create a manual journal in Xero. It needs at least two lines whose amounts
    sum to zero (debits positive, credits negative); account codes come from
    list_accounts. Status defaults to DRAFT. Returns the journal and a deep
    link. Not idempotent.
    """
    body = _journal_body(
        narration,
        manualJournalLines,
        date,
        lineAmountTypes,
        status,
        url,
        showOnCashBasisReports,
    )
    data = _request(
        "PUT",
        "ManualJournals",
        scope="manualjournals",
        json_data={"ManualJournals": [body]},
    )
    record = _first(data, "ManualJournals")
    return _success(
        manual_journal=_manual_journal_view(record),
        deep_link=_deep_link("manual_journal", record.get("ManualJournalID")),
    )


@mcp.tool(annotations=CREATE)
def create_tracking_category(name: str) -> str:
    """Create a tracking category in Xero. Not idempotent."""
    data = _request(
        "PUT", "TrackingCategories", scope="settings", json_data={"Name": name}
    )
    return _success(
        tracking_category=_tracking_category_view(_first(data, "TrackingCategories"))
    )


@mcp.tool(annotations=CREATE)
def create_tracking_options(trackingCategoryId: str, optionNames: list[str]) -> str:
    """Create up to 10 tracking options in a tracking category. Not idempotent."""
    if not 1 <= len(optionNames) <= 10:
        raise ValueError("optionNames must contain between 1 and 10 names")
    category_id = url_path_id(trackingCategoryId, "trackingCategoryId")
    created = []
    for option_name in optionNames:
        data = _request(
            "PUT",
            f"TrackingCategories/{category_id}/Options",
            scope="settings",
            json_data={"Name": option_name},
        )
        for option in data.get("Options") or []:
            created.append(
                _compact(
                    {
                        "option_id": option.get("TrackingOptionID"),
                        "name": option.get("Name"),
                        "status": option.get("Status"),
                    }
                )
            )
    return _success(options=created, total_count=len(created))


# --------------------------------------------------------------------------- #
# Updates
# --------------------------------------------------------------------------- #


@mcp.tool(annotations=UPDATE)
def update_invoice(
    invoiceId: str,
    lineItems: list[LineItem] | None = None,
    reference: str | None = None,
    dueDate: str | None = None,
    date: str | None = None,
    contactId: str | None = None,
) -> str:
    """
    Update a DRAFT or SUBMITTED invoice in Xero. If lineItems is given it
    replaces ALL existing lines: send every line to keep, including unchanged
    ones. Returns the invoice and a deep link to show the user.
    """
    safe_id = url_path_id(invoiceId, "invoiceId")
    existing = _get_one(f"Invoices/{safe_id}", "Invoices", scope="invoices")
    _require_draft(existing, "invoice", {"DRAFT", "SUBMITTED"})
    body: dict[str, Any] = {"InvoiceID": safe_id}
    if lineItems is not None:
        body["LineItems"] = _line_item_bodies(lineItems)
    if reference:
        body["Reference"] = reference
    if dueDate:
        body["DueDate"] = _iso_date(dueDate, "dueDate").isoformat()
    if date:
        body["Date"] = _iso_date(date, "date").isoformat()
    if contactId:
        body["Contact"] = {"ContactID": _guid(contactId, "contactId")}
    data = _request(
        "POST", f"Invoices/{safe_id}", scope="invoices", json_data={"Invoices": [body]}
    )
    record = _first(data, "Invoices")
    link = _deep_link(
        "bill" if record.get("Type") == "ACCPAY" else "invoice", record.get("InvoiceID")
    )
    return _success(invoice=_invoice_view(record, with_lines=True), deep_link=link)


@mcp.tool(annotations=UPDATE)
def update_contact(
    contactId: str,
    name: str,
    firstName: str | None = None,
    lastName: str | None = None,
    email: str | None = None,
    phone: str | None = None,
    address: Address | None = None,
) -> str:
    """Update a contact in Xero. Returns the contact and a deep link to show the
    user."""
    safe_id = url_path_id(contactId, "contactId")
    body: dict[str, Any] = {"ContactID": safe_id, "Name": name}
    if firstName:
        body["FirstName"] = firstName
    if lastName:
        body["LastName"] = lastName
    if email:
        body["EmailAddress"] = email
    if phone:
        body["Phones"] = [{"PhoneType": "MOBILE", "PhoneNumber": phone}]
    if address:
        body["Addresses"] = [
            _compact(
                {
                    "AddressType": "STREET",
                    "AddressLine1": address.addressLine1,
                    "AddressLine2": address.addressLine2,
                    "City": address.city,
                    "Region": address.region,
                    "PostalCode": address.postalCode,
                    "Country": address.country,
                }
            )
        ]
    data = _request(
        "POST", f"Contacts/{safe_id}", scope="contacts", json_data={"Contacts": [body]}
    )
    record = _first(data, "Contacts")
    return _success(
        contact=_contact_view(record),
        deep_link=_deep_link("contact", record.get("ContactID")),
    )


@mcp.tool(annotations=UPDATE)
def update_credit_note(
    creditNoteId: str,
    lineItems: list[LineItem] | None = None,
    reference: str | None = None,
    date: str | None = None,
    contactId: str | None = None,
) -> str:
    """
    Update a DRAFT credit note in Xero. If lineItems is given it replaces ALL
    existing lines: send every line to keep. Returns the credit note and a deep
    link.
    """
    safe_id = url_path_id(creditNoteId, "creditNoteId")
    existing = _get_one(f"CreditNotes/{safe_id}", "CreditNotes", scope="invoices")
    _require_draft(existing, "credit note", {"DRAFT"})
    body: dict[str, Any] = {"CreditNoteID": safe_id}
    if lineItems is not None:
        body["LineItems"] = _line_item_bodies(lineItems)
    if reference:
        body["Reference"] = reference
    if date:
        body["Date"] = _iso_date(date, "date").isoformat()
    if contactId:
        body["Contact"] = {"ContactID": _guid(contactId, "contactId")}
    data = _request(
        "POST",
        f"CreditNotes/{safe_id}",
        scope="invoices",
        json_data={"CreditNotes": [body]},
    )
    record = _first(data, "CreditNotes")
    return _success(
        credit_note=_credit_note_view(record, with_lines=True),
        deep_link=_deep_link("credit_note", record.get("CreditNoteID")),
    )


@mcp.tool(annotations=UPDATE)
def update_quote(
    quoteId: str,
    lineItems: list[LineItem] | None = None,
    reference: str | None = None,
    terms: str | None = None,
    title: str | None = None,
    summary: str | None = None,
    quoteNumber: str | None = None,
    contactId: str | None = None,
    date: str | None = None,
    expiryDate: str | None = None,
) -> str:
    """
    Update a DRAFT quote in Xero. If lineItems is given it replaces ALL existing
    lines: send every line to keep. Returns the quote and a deep link.
    """
    safe_id = url_path_id(quoteId, "quoteId")
    existing = _get_one(f"Quotes/{safe_id}", "Quotes", scope="invoices")
    _require_draft(existing, "quote", {"DRAFT"})
    # Xero rejects a quote update that omits its contact and date.
    existing_contact = existing.get("Contact") or {}
    body: dict[str, Any] = {"QuoteID": safe_id}
    if existing_contact.get("ContactID"):
        body["Contact"] = {"ContactID": existing_contact["ContactID"]}
    if _doc_date(existing, "Date"):
        body["Date"] = _doc_date(existing, "Date")
    if contactId:
        body["Contact"] = {"ContactID": _guid(contactId, "contactId")}
    if date:
        body["Date"] = _iso_date(date, "date").isoformat()
    if expiryDate:
        body["ExpiryDate"] = _iso_date(expiryDate, "expiryDate").isoformat()
    if lineItems is not None:
        body["LineItems"] = _line_item_bodies(lineItems)
    for key, value in (
        ("Reference", reference),
        ("Terms", terms),
        ("Title", title),
        ("Summary", summary),
        ("QuoteNumber", quoteNumber),
    ):
        if value:
            body[key] = value
    data = _request(
        "POST", f"Quotes/{safe_id}", scope="invoices", json_data={"Quotes": [body]}
    )
    record = _first(data, "Quotes")
    return _success(
        quote=_quote_view(record, with_lines=True),
        deep_link=_deep_link("quote", record.get("QuoteID")),
    )


@mcp.tool(annotations=UPDATE)
def update_bank_transaction(
    bankTransactionId: str,
    type: Literal["RECEIVE", "SPEND"] | None = None,
    contactId: str | None = None,
    lineItems: list[LineItem] | None = None,
    reference: str | None = None,
    date: str | None = None,
) -> str:
    """
    Update a bank transaction in Xero. If lineItems is given it replaces ALL
    existing lines: send every line to keep.
    """
    safe_id = url_path_id(bankTransactionId, "bankTransactionId")
    existing = _get_one(
        f"BankTransactions/{safe_id}", "BankTransactions", scope="banktransactions"
    )
    body: dict[str, Any] = {**existing, "BankTransactionID": safe_id}
    if type:
        body["Type"] = type
    if contactId:
        body["Contact"] = {"ContactID": _guid(contactId, "contactId")}
    if lineItems is not None:
        body["LineItems"] = _line_item_bodies(lineItems)
    if reference:
        body["Reference"] = reference
    if date:
        body["Date"] = _iso_date(date, "date").isoformat()
    data = _request(
        "POST",
        f"BankTransactions/{safe_id}",
        scope="banktransactions",
        json_data={"BankTransactions": [body]},
    )
    record = _first(data, "BankTransactions")
    return _success(bank_transaction=_bank_transaction_view(record, with_lines=True))


@mcp.tool(annotations=UPDATE)
def update_item(
    itemId: str,
    code: str,
    name: str,
    description: str | None = None,
    purchaseDescription: str | None = None,
    purchaseDetails: ItemDetails | None = None,
    salesDetails: ItemDetails | None = None,
    isTrackedAsInventory: bool | None = None,
    inventoryAssetAccountCode: str | None = None,
) -> str:
    """Update an item in Xero."""
    safe_id = url_path_id(itemId, "itemId")
    body: dict[str, Any] = {"ItemID": safe_id, "Code": code, "Name": name}
    if description:
        body["Description"] = description
    if purchaseDescription:
        body["PurchaseDescription"] = purchaseDescription
    if purchaseDetails:
        body["PurchaseDetails"] = _item_details_body(purchaseDetails)
    if salesDetails:
        body["SalesDetails"] = _item_details_body(salesDetails)
    if isTrackedAsInventory is not None:
        body["IsTrackedAsInventory"] = isTrackedAsInventory
    if inventoryAssetAccountCode:
        body["InventoryAssetAccountCode"] = inventoryAssetAccountCode
    data = _request(
        "POST", f"Items/{safe_id}", scope="settings", json_data={"Items": [body]}
    )
    return _success(item=_item_view(_first(data, "Items")))


@mcp.tool(annotations=UPDATE)
def update_manual_journal(
    manualJournalId: str,
    narration: str,
    manualJournalLines: list[ManualJournalLine],
    date: str | None = None,
    lineAmountTypes: Literal["EXCLUSIVE", "INCLUSIVE", "NO_TAX"] | None = None,
    status: Literal["DRAFT", "POSTED", "DELETED", "VOIDED", "ARCHIVED"] | None = None,
    url: str | None = None,
    showOnCashBasisReports: bool | None = None,
) -> str:
    """
    Update a manual journal in Xero. The lines replace the existing ones and
    must sum to zero (debits positive, credits negative). Returns the journal
    and a deep link.
    """
    safe_id = url_path_id(manualJournalId, "manualJournalId")
    body = _journal_body(
        narration,
        manualJournalLines,
        date,
        lineAmountTypes,
        status,
        url,
        showOnCashBasisReports,
    )
    body["ManualJournalID"] = safe_id
    data = _request(
        "POST",
        f"ManualJournals/{safe_id}",
        scope="manualjournals",
        json_data={"ManualJournals": [body]},
    )
    record = _first(data, "ManualJournals")
    return _success(
        manual_journal=_manual_journal_view(record),
        deep_link=_deep_link("manual_journal", record.get("ManualJournalID")),
    )


@mcp.tool(annotations=UPDATE)
def update_tracking_category(
    trackingCategoryId: str,
    name: str | None = None,
    status: Literal["ACTIVE", "ARCHIVED"] | None = None,
) -> str:
    """Rename or archive a tracking category in Xero."""
    safe_id = url_path_id(trackingCategoryId, "trackingCategoryId")
    existing = _get_one(
        f"TrackingCategories/{safe_id}", "TrackingCategories", scope="settings"
    )
    body = {
        "TrackingCategoryID": safe_id,
        "Name": name or existing.get("Name"),
        "Status": status or existing.get("Status"),
    }
    data = _request(
        "POST", f"TrackingCategories/{safe_id}", scope="settings", json_data=body
    )
    return _success(
        tracking_category=_tracking_category_view(_first(data, "TrackingCategories"))
    )


@mcp.tool(annotations=UPDATE)
def update_tracking_options(
    trackingCategoryId: str, options: list[TrackingOptionUpdate]
) -> str:
    """Rename or archive up to 10 tracking options of a tracking category."""
    if not 1 <= len(options) <= 10:
        raise ValueError("options must contain between 1 and 10 entries")
    category_id = url_path_id(trackingCategoryId, "trackingCategoryId")
    existing = _get_one(
        f"TrackingCategories/{category_id}", "TrackingCategories", scope="settings"
    )
    current = {
        option.get("TrackingOptionID"): option
        for option in existing.get("Options") or []
    }
    updated = []
    for option in options:
        option_id = url_path_id(option.trackingOptionId, "trackingOptionId")
        known = current.get(option_id)
        if known is None:
            raise ValueError(
                f"Tracking option {option_id} is not in tracking category {category_id}"
            )
        body = {
            "TrackingOptionID": option_id,
            "Name": option.name or known.get("Name"),
            "Status": option.status or known.get("Status"),
        }
        data = _request(
            "POST",
            f"TrackingCategories/{category_id}/Options/{option_id}",
            scope="settings",
            json_data=body,
        )
        for result in data.get("Options") or []:
            updated.append(
                _compact(
                    {
                        "option_id": result.get("TrackingOptionID"),
                        "name": result.get("Name"),
                        "status": result.get("Status"),
                    }
                )
            )
    return _success(options=updated, total_count=len(updated))


if __name__ == "__main__":
    mcp.run()
