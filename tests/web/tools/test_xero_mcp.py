import asyncio
import json
from datetime import date
from typing import Any, Callable

import pytest

from xagent.web.tools.mcp import xero

TENANT = "tenant-1"
CONTACT_ID = "11111111-2222-3333-4444-555555555555"
INVOICE_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


class MockResponse:
    def __init__(
        self,
        json_data: Any = None,
        status_code: int = 200,
        text: str = "",
        headers: dict[str, str] | None = None,
    ):
        self._json_data = json_data if json_data is not None else {}
        self.status_code = status_code
        self.text = text or (json.dumps(self._json_data) if json_data else "")
        self.content = self.text.encode()
        self.headers = headers or {}

    def json(self) -> Any:
        return self._json_data


class FakeXero:
    """Stands in for ``requests``: records calls, answers from ``responder``."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.responder: Callable[[dict[str, Any]], MockResponse] = lambda call: (
            MockResponse({})
        )

    def request(self, **kwargs: Any) -> MockResponse:
        call = {key: kwargs.get(key) for key in ("method", "url", "params", "json")}
        call["path"] = kwargs["url"].removeprefix(xero.XERO_API_BASE)
        call["headers"] = kwargs.get("headers")
        self.calls.append(call)
        return self.responder(call)

    def get(self, url: str, **kwargs: Any) -> MockResponse:
        assert url == xero.XERO_CONNECTIONS_URL
        self.calls.append({"method": "GET", "url": url, "path": "connections"})
        return MockResponse([{"tenantId": TENANT}])


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeXero:
    monkeypatch.setenv("XERO_CLIENT_BEARER_TOKEN", "token")
    monkeypatch.delenv("XERO_TENANT_ID", raising=False)
    xero._connection_cache.clear()
    fake_xero = FakeXero()
    monkeypatch.setattr(xero.requests, "request", fake_xero.request)
    monkeypatch.setattr(xero.requests, "get", fake_xero.get)
    return fake_xero


def _invoice(
    number: str,
    due: str,
    amount_due: float,
    *,
    contact: str = "Acme",
    contact_id: str = CONTACT_ID,
    currency: str = "AUD",
    status: str = "AUTHORISED",
) -> dict[str, Any]:
    return {
        "InvoiceID": f"id-{number}",
        "InvoiceNumber": number,
        "Type": "ACCREC",
        "Status": status,
        "Contact": {"ContactID": contact_id, "Name": contact},
        "DateString": "2026-08-01T00:00:00",
        "DueDateString": f"{due}T00:00:00",
        "CurrencyCode": currency,
        "Total": amount_due,
        "AmountDue": amount_due,
        "AmountPaid": 0,
    }


def _call_tool(name: str, **arguments: Any) -> dict[str, Any]:
    async def run() -> Any:
        return await xero.mcp.call_tool(name, arguments)

    result = asyncio.run(run())
    content = result[0] if isinstance(result, tuple) else result
    return json.loads(content[0].text)


# --------------------------------------------------------------------------- #
# Connection
# --------------------------------------------------------------------------- #


def test_token_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XERO_CLIENT_BEARER_TOKEN", raising=False)

    with pytest.raises(ValueError, match="XERO_CLIENT_BEARER_TOKEN"):
        xero._access_token()


def test_tenant_is_looked_up_once_and_sent_on_every_call(fake: FakeXero) -> None:
    fake.responder = lambda call: MockResponse({"Accounts": []})

    xero.list_accounts()
    xero.list_accounts()

    assert [call["path"] for call in fake.calls].count("connections") == 1
    api_calls = [call for call in fake.calls if call["path"] == "Accounts"]
    assert len(api_calls) == 2
    assert all(call["headers"]["xero-tenant-id"] == TENANT for call in api_calls)
    assert api_calls[0]["headers"]["Authorization"] == "Bearer token"


def test_explicit_tenant_id_skips_the_connections_lookup(
    fake: FakeXero, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XERO_TENANT_ID", "pinned")
    fake.responder = lambda call: MockResponse({"Accounts": []})

    xero.list_accounts()

    assert [call["path"] for call in fake.calls] == ["Accounts"]
    assert fake.calls[0]["headers"]["xero-tenant-id"] == "pinned"


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("status", [401, 403])
def test_denied_request_names_the_missing_scope(fake: FakeXero, status: int) -> None:
    fake.responder = lambda call: MockResponse(
        {"Title": "Forbidden"}, status_code=status
    )

    with pytest.raises(RuntimeError) as error:
        xero.list_aged_receivables_by_contact(contactId=CONTACT_ID)

    message = str(error.value)
    assert f"status {status}" in message
    assert "accounting.reports.aged.read" in message
    assert "reconnect" in message.lower()


def test_rate_limit_error_reports_retry_after(fake: FakeXero) -> None:
    fake.responder = lambda call: MockResponse(
        {}, status_code=429, headers={"Retry-After": "17"}
    )

    with pytest.raises(RuntimeError, match="Retry after 17 seconds"):
        xero.list_accounts()


def test_validation_errors_are_surfaced(fake: FakeXero) -> None:
    fake.responder = lambda call: MockResponse(
        {
            "Message": "A validation exception occurred",
            "Elements": [
                {"ValidationErrors": [{"Message": "Account code '999' is not valid"}]}
            ],
        },
        status_code=400,
    )

    with pytest.raises(RuntimeError, match="Account code '999' is not valid"):
        xero.create_contact(name="Acme")


def test_failures_reach_the_agent_as_tool_errors(fake: FakeXero) -> None:
    """A failed call must come back flagged ``isError`` so it is never counted
    as a success."""
    fake.responder = lambda call: MockResponse({}, status_code=500, text="boom")

    async def run() -> Any:
        from mcp.server.fastmcp.exceptions import ToolError

        with pytest.raises(ToolError, match="status 500"):
            await xero.mcp.call_tool("list_accounts", {})

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# Paged lists
# --------------------------------------------------------------------------- #


def test_list_invoices_reports_exact_paging_facts(fake: FakeXero) -> None:
    fake.responder = lambda call: MockResponse(
        {
            "pagination": {"page": 2, "pageSize": 20, "pageCount": 5, "itemCount": 93},
            "Invoices": [_invoice("INV-1", "2026-09-01", 10)],
        }
    )

    result = _call_tool("list_invoices", page=2)

    assert result["status"] == "success"
    assert result["total_count"] == 93
    assert result["page_count"] == 5
    assert result["has_more"] is True
    assert result["next_page"] == 3
    assert result["invoices"][0]["due_date"] == "2026-09-01T00:00:00"
    sent = fake.calls[-1]["params"]
    assert sent["page"] == 2 and sent["pageSize"] == 20


def test_list_invoices_last_page_has_no_next_page(fake: FakeXero) -> None:
    fake.responder = lambda call: MockResponse(
        {
            "pagination": {"page": 5, "pageSize": 20, "pageCount": 5, "itemCount": 93},
            "Invoices": [],
        }
    )

    result = _call_tool("list_invoices", page=5)

    assert result["has_more"] is False
    assert "next_page" not in result


def test_list_invoices_builds_filters(fake: FakeXero) -> None:
    fake.responder = lambda call: MockResponse({"Invoices": []})

    _call_tool(
        "list_invoices",
        type="ACCREC",
        statuses=["authorised", "PAID"],
        contactIds=[CONTACT_ID],
        dueDateBefore="2026-10-13",
        page_size=500,
    )

    params = fake.calls[-1]["params"]
    assert params["where"] == 'Type=="ACCREC" AND DueDate<DateTime(2026,10,13)'
    assert params["Statuses"] == "AUTHORISED,PAID"
    assert params["ContactIDs"] == CONTACT_ID
    assert params["pageSize"] == 100


def test_list_invoices_includes_lines_only_for_named_invoices(fake: FakeXero) -> None:
    record = _invoice("INV-9", "2026-09-01", 10)
    record["LineItems"] = [
        {"Description": "Widget", "Quantity": 2, "UnitAmount": 5, "AccountCode": "200"}
    ]
    fake.responder = lambda call: MockResponse({"Invoices": [record]})

    plain = _call_tool("list_invoices")
    named = _call_tool("list_invoices", invoiceNumbers=["INV-9"])

    assert "line_items" not in plain["invoices"][0]
    assert named["invoices"][0]["line_items"][0]["description"] == "Widget"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"statuses": ["BOGUS"]}, "Unknown invoice status"),
        ({"contactIds": ['x"; DROP']}, "contactIds must be a Xero GUID"),
        ({"dueDateBefore": "13/10/2026"}, "dueDateBefore must be YYYY-MM-DD"),
        ({"invoiceNumbers": ['INV"1']}, "invoiceNumbers contains characters"),
    ],
)
def test_list_invoices_rejects_unsafe_input(
    fake: FakeXero, arguments: dict[str, Any], message: str
) -> None:
    from mcp.server.fastmcp.exceptions import ToolError

    with pytest.raises(ToolError, match=message):
        asyncio.run(xero.mcp.call_tool("list_invoices", arguments))

    assert fake.calls == []


def test_list_payments_where_clause_is_validated_and_built(fake: FakeXero) -> None:
    fake.responder = lambda call: MockResponse({"Payments": []})

    xero.list_payments(invoiceId=INVOICE_ID, reference="REF-1")

    assert fake.calls[-1]["params"]["where"] == (
        f'Invoice.InvoiceID==Guid("{INVOICE_ID}") AND Reference=="REF-1"'
    )
    with pytest.raises(ValueError, match="reference contains characters"):
        xero.list_payments(reference='x" OR 1==1')


def test_unpaged_list_falls_back_to_full_page_heuristic(fake: FakeXero) -> None:
    fake.responder = lambda call: MockResponse(
        {"Quotes": [{"QuoteID": str(i)} for i in range(100)]}
    )

    result = _call_tool("list_quotes", page=1)

    assert result["has_more"] is True
    assert result["next_page"] == 2


def test_oversized_page_is_trimmed_and_flagged(
    fake: FakeXero, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(xero, "get_tool_max_output_length", lambda: 1500)
    fake.responder = lambda call: MockResponse(
        {
            "pagination": {"page": 1, "pageSize": 20, "pageCount": 1, "itemCount": 20},
            "Invoices": [_invoice(f"INV-{i}", "2026-09-01", 10) for i in range(20)],
        }
    )

    result = _call_tool("list_invoices")

    assert result["truncated"] is True
    assert result["returned"] < 20
    assert "smaller page_size" in result["note"]
    assert "next_page" not in result


# --------------------------------------------------------------------------- #
# Overdue invoices
# --------------------------------------------------------------------------- #


def _overdue_pages(pages: list[list[dict[str, Any]]]) -> Callable[..., MockResponse]:
    def respond(call: dict[str, Any]) -> MockResponse:
        page = call["params"]["page"]
        return MockResponse(
            {
                "pagination": {
                    "page": page,
                    "pageSize": 100,
                    "pageCount": len(pages),
                    "itemCount": sum(len(p) for p in pages),
                },
                "Invoices": pages[page - 1],
            }
        )

    return respond


def test_overdue_reads_every_page(fake: FakeXero) -> None:
    pages = [
        [_invoice(f"INV-{p}{i}", "2026-09-01", 100) for i in range(3)] for p in range(4)
    ]
    fake.responder = _overdue_pages(pages)

    result = _call_tool("list_overdue_invoices", asOfDate="2026-10-13", limit=200)

    assert [call["params"]["page"] for call in fake.calls if "params" in call] == [
        1,
        2,
        3,
        4,
    ]
    assert result["total_count"] == 12
    assert result["returned"] == 12
    assert result["complete"] is True
    assert result["has_more"] is False
    assert result["total_amount_due_by_currency"] == {"AUD": "1200.00"}


def test_overdue_query_asks_xero_for_authorised_invoices_due_before_the_date(
    fake: FakeXero,
) -> None:
    fake.responder = _overdue_pages([[]])

    _call_tool("list_overdue_invoices", asOfDate="2026-10-13")

    params = fake.calls[-1]["params"]
    assert params["where"] == 'Type=="ACCREC" AND DueDate<DateTime(2026,10,13)'
    assert params["Statuses"] == "AUTHORISED"
    assert params["order"] == "DueDate"
    assert params["pageSize"] == 100


def test_overdue_min_days_moves_the_cutoff(fake: FakeXero) -> None:
    fake.responder = _overdue_pages([[]])

    _call_tool("list_overdue_invoices", asOfDate="2026-10-13", minDaysOverdue=30)

    assert fake.calls[-1]["params"]["where"].endswith("DueDate<DateTime(2026,9,14)")


def test_overdue_drops_invoices_that_are_not_actually_overdue(fake: FakeXero) -> None:
    fake.responder = _overdue_pages(
        [
            [
                _invoice("INV-OLD", "2026-09-01", 100),
                _invoice("INV-TODAY", "2026-10-13", 50),
                _invoice("INV-SETTLED", "2026-09-01", 0),
            ]
        ]
    )

    result = _call_tool("list_overdue_invoices", asOfDate="2026-10-13")

    assert [inv["invoice_number"] for inv in result["invoices"]] == ["INV-OLD"]
    assert result["invoices"][0]["days_overdue"] == 42


def test_overdue_falls_back_when_xero_rejects_the_due_date_filter(
    fake: FakeXero,
) -> None:
    pages = [
        [
            _invoice("INV-OLD", "2026-09-01", 100),
            _invoice("INV-NOT-DUE", "2026-11-01", 100),
        ]
    ]
    respond = _overdue_pages(pages)

    def reject_due_date(call: dict[str, Any]) -> MockResponse:
        if "DueDate" in (call["params"].get("where") or ""):
            return MockResponse({"Message": "Invalid filter"}, status_code=400)
        return respond(call)

    fake.responder = reject_due_date

    result = _call_tool("list_overdue_invoices", asOfDate="2026-10-13")

    wheres = [call["params"]["where"] for call in fake.calls if "params" in call]
    assert wheres == [
        'Type=="ACCREC" AND DueDate<DateTime(2026,10,13)',
        'Type=="ACCREC"',
    ]
    # Without the server-side clause, the not-yet-due invoice is dropped locally.
    assert [inv["invoice_number"] for inv in result["invoices"]] == ["INV-OLD"]


def test_overdue_does_not_swallow_other_errors(fake: FakeXero) -> None:
    fake.responder = lambda call: MockResponse({}, status_code=500, text="boom")

    with pytest.raises(RuntimeError, match="status 500"):
        xero.list_overdue_invoices(asOfDate="2026-10-13")

    assert len([call for call in fake.calls if "params" in call]) == 1


def test_overdue_ignores_invoices_of_the_other_type(fake: FakeXero) -> None:
    bill = _invoice("BILL-1", "2026-09-01", 100)
    bill["Type"] = "ACCPAY"
    fake.responder = _overdue_pages([[bill, _invoice("INV-1", "2026-09-01", 50)]])

    result = _call_tool("list_overdue_invoices", asOfDate="2026-10-13")

    assert [inv["invoice_number"] for inv in result["invoices"]] == ["INV-1"]


def test_overdue_sorts_oldest_first_and_summarises(fake: FakeXero) -> None:
    fake.responder = _overdue_pages(
        [
            [
                _invoice("INV-B", "2026-10-01", 200, contact="Beta", contact_id="b"),
                _invoice("INV-A", "2026-06-01", 300, contact="Acme", contact_id="a"),
                _invoice("INV-C", "2026-09-20", 50, contact="Acme", contact_id="a"),
                _invoice("INV-D", "2026-09-01", 25, currency="USD"),
            ]
        ]
    )

    result = _call_tool("list_overdue_invoices", asOfDate="2026-10-13")

    assert [inv["invoice_number"] for inv in result["invoices"]] == [
        "INV-A",
        "INV-D",
        "INV-C",
        "INV-B",
    ]
    assert result["total_amount_due_by_currency"] == {"AUD": "550.00", "USD": "25.00"}
    assert result["aging_buckets"]["90+"]["count"] == 1
    assert result["aging_buckets"]["31-60"]["count"] == 1
    assert result["aging_buckets"]["1-30"]["count"] == 2
    assert result["aging_buckets"]["61-90"]["count"] == 0
    top = result["by_contact"][0]
    assert top["contact_name"] == "Acme"
    assert top["invoice_count"] == 2
    assert top["amount_due_by_currency"] == {"AUD": "350.00"}
    assert top["oldest_due_date"] == "2026-06-01T00:00:00"


def test_overdue_window_resumes_with_next_skip(fake: FakeXero) -> None:
    pages = [[_invoice(f"INV-{i:02d}", f"2026-09-{i + 1:02d}", 10) for i in range(10)]]
    fake.responder = _overdue_pages(pages)

    first = _call_tool("list_overdue_invoices", asOfDate="2026-10-13", limit=4)
    second = _call_tool(
        "list_overdue_invoices",
        asOfDate="2026-10-13",
        limit=4,
        skip=first["next_skip"],
    )

    assert first["total_count"] == second["total_count"] == 10
    assert first["has_more"] is True and first["next_skip"] == 4
    assert [i["invoice_number"] for i in second["invoices"]] == [
        "INV-04",
        "INV-05",
        "INV-06",
        "INV-07",
    ]
    # The summary covers the whole list even when only a window is returned.
    assert second["total_amount_due_by_currency"] == {"AUD": "100.00"}


def test_overdue_trims_a_window_that_does_not_fit_and_still_resumes(
    fake: FakeXero, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(xero, "get_tool_max_output_length", lambda: 4000)
    pages = [[_invoice(f"INV-{i:03d}", "2026-09-01", 10) for i in range(60)]]
    fake.responder = _overdue_pages(pages)

    result = _call_tool("list_overdue_invoices", asOfDate="2026-10-13", limit=60)

    assert result["truncated"] is True
    assert result["returned"] < 60
    assert result["has_more"] is True
    assert result["next_skip"] == result["returned"]
    assert result["total_count"] == 60


def test_overdue_reports_when_it_stops_early(
    fake: FakeXero, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(xero, "OVERDUE_MAX_PAGES", 2)
    pages = [[_invoice(f"INV-{p}", "2026-09-01", 10)] for p in range(5)]
    fake.responder = _overdue_pages(pages)

    result = _call_tool("list_overdue_invoices", asOfDate="2026-10-13")

    assert result["complete"] is False
    assert result["total_count"] == 2
    assert "Stopped after 2 pages" in result["note"]


def test_overdue_without_page_metadata_stops_on_a_short_page(fake: FakeXero) -> None:
    fake.responder = lambda call: MockResponse(
        {"Invoices": [_invoice("INV-1", "2026-09-01", 10)]}
    )

    result = _call_tool("list_overdue_invoices", asOfDate="2026-10-13")

    assert result["total_count"] == 1
    assert len(fake.calls) == 2  # connections + one page


def test_overdue_defaults_to_today(
    fake: FakeXero, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(xero, "_today", lambda: date(2026, 10, 13))
    fake.responder = _overdue_pages([[]])

    result = _call_tool("list_overdue_invoices")

    assert result["as_of_date"] == "2026-10-13"


# --------------------------------------------------------------------------- #
# Dates and reports
# --------------------------------------------------------------------------- #


def test_xero_date_formats() -> None:
    assert xero._xero_date("/Date(1790000000000+0000)/") == "2026-09-21T14:13:20Z"
    assert xero._xero_date("2026-09-01T00:00:00") == "2026-09-01T00:00:00"
    assert xero._xero_date(None) is None
    assert (
        xero._doc_date({"DueDate": "/Date(1790000000000+0000)/"}, "DueDate")
        == "2026-09-21T14:13:20Z"
    )


def test_report_rows_are_flattened(fake: FakeXero) -> None:
    fake.responder = lambda call: MockResponse(
        {
            "Reports": [
                {
                    "ReportName": "Trial Balance",
                    "ReportDate": "13 October 2026",
                    "Rows": [
                        {
                            "RowType": "Header",
                            "Cells": [{"Value": "Account"}, {"Value": "Debit"}],
                        },
                        {
                            "RowType": "Section",
                            "Title": "Revenue",
                            "Rows": [
                                {
                                    "RowType": "Row",
                                    "Cells": [{"Value": "Sales"}, {"Value": "100"}],
                                }
                            ],
                        },
                    ],
                }
            ]
        }
    )

    result = _call_tool("list_trial_balance", date="2026-10-13")

    assert result["report_name"] == "Trial Balance"
    assert result["rows"][1] == {"row_type": "Section", "title": "Revenue"}
    assert result["rows"][2] == {
        "row_type": "Row",
        "cells": ["Sales", "100"],
        "section": "Revenue",
    }
    assert fake.calls[-1]["params"] == {"date": "2026-10-13"}


def test_aged_receivables_maps_parameters(fake: FakeXero) -> None:
    fake.responder = lambda call: MockResponse({"Reports": [{"Rows": []}]})

    xero.list_aged_receivables_by_contact(
        contactId=CONTACT_ID,
        reportDate="2026-10-13",
        invoicesFromDate="2026-01-01",
        invoicesToDate="2026-10-01",
    )

    call = fake.calls[-1]
    assert call["path"] == "Reports/AgedReceivablesByContact"
    assert call["params"] == {
        "contactId": CONTACT_ID,
        "date": "2026-10-13",
        "fromDate": "2026-01-01",
        "toDate": "2026-10-01",
    }


# --------------------------------------------------------------------------- #
# Writes
# --------------------------------------------------------------------------- #


LINE = xero.LineItem(
    description="Consulting",
    quantity=2,
    unitAmount=150,
    accountCode="200",
    taxType="OUTPUT",
)


def _created(key: str, record: dict[str, Any]) -> Callable[..., MockResponse]:
    def respond(call: dict[str, Any]) -> MockResponse:
        if call["path"] == "Organisation":
            return MockResponse({"Organisations": [{"ShortCode": "!abc12"}]})
        return MockResponse({key: [record]})

    return respond


def test_create_invoice_makes_a_draft_and_returns_a_link(
    fake: FakeXero, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(xero, "_today", lambda: date(2026, 10, 13))
    fake.responder = _created("Invoices", _invoice("INV-7", "2026-11-12", 300))

    result = xero.create_invoice(
        contactId=CONTACT_ID, lineItems=[LINE], reference="PO-1", date="2026-10-01"
    )

    put = next(call for call in fake.calls if call["method"] == "PUT")
    body = put["json"]["Invoices"][0]
    assert put["path"] == "Invoices"
    assert body["Status"] == "DRAFT"
    assert body["Type"] == "ACCREC"
    assert body["Reference"] == "PO-1"
    assert body["Date"] == "2026-10-01"
    assert body["DueDate"] == "2026-11-12"
    assert body["LineItems"][0] == {
        "Description": "Consulting",
        "Quantity": 2,
        "UnitAmount": 150,
        "AccountCode": "200",
        "TaxType": "OUTPUT",
    }
    assert "summarizeErrors" not in (put["params"] or {})
    payload = json.loads(result)
    assert payload["invoice"]["invoice_number"] == "INV-7"
    assert (
        payload["deep_link"] == "https://go.xero.com/app/!abc12/invoicing/view/id-INV-7"
    )


def test_create_bill_uses_the_reference_as_invoice_number(fake: FakeXero) -> None:
    fake.responder = _created("Invoices", _invoice("BILL-1", "2026-11-12", 300))

    xero.create_invoice(
        contactId=CONTACT_ID, lineItems=[LINE], type="ACCPAY", reference="BILL-1"
    )

    body = next(c for c in fake.calls if c["method"] == "PUT")["json"]["Invoices"][0]
    assert body["InvoiceNumber"] == "BILL-1"
    assert "Reference" not in body


def test_a_missing_deep_link_never_fails_a_successful_write(fake: FakeXero) -> None:
    def respond(call: dict[str, Any]) -> MockResponse:
        if call["path"] == "Organisation":
            return MockResponse({}, status_code=403)
        return MockResponse({"Contacts": [{"ContactID": CONTACT_ID, "Name": "Acme"}]})

    fake.responder = respond

    result = json.loads(xero.create_contact(name="Acme"))

    assert result["status"] == "success"
    assert result["deep_link"] is None


def test_update_invoice_refuses_an_authorised_invoice(fake: FakeXero) -> None:
    fake.responder = lambda call: MockResponse(
        {"Invoices": [_invoice("INV-1", "2026-09-01", 10, status="AUTHORISED")]}
    )

    with pytest.raises(ValueError, match="this one is AUTHORISED"):
        xero.update_invoice(invoiceId=INVOICE_ID, reference="new")

    assert all(call["method"] == "GET" for call in fake.calls)


def test_update_invoice_sends_only_the_changed_fields(fake: FakeXero) -> None:
    def respond(call: dict[str, Any]) -> MockResponse:
        if call["method"] == "GET" and call["path"].startswith("Invoices/"):
            return MockResponse(
                {"Invoices": [_invoice("INV-1", "2026-09-01", 10, status="DRAFT")]}
            )
        if call["path"] == "Organisation":
            return MockResponse({"Organisations": [{"ShortCode": "!abc12"}]})
        return MockResponse({"Invoices": [_invoice("INV-1", "2026-09-01", 10)]})

    fake.responder = respond

    xero.update_invoice(invoiceId=INVOICE_ID, reference="new", dueDate="2026-12-01")

    post = next(call for call in fake.calls if call["method"] == "POST")
    assert post["path"] == f"Invoices/{INVOICE_ID}"
    assert post["json"] == {
        "Invoices": [
            {"InvoiceID": INVOICE_ID, "Reference": "new", "DueDate": "2026-12-01"}
        ]
    }


def test_update_quote_keeps_the_contact_and_date_xero_requires(fake: FakeXero) -> None:
    existing = {
        "QuoteID": "q1",
        "Status": "DRAFT",
        "Contact": {"ContactID": CONTACT_ID},
        "DateString": "2026-10-01T00:00:00",
    }

    def respond(call: dict[str, Any]) -> MockResponse:
        if call["path"] == "Organisation":
            return MockResponse({"Organisations": [{"ShortCode": "!abc12"}]})
        return MockResponse({"Quotes": [existing]})

    fake.responder = respond

    xero.update_quote(quoteId=INVOICE_ID, title="Renewal")

    post = next(call for call in fake.calls if call["method"] == "POST")
    body = post["json"]["Quotes"][0]
    assert body["Contact"] == {"ContactID": CONTACT_ID}
    assert body["Date"] == "2026-10-01T00:00:00"
    assert body["Title"] == "Renewal"


def test_create_payment_requires_a_positive_amount(fake: FakeXero) -> None:
    with pytest.raises(ValueError, match="amount must be positive"):
        xero.create_payment(invoiceId=INVOICE_ID, accountId=CONTACT_ID, amount=0)

    assert fake.calls == []


def test_manual_journal_must_balance(fake: FakeXero) -> None:
    lines = [
        xero.ManualJournalLine(lineAmount=100, accountCode="200"),
        xero.ManualJournalLine(lineAmount=-90, accountCode="400"),
    ]

    with pytest.raises(ValueError, match="sum to 10"):
        xero.create_manual_journal(narration="Accrual", manualJournalLines=lines)
    with pytest.raises(ValueError, match="at least two"):
        xero.create_manual_journal(narration="Accrual", manualJournalLines=lines[:1])

    assert fake.calls == []


def test_balanced_manual_journal_is_created(fake: FakeXero) -> None:
    fake.responder = _created(
        "ManualJournals", {"ManualJournalID": "mj1", "Narration": "Accrual"}
    )
    lines = [
        xero.ManualJournalLine(lineAmount=0.1, accountCode="200"),
        xero.ManualJournalLine(lineAmount=0.2, accountCode="201"),
        xero.ManualJournalLine(lineAmount=-0.3, accountCode="400", description="x"),
    ]

    result = json.loads(
        xero.create_manual_journal(narration="Accrual", manualJournalLines=lines)
    )

    put = next(call for call in fake.calls if call["method"] == "PUT")
    assert put["json"]["ManualJournals"][0]["JournalLines"][2] == {
        "LineAmount": -0.3,
        "AccountCode": "400",
        "Description": "x",
    }
    assert result["manual_journal"]["manual_journal_id"] == "mj1"


def test_tracking_options_are_created_one_by_one(fake: FakeXero) -> None:
    fake.responder = lambda call: MockResponse(
        {"Options": [{"TrackingOptionID": "o", "Name": "North"}]}
    )

    result = json.loads(
        xero.create_tracking_options(
            trackingCategoryId=CONTACT_ID, optionNames=["North", "South"]
        )
    )

    puts = [call for call in fake.calls if call["method"] == "PUT"]
    assert [call["json"] for call in puts] == [{"Name": "North"}, {"Name": "South"}]
    assert result["total_count"] == 2
    with pytest.raises(ValueError, match="between 1 and 10"):
        xero.create_tracking_options(trackingCategoryId=CONTACT_ID, optionNames=[])


def test_update_tracking_options_rejects_a_foreign_option(fake: FakeXero) -> None:
    fake.responder = lambda call: MockResponse(
        {"TrackingCategories": [{"Options": [{"TrackingOptionID": "known"}]}]}
    )

    with pytest.raises(ValueError, match="is not in tracking category"):
        xero.update_tracking_options(
            trackingCategoryId=CONTACT_ID,
            options=[xero.TrackingOptionUpdate(trackingOptionId="other", name="x")],
        )


def test_ids_cannot_escape_their_url_segment(fake: FakeXero) -> None:
    fake.responder = lambda call: MockResponse({}, status_code=404)

    with pytest.raises(RuntimeError, match="status 404"):
        xero.update_invoice(invoiceId="../Contacts", reference="x")

    # The id stays one percent-encoded path segment; it never reaches /Contacts.
    assert [call["path"] for call in fake.calls if "path" in call] == [
        "connections",
        "Invoices/..%2FContacts",
    ]
    with pytest.raises(ValueError):
        xero.update_invoice(invoiceId="..", reference="x")


# --------------------------------------------------------------------------- #
# Tool surface
# --------------------------------------------------------------------------- #

# The accounting tools the third-party @xeroapi/xero-mcp-server exposed, which
# Toby's prompts and memories refer to by name.
PRESERVED_TOOLS = {
    "create_bank_transaction",
    "create_contact",
    "create_credit_note",
    "create_invoice",
    "create_item",
    "create_manual_journal",
    "create_payment",
    "create_quote",
    "create_tracking_category",
    "create_tracking_options",
    "list_accounts",
    "list_aged_payables_by_contact",
    "list_aged_receivables_by_contact",
    "list_bank_transactions",
    "list_contact_groups",
    "list_contacts",
    "list_credit_notes",
    "list_invoices",
    "list_items",
    "list_manual_journals",
    "list_organisation_details",
    "list_payments",
    "list_profit_and_loss",
    "list_quotes",
    "list_report_balance_sheet",
    "list_tax_rates",
    "list_tracking_categories",
    "list_trial_balance",
    "update_bank_transaction",
    "update_contact",
    "update_credit_note",
    "update_invoice",
    "update_item",
    "update_manual_journal",
    "update_quote",
    "update_tracking_category",
    "update_tracking_options",
}


def _registered_tools() -> dict[str, Any]:
    tools = asyncio.run(xero.mcp.list_tools())
    return {tool.name: tool for tool in tools}


def test_every_preserved_tool_is_registered_plus_the_overdue_tool() -> None:
    assert set(_registered_tools()) == PRESERVED_TOOLS | {"list_overdue_invoices"}


def test_read_tools_are_declared_read_only_and_writes_are_not() -> None:
    for name, tool in _registered_tools().items():
        read_only = tool.annotations is not None and tool.annotations.readOnlyHint
        assert bool(read_only) == name.startswith("list_"), name


def test_paged_tools_no_longer_require_a_page_argument() -> None:
    tools = _registered_tools()

    for name in ("list_invoices", "list_credit_notes", "list_bank_transactions"):
        assert "page" not in tools[name].inputSchema.get("required", []), name
