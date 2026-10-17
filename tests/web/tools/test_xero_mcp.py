import asyncio
import json
from datetime import date, datetime, timezone
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
    issued: str = "2026-08-01",
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
        "DateString": f"{issued}T00:00:00",
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


@pytest.mark.parametrize(
    ("connections", "message"),
    [([], "No Xero organisation"), ([{}], "without a tenant id")],
)
def test_tenant_lookup_rejects_unusable_connections(
    fake: FakeXero,
    monkeypatch: pytest.MonkeyPatch,
    connections: list[dict[str, Any]],
    message: str,
) -> None:
    def get(url: str, **kwargs: Any) -> MockResponse:
        fake.calls.append({"method": "GET", "url": url, "path": "connections"})
        return MockResponse(connections)

    monkeypatch.setattr(xero.requests, "get", get)

    with pytest.raises(RuntimeError, match=message):
        xero.list_accounts()

    assert [call["path"] for call in fake.calls] == ["connections"]


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


def test_unauthorized_request_reports_expired_connection_not_scope(
    fake: FakeXero,
) -> None:
    fake.responder = lambda call: MockResponse(
        {"Title": "Token expired"}, status_code=401
    )

    with pytest.raises(RuntimeError) as error:
        xero.list_aged_receivables_by_contact(contactId=CONTACT_ID)

    message = str(error.value)
    assert "status 401" in message
    assert "Token expired" in message
    assert "scope" not in message
    assert "reconnect" in message.lower()


def test_forbidden_request_names_the_missing_scope(fake: FakeXero) -> None:
    fake.responder = lambda call: MockResponse({"Title": "Forbidden"}, status_code=403)

    with pytest.raises(RuntimeError) as error:
        xero.list_aged_receivables_by_contact(contactId=CONTACT_ID)

    message = str(error.value)
    assert "status 403" in message
    assert "accounting.reports.aged.read" in message
    assert "Forbidden" in message
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


def test_list_invoices_exposes_expected_and_planned_payment_dates(
    fake: FakeXero,
) -> None:
    receivable = _invoice("INV-AR", "2026-10-20", 10)
    receivable["ExpectedPaymentDate"] = "/Date(1792454400000+0000)/"
    payable = _invoice("INV-AP", "2026-10-21", 20)
    payable["Type"] = "ACCPAY"
    payable["PlannedPaymentDate"] = "2026-10-22T00:00:00"
    fake.responder = lambda call: MockResponse({"Invoices": [receivable, payable]})

    result = _call_tool("list_invoices")

    assert result["invoices"][0]["expected_payment_date"] == "2026-10-20T00:00:00Z"
    assert result["invoices"][1]["planned_payment_date"] == "2026-10-22T00:00:00"
    assert "planned_payment_date" not in result["invoices"][0]
    assert "expected_payment_date" not in result["invoices"][1]


def test_list_invoice_history_returns_xero_notes_oldest_first(fake: FakeXero) -> None:
    fake.responder = lambda call: MockResponse(
        {
            "HistoryRecords": [
                {
                    "Details": "Customer promised payment on Friday",
                    "Changes": "Note",
                    "User": "Hannah Example",
                    "DateUTC": "/Date(1792458000000+0000)/",
                },
                {
                    "Details": "Invoice authorised",
                    "Changes": "Approved",
                    "User": "Toby Example",
                    "DateUTCString": "2026-10-19T23:00:00Z",
                },
                {
                    "Details": "Imported note without timestamp",
                    "Changes": "Note",
                    "User": "Migration",
                },
            ]
        }
    )

    result = _call_tool("list_invoice_history", invoiceId=INVOICE_ID)

    assert result == {
        "status": "success",
        "history": [
            {
                "details": "Invoice authorised",
                "changes": "Approved",
                "user": "Toby Example",
                "date_utc": "2026-10-19T23:00:00Z",
            },
            {
                "details": "Customer promised payment on Friday",
                "changes": "Note",
                "user": "Hannah Example",
                "date_utc": "2026-10-20T01:00:00Z",
            },
            {
                "details": "Imported note without timestamp",
                "changes": "Note",
                "user": "Migration",
            },
        ],
        "total_count": 3,
        "complete": True,
        "skip": 0,
        "returned": 3,
        "has_more": False,
    }
    assert fake.calls[-1]["path"] == f"Invoices/{INVOICE_ID}/History"
    assert fake.calls[-1]["method"] == "GET"

    first = _call_tool("list_invoice_history", invoiceId=INVOICE_ID, limit=1)
    assert first["returned"] == 1
    assert first["has_more"] is True
    assert first["next_skip"] == 1
    assert first["history"][0]["details"] == "Invoice authorised"

    second = _call_tool(
        "list_invoice_history",
        invoiceId=INVOICE_ID,
        skip=first["next_skip"],
        limit=1,
    )
    assert second["skip"] == 1
    assert second["returned"] == 1
    assert second["has_more"] is True
    assert second["next_skip"] == 2
    assert second["history"][0]["details"] == ("Customer promised payment on Friday")

    last = _call_tool(
        "list_invoice_history",
        invoiceId=INVOICE_ID,
        skip=second["next_skip"],
        limit=1,
    )
    assert last["skip"] == 2
    assert last["returned"] == 1
    assert last["has_more"] is False
    assert "next_skip" not in last
    assert last["history"][0]["details"] == "Imported note without timestamp"


def test_list_invoice_history_handles_empty_history_and_clamps_window(
    fake: FakeXero,
) -> None:
    fake.responder = lambda call: MockResponse({})

    result = _call_tool(
        "list_invoice_history",
        invoiceId=INVOICE_ID,
        skip=-10,
        limit=0,
    )

    assert result == {
        "status": "success",
        "total_count": 0,
        "complete": True,
        "skip": 0,
        "returned": 0,
        "has_more": False,
        "history": [],
    }


def test_list_invoice_history_rejects_unsafe_invoice_id(fake: FakeXero) -> None:
    with pytest.raises(ValueError, match="invoiceId"):
        xero.list_invoice_history(invoiceId="../Contacts")

    assert fake.calls == []


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


def _paid_invoice(
    number: str,
    paid_on: str,
    *,
    total: float = 10,
    currency: str = "AUD",
    contact_id: str = CONTACT_ID,
) -> dict[str, Any]:
    record = _invoice(
        number,
        "2026-10-01",
        0,
        issued="2026-09-01",
        currency=currency,
        contact_id=contact_id,
        status="PAID",
    )
    moment = datetime.fromisoformat(paid_on).replace(tzinfo=timezone.utc)
    milliseconds = int(moment.timestamp() * 1000)
    record["FullyPaidOnDate"] = f"/Date({milliseconds}+0000)/"
    record["Total"] = total
    record["AmountPaid"] = total
    return record


def _paid_pages(pages: list[list[dict[str, Any]]]) -> Callable[..., MockResponse]:
    def respond(call: dict[str, Any]) -> MockResponse:
        page = call["params"]["page"]
        return MockResponse(
            {
                "pagination": {
                    "page": page,
                    "pageSize": xero.XERO_BULK_PAGE_SIZE,
                    "pageCount": len(pages),
                    "itemCount": sum(len(items) for items in pages),
                },
                "Invoices": pages[page - 1],
            }
        )

    return respond


def test_list_paid_invoices_reads_later_pages_after_mixed_dates(
    fake: FakeXero,
) -> None:
    # Production regression: the first page contained ten 13 Oct settlements;
    # the next page mixed five more with other dates. Updated ordering is not a
    # valid boundary for FullyPaidOnDate.
    first = [_paid_invoice(f"INV-{number}", "2026-10-13") for number in range(1, 11)]
    second = [
        *[_paid_invoice(f"INV-{number}", "2026-10-13") for number in range(11, 16)],
        _paid_invoice("INV-OLDER", "2026-10-12"),
    ]
    fake.responder = _paid_pages([first, second])

    result = _call_tool("list_paid_invoices", paidOnDate="2026-10-13")

    assert result["complete"] is True
    assert result["date_basis"] == "FullyPaidOnDate"
    assert result["total_count"] == 15
    assert result["returned"] == 15
    assert result["has_more"] is False
    assert {row["invoice_number"] for row in result["invoices"]} == {
        f"INV-{number}" for number in range(1, 16)
    }
    assert result["total_invoice_amount_by_currency"] == {"AUD": "150.00"}
    api_calls = [call for call in fake.calls if call.get("path") == "Invoices"]
    assert [call["params"]["page"] for call in api_calls] == [1, 2]
    assert api_calls[0]["params"]["Statuses"] == "PAID"
    assert api_calls[0]["params"]["where"] == 'Type=="ACCREC"'
    assert api_calls[0]["params"]["summaryOnly"] == "true"


def test_list_paid_invoices_windows_totals_and_contact_filter(fake: FakeXero) -> None:
    fake.responder = _paid_pages(
        [
            [
                _paid_invoice("INV-A", "2026-10-13", total=12.5),
                _paid_invoice(
                    "INV-B",
                    "2026-10-13",
                    total=20.25,
                    currency="USD",
                ),
            ]
        ]
    )

    first = _call_tool(
        "list_paid_invoices",
        paidOnDate="2026-10-13",
        contactIds=[CONTACT_ID],
        limit=1,
    )

    assert first["total_count"] == 2
    assert first["total_invoice_amount_by_currency"] == {
        "AUD": "12.50",
        "USD": "20.25",
    }
    assert first["returned"] == 1
    assert first["has_more"] is True
    assert first["next_skip"] == 1
    assert first["invoices"][0]["invoice_number"] == "INV-A"
    assert fake.calls[-1]["params"]["ContactIDs"] == CONTACT_ID

    second = _call_tool(
        "list_paid_invoices",
        paidOnDate="2026-10-13",
        contactIds=[CONTACT_ID],
        skip=first["next_skip"],
        limit=1,
    )

    assert second["returned"] == 1
    assert second["has_more"] is False
    assert second["invoices"][0]["invoice_number"] == "INV-B"


def test_list_paid_invoices_rejects_paid_record_without_settlement_date(
    fake: FakeXero,
) -> None:
    missing_date = _paid_invoice("INV-MISSING", "2026-10-13")
    missing_date.pop("FullyPaidOnDate")
    fake.responder = _paid_pages([[missing_date]])

    with pytest.raises(RuntimeError, match="without FullyPaidOnDate"):
        xero.list_paid_invoices(paidOnDate="2026-10-13")


def test_list_paid_invoices_rejects_partial_scan_without_pagination(
    fake: FakeXero,
) -> None:
    fake.responder = lambda call: MockResponse(
        {"Invoices": [_paid_invoice("INV-1", "2026-10-13")]}
    )

    with pytest.raises(
        RuntimeError, match="refusing to report a partial settlement-date list"
    ):
        xero.list_paid_invoices(paidOnDate="2026-10-13")


def test_list_paid_invoices_rejects_partial_scan_at_safety_limit(
    fake: FakeXero, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(xero, "PAID_INVOICE_MAX_PAGES", 1)
    fake.responder = _paid_pages(
        [
            [_paid_invoice("INV-1", "2026-10-13")],
            [_paid_invoice("INV-2", "2026-10-13")],
        ]
    )

    with pytest.raises(RuntimeError, match="paid-invoice scan reached the safety"):
        xero.list_paid_invoices(paidOnDate="2026-10-13")


def test_list_paid_invoices_rejects_invalid_date_before_calling_xero(
    fake: FakeXero,
) -> None:
    with pytest.raises(ValueError, match="paidOnDate must be YYYY-MM-DD"):
        xero.list_paid_invoices(paidOnDate="13/10/2026")

    assert fake.calls == []


def test_quotes_use_xero_pagination_metadata(fake: FakeXero) -> None:
    fake.responder = lambda call: MockResponse(
        {
            "pagination": {"page": 1, "pageSize": 20, "pageCount": 3, "itemCount": 45},
            "Quotes": [{"QuoteID": str(i)} for i in range(20)],
        }
    )

    result = _call_tool("list_quotes", page=1)

    assert result["total_count"] == 45
    assert result["has_more"] is True
    assert result["next_page"] == 2
    assert fake.calls[-1]["params"]["pageSize"] == 20


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
                    "pageSize": xero.XERO_BULK_PAGE_SIZE,
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
    assert params["where"] == (
        'Type=="ACCREC" AND Date<=DateTime(2026,10,13) AND DueDate<DateTime(2026,10,13)'
    )
    assert params["Statuses"] == "AUTHORISED"
    assert "order" not in params
    assert params["pageSize"] == xero.XERO_BULK_PAGE_SIZE
    assert params["summaryOnly"] == "true"


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
                # Defend locally even if Xero ignores Statuses=AUTHORISED or
                # returns a stale Paid row with an inconsistent amount due.
                _invoice("INV-PAID", "2026-09-01", 75, status="PAID"),
                _invoice(
                    "INV-ISSUED-LATER",
                    "2026-09-01",
                    80,
                    issued="2026-10-14",
                ),
            ]
        ]
    )

    result = _call_tool("list_overdue_invoices", asOfDate="2026-10-13")

    assert [inv["invoice_number"] for inv in result["invoices"]] == ["INV-OLD"]
    assert result["invoices"][0]["days_overdue"] == 42
    assert result["balance_basis"] == "current_open_invoice_state"
    assert result["historical_payments_reversed"] is False
    assert result["excludes"] == ["credit_notes", "overpayments", "prepayments"]


def test_overdue_deduplicates_invoice_ids_across_pages(fake: FakeXero) -> None:
    repeated = _invoice("INV-1", "2026-09-01", 100)
    fake.responder = _overdue_pages([[repeated], [repeated]])

    result = _call_tool("list_overdue_invoices", asOfDate="2026-10-13")

    assert result["total_count"] == 1
    assert [invoice["invoice_number"] for invoice in result["invoices"]] == ["INV-1"]


def test_overdue_does_not_retry_a_bad_request_without_filters(fake: FakeXero) -> None:
    fake.responder = lambda call: MockResponse(
        {"Message": "Invalid filter"}, status_code=400
    )

    with pytest.raises(xero.XeroAPIError, match="Invalid filter") as error:
        xero.list_overdue_invoices(asOfDate="2026-10-13")

    assert error.value.status_code == 400
    assert len([call for call in fake.calls if "params" in call]) == 1


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
    assert result["ageing_buckets"]["90+"]["count"] == 1
    assert result["ageing_buckets"]["31-60"]["count"] == 1
    assert result["ageing_buckets"]["1-30"]["count"] == 2
    assert result["ageing_buckets"]["61-90"]["count"] == 0
    top = result["by_contact"][0]
    assert top["contact_name"] == "Acme"
    assert top["currency"] == "AUD"
    assert top["invoice_count"] == 2
    assert top["amount_due"] == "350.00"
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


def test_overdue_fails_instead_of_reporting_partial_totals(
    fake: FakeXero, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(xero, "OPEN_INVOICE_MAX_PAGES", 2)
    pages = [[_invoice(f"INV-{p}", "2026-09-01", 10)] for p in range(5)]
    fake.responder = _overdue_pages(pages)

    with pytest.raises(RuntimeError, match="safety limit of 2 pages"):
        xero.list_overdue_invoices(asOfDate="2026-10-13")


def test_overdue_without_page_metadata_fails_closed(fake: FakeXero) -> None:
    fake.responder = lambda call: MockResponse(
        {"Invoices": [_invoice("INV-1", "2026-09-01", 10)]}
    )

    with pytest.raises(RuntimeError, match="no pagination metadata"):
        xero.list_overdue_invoices(asOfDate="2026-10-13")


def test_overdue_defaults_to_today(
    fake: FakeXero, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(xero, "_organisation_today", lambda: date(2026, 10, 13))
    fake.responder = _overdue_pages([[]])

    result = _call_tool("list_overdue_invoices")

    assert result["as_of_date"] == "2026-10-13"


# --------------------------------------------------------------------------- #
# Receivables ageing
# --------------------------------------------------------------------------- #


RECEIVABLES_SPREADSHEET_COLUMNS = [
    {"header": "Company Name", "field": "contact_name"},
    {"header": "Currency", "field": "currency"},
    {"header": "Invoice Count", "field": "invoice_count"},
    {"header": "Current", "field": "buckets.Current"},
    {"header": "1-15 Days", "field": "buckets.1-15"},
    {"header": "16-30 Days", "field": "buckets.16-30"},
    {"header": "31-45 Days", "field": "buckets.31-45"},
    {"header": "46-60 Days", "field": "buckets.46-60"},
    {"header": "61-75 Days", "field": "buckets.61-75"},
    {"header": "76+ Days", "field": "buckets.76+"},
    {"header": "Total Due", "field": "total_due"},
]


def _resolve_field(row: dict[str, Any], field: str) -> Any:
    value: Any = row
    for part in field.split("."):
        value = value[part]
    return value


@pytest.mark.parametrize(
    ("days", "bucket"),
    [
        (-3, "Current"),
        (0, "Current"),
        (1, "1-15"),
        (15, "1-15"),
        (16, "16-30"),
        (30, "16-30"),
        (31, "31-45"),
        (45, "31-45"),
        (46, "46-60"),
        (60, "46-60"),
        (61, "61-75"),
        (75, "61-75"),
        (76, "76+"),
        (542, "76+"),
    ],
)
def test_receivable_bucket_edges(days: int, bucket: str) -> None:
    assert xero._receivable_bucket(days) == bucket


def test_ageing_has_one_named_row_per_contact_and_currency_with_every_bucket(
    fake: FakeXero,
) -> None:
    fake.responder = _overdue_pages(
        [
            [
                _invoice("INV-1", "2026-11-01", 100, contact="Acme", contact_id="a"),
                _invoice("INV-2", "2026-10-13", 40, contact="Acme", contact_id="a"),
                _invoice("INV-3", "2026-10-01", 10, contact="Acme", contact_id="a"),
                _invoice("INV-4", "2025-04-20", 1617, contact="Heebie", contact_id="h"),
                _invoice("INV-5", "2026-10-05", 5, contact="Beta", contact_id="b"),
                _invoice("INV-6", "2026-10-05", 0, contact="Beta", contact_id="b"),
                _invoice(
                    "INV-PAID",
                    "2026-09-01",
                    999,
                    contact="Beta",
                    contact_id="b",
                    status="PAID",
                ),
                _invoice(
                    "INV-7",
                    "2026-09-01",
                    7,
                    contact="Acme",
                    contact_id="a",
                    currency="USD",
                ),
            ]
        ]
    )

    result = _call_tool("list_receivables_ageing", asOfDate="2026-10-13")

    assert result["total_invoice_count"] == 6
    assert result["bucket_names"] == list(xero.RECEIVABLE_BUCKETS)
    assert result["spreadsheet_export"] == {
        "row_source": "rows",
        "columns": RECEIVABLES_SPREADSHEET_COLUMNS,
        "retrieve_all_pages": True,
    }
    rows = result["rows"]
    for row in rows:
        assert all(
            _resolve_field(row, column["field"]) is not None
            for column in result["spreadsheet_export"]["columns"]
        )
    assert [(row["contact_name"], row["currency"]) for row in rows] == [
        ("Heebie", "AUD"),
        ("Acme", "AUD"),
        ("Beta", "AUD"),
        ("Acme", "USD"),
    ]
    heebie = rows[0]
    assert heebie["buckets"]["76+"] == "1617.00"
    assert heebie["total_due"] == "1617.00"
    assert set(heebie["buckets"]) == set(xero.RECEIVABLE_BUCKETS)
    acme_aud = rows[1]
    assert acme_aud["currency"] == "AUD"
    assert acme_aud["buckets"]["Current"] == "140.00"
    assert acme_aud["buckets"]["1-15"] == "10.00"
    assert acme_aud["invoice_count"] == 3
    assert rows[2]["total_due"] == "5.00"
    aud = result["totals_by_currency"]["AUD"]
    assert aud["total_due"] == "1772.00"
    assert aud["buckets"]["Current"] == "140.00"
    assert aud["invoice_count"] == 5
    assert result["totals_by_currency"]["USD"]["buckets"]["31-45"] == "7.00"


def test_ageing_reads_every_page_and_does_not_narrow_by_due_date(
    fake: FakeXero,
) -> None:
    pages = [[_invoice(f"INV-{p}", "2026-12-01", 10)] for p in range(3)]
    fake.responder = _overdue_pages(pages)

    result = _call_tool("list_receivables_ageing", asOfDate="2026-10-13")

    sent = [call["params"] for call in fake.calls if "params" in call]
    assert [params["page"] for params in sent] == [1, 2, 3]
    assert sent[0]["where"] == ('Type=="ACCREC" AND Date<=DateTime(2026,10,13)')
    assert sent[0]["Statuses"] == "AUTHORISED"
    assert result["total_invoice_count"] == 3
    assert result["totals_by_currency"]["AUD"]["buckets"]["Current"] == "30.00"


def test_ageing_rows_window_resumes_with_next_skip(fake: FakeXero) -> None:
    invoices = [
        _invoice(f"INV-{i}", "2026-09-01", 100 - i, contact=f"C{i}", contact_id=f"c{i}")
        for i in range(6)
    ]
    fake.responder = _overdue_pages([invoices])

    first = _call_tool("list_receivables_ageing", asOfDate="2026-10-13", limit=4)
    second = _call_tool(
        "list_receivables_ageing",
        asOfDate="2026-10-13",
        limit=4,
        skip=first["next_skip"],
    )

    assert first["has_more"] is True and first["next_skip"] == 4
    assert first["rows_complete"] is False
    assert first["spreadsheet_export"]["row_source"] == "rows"
    assert [r["contact_name"] for r in second["rows"]] == ["C4", "C5"]
    assert second["has_more"] is False
    assert second["rows_complete"] is True
    assert second["spreadsheet_export"]["columns"][0] == {
        "header": "Company Name",
        "field": "contact_name",
    }
    assert first["totals_by_currency"] == second["totals_by_currency"]


def test_ageing_trims_rows_and_preserves_resume_cursor(
    fake: FakeXero, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(xero, "get_tool_max_output_length", lambda: 3500)
    invoices = [
        _invoice(f"INV-{i:03d}", "2026-09-01", 10, contact=f"C{i}", contact_id=f"c{i}")
        for i in range(60)
    ]
    fake.responder = _overdue_pages([invoices])

    result = _call_tool("list_receivables_ageing", asOfDate="2026-10-13", limit=60)

    assert result["truncated"] is True
    assert result["returned"] < 60
    assert result["has_more"] is True
    assert result["next_skip"] == result["returned"]
    assert result["total_rows"] == 60
    assert result["rows_complete"] is False
    assert result["spreadsheet_export"]["retrieve_all_pages"] is True


def test_ageing_export_contract_survives_an_empty_result(fake: FakeXero) -> None:
    fake.responder = _overdue_pages([[]])

    result = _call_tool("list_receivables_ageing", asOfDate="2026-10-13")

    assert result["rows"] == []
    assert result["rows_complete"] is True
    assert result["spreadsheet_export"] == {
        "row_source": "rows",
        "columns": RECEIVABLES_SPREADSHEET_COLUMNS,
        "retrieve_all_pages": True,
    }


def test_ageing_export_keeps_company_column_for_an_unnamed_contact(
    fake: FakeXero,
) -> None:
    fake.responder = _overdue_pages(
        [[_invoice("INV-1", "2026-09-01", 10, contact="", contact_id="contact-1")]]
    )

    result = _call_tool("list_receivables_ageing", asOfDate="2026-10-13")

    assert result["rows"][0]["contact_name"] == "contact-1"
    assert _resolve_field(result["rows"][0], "contact_name") == "contact-1"


def test_ageing_fails_instead_of_reporting_partial_totals(
    fake: FakeXero, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(xero, "OPEN_INVOICE_MAX_PAGES", 2)
    fake.responder = _overdue_pages(
        [[_invoice(f"INV-{page}", "2026-09-01", 10)] for page in range(5)]
    )

    with pytest.raises(RuntimeError, match="safety limit of 2 pages"):
        xero.list_receivables_ageing(asOfDate="2026-10-13")


def test_ageing_supports_payables(fake: FakeXero) -> None:
    bill = _invoice("BILL-1", "2026-09-01", 50)
    bill["Type"] = "ACCPAY"
    fake.responder = _overdue_pages([[bill, _invoice("INV-1", "2026-09-01", 99)]])

    result = _call_tool("list_receivables_ageing", asOfDate="2026-10-13", type="ACCPAY")

    assert result["invoice_type"] == "ACCPAY"
    assert result["total_invoice_count"] == 1
    assert fake.calls[-1]["params"]["where"] == (
        'Type=="ACCPAY" AND Date<=DateTime(2026,10,13)'
    )


# --------------------------------------------------------------------------- #
# Dates and reports
# --------------------------------------------------------------------------- #


def test_organisation_today_uses_xero_accounting_timezone(
    fake: FakeXero, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            moment = datetime(2026, 10, 13, 13, 30, tzinfo=timezone.utc)
            return moment.astimezone(tz) if tz else moment.replace(tzinfo=None)

    monkeypatch.setattr(xero, "datetime", FixedDateTime)
    fake.responder = lambda call: MockResponse(
        {"Organisations": [{"Timezone": "NEWZEALANDSTANDARDTIME"}]}
    )

    assert xero._organisation_today() == date(2026, 10, 14)


@pytest.mark.parametrize(
    ("xero_zone", "iana_zone"),
    [
        ("AUSCENTRALSTANDARDTIME", "Australia/Darwin"),
        ("CENAUSTRALIASTANDARDTIME", "Australia/Adelaide"),
        ("EAUSTRALIASTANDARDTIME", "Australia/Brisbane"),
        ("WAUSTRALIASTANDARDTIME", "Australia/Perth"),
        ("CHATHAMISLANDSSTANDARDTIME", "Pacific/Chatham"),
    ],
)
def test_xero_au_nz_timezone_mapping(xero_zone: str, iana_zone: str) -> None:
    assert xero._XERO_TIME_ZONES[xero_zone] == iana_zone


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


def test_aged_receivables_rejects_placeholder_contact_before_request(
    fake: FakeXero,
) -> None:
    with pytest.raises(ValueError, match="contactId must be a Xero GUID"):
        xero.list_aged_receivables_by_contact(contactId="PLACEHOLDER")

    assert fake.calls == []


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


def test_create_payment_returns_created_payment(fake: FakeXero) -> None:
    fake.responder = lambda call: MockResponse(
        {
            "Payments": [
                {
                    "PaymentID": "payment-1",
                    "Invoice": {"InvoiceID": INVOICE_ID, "InvoiceNumber": "INV-1"},
                    "Account": {"AccountID": CONTACT_ID, "Code": "090"},
                    "Amount": 125,
                    "DateString": "2026-10-13T00:00:00",
                    "Reference": "STRIPE-1",
                }
            ]
        }
    )

    result = json.loads(
        xero.create_payment(
            invoiceId=INVOICE_ID,
            accountId=CONTACT_ID,
            amount=125,
            date="2026-10-13",
            reference="STRIPE-1",
        )
    )

    assert result["payment"]["payment_id"] == "payment-1"
    post = fake.calls[-1]
    assert post["method"] == "POST"
    assert post["path"] == "Payments"
    assert post["json"]["Amount"] == 125


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
# Existing prompts and memories refer to these tools by name.
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


def test_every_preserved_tool_is_registered_plus_the_new_ones() -> None:
    assert set(_registered_tools()) == PRESERVED_TOOLS | {
        "list_paid_invoices",
        "list_invoice_history",
        "list_overdue_invoices",
        "list_receivables_ageing",
    }


def test_read_tools_are_declared_read_only_and_writes_are_not() -> None:
    for name, tool in _registered_tools().items():
        read_only = tool.annotations is not None and tool.annotations.readOnlyHint
        assert bool(read_only) == name.startswith("list_"), name


def test_paged_tools_no_longer_require_a_page_argument() -> None:
    tools = _registered_tools()

    for name in ("list_invoices", "list_credit_notes", "list_bank_transactions"):
        assert "page" not in tools[name].inputSchema.get("required", []), name


def test_receivables_ageing_description_requires_named_spreadsheet_rows() -> None:
    description = " ".join(
        _registered_tools()["list_receivables_ageing"].description.split()
    )

    assert "follow spreadsheet_export" in description
    assert "include Company Name from contact_name as the first column" in description
    assert "do not substitute it for the contact rows" in description
