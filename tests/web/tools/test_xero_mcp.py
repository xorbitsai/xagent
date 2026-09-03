import json
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import xero


class MockResponse:
    def __init__(
        self,
        json_data=None,
        text: str = "",
        status_code: int = 200,
        url: str = "",
        json_raises: bool = False,
    ):
        self._json_data = json_data if json_data is not None else {}
        self._json_raises = json_raises
        self.text = text or (json.dumps(self._json_data) if json_data else "")
        self.status_code = status_code
        self.content = self.text.encode()
        self.url = url

    def json(self):
        if self._json_raises:
            raise json.JSONDecodeError("Expecting value", self.text, 0)
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} Client Error for url: {self.url}", response=self
            )


_GUID = "12345678-1234-1234-1234-123456789012"


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("XERO_ACCESS_TOKEN", "access-token")


def test_headers_require_access_token(monkeypatch):
    monkeypatch.delenv("XERO_ACCESS_TOKEN")

    with pytest.raises(ValueError, match="XERO_ACCESS_TOKEN"):
        xero._headers()


def test_headers_include_bearer_token_without_tenant():
    headers = xero._headers()

    assert headers["Authorization"] == "Bearer access-token"
    assert "Xero-tenant-id" not in headers


def test_headers_include_tenant_id_when_provided():
    headers = xero._headers("tenant-123")

    assert headers["Xero-tenant-id"] == "tenant-123"


@pytest.mark.parametrize("value", ["", "   ", None])
def test_require_non_blank_rejects_empty_values(value):
    with pytest.raises(ValueError, match="field"):
        xero._require_non_blank(value, "field")


def test_validate_guid_accepts_well_formed_guid():
    assert xero._validate_guid(_GUID, "contact_id") == _GUID


@pytest.mark.parametrize("value", ["not-a-guid", "c1", _GUID + "x", '"; DROP TABLE'])
def test_validate_guid_rejects_malformed_values(value):
    with pytest.raises(ValueError, match="contact_id"):
        xero._validate_guid(value, "contact_id")


def test_reject_quote_passes_through_clean_value():
    assert xero._reject_quote("BANK", "account_type") == "BANK"


def test_reject_quote_rejects_embedded_quote():
    with pytest.raises(ValueError, match="account_type"):
        xero._reject_quote('BANK" || Type!="', "account_type")


def test_extract_error_detail_returns_plain_message():
    response = MockResponse(json_data={"Message": "Invalid contact"})

    assert xero._extract_error_detail(response) == "Invalid contact"


def test_extract_error_detail_folds_in_validation_errors():
    response = MockResponse(
        json_data={
            "Message": "A validation exception occurred",
            "Elements": [{"ValidationErrors": [{"Message": "Name must not be blank"}]}],
        }
    )

    assert (
        xero._extract_error_detail(response)
        == "A validation exception occurred: Name must not be blank"
    )


def test_extract_error_detail_returns_none_for_non_json_body():
    response = MockResponse(status_code=500, text="not json", json_raises=True)

    assert xero._extract_error_detail(response) is None


def test_build_line_items_requires_description():
    with pytest.raises(ValueError, match="description"):
        xero._build_line_items([{"quantity": 1}])


def test_build_line_items_translates_snake_case_to_pascal_case():
    result = xero._build_line_items(
        [
            {
                "description": "Consulting",
                "quantity": 2,
                "unit_amount": 150.0,
                "account_code": "200",
            }
        ]
    )

    assert result == [
        {
            "Description": "Consulting",
            "Quantity": 2,
            "UnitAmount": 150.0,
            "AccountCode": "200",
        }
    ]


def test_build_line_items_allows_description_only():
    result = xero._build_line_items([{"description": "Note line"}])

    assert result == [{"Description": "Note line"}]


def test_build_line_items_allows_zero_quantity_and_amount():
    result = xero._build_line_items(
        [{"description": "Free sample", "quantity": 0, "unit_amount": 0}]
    )

    assert result == [{"Description": "Free sample", "Quantity": 0, "UnitAmount": 0}]


def test_build_line_items_rejects_non_numeric_quantity():
    with pytest.raises(ValueError, match="quantity"):
        xero._build_line_items([{"description": "x", "quantity": "two"}])


def test_build_line_items_rejects_non_numeric_unit_amount():
    with pytest.raises(ValueError, match="unit_amount"):
        xero._build_line_items([{"description": "x", "unit_amount": "ten"}])


def test_request_uses_bearer_and_tenant_headers(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(xero.requests, "request", mock_request)

    result = xero._request("GET", "https://api.xero.com/x", tenant_id="tenant-123")

    assert result == {"ok": True}
    assert (
        mock_request.call_args.kwargs["headers"]["Authorization"]
        == "Bearer access-token"
    )
    assert mock_request.call_args.kwargs["headers"]["Xero-tenant-id"] == "tenant-123"


def test_request_returns_empty_dict_for_204(monkeypatch):
    monkeypatch.setattr(
        xero.requests,
        "request",
        Mock(return_value=MockResponse(status_code=204, text="")),
    )

    assert xero._request("DELETE", "https://api.xero.com/x") == {}


def test_request_raises_with_structured_error_detail(monkeypatch):
    monkeypatch.setattr(
        xero.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=400, json_data={"Message": "Bad request"}
            )
        ),
    )

    with pytest.raises(RuntimeError, match="Bad request"):
        xero._request("GET", "https://api.xero.com/x")


def test_request_raises_clear_error_for_non_json_2xx_body(monkeypatch):
    monkeypatch.setattr(
        xero.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=200, text="<html>not json</html>", json_raises=True
            )
        ),
    )

    with pytest.raises(RuntimeError, match="non-JSON"):
        xero._request("GET", "https://api.xero.com/x")


def test_accounting_request_requires_tenant_id():
    with pytest.raises(ValueError, match="tenant_id"):
        xero._accounting_request("GET", "  ", "/Contacts")


def test_accounting_request_builds_accounting_url(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(xero.requests, "request", mock_request)

    xero._accounting_request("GET", "tenant-1", "/Contacts")

    assert (
        mock_request.call_args.kwargs["url"]
        == "https://api.xero.com/api.xro/2.0/Contacts"
    )


def test_first_item_raises_when_empty():
    with pytest.raises(ValueError, match="nope"):
        xero._first_item({"Contacts": []}, "Contacts", "nope")


def test_first_item_returns_first_element():
    assert xero._first_item({"Contacts": [{"a": 1}, {"a": 2}]}, "Contacts", "x") == {
        "a": 1
    }


def test_list_organisations_returns_summaries(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data=[
                {"tenantId": "t1", "tenantName": "Acme", "tenantType": "ORGANISATION"}
            ]
        )
    )
    monkeypatch.setattr(xero.requests, "request", mock_request)

    result = json.loads(xero.xero_list_organisations())

    assert result["status"] == "success"
    assert result["organisations"] == [
        {"tenant_id": "t1", "tenant_name": "Acme", "tenant_type": "ORGANISATION"}
    ]
    assert mock_request.call_args.kwargs["url"] == xero.XERO_CONNECTIONS_URL
    assert "Xero-tenant-id" not in mock_request.call_args.kwargs["headers"]


def test_get_organisation_returns_first_entry(monkeypatch):
    monkeypatch.setattr(
        xero.requests,
        "request",
        Mock(
            return_value=MockResponse(json_data={"Organisations": [{"Name": "Acme"}]})
        ),
    )

    result = json.loads(xero.xero_get_organisation("tenant-1"))

    assert result["status"] == "success"
    assert result["organisation"]["Name"] == "Acme"


def test_list_contacts_sends_search_term_and_page(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "Contacts": [{"ContactID": "c1", "Name": "Jane"}],
                "pagination": {"page": 2, "pageCount": 3, "itemCount": 250},
            }
        )
    )
    monkeypatch.setattr(xero.requests, "request", mock_request)

    result = json.loads(xero.xero_list_contacts("tenant-1", search_term="jane", page=2))

    assert result["status"] == "success"
    assert result["contacts"]["contacts"][0]["name"] == "Jane"
    assert result["contacts"]["page_count"] == 3
    params = mock_request.call_args.kwargs["params"]
    assert params == {"page": 2, "searchTerm": "jane"}


def test_get_contact_returns_error_when_not_found(monkeypatch):
    monkeypatch.setattr(
        xero.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"Contacts": []})),
    )

    result = json.loads(xero.xero_get_contact("tenant-1", "missing"))

    assert result["status"] == "error"


def test_create_contact_requires_name():
    result = json.loads(xero.xero_create_contact("tenant-1", ""))

    assert result["status"] == "error"


def test_create_contact_sends_expected_body(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"Contacts": [{"ContactID": "c1", "Name": "Jane"}]}
        )
    )
    monkeypatch.setattr(xero.requests, "request", mock_request)

    result = json.loads(
        xero.xero_create_contact(
            "tenant-1", "Jane", email="jane@example.com", phone="555"
        )
    )

    assert result["status"] == "success"
    body = mock_request.call_args.kwargs["json"]["Contacts"][0]
    assert body["Name"] == "Jane"
    assert body["EmailAddress"] == "jane@example.com"
    assert body["Phones"] == [{"PhoneType": "DEFAULT", "PhoneNumber": "555"}]
    assert mock_request.call_args.kwargs["method"] == "POST"


def test_update_contact_requires_at_least_one_field():
    result = json.loads(xero.xero_update_contact("tenant-1", "c1"))

    assert result["status"] == "error"


def test_update_contact_sends_only_provided_fields(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"Contacts": [{"ContactID": "c1"}]})
    )
    monkeypatch.setattr(xero.requests, "request", mock_request)

    xero.xero_update_contact("tenant-1", "c1", email="new@example.com")

    body = mock_request.call_args.kwargs["json"]["Contacts"][0]
    assert body == {"EmailAddress": "new@example.com"}
    assert mock_request.call_args.kwargs["url"].endswith("/Contacts/c1")


def test_update_contact_clears_phone_with_empty_string(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"Contacts": [{"ContactID": "c1"}]})
    )
    monkeypatch.setattr(xero.requests, "request", mock_request)

    xero.xero_update_contact("tenant-1", "c1", phone="")

    body = mock_request.call_args.kwargs["json"]["Contacts"][0]
    assert body == {"Phones": []}


def test_update_contact_sets_single_default_phone(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"Contacts": [{"ContactID": "c1"}]})
    )
    monkeypatch.setattr(xero.requests, "request", mock_request)

    xero.xero_update_contact("tenant-1", "c1", phone="555-1234")

    body = mock_request.call_args.kwargs["json"]["Contacts"][0]
    assert body == {"Phones": [{"PhoneType": "DEFAULT", "PhoneNumber": "555-1234"}]}


def test_list_invoices_builds_where_clause_from_status_and_contact(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"Invoices": []}))
    monkeypatch.setattr(xero.requests, "request", mock_request)

    xero.xero_list_invoices("tenant-1", status="DRAFT", contact_id=_GUID)

    params = mock_request.call_args.kwargs["params"]
    assert params["where"] == f'Status=="DRAFT" AND Contact.ContactID==Guid("{_GUID}")'


def test_list_invoices_rejects_invalid_status():
    result = json.loads(xero.xero_list_invoices("tenant-1", status="BOGUS"))

    assert result["status"] == "error"


def test_list_invoices_rejects_non_guid_contact_id():
    result = json.loads(xero.xero_list_invoices("tenant-1", contact_id="not-a-guid"))

    assert result["status"] == "error"


def test_list_invoices_omits_where_when_no_filters(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"Invoices": []}))
    monkeypatch.setattr(xero.requests, "request", mock_request)

    xero.xero_list_invoices("tenant-1")

    assert "where" not in mock_request.call_args.kwargs["params"]


def test_get_invoice_includes_line_items(monkeypatch):
    monkeypatch.setattr(
        xero.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "Invoices": [
                        {
                            "InvoiceID": "i1",
                            "LineItems": [{"Description": "Consulting"}],
                        }
                    ]
                }
            )
        ),
    )

    result = json.loads(xero.xero_get_invoice("tenant-1", "i1"))

    assert result["status"] == "success"
    assert result["invoice"]["line_items"] == [{"Description": "Consulting"}]


def test_create_invoice_requires_line_items():
    result = json.loads(xero.xero_create_invoice("tenant-1", "c1", []))

    assert result["status"] == "error"


def test_create_invoice_rejects_invalid_type():
    result = json.loads(
        xero.xero_create_invoice(
            "tenant-1", "c1", [{"description": "x"}], invoice_type="BOGUS"
        )
    )

    assert result["status"] == "error"


def test_create_invoice_rejects_invalid_status():
    result = json.loads(
        xero.xero_create_invoice(
            "tenant-1", "c1", [{"description": "x"}], status="BOGUS"
        )
    )

    assert result["status"] == "error"


def test_create_invoice_sends_expected_body(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"Invoices": [{"InvoiceID": "i1"}]})
    )
    monkeypatch.setattr(xero.requests, "request", mock_request)

    result = json.loads(
        xero.xero_create_invoice(
            "tenant-1",
            "c1",
            [{"description": "Consulting", "quantity": 1, "unit_amount": 100.0}],
            date="2026-01-01",
            due_date="2026-01-15",
        )
    )

    assert result["status"] == "success"
    body = mock_request.call_args.kwargs["json"]["Invoices"][0]
    assert body["Type"] == "ACCREC"
    assert body["Contact"] == {"ContactID": "c1"}
    assert body["Status"] == "DRAFT"
    assert body["Date"] == "2026-01-01"
    assert body["DueDate"] == "2026-01-15"
    assert body["LineItems"] == [
        {"Description": "Consulting", "Quantity": 1, "UnitAmount": 100.0}
    ]


def test_update_invoice_status_sends_status_only(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"Invoices": [{"InvoiceID": "i1", "Status": "AUTHORISED"}]}
        )
    )
    monkeypatch.setattr(xero.requests, "request", mock_request)

    result = json.loads(xero.xero_update_invoice_status("tenant-1", "i1", "AUTHORISED"))

    assert result["status"] == "success"
    body = mock_request.call_args.kwargs["json"]["Invoices"][0]
    assert body == {"Status": "AUTHORISED"}
    assert mock_request.call_args.kwargs["url"].endswith("/Invoices/i1")


def test_update_invoice_status_rejects_invalid_status():
    result = json.loads(xero.xero_update_invoice_status("tenant-1", "i1", "BOGUS"))

    assert result["status"] == "error"


def test_list_accounts_sends_type_filter(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"Accounts": []}))
    monkeypatch.setattr(xero.requests, "request", mock_request)

    xero.xero_list_accounts("tenant-1", account_type="BANK")

    assert mock_request.call_args.kwargs["params"]["where"] == 'Type=="BANK"'


def test_list_accounts_rejects_account_type_with_embedded_quote():
    result = json.loads(
        xero.xero_list_accounts("tenant-1", account_type='BANK" || Type!="')
    )

    assert result["status"] == "error"


def test_list_accounts_omits_where_when_no_type(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"Accounts": []}))
    monkeypatch.setattr(xero.requests, "request", mock_request)

    xero.xero_list_accounts("tenant-1")

    assert "where" not in mock_request.call_args.kwargs["params"]


def test_list_payments_returns_summaries(monkeypatch):
    monkeypatch.setattr(
        xero.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "Payments": [
                        {
                            "PaymentID": "p1",
                            "Amount": 10.0,
                            "Invoice": {"InvoiceID": "i1", "InvoiceNumber": "INV-1"},
                        }
                    ]
                }
            )
        ),
    )

    result = json.loads(xero.xero_list_payments("tenant-1"))

    assert result["status"] == "success"
    assert result["payments"]["payments"][0]["invoice_number"] == "INV-1"


def test_xero_app_registry_uses_oauth_transport_and_provider():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    xero_app = next(
        row for row in get_builtin_public_mcp_app_rows() if row["app_id"] == "xero"
    )
    assert xero_app["transport"] == "oauth"
    assert xero_app["provider_name"] == "xero"
    assert xero_app["category"] == "Accounting"
    assert xero_app["launch_config"]["env_mapping"] == {
        "XERO_ACCESS_TOKEN": "access_token"
    }


def test_xero_oauth_provider_registry_has_expected_endpoints():
    from xagent.web.builtin_mcp_registry import get_builtin_oauth_provider_rows

    xero_provider = next(
        row
        for row in get_builtin_oauth_provider_rows()
        if row["provider_name"] == "xero"
    )
    assert (
        xero_provider["auth_url"] == "https://login.xero.com/identity/connect/authorize"
    )
    assert xero_provider["token_url"] == "https://identity.xero.com/connect/token"
