import json
from unittest.mock import Mock

import pytest

from xagent.web.tools.mcp import myob
from xagent.web.tools.mcp import utils as mcp_utils


class MockResponse:
    def __init__(
        self,
        json_data=None,
        status_code: int = 200,
        text: str = "",
        headers: dict | None = None,
    ):
        self._json_data = json_data if json_data is not None else {}
        self.status_code = status_code
        self.text = text or (json.dumps(self._json_data) if json_data else "")
        self.content = self.text.encode()
        self.headers = headers or {}

    def json(self):
        return self._json_data


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("MYOB_ACCESS_TOKEN", "access-token")
    monkeypatch.setenv("MYOB_API_KEY", "consumer-key")
    monkeypatch.setenv("MYOB_BUSINESS_ID", "11111111-2222-3333-4444-555555555555")


def test_headers_require_access_token(monkeypatch):
    monkeypatch.delenv("MYOB_ACCESS_TOKEN")

    with pytest.raises(ValueError, match="MYOB_ACCESS_TOKEN"):
        myob._headers()


def test_headers_require_api_key(monkeypatch):
    monkeypatch.delenv("MYOB_API_KEY")

    with pytest.raises(ValueError, match="MYOB_API_KEY"):
        myob._headers()


def test_headers_do_not_include_cftoken():
    # cftoken (x-myobapi-cftoken) is a dead concept in the current MYOB auth
    # model -- the OAuth token plus x-myobapi-key is sufficient on its own,
    # confirmed against uptick/pymyob's own build_request_kwargs.
    headers = myob._headers()

    assert headers == {
        "Authorization": "Bearer access-token",
        "x-myobapi-key": "consumer-key",
        "x-myobapi-version": "v2",
    }


def test_business_id_requires_env_var(monkeypatch):
    monkeypatch.delenv("MYOB_BUSINESS_ID")

    with pytest.raises(ValueError, match="MYOB_BUSINESS_ID"):
        myob._business_id()


def test_base_url_includes_business_id():
    assert myob._base_url() == (
        "https://api.myob.com/accountright/11111111-2222-3333-4444-555555555555/"
    )


def test_request_uses_base_url_and_headers(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(myob.requests, "request", mock_request)

    result = myob._request("GET", "Contact/Customer/")

    assert result == {"ok": True}
    assert mock_request.call_args.kwargs["url"] == (
        "https://api.myob.com/accountright/11111111-2222-3333-4444-555555555555/"
        "Contact/Customer/"
    )
    assert mock_request.call_args.kwargs["headers"]["x-myobapi-key"] == "consumer-key"


def test_request_raises_with_joined_errors_array(monkeypatch):
    monkeypatch.setattr(
        myob.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=400,
                json_data={
                    "Errors": [
                        {
                            "Name": "ValidationException",
                            "Message": "CompanyName is required",
                            "AdditionalDetails": "",
                        }
                    ]
                },
            )
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        myob._request("GET", "Contact/Customer/")

    assert "ValidationException" in str(excinfo.value)
    assert "CompanyName is required" in str(excinfo.value)


def test_request_joins_list_shaped_additional_details(monkeypatch):
    # At least one documented MYOB error shape carries AdditionalDetails as
    # a list of per-field messages, not a plain string -- must be joined
    # into readable text, not fall through to a raw Python list repr like
    # "['TaxCode is required', 'Customer is required']".
    monkeypatch.setattr(
        myob.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=400,
                json_data={
                    "Errors": [
                        {
                            "Name": "ValidationException",
                            "Message": "Required fields are missing",
                            "AdditionalDetails": [
                                "TaxCode is required",
                                "Customer is required",
                            ],
                        }
                    ]
                },
            )
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        myob._request("GET", "Contact/Customer/")

    assert "TaxCode is required, Customer is required" in str(excinfo.value)
    assert "[" not in str(excinfo.value)


def test_request_falls_back_to_raw_text_for_unstructured_error(monkeypatch):
    monkeypatch.setattr(
        myob.requests,
        "request",
        Mock(return_value=MockResponse(status_code=500, text="Gateway error")),
    )

    with pytest.raises(RuntimeError) as excinfo:
        myob._request("GET", "Contact/Customer/")

    assert "Gateway error" in str(excinfo.value)


def test_request_truncates_long_error_body(monkeypatch):
    long_body = "x" * 5000
    monkeypatch.setattr(
        myob.requests,
        "request",
        Mock(return_value=MockResponse(status_code=500, text=long_body)),
    )

    with pytest.raises(RuntimeError) as excinfo:
        myob._request("GET", "Contact/Customer/")

    assert "[truncated]" in str(excinfo.value)
    assert len(str(excinfo.value)) < len(long_body)


def test_request_truncates_long_structured_error_detail(monkeypatch):
    # Truncation must apply to a successfully-extracted {"Errors": [...]}
    # detail too, not just the raw-text fallback used when that extraction
    # fails -- a single AdditionalDetails field (or many stacked errors) can
    # be just as unbounded as raw response text.
    long_message = "x" * 5000
    monkeypatch.setattr(
        myob.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=400,
                json_data={
                    "Errors": [
                        {
                            "Name": "ValidationException",
                            "Message": long_message,
                            "AdditionalDetails": "",
                        }
                    ]
                },
            )
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        myob._request("GET", "Contact/Customer/")

    assert "[truncated]" in str(excinfo.value)
    assert len(str(excinfo.value)) < len(long_message)


def test_list_items_unwraps_items_and_count():
    items, total_count = myob._list_items(
        {"Count": 2, "Items": [{"UID": "1"}, {"UID": "2"}]}
    )

    assert items == [{"UID": "1"}, {"UID": "2"}]
    assert total_count == 2


def test_list_items_tolerates_bare_array():
    items, total_count = myob._list_items([{"UID": "1"}])

    assert items == [{"UID": "1"}]
    assert total_count == 1


def test_list_customers_sends_filter_orderby_and_paging(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"Count": 1, "Items": [{"UID": "c1"}]})
    )
    monkeypatch.setattr(myob.requests, "request", mock_request)

    result = json.loads(
        myob.myob_list_customers(
            filter="IsActive eq true", orderby="CompanyName", top=10, skip=5
        )
    )

    assert result["status"] == "success"
    assert result["customers"] == [{"UID": "c1"}]
    assert result["total_count"] == 1
    assert mock_request.call_args.kwargs["params"] == {
        "$top": 10,
        "$skip": 5,
        "$filter": "IsActive eq true",
        "$orderby": "CompanyName",
    }
    assert mock_request.call_args.kwargs["url"].endswith("Contact/Customer/")


def test_list_customers_clamps_page_size(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"Items": []}))
    monkeypatch.setattr(myob.requests, "request", mock_request)

    myob.myob_list_customers(top=99999, skip=-5)

    params = mock_request.call_args.kwargs["params"]
    assert params["$top"] == myob.MAX_PAGE_LIMIT
    assert params["$skip"] == 0


def test_get_customer_percent_encodes_uid(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"UID": "c1"}))
    monkeypatch.setattr(myob.requests, "request", mock_request)

    myob.myob_get_customer("c1/../evil")

    url = mock_request.call_args.kwargs["url"]
    assert url.endswith("Contact/Customer/c1%2F..%2Fevil/")


def test_get_sales_invoice_caps_output_size(monkeypatch):
    # A real invoice's Lines array (unlike a flat customer/supplier record)
    # is open-ended -- this is the shape success_with_capped_dict exists to
    # protect against, unlike the list endpoints above which are already
    # bounded by top/skip.
    big_invoice = {
        "UID": "inv1",
        "Lines": [{"Description": "x" * 200} for _ in range(50)],
    }
    monkeypatch.setattr(
        myob.requests, "request", Mock(return_value=MockResponse(json_data=big_invoice))
    )
    monkeypatch.setattr(mcp_utils, "get_tool_max_output_length", lambda: 500)

    raw = myob.myob_get_sales_invoice("inv1")
    result = json.loads(raw)

    assert len(raw) <= 500
    assert result["status"] == "success"
    assert result["truncated"] is True


def test_create_sales_invoice_caps_output_size(monkeypatch):
    # returnBody=true means create/update tools echo back the full object
    # too, so they carry the same open-ended-Lines-array risk as the get
    # tool above, not just the list/get tools.
    big_invoice = {
        "UID": "inv1",
        "Lines": [{"Description": "x" * 200} for _ in range(50)],
    }
    monkeypatch.setattr(
        myob.requests, "request", Mock(return_value=MockResponse(json_data=big_invoice))
    )
    monkeypatch.setattr(mcp_utils, "get_tool_max_output_length", lambda: 500)

    raw = myob.myob_create_sales_invoice({"Customer": {"UID": "c1"}})
    result = json.loads(raw)

    assert len(raw) <= 500
    assert result["status"] == "success"
    assert result["truncated"] is True


def test_update_sales_invoice_caps_output_size(monkeypatch):
    current = {"UID": "inv1"}
    big_updated = {
        "UID": "inv1",
        "Lines": [{"Description": "x" * 200} for _ in range(50)],
    }
    monkeypatch.setattr(
        myob.requests,
        "request",
        Mock(
            side_effect=[
                MockResponse(json_data=current),
                MockResponse(json_data=big_updated),
            ]
        ),
    )
    monkeypatch.setattr(mcp_utils, "get_tool_max_output_length", lambda: 500)

    raw = myob.myob_update_sales_invoice("inv1", {"Status": "Closed"})
    result = json.loads(raw)

    assert len(raw) <= 500
    assert result["status"] == "success"
    assert result["truncated"] is True


def test_get_customer_rejects_empty_uid(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(myob.requests, "request", mock_request)

    result = json.loads(myob.myob_get_customer(""))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_create_customer_sends_return_body_and_json(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"UID": "c1", "CompanyName": "Acme"})
    )
    monkeypatch.setattr(myob.requests, "request", mock_request)

    result = json.loads(
        myob.myob_create_customer({"CompanyName": "Acme", "IsIndividual": False})
    )

    assert result["status"] == "success"
    assert result["customer"]["UID"] == "c1"
    assert mock_request.call_args.kwargs["json"] == {
        "CompanyName": "Acme",
        "IsIndividual": False,
    }
    assert mock_request.call_args.kwargs["params"] == {"returnBody": "true"}
    assert mock_request.call_args.kwargs["method"] == "POST"
    assert mock_request.call_args.kwargs["url"].endswith("Contact/Customer/")


def test_create_customer_rejects_empty_fields(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(myob.requests, "request", mock_request)

    result = json.loads(myob.myob_create_customer({}))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_create_customer_falls_back_to_location_header_when_body_is_empty(
    monkeypatch,
):
    mock_request = Mock(
        return_value=MockResponse(
            status_code=201,
            text="",
            headers={
                "Location": (
                    "https://api.myob.com/accountright/biz-id/Contact/Customer/c1-uid/"
                )
            },
        )
    )
    monkeypatch.setattr(myob.requests, "request", mock_request)

    result = json.loads(myob.myob_create_customer({"CompanyName": "Acme"}))

    assert result["status"] == "success"
    assert result["customer"]["UID"] == "c1-uid"


def test_update_customer_merges_fetched_record_with_changes(monkeypatch):
    current = {
        "UID": "c1",
        "CompanyName": "Old Name",
        "IsIndividual": False,
        "RowVersion": "abc123==",
    }
    updated = {**current, "CompanyName": "New Name"}
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=current),  # the GET inside _update_resource
            MockResponse(json_data=updated),  # the PUT
        ]
    )
    monkeypatch.setattr(myob.requests, "request", mock_request)

    result = json.loads(myob.myob_update_customer("c1", {"CompanyName": "New Name"}))

    assert result["status"] == "success"
    assert result["customer"]["CompanyName"] == "New Name"
    put_call = mock_request.call_args_list[1]
    assert put_call.kwargs["method"] == "PUT"
    # RowVersion/IsIndividual carried forward from the GET untouched, not
    # dropped just because the caller only named CompanyName.
    assert put_call.kwargs["json"] == updated
    assert put_call.kwargs["params"] == {"returnBody": "true"}


def test_update_customer_rejects_empty_fields(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(myob.requests, "request", mock_request)

    result = json.loads(myob.myob_update_customer("c1", {}))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_update_resource_rejects_an_empty_get_response(monkeypatch):
    # {} passes `isinstance(current, dict)` but must still be rejected --
    # letting it through would merge as `{**{}, **fields}`, silently
    # wiping every existing field (including RowVersion) on the PUT.
    mock_request = Mock(return_value=MockResponse(json_data={}))
    monkeypatch.setattr(myob.requests, "request", mock_request)

    with pytest.raises(RuntimeError, match="no existing record"):
        myob._update_resource("Contact/Customer/", "c1", {"CompanyName": "New Name"})

    mock_request.assert_called_once()  # the GET only -- no PUT attempted


def test_update_resource_rejects_a_non_dict_get_response(monkeypatch):
    # A truthy non-dict (e.g. a bare list) passes `not current` alone --
    # must still be rejected with the same clear message, not left to blow
    # up as an opaque TypeError at the dict-spread merge below.
    mock_request = Mock(return_value=MockResponse(json_data=[1, 2, 3]))
    monkeypatch.setattr(myob.requests, "request", mock_request)

    with pytest.raises(RuntimeError, match="no existing record"):
        myob._update_resource("Contact/Customer/", "c1", {"CompanyName": "New Name"})

    mock_request.assert_called_once()  # the GET only -- no PUT attempted


def test_update_supplier_merges_fetched_record_with_changes(monkeypatch):
    current = {"UID": "s1", "CompanyName": "Old Supplier", "RowVersion": "abc=="}
    updated = {**current, "CompanyName": "New Supplier"}
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=current),
            MockResponse(json_data=updated),
        ]
    )
    monkeypatch.setattr(myob.requests, "request", mock_request)

    result = json.loads(
        myob.myob_update_supplier("s1", {"CompanyName": "New Supplier"})
    )

    assert result["status"] == "success"
    assert result["supplier"]["CompanyName"] == "New Supplier"
    put_call = mock_request.call_args_list[1]
    assert put_call.kwargs["method"] == "PUT"
    assert put_call.kwargs["json"] == updated


def test_update_sales_invoice_merges_fetched_record_with_changes(monkeypatch):
    current = {"UID": "inv1", "Status": "Open", "RowVersion": "abc=="}
    updated = {**current, "Status": "Closed"}
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=current),
            MockResponse(json_data=updated),
        ]
    )
    monkeypatch.setattr(myob.requests, "request", mock_request)

    result = json.loads(myob.myob_update_sales_invoice("inv1", {"Status": "Closed"}))

    assert result["status"] == "success"
    assert result["invoice"]["Status"] == "Closed"
    put_call = mock_request.call_args_list[1]
    assert put_call.kwargs["method"] == "PUT"
    assert put_call.kwargs["json"] == updated


def test_update_purchase_bill_merges_fetched_record_with_changes(monkeypatch):
    current = {"UID": "bill1", "Status": "Open", "RowVersion": "abc=="}
    updated = {**current, "Status": "Closed"}
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=current),
            MockResponse(json_data=updated),
        ]
    )
    monkeypatch.setattr(myob.requests, "request", mock_request)

    result = json.loads(myob.myob_update_purchase_bill("bill1", {"Status": "Closed"}))

    assert result["status"] == "success"
    assert result["bill"]["Status"] == "Closed"
    put_call = mock_request.call_args_list[1]
    assert put_call.kwargs["method"] == "PUT"
    assert put_call.kwargs["json"] == updated


def test_get_business_info_unwraps_company_file(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"CompanyFile": {"Name": "Acme Pty Ltd", "Uri": "..."}}
        )
    )
    monkeypatch.setattr(myob.requests, "request", mock_request)

    result = json.loads(myob.myob_get_business_info())

    assert result["status"] == "success"
    assert result["business"]["Name"] == "Acme Pty Ltd"
    # The business-info call hits the base business URL itself, not a
    # resource path underneath it.
    assert mock_request.call_args.kwargs["url"] == myob._base_url()


def test_list_sales_invoices_uses_item_layout_endpoint(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"Items": []}))
    monkeypatch.setattr(myob.requests, "request", mock_request)

    myob.myob_list_sales_invoices()

    assert mock_request.call_args.kwargs["url"].endswith("Sale/Invoice/Item/")


def test_list_purchase_bills_uses_item_layout_endpoint(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"Items": []}))
    monkeypatch.setattr(myob.requests, "request", mock_request)

    myob.myob_list_purchase_bills()

    assert mock_request.call_args.kwargs["url"].endswith("Purchase/Bill/Item/")


def test_list_accounts_uses_general_ledger_endpoint(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"Items": []}))
    monkeypatch.setattr(myob.requests, "request", mock_request)

    myob.myob_list_accounts()

    assert mock_request.call_args.kwargs["url"].endswith("GeneralLedger/Account/")


def test_list_tax_codes_uses_general_ledger_endpoint(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"Items": []}))
    monkeypatch.setattr(myob.requests, "request", mock_request)

    myob.myob_list_tax_codes()

    assert mock_request.call_args.kwargs["url"].endswith("GeneralLedger/TaxCode/")


def test_capped_list_halves_items_to_fit_output_limit(monkeypatch):
    monkeypatch.setattr(myob, "get_tool_max_output_length", lambda: 200)

    raw = myob._success_with_capped_list(
        "customers",
        [{"UID": str(i), "Name": "x" * 50} for i in range(20)],
        total_count=20,
    )
    result = json.loads(raw)

    assert len(raw) <= 200
    assert result["truncated"] is True
    assert len(result["customers"]) < 20


def test_myob_app_registry_scopes_exclude_unused_families():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    myob_app = next(
        row for row in get_builtin_public_mcp_app_rows() if row["app_id"] == "myob"
    )
    assert "sme-contacts-customer" in myob_app["oauth_scopes"]
    assert "sme-general-ledger" in myob_app["oauth_scopes"]
    # No payroll/employee/personal/timebilling/banking tools are exposed, so
    # none of those scopes should be requested.
    for unused_scope in (
        "sme-banking",
        "sme-payroll",
        "sme-contacts-employee",
        "sme-contacts-personal",
        "sme-timebilling",
    ):
        assert unused_scope not in myob_app["oauth_scopes"]
    assert myob_app["launch_config"]["env_mapping"] == {
        "MYOB_ACCESS_TOKEN": "access_token",
        "MYOB_BUSINESS_ID": "instance_url",
    }
    assert myob_app["launch_config"]["static_env"] == {"MYOB_API_KEY": "MYOB_CLIENT_ID"}
