import json
from unittest.mock import Mock

import pytest

from xagent.web.tools.mcp import chartmogul


class MockResponse:
    def __init__(self, json_data=None, status_code: int = 200, text: str = ""):
        self.status_code = status_code
        # `json_data is not None`, not truthiness: an explicit `{}` must
        # still serialize to the text "{}" rather than falling through to
        # the empty-string default, or a test constructing MockResponse({})
        # to check error-detail-extraction-on-empty-body would silently
        # exercise the wrong code path.
        self.text = text or (json.dumps(json_data) if json_data is not None else "")
        self.content = self.text.encode()

    def json(self):
        # Parses ``self.text`` for real, rather than returning a
        # precomputed dict unconditionally -- otherwise a test constructing
        # MockResponse(text="<html>...</html>") to exercise
        # _extract_error_detail's `except ValueError` branch would pass
        # without ever actually reaching it, since .json() would never
        # raise. Matches real requests.Response.json()'s contract: invalid
        # JSON raises, it doesn't silently return something else.
        try:
            return json.loads(self.text)
        except json.JSONDecodeError as e:
            raise ValueError("No JSON object could be decoded") from e


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("CHARTMOGUL_API_KEY", "test-api-key")


# ---------------------------------------------------------------------------
# _api_key
# ---------------------------------------------------------------------------


def test_api_key_requires_env_var(monkeypatch):
    monkeypatch.delenv("CHARTMOGUL_API_KEY")

    with pytest.raises(ValueError, match="CHARTMOGUL_API_KEY"):
        chartmogul._api_key()


def test_api_key_rejects_whitespace_only(monkeypatch):
    monkeypatch.setenv("CHARTMOGUL_API_KEY", "   ")

    with pytest.raises(ValueError, match="CHARTMOGUL_API_KEY"):
        chartmogul._api_key()


def test_api_key_strips_surrounding_whitespace(monkeypatch):
    monkeypatch.setenv("CHARTMOGUL_API_KEY", "  test-api-key\n")

    assert chartmogul._api_key() == "test-api-key"


# ---------------------------------------------------------------------------
# _extract_error_detail
# ---------------------------------------------------------------------------


def test_extract_error_detail_prefers_message_key():
    response = MockResponse(json_data={"message": "Invalid API key", "error": "auth"})

    assert chartmogul._extract_error_detail(response) == "Invalid API key"


def test_extract_error_detail_handles_a_list_of_field_errors():
    response = MockResponse(json_data={"errors": ["external_id can't be blank"]})

    detail = chartmogul._extract_error_detail(response)

    assert detail is not None
    assert "external_id can't be blank" in detail


def test_extract_error_detail_returns_none_for_non_json():
    response = MockResponse(text="<html>gateway error</html>")

    assert chartmogul._extract_error_detail(response) is None


# ---------------------------------------------------------------------------
# _request
# ---------------------------------------------------------------------------


def test_request_uses_basic_auth_with_empty_password(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    chartmogul._request("GET", "/customers")

    assert mock_request.call_args.kwargs["auth"] == ("test-api-key", "")
    assert mock_request.call_args.kwargs["url"] == (
        "https://api.chartmogul.com/v1/customers"
    )


def test_request_drops_none_valued_params(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    chartmogul._request("GET", "/customers", params={"status": None, "cursor": "abc"})

    assert mock_request.call_args.kwargs["params"] == {"cursor": "abc"}


def test_request_raises_with_error_detail_on_failure(monkeypatch):
    monkeypatch.setattr(
        chartmogul.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=401, json_data={"message": "Invalid API key"}
            )
        ),
    )

    with pytest.raises(RuntimeError, match="Invalid API key"):
        chartmogul._request("GET", "/customers")


def test_request_returns_empty_dict_on_204(monkeypatch):
    monkeypatch.setattr(
        chartmogul.requests,
        "request",
        Mock(return_value=MockResponse(status_code=204, text="")),
    )

    assert chartmogul._request("PATCH", "/customers/uuid-1") == {}


# ---------------------------------------------------------------------------
# chartmogul_list_customers
# ---------------------------------------------------------------------------


def test_list_customers_returns_entries(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "entries": [{"uuid": "cus_1", "name": "Acme"}],
                "has_more": False,
                "cursor": None,
            }
        )
    )
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    result = json.loads(chartmogul.chartmogul_list_customers())

    assert result["status"] == "success"
    assert result["entries"][0]["name"] == "Acme"
    assert result["has_more"] is False
    assert mock_request.call_args.kwargs["url"] == (
        "https://api.chartmogul.com/v1/customers"
    )
    assert mock_request.call_args.kwargs["method"] == "GET"


def test_list_customers_rejects_unexpected_shape(monkeypatch):
    monkeypatch.setattr(
        chartmogul.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"unexpected": True})),
    )

    result = json.loads(chartmogul.chartmogul_list_customers())

    assert result["status"] == "error"


def test_list_customers_clamps_per_page(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"entries": [], "has_more": False})
    )
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    chartmogul.chartmogul_list_customers(per_page=10000)

    assert (
        mock_request.call_args.kwargs["params"]["per_page"] == chartmogul.MAX_PER_PAGE
    )


# ---------------------------------------------------------------------------
# chartmogul_create_customer / chartmogul_get_customer / chartmogul_update_customer
# ---------------------------------------------------------------------------


def test_create_customer_returns_created_record(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"uuid": "cus_1", "name": "Acme"})
    )
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    result = json.loads(
        chartmogul.chartmogul_create_customer(
            {"data_source_uuid": "ds_1", "external_id": "acme-1", "name": "Acme"}
        )
    )

    assert result["status"] == "success"
    assert result["customer"]["uuid"] == "cus_1"
    assert mock_request.call_args.kwargs["method"] == "POST"
    assert mock_request.call_args.kwargs["json"]["name"] == "Acme"


def test_get_customer_percent_encodes_uuid(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"uuid": "cus_1"}))
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    chartmogul.chartmogul_get_customer("cus/1")

    assert mock_request.call_args.kwargs["url"] == (
        "https://api.chartmogul.com/v1/customers/cus%2F1"
    )


def test_update_customer_sends_patch(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"uuid": "cus_1"}))
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    chartmogul.chartmogul_update_customer("cus_1", {"company": "Acme Inc."})

    assert mock_request.call_args.kwargs["method"] == "PATCH"
    assert mock_request.call_args.kwargs["url"] == (
        "https://api.chartmogul.com/v1/customers/cus_1"
    )


def test_get_customer_returns_error_payload_on_failure(monkeypatch):
    monkeypatch.setattr(
        chartmogul.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=404, json_data={"message": "Customer not found"}
            )
        ),
    )

    result = json.loads(chartmogul.chartmogul_get_customer("cus_missing"))

    assert result["status"] == "error"
    assert "Customer not found" in result["message"]


# ---------------------------------------------------------------------------
# chartmogul_list_contacts / chartmogul_create_contact / chartmogul_get_contact /
# chartmogul_update_contact
# ---------------------------------------------------------------------------


def test_list_contacts_returns_entries(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"entries": [{"uuid": "con_1", "email": "a@example.com"}]}
        )
    )
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    result = json.loads(chartmogul.chartmogul_list_contacts(email="a@example.com"))

    assert result["status"] == "success"
    assert result["entries"][0]["email"] == "a@example.com"
    assert mock_request.call_args.kwargs["url"] == (
        "https://api.chartmogul.com/v1/contacts"
    )


def test_create_contact_sends_post(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"uuid": "con_1"}))
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    chartmogul.chartmogul_create_contact(
        {"customer_uuid": "cus_1", "email": "a@example.com"}
    )

    assert mock_request.call_args.kwargs["method"] == "POST"
    assert mock_request.call_args.kwargs["url"] == (
        "https://api.chartmogul.com/v1/contacts"
    )


def test_get_contact_returns_record(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"uuid": "con_1"}))
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    result = json.loads(chartmogul.chartmogul_get_contact("con_1"))

    assert result["status"] == "success"
    assert result["contact"]["uuid"] == "con_1"


def test_update_contact_sends_patch(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"uuid": "con_1"}))
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    chartmogul.chartmogul_update_contact("con_1", {"title": "VP"})

    assert mock_request.call_args.kwargs["method"] == "PATCH"
    assert mock_request.call_args.kwargs["json"] == {"title": "VP"}


# ---------------------------------------------------------------------------
# chartmogul_list_opportunities / chartmogul_create_opportunity /
# chartmogul_get_opportunity / chartmogul_update_opportunity
# ---------------------------------------------------------------------------


def test_list_opportunities_returns_entries(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"entries": [{"uuid": "opp_1"}]})
    )
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    result = json.loads(chartmogul.chartmogul_list_opportunities(customer_uuid="cus_1"))

    assert result["status"] == "success"
    assert result["entries"][0]["uuid"] == "opp_1"
    assert mock_request.call_args.kwargs["url"] == (
        "https://api.chartmogul.com/v1/opportunities"
    )


def test_create_opportunity_sends_post(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"uuid": "opp_1"}))
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    chartmogul.chartmogul_create_opportunity(
        {
            "customer_uuid": "cus_1",
            "owner": "sales@example.com",
            "pipeline": "Default",
            "pipeline_stage": "Discovery",
            "estimated_close_date": "2026-12-31",
            "currency": "USD",
            "amount_in_cents": 100000,
        }
    )

    assert mock_request.call_args.kwargs["method"] == "POST"
    assert mock_request.call_args.kwargs["url"] == (
        "https://api.chartmogul.com/v1/opportunities"
    )


def test_get_opportunity_returns_record(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"uuid": "opp_1"}))
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    result = json.loads(chartmogul.chartmogul_get_opportunity("opp_1"))

    assert result["status"] == "success"
    assert result["opportunity"]["uuid"] == "opp_1"


def test_update_opportunity_sends_patch(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"uuid": "opp_1"}))
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    chartmogul.chartmogul_update_opportunity("opp_1", {"pipeline_stage": "Won"})

    assert mock_request.call_args.kwargs["method"] == "PATCH"
    assert mock_request.call_args.kwargs["json"] == {"pipeline_stage": "Won"}


def test_list_opportunities_rejects_unexpected_shape(monkeypatch):
    monkeypatch.setattr(
        chartmogul.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"unexpected": True})),
    )

    result = json.loads(chartmogul.chartmogul_list_opportunities())

    assert result["status"] == "error"


def test_list_opportunities_passes_estimated_close_date_filters(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"entries": []}))
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    chartmogul.chartmogul_list_opportunities(
        estimated_close_date_on_or_after="2026-01-01",
        estimated_close_date_on_or_before="2026-12-31",
    )

    params = mock_request.call_args.kwargs["params"]
    assert params["estimated_close_date_on_or_after"] == "2026-01-01"
    assert params["estimated_close_date_on_or_before"] == "2026-12-31"


def test_list_customers_passes_email_and_system_filters(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"entries": []}))
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    chartmogul.chartmogul_list_customers(email="a@example.com", system="Stripe")

    params = mock_request.call_args.kwargs["params"]
    assert params["email"] == "a@example.com"
    assert params["system"] == "Stripe"


def test_list_contacts_passes_data_source_uuid_filter(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"entries": []}))
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    chartmogul.chartmogul_list_contacts(data_source_uuid="ds_1")

    assert mock_request.call_args.kwargs["params"]["data_source_uuid"] == "ds_1"


# ---------------------------------------------------------------------------
# Delete tools
# ---------------------------------------------------------------------------


def test_delete_customer_sends_delete(monkeypatch):
    mock_request = Mock(return_value=MockResponse(status_code=204, text=""))
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    result = json.loads(chartmogul.chartmogul_delete_customer("cus_1"))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["method"] == "DELETE"
    assert mock_request.call_args.kwargs["url"] == (
        "https://api.chartmogul.com/v1/customers/cus_1"
    )


def test_delete_contact_sends_delete(monkeypatch):
    mock_request = Mock(return_value=MockResponse(status_code=204, text=""))
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    result = json.loads(chartmogul.chartmogul_delete_contact("con_1"))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["method"] == "DELETE"
    assert mock_request.call_args.kwargs["url"] == (
        "https://api.chartmogul.com/v1/contacts/con_1"
    )


def test_delete_opportunity_sends_delete(monkeypatch):
    mock_request = Mock(return_value=MockResponse(status_code=204, text=""))
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    result = json.loads(chartmogul.chartmogul_delete_opportunity("opp_1"))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["method"] == "DELETE"
    assert mock_request.call_args.kwargs["url"] == (
        "https://api.chartmogul.com/v1/opportunities/opp_1"
    )


def test_delete_customer_returns_error_payload_on_failure(monkeypatch):
    monkeypatch.setattr(
        chartmogul.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=404, json_data={"message": "Customer not found"}
            )
        ),
    )

    result = json.loads(chartmogul.chartmogul_delete_customer("cus_missing"))

    assert result["status"] == "error"
    assert "Customer not found" in result["message"]


# ---------------------------------------------------------------------------
# Proxy/connection failure redaction and truncated-list messaging
# ---------------------------------------------------------------------------


def test_request_redacts_credentials_from_connection_failure(monkeypatch):
    def _raise(*args, **kwargs):
        raise chartmogul.requests.exceptions.ProxyError(
            "Unable to connect to proxy: https://user:sup3rsecret@proxy.example.com"
        )

    monkeypatch.setattr(chartmogul.requests, "request", _raise)

    with pytest.raises(RuntimeError) as exc_info:
        chartmogul._request("GET", "/customers")

    assert "sup3rsecret" not in str(exc_info.value)


def test_success_with_capped_list_message_warns_data_is_not_recoverable(monkeypatch):
    big_item = {"uuid": "cus_1", "blob": "x" * 100000}
    monkeypatch.setattr(chartmogul, "get_tool_max_output_length", lambda: 5000)

    result = json.loads(
        chartmogul._success_with_capped_list(
            "entries",
            {"entries": [big_item, big_item], "has_more": True, "cursor": "abc"},
        )
    )

    assert result["truncated"] is True
    assert "cannot be recovered" in result["message"]
    # cursor/has_more still describe the original fetched page, unchanged
    assert result["cursor"] == "abc"
    assert result["has_more"] is True
