import json
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import google_ads


class MockResponse:
    def __init__(self, json_data=None, text="", status_code=200):
        self._json_data = json_data if json_data is not None else {}
        self.text = text or (json.dumps(self._json_data) if json_data else "")
        self.status_code = status_code
        self.content = self.text.encode()

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error", response=self)


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "access-token")
    monkeypatch.setenv("GOOGLE_ADS_DEVELOPER_TOKEN", "dev-token")


def test_headers_require_access_token(monkeypatch):
    monkeypatch.delenv("GOOGLE_ACCESS_TOKEN")

    with pytest.raises(ValueError, match="GOOGLE_ACCESS_TOKEN"):
        google_ads._headers()


def test_headers_require_developer_token(monkeypatch):
    monkeypatch.delenv("GOOGLE_ADS_DEVELOPER_TOKEN")

    with pytest.raises(ValueError, match="GOOGLE_ADS_DEVELOPER_TOKEN"):
        google_ads._headers()


def test_headers_include_bearer_and_developer_token():
    headers = google_ads._headers()

    assert headers["Authorization"] == "Bearer access-token"
    assert headers["developer-token"] == "dev-token"
    assert "login-customer-id" not in headers


def test_headers_include_normalized_login_customer_id():
    headers = google_ads._headers(login_customer_id="123-456-7890")

    assert headers["login-customer-id"] == "1234567890"


def test_headers_reject_login_customer_id_with_invalid_characters():
    with pytest.raises(ValueError, match="login_customer_id"):
        google_ads._headers(login_customer_id="1234567890\r\nX-Injected: 1")
    with pytest.raises(ValueError, match="login_customer_id"):
        google_ads._headers(login_customer_id="1234567890\n")


def test_headers_reject_all_dash_login_customer_id():
    """An all-dash value would otherwise strip to "" and send an empty header."""
    with pytest.raises(ValueError, match="login_customer_id"):
        google_ads._headers(login_customer_id="---")


def test_request_wraps_http_error_with_response_body(monkeypatch):
    monkeypatch.setattr(
        google_ads.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=400, text='{"error": {"message": "invalid query"}}'
            )
        ),
    )

    with pytest.raises(RuntimeError, match="invalid query"):
        google_ads._request("GET", "/customers:listAccessibleCustomers")


def test_request_extracts_structured_error_message(monkeypatch):
    monkeypatch.setattr(
        google_ads.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=400,
                json_data={
                    "error": {
                        "message": "Request contains an invalid argument.",
                        "details": [
                            {
                                "errors": [
                                    {"message": "Unrecognized field in GAQL query."}
                                ]
                            }
                        ],
                    }
                },
            )
        ),
    )

    with pytest.raises(RuntimeError, match="Unrecognized field in GAQL query"):
        google_ads._request("GET", "/customers:listAccessibleCustomers")


def test_request_falls_back_to_raw_text_for_unstructured_error_body(monkeypatch):
    monkeypatch.setattr(
        google_ads.requests,
        "request",
        Mock(return_value=MockResponse(status_code=500, text="upstream 500")),
    )

    with pytest.raises(RuntimeError, match="upstream 500"):
        google_ads._request("GET", "/customers:listAccessibleCustomers")


def test_list_accessible_customers_strips_resource_prefix(monkeypatch):
    monkeypatch.setattr(
        google_ads.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "resourceNames": ["customers/1112223333", "customers/4445556666"]
                }
            )
        ),
    )

    result = json.loads(google_ads.google_ads_list_accessible_customers())

    assert result["status"] == "success"
    assert result["customer_ids"] == ["1112223333", "4445556666"]


def test_list_accessible_customers_filters_non_string_resource_names(monkeypatch):
    monkeypatch.setattr(
        google_ads.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"resourceNames": ["customers/111", 42, None]}
            )
        ),
    )

    result = json.loads(google_ads.google_ads_list_accessible_customers())

    assert result["status"] == "success"
    assert result["customer_ids"] == ["111"]


def test_list_accessible_customers_rejects_non_dict_response(monkeypatch):
    monkeypatch.setattr(
        google_ads.requests,
        "request",
        Mock(return_value=MockResponse(json_data=["unexpected", "list"])),
    )

    result = json.loads(google_ads.google_ads_list_accessible_customers())

    assert result["status"] == "error"
    assert "Unexpected response format" in result["message"]


def test_google_ads_search_sends_gaql_query_and_login_customer_id(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"results": [{"foo": 1}]}))
    monkeypatch.setattr(google_ads.requests, "request", mock_request)

    result = json.loads(
        google_ads.google_ads_search(
            "111-222-3333",
            "SELECT campaign.id FROM campaign",
            login_customer_id="999-888-7777",
        )
    )

    assert result["status"] == "success"
    assert result["results"] == [{"foo": 1}]

    call_kwargs = mock_request.call_args.kwargs
    assert call_kwargs["url"].endswith("/customers/1112223333/googleAds:search")
    assert call_kwargs["json"] == {"query": "SELECT campaign.id FROM campaign"}
    assert call_kwargs["headers"]["login-customer-id"] == "9998887777"


def test_google_ads_search_returns_error_payload_on_failure(monkeypatch):
    monkeypatch.setattr(
        google_ads.requests,
        "request",
        Mock(return_value=MockResponse(status_code=400, text="bad request")),
    )

    result = json.loads(
        google_ads.google_ads_search("1112223333", "SELECT campaign.id")
    )

    assert result["status"] == "error"
    assert "bad request" in result["message"]


def test_google_ads_search_handles_empty_dict_result(monkeypatch):
    monkeypatch.setattr(
        google_ads.requests,
        "request",
        Mock(return_value=MockResponse(json_data={})),
    )

    result = json.loads(
        google_ads.google_ads_search("1112223333", "SELECT campaign.id")
    )

    assert result["status"] == "success"
    assert result["results"] == []


def test_google_ads_search_rejects_customer_id_with_invalid_characters(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(google_ads.requests, "request", mock_request)

    result = json.loads(
        google_ads.google_ads_search("111/../other", "SELECT campaign.id")
    )

    assert result["status"] == "error"
    assert "customer_id" in result["message"]
    mock_request.assert_not_called()


def test_google_ads_search_rejects_all_dash_customer_id(monkeypatch):
    """An all-dash value would otherwise strip to "" and hit /customers//... ."""
    mock_request = Mock()
    monkeypatch.setattr(google_ads.requests, "request", mock_request)

    result = json.loads(google_ads.google_ads_search("---", "SELECT campaign.id"))

    assert result["status"] == "error"
    assert "customer_id" in result["message"]
    mock_request.assert_not_called()
