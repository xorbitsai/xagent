import json
from unittest.mock import Mock

import pytest

from xagent.web.tools.mcp import chartmogul
from xagent.web.tools.mcp import utils as mcp_utils


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


def test_request_drops_empty_string_valued_params(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    chartmogul._request("GET", "/customers", params={"email": "", "cursor": "abc"})

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
                "entries": [
                    {
                        "uuid": "cus_1",
                        "name": "Acme",
                        "mrr": 935700,
                        "arr": 11228400,
                        "currency": "PHP",
                    }
                ],
                "has_more": False,
                "cursor": None,
            }
        )
    )
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    result = json.loads(chartmogul.chartmogul_list_customers())

    assert result["status"] == "success"
    assert result["entries"][0]["name"] == "Acme"
    assert result["entries"][0]["mrr"] == "9357.00"
    assert result["entries"][0]["mrr_cents"] == 935700
    assert result["entries"][0]["arr"] == "112284.00"
    assert result["entries"][0]["arr_cents"] == 11228400
    assert result["entries"][0]["currency"] == "PHP"
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


def test_list_customers_treats_null_entries_as_empty_page(monkeypatch):
    monkeypatch.setattr(
        chartmogul.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"entries": None, "has_more": False, "cursor": None}
            )
        ),
    )

    result = json.loads(chartmogul.chartmogul_list_customers())

    assert result["status"] == "success"
    assert result["entries"] == []


def test_list_customers_rejects_non_object_entry(monkeypatch):
    monkeypatch.setattr(
        chartmogul.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"entries": [42]})),
    )

    result = json.loads(chartmogul.chartmogul_list_customers())

    assert result["status"] == "error"
    assert "unexpected response for /customers" in result["message"]


def test_list_customers_wraps_non_integer_money_as_error(monkeypatch):
    monkeypatch.setattr(
        chartmogul.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"entries": [{"uuid": "cus_1", "mrr": "935700"}]}
            )
        ),
    )

    result = json.loads(chartmogul.chartmogul_list_customers())

    assert result["status"] == "error"
    assert "non-integer value for mrr: str" in result["message"]


def test_list_customers_clamps_per_page(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"entries": [], "has_more": False})
    )
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    chartmogul.chartmogul_list_customers(per_page=10000)

    assert (
        mock_request.call_args.kwargs["params"]["per_page"] == chartmogul.MAX_PER_PAGE
    )


@pytest.mark.parametrize(
    ("cents", "expected"),
    [(0, "0.00"), (1, "0.01"), (-1, "-0.01"), (935700, "9357.00")],
)
def test_major_units_is_exact(cents, expected):
    assert chartmogul._major_units(cents, "amount") == expected


@pytest.mark.parametrize("value", [True, 1.0, "100", {}])
def test_major_units_rejects_non_integer_source_values(value):
    with pytest.raises(TypeError, match="non-integer value"):
        chartmogul._major_units(value, "amount")


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


@pytest.mark.parametrize(
    "invoke",
    [
        lambda: chartmogul.chartmogul_create_customer(
            {"data_source_uuid": "ds_1", "external_id": "acme-1"}
        ),
        lambda: chartmogul.chartmogul_get_customer("cus_1"),
        lambda: chartmogul.chartmogul_update_customer(
            "cus_1", {"company": "Acme Inc."}
        ),
    ],
)
def test_single_customer_results_use_major_currency_units(monkeypatch, invoke):
    monkeypatch.setattr(
        chartmogul.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "uuid": "cus_1",
                    "mrr": 935700,
                    "arr": 11228400,
                    "currency": "PHP",
                }
            )
        ),
    )

    customer = json.loads(invoke())["customer"]

    assert customer["mrr"] == "9357.00"
    assert customer["mrr_cents"] == 935700
    assert customer["arr"] == "112284.00"
    assert customer["arr_cents"] == 11228400
    assert customer["currency"] == "PHP"


def test_customer_money_normalization_preserves_null_and_missing_fields():
    assert chartmogul._customer_with_major_units({"mrr": None}) == {"mrr": None}
    assert chartmogul._customer_with_major_units({"uuid": "cus_1"}) == {"uuid": "cus_1"}


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
        "cus_1", {"data_source_uuid": "ds_1", "email": "a@example.com"}
    )

    assert mock_request.call_args.kwargs["method"] == "POST"
    assert mock_request.call_args.kwargs["url"] == (
        "https://api.chartmogul.com/v1/customers/cus_1/contacts"
    )
    assert mock_request.call_args.kwargs["json"] == {
        "data_source_uuid": "ds_1",
        "email": "a@example.com",
    }


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
        return_value=MockResponse(
            json_data={
                "entries": [
                    {
                        "uuid": "opp_1",
                        "amount_in_cents": 100000,
                        "currency": "USD",
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    result = json.loads(chartmogul.chartmogul_list_opportunities(customer_uuid="cus_1"))

    assert result["status"] == "success"
    assert result["entries"][0]["uuid"] == "opp_1"
    assert result["entries"][0]["amount"] == "1000.00"
    assert result["entries"][0]["amount_in_cents"] == 100000
    assert result["entries"][0]["currency"] == "USD"
    assert mock_request.call_args.kwargs["url"] == (
        "https://api.chartmogul.com/v1/opportunities"
    )


def test_create_opportunity_sends_post(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"uuid": "opp_1"}))
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    data = {
        "customer_uuid": "cus_1",
        "owner": "sales@example.com",
        "pipeline": "Default",
        "pipeline_stage": "Discovery",
        "estimated_close_date": "2026-12-31",
        "currency": "USD",
        "amount_in_cents": 100000,
    }
    chartmogul.chartmogul_create_opportunity(data)

    assert mock_request.call_args.kwargs["method"] == "POST"
    assert mock_request.call_args.kwargs["url"] == (
        "https://api.chartmogul.com/v1/opportunities"
    )
    assert mock_request.call_args.kwargs["json"] == data


def test_get_opportunity_returns_record(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"uuid": "opp_1"}))
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    result = json.loads(chartmogul.chartmogul_get_opportunity("opp_1"))

    assert result["status"] == "success"
    assert result["opportunity"]["uuid"] == "opp_1"


def test_update_opportunity_sends_patch(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"uuid": "opp_1"}))
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    data = {"pipeline_stage": "Won", "amount_in_cents": 100000}
    chartmogul.chartmogul_update_opportunity("opp_1", data)

    assert mock_request.call_args.kwargs["method"] == "PATCH"
    assert mock_request.call_args.kwargs["json"] == data


@pytest.mark.parametrize(
    "invoke",
    [
        lambda: chartmogul.chartmogul_create_opportunity(
            {
                "customer_uuid": "cus_1",
                "owner": "sales@example.com",
                "pipeline": "Default",
                "pipeline_stage": "Discovery",
                "estimated_close_date": "2026-12-31",
                "currency": "USD",
                "amount_in_cents": 100000,
            }
        ),
        lambda: chartmogul.chartmogul_get_opportunity("opp_1"),
        lambda: chartmogul.chartmogul_update_opportunity(
            "opp_1", {"amount_in_cents": 100000}
        ),
    ],
)
def test_single_opportunity_results_include_major_currency_amount(monkeypatch, invoke):
    monkeypatch.setattr(
        chartmogul.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "uuid": "opp_1",
                    "amount_in_cents": 100000,
                    "currency": "USD",
                }
            )
        ),
    )

    opportunity = json.loads(invoke())["opportunity"]

    assert opportunity["amount"] == "1000.00"
    assert opportunity["amount_in_cents"] == 100000
    assert opportunity["currency"] == "USD"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            {"uuid": "opp_1", "amount_in_cents": None},
            {"uuid": "opp_1", "amount_in_cents": None},
        ),
        ({"uuid": "opp_1"}, {"uuid": "opp_1"}),
    ],
)
def test_opportunity_money_normalization_preserves_null_and_missing_fields(
    source, expected
):
    assert chartmogul._opportunity_with_major_units(source) == expected


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
# Metrics
# ---------------------------------------------------------------------------


def test_get_metrics_mrr_normalizes_money_and_passes_filters(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "entries": [
                    {
                        "date": "2026-06-30",
                        "mrr": 2183300,
                        "mrr-new-business": 10000,
                        "mrr-expansion": 5000,
                        "mrr-contraction": -2500,
                        "mrr-churn": -1000,
                        "mrr-reactivation": 0,
                        "percentage-change": 1.2,
                    },
                    {"date": "2026-07-31", "mrr": 2190000},
                    {"date": "2026-08-31", "mrr": 2200000},
                    {"date": "2026-09-30", "mrr": 2210000},
                ],
                "summary": {
                    "current": 2183300,
                    "previous": 2100000,
                    "percentage-change": 3.97,
                },
            }
        )
    )
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    result = json.loads(
        chartmogul.chartmogul_get_metrics(
            "mrr",
            "2026-06-01",
            "2026-09-30",
            geo="SG",
            plans="Gold",
        )
    )

    assert result["status"] == "success"
    assert result["metric"] == "mrr"
    assert [entry["date"] for entry in result["metrics"]["entries"]] == [
        "2026-06-30",
        "2026-07-31",
        "2026-08-31",
        "2026-09-30",
    ]
    entry = result["metrics"]["entries"][0]
    assert entry["mrr"] == "21833.00"
    assert entry["mrr_cents"] == 2183300
    assert entry["mrr-contraction"] == "-25.00"
    assert entry["mrr_contraction_cents"] == -2500
    assert entry["percentage-change"] == 1.2
    summary = result["metrics"]["summary"]
    assert summary["current"] == "21833.00"
    assert summary["current_cents"] == 2183300
    assert summary["percentage-change"] == 3.97

    request = mock_request.call_args.kwargs
    assert request["method"] == "GET"
    assert request["url"] == "https://api.chartmogul.com/v1/metrics/mrr"
    assert request["auth"] == ("test-api-key", "")
    assert request["params"] == {
        "start-date": "2026-06-01",
        "end-date": "2026-09-30",
        "interval": "month",
        "geo": "SG",
        "plans": "Gold",
    }


def test_get_metrics_all_normalizes_only_monetary_fields(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "entries": [
                    {
                        "date": "2026-09-30",
                        "mrr": 2183300,
                        "arr": 26199600,
                        "ltv": 145000.25,
                        "asp": 9900,
                        "arpa": 545825.0,
                        "customers": 4,
                        "customer-churn-rate": 2.5,
                        "mrr-churn-rate": 1.25,
                    }
                ],
                "summary": {
                    "current-mrr": 2183300,
                    "previous-mrr": 2100000,
                    "current-ltv": 145000.25,
                    "current-customers": 4,
                    "previous-customers": 3,
                    "current-customer-churn-rate": 2.5,
                },
            }
        )
    )
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    result = json.loads(
        chartmogul.chartmogul_get_metrics(
            "all", "2026-09-01", "2026-09-30", interval="day"
        )
    )

    entry = result["metrics"]["entries"][0]
    assert entry["mrr"] == "21833.00"
    assert entry["mrr_cents"] == 2183300
    assert entry["arpa"] == "5458.25"
    assert entry["arpa_cents"] == 545825.0
    assert entry["ltv"] == "1450.00"
    assert entry["ltv_cents"] == 145000.25
    assert entry["customers"] == 4
    assert entry["customer-churn-rate"] == 2.5
    summary = result["metrics"]["summary"]
    assert summary["current-mrr"] == "21833.00"
    assert summary["current_mrr_cents"] == 2183300
    assert summary["current-ltv"] == "1450.00"
    assert summary["current_ltv_cents"] == 145000.25
    assert summary["current-customers"] == 4
    assert summary["current-customer-churn-rate"] == 2.5
    request = mock_request.call_args.kwargs
    assert request["url"] == "https://api.chartmogul.com/v1/metrics/all"
    assert request["params"]["interval"] == "day"


def test_get_metrics_truncation_keeps_latest_entries(monkeypatch):
    entries = [{"date": f"2026-01-{day:02d}", "mrr": day * 100} for day in range(1, 29)]
    monkeypatch.setattr(
        chartmogul.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "entries": entries,
                    "summary": {"current": 2800, "previous": 2700},
                }
            )
        ),
    )
    monkeypatch.setattr(mcp_utils, "get_tool_max_output_length", lambda: 800)

    result = json.loads(
        chartmogul.chartmogul_get_metrics(
            "mrr", "2026-01-01", "2026-01-28", interval="day"
        )
    )

    returned_dates = [entry["date"] for entry in result["metrics"]["entries"]]
    assert result["truncated"] is True
    assert 0 < len(returned_dates) < len(entries)
    assert returned_dates == [
        entry["date"] for entry in entries[-len(returned_dates) :]
    ]


def test_get_metrics_count_values_are_not_treated_as_money(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "entries": [{"date": "2026-09-30", "customers": 4}],
                "summary": {"current": 4, "previous": 3},
            }
        )
    )
    monkeypatch.setattr(
        chartmogul.requests,
        "request",
        mock_request,
    )

    result = json.loads(
        chartmogul.chartmogul_get_metrics("customer-count", "2026-09-01", "2026-09-30")
    )

    assert result["metrics"]["summary"] == {"current": 4, "previous": 3}
    assert "geo" not in mock_request.call_args.kwargs["params"]
    assert "plans" not in mock_request.call_args.kwargs["params"]


def test_get_metrics_asp_allows_documented_quarter_interval(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "entries": [{"date": "2026-09-30", "asp": 9900}],
                "summary": {"current": 9900, "previous": 9500},
            }
        )
    )
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    result = json.loads(
        chartmogul.chartmogul_get_metrics(
            "asp", "2026-07-01", "2026-09-30", interval="quarter"
        )
    )

    assert result["status"] == "success"
    assert result["metrics"]["entries"][0]["asp"] == "99.00"
    assert mock_request.call_args.kwargs["params"]["interval"] == "quarter"


def test_get_metrics_accepts_fractional_ltv_cents(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "entries": [{"date": "2026-09-30", "ltv": 1862989.7959183701}],
                "summary": {
                    "current": 1862989.7959183701,
                    "previous": 1838875.5102040817,
                },
            }
        )
    )
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    result = json.loads(
        chartmogul.chartmogul_get_metrics("ltv", "2026-09-01", "2026-09-30")
    )

    assert result["status"] == "success"
    entry = result["metrics"]["entries"][0]
    assert entry["ltv"] == "18629.90"
    assert entry["ltv_cents"] == 1862989.7959183701
    assert result["metrics"]["summary"]["current"] == "18629.90"
    assert result["metrics"]["summary"]["previous"] == "18388.76"
    assert "interval" not in mock_request.call_args.kwargs["params"]


@pytest.mark.parametrize("value", [True, "1862989.7959183701", {}, float("nan")])
def test_get_metrics_rejects_invalid_ltv_money(monkeypatch, value):
    monkeypatch.setattr(
        chartmogul.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "entries": [{"date": "2026-09-30", "ltv": value}],
                    "summary": {"current": 145000, "previous": 140000},
                }
            )
        ),
    )

    result = json.loads(
        chartmogul.chartmogul_get_metrics("ltv", "2026-09-01", "2026-09-30")
    )

    assert result["status"] == "error"
    assert "value for ltv" in result["message"]


@pytest.mark.parametrize(
    ("start_date", "end_date", "expected_message"),
    [
        ("2026-02-30", "2026-03-01", "start_date"),
        ("2026-W01-1", "2026-03-01", "start_date"),
        ("2026-10-01", "2026-09-30", "end_date must be on or after"),
    ],
)
def test_get_metrics_rejects_invalid_date_ranges(
    monkeypatch, start_date, end_date, expected_message
):
    mock_request = Mock()
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    result = json.loads(chartmogul.chartmogul_get_metrics("mrr", start_date, end_date))

    assert result["status"] == "error"
    assert expected_message in result["message"]
    mock_request.assert_not_called()


@pytest.mark.parametrize(
    ("metric", "interval", "expected_message"),
    [
        ("invoices", "month", "metric must be one of"),
        ("mrr", "hour", "interval for mrr must be one of"),
        ("asp", "day", "interval for asp must be one of"),
        ("ltv", "year", "interval for ltv must be one of"),
    ],
)
def test_get_metrics_rejects_unsupported_values(
    monkeypatch, metric, interval, expected_message
):
    mock_request = Mock()
    monkeypatch.setattr(chartmogul.requests, "request", mock_request)

    result = json.loads(
        chartmogul.chartmogul_get_metrics(
            metric, "2026-09-01", "2026-09-30", interval=interval
        )
    )

    assert result["status"] == "error"
    assert expected_message in result["message"]
    mock_request.assert_not_called()


def test_get_metrics_rejects_unexpected_response_shape(monkeypatch):
    monkeypatch.setattr(
        chartmogul.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"entries": []})),
    )

    result = json.loads(
        chartmogul.chartmogul_get_metrics("mrr", "2026-09-01", "2026-09-30")
    )

    assert result["status"] == "error"
    assert "unexpected response" in result["message"]


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


def test_success_with_capped_list_drops_message_as_last_resort(monkeypatch):
    # Small enough that even an empty item list plus the added "message"
    # text doesn't fit -- forces the post-loop fallback that drops the
    # message itself rather than returning an over-limit response.
    monkeypatch.setattr(chartmogul, "get_tool_max_output_length", lambda: 40)

    result = json.loads(
        chartmogul._success_with_capped_list(
            "entries",
            {
                "entries": [{"uuid": "cus_1"}, {"uuid": "cus_2"}],
                "has_more": True,
                "cursor": "abc",
            },
        )
    )

    assert result["entries"] == []
    assert result["truncated"] is True
    assert "message" not in result


async def test_mcp_registers_all_sixteen_tools():
    """Direct-call unit tests above exercise each tool's Python body, but
    none of them go through FastMCP's own registration/schema layer -- a
    tool whose @mcp.tool() decorator was dropped, or whose name was
    typo'd, would still pass every test above while being unreachable to
    an agent."""
    tools = await chartmogul.mcp.list_tools()

    assert {tool.name for tool in tools} == {
        "chartmogul_list_customers",
        "chartmogul_create_customer",
        "chartmogul_get_customer",
        "chartmogul_update_customer",
        "chartmogul_delete_customer",
        "chartmogul_list_contacts",
        "chartmogul_create_contact",
        "chartmogul_get_contact",
        "chartmogul_update_contact",
        "chartmogul_delete_contact",
        "chartmogul_list_opportunities",
        "chartmogul_create_opportunity",
        "chartmogul_get_opportunity",
        "chartmogul_update_opportunity",
        "chartmogul_delete_opportunity",
        "chartmogul_get_metrics",
    }
