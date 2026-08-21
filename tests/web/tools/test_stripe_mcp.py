import json
from unittest.mock import Mock

import pytest

from xagent.web.tools.mcp import stripe


class MockResponse:
    def __init__(
        self, json_data=None, status_code: int = 200, text: str = "", url: str = ""
    ):
        self._json_data = json_data if json_data is not None else {}
        self.status_code = status_code
        self.text = text or (json.dumps(self._json_data) if json_data else "")
        self.content = self.text.encode()
        self.url = url

    def json(self):
        return self._json_data


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("STRIPE_API_KEY", "rk_test_123")


def test_headers_require_api_key(monkeypatch):
    monkeypatch.delenv("STRIPE_API_KEY")

    with pytest.raises(ValueError, match="STRIPE_API_KEY"):
        stripe._headers()


def test_headers_include_bearer_token_only():
    assert stripe._headers() == {"Authorization": "Bearer rk_test_123"}


def test_headers_include_idempotency_key_when_given():
    headers = stripe._headers(idempotency_key="abc123")

    assert headers == {
        "Authorization": "Bearer rk_test_123",
        "Idempotency-Key": "abc123",
    }


def test_path_segment_percent_encodes_path_traversal_characters():
    assert stripe._path_segment("cus_1/../account") == "cus_1%2F..%2Faccount"


def test_idempotency_key_is_stable_for_identical_arguments():
    first = stripe._idempotency_key("POST", "/refunds", {"charge": "ch_1"})
    second = stripe._idempotency_key("POST", "/refunds", {"charge": "ch_1"})

    assert first == second


def test_idempotency_key_differs_for_different_arguments():
    first = stripe._idempotency_key("POST", "/refunds", {"charge": "ch_1"})
    second = stripe._idempotency_key("POST", "/refunds", {"charge": "ch_2"})

    assert first != second


def test_flatten_form_params_flattens_nested_dict():
    result = stripe._flatten_form_params({"metadata": {"order_id": "6735"}})

    assert result == [("metadata[order_id]", "6735")]


def test_flatten_form_params_flattens_list():
    result = stripe._flatten_form_params({"expand": ["customer", "invoice"]})

    assert result == [("expand[0]", "customer"), ("expand[1]", "invoice")]


def test_flatten_form_params_omits_none_values():
    result = stripe._flatten_form_params({"name": "Acme", "description": None})

    assert result == [("name", "Acme")]


def test_flatten_form_params_serializes_booleans_lowercase():
    result = stripe._flatten_form_params(
        {"metadata": {"is_active": True, "is_pending": False}}
    )

    assert result == [
        ("metadata[is_active]", "true"),
        ("metadata[is_pending]", "false"),
    ]


def test_request_uses_base_url_and_headers(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": "acct_123"}))
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    result = stripe._request("GET", "/account")

    assert result == {"id": "acct_123"}
    assert mock_request.call_args.kwargs["url"] == "https://api.stripe.com/v1/account"
    assert (
        mock_request.call_args.kwargs["headers"]["Authorization"]
        == "Bearer rk_test_123"
    )


def test_request_form_encodes_nested_form_data(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": "cus_123"}))
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    stripe._request(
        "POST",
        "/customers",
        form_data={"name": "Acme", "metadata": {"order_id": "6735"}},
    )

    assert mock_request.call_args.kwargs["data"] == [
        ("name", "Acme"),
        ("metadata[order_id]", "6735"),
    ]


def test_request_sends_idempotency_key_only_for_post(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": "cus_123"}))
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    stripe._request("POST", "/customers", form_data={"name": "Acme"})

    assert "Idempotency-Key" in mock_request.call_args.kwargs["headers"]


def test_request_omits_idempotency_key_for_get(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": "acct_123"}))
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    stripe._request("GET", "/account")

    assert "Idempotency-Key" not in mock_request.call_args.kwargs["headers"]


def test_request_reuses_idempotency_key_for_identical_retry_arguments(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": "re_1"}))
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    stripe._request("POST", "/refunds", form_data={"charge": "ch_1"})
    stripe._request("POST", "/refunds", form_data={"charge": "ch_1"})

    first_key = mock_request.call_args_list[0].kwargs["headers"]["Idempotency-Key"]
    second_key = mock_request.call_args_list[1].kwargs["headers"]["Idempotency-Key"]
    assert first_key == second_key


def test_request_retries_once_on_429_with_retry_after(monkeypatch):
    responses = [
        MockResponse(status_code=429, text="rate limited"),
        MockResponse(json_data={"id": "acct_123"}),
    ]
    responses[0].headers = {"Retry-After": "1"}
    mock_request = Mock(side_effect=responses)
    monkeypatch.setattr(stripe.requests, "request", mock_request)
    mock_sleep = Mock()
    monkeypatch.setattr(stripe.time, "sleep", mock_sleep)

    result = stripe._request("GET", "/account")

    assert result == {"id": "acct_123"}
    assert mock_request.call_count == 2
    mock_sleep.assert_called_once_with(1)


def test_request_does_not_retry_past_max_retry_after(monkeypatch):
    response = MockResponse(status_code=429, text="rate limited")
    response.headers = {"Retry-After": str(stripe.MAX_RETRY_AFTER_SECONDS + 1)}
    mock_request = Mock(return_value=response)
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    with pytest.raises(RuntimeError, match="429"):
        stripe._request("GET", "/account")

    assert mock_request.call_count == 1


def test_request_raises_with_structured_error_detail(monkeypatch):
    monkeypatch.setattr(
        stripe.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=402,
                json_data={
                    "error": {
                        "type": "card_error",
                        "code": "card_declined",
                        "message": "Your card was declined.",
                    }
                },
            )
        ),
    )

    with pytest.raises(RuntimeError, match="Your card was declined"):
        stripe._request("GET", "/charges/ch_123")


def test_request_truncates_unstructured_error_body(monkeypatch):
    long_body = "x" * 5000
    monkeypatch.setattr(
        stripe.requests,
        "request",
        Mock(return_value=MockResponse(status_code=500, text=long_body)),
    )

    with pytest.raises(RuntimeError) as excinfo:
        stripe._request("GET", "/charges/ch_123")

    assert "[truncated]" in str(excinfo.value)
    assert len(str(excinfo.value)) < len(long_body)


def test_get_account_info_returns_profile(monkeypatch):
    monkeypatch.setattr(
        stripe.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "id": "acct_123",
                    "business_profile": {"name": "Acme Corp"},
                    "email": "billing@acme.com",
                    "country": "US",
                    "default_currency": "usd",
                }
            )
        ),
    )

    result = json.loads(stripe.stripe_get_account_info())

    assert result["status"] == "success"
    assert result["account"]["business_name"] == "Acme Corp"
    assert result["account"]["id"] == "acct_123"


def test_get_account_info_returns_error_payload_on_failure(monkeypatch):
    monkeypatch.setattr(
        stripe.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=401,
                json_data={
                    "error": {
                        "type": "authentication_error",
                        "message": "Invalid API key",
                    }
                },
            )
        ),
    )

    result = json.loads(stripe.stripe_get_account_info())

    assert result["status"] == "error"
    assert "Invalid API key" in result["message"]


def test_get_balance_returns_available_and_pending(monkeypatch):
    monkeypatch.setattr(
        stripe.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "available": [{"amount": 1000, "currency": "usd"}],
                    "pending": [{"amount": 500, "currency": "usd"}],
                }
            )
        ),
    )

    result = json.loads(stripe.stripe_get_balance())

    assert result["status"] == "success"
    assert result["available"] == [{"amount": 1000, "currency": "usd"}]
    assert result["pending"] == [{"amount": 500, "currency": "usd"}]


def test_list_customers_includes_email_filter_and_truncated_flag(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "data": [{"id": "cus_1", "email": "ada@example.com"}],
                "has_more": True,
            }
        )
    )
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    result = json.loads(stripe.stripe_list_customers(email="ada@example.com"))

    assert result["status"] == "success"
    assert result["customers"] == [{"id": "cus_1", "email": "ada@example.com"}]
    assert result["truncated"] is True
    assert mock_request.call_args.kwargs["params"]["email"] == "ada@example.com"


def test_list_customers_not_truncated_when_no_more(monkeypatch):
    monkeypatch.setattr(
        stripe.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"data": [], "has_more": False})),
    )

    result = json.loads(stripe.stripe_list_customers())

    assert result["status"] == "success"
    assert result["truncated"] is False


def test_get_customer_returns_customer(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": "cus_1"}))
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    result = json.loads(stripe.stripe_get_customer("cus_1"))

    assert result["status"] == "success"
    assert result["customer"]["id"] == "cus_1"
    assert mock_request.call_args.kwargs["url"].endswith("/customers/cus_1")


def test_get_customer_percent_encodes_customer_id(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": "cus_1"}))
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    stripe.stripe_get_customer("cus_1/../account")

    assert mock_request.call_args.kwargs["url"].endswith(
        "/customers/cus_1%2F..%2Faccount"
    )


def test_create_customer_sends_form_data_with_metadata(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": "cus_1"}))
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    result = json.loads(
        stripe.stripe_create_customer(
            name="Acme", email="billing@acme.com", metadata={"order_id": "6735"}
        )
    )

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["data"] == [
        ("name", "Acme"),
        ("email", "billing@acme.com"),
        ("metadata[order_id]", "6735"),
    ]


def test_create_customer_omits_optional_fields_when_not_provided(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": "cus_1"}))
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    stripe.stripe_create_customer()

    assert mock_request.call_args.kwargs["data"] is None


def test_list_charges_uses_customer_filter(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"data": [{"id": "ch_1"}], "has_more": False}
        )
    )
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    result = json.loads(stripe.stripe_list_charges(customer_id="cus_1"))

    assert result["status"] == "success"
    assert result["charges"] == [{"id": "ch_1"}]
    assert mock_request.call_args.kwargs["params"]["customer"] == "cus_1"


def test_get_charge_returns_charge(monkeypatch):
    monkeypatch.setattr(
        stripe.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"id": "ch_1", "amount": 500})),
    )

    result = json.loads(stripe.stripe_get_charge("ch_1"))

    assert result["status"] == "success"
    assert result["charge"]["amount"] == 500


def test_create_refund_requires_charge_or_payment_intent():
    result = json.loads(stripe.stripe_create_refund())

    assert result["status"] == "error"
    assert "charge_id or payment_intent_id" in result["message"]


def test_create_refund_sends_charge_and_amount(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": "re_1"}))
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    result = json.loads(stripe.stripe_create_refund(charge_id="ch_1", amount=500))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["data"] == [
        ("charge", "ch_1"),
        ("amount", 500),
    ]


def test_create_refund_sends_payment_intent(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": "re_1"}))
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    stripe.stripe_create_refund(
        payment_intent_id="pi_1", reason="requested_by_customer"
    )

    assert mock_request.call_args.kwargs["data"] == [
        ("payment_intent", "pi_1"),
        ("reason", "requested_by_customer"),
    ]


def test_list_payment_intents_returns_results(monkeypatch):
    monkeypatch.setattr(
        stripe.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"data": [{"id": "pi_1"}], "has_more": False}
            )
        ),
    )

    result = json.loads(stripe.stripe_list_payment_intents())

    assert result["status"] == "success"
    assert result["payment_intents"] == [{"id": "pi_1"}]


def test_list_invoices_uses_status_filter(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"data": [{"id": "in_1"}], "has_more": False}
        )
    )
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    result = json.loads(stripe.stripe_list_invoices(status="open"))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["params"]["status"] == "open"


def test_get_invoice_returns_invoice(monkeypatch):
    monkeypatch.setattr(
        stripe.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"id": "in_1", "total": 1000})),
    )

    result = json.loads(stripe.stripe_get_invoice("in_1"))

    assert result["status"] == "success"
    assert result["invoice"]["total"] == 1000


def test_list_subscriptions_uses_status_filter(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"data": [{"id": "sub_1"}], "has_more": False}
        )
    )
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    result = json.loads(stripe.stripe_list_subscriptions(status="past_due"))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["params"]["status"] == "past_due"


def test_list_products_uses_active_filter(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"data": [{"id": "prod_1"}], "has_more": False}
        )
    )
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    result = json.loads(stripe.stripe_list_products(active=True))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["params"]["active"] == "true"


def test_list_products_serializes_active_false_as_lowercase_string(monkeypatch):
    """requests would otherwise serialize a bare Python bool as "True"/"False"
    in the query string, which Stripe's API does not accept."""
    mock_request = Mock(
        return_value=MockResponse(json_data={"data": [], "has_more": False})
    )
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    stripe.stripe_list_products(active=False)

    assert mock_request.call_args.kwargs["params"]["active"] == "false"


def test_list_prices_uses_product_filter(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"data": [{"id": "price_1"}], "has_more": False}
        )
    )
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    result = json.loads(stripe.stripe_list_prices(product_id="prod_1"))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["params"]["product"] == "prod_1"


def test_stripe_app_registry_requires_api_key():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    stripe_app = next(
        row for row in get_builtin_public_mcp_app_rows() if row["app_id"] == "stripe"
    )
    assert stripe_app["provider_name"] is None
    assert stripe_app["launch_config"]["required_env"] == ["STRIPE_API_KEY"]
