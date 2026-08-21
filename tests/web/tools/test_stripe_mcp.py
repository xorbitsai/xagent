import json
from unittest.mock import Mock

import pytest

from xagent.web.tools.mcp import stripe


class MockResponse:
    def __init__(
        self,
        json_data=None,
        status_code: int = 200,
        text: str = "",
        url: str = "",
        headers=None,
    ):
        self._json_data = json_data if json_data is not None else {}
        self.status_code = status_code
        self.text = text or (json.dumps(self._json_data) if json_data else "")
        self.content = self.text.encode()
        self.url = url
        self.headers = headers or {}

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


def test_headers_accept_live_restricted_key(monkeypatch):
    monkeypatch.setenv("STRIPE_API_KEY", "rk_live_abc")

    assert stripe._headers() == {"Authorization": "Bearer rk_live_abc"}


def test_headers_reject_full_secret_key(monkeypatch):
    monkeypatch.setenv("STRIPE_API_KEY", "sk_live_abc")

    with pytest.raises(ValueError, match="Restricted API Key"):
        stripe._headers()


def test_headers_reject_test_mode_full_secret_key(monkeypatch):
    monkeypatch.setenv("STRIPE_API_KEY", "sk_test_abc")

    with pytest.raises(ValueError, match="Restricted API Key"):
        stripe._headers()


def test_headers_include_idempotency_key_when_given():
    headers = stripe._headers(idempotency_key="abc123")

    assert headers == {
        "Authorization": "Bearer rk_test_123",
        "Idempotency-Key": "abc123",
    }


def test_path_segment_percent_encodes_path_traversal_characters():
    assert stripe._path_segment("cus_1/../account") == "cus_1%2F..%2Faccount"


def test_generate_idempotency_key_is_not_stable_across_calls():
    first = stripe._generate_idempotency_key()
    second = stripe._generate_idempotency_key()

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


def test_flatten_form_params_rejects_bracket_in_key():
    with pytest.raises(ValueError, match="Invalid form field name"):
        stripe._flatten_form_params({"metadata": {"foo]bar": "1"}})


def test_flatten_form_params_rejects_empty_key():
    with pytest.raises(ValueError, match="must be non-empty"):
        stripe._flatten_form_params({"": "1"})


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


def test_request_logs_warning_on_idempotent_replay(monkeypatch, caplog):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"id": "cus_1"}, headers={"Idempotent-Replayed": "true"}
        )
    )
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    with caplog.at_level("WARNING", logger=stripe.logger.name):
        stripe._request("POST", "/customers", form_data={"name": "Acme"})

    assert "idempotent replay" in caplog.text.lower()


def test_request_does_not_log_warning_on_fresh_create(monkeypatch, caplog):
    mock_request = Mock(return_value=MockResponse(json_data={"id": "cus_1"}))
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    with caplog.at_level("WARNING", logger=stripe.logger.name):
        stripe._request("POST", "/customers", form_data={"name": "Acme"})

    assert "idempotent replay" not in caplog.text.lower()


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


def test_request_generates_different_key_for_identical_default_calls(monkeypatch):
    """Two independent calls with the same arguments and no explicit
    idempotency_key must NOT collide -- see _generate_idempotency_key's
    docstring for why a content-derived key was wrong."""
    mock_request = Mock(return_value=MockResponse(json_data={"id": "re_1"}))
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    stripe._request("POST", "/refunds", form_data={"charge": "ch_1"})
    stripe._request("POST", "/refunds", form_data={"charge": "ch_1"})

    first_key = mock_request.call_args_list[0].kwargs["headers"]["Idempotency-Key"]
    second_key = mock_request.call_args_list[1].kwargs["headers"]["Idempotency-Key"]
    assert first_key != second_key


def test_request_reuses_caller_supplied_idempotency_key(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": "re_1"}))
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    stripe._request(
        "POST", "/refunds", form_data={"charge": "ch_1"}, idempotency_key="retry-1"
    )
    stripe._request(
        "POST", "/refunds", form_data={"charge": "ch_2"}, idempotency_key="retry-1"
    )

    first_key = mock_request.call_args_list[0].kwargs["headers"]["Idempotency-Key"]
    second_key = mock_request.call_args_list[1].kwargs["headers"]["Idempotency-Key"]
    assert first_key == second_key == "retry-1"


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


def test_request_reuses_same_headers_and_data_across_429_retry(monkeypatch):
    responses = [
        MockResponse(status_code=429, text="rate limited"),
        MockResponse(json_data={"id": "cus_1"}),
    ]
    responses[0].headers = {"Retry-After": "1"}
    mock_request = Mock(side_effect=responses)
    monkeypatch.setattr(stripe.requests, "request", mock_request)
    monkeypatch.setattr(stripe.time, "sleep", Mock())

    stripe._request("POST", "/customers", form_data={"name": "Acme"})

    first_call, second_call = mock_request.call_args_list
    assert first_call.kwargs["headers"] == second_call.kwargs["headers"]
    assert first_call.kwargs["data"] == second_call.kwargs["data"]


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


def test_request_redacts_structured_error_message(monkeypatch):
    monkeypatch.setattr(
        stripe.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=402,
                json_data={
                    "error": {
                        "message": "Gateway error, Authorization: Bearer sk_live_leaked"
                    }
                },
            )
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        stripe._request("GET", "/charges/ch_123")

    assert "sk_live_leaked" not in str(excinfo.value)


def test_request_redacts_raw_error_body(monkeypatch):
    monkeypatch.setattr(
        stripe.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=500,
                text="<html>Authorization: Bearer sk_live_leaked</html>",
            )
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        stripe._request("GET", "/charges/ch_123")

    assert "sk_live_leaked" not in str(excinfo.value)


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


def test_request_flags_replay_on_error_response(monkeypatch):
    monkeypatch.setattr(
        stripe.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=402,
                json_data={"error": {"message": "Your card was declined."}},
                headers={"Idempotent-Replayed": "true"},
            )
        ),
    )

    with pytest.raises(RuntimeError, match="replay of a previous failed attempt"):
        stripe._request("POST", "/refunds", form_data={"charge": "ch_1"})


def test_request_does_not_mention_replay_on_fresh_error(monkeypatch):
    monkeypatch.setattr(
        stripe.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=402,
                json_data={"error": {"message": "Your card was declined."}},
            )
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        stripe._request("POST", "/refunds", form_data={"charge": "ch_1"})

    assert "replay" not in str(excinfo.value).lower()


def test_request_wraps_network_exception(monkeypatch):
    monkeypatch.setattr(
        stripe.requests,
        "request",
        Mock(side_effect=stripe.requests.ConnectionError("boom at http://x:y@host")),
    )

    with pytest.raises(RuntimeError, match="Stripe request failed"):
        stripe._request("GET", "/account")


def test_request_redacts_credentials_from_network_exception(monkeypatch):
    monkeypatch.setattr(
        stripe.requests,
        "request",
        Mock(
            side_effect=stripe.requests.ConnectionError(
                "ProxyError connecting via http://user:secret@proxyhost:8080"
            )
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        stripe._request("GET", "/account")

    assert "secret" not in str(excinfo.value)


def test_request_network_exception_suggests_idempotency_key_for_post(monkeypatch):
    monkeypatch.setattr(
        stripe.requests,
        "request",
        Mock(side_effect=stripe.requests.ConnectionError("timed out")),
    )

    with pytest.raises(RuntimeError) as excinfo:
        stripe._request(
            "POST",
            "/refunds",
            form_data={"charge": "ch_1"},
            idempotency_key="retry-me",
        )

    assert 'idempotency_key="retry-me"' in str(excinfo.value)


def test_request_network_exception_omits_idempotency_hint_for_get(monkeypatch):
    monkeypatch.setattr(
        stripe.requests,
        "request",
        Mock(side_effect=stripe.requests.ConnectionError("timed out")),
    )

    with pytest.raises(RuntimeError) as excinfo:
        stripe._request("GET", "/account")

    assert "idempotency_key" not in str(excinfo.value)


def test_request_truncates_long_network_exception_text(monkeypatch):
    monkeypatch.setattr(
        stripe.requests,
        "request",
        Mock(side_effect=stripe.requests.ConnectionError("x" * 5000)),
    )

    with pytest.raises(RuntimeError) as excinfo:
        stripe._request("GET", "/account")

    assert "[truncated]" in str(excinfo.value)
    assert len(str(excinfo.value)) < 5000


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


def test_create_customer_reports_idempotent_replayed_true_on_dedup(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"id": "cus_1"}, headers={"Idempotent-Replayed": "true"}
        )
    )
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    result = json.loads(stripe.stripe_create_customer(name="Acme"))

    assert result["idempotent_replayed"] is True


def test_create_customer_reports_idempotent_replayed_false_on_fresh_create(
    monkeypatch,
):
    mock_request = Mock(return_value=MockResponse(json_data={"id": "cus_1"}))
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    result = json.loads(stripe.stripe_create_customer(name="Acme"))

    assert result["idempotent_replayed"] is False


def test_create_customer_passes_through_caller_supplied_idempotency_key(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": "cus_1"}))
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    stripe.stripe_create_customer(name="Acme", idempotency_key="my-retry-key")

    assert mock_request.call_args.kwargs["headers"]["Idempotency-Key"] == "my-retry-key"


def test_get_customer_requires_non_empty_id():
    result = json.loads(stripe.stripe_get_customer(""))

    assert result["status"] == "error"
    assert "customer_id is required" in result["message"]


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


def test_list_charges_clamps_limit(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"data": [], "has_more": False})
    )
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    stripe.stripe_list_charges(limit=500)

    assert mock_request.call_args.kwargs["params"]["limit"] == stripe.MAX_LIMIT


def test_list_charges_forwards_starting_after(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"data": [], "has_more": False})
    )
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    stripe.stripe_list_charges(starting_after="ch_1")

    assert mock_request.call_args.kwargs["params"]["starting_after"] == "ch_1"


def test_list_charges_reports_truncated_flag(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"data": [{"id": "ch_1"}], "has_more": True}
        )
    )
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    result = json.loads(stripe.stripe_list_charges())

    assert result["truncated"] is True


def test_list_charges_returns_error_payload_on_failure(monkeypatch):
    monkeypatch.setattr(
        stripe.requests,
        "request",
        Mock(return_value=MockResponse(status_code=500, text="boom")),
    )

    result = json.loads(stripe.stripe_list_charges())

    assert result["status"] == "error"


def test_create_refund_returns_error_payload_on_failure(monkeypatch):
    monkeypatch.setattr(
        stripe.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=402, json_data={"error": {"message": "card declined"}}
            )
        ),
    )

    result = json.loads(stripe.stripe_create_refund(charge_id="ch_1"))

    assert result["status"] == "error"
    assert "card declined" in result["message"]


def test_get_charge_returns_charge(monkeypatch):
    monkeypatch.setattr(
        stripe.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"id": "ch_1", "amount": 500})),
    )

    result = json.loads(stripe.stripe_get_charge("ch_1"))

    assert result["status"] == "success"
    assert result["charge"]["amount"] == 500


def test_get_charge_requires_non_empty_id():
    result = json.loads(stripe.stripe_get_charge(""))

    assert result["status"] == "error"
    assert "charge_id is required" in result["message"]


def test_create_refund_requires_charge_or_payment_intent():
    result = json.loads(stripe.stripe_create_refund())

    assert result["status"] == "error"
    assert "charge_id or payment_intent_id" in result["message"]


def test_create_refund_rejects_both_charge_and_payment_intent(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": "re_1"}))
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    result = json.loads(
        stripe.stripe_create_refund(charge_id="ch_1", payment_intent_id="pi_1")
    )

    assert result["status"] == "error"
    assert "not both" in result["message"]
    mock_request.assert_not_called()


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


def test_create_refund_sends_metadata(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": "re_1"}))
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    stripe.stripe_create_refund(charge_id="ch_1", metadata={"internal_id": "6735"})

    assert mock_request.call_args.kwargs["data"] == [
        ("charge", "ch_1"),
        ("metadata[internal_id]", "6735"),
    ]


def test_create_refund_reports_idempotent_replayed_true_on_dedup(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"id": "re_1"}, headers={"Idempotent-Replayed": "true"}
        )
    )
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    result = json.loads(stripe.stripe_create_refund(charge_id="ch_1"))

    assert result["idempotent_replayed"] is True


def test_create_refund_reports_idempotent_replayed_false_on_fresh_create(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": "re_1"}))
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    result = json.loads(stripe.stripe_create_refund(charge_id="ch_1"))

    assert result["idempotent_replayed"] is False


def test_create_refund_passes_through_caller_supplied_idempotency_key(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": "re_1"}))
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    stripe.stripe_create_refund(charge_id="ch_1", idempotency_key="my-retry-key")

    assert mock_request.call_args.kwargs["headers"]["Idempotency-Key"] == "my-retry-key"


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


def test_get_invoice_requires_non_empty_id():
    result = json.loads(stripe.stripe_get_invoice(""))

    assert result["status"] == "error"
    assert "invoice_id is required" in result["message"]


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


def test_list_prices_uses_active_filter(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"data": [{"id": "price_1"}], "has_more": False}
        )
    )
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    result = json.loads(stripe.stripe_list_prices(active=True))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["params"]["active"] == "true"


def test_list_prices_serializes_active_false_as_lowercase_string(monkeypatch):
    """requests would otherwise serialize a bare Python bool as "True"/"False"
    in the query string, which Stripe's API does not accept."""
    mock_request = Mock(
        return_value=MockResponse(json_data={"data": [], "has_more": False})
    )
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    stripe.stripe_list_prices(active=False)

    assert mock_request.call_args.kwargs["params"]["active"] == "false"


def test_paginated_results_truncates_when_data_exceeds_limit():
    data, truncated = stripe._paginated_results(
        {"data": [{"id": f"cus_{i}"} for i in range(5)], "has_more": False}, limit=3
    )

    assert data == [{"id": "cus_0"}, {"id": "cus_1"}, {"id": "cus_2"}]
    assert truncated is True


def test_paginated_results_rejects_non_list_data():
    with pytest.raises(RuntimeError, match="non-list"):
        stripe._paginated_results({"data": "not-a-list"}, limit=3)


def test_request_raises_on_non_json_success_body(monkeypatch):
    response = Mock()
    response.status_code = 200
    response.content = b"not json"
    response.headers = {}
    response.json.side_effect = ValueError("Expecting value")
    monkeypatch.setattr(stripe.requests, "request", Mock(return_value=response))

    with pytest.raises(RuntimeError, match="non-JSON body"):
        stripe._request("GET", "/account")


def test_stripe_app_registry_requires_api_key():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    stripe_app = next(
        row for row in get_builtin_public_mcp_app_rows() if row["app_id"] == "stripe"
    )
    assert stripe_app["provider_name"] is None
    assert stripe_app["launch_config"]["required_env"] == ["STRIPE_API_KEY"]


async def test_mcp_registers_all_fourteen_tools():
    """Direct-call unit tests above exercise each tool's Python body, but
    none of them go through FastMCP's own registration/schema layer -- a
    tool whose @mcp.tool() decorator was dropped, or whose name was typo'd,
    would still pass every test above while being unreachable to an agent."""
    tools = await stripe.mcp.list_tools()

    assert {tool.name for tool in tools} == {
        "stripe_get_account_info",
        "stripe_get_balance",
        "stripe_list_customers",
        "stripe_get_customer",
        "stripe_create_customer",
        "stripe_list_charges",
        "stripe_get_charge",
        "stripe_create_refund",
        "stripe_list_payment_intents",
        "stripe_list_invoices",
        "stripe_get_invoice",
        "stripe_list_subscriptions",
        "stripe_list_products",
        "stripe_list_prices",
    }


async def test_create_customer_via_mcp_layer_parses_json_string_metadata(monkeypatch):
    """MCP tool arguments arrive as JSON over the wire; some callers send a
    dict-typed argument as a JSON-encoded string rather than a native JSON
    object (a real, observed LLM tool-calling behavior). FastMCP's schema
    validation pre-parses this before the tool body runs -- a direct Python
    call bypasses that layer entirely, so metadata's dict[str, str] contract
    is only actually exercised through mcp.call_tool, not stripe_create_
    customer(...) called directly."""
    mock_request = Mock(return_value=MockResponse(json_data={"id": "cus_1"}))
    monkeypatch.setattr(stripe.requests, "request", mock_request)

    await stripe.mcp.call_tool(
        "stripe_create_customer",
        {"name": "Acme", "metadata": '{"internal_id": "6735"}'},
    )

    assert mock_request.call_args.kwargs["data"] == [
        ("name", "Acme"),
        ("metadata[internal_id]", "6735"),
    ]
