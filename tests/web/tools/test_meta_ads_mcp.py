import json
from unittest.mock import Mock

import requests

from xagent.web.tools.mcp import meta_ads


class MockResponse:
    def __init__(self, json_data=None, text="", status_code=200):
        self._json_data = json_data or {}
        self.text = text
        self.status_code = status_code
        self.content = text.encode("utf-8") if text else b"{}"

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error


def _payload(result: str):
    return json.loads(result)


def test_auth_status_uses_injected_meta_token(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    mock_request = Mock(return_value=MockResponse({"id": "user-1", "name": "Alice"}))
    monkeypatch.setattr(meta_ads.requests, "request", mock_request)

    result = _payload(meta_ads.meta_ads_auth_status())

    assert result == {
        "status": "success",
        "authenticated": True,
        "user": {"id": "user-1", "name": "Alice"},
    }
    mock_request.assert_called_once_with(
        method="GET",
        url="https://graph.facebook.com/v25.0/me",
        headers={
            "Authorization": "Bearer user-token",
            "Accept": "application/json",
        },
        params={"fields": "id,name"},
        data=None,
        timeout=30,
    )


def test_list_ad_accounts_returns_data_and_next_link(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    mock_request = Mock(
        return_value=MockResponse(
            {
                "data": [{"id": "act_123", "name": "Launch Ads"}],
                "paging": {"next": "https://graph.facebook.com/next"},
            }
        )
    )
    monkeypatch.setattr(meta_ads.requests, "request", mock_request)

    result = _payload(meta_ads.meta_ads_list_ad_accounts(limit=5))

    assert result == {
        "status": "success",
        "ad_accounts": [{"id": "act_123", "name": "Launch Ads"}],
        "next_link": "https://graph.facebook.com/next",
    }
    assert mock_request.call_args.kwargs["url"] == (
        "https://graph.facebook.com/v25.0/me/adaccounts"
    )
    assert mock_request.call_args.kwargs["params"] == {
        "fields": meta_ads.AD_ACCOUNT_FIELDS,
        "limit": 5,
    }


def test_get_ad_account_accepts_id_without_act_prefix(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    mock_request = Mock(return_value=MockResponse({"id": "act_123", "name": "Launch"}))
    monkeypatch.setattr(meta_ads.requests, "request", mock_request)

    result = _payload(meta_ads.meta_ads_get_ad_account("123"))

    assert result == {
        "status": "success",
        "ad_account": {"id": "act_123", "name": "Launch"},
    }
    assert mock_request.call_args.kwargs["url"] == (
        "https://graph.facebook.com/v25.0/act_123"
    )


def test_get_ad_account_rejects_non_numeric_id(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    mock_request = Mock()
    monkeypatch.setattr(meta_ads.requests, "request", mock_request)

    result = _payload(meta_ads.meta_ads_get_ad_account("act_123; DROP TABLE"))

    assert result == {
        "status": "error",
        "message": "ad_account_id must be numeric, optionally prefixed with 'act_'",
    }
    mock_request.assert_not_called()


def test_get_ad_account_rejects_non_ascii_digits(monkeypatch):
    """Bare \\d in a regex matches any Unicode decimal digit, not just
    ASCII 0-9 -- these Arabic-Indic digits must not slip past validation."""
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    mock_request = Mock()
    monkeypatch.setattr(meta_ads.requests, "request", mock_request)

    result = _payload(meta_ads.meta_ads_get_ad_account("act_١٢٣"))

    assert result == {
        "status": "error",
        "message": "ad_account_id must be numeric, optionally prefixed with 'act_'",
    }
    mock_request.assert_not_called()


def test_list_campaigns_builds_expected_path(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    mock_request = Mock(return_value=MockResponse({"data": [{"id": "campaign-1"}]}))
    monkeypatch.setattr(meta_ads.requests, "request", mock_request)

    result = _payload(meta_ads.meta_ads_list_campaigns("act_123"))

    assert result == {
        "status": "success",
        "campaigns": [{"id": "campaign-1"}],
        "next_link": None,
    }
    assert mock_request.call_args.kwargs["url"] == (
        "https://graph.facebook.com/v25.0/act_123/campaigns"
    )
    assert mock_request.call_args.kwargs["params"]["fields"] == meta_ads.CAMPAIGN_FIELDS


def test_list_ad_sets_applies_campaign_filter(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    mock_request = Mock(return_value=MockResponse({"data": []}))
    monkeypatch.setattr(meta_ads.requests, "request", mock_request)

    _payload(meta_ads.meta_ads_list_ad_sets("act_123", campaign_id="1001"))

    params = mock_request.call_args.kwargs["params"]
    assert json.loads(params["filtering"]) == [
        {"field": "campaign.id", "operator": "EQUAL", "value": "1001"}
    ]


def test_list_ad_sets_rejects_non_numeric_campaign_id(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    mock_request = Mock()
    monkeypatch.setattr(meta_ads.requests, "request", mock_request)

    result = _payload(
        meta_ads.meta_ads_list_ad_sets("act_123", campaign_id="1; DROP TABLE")
    )

    assert result == {"status": "error", "message": "campaign_id must be numeric"}
    mock_request.assert_not_called()


def test_list_ads_applies_campaign_and_ad_set_filters(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    mock_request = Mock(return_value=MockResponse({"data": []}))
    monkeypatch.setattr(meta_ads.requests, "request", mock_request)

    _payload(
        meta_ads.meta_ads_list_ads("act_123", campaign_id="1001", ad_set_id="2002")
    )

    params = mock_request.call_args.kwargs["params"]
    assert json.loads(params["filtering"]) == [
        {"field": "campaign.id", "operator": "EQUAL", "value": "1001"},
        {"field": "adset.id", "operator": "EQUAL", "value": "2002"},
    ]


def test_list_ads_rejects_non_numeric_ad_set_id(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    mock_request = Mock()
    monkeypatch.setattr(meta_ads.requests, "request", mock_request)

    result = _payload(meta_ads.meta_ads_list_ads("act_123", ad_set_id="not-numeric"))

    assert result == {"status": "error", "message": "ad_set_id must be numeric"}
    mock_request.assert_not_called()


def test_get_insights_defaults_to_date_preset(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    mock_request = Mock(return_value=MockResponse({"data": [{"impressions": "100"}]}))
    monkeypatch.setattr(meta_ads.requests, "request", mock_request)

    result = _payload(meta_ads.meta_ads_get_insights("1001"))

    assert result == {
        "status": "success",
        "insights": [{"impressions": "100"}],
        "next_link": None,
    }
    assert mock_request.call_args.kwargs["url"] == (
        "https://graph.facebook.com/v25.0/1001/insights"
    )
    params = mock_request.call_args.kwargs["params"]
    assert params["date_preset"] == "last_30d"
    assert params["fields"] == meta_ads.INSIGHTS_FIELDS
    assert "level" not in params


def test_get_insights_accepts_ad_account_id_and_level(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    mock_request = Mock(return_value=MockResponse({"data": []}))
    monkeypatch.setattr(meta_ads.requests, "request", mock_request)

    _payload(meta_ads.meta_ads_get_insights("act_123", level="campaign"))

    assert mock_request.call_args.kwargs["url"] == (
        "https://graph.facebook.com/v25.0/act_123/insights"
    )
    assert mock_request.call_args.kwargs["params"]["level"] == "campaign"


def test_get_insights_uses_explicit_time_range(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    mock_request = Mock(return_value=MockResponse({"data": []}))
    monkeypatch.setattr(meta_ads.requests, "request", mock_request)

    _payload(
        meta_ads.meta_ads_get_insights("1001", since="2026-01-01", until="2026-01-31")
    )

    params = mock_request.call_args.kwargs["params"]
    assert json.loads(params["time_range"]) == {
        "since": "2026-01-01",
        "until": "2026-01-31",
    }
    assert "date_preset" not in params


def test_get_insights_accepts_data_maximum_date_preset(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    mock_request = Mock(return_value=MockResponse({"data": []}))
    monkeypatch.setattr(meta_ads.requests, "request", mock_request)

    result = _payload(
        meta_ads.meta_ads_get_insights("1001", date_preset="data_maximum")
    )

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["params"]["date_preset"] == "data_maximum"


def test_get_insights_rejects_invalid_date_preset(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    mock_request = Mock()
    monkeypatch.setattr(meta_ads.requests, "request", mock_request)

    result = _payload(
        meta_ads.meta_ads_get_insights("1001", date_preset="not_a_preset")
    )

    assert result["status"] == "error"
    assert "date_preset must be one of" in result["message"]
    mock_request.assert_not_called()


def test_get_insights_rejects_partial_time_range(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    mock_request = Mock()
    monkeypatch.setattr(meta_ads.requests, "request", mock_request)

    result = _payload(meta_ads.meta_ads_get_insights("1001", since="2026-01-01"))

    assert result == {
        "status": "error",
        "message": "since and until must be provided together",
    }
    mock_request.assert_not_called()


def test_get_insights_rejects_malformed_date(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    mock_request = Mock()
    monkeypatch.setattr(meta_ads.requests, "request", mock_request)

    result = _payload(
        meta_ads.meta_ads_get_insights("1001", since="01/01/2026", until="2026-01-31")
    )

    assert result == {
        "status": "error",
        "message": "since/until must be in YYYY-MM-DD format",
    }
    mock_request.assert_not_called()


def test_get_insights_rejects_invalid_level(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    mock_request = Mock()
    monkeypatch.setattr(meta_ads.requests, "request", mock_request)

    result = _payload(meta_ads.meta_ads_get_insights("1001", level="bogus"))

    assert result["status"] == "error"
    assert "level must be one of" in result["message"]
    mock_request.assert_not_called()


def test_get_insights_rejects_malformed_object_id(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    mock_request = Mock()
    monkeypatch.setattr(meta_ads.requests, "request", mock_request)

    result = _payload(meta_ads.meta_ads_get_insights("campaign-1/insights"))

    assert result == {
        "status": "error",
        "message": "object_id must be 'act_<digits>' or a numeric id",
    }
    mock_request.assert_not_called()


def test_graph_api_errors_are_structured_and_redact_tokens(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    monkeypatch.setattr(
        meta_ads.requests,
        "request",
        Mock(
            return_value=MockResponse(
                {"error": {"message": "Bad OAuth token user-token", "code": 190}},
                text='{"error":{"message":"Bad OAuth token user-token","code":190}}',
                status_code=400,
            )
        ),
    )

    result = _payload(meta_ads.meta_ads_auth_status())

    assert result["status"] == "error"
    assert "user-token" not in json.dumps(result)
    assert "[redacted]" in json.dumps(result)
    assert result["details"]["error"]["code"] == 190


def test_graph_api_error_logging_redacts_the_token(monkeypatch, caplog):
    """The Graph API can echo the access token back inside an error message
    (as above); the logged line must be redacted the same way the JSON
    response is, not just the response."""
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    monkeypatch.setattr(
        meta_ads.requests,
        "request",
        Mock(
            return_value=MockResponse(
                {"error": {"message": "Bad OAuth token user-token", "code": 190}},
                text='{"error":{"message":"Bad OAuth token user-token","code":190}}',
                status_code=400,
            )
        ),
    )

    with caplog.at_level("ERROR", logger="meta-ads-mcp"):
        meta_ads.meta_ads_auth_status()

    assert "user-token" not in caplog.text
    assert "[redacted]" in caplog.text


def test_generic_exception_logging_also_redacts_the_token(monkeypatch, caplog):
    """_log_error covers the generic `except Exception` branch too, not just
    GraphAPIError -- an unexpected failure that happens to embed the token
    in its message must not leak it into logs either."""
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
    monkeypatch.setattr(
        meta_ads.requests,
        "request",
        Mock(side_effect=RuntimeError("boom: token was user-token")),
    )

    with caplog.at_level("ERROR", logger="meta-ads-mcp"):
        result = _payload(meta_ads.meta_ads_auth_status())

    assert result["status"] == "error"
    assert "user-token" not in caplog.text
    assert "[redacted]" in caplog.text
