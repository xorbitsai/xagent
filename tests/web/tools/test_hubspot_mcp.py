import json
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import hubspot


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
def _access_token(monkeypatch):
    monkeypatch.setenv("HUBSPOT_ACCESS_TOKEN", "token")


def test_headers_require_access_token(monkeypatch):
    monkeypatch.delenv("HUBSPOT_ACCESS_TOKEN")

    with pytest.raises(ValueError, match="HUBSPOT_ACCESS_TOKEN"):
        hubspot._headers()


def test_request_wraps_http_error_with_response_body(monkeypatch):
    monkeypatch.setattr(
        hubspot.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=400, text='{"message": "Property does not exist"}'
            )
        ),
    )

    with pytest.raises(RuntimeError, match="Property does not exist"):
        hubspot._request("GET", "/crm/v3/objects/contacts/1")


def test_request_returns_empty_dict_on_no_content(monkeypatch):
    monkeypatch.setattr(
        hubspot.requests,
        "request",
        Mock(return_value=MockResponse(status_code=204)),
    )

    assert hubspot._request("DELETE", "/crm/v3/objects/contacts/1") == {}


def test_request_defaults_to_the_short_timeout(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    hubspot._request("GET", "/crm/v3/objects/contacts/1")

    assert mock_request.call_args.kwargs["timeout"] == hubspot.DEFAULT_TIMEOUT_SECONDS


def test_association_listing_uses_the_longer_timeout(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"results": []}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    hubspot._list_association_ids("/crm/v3/objects/contacts/c1/associations/deals", 10)

    assert (
        mock_request.call_args.kwargs["timeout"]
        == hubspot.ASSOCIATION_LISTING_TIMEOUT_SECONDS
    )


def test_create_note_assembles_associations(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": "note-1"}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_create_note("Call summary", contact_id="c1", deal_id="d1")
    )

    assert result["status"] == "success"
    body = mock_request.call_args.kwargs["json"]
    assert body["properties"]["hs_note_body"] == "Call summary"
    assert body["associations"] == [
        {
            "to": {"id": "c1"},
            "types": [
                {"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 202}
            ],
        },
        {
            "to": {"id": "d1"},
            "types": [
                {"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 214}
            ],
        },
    ]


def test_create_note_requires_at_least_one_target(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_create_note("orphan note"))

    assert result["status"] == "error"
    assert "At least one of" in result["message"]
    mock_request.assert_not_called()


def test_get_contact_notes_paginates_and_reports_has_more(monkeypatch):
    association_pages = {
        None: {
            "results": [{"id": f"n{i}"} for i in range(3)],
            "paging": {"next": {"after": "cursor-1"}},
        },
        "cursor-1": {
            "results": [{"id": "n3"}, {"id": "n4"}],
            "paging": {"next": {"after": "cursor-2"}},
        },
    }
    association_calls = []

    def fake_request(method, url, headers, params, json, timeout):
        if "/associations/notes" in url:
            association_calls.append(params)
            return MockResponse(json_data=association_pages[params.get("after")])
        assert url.endswith("/crm/v3/objects/notes/batch/read")
        return MockResponse(
            json_data={
                "results": [
                    {"id": item["id"], "properties": {"hs_note_body": "x"}}
                    for item in json["inputs"]
                ]
            }
        )

    monkeypatch.setattr(hubspot.requests, "request", Mock(side_effect=fake_request))

    result = json.loads(hubspot.hubspot_get_contact_notes("c1", limit=5))

    assert result["status"] == "success"
    assert [note["id"] for note in result["notes"]] == ["n0", "n1", "n2", "n3", "n4"]
    assert result["has_more"] is True
    assert association_calls == [{"limit": 5}, {"limit": 2, "after": "cursor-1"}]


def test_get_contact_deals_single_page_has_no_more(monkeypatch):
    def fake_request(method, url, headers, params, json, timeout):
        if "/associations/deals" in url:
            return MockResponse(json_data={"results": [{"id": "d1"}, {"id": "d2"}]})
        assert url.endswith("/crm/v3/objects/deals/batch/read")
        return MockResponse(
            json_data={
                "results": [
                    {"id": item["id"], "properties": {"dealname": "Deal"}}
                    for item in json["inputs"]
                ]
            }
        )

    monkeypatch.setattr(hubspot.requests, "request", Mock(side_effect=fake_request))

    result = json.loads(hubspot.hubspot_get_contact_deals("c1"))

    assert result["status"] == "success"
    assert [deal["id"] for deal in result["deals"]] == ["d1", "d2"]
    assert result["has_more"] is False


def test_association_listing_stops_on_empty_page_with_cursor(monkeypatch):
    """A page with no results but a next cursor must terminate, not loop."""
    request_mock = Mock(
        return_value=MockResponse(
            json_data={"results": [], "paging": {"next": {"after": "cursor-1"}}}
        )
    )
    monkeypatch.setattr(hubspot.requests, "request", request_mock)

    result = json.loads(hubspot.hubspot_get_contact_deals("c1"))

    assert result == {"status": "success", "deals": [], "has_more": True}
    assert request_mock.call_count == 1


def test_get_contact_deals_empty_returns_no_more(monkeypatch):
    monkeypatch.setattr(
        hubspot.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"results": []})),
    )

    result = json.loads(hubspot.hubspot_get_contact_deals("c1"))

    assert result == {"status": "success", "deals": [], "has_more": False}


def test_list_forms_returns_raw_results(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"results": [{"id": "f1", "name": "Contact Us"}]}
        )
    )
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_list_forms(limit=5))

    assert result == {
        "status": "success",
        "forms": [{"id": "f1", "name": "Contact Us"}],
        "has_more": False,
    }
    assert mock_request.call_args.kwargs["url"].endswith("/marketing/v3/forms")
    assert mock_request.call_args.kwargs["params"] == {"limit": 5}


def test_get_form_submissions_reports_has_more(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "results": [{"submittedAt": 1, "values": []}],
                "paging": {"next": {"after": "cursor-1"}},
            }
        )
    )
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_get_form_submissions("form-1", limit=10))

    assert result["status"] == "success"
    assert result["has_more"] is True
    assert mock_request.call_args.kwargs["url"].endswith(
        "/form-integrations/v1/submissions/forms/form-1"
    )


def test_get_analytics_report_rejects_invalid_breakdown(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_get_analytics_report("sources", "yearly"))

    assert result["status"] == "error"
    assert "breakdown must be one of" in result["message"]
    mock_request.assert_not_called()


def test_get_analytics_report_passes_date_range(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"totals": {}}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_get_analytics_report(
            "sources", "day", start_date="20260101", end_date="20260131"
        )
    )

    assert result == {"status": "success", "report": {"totals": {}}}
    assert mock_request.call_args.kwargs["url"].endswith(
        "/analytics/v2/reports/sources/day"
    )
    assert mock_request.call_args.kwargs["params"] == {
        "start": "20260101",
        "end": "20260131",
    }


def test_list_marketing_emails_returns_raw_results(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"results": [{"id": "e1"}]})
    )
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_list_marketing_emails())

    assert result == {
        "status": "success",
        "emails": [{"id": "e1"}],
        "has_more": False,
    }
    assert mock_request.call_args.kwargs["url"].endswith("/marketing/v3/emails")


def test_get_marketing_email_statistics_passes_email_ids(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"results": [{"id": "e1"}]})
    )
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_get_marketing_email_statistics("e1, e2"))

    assert result == {"status": "success", "statistics": [{"id": "e1"}]}
    assert mock_request.call_args.kwargs["params"] == {"emailIds": ["e1", "e2"]}


def test_list_campaigns_returns_raw_results(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"results": [{"id": "c1"}]})
    )
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_list_campaigns())

    assert result == {
        "status": "success",
        "campaigns": [{"id": "c1"}],
        "has_more": False,
    }
    assert mock_request.call_args.kwargs["url"].endswith("/marketing/v3/campaigns")


def test_get_campaign_metrics_passes_date_range(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"sessions": 10}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_get_campaign_metrics(
            "campaign-1", start_date="2026-01-01", end_date="2026-01-31"
        )
    )

    assert result == {"status": "success", "metrics": {"sessions": 10}}
    assert mock_request.call_args.kwargs["url"].endswith(
        "/marketing/v3/campaigns/campaign-1/reports/metrics"
    )
    assert mock_request.call_args.kwargs["params"] == {
        "startDate": "2026-01-01",
        "endDate": "2026-01-31",
    }
