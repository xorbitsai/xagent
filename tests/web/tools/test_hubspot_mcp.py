import json
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import hubspot
from xagent.web.tools.mcp import utils as mcp_utils


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


def test_url_path_id_percent_encodes_path_and_query_metacharacters():
    """Percent-encoding, not a blocklist, is what actually prevents an id
    from escaping its intended URL path segment."""
    assert (
        hubspot._url_path_id("x?limit=1&foo=/reports/metrics", "campaign_id")
        == "x%3Flimit%3D1%26foo%3D%2Freports%2Fmetrics"
    )


def test_url_path_id_rejects_whitespace_padded_value():
    with pytest.raises(ValueError, match="contact_id"):
        hubspot._url_path_id(" c1 ", "contact_id")


def test_get_contact_encodes_and_fetches_by_id(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": "c1"}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_get_contact("c1/../2"))

    assert result == {"status": "success", "contact": {"id": "c1"}}
    assert mock_request.call_args.kwargs["url"].endswith(
        "/crm/v3/objects/contacts/c1%2F..%2F2"
    )


def test_get_contact_rejects_whitespace_padded_contact_id(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_get_contact(" c1 "))

    assert result["status"] == "error"
    assert "contact_id" in result["message"]
    mock_request.assert_not_called()


def test_update_company_encodes_company_id(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": "co1"}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_update_company("co 1/2", '{"name": "Acme"}'))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"].endswith(
        "/crm/v3/objects/companies/co%201%2F2"
    )


def test_update_company_rejects_whitespace_padded_company_id(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_update_company(" co1 ", '{"name": "Acme"}'))

    assert result["status"] == "error"
    assert "company_id" in result["message"]
    mock_request.assert_not_called()


def test_get_contact_deals_rejects_whitespace_padded_contact_id(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_get_contact_deals(" c1 "))

    assert result["status"] == "error"
    assert "contact_id" in result["message"]
    mock_request.assert_not_called()


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

    assert result == {
        "status": "success",
        "deals": [],
        "has_more": True,
        "after": "cursor-1",
        "truncated": False,
        "missing_deal_ids": [],
    }
    assert request_mock.call_count == 1


def test_get_contact_deals_empty_returns_no_more(monkeypatch):
    monkeypatch.setattr(
        hubspot.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"results": []})),
    )

    result = json.loads(hubspot.hubspot_get_contact_deals("c1"))

    assert result == {
        "status": "success",
        "deals": [],
        "has_more": False,
        "after": None,
        "truncated": False,
        "missing_deal_ids": [],
    }


def test_get_company_deals_single_page_has_no_more(monkeypatch):
    association_urls = []

    def fake_request(method, url, headers, params, json, timeout):
        if "/associations/deals" in url:
            association_urls.append(url)
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

    result = json.loads(hubspot.hubspot_get_company_deals("co1"))

    assert result["status"] == "success"
    assert [deal["id"] for deal in result["deals"]] == ["d1", "d2"]
    assert result["has_more"] is False
    assert association_urls == [
        "https://api.hubapi.com/crm/v3/objects/companies/co1/associations/deals"
    ]


def test_get_company_deals_empty_returns_no_more(monkeypatch):
    monkeypatch.setattr(
        hubspot.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"results": []})),
    )

    result = json.loads(hubspot.hubspot_get_company_deals("co1"))

    assert result == {
        "status": "success",
        "deals": [],
        "has_more": False,
        "after": None,
        "truncated": False,
        "missing_deal_ids": [],
    }


def test_get_company_deals_returns_and_accepts_pagination_cursor(monkeypatch):
    deal_ids = ["d0", "d1", "d2", "d3"]
    association_params = []

    def fake_request(method, url, headers, params, json, timeout):
        if "/associations/deals" in url:
            association_params.append(params)
            start = int(params.get("after", "0"))
            end = min(start + params["limit"], len(deal_ids))
            page = {"results": [{"id": deal_id} for deal_id in deal_ids[start:end]]}
            if end < len(deal_ids):
                page["paging"] = {"next": {"after": str(end)}}
            return MockResponse(json_data=page)
        return MockResponse(
            json_data={
                "results": [
                    {"id": item["id"], "properties": {}} for item in json["inputs"]
                ]
            }
        )

    monkeypatch.setattr(hubspot.requests, "request", Mock(side_effect=fake_request))

    first = json.loads(hubspot.hubspot_get_company_deals("co1", limit=2))
    second = json.loads(
        hubspot.hubspot_get_company_deals("co1", limit=2, after=first["after"])
    )

    assert [deal["id"] for deal in first["deals"]] == ["d0", "d1"]
    assert first["after"] == "2"
    assert first["has_more"] is True
    assert [deal["id"] for deal in second["deals"]] == ["d2", "d3"]
    assert second["after"] is None
    assert second["has_more"] is False
    assert association_params == [{"limit": 2}, {"limit": 2, "after": "2"}]


@pytest.mark.parametrize(
    "tool, object_id",
    [
        (hubspot.hubspot_get_contact_deals, "c1"),
        (hubspot.hubspot_get_company_deals, "co1"),
    ],
)
def test_get_associated_deals_clamps_limit_to_valid_range(monkeypatch, tool, object_id):
    request_mock = Mock(return_value=MockResponse(json_data={"results": []}))
    monkeypatch.setattr(hubspot.requests, "request", request_mock)

    tool(object_id, limit=0)
    assert request_mock.call_args.kwargs["params"]["limit"] == 1

    tool(object_id, limit=1000)
    assert request_mock.call_args.kwargs["params"]["limit"] == 100


@pytest.mark.parametrize(
    "tool, object_id",
    [
        (hubspot.hubspot_get_contact_deals, "c1"),
        (hubspot.hubspot_get_company_deals, "co1"),
    ],
)
def test_get_associated_deals_wraps_request_errors(monkeypatch, tool, object_id):
    monkeypatch.setattr(
        hubspot.requests,
        "request",
        Mock(return_value=MockResponse(status_code=500, text="boom")),
    )

    result = json.loads(tool(object_id))

    assert result["status"] == "error"
    assert "boom" in result["message"]


def test_get_company_deals_rejects_whitespace_padded_company_id(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_get_company_deals(" co1 "))

    assert result["status"] == "error"
    assert "company_id" in result["message"]
    mock_request.assert_not_called()


def test_get_associated_deals_requests_authoritative_closed_state_flags(monkeypatch):
    """dealstage/pipeline are opaque IDs and closedate is set on open deals
    too, so a caller needs hs_is_closed(_won|_lost) to tell open from closed
    without guessing at stage semantics.
    """

    def fake_request(method, url, headers, params, json, timeout):
        if "/associations/deals" in url:
            return MockResponse(json_data={"results": [{"id": "d1"}]})
        assert url.endswith("/crm/v3/objects/deals/batch/read")
        requested_properties.append(json["properties"])
        return MockResponse(json_data={"results": [{"id": "d1", "properties": {}}]})

    requested_properties = []
    monkeypatch.setattr(hubspot.requests, "request", Mock(side_effect=fake_request))

    contact_result = json.loads(hubspot.hubspot_get_contact_deals("c1"))
    company_result = json.loads(hubspot.hubspot_get_company_deals("co1"))

    assert contact_result["status"] == "success"
    assert company_result["status"] == "success"
    assert len(requested_properties) == 2
    for properties in requested_properties:
        assert "hs_is_closed" in properties
        assert "hs_is_closed_won" in properties
        assert "hs_is_closed_lost" in properties


def test_get_associated_deals_reports_ids_batch_read_silently_dropped(monkeypatch):
    """A partial batch-read (e.g. an archived deal) must be surfaced, not
    returned as if every requested id came back.
    """

    def fake_request(method, url, headers, params, json, timeout):
        if "/associations/deals" in url:
            return MockResponse(json_data={"results": [{"id": "d1"}, {"id": "d2"}]})
        assert url.endswith("/crm/v3/objects/deals/batch/read")
        # HubSpot dropped d2 from the batch-read response entirely.
        return MockResponse(json_data={"results": [{"id": "d1", "properties": {}}]})

    monkeypatch.setattr(hubspot.requests, "request", Mock(side_effect=fake_request))

    result = json.loads(hubspot.hubspot_get_contact_deals("c1"))

    assert result["status"] == "success"
    assert [deal["id"] for deal in result["deals"]] == ["d1"]
    assert result["missing_deal_ids"] == ["d2"]


def test_get_associated_deals_halves_response_past_output_limit(monkeypatch):
    """A large enough deal list must be halved rather than returned whole
    and hard-truncated into invalid JSON by the platform's output filter.
    """

    def fake_request(method, url, headers, params, json, timeout):
        if "/associations/deals" in url:
            return MockResponse(
                json_data={"results": [{"id": f"d{i}"} for i in range(8)]}
            )
        assert url.endswith("/crm/v3/objects/deals/batch/read")
        return MockResponse(
            json_data={
                "results": [
                    {"id": item["id"], "properties": {"dealname": "x" * 100}}
                    for item in json["inputs"]
                ]
            }
        )

    monkeypatch.setattr(hubspot.requests, "request", Mock(side_effect=fake_request))
    monkeypatch.setattr(hubspot, "get_tool_max_output_length", lambda: 400)

    response = hubspot.hubspot_get_contact_deals("c1", limit=8)
    result = json.loads(response)

    assert len(response) <= 400
    assert result["status"] == "success"
    assert 0 < len(result["deals"]) < 8
    assert result["truncated"] is True
    assert result["has_more"] is True
    assert result["after"] is None
    assert result["missing_deal_ids"] == []


def test_get_associated_deals_reports_input_after_when_deals_are_trimmed(
    monkeypatch,
):
    """A trimmed page must be retried from its input cursor."""

    def fake_request(method, url, headers, params, json, timeout):
        if "/associations/deals" in url:
            assert params["after"] == "caller-cursor"
            return MockResponse(
                json_data={
                    "results": [{"id": f"d{i}"} for i in range(4)],
                    "paging": {"next": {"after": "server-next-cursor"}},
                }
            )
        return MockResponse(
            json_data={
                "results": [
                    {"id": item["id"], "properties": {"dealname": "x" * 200}}
                    for item in json["inputs"]
                ]
            }
        )

    monkeypatch.setattr(hubspot.requests, "request", Mock(side_effect=fake_request))
    monkeypatch.setattr(hubspot, "get_tool_max_output_length", lambda: 500)

    result = json.loads(
        hubspot.hubspot_get_contact_deals("c1", limit=4, after="caller-cursor")
    )

    assert result["truncated"] is True
    assert 0 < len(result["deals"]) < 4
    assert result["has_more"] is True
    assert result["after"] == "caller-cursor"


def test_get_associated_deals_prioritizes_shrinking_missing_ids_over_real_deals(
    monkeypatch,
):
    """When both deals and missing_deal_ids are populated and halving is
    needed, missing_deal_ids (a diagnostic list of ids HubSpot didn't
    return) must shrink before real, successfully-fetched deal data is
    thrown away.
    """

    def fake_request(method, url, headers, params, json, timeout):
        if "/associations/deals" in url:
            found = [{"id": f"d{i}"} for i in range(5)]
            missing = [{"id": f"missing-deal-id-{i:03d}"} for i in range(20)]
            return MockResponse(json_data={"results": found + missing})
        assert url.endswith("/crm/v3/objects/deals/batch/read")
        # Only the 5 "found" ids actually come back; the 20 "missing" ones
        # were requested but silently dropped by the batch-read.
        return MockResponse(
            json_data={
                "results": [
                    {"id": item["id"], "properties": {}}
                    for item in json["inputs"]
                    if not item["id"].startswith("missing-deal-id-")
                ]
            }
        )

    monkeypatch.setattr(hubspot.requests, "request", Mock(side_effect=fake_request))
    monkeypatch.setattr(hubspot, "get_tool_max_output_length", lambda: 300)

    response = hubspot.hubspot_get_contact_deals("c1", limit=25)
    result = json.loads(response)

    assert len(response) <= 300
    assert result["status"] == "success"
    assert len(result["deals"]) == 5
    assert 0 < len(result["missing_deal_ids"]) < 20
    assert result["truncated"] is True


def test_get_associated_deals_halves_missing_deal_ids_past_output_limit(monkeypatch):
    """missing_deal_ids must also be halved when it alone keeps the response
    over the output limit - halving only `deals` isn't enough once `deals`
    is already empty.
    """

    def fake_request(method, url, headers, params, json, timeout):
        if "/associations/deals" in url:
            return MockResponse(
                json_data={"results": [{"id": f"deal-id-{i:03d}"} for i in range(8)]}
            )
        assert url.endswith("/crm/v3/objects/deals/batch/read")
        # HubSpot dropped every requested deal from the batch-read.
        return MockResponse(json_data={"results": []})

    monkeypatch.setattr(hubspot.requests, "request", Mock(side_effect=fake_request))
    monkeypatch.setattr(hubspot, "get_tool_max_output_length", lambda: 140)

    response = hubspot.hubspot_get_contact_deals("c1", limit=8)
    result = json.loads(response)

    assert len(response) <= 140
    assert result["status"] == "success"
    assert result["deals"] == []
    assert 0 < len(result["missing_deal_ids"]) < 8
    assert result["truncated"] is True
    assert result["has_more"] is False
    assert result["after"] is None


def test_get_associated_deals_explains_when_everything_collapses(monkeypatch):
    """Mirrors _paged_list: when halving empties deals and missing_deal_ids
    and the response still doesn't fit, say retrying won't help rather than
    returning an empty result indistinguishable from "nothing exists".
    """

    def fake_request(method, url, headers, params, json, timeout):
        if "/associations/deals" in url:
            return MockResponse(json_data={"results": [{"id": "d1"}]})
        assert url.endswith("/crm/v3/objects/deals/batch/read")
        return MockResponse(
            json_data={
                "results": [{"id": "d1", "properties": {"dealname": "x" * 1000}}]
            }
        )

    monkeypatch.setattr(hubspot.requests, "request", Mock(side_effect=fake_request))
    monkeypatch.setattr(hubspot, "get_tool_max_output_length", lambda: 450)

    response = hubspot.hubspot_get_contact_deals("c1")
    result = json.loads(response)

    assert len(response) <= 450
    assert result["status"] == "success"
    assert result["deals"] == []
    assert result["truncated"] is True
    assert "too large to fit" in result["message"]


def test_get_associated_deals_drops_message_when_it_alone_exceeds_the_limit(
    monkeypatch,
):
    """Mirrors _paged_list's equivalent test: when even the "everything
    collapsed" explanatory message wouldn't fit, drop it rather than
    exceeding the limit anyway.
    """

    def fake_request(method, url, headers, params, json, timeout):
        if "/associations/deals" in url:
            return MockResponse(json_data={"results": [{"id": "d1"}]})
        assert url.endswith("/crm/v3/objects/deals/batch/read")
        return MockResponse(
            json_data={"results": [{"id": "d1", "properties": {"dealname": "x" * 200}}]}
        )

    monkeypatch.setattr(hubspot.requests, "request", Mock(side_effect=fake_request))
    monkeypatch.setattr(hubspot, "get_tool_max_output_length", lambda: 150)

    response = hubspot.hubspot_get_contact_deals("c1")
    result = json.loads(response)

    assert len(response) <= 150
    assert result["status"] == "success"
    assert result["deals"] == []
    assert result["truncated"] is True
    assert "message" not in result


def test_get_associated_deals_handles_association_result_missing_id(monkeypatch):
    """A malformed association item must not send a null batch-read id."""

    batch_inputs = []

    def fake_request(method, url, headers, params, json, timeout):
        if "/associations/deals" in url:
            return MockResponse(json_data={"results": [{"id": "d1"}, {}]})
        assert url.endswith("/crm/v3/objects/deals/batch/read")
        batch_inputs.extend(json["inputs"])
        return MockResponse(json_data={"results": [{"id": "d1", "properties": {}}]})

    monkeypatch.setattr(hubspot.requests, "request", Mock(side_effect=fake_request))

    result = json.loads(hubspot.hubspot_get_contact_deals("c1"))

    assert result["status"] == "success"
    assert [deal["id"] for deal in result["deals"]] == ["d1"]
    assert result["missing_deal_ids"] == []
    assert batch_inputs == [{"id": "d1"}]


def test_get_associated_deals_deduplicates_and_normalizes_ids(monkeypatch):
    batch_inputs = []

    def fake_request(method, url, headers, params, json, timeout):
        if "/associations/deals" in url:
            return MockResponse(
                json_data={"results": [{"id": 1}, {"id": "1"}, {"id": "2"}]}
            )
        batch_inputs.extend(json["inputs"])
        return MockResponse(
            json_data={
                "results": [
                    {"id": item["id"], "properties": {}} for item in json["inputs"]
                ]
            }
        )

    monkeypatch.setattr(hubspot.requests, "request", Mock(side_effect=fake_request))

    result = json.loads(hubspot.hubspot_get_company_deals("co1"))

    assert result["status"] == "success"
    assert [deal["id"] for deal in result["deals"]] == ["1", "2"]
    assert result["missing_deal_ids"] == []
    assert batch_inputs == [{"id": "1"}, {"id": "2"}]


def test_create_deal_without_contact_id_sends_no_associations(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": "d1"}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_create_deal('{"dealname": "Acme - Onboarding"}')
    )

    assert result == {"status": "success", "deal": {"id": "d1"}}
    body = mock_request.call_args.kwargs["json"]
    assert body == {"properties": {"dealname": "Acme - Onboarding"}}
    assert mock_request.call_args.kwargs["url"].endswith("/crm/v3/objects/deals")
    assert mock_request.call_args.kwargs["method"] == "POST"


def test_create_deal_with_contact_id_assembles_association(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": "d1"}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_create_deal(
            '{"dealname": "Acme - Onboarding"}', contact_id="c1"
        )
    )

    assert result["status"] == "success"
    body = mock_request.call_args.kwargs["json"]
    assert body["associations"] == [
        {
            "to": {"id": "c1"},
            "types": [
                {"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 3}
            ],
        }
    ]


def test_create_deal_sends_contact_id_verbatim_not_url_encoded(monkeypatch):
    """contact_id is placed in the JSON request body, not a URL path -
    percent-encoding it (as url_path_id would) sends HubSpot a mangled id
    that doesn't match any real contact instead of the real one."""
    mock_request = Mock(return_value=MockResponse(json_data={"id": "d1"}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    hubspot.hubspot_create_deal('{"dealname": "x"}', contact_id="c1/2")

    body = mock_request.call_args.kwargs["json"]
    assert body["associations"][0]["to"]["id"] == "c1/2"


def test_create_deal_rejects_whitespace_padded_contact_id(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_create_deal('{"dealname": "x"}', contact_id=" c1 ")
    )

    assert result["status"] == "error"
    assert "contact_id" in result["message"]
    mock_request.assert_not_called()


def test_create_deal_wraps_request_errors(monkeypatch):
    """Regression coverage for the exact incident this PR fixes: a HubSpot
    write returning 401 must come back as a normal {"status": "error"}
    result, not an uncaught exception - update_deal already had this test,
    create_deal did not."""
    monkeypatch.setattr(
        hubspot.requests,
        "request",
        Mock(return_value=MockResponse(status_code=401, text="Unauthorized")),
    )

    result = json.loads(hubspot.hubspot_create_deal('{"dealname": "x"}'))

    assert result["status"] == "error"
    assert "Unauthorized" in result["message"]


def test_update_deal_encodes_deal_id_and_sends_properties(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"id": "d1", "properties": {}})
    )
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_update_deal("d 1/2", '{"dealstage": "qualifiedtobuy"}')
    )

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"].endswith(
        "/crm/v3/objects/deals/d%201%2F2"
    )
    assert mock_request.call_args.kwargs["method"] == "PATCH"
    assert mock_request.call_args.kwargs["json"] == {
        "properties": {"dealstage": "qualifiedtobuy"}
    }


def test_update_deal_rejects_whitespace_padded_deal_id(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_update_deal(" d1 ", '{"dealstage": "qualifiedtobuy"}')
    )

    assert result["status"] == "error"
    assert "deal_id" in result["message"]
    mock_request.assert_not_called()


def test_update_deal_wraps_request_errors(monkeypatch):
    monkeypatch.setattr(
        hubspot.requests,
        "request",
        Mock(return_value=MockResponse(status_code=401, text="Unauthorized")),
    )

    result = json.loads(
        hubspot.hubspot_update_deal("d1", '{"dealstage": "qualifiedtobuy"}')
    )

    assert result["status"] == "error"
    assert "Unauthorized" in result["message"]


def test_list_forms_projects_summary_fields(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "results": [
                    {
                        "id": "f1",
                        "name": "Contact Us",
                        "formType": "hubspot",
                        "createdAt": "2026-01-01T00:00:00Z",
                        "updatedAt": "2026-01-02T00:00:00Z",
                        "archived": False,
                        "fieldGroups": [{"huge": "payload"}],
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_list_forms(limit=5))

    assert result == {
        "status": "success",
        "forms": [
            {
                "id": "f1",
                "name": "Contact Us",
                "formType": "hubspot",
                "createdAt": "2026-01-01T00:00:00Z",
                "updatedAt": "2026-01-02T00:00:00Z",
                "archived": False,
            }
        ],
        "truncated": False,
        "has_more": False,
        "after": None,
    }
    assert mock_request.call_args.kwargs["url"].endswith("/marketing/v3/forms")
    assert mock_request.call_args.kwargs["params"] == {"limit": 5}


def test_list_forms_caps_output_size(monkeypatch):
    big_forms = [{"id": str(i), "name": "x" * 1000} for i in range(50)]
    mock_request = Mock(return_value=MockResponse(json_data={"results": big_forms}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)
    monkeypatch.setattr(hubspot, "get_tool_max_output_length", lambda: 2000)

    result = json.loads(hubspot.hubspot_list_forms(limit=100))

    assert result["status"] == "success"
    assert result["truncated"] is True
    assert len(result["forms"]) < len(big_forms)
    assert len(json.dumps(result)) <= 2000 + 200  # last halving step can overshoot


def test_paged_list_truncates_to_empty_when_single_item_still_oversized(monkeypatch):
    """Regression for an off-by-one in the halving loop: a page that
    shrinks to exactly one item which is STILL over budget on its own must
    still truncate down to zero items with truncated=True, not silently
    return the oversized single item untouched (a `len(items) > 1` guard
    would stop shrinking right before this point)."""
    huge_form = {"id": "f1", "name": "x" * 5000}
    mock_request = Mock(return_value=MockResponse(json_data={"results": [huge_form]}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)
    monkeypatch.setattr(hubspot, "get_tool_max_output_length", lambda: 200)

    result = json.loads(hubspot.hubspot_list_forms(limit=1))

    assert result["status"] == "success"
    assert result["forms"] == []
    assert result["truncated"] is True
    assert result["has_more"] is True


def test_paged_list_explains_the_dead_end_when_collapsed_to_empty(monkeypatch):
    """has_more=true with no cursor and an empty page (the collapse case
    above) is otherwise a dead end for the caller - a smaller `limit` than
    1 doesn't exist, and there's no cursor to retry a different page with.
    An explicit message must say so instead of leaving that implicit.

    Limit is large enough to fit the message itself (unlike the sibling
    test below): this test is specifically about the message being added
    when there's room for it, not about the size-cap interaction."""
    huge_form = {"id": "f1", "name": "x" * 5000}
    mock_request = Mock(return_value=MockResponse(json_data={"results": [huge_form]}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)
    monkeypatch.setattr(hubspot, "get_tool_max_output_length", lambda: 400)

    result = json.loads(hubspot.hubspot_list_forms(limit=1))

    assert "message" in result
    assert "too large" in result["message"]


def test_paged_list_drops_message_when_it_alone_exceeds_the_limit(monkeypatch):
    """The dead-end message added above must not be appended
    unconditionally: rebuilding with it included must still respect
    max_output_length, or appending it would silently reintroduce the
    hard-truncated-into-broken-JSON failure mode this function exists to
    prevent. A limit that fits the empty-page envelope (86 chars) but not
    envelope-plus-message (339 chars) must fall back to omitting the
    message rather than exceeding the cap."""
    huge_form = {"id": "f1", "name": "x" * 5000}
    mock_request = Mock(return_value=MockResponse(json_data={"results": [huge_form]}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)
    monkeypatch.setattr(hubspot, "get_tool_max_output_length", lambda: 200)

    raw = hubspot.hubspot_list_forms(limit=1)
    result = json.loads(raw)

    assert len(raw) <= 200
    assert result["status"] == "success"
    assert result["truncated"] is True
    assert result["forms"] == []
    assert "message" not in result


def test_paged_list_reports_input_after_not_next_after_when_truncated(monkeypatch):
    """A truncated page must report ITS OWN input cursor, not the server's
    next-page cursor: the server already considers the full page consumed,
    so advancing to its next page would permanently skip whatever this
    page didn't return for space reasons."""
    forms = [{"id": str(i), "name": "x" * 200} for i in range(4)]
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "results": forms,
                "paging": {"next": {"after": "server-next-cursor"}},
            }
        )
    )
    monkeypatch.setattr(hubspot.requests, "request", mock_request)
    monkeypatch.setattr(hubspot, "get_tool_max_output_length", lambda: 500)

    result = json.loads(hubspot.hubspot_list_forms(after="caller-input-cursor"))

    assert result["truncated"] is True
    assert 0 < len(result["forms"]) < 4
    assert result["has_more"] is True
    assert result["after"] == "caller-input-cursor"


def test_list_forms_clamps_limit_to_valid_range(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"results": []}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    hubspot.hubspot_list_forms(limit=0)
    assert mock_request.call_args.kwargs["params"]["limit"] == 1

    hubspot.hubspot_list_forms(limit=1000)
    assert mock_request.call_args.kwargs["params"]["limit"] == 100


def test_list_forms_reports_has_more_and_passes_after_cursor(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "results": [{"id": "f1"}],
                "paging": {"next": {"after": "cursor-1"}},
            }
        )
    )
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_list_forms(after="cursor-0"))

    assert result["has_more"] is True
    assert result["after"] == "cursor-1"
    assert mock_request.call_args.kwargs["params"]["after"] == "cursor-0"


def test_list_forms_wraps_request_errors(monkeypatch):
    monkeypatch.setattr(
        hubspot.requests,
        "request",
        Mock(return_value=MockResponse(status_code=500, text="boom")),
    )

    result = json.loads(hubspot.hubspot_list_forms())

    assert result["status"] == "error"
    assert "boom" in result["message"]


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

    result = json.loads(
        hubspot.hubspot_get_form_submissions("form-1", limit=10, after="cursor-0")
    )

    assert result["status"] == "success"
    assert result["has_more"] is True
    assert result["after"] == "cursor-1"
    assert mock_request.call_args.kwargs["params"]["after"] == "cursor-0"
    assert mock_request.call_args.kwargs["url"].endswith(
        "/form-integrations/v1/submissions/forms/form-1"
    )


def test_get_form_submissions_clamps_limit_to_valid_range(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"results": []}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    hubspot.hubspot_get_form_submissions("form-1", limit=0)
    assert mock_request.call_args.kwargs["params"]["limit"] == 1

    hubspot.hubspot_get_form_submissions("form-1", limit=1000)
    assert mock_request.call_args.kwargs["params"]["limit"] == 50


def test_get_form_submissions_encodes_special_characters_in_form_id(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"results": []}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    hubspot.hubspot_get_form_submissions("form/1?x=1")

    assert mock_request.call_args.kwargs["url"].endswith(
        "/form-integrations/v1/submissions/forms/form%2F1%3Fx%3D1"
    )


def test_get_form_submissions_rejects_whitespace_padded_form_id(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_get_form_submissions(" form-1 "))

    assert result["status"] == "error"
    assert "form_id" in result["message"]
    mock_request.assert_not_called()


def test_get_form_submissions_wraps_request_errors(monkeypatch):
    monkeypatch.setattr(
        hubspot.requests,
        "request",
        Mock(return_value=MockResponse(status_code=500, text="boom")),
    )

    result = json.loads(hubspot.hubspot_get_form_submissions("form-1"))

    assert result["status"] == "error"
    assert "boom" in result["message"]


def test_get_analytics_report_rejects_invalid_report_type(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_get_analytics_report("not-a-real-type", "20260101", "20260131")
    )

    assert result["status"] == "error"
    assert "report_type must be one of" in result["message"]
    mock_request.assert_not_called()


# Hardcoded literal, deliberately NOT hubspot._VALID_ANALYTICS_REPORT_TYPES:
# testing against the same constant the code validates against can't catch a
# wrong entry in that constant (exactly how "landing-pages" - a non-existent
# HubSpot value - made it through an earlier version of this allowlist).
_DOCUMENTED_ANALYTICS_REPORT_TYPES = [
    "totals",
    "sessions",
    "sources",
    "geolocation",
    "utm-campaigns",
    "utm-contents",
    "utm-mediums",
    "utm-sources",
    "utm-terms",
    "event-completions",
    "forms",
    "pages",
    "social-assists",
]


def test_get_analytics_report_accepts_all_documented_report_types(monkeypatch):
    """Every documented dimension and content object type must pass the
    allowlist - a wrong entry here actively blocks valid HubSpot values."""
    mock_request = Mock(return_value=MockResponse(json_data={}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    for report_type in _DOCUMENTED_ANALYTICS_REPORT_TYPES:
        result = json.loads(
            hubspot.hubspot_get_analytics_report(report_type, "20260101", "20260131")
        )
        assert result["status"] == "success", report_type
        assert mock_request.call_args.kwargs["url"].endswith(
            f"/analytics/v2/reports/{report_type}/total"
        )


def test_get_analytics_report_normalizes_report_type_and_breakdown_case(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_get_analytics_report(
            " Sources ", "20260101", "20260131", breakdown=" Daily "
        )
    )

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"].endswith(
        "/analytics/v2/reports/sources/daily"
    )


def test_get_analytics_report_rejects_invalid_breakdown(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_get_analytics_report(
            "sources", "20260101", "20260131", breakdown="yearly"
        )
    )

    assert result["status"] == "error"
    assert "breakdown must be one of" in result["message"]
    mock_request.assert_not_called()


def test_get_analytics_report_passes_date_range(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"totals": {}}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_get_analytics_report(
            "sources", "20260101", "20260131", breakdown="daily"
        )
    )

    assert result == {
        "status": "success",
        "report": {"totals": {}},
        "truncated": False,
    }
    assert mock_request.call_args.kwargs["url"].endswith(
        "/analytics/v2/reports/sources/daily"
    )
    assert mock_request.call_args.kwargs["params"] == {
        "start": "20260101",
        "end": "20260131",
    }


def test_get_analytics_report_wraps_request_errors(monkeypatch):
    monkeypatch.setattr(
        hubspot.requests,
        "request",
        Mock(return_value=MockResponse(status_code=500, text="boom")),
    )

    result = json.loads(
        hubspot.hubspot_get_analytics_report("sources", "20260101", "20260131")
    )

    assert result["status"] == "error"
    assert "boom" in result["message"]


def test_get_analytics_report_rejects_wrong_date_format(monkeypatch):
    """The analytics API wants YYYYMMDD; the campaign-metrics API wants
    YYYY-MM-DD. Mixing them up must produce a message naming the fix, not
    an opaque upstream 400."""
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_get_analytics_report("sources", "2026-01-01", "2026-01-31")
    )

    assert result["status"] == "error"
    assert "YYYYMMDD" in result["message"]
    mock_request.assert_not_called()


def test_get_analytics_report_rejects_bad_end_date_with_valid_start_date(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_get_analytics_report("sources", "20260101", "2026-01-31")
    )

    assert result["status"] == "error"
    assert "end_date" in result["message"]
    mock_request.assert_not_called()


def test_get_analytics_report_rejects_non_calendar_date(monkeypatch):
    """ "20261332" matches a digit-position-only regex but isn't a real
    date - month 13, day 32."""
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_get_analytics_report("sources", "20261332", "20261332")
    )

    assert result["status"] == "error"
    assert "start_date" in result["message"]
    mock_request.assert_not_called()


def test_get_analytics_report_rejects_start_after_end(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_get_analytics_report("sources", "20260201", "20260101")
    )

    assert result["status"] == "error"
    assert "start_date must not be after end_date" in result["message"]
    mock_request.assert_not_called()


def test_get_analytics_report_rejects_daily_span_over_500_days(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_get_analytics_report(
            "sources", "20250101", "20260601", breakdown="daily"
        )
    )

    assert result["status"] == "error"
    assert "500 days" in result["message"]
    mock_request.assert_not_called()


def test_get_analytics_report_allows_a_wide_span_for_non_daily_breakdown(monkeypatch):
    """The 500-day cap is documented for "daily" specifically; the same
    span must not be rejected for "monthly"."""
    mock_request = Mock(return_value=MockResponse(json_data={"totals": {}}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_get_analytics_report(
            "sources", "20250101", "20260601", breakdown="monthly"
        )
    )

    assert result["status"] == "success"


def test_get_analytics_report_accepts_summarize_breakdown_variants(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"totals": {}}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_get_analytics_report(
            "sources", "20260101", "20260131", breakdown="summarize/daily"
        )
    )

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"].endswith(
        "/analytics/v2/reports/sources/summarize/daily"
    )


def test_get_analytics_report_caps_output_preserving_scalar_keys(monkeypatch):
    """The cap must shrink the CONTENTS of the largest nested list/dict
    value, not drop whole top-level keys - dropping keys could discard a
    small summary field (offset, total) on the very first step while the
    oversized field it summarizes survives untouched, and there's no
    cursor to retry with to recover it."""
    report = {
        "offset": 0,
        "total": 42,
        "breakdowns": [
            {"date": f"202601{day:02d}", "value": "x" * 200} for day in range(1, 40)
        ],
    }
    mock_request = Mock(return_value=MockResponse(json_data=report))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)
    monkeypatch.setattr(mcp_utils, "get_tool_max_output_length", lambda: 2000)

    result = json.loads(
        hubspot.hubspot_get_analytics_report(
            "sources", "20260101", "20260131", breakdown="daily"
        )
    )

    assert result["status"] == "success"
    assert result["truncated"] is True
    assert result["report"]["offset"] == 0
    assert result["report"]["total"] == 42
    assert 0 < len(result["report"]["breakdowns"]) < len(report["breakdowns"])
    assert len(json.dumps(result)) <= 2000 + 400  # last halving step can overshoot


def test_get_analytics_report_falls_back_to_dropping_keys_with_no_collections(
    monkeypatch,
):
    """When a dict has no list/dict-valued keys to shrink into - e.g. a
    handful of scalar fields with huge string values - the fallback must
    still guarantee termination by dropping whole keys, down to {}."""
    report = {"totals": "x" * 5000}
    mock_request = Mock(return_value=MockResponse(json_data=report))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)
    monkeypatch.setattr(mcp_utils, "get_tool_max_output_length", lambda: 200)

    result = json.loads(
        hubspot.hubspot_get_analytics_report("sources", "20260101", "20260131")
    )

    assert result["status"] == "success"
    assert result["truncated"] is True
    assert result["report"] == {}


def test_get_analytics_report_does_not_truncate_a_small_response(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"totals": {"visits": 1}}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_get_analytics_report("sources", "20260101", "20260131")
    )

    assert result == {
        "status": "success",
        "report": {"totals": {"visits": 1}},
        "truncated": False,
    }


def test_get_analytics_report_passes_through_a_non_dict_body(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data=[1, 2, 3]))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_get_analytics_report("sources", "20260101", "20260131")
    )

    assert result == {"status": "success", "report": [1, 2, 3], "truncated": False}


def test_list_marketing_emails_projects_summary_fields(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "results": [
                    {
                        "id": "e1",
                        "name": "Welcome",
                        "subject": "Hi there",
                        "state": "PUBLISHED",
                        "publishDate": "2026-01-01T00:00:00Z",
                        "createdAt": "2025-12-01T00:00:00Z",
                        "updatedAt": "2026-01-01T00:00:00Z",
                        "htmlBody": "<html>huge payload</html>",
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_list_marketing_emails())

    assert result == {
        "status": "success",
        "emails": [
            {
                "id": "e1",
                "name": "Welcome",
                "subject": "Hi there",
                "state": "PUBLISHED",
                "publishDate": "2026-01-01T00:00:00Z",
                "createdAt": "2025-12-01T00:00:00Z",
                "updatedAt": "2026-01-01T00:00:00Z",
            }
        ],
        "truncated": False,
        "has_more": False,
        "after": None,
    }
    assert mock_request.call_args.kwargs["url"].endswith("/marketing/v3/emails")


def test_list_marketing_emails_clamps_limit_to_valid_range(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"results": []}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    hubspot.hubspot_list_marketing_emails(limit=0)
    assert mock_request.call_args.kwargs["params"]["limit"] == 1

    hubspot.hubspot_list_marketing_emails(limit=1000)
    assert mock_request.call_args.kwargs["params"]["limit"] == 100


def test_list_marketing_emails_reports_has_more_and_after(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "results": [{"id": "e1"}],
                "paging": {"next": {"after": "cursor-1"}},
            }
        )
    )
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_list_marketing_emails(after="cursor-0"))

    assert result["has_more"] is True
    assert result["after"] == "cursor-1"
    assert mock_request.call_args.kwargs["params"]["after"] == "cursor-0"


def test_list_marketing_emails_wraps_request_errors(monkeypatch):
    monkeypatch.setattr(
        hubspot.requests,
        "request",
        Mock(return_value=MockResponse(status_code=500, text="boom")),
    )

    result = json.loads(hubspot.hubspot_list_marketing_emails())

    assert result["status"] == "error"
    assert "boom" in result["message"]


def test_get_marketing_email_statistics_passes_email_ids(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"results": [{"id": "e1"}]})
    )
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_get_marketing_email_statistics("e1,e2"))

    assert result == {
        "status": "success",
        "statistics": {"results": [{"id": "e1"}]},
        "truncated": False,
    }
    assert mock_request.call_args.kwargs["params"] == {"emailIds": ["e1", "e2"]}
    assert mock_request.call_args.kwargs["method"] == "GET"


def test_get_marketing_email_statistics_passes_through_a_non_dict_body(monkeypatch):
    """A non-dict response body has nothing generically safe to trim
    without guessing at a shape the capping helper doesn't know, so it
    must pass through unmodified with truncated=False rather than crash
    or silently corrupt the value."""
    mock_request = Mock(return_value=MockResponse(json_data=[{"id": "e1"}]))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_get_marketing_email_statistics("e1"))

    assert result == {
        "status": "success",
        "statistics": [{"id": "e1"}],
        "truncated": False,
    }


def test_get_marketing_email_statistics_rejects_bad_end_date_with_valid_start_date(
    monkeypatch,
):
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_get_marketing_email_statistics(
            "e1", start_date="2026-01-01T00:00:00Z", end_date="not-a-timestamp"
        )
    )

    assert result["status"] == "error"
    assert "end_date" in result["message"]
    mock_request.assert_not_called()


def test_get_marketing_email_statistics_passes_time_window(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    hubspot.hubspot_get_marketing_email_statistics(
        "e1", start_date="2026-01-01T00:00:00Z", end_date="2026-01-31T00:00:00Z"
    )

    assert mock_request.call_args.kwargs["params"] == {
        "emailIds": ["e1"],
        "startTimestamp": "2026-01-01T00:00:00Z",
        "endTimestamp": "2026-01-31T00:00:00Z",
    }


def test_get_marketing_email_statistics_rejects_whitespace_padded_id(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_get_marketing_email_statistics("e1, e2"))

    assert result["status"] == "error"
    assert "each email id" in result["message"]
    mock_request.assert_not_called()


def test_get_marketing_email_statistics_rejects_empty_email_ids(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_get_marketing_email_statistics(""))

    assert result["status"] == "error"
    assert "each email id" in result["message"]
    mock_request.assert_not_called()


def test_get_marketing_email_statistics_rejects_too_many_ids(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    too_many = ",".join(str(i) for i in range(101))
    result = json.loads(hubspot.hubspot_get_marketing_email_statistics(too_many))

    assert result["status"] == "error"
    assert "at most 100 ids" in result["message"]
    mock_request.assert_not_called()


def test_get_marketing_email_statistics_accepts_exactly_the_id_cap(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    exactly_100 = ",".join(str(i) for i in range(100))
    result = json.loads(hubspot.hubspot_get_marketing_email_statistics(exactly_100))

    assert result["status"] == "success"
    assert len(mock_request.call_args.kwargs["params"]["emailIds"]) == 100


def test_get_marketing_email_statistics_wraps_request_errors(monkeypatch):
    monkeypatch.setattr(
        hubspot.requests,
        "request",
        Mock(return_value=MockResponse(status_code=500, text="boom")),
    )

    result = json.loads(hubspot.hubspot_get_marketing_email_statistics("e1"))

    assert result["status"] == "error"
    assert "boom" in result["message"]


def test_list_campaigns_sends_default_properties_and_trims_results(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "results": [
                    {
                        "id": "c1",
                        "properties": {"hs_name": "Spring Launch"},
                        "extraServerField": "should be dropped",
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_list_campaigns())

    assert result == {
        "status": "success",
        "campaigns": [{"id": "c1", "properties": {"hs_name": "Spring Launch"}}],
        "truncated": False,
        "has_more": False,
        "after": None,
    }
    assert mock_request.call_args.kwargs["url"].endswith("/marketing/v3/campaigns")
    assert mock_request.call_args.kwargs["params"]["properties"] == ",".join(
        hubspot.DEFAULT_CAMPAIGN_PROPERTIES
    )


def test_list_campaigns_clamps_limit_to_valid_range(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"results": []}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    hubspot.hubspot_list_campaigns(limit=0)
    assert mock_request.call_args.kwargs["params"]["limit"] == 1

    hubspot.hubspot_list_campaigns(limit=1000)
    assert mock_request.call_args.kwargs["params"]["limit"] == 100


def test_list_campaigns_reports_has_more_and_after(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "results": [{"id": "c1", "properties": {}}],
                "paging": {"next": {"after": "cursor-1"}},
            }
        )
    )
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_list_campaigns(after="cursor-0"))

    assert result["has_more"] is True
    assert result["after"] == "cursor-1"
    assert mock_request.call_args.kwargs["params"]["after"] == "cursor-0"


def test_list_campaigns_wraps_request_errors(monkeypatch):
    monkeypatch.setattr(
        hubspot.requests,
        "request",
        Mock(return_value=MockResponse(status_code=500, text="boom")),
    )

    result = json.loads(hubspot.hubspot_list_campaigns())

    assert result["status"] == "error"
    assert "boom" in result["message"]


def test_get_campaign_metrics_passes_date_range(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"sessions": 10}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_get_campaign_metrics(
            "campaign-1", start_date="2026-01-01", end_date="2026-01-31"
        )
    )

    assert result == {
        "status": "success",
        "metrics": {"sessions": 10},
        "truncated": False,
    }
    assert mock_request.call_args.kwargs["url"].endswith(
        "/marketing/v3/campaigns/campaign-1/reports/metrics"
    )
    assert mock_request.call_args.kwargs["params"] == {
        "startDate": "2026-01-01",
        "endDate": "2026-01-31",
    }


def test_get_campaign_metrics_rejects_whitespace_padded_campaign_id(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_get_campaign_metrics(" campaign-1 "))

    assert result["status"] == "error"
    assert "campaign_id" in result["message"]
    mock_request.assert_not_called()


def test_get_campaign_metrics_rejects_wrong_date_format(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_get_campaign_metrics("campaign-1", start_date="20260101")
    )

    assert result["status"] == "error"
    assert "YYYY-MM-DD" in result["message"]
    mock_request.assert_not_called()


def test_get_campaign_metrics_rejects_bad_end_date_with_valid_start_date(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_get_campaign_metrics(
            "campaign-1", start_date="2026-01-01", end_date="20260131"
        )
    )

    assert result["status"] == "error"
    assert "end_date" in result["message"]
    mock_request.assert_not_called()


def test_get_campaign_metrics_rejects_non_calendar_date(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_get_campaign_metrics("campaign-1", start_date="2026-02-30")
    )

    assert result["status"] == "error"
    assert "start_date" in result["message"]
    mock_request.assert_not_called()


def test_get_campaign_metrics_rejects_non_zero_padded_date(monkeypatch):
    """Regression: datetime.strptime's %m/%d accept non-zero-padded values
    (and %Y can under-consume digits when the remaining format directives
    still find something to match), so a strptime-only check would parse
    "2026-1-1" or the 6-digit "202611" into a DIFFERENT, wrong date instead
    of rejecting it - not just be more lenient, but silently misparse."""
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_get_campaign_metrics("campaign-1", start_date="2026-1-1")
    )

    assert result["status"] == "error"
    assert "start_date" in result["message"]
    mock_request.assert_not_called()


def test_get_analytics_report_rejects_short_yyyymmdd(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_get_analytics_report("sources", "202611", "20260131")
    )

    assert result["status"] == "error"
    assert "start_date" in result["message"]
    mock_request.assert_not_called()


def test_get_campaign_metrics_rejects_start_after_end(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_get_campaign_metrics(
            "campaign-1", start_date="2026-02-01", end_date="2026-01-01"
        )
    )

    assert result["status"] == "error"
    assert "start_date must not be after end_date" in result["message"]
    mock_request.assert_not_called()


def test_get_campaign_metrics_passes_through_a_non_dict_body(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data=[1, 2, 3]))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_get_campaign_metrics("campaign-1"))

    assert result == {"status": "success", "metrics": [1, 2, 3], "truncated": False}


def test_list_campaigns_skips_non_dict_items(monkeypatch):
    """A non-dict item in the results (a malformed/unexpected upstream
    shape) must be dropped rather than crash the whole page, matching the
    guard _project_fields already applies for forms/emails."""
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"results": ["not-a-dict", {"id": "c1", "properties": {}}]}
        )
    )
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_list_campaigns())

    assert result["status"] == "success"
    assert result["campaigns"] == [{"id": "c1", "properties": {}}]


def test_create_note_treats_empty_string_id_as_not_provided(monkeypatch):
    """An empty-string id keeps its pre-existing meaning of "not provided"
    (an LLM caller often sends "" instead of omitting the param); only a
    non-empty id gets whitespace validation."""
    mock_request = Mock(return_value=MockResponse(json_data={"id": "note-1"}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_create_note("summary", contact_id="", deal_id="d1")
    )

    assert result["status"] == "success"
    associations = mock_request.call_args.kwargs["json"]["associations"]
    assert [a["to"]["id"] for a in associations] == ["d1"]


def test_create_note_rejects_whitespace_padded_id(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_create_note("summary", contact_id=" c1 "))

    assert result["status"] == "error"
    assert "contact_id" in result["message"]
    mock_request.assert_not_called()


def test_get_campaign_metrics_wraps_request_errors(monkeypatch):
    monkeypatch.setattr(
        hubspot.requests,
        "request",
        Mock(return_value=MockResponse(status_code=500, text="boom")),
    )

    result = json.loads(hubspot.hubspot_get_campaign_metrics("campaign-1"))

    assert result["status"] == "error"
    assert "boom" in result["message"]


def test_list_deals_hits_the_bare_deals_endpoint_with_properties(monkeypatch):
    """Regression: a request for "all deals" (not scoped to a contact or
    company) previously had no dedicated MCP tool at all, so an agent
    would fall back to the credential-less generic api_call tool and
    always get a 401 - see the 2026-09-16 incident. hubspot_list_deals
    covers that gap the same way hubspot_list_forms already covers forms."""
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "results": [
                    {
                        "id": "d1",
                        "properties": {"dealname": "Acme", "dealstage": "closedwon"},
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_list_deals(limit=50))

    assert result == {
        "status": "success",
        "deals": [
            {"id": "d1", "properties": {"dealname": "Acme", "dealstage": "closedwon"}}
        ],
        "truncated": False,
        "has_more": False,
        "after": None,
    }
    assert mock_request.call_args.kwargs["url"].endswith("/crm/v3/objects/deals")
    params = mock_request.call_args.kwargs["params"]
    assert params["limit"] == 50
    assert "dealname" in params["properties"]


def test_list_companies_hits_the_bare_companies_endpoint(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"results": [{"id": "c1", "properties": {"name": "Acme"}}]}
        )
    )
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_list_companies())

    assert result["status"] == "success"
    assert result["companies"] == [{"id": "c1", "properties": {"name": "Acme"}}]
    assert mock_request.call_args.kwargs["url"].endswith("/crm/v3/objects/companies")


def test_list_contacts_hits_the_bare_contacts_endpoint(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"results": [{"id": "p1", "properties": {"email": "a@b.com"}}]}
        )
    )
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_list_contacts())

    assert result["status"] == "success"
    assert result["contacts"] == [{"id": "p1", "properties": {"email": "a@b.com"}}]
    assert mock_request.call_args.kwargs["url"].endswith("/crm/v3/objects/contacts")


@pytest.mark.parametrize(
    "tool, id_prefix",
    [
        (hubspot.hubspot_list_deals, "d"),
        (hubspot.hubspot_list_companies, "c"),
        (hubspot.hubspot_list_contacts, "p"),
    ],
)
def test_list_passes_after_cursor_and_reports_next_one(monkeypatch, tool, id_prefix):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "results": [{"id": f"{id_prefix}1", "properties": {}}],
                "paging": {"next": {"after": "cursor-1"}},
            }
        )
    )
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(tool(after="cursor-0"))

    assert mock_request.call_args.kwargs["params"]["after"] == "cursor-0"
    assert result["has_more"] is True
    assert result["after"] == "cursor-1"


def test_list_deals_handles_an_explicit_null_results_field(monkeypatch):
    """Regression: `result.get("results", [])` only falls back to `[]` when
    the "results" key is ABSENT, not when HubSpot returns it as an explicit
    JSON null - which is still a key that exists, just with value None. That
    None used to reach _project_id_and_properties's `for item in items` and
    raise TypeError instead of the tool's usual structured error."""
    mock_request = Mock(return_value=MockResponse(json_data={"results": None}))
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_list_deals())

    assert result == {
        "status": "success",
        "deals": [],
        "truncated": False,
        "has_more": False,
        "after": None,
    }


def test_project_id_and_properties_treats_a_null_properties_value_as_empty():
    """A single item with `"properties": null` (key present, value None)
    must come back as `{}`, not None - item.get("properties", {}) only
    substitutes the default when the key is missing, not when it's None."""
    result = hubspot._project_id_and_properties(
        [{"id": "d1", "properties": None}, {"id": "d2", "properties": {"amount": "5"}}]
    )

    assert result == [
        {"id": "d1", "properties": {}},
        {"id": "d2", "properties": {"amount": "5"}},
    ]


def test_list_deals_wraps_request_errors(monkeypatch):
    monkeypatch.setattr(
        hubspot.requests,
        "request",
        Mock(return_value=MockResponse(status_code=500, text="boom")),
    )

    result = json.loads(hubspot.hubspot_list_deals())

    assert result["status"] == "error"
    assert "boom" in result["message"]


def test_search_contacts_sends_free_text_query_only(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"total": 0, "results": []})
    )
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    hubspot.hubspot_search_contacts(query="chelsea")

    body = mock_request.call_args.kwargs["json"]
    assert body["query"] == "chelsea"
    assert "filterGroups" not in body


def test_search_contacts_sends_filter_groups_with_no_query(monkeypatch):
    """Regression: the 2026-09-16 incident's failed request had a
    filterGroups-only body with no "query" key at all (filtering contacts
    created after a date) - the generic api_call tool was the only way to
    express that. filter_groups_json must let hubspot_search_contacts send
    exactly that shape without a query."""
    mock_request = Mock(
        return_value=MockResponse(json_data={"total": 0, "results": []})
    )
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    filter_groups_json = json.dumps(
        [
            {
                "filters": [
                    {
                        "propertyName": "createdate",
                        "operator": "GTE",
                        "value": "2026-09-01",
                    }
                ]
            }
        ]
    )

    hubspot.hubspot_search_contacts(filter_groups_json=filter_groups_json)

    body = mock_request.call_args.kwargs["json"]
    assert "query" not in body
    assert body["filterGroups"] == [
        {
            "filters": [
                {"propertyName": "createdate", "operator": "GTE", "value": "2026-09-01"}
            ]
        }
    ]


def test_search_contacts_combines_query_and_filter_groups(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"total": 0, "results": []})
    )
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    filter_groups_json = json.dumps(
        [
            {
                "filters": [
                    {
                        "propertyName": "lifecyclestage",
                        "operator": "EQ",
                        "value": "lead",
                    }
                ]
            }
        ]
    )

    hubspot.hubspot_search_contacts(query="acme", filter_groups_json=filter_groups_json)

    body = mock_request.call_args.kwargs["json"]
    assert body["query"] == "acme"
    assert "filterGroups" in body


def test_search_contacts_rejects_non_array_filter_groups_json(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_search_contacts(filter_groups_json=json.dumps({"filters": []}))
    )

    assert result["status"] == "error"
    assert "filter_groups_json" in result["message"]
    mock_request.assert_not_called()


def test_search_contacts_rejects_a_flat_list_of_non_group_conditions(monkeypatch):
    """Regression: _parse_filter_groups previously only checked the
    top-level JSON was a list, so a caller mistake like a flat list of
    condition strings (rather than HubSpot's `{"filters": [...]}` group
    objects) passed local validation and reached HubSpot as-is, surfacing
    only as an opaque upstream 400 instead of the same actionable local
    error the top-level-not-a-list case already gets."""
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_search_contacts(filter_groups_json=json.dumps(["stage=won"]))
    )

    assert result["status"] == "error"
    assert "filter_groups_json" in result["message"]
    mock_request.assert_not_called()


def test_search_contacts_rejects_a_filter_group_with_empty_filters(monkeypatch):
    """Regression (Blocking): a group with an empty (or missing) "filters"
    list is truthy as a Python list-of-one-dict, so it passed both the
    old element-shape check and the no-criteria guard in _search - HubSpot
    treats a filter group with no conditions as matching everything,
    silently defeating the guard's entire purpose. A caller building
    filter_groups_json programmatically (e.g. all conditions pruned
    upstream but the outer group kept) can plausibly produce exactly this
    shape."""
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_search_contacts(filter_groups_json=json.dumps([{}]))
    )

    assert result["status"] == "error"
    assert "filter_groups_json" in result["message"]
    mock_request.assert_not_called()


def test_search_contacts_rejects_a_filter_group_with_non_list_filters(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_search_contacts(
            filter_groups_json=json.dumps([{"filters": "not-a-list"}])
        )
    )

    assert result["status"] == "error"
    assert "filter_groups_json" in result["message"]
    mock_request.assert_not_called()


def test_search_contacts_rejects_a_group_with_an_empty_filters_list_directly(
    monkeypatch,
):
    """Isolates the "filters" list being present but empty, as distinct
    from missing entirely ({}) or wrong-typed ("not-a-list") - all three
    are separate branches of _is_valid_filter_group."""
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_search_contacts(
            filter_groups_json=json.dumps([{"filters": []}])
        )
    )

    assert result["status"] == "error"
    assert "filter_groups_json" in result["message"]
    mock_request.assert_not_called()


def test_search_contacts_rejects_a_non_empty_filters_list_of_empty_filters(
    monkeypatch,
):
    """Regression: the previous round's fix only checked the filters LIST
    was non-empty, not that its individual filter objects were meaningful -
    filter_groups_json='[{"filters": [{}]}]' is a non-empty list containing
    one filter with neither propertyName nor operator, the same "matches
    everything" hole one level deeper. Verified against HubSpot's CRM
    Search API docs: propertyName and operator are required on every
    filter regardless of operator type."""
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_search_contacts(
            filter_groups_json=json.dumps([{"filters": [{}]}])
        )
    )

    assert result["status"] == "error"
    assert "filter_groups_json" in result["message"]
    mock_request.assert_not_called()


def test_search_contacts_rejects_a_filters_list_of_non_dict_entries(monkeypatch):
    """filter_groups_json='[{"filters": ["dealstage"]}]' - a non-empty list
    of strings rather than filter objects - must be rejected locally
    instead of reaching HubSpot for an opaque 400."""
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_search_contacts(
            filter_groups_json=json.dumps([{"filters": ["dealstage"]}])
        )
    )

    assert result["status"] == "error"
    assert "filter_groups_json" in result["message"]
    mock_request.assert_not_called()


def test_search_contacts_rejects_a_filter_missing_property_name_or_operator(
    monkeypatch,
):
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(
        hubspot.hubspot_search_contacts(
            filter_groups_json=json.dumps(
                [{"filters": [{"propertyName": "dealstage"}]}]
            )
        )
    )

    assert result["status"] == "error"
    assert "filter_groups_json" in result["message"]
    mock_request.assert_not_called()


def test_search_contacts_accepts_a_has_property_filter_with_no_value(monkeypatch):
    """HAS_PROPERTY/NOT_HAS_PROPERTY are real HubSpot operators that take
    no "value" field at all - the propertyName/operator check must not
    reject a legitimate filter just because it has neither "value" nor
    "values" nor "highValue". Verified against HubSpot's CRM Search API
    docs."""
    mock_request = Mock(
        return_value=MockResponse(json_data={"total": 0, "results": []})
    )
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    filter_groups_json = json.dumps(
        [{"filters": [{"propertyName": "email", "operator": "HAS_PROPERTY"}]}]
    )

    result = json.loads(
        hubspot.hubspot_search_contacts(filter_groups_json=filter_groups_json)
    )

    assert result["status"] == "success"
    mock_request.assert_called_once()


def test_parse_filter_groups_rejects_malformed_json_with_an_actionable_message(
    monkeypatch,
):
    """Genuinely malformed (non-JSON) filter_groups_json used to raise a raw
    json.JSONDecodeError with no parameter name context - still caught by
    the tool's try/except (JSONDecodeError is a ValueError subclass) so it
    never crashed, but the message didn't say which parameter was wrong."""
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_search_contacts(filter_groups_json="not json"))

    assert result["status"] == "error"
    assert "filter_groups_json" in result["message"]
    mock_request.assert_not_called()


def test_search_contacts_rejects_whitespace_only_query(monkeypatch):
    """A whitespace-only query is not meaningfully different from no query
    at all - it must not (a) be sent to HubSpot as a real search term, or
    (b) count as "a query was given" and bypass the no-criteria guard."""
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_search_contacts(query="   "))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_search_contacts_strips_query_before_sending(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"total": 0, "results": []})
    )
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    hubspot.hubspot_search_contacts(query="  acme  ")

    assert mock_request.call_args.kwargs["json"]["query"] == "acme"


@pytest.mark.parametrize(
    "tool, path",
    [
        (hubspot.hubspot_search_contacts, "/crm/v3/objects/contacts/search"),
        (hubspot.hubspot_search_companies, "/crm/v3/objects/companies/search"),
        (hubspot.hubspot_search_deals, "/crm/v3/objects/deals/search"),
    ],
)
def test_search_hits_the_expected_search_endpoint(monkeypatch, tool, path):
    mock_request = Mock(
        return_value=MockResponse(json_data={"total": 0, "results": []})
    )
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    tool(query="acme")

    assert mock_request.call_args.kwargs["url"].endswith(path)


@pytest.mark.parametrize(
    "tool, list_tool_name",
    [
        (hubspot.hubspot_search_contacts, "hubspot_list_contacts"),
        (hubspot.hubspot_search_companies, "hubspot_list_companies"),
        (hubspot.hubspot_search_deals, "hubspot_list_deals"),
    ],
)
def test_search_rejects_no_query_and_no_filter(monkeypatch, tool, list_tool_name):
    """Regression: query used to be a required, non-empty `str` (no
    default), so an unfiltered search was structurally unreachable. Making
    it optional (for the filter-only case) also made a call with NEITHER
    query nor filter_groups_json newly possible - which would otherwise
    silently return an arbitrary, unfiltered page that looks like a real
    match instead of the caller's evidently-missing search criteria."""
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(tool())

    assert result["status"] == "error"
    assert list_tool_name in result["message"]
    mock_request.assert_not_called()


def test_search_contacts_rejects_an_empty_filter_groups_array(monkeypatch):
    """An explicit `filter_groups_json="[]"` parses to an empty list, which
    is exactly as unscoped as omitting it - it must not be treated as "a
    filter was given" and bypass the no-criteria guard above."""
    mock_request = Mock()
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_search_contacts(filter_groups_json="[]"))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_search_contacts_handles_an_explicit_null_total(monkeypatch):
    """Regression: result.get("total", 0) only substitutes the default when
    the key is absent, not when HubSpot returns an explicit JSON null -
    the same class of bug already fixed for the "results" field."""
    mock_request = Mock(
        return_value=MockResponse(json_data={"total": None, "results": []})
    )
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    result = json.loads(hubspot.hubspot_search_contacts(query="acme"))

    assert result == {"status": "success", "total": 0, "results": []}


def test_search_companies_sends_filter_groups(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"total": 0, "results": []})
    )
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    filter_groups_json = json.dumps(
        [{"filters": [{"propertyName": "industry", "operator": "EQ", "value": "SAAS"}]}]
    )

    hubspot.hubspot_search_companies(filter_groups_json=filter_groups_json)

    body = mock_request.call_args.kwargs["json"]
    assert "query" not in body
    assert body["filterGroups"] == [
        {"filters": [{"propertyName": "industry", "operator": "EQ", "value": "SAAS"}]}
    ]


def test_search_deals_hits_the_deals_search_endpoint(monkeypatch):
    """New tool: previously there was no way to search/filter deals at all
    across the whole portal (only per-contact/per-company lookups), so an
    agent asking for e.g. "deals in the qualifiedtobuy stage" had no MCP
    tool for it and had to fall back to the credential-less api_call tool."""
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "total": 1,
                "results": [
                    {"id": "d1", "properties": {"dealstage": "qualifiedtobuy"}}
                ],
            }
        )
    )
    monkeypatch.setattr(hubspot.requests, "request", mock_request)

    filter_groups_json = json.dumps(
        [
            {
                "filters": [
                    {
                        "propertyName": "dealstage",
                        "operator": "EQ",
                        "value": "qualifiedtobuy",
                    }
                ]
            }
        ]
    )

    result = json.loads(
        hubspot.hubspot_search_deals(filter_groups_json=filter_groups_json)
    )

    assert result == {
        "status": "success",
        "total": 1,
        "results": [{"id": "d1", "properties": {"dealstage": "qualifiedtobuy"}}],
    }
    assert mock_request.call_args.kwargs["url"].endswith("/crm/v3/objects/deals/search")
    body = mock_request.call_args.kwargs["json"]
    assert "query" not in body
    assert body["filterGroups"] == [
        {
            "filters": [
                {
                    "propertyName": "dealstage",
                    "operator": "EQ",
                    "value": "qualifiedtobuy",
                }
            ]
        }
    ]


def test_search_deals_wraps_request_errors(monkeypatch):
    monkeypatch.setattr(
        hubspot.requests,
        "request",
        Mock(return_value=MockResponse(status_code=500, text="boom")),
    )

    result = json.loads(hubspot.hubspot_search_deals(query="acme"))

    assert result["status"] == "error"
    assert "boom" in result["message"]
