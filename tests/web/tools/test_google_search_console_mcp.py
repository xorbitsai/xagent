import json
from unittest.mock import Mock

import pytest
import requests

from xagent.config import get_tool_max_output_length
from xagent.web.tools.mcp import google_search_console


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


def test_headers_require_access_token(monkeypatch):
    monkeypatch.delenv("GOOGLE_ACCESS_TOKEN")

    with pytest.raises(ValueError, match="GOOGLE_ACCESS_TOKEN"):
        google_search_console._headers()


def test_headers_include_bearer_token():
    headers = google_search_console._headers()

    assert headers["Authorization"] == "Bearer access-token"


def test_encoded_site_url_percent_encodes_url_prefix_property():
    assert (
        google_search_console._encoded_site_url("https://example.com/")
        == "https%3A%2F%2Fexample.com%2F"
    )


def test_encoded_site_url_percent_encodes_domain_property():
    assert (
        google_search_console._encoded_site_url("sc-domain:example.com")
        == "sc-domain%3Aexample.com"
    )


def test_encoded_site_url_rejects_empty():
    with pytest.raises(ValueError, match="site_url"):
        google_search_console._encoded_site_url("")


def test_request_wraps_http_error_with_google_error_message(monkeypatch):
    monkeypatch.setattr(
        google_search_console.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=400,
                json_data={
                    "error": {
                        "code": 400,
                        "message": "Invalid site URL.",
                        "status": "INVALID_ARGUMENT",
                    }
                },
            )
        ),
    )

    with pytest.raises(RuntimeError, match="Invalid site URL"):
        google_search_console._request(
            "GET", f"{google_search_console.WEBMASTERS_API_BASE_URL}/sites"
        )


def test_request_falls_back_to_raw_text_for_unstructured_error_body(monkeypatch):
    monkeypatch.setattr(
        google_search_console.requests,
        "request",
        Mock(return_value=MockResponse(status_code=500, text="upstream 500")),
    )

    with pytest.raises(RuntimeError, match="upstream 500"):
        google_search_console._request(
            "GET", f"{google_search_console.WEBMASTERS_API_BASE_URL}/sites"
        )


def test_request_caps_unstructured_error_body(monkeypatch):
    monkeypatch.setattr(
        google_search_console.requests,
        "request",
        Mock(return_value=MockResponse(status_code=502, text="x" * 5000)),
    )

    with pytest.raises(RuntimeError) as excinfo:
        google_search_console._request(
            "GET", f"{google_search_console.WEBMASTERS_API_BASE_URL}/sites"
        )
    assert len(str(excinfo.value)) < 1200


def test_request_raises_runtime_error_on_non_json_success_body(monkeypatch):
    """A 200 response whose body isn't JSON (e.g. an HTML page from an
    intermediate proxy) must fail with a clear RuntimeError, not a raw
    ValueError/JSONDecodeError from response.json()."""
    bad_response = Mock()
    bad_response.status_code = 200
    bad_response.content = b"<html>not json</html>"
    bad_response.text = "<html>not json</html>"
    bad_response.raise_for_status = Mock()
    bad_response.json = Mock(side_effect=ValueError("Expecting value"))
    monkeypatch.setattr(
        google_search_console.requests,
        "request",
        Mock(return_value=bad_response),
    )

    with pytest.raises(RuntimeError, match="Failed to parse JSON response"):
        google_search_console._request(
            "GET", f"{google_search_console.WEBMASTERS_API_BASE_URL}/sites"
        )


def test_list_sites_projects_site_entries(monkeypatch):
    monkeypatch.setattr(
        google_search_console.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "siteEntry": [
                        {
                            "siteUrl": "https://example.com/",
                            "permissionLevel": "siteOwner",
                        },
                        {
                            "siteUrl": "sc-domain:example.com",
                            "permissionLevel": "siteFullUser",
                        },
                    ]
                }
            )
        ),
    )

    result = json.loads(google_search_console.google_search_console_list_sites())

    assert result["status"] == "success"
    assert result["sites"] == [
        {"site_url": "https://example.com/", "permission_level": "siteOwner"},
        {"site_url": "sc-domain:example.com", "permission_level": "siteFullUser"},
    ]


def test_list_sites_filters_out_entries_without_site_url(monkeypatch):
    monkeypatch.setattr(
        google_search_console.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "siteEntry": [
                        {"siteUrl": None, "permissionLevel": "siteOwner"},
                        {"permissionLevel": "siteOwner"},
                        {
                            "siteUrl": "https://example.com/",
                            "permissionLevel": "siteOwner",
                        },
                    ]
                }
            )
        ),
    )

    result = json.loads(google_search_console.google_search_console_list_sites())

    assert result["status"] == "success"
    assert result["sites"] == [
        {"site_url": "https://example.com/", "permission_level": "siteOwner"}
    ]


def test_list_sites_handles_no_sites(monkeypatch):
    monkeypatch.setattr(
        google_search_console.requests,
        "request",
        Mock(return_value=MockResponse(json_data={})),
    )

    result = json.loads(google_search_console.google_search_console_list_sites())

    assert result["status"] == "success"
    assert result["sites"] == []


def test_list_sites_returns_error_payload_on_failure(monkeypatch):
    monkeypatch.setattr(
        google_search_console.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=403,
                json_data={"error": {"message": "insufficient permissions"}},
            )
        ),
    )

    result = json.loads(google_search_console.google_search_console_list_sites())

    assert result["status"] == "error"
    assert "insufficient permissions" in result["message"]


def test_list_sitemaps_returns_sitemap_list(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "sitemap": [
                    {"path": "https://example.com/sitemap.xml", "isPending": False}
                ]
            }
        )
    )
    monkeypatch.setattr(google_search_console.requests, "request", mock_request)

    result = json.loads(
        google_search_console.google_search_console_list_sitemaps(
            "https://example.com/"
        )
    )

    assert result["status"] == "success"
    assert result["sitemaps"][0]["path"] == "https://example.com/sitemap.xml"
    assert "https%3A%2F%2Fexample.com%2F" in mock_request.call_args.kwargs["url"]


def test_query_search_analytics_builds_body_and_returns_rows(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "rows": [
                    {
                        "keys": ["some query"],
                        "clicks": 10,
                        "impressions": 100,
                        "ctr": 0.1,
                        "position": 3.5,
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(google_search_console.requests, "request", mock_request)

    result = json.loads(
        google_search_console.google_search_console_query_search_analytics(
            "https://example.com/",
            start_date="2026-07-01",
            end_date="2026-07-28",
            dimensions=["query"],
        )
    )

    assert result["status"] == "success"
    assert result["row_count"] == 1
    assert result["rows"][0]["clicks"] == 10

    call_kwargs = mock_request.call_args.kwargs
    assert call_kwargs["url"].endswith(
        "/sites/https%3A%2F%2Fexample.com%2F/searchAnalytics/query"
    )
    body = call_kwargs["json"]
    assert body["startDate"] == "2026-07-01"
    assert body["endDate"] == "2026-07-28"
    assert body["dimensions"] == ["query"]
    assert body["type"] == "web"
    assert body["rowLimit"] == google_search_console.QUERY_DEFAULT_ROW_LIMIT
    assert body["startRow"] == 0


def test_query_search_analytics_passes_through_filter_groups_and_search_type(
    monkeypatch,
):
    mock_request = Mock(return_value=MockResponse(json_data={}))
    monkeypatch.setattr(google_search_console.requests, "request", mock_request)

    filter_groups = [
        {
            "filters": [
                {"dimension": "country", "operator": "equals", "expression": "usa"}
            ]
        }
    ]

    result = json.loads(
        google_search_console.google_search_console_query_search_analytics(
            "sc-domain:example.com",
            start_date="2026-07-01",
            end_date="2026-07-28",
            search_type="image",
            dimension_filter_groups=filter_groups,
            row_limit=50,
            start_row=100,
        )
    )

    assert result["status"] == "success"
    assert result["rows"] == []
    body = mock_request.call_args.kwargs["json"]
    assert body["type"] == "image"
    assert body["dimensionFilterGroups"] == filter_groups
    assert body["rowLimit"] == 50
    assert body["startRow"] == 100


def test_query_search_analytics_rejects_invalid_date(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(google_search_console.requests, "request", mock_request)

    result = json.loads(
        google_search_console.google_search_console_query_search_analytics(
            "https://example.com/", start_date="7daysAgo", end_date="2026-07-28"
        )
    )

    assert result["status"] == "error"
    assert "start_date" in result["message"]
    mock_request.assert_not_called()


def test_query_search_analytics_rejects_invalid_calendar_date(monkeypatch):
    """The date regex alone accepts "2026-02-31" (right shape, impossible
    date); fromisoformat catches what the regex can't."""
    mock_request = Mock()
    monkeypatch.setattr(google_search_console.requests, "request", mock_request)

    result = json.loads(
        google_search_console.google_search_console_query_search_analytics(
            "https://example.com/", start_date="2026-02-31", end_date="2026-07-28"
        )
    )

    assert result["status"] == "error"
    assert "valid calendar date" in result["message"]
    mock_request.assert_not_called()


def test_query_search_analytics_rejects_start_date_after_end_date(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(google_search_console.requests, "request", mock_request)

    result = json.loads(
        google_search_console.google_search_console_query_search_analytics(
            "https://example.com/", start_date="2026-08-01", end_date="2026-01-01"
        )
    )

    assert result["status"] == "error"
    assert "start_date" in result["message"]
    mock_request.assert_not_called()


def test_query_search_analytics_rejects_invalid_search_type(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(google_search_console.requests, "request", mock_request)

    result = json.loads(
        google_search_console.google_search_console_query_search_analytics(
            "https://example.com/",
            start_date="2026-07-01",
            end_date="2026-07-28",
            search_type="carrier-pigeon",
        )
    )

    assert result["status"] == "error"
    assert "search_type" in result["message"]
    mock_request.assert_not_called()


def test_query_search_analytics_rejects_invalid_dimensions(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(google_search_console.requests, "request", mock_request)

    result = json.loads(
        google_search_console.google_search_console_query_search_analytics(
            "https://example.com/",
            start_date="2026-07-01",
            end_date="2026-07-28",
            dimensions=["query", "bogus"],
        )
    )

    assert result["status"] == "error"
    assert "bogus" in result["message"]
    mock_request.assert_not_called()


def test_query_search_analytics_rejects_zero_row_limit(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(google_search_console.requests, "request", mock_request)

    result = json.loads(
        google_search_console.google_search_console_query_search_analytics(
            "https://example.com/",
            start_date="2026-07-01",
            end_date="2026-07-28",
            row_limit=0,
        )
    )

    assert result["status"] == "error"
    assert "row_limit" in result["message"]
    mock_request.assert_not_called()


def test_query_search_analytics_rejects_row_limit_above_max(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(google_search_console.requests, "request", mock_request)

    result = json.loads(
        google_search_console.google_search_console_query_search_analytics(
            "https://example.com/",
            start_date="2026-07-01",
            end_date="2026-07-28",
            row_limit=google_search_console.QUERY_MAX_ROW_LIMIT + 1,
        )
    )

    assert result["status"] == "error"
    assert "row_limit" in result["message"]
    mock_request.assert_not_called()


def test_query_search_analytics_rejects_negative_start_row(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(google_search_console.requests, "request", mock_request)

    result = json.loads(
        google_search_console.google_search_console_query_search_analytics(
            "https://example.com/",
            start_date="2026-07-01",
            end_date="2026-07-28",
            start_row=-1,
        )
    )

    assert result["status"] == "error"
    assert "start_row" in result["message"]
    mock_request.assert_not_called()


def test_query_search_analytics_trims_rows_and_flags_truncated_when_over_budget(
    monkeypatch,
):
    rows = [
        {
            "keys": [f"https://example.com/some/long/page/path/{i}".ljust(70, "x")],
            "clicks": i,
            "impressions": i * 10,
            "ctr": 0.1,
            "position": 3.5,
        }
        for i in range(google_search_console.QUERY_MAX_ROW_LIMIT)
    ]
    mock_request = Mock(return_value=MockResponse(json_data={"rows": rows}))
    monkeypatch.setattr(google_search_console.requests, "request", mock_request)

    response = google_search_console.google_search_console_query_search_analytics(
        "https://example.com/",
        start_date="2026-07-01",
        end_date="2026-07-28",
        dimensions=["page"],
        row_limit=google_search_console.QUERY_MAX_ROW_LIMIT,
    )

    result = json.loads(response)
    assert result["status"] == "success"
    assert result["truncated"] is True
    assert len(result["rows"]) < google_search_console.QUERY_MAX_ROW_LIMIT
    assert len(response) < get_tool_max_output_length()


def test_query_search_analytics_trims_to_empty_when_single_row_exceeds_budget(
    monkeypatch,
):
    """A single row whose own serialized size exceeds max_output_length must
    still end up under budget — the halving loop must not stop at
    len(rows) == 1 just because it can't halve further."""
    max_output_length = get_tool_max_output_length()
    oversized_row = {"keys": ["x" * (max_output_length + 1000)]}
    mock_request = Mock(return_value=MockResponse(json_data={"rows": [oversized_row]}))
    monkeypatch.setattr(google_search_console.requests, "request", mock_request)

    response = google_search_console.google_search_console_query_search_analytics(
        "https://example.com/", start_date="2026-07-01", end_date="2026-07-28"
    )

    result = json.loads(response)
    assert result["status"] == "success"
    assert result["rows"] == []
    assert result["truncated"] is True
    assert len(response) < max_output_length


def test_query_search_analytics_treats_non_list_rows_as_empty(monkeypatch):
    """A malformed API response where "rows" isn't a list (e.g. null or a
    dict) must not raise a TypeError out of len()/slicing; it degrades to an
    empty result instead."""
    mock_request = Mock(return_value=MockResponse(json_data={"rows": "not-a-list"}))
    monkeypatch.setattr(google_search_console.requests, "request", mock_request)

    result = json.loads(
        google_search_console.google_search_console_query_search_analytics(
            "https://example.com/", start_date="2026-07-01", end_date="2026-07-28"
        )
    )

    assert result["status"] == "success"
    assert result["rows"] == []
    assert result["row_count"] == 0


def test_query_search_analytics_returns_error_payload_on_api_failure(monkeypatch):
    monkeypatch.setattr(
        google_search_console.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=400,
                json_data={"error": {"message": "bad request"}},
            )
        ),
    )

    result = json.loads(
        google_search_console.google_search_console_query_search_analytics(
            "https://example.com/", start_date="2026-07-01", end_date="2026-07-28"
        )
    )

    assert result["status"] == "error"
    assert "bad request" in result["message"]


def test_inspect_url_returns_inspection_result(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "inspectionResult": {
                    "indexStatusResult": {"verdict": "PASS"},
                }
            }
        )
    )
    monkeypatch.setattr(google_search_console.requests, "request", mock_request)

    result = json.loads(
        google_search_console.google_search_console_inspect_url(
            "https://example.com/", "https://example.com/page"
        )
    )

    assert result["status"] == "success"
    assert result["inspection_result"]["indexStatusResult"]["verdict"] == "PASS"

    body = mock_request.call_args.kwargs["json"]
    assert body["inspectionUrl"] == "https://example.com/page"
    assert body["siteUrl"] == "https://example.com/"
    assert body["languageCode"] == "en-US"
    assert mock_request.call_args.kwargs["url"].endswith("/urlInspection/index:inspect")


def test_inspect_url_rejects_empty_inspection_url(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(google_search_console.requests, "request", mock_request)

    result = json.loads(
        google_search_console.google_search_console_inspect_url(
            "https://example.com/", ""
        )
    )

    assert result["status"] == "error"
    assert "inspection_url" in result["message"]
    mock_request.assert_not_called()


def test_inspect_url_returns_error_payload_on_api_failure(monkeypatch):
    monkeypatch.setattr(
        google_search_console.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=404,
                json_data={"error": {"message": "URL not found"}},
            )
        ),
    )

    result = json.loads(
        google_search_console.google_search_console_inspect_url(
            "https://example.com/", "https://example.com/missing"
        )
    )

    assert result["status"] == "error"
    assert "URL not found" in result["message"]
