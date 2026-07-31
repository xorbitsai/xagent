import json
from unittest.mock import Mock

import pytest
import requests

from xagent.config import get_tool_max_output_length
from xagent.web.tools.mcp import google_analytics


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
        google_analytics._headers()


def test_headers_include_bearer_token():
    headers = google_analytics._headers()

    assert headers["Authorization"] == "Bearer access-token"


def test_normalize_property_id_accepts_bare_numeric():
    assert google_analytics._normalize_property_id("123456") == "123456"


def test_normalize_property_id_strips_resource_prefix():
    assert google_analytics._normalize_property_id("properties/123456") == "123456"


def test_normalize_property_id_rejects_non_numeric():
    with pytest.raises(ValueError, match="property_id"):
        google_analytics._normalize_property_id("123/../456")


def test_normalize_property_id_rejects_empty_after_prefix():
    with pytest.raises(ValueError, match="property_id"):
        google_analytics._normalize_property_id("properties/")


def test_request_wraps_http_error_with_google_error_message(monkeypatch):
    monkeypatch.setattr(
        google_analytics.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=400,
                json_data={
                    "error": {
                        "code": 400,
                        "message": "Field sessions is not a valid metric.",
                        "status": "INVALID_ARGUMENT",
                    }
                },
            )
        ),
    )

    with pytest.raises(RuntimeError, match="not a valid metric"):
        google_analytics._request(
            "POST", f"{google_analytics.DATA_API_BASE_URL}/properties/1:runReport"
        )


def test_request_falls_back_to_raw_text_for_unstructured_error_body(monkeypatch):
    monkeypatch.setattr(
        google_analytics.requests,
        "request",
        Mock(return_value=MockResponse(status_code=500, text="upstream 500")),
    )

    with pytest.raises(RuntimeError, match="upstream 500"):
        google_analytics._request(
            "GET", f"{google_analytics.ADMIN_API_BASE_URL}/accountSummaries"
        )


def test_list_properties_flattens_account_summaries(monkeypatch):
    monkeypatch.setattr(
        google_analytics.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "accountSummaries": [
                        {
                            "displayName": "Acme Inc",
                            "propertySummaries": [
                                {
                                    "property": "properties/111",
                                    "displayName": "Acme Website",
                                },
                                {
                                    "property": "properties/222",
                                    "displayName": "Acme App",
                                },
                            ],
                        }
                    ]
                }
            )
        ),
    )

    result = json.loads(google_analytics.google_analytics_list_properties())

    assert result["status"] == "success"
    assert result["properties"] == [
        {
            "account_name": "Acme Inc",
            "property_id": "111",
            "property_display_name": "Acme Website",
        },
        {
            "account_name": "Acme Inc",
            "property_id": "222",
            "property_display_name": "Acme App",
        },
    ]


def test_list_properties_follows_pagination(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(
                json_data={
                    "accountSummaries": [
                        {
                            "displayName": "Acme Inc",
                            "propertySummaries": [
                                {"property": "properties/111", "displayName": "Site A"}
                            ],
                        }
                    ],
                    "nextPageToken": "page-2",
                }
            ),
            MockResponse(
                json_data={
                    "accountSummaries": [
                        {
                            "displayName": "Acme Inc",
                            "propertySummaries": [
                                {"property": "properties/222", "displayName": "Site B"}
                            ],
                        }
                    ]
                }
            ),
        ]
    )
    monkeypatch.setattr(google_analytics.requests, "request", mock_request)

    result = json.loads(google_analytics.google_analytics_list_properties())

    assert result["status"] == "success"
    assert [p["property_id"] for p in result["properties"]] == ["111", "222"]
    assert mock_request.call_count == 2
    second_call = mock_request.call_args_list[1]
    assert second_call.kwargs["params"]["pageToken"] == "page-2"


def test_list_properties_handles_account_without_properties(monkeypatch):
    monkeypatch.setattr(
        google_analytics.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"accountSummaries": [{"displayName": "Empty Account"}]}
            )
        ),
    )

    result = json.loads(google_analytics.google_analytics_list_properties())

    assert result["status"] == "success"
    assert result["properties"] == []


def test_list_properties_tolerates_null_values_in_response(monkeypatch):
    """Keys present with an explicit null (accountSummaries, property) must not
    crash iteration/rsplit; entries without a usable property resource name are
    skipped rather than emitted with an empty property_id."""
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data={"accountSummaries": None}),
            MockResponse(
                json_data={
                    "accountSummaries": [
                        {
                            "displayName": "Acme Inc",
                            "propertySummaries": [
                                {"property": None, "displayName": "broken"},
                                {"property": "properties/111", "displayName": "ok"},
                            ],
                        }
                    ]
                }
            ),
        ]
    )
    monkeypatch.setattr(google_analytics.requests, "request", mock_request)

    first = json.loads(google_analytics.google_analytics_list_properties())
    second = json.loads(google_analytics.google_analytics_list_properties())

    assert first["status"] == "success"
    assert first["properties"] == []
    assert second["status"] == "success"
    assert [p["property_id"] for p in second["properties"]] == ["111"]


def test_list_properties_returns_error_payload_on_failure(monkeypatch):
    monkeypatch.setattr(
        google_analytics.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=403,
                json_data={"error": {"message": "insufficient permissions"}},
            )
        ),
    )

    result = json.loads(google_analytics.google_analytics_list_properties())

    assert result["status"] == "error"
    assert "insufficient permissions" in result["message"]


def test_get_metadata_projects_entries_to_name_fields(monkeypatch):
    """The full GA4 metadata document (with descriptions) can exceed the
    platform output filter's truncation threshold; only the name-picking
    fields survive the projection."""
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "dimensions": [
                    {
                        "apiName": "country",
                        "uiName": "Country",
                        "category": "Geography",
                        "description": "x" * 500,
                        "customDefinition": False,
                    }
                ],
                "metrics": [
                    {
                        "apiName": "sessions",
                        "uiName": "Sessions",
                        "category": "Session",
                        "description": "y" * 500,
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(google_analytics.requests, "request", mock_request)

    result = json.loads(google_analytics.google_analytics_get_metadata("properties/42"))

    assert result["status"] == "success"
    assert result["dimensions"] == [
        {"apiName": "country", "uiName": "Country", "category": "Geography"}
    ]
    assert result["metrics"] == [
        {"apiName": "sessions", "uiName": "Sessions", "category": "Session"}
    ]
    assert mock_request.call_args.kwargs["url"].endswith("/properties/42/metadata")


def test_get_metadata_skips_entries_without_usable_api_name(monkeypatch):
    """An entry with no apiName (or a non-string one) can't be passed back into
    run_report's metrics/dimensions, so it must be dropped rather than
    projected with an empty name."""
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "dimensions": [
                    {"apiName": None, "uiName": "broken"},
                    {"uiName": "also broken"},
                    {"apiName": "country", "uiName": "Country"},
                ],
                "metrics": [],
            }
        )
    )
    monkeypatch.setattr(google_analytics.requests, "request", mock_request)

    result = json.loads(google_analytics.google_analytics_get_metadata("42"))

    assert result["status"] == "success"
    assert [d["apiName"] for d in result["dimensions"]] == ["country"]


def test_get_metadata_search_filters_by_name(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "dimensions": [
                    {"apiName": "country", "uiName": "Country"},
                    {"apiName": "sessionSource", "uiName": "Session source"},
                ],
                "metrics": [
                    {"apiName": "sessions", "uiName": "Sessions"},
                    {"apiName": "totalRevenue", "uiName": "Total revenue"},
                ],
            }
        )
    )
    monkeypatch.setattr(google_analytics.requests, "request", mock_request)

    result = json.loads(
        google_analytics.google_analytics_get_metadata("42", search="SESSION")
    )

    assert result["status"] == "success"
    assert [d["apiName"] for d in result["dimensions"]] == ["sessionSource"]
    assert [m["apiName"] for m in result["metrics"]] == ["sessions"]


def test_get_metadata_rejects_invalid_property_id(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(google_analytics.requests, "request", mock_request)

    result = json.loads(google_analytics.google_analytics_get_metadata("not-a-number"))

    assert result["status"] == "error"
    assert "property_id" in result["message"]
    mock_request.assert_not_called()


def test_run_report_builds_body_and_returns_rows(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "dimensionHeaders": [{"name": "sessionDefaultChannelGroup"}],
                "metricHeaders": [{"name": "sessions", "type": "TYPE_INTEGER"}],
                "rows": [
                    {
                        "dimensionValues": [{"value": "Organic Search"}],
                        "metricValues": [{"value": "1200"}],
                    }
                ],
                "rowCount": 1,
            }
        )
    )
    monkeypatch.setattr(google_analytics.requests, "request", mock_request)

    result = json.loads(
        google_analytics.google_analytics_run_report(
            "properties/42",
            metrics=["sessions", "conversions"],
            date_ranges=[
                {
                    "start_date": "2026-07-01",
                    "end_date": "2026-07-28",
                    "name": "current",
                },
                {
                    "start_date": "2026-06-03",
                    "end_date": "2026-06-30",
                    "name": "previous",
                },
            ],
            dimensions=["sessionDefaultChannelGroup"],
            limit=50,
        )
    )

    assert result["status"] == "success"
    assert result["row_count"] == 1
    assert result["rows"][0]["metricValues"] == [{"value": "1200"}]

    call_kwargs = mock_request.call_args.kwargs
    assert call_kwargs["url"].endswith("/properties/42:runReport")
    body = call_kwargs["json"]
    assert body["metrics"] == [{"name": "sessions"}, {"name": "conversions"}]
    assert body["dateRanges"] == [
        {"startDate": "2026-07-01", "endDate": "2026-07-28", "name": "current"},
        {"startDate": "2026-06-03", "endDate": "2026-06-30", "name": "previous"},
    ]
    assert body["dimensions"] == [{"name": "sessionDefaultChannelGroup"}]
    assert body["limit"] == "50"
    assert "dimensionFilter" not in body
    assert "orderBys" not in body


def test_run_report_passes_through_filter_and_order(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"rowCount": 0}))
    monkeypatch.setattr(google_analytics.requests, "request", mock_request)

    dimension_filter = {
        "filter": {
            "fieldName": "sessionCampaignName",
            "stringFilter": {"value": "summer-sale"},
        }
    }
    order_bys = [{"metric": {"metricName": "sessions"}, "desc": True}]

    result = json.loads(
        google_analytics.google_analytics_run_report(
            "42",
            metrics=["sessions"],
            date_ranges=[{"start_date": "7daysAgo", "end_date": "today"}],
            dimension_filter=dimension_filter,
            order_bys=order_bys,
        )
    )

    assert result["status"] == "success"
    assert result["rows"] == []
    body = mock_request.call_args.kwargs["json"]
    assert body["dimensionFilter"] == dimension_filter
    assert body["orderBys"] == order_bys
    assert body["dateRanges"] == [{"startDate": "7daysAgo", "endDate": "today"}]


def test_run_report_sends_default_limit_and_omits_zero_offset(monkeypatch):
    """GA4's own default returns up to 10k rows; the tool must always send an
    explicit limit, and the default itself must stay small enough that a wide
    report's serialized response doesn't cross the MCP output truncation
    threshold."""
    mock_request = Mock(return_value=MockResponse(json_data={"rowCount": 0}))
    monkeypatch.setattr(google_analytics.requests, "request", mock_request)

    google_analytics.google_analytics_run_report(
        "42",
        metrics=["sessions"],
        date_ranges=[{"start_date": "7daysAgo", "end_date": "today"}],
    )

    body = mock_request.call_args.kwargs["json"]
    assert body["limit"] == str(google_analytics.RUN_REPORT_DEFAULT_LIMIT)
    assert "offset" not in body


def test_run_report_rejects_empty_metrics(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(google_analytics.requests, "request", mock_request)

    result = json.loads(
        google_analytics.google_analytics_run_report(
            "42",
            metrics=[],
            date_ranges=[{"start_date": "7daysAgo", "end_date": "today"}],
        )
    )

    assert result["status"] == "error"
    assert "metrics" in result["message"]
    mock_request.assert_not_called()


@pytest.mark.parametrize(
    "bad_limit",
    [0, google_analytics.RUN_REPORT_MAX_LIMIT + 1],
)
def test_run_report_rejects_out_of_range_limit(monkeypatch, bad_limit):
    mock_request = Mock()
    monkeypatch.setattr(google_analytics.requests, "request", mock_request)

    result = json.loads(
        google_analytics.google_analytics_run_report(
            "42",
            metrics=["sessions"],
            date_ranges=[{"start_date": "7daysAgo", "end_date": "today"}],
            limit=bad_limit,
        )
    )

    assert result["status"] == "error"
    assert "limit" in result["message"]
    mock_request.assert_not_called()


def test_run_report_rejects_negative_offset(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(google_analytics.requests, "request", mock_request)

    result = json.loads(
        google_analytics.google_analytics_run_report(
            "42",
            metrics=["sessions"],
            date_ranges=[{"start_date": "7daysAgo", "end_date": "today"}],
            offset=-1,
        )
    )

    assert result["status"] == "error"
    assert "offset" in result["message"]
    mock_request.assert_not_called()


@pytest.mark.parametrize(
    "limit",
    [google_analytics.RUN_REPORT_DEFAULT_LIMIT, google_analytics.RUN_REPORT_MAX_LIMIT],
)
def test_run_report_keeps_serialized_response_under_truncation_threshold(
    monkeypatch, limit
):
    """Regression for the sizing bug: a report at both the tool's default and
    max row limit must serialize to under the MCP output truncation threshold
    enforced by src/xagent/core/tools/adapters/vibe/output_filter.py, even
    with dimension values wide enough (~30-40 chars, e.g. a landing-page path)
    to actually stress the boundary — not just the narrow 11-char case."""
    row = {
        "dimensionValues": [
            {"value": f"dim-value-{i:02d}".ljust(35, "x")} for i in range(5)
        ],
        "metricValues": [{"value": str(1000 + i)} for i in range(5)],
    }
    rows = [row] * limit
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "dimensionHeaders": [{"name": f"dim{i}"} for i in range(5)],
                "metricHeaders": [{"name": f"metric{i}"} for i in range(5)],
                "rows": rows,
                "rowCount": len(rows),
            }
        )
    )
    monkeypatch.setattr(google_analytics.requests, "request", mock_request)

    response = google_analytics.google_analytics_run_report(
        "42",
        metrics=[f"metric{i}" for i in range(5)],
        date_ranges=[{"start_date": "7daysAgo", "end_date": "today"}],
        dimensions=[f"dim{i}" for i in range(5)],
        limit=limit,
    )

    result = json.loads(response)
    assert result["status"] == "success"
    assert len(response) < get_tool_max_output_length()


def test_run_report_trims_rows_and_flags_truncated_when_still_over_budget(monkeypatch):
    """Several long-valued dimensions at once (all 5 padded to 35 chars) at
    RUN_REPORT_MAX_LIMIT rows crosses the output threshold even after the
    default/max regression test's mix stays under it. The tool must trim
    rows and set truncated=True rather than return an oversized payload that
    the platform would hard-truncate into invalid JSON."""
    row = {
        "dimensionValues": [
            {"value": f"dim-value-{i:02d}".ljust(35, "x")} for i in range(5)
        ],
        "metricValues": [{"value": str(1000 + i)} for i in range(5)],
    }
    rows = [row] * google_analytics.RUN_REPORT_MAX_LIMIT
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "dimensionHeaders": [{"name": f"dim{i}"} for i in range(5)],
                "metricHeaders": [{"name": f"metric{i}"} for i in range(5)],
                "rows": rows,
                "rowCount": len(rows),
            }
        )
    )
    monkeypatch.setattr(google_analytics.requests, "request", mock_request)

    response = google_analytics.google_analytics_run_report(
        "42",
        metrics=[f"metric{i}" for i in range(5)],
        date_ranges=[{"start_date": "7daysAgo", "end_date": "today"}],
        dimensions=[f"dim{i}" for i in range(5)],
        limit=google_analytics.RUN_REPORT_MAX_LIMIT,
    )

    result = json.loads(response)
    assert result["status"] == "success"
    assert result["truncated"] is True
    assert len(result["rows"]) < google_analytics.RUN_REPORT_MAX_LIMIT
    assert len(response) < get_tool_max_output_length()


def test_run_report_passes_metric_filter_and_offset(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"rowCount": 0}))
    monkeypatch.setattr(google_analytics.requests, "request", mock_request)

    metric_filter = {
        "filter": {
            "fieldName": "sessions",
            "numericFilter": {
                "operation": "GREATER_THAN",
                "value": {"int64Value": "100"},
            },
        }
    }

    google_analytics.google_analytics_run_report(
        "42",
        metrics=["sessions"],
        date_ranges=[{"start_date": "7daysAgo", "end_date": "today"}],
        metric_filter=metric_filter,
        offset=1000,
    )

    body = mock_request.call_args.kwargs["json"]
    assert body["metricFilter"] == metric_filter
    assert body["offset"] == "1000"


@pytest.mark.parametrize(
    ("bad_ranges", "expected_fragment"),
    [
        ([], "at least one"),
        (
            [{"start_date": "7daysAgo", "end_date": "today"}] * 5,
            "at most 4",
        ),
    ],
)
def test_run_report_rejects_bad_date_range_counts(
    monkeypatch, bad_ranges, expected_fragment
):
    mock_request = Mock()
    monkeypatch.setattr(google_analytics.requests, "request", mock_request)

    result = json.loads(
        google_analytics.google_analytics_run_report(
            "42", metrics=["sessions"], date_ranges=bad_ranges
        )
    )

    assert result["status"] == "error"
    assert expected_fragment in result["message"]
    mock_request.assert_not_called()


def test_list_properties_flags_truncation_when_pages_run_out(monkeypatch):
    monkeypatch.setattr(google_analytics, "MAX_ACCOUNT_SUMMARY_PAGES", 1)
    monkeypatch.setattr(
        google_analytics.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "accountSummaries": [
                        {
                            "displayName": "Acme Inc",
                            "propertySummaries": [
                                {"property": "properties/111", "displayName": "Site A"}
                            ],
                        }
                    ],
                    "nextPageToken": "page-2",
                }
            )
        ),
    )

    result = json.loads(google_analytics.google_analytics_list_properties())

    assert result["status"] == "success"
    assert result["truncated"] is True


def test_request_caps_unstructured_error_body(monkeypatch):
    monkeypatch.setattr(
        google_analytics.requests,
        "request",
        Mock(return_value=MockResponse(status_code=502, text="x" * 5000)),
    )

    with pytest.raises(RuntimeError) as excinfo:
        google_analytics._request(
            "GET", f"{google_analytics.ADMIN_API_BASE_URL}/accountSummaries"
        )
    assert len(str(excinfo.value)) < 1200


def test_run_report_rejects_date_range_missing_required_keys(monkeypatch):
    """An empty dict or misnamed keys (e.g. camelCase "startDate") pass the
    tool-signature validation but must fail here with an actionable message
    instead of sending nulls to the API."""
    mock_request = Mock()
    monkeypatch.setattr(google_analytics.requests, "request", mock_request)

    for bad_range in [{}, {"startDate": "7daysAgo", "endDate": "today"}]:
        result = json.loads(
            google_analytics.google_analytics_run_report(
                "42", metrics=["sessions"], date_ranges=[bad_range]
            )
        )
        assert result["status"] == "error"
        assert "start_date" in result["message"]
    mock_request.assert_not_called()


def test_run_report_rejects_invalid_property_id(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(google_analytics.requests, "request", mock_request)

    result = json.loads(
        google_analytics.google_analytics_run_report(
            "42; DROP TABLE",
            metrics=["sessions"],
            date_ranges=[{"start_date": "7daysAgo", "end_date": "today"}],
        )
    )

    assert result["status"] == "error"
    assert "property_id" in result["message"]
    mock_request.assert_not_called()


def test_run_report_returns_error_payload_on_api_failure(monkeypatch):
    monkeypatch.setattr(
        google_analytics.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=400,
                json_data={"error": {"message": "bad metric name"}},
            )
        ),
    )

    result = json.loads(
        google_analytics.google_analytics_run_report(
            "42",
            metrics=["not-real"],
            date_ranges=[{"start_date": "7daysAgo", "end_date": "today"}],
        )
    )

    assert result["status"] == "error"
    assert "bad metric name" in result["message"]
