import asyncio
import json
from unittest.mock import Mock

import pytest

from xagent.web.tools.mcp import employment_hero


class MockResponse:
    def __init__(self, json_data=None, status_code: int = 200, text: str = ""):
        self._json_data = json_data if json_data is not None else {}
        self.status_code = status_code
        self.text = text or (json.dumps(self._json_data) if json_data else "")
        self.content = self.text.encode()

    def json(self):
        return self._json_data


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("EMPLOYMENT_HERO_ACCESS_TOKEN", "access-token")


def test_headers_require_access_token(monkeypatch):
    monkeypatch.delenv("EMPLOYMENT_HERO_ACCESS_TOKEN")

    with pytest.raises(ValueError, match="EMPLOYMENT_HERO_ACCESS_TOKEN"):
        employment_hero._headers()


def test_headers_include_bearer_token():
    assert employment_hero._headers() == {"Authorization": "Bearer access-token"}


def test_request_raises_with_structured_message(monkeypatch):
    # `text` deliberately does NOT contain the expected message: MockResponse's
    # raw-text fallback would otherwise happen to satisfy this test's assertion
    # even if _extract_error_detail's JSON parsing were broken (returning
    # None), since the raw-text fallback path is exercised regardless.
    monkeypatch.setattr(
        employment_hero.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=401,
                json_data={"message": "invalid or expired token"},
                text="raw fallback body",
            )
        ),
    )

    with pytest.raises(RuntimeError, match="invalid or expired token") as excinfo:
        employment_hero._request("GET", "/organisations")
    assert "401" in str(excinfo.value)


def test_request_raises_with_errors_list_of_dicts(monkeypatch):
    # See test_request_raises_with_structured_message for why `text` must not
    # itself contain the expected message.
    monkeypatch.setattr(
        employment_hero.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=422,
                json_data={"errors": [{"message": "date is required"}]},
                text="raw fallback body",
            )
        ),
    )

    with pytest.raises(RuntimeError, match="date is required"):
        employment_hero._request("GET", "/organisations")


def test_request_raises_with_errors_list_of_strings(monkeypatch):
    monkeypatch.setattr(
        employment_hero.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=422,
                json_data={"errors": ["date is required"]},
                text="raw fallback body",
            )
        ),
    )

    with pytest.raises(RuntimeError, match="date is required"):
        employment_hero._request("GET", "/organisations")


def test_request_raises_with_errors_dict_of_field_to_list(monkeypatch):
    monkeypatch.setattr(
        employment_hero.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=422,
                json_data={"errors": {"date": ["is required"]}},
                text="raw fallback body",
            )
        ),
    )

    with pytest.raises(RuntimeError, match="date: is required"):
        employment_hero._request("GET", "/organisations")


def test_request_falls_back_to_raw_text_for_unstructured_error_body(monkeypatch):
    monkeypatch.setattr(
        employment_hero.requests,
        "request",
        Mock(return_value=MockResponse(status_code=500, text="upstream 500")),
    )

    with pytest.raises(RuntimeError, match="upstream 500"):
        employment_hero._request("GET", "/organisations")


def test_request_truncates_long_unstructured_error_body(monkeypatch):
    long_body = "x" * 5000
    monkeypatch.setattr(
        employment_hero.requests,
        "request",
        Mock(return_value=MockResponse(status_code=502, text=long_body)),
    )

    with pytest.raises(RuntimeError) as excinfo:
        employment_hero._request("GET", "/organisations")
    message = str(excinfo.value)
    assert "[truncated]" in message
    assert len(message) < len(long_body)


def test_request_returns_empty_dict_for_204(monkeypatch):
    monkeypatch.setattr(
        employment_hero.requests,
        "request",
        Mock(return_value=MockResponse(status_code=204, text="")),
    )

    assert employment_hero._request("DELETE", "/organisations/org-1") == {}


def test_request_raises_clean_error_for_non_json_200_response(monkeypatch):
    bad_response = Mock(
        status_code=200, content=b"<html>not json</html>", text="<html>not json</html>"
    )
    bad_response.json.side_effect = ValueError("no JSON object could be decoded")
    monkeypatch.setattr(
        employment_hero.requests, "request", Mock(return_value=bad_response)
    )

    with pytest.raises(RuntimeError, match="non-JSON response"):
        employment_hero._request("GET", "/organisations")


def test_list_payload_raises_on_non_dict_data():
    with pytest.raises(RuntimeError, match="unexpected response"):
        employment_hero._list_payload({"data": ["not", "a", "dict"]}, "organisations")


def test_list_payload_returns_dict_data():
    assert employment_hero._list_payload({"data": {"items": []}}, "organisations") == {
        "items": []
    }


def test_unwrap_data_returns_data_key_when_present():
    assert employment_hero._unwrap_data({"data": {"id": "org-1"}}) == {"id": "org-1"}


def test_unwrap_data_passes_through_when_no_data_key():
    assert employment_hero._unwrap_data({"id": "org-1"}) == {"id": "org-1"}


@pytest.mark.parametrize(
    "page_index,item_per_page,expected",
    [
        (1, 20, {"page_index": 1, "item_per_page": 20}),
        (0, 20, {"page_index": 1, "item_per_page": 20}),  # clamped up to 1
        (-5, 20, {"page_index": 1, "item_per_page": 20}),
        (2, 0, {"page_index": 2, "item_per_page": 1}),  # clamped up to 1
        (2, 500, {"page_index": 2, "item_per_page": 100}),  # clamped to max
    ],
)
def test_pagination_params_clamps_out_of_range_values(
    page_index, item_per_page, expected
):
    assert employment_hero._pagination_params(page_index, item_per_page) == expected


def test_list_organisations_requests_expected_path_and_params(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "data": {
                    "items": [{"id": "org-1", "name": "Acme"}],
                    "page_index": 1,
                    "item_per_page": 20,
                    "total_items": 1,
                    "total_pages": 1,
                }
            }
        )
    )
    monkeypatch.setattr(employment_hero.requests, "request", mock_request)

    result = json.loads(employment_hero.employment_hero_list_organisations())

    assert result["status"] == "success"
    assert result["organisations"]["items"] == [{"id": "org-1", "name": "Acme"}]
    call = mock_request.call_args
    assert (
        call.kwargs["url"]
        == f"{employment_hero.EMPLOYMENT_HERO_BASE_URL}/organisations"
    )
    assert call.kwargs["params"] == {"page_index": 1, "item_per_page": 20}
    assert call.kwargs["headers"] == {"Authorization": "Bearer access-token"}


def test_list_organisations_errors_on_non_dict_data_shape(monkeypatch):
    """A live response deviating from the documented {"items": [...]}
    envelope (e.g. `data` is a bare list) must surface as a clear error --
    success_with_capped_dict only caps output size for a dict payload, so
    silently accepting a non-dict shape would skip that protection."""
    mock_request = Mock(
        return_value=MockResponse(json_data={"data": ["not", "a", "dict"]})
    )
    monkeypatch.setattr(employment_hero.requests, "request", mock_request)

    result = json.loads(employment_hero.employment_hero_list_organisations())

    assert result["status"] == "error"


def test_list_employees_includes_member_type_only_when_provided(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"data": {"items": []}}))
    monkeypatch.setattr(employment_hero.requests, "request", mock_request)

    json.loads(employment_hero.employment_hero_list_employees("org-1"))

    call = mock_request.call_args
    assert (
        call.kwargs["url"]
        == f"{employment_hero.EMPLOYMENT_HERO_BASE_URL}/organisations/org-1/employees"
    )
    assert "member_type" not in call.kwargs["params"]

    json.loads(
        employment_hero.employment_hero_list_employees(
            "org-1", member_type="contractor"
        )
    )
    call = mock_request.call_args
    assert call.kwargs["params"]["member_type"] == "contractor"


def test_list_employees_rejects_invalid_member_type(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"data": {"items": []}}))
    monkeypatch.setattr(employment_hero.requests, "request", mock_request)

    result = json.loads(
        employment_hero.employment_hero_list_employees("org-1", member_type="staff")
    )

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_list_employees_forwards_query_without_loading_the_whole_directory(
    monkeypatch,
):
    mock_request = Mock(return_value=MockResponse(json_data={"data": {"items": []}}))
    monkeypatch.setattr(employment_hero.requests, "request", mock_request)

    json.loads(
        employment_hero.employment_hero_list_employees(
            "org-1", item_per_page=20, query="  Summer Smith  "
        )
    )

    assert mock_request.call_args.kwargs["params"] == {
        "page_index": 1,
        "item_per_page": 20,
        "query": "Summer Smith",
    }


def test_get_employee_url_encodes_path_segments(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"data": {"id": "emp-1"}}))
    monkeypatch.setattr(employment_hero.requests, "request", mock_request)

    result = json.loads(employment_hero.employment_hero_get_employee("org 1", "emp/1"))

    assert result["status"] == "success"
    assert result["employee"] == {"id": "emp-1"}
    call = mock_request.call_args
    assert (
        call.kwargs["url"]
        == f"{employment_hero.EMPLOYMENT_HERO_BASE_URL}/organisations/org%201/employees/emp%2F1"
    )


def test_get_employee_errors_on_null_payload(monkeypatch):
    """A 200 with a null `data` (e.g. a soft-deleted or access-restricted
    record) must surface as a clear error, not a "success" envelope
    wrapping employee: null."""
    mock_request = Mock(return_value=MockResponse(json_data={"data": None}))
    monkeypatch.setattr(employment_hero.requests, "request", mock_request)

    result = json.loads(employment_hero.employment_hero_get_employee("org-1", "emp-1"))

    assert result["status"] == "error"


def test_list_team_employees_requests_nested_path(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"data": {"items": []}}))
    monkeypatch.setattr(employment_hero.requests, "request", mock_request)

    json.loads(employment_hero.employment_hero_list_team_employees("org-1", "team-1"))

    call = mock_request.call_args
    assert (
        call.kwargs["url"]
        == f"{employment_hero.EMPLOYMENT_HERO_BASE_URL}/organisations/org-1/teams/team-1/employees"
    )


def test_list_timesheet_entries_includes_date_range_only_when_provided(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"data": {"items": []}}))
    monkeypatch.setattr(employment_hero.requests, "request", mock_request)

    json.loads(employment_hero.employment_hero_list_timesheet_entries("org-1", "emp-1"))
    call = mock_request.call_args
    assert (
        call.kwargs["url"]
        == f"{employment_hero.EMPLOYMENT_HERO_BASE_URL}/organisations/org-1/employees/emp-1/timesheet_entries"
    )
    assert "start_date" not in call.kwargs["params"]
    assert "end_date" not in call.kwargs["params"]

    json.loads(
        employment_hero.employment_hero_list_timesheet_entries(
            "org-1", "emp-1", start_date="01/08/2026", end_date="31/08/2026"
        )
    )
    call = mock_request.call_args
    assert call.kwargs["params"]["start_date"] == "01/08/2026"
    assert call.kwargs["params"]["end_date"] == "31/08/2026"


@pytest.mark.parametrize(
    "call_tool",
    [
        lambda: employment_hero.employment_hero_list_employees("org-1"),
        lambda: employment_hero.employment_hero_list_teams("org-1"),
        lambda: employment_hero.employment_hero_list_team_employees("org-1", "team-1"),
        lambda: employment_hero.employment_hero_list_timesheet_entries(
            "org-1", "emp-1"
        ),
    ],
    ids=[
        "list_employees",
        "list_teams",
        "list_team_employees",
        "list_timesheet_entries",
    ],
)
def test_list_tools_error_on_non_dict_data_shape(monkeypatch, call_tool):
    """The non-dict-data-shape guard (_list_payload) added for
    employment_hero_list_organisations must actually be wired into every
    other list tool's own call site too, not just exercised via the shared
    helper's unit tests -- this end-to-end check would catch a future
    tool accidentally reverting to a bare _unwrap_data(result) call, or
    passing the wrong resource/result variable into _list_payload."""
    mock_request = Mock(
        return_value=MockResponse(json_data={"data": ["not", "a", "dict"]})
    )
    monkeypatch.setattr(employment_hero.requests, "request", mock_request)

    result = json.loads(call_tool())

    assert result["status"] == "error"


def test_tool_functions_return_error_envelope_on_failure(monkeypatch):
    monkeypatch.setattr(
        employment_hero.requests,
        "request",
        Mock(
            return_value=MockResponse(status_code=403, text='{"message": "forbidden"}')
        ),
    )

    result = json.loads(employment_hero.employment_hero_list_organisations())

    assert result["status"] == "error"
    assert "forbidden" in result["message"]


def test_request_adds_actionable_scope_hint_to_403(monkeypatch):
    monkeypatch.setattr(
        employment_hero.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=403, json_data={"message": "forbidden"}
            )
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        employment_hero._request(
            "GET",
            "/organisations/org-1/leave_categories",
            required_scope=employment_hero.LEAVE_CATEGORIES_LIST_SCOPE,
        )

    message = str(excinfo.value)
    assert employment_hero.LEAVE_CATEGORIES_LIST_SCOPE in message
    assert "disconnect and reconnect" in message


def test_list_leave_categories_requests_expected_path(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "data": {
                    "items": [
                        {
                            "id": "category-1",
                            "name": "Annual Leave",
                            "unit_type": "days",
                        }
                    ],
                    "total_items": 1,
                }
            }
        )
    )
    monkeypatch.setattr(employment_hero.requests, "request", mock_request)

    result = json.loads(employment_hero.employment_hero_list_leave_categories("org/1"))

    assert result["status"] == "success"
    assert result["leave_categories"]["items"][0]["name"] == "Annual Leave"
    assert (
        mock_request.call_args.kwargs["url"]
        == f"{employment_hero.EMPLOYMENT_HERO_BASE_URL}/organisations/org%2F1/leave_categories"
    )


def test_get_leave_balances_requests_employee_path(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "data": {
                    "items": [
                        {
                            "leave_category_name": "Annual Leave",
                            "available_balance": 10,
                        }
                    ]
                }
            }
        )
    )
    monkeypatch.setattr(employment_hero.requests, "request", mock_request)

    result = json.loads(
        employment_hero.employment_hero_get_leave_balances("org-1", "emp/1")
    )

    assert result["status"] == "success"
    assert (
        mock_request.call_args.kwargs["url"]
        == f"{employment_hero.EMPLOYMENT_HERO_BASE_URL}/organisations/org-1/employees/emp%2F1/leave_balances"
    )


def test_list_leave_requests_supports_manager_filters(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"data": {"items": []}}))
    monkeypatch.setattr(employment_hero.requests, "request", mock_request)

    result = json.loads(
        employment_hero.employment_hero_list_leave_requests(
            "org-1",
            statuses=[" pending ", "approved"],
            start_date="2026-11-01",
            end_date="2026-11-30",
            leave_category_id="category/1",
            page_index=2,
            item_per_page=50,
        )
    )

    assert result["status"] == "success"
    call = mock_request.call_args
    assert (
        call.kwargs["url"]
        == f"{employment_hero.EMPLOYMENT_HERO_BASE_URL}/organisations/org-1/leave_requests"
    )
    assert call.kwargs["params"] == {
        "page_index": 2,
        "item_per_page": 50,
        "statuses": ["pending", "approved"],
        "start_date": "2026-11-01",
        "end_date": "2026-11-30",
        "leave_category_id": "category/1",
    }


def test_list_leave_requests_normalizes_and_validates_statuses(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"data": {"items": []}}))
    monkeypatch.setattr(employment_hero.requests, "request", mock_request)

    result = json.loads(
        employment_hero.employment_hero_list_leave_requests(
            "org-1", statuses=[" Pending ", "DECLINED"]
        )
    )

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["params"]["statuses"] == [
        "pending",
        "declined",
    ]

    mock_request.reset_mock()
    result = json.loads(
        employment_hero.employment_hero_list_leave_requests(
            "org-1", statuses=["cancelled"]
        )
    )

    assert result["status"] == "error"
    assert "pending, approved, declined" in result["message"]
    mock_request.assert_not_called()


def test_list_leave_requests_supports_employee_view(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"data": {"items": []}}))
    monkeypatch.setattr(employment_hero.requests, "request", mock_request)

    json.loads(
        employment_hero.employment_hero_list_leave_requests(
            "org-1", employee_id="emp/1"
        )
    )

    assert (
        mock_request.call_args.kwargs["url"]
        == f"{employment_hero.EMPLOYMENT_HERO_BASE_URL}/organisations/org-1/employees/emp%2F1/leave_requests"
    )


@pytest.mark.parametrize(
    "start_date,end_date,message",
    [
        ("2026-02-30", "2026-03-01", "valid YYYY-MM-DD"),
        ("2026-11-02", "2026-11-01", "on or after"),
    ],
)
def test_list_leave_requests_rejects_invalid_date_ranges(
    monkeypatch, start_date, end_date, message
):
    mock_request = Mock()
    monkeypatch.setattr(employment_hero.requests, "request", mock_request)

    result = json.loads(
        employment_hero.employment_hero_list_leave_requests(
            "org-1", start_date=start_date, end_date=end_date
        )
    )

    assert result["status"] == "error"
    assert message in result["message"]
    mock_request.assert_not_called()


def test_list_leave_requests_accepts_one_sided_date_filter(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"data": {"items": []}}))
    monkeypatch.setattr(employment_hero.requests, "request", mock_request)

    result = json.loads(
        employment_hero.employment_hero_list_leave_requests(
            "org-1", start_date="2026-11-01"
        )
    )

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["params"]["start_date"] == "2026-11-01"
    assert "end_date" not in mock_request.call_args.kwargs["params"]


def test_create_leave_request_sends_validated_json_body(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "data": {
                    "id": "leave-1",
                    "employee_id": "emp-1",
                    "status": "Pending",
                }
            }
        )
    )
    monkeypatch.setattr(employment_hero.requests, "request", mock_request)

    result = json.loads(
        employment_hero.employment_hero_create_leave_request(
            "org-1",
            "emp-1",
            "category-1",
            "2026-12-24",
            "2026-12-25",
            comment="Holiday",
            hours_per_day=[{"date": "2026-12-24", "hours": 7.6}],
        )
    )

    assert result["status"] == "success"
    assert result["leave_request"]["id"] == "leave-1"
    call = mock_request.call_args
    assert call.kwargs["method"] == "POST"
    assert call.kwargs["json"] == {
        "leave_category_id": "category-1",
        "start_date": "2026-12-24",
        "end_date": "2026-12-25",
        "comment": "Holiday",
        "hours_per_day": [{"date": "2026-12-24", "hours": 7.6}],
    }


@pytest.mark.parametrize(
    "hours_per_day,message",
    [
        ([{"date": "2026-12-23", "hours": 7.6}], "must fall within"),
        (
            [
                {"date": "2026-12-24", "hours": 7.6},
                {"date": "2026-12-24", "hours": 4},
            ],
            "duplicate date",
        ),
        ([{"date": "2026-12-24", "hours": -1}], "non-negative number"),
        ([{"date": "2026-12-24", "hours": float("nan")}], "non-negative number"),
        ([{"date": "2026-12-24"}], "exactly 'date' and 'hours'"),
    ],
)
def test_create_leave_request_rejects_invalid_hours_per_day(
    monkeypatch, hours_per_day, message
):
    mock_request = Mock()
    monkeypatch.setattr(employment_hero.requests, "request", mock_request)

    result = json.loads(
        employment_hero.employment_hero_create_leave_request(
            "org-1",
            "emp-1",
            "category-1",
            "2026-12-24",
            "2026-12-25",
            hours_per_day=hours_per_day,
        )
    )

    assert result["status"] == "error"
    assert message in result["message"]
    mock_request.assert_not_called()


def test_create_leave_request_does_not_report_empty_response_as_success(monkeypatch):
    monkeypatch.setattr(
        employment_hero.requests,
        "request",
        Mock(return_value=MockResponse(status_code=204, text="")),
    )

    result = json.loads(
        employment_hero.employment_hero_create_leave_request(
            "org-1", "emp-1", "category-1", "2026-12-24", "2026-12-25"
        )
    )

    assert result["status"] == "error"
    assert "unexpected response" in result["message"]


def test_approve_leave_request_requests_expected_path(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"data": {"id": "leave-1", "status": "Approved"}}
        )
    )
    monkeypatch.setattr(employment_hero.requests, "request", mock_request)

    result = json.loads(
        employment_hero.employment_hero_approve_leave_request(
            "org-1", "emp/1", "leave/1"
        )
    )

    assert result["status"] == "success"
    assert result["leave_request"]["status"] == "Approved"
    call = mock_request.call_args
    assert call.kwargs["method"] == "POST"
    assert "json" not in call.kwargs
    assert (
        call.kwargs["url"]
        == f"{employment_hero.EMPLOYMENT_HERO_BASE_URL}/organisations/org-1/employees/emp%2F1/leave_requests/leave%2F1/approve"
    )


def test_leave_tools_are_discoverable_with_expected_annotations_and_search_schema():
    tools = {tool.name: tool for tool in asyncio.run(employment_hero.mcp.list_tools())}

    expected_leave_tools = {
        "employment_hero_list_leave_categories",
        "employment_hero_get_leave_balances",
        "employment_hero_list_leave_requests",
        "employment_hero_create_leave_request",
        "employment_hero_approve_leave_request",
    }
    assert expected_leave_tools <= set(tools)
    assert "query" in tools["employment_hero_list_employees"].inputSchema["properties"]

    for name in (
        "employment_hero_list_leave_categories",
        "employment_hero_get_leave_balances",
        "employment_hero_list_leave_requests",
    ):
        assert tools[name].annotations is not None
        assert tools[name].annotations.readOnlyHint is True

    create = tools["employment_hero_create_leave_request"].annotations
    assert create is not None
    assert create.destructiveHint is False
    assert create.idempotentHint is False

    approve = tools["employment_hero_approve_leave_request"].annotations
    assert approve is not None
    assert approve.destructiveHint is True
    assert approve.idempotentHint is True
