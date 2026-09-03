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
