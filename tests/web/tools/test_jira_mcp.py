import json
import logging
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import jira


class MockResponse:
    def __init__(
        self,
        json_data=None,
        status_code: int = 200,
        text: str = "",
        url: str = "",
        headers: dict | None = None,
    ):
        self._json_data = json_data if json_data is not None else {}
        self.status_code = status_code
        self.text = text or (
            json.dumps(self._json_data) if json_data is not None else ""
        )
        self.content = self.text.encode()
        self.url = url
        self.headers = headers or {}

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} Client Error for url: {self.url}", response=self
            )


_SITE_A = {"id": "site-a", "name": "Acme", "url": "https://acme.atlassian.net"}
_SITE_B = {"id": "site-b", "name": "Beta", "url": "https://beta.atlassian.net"}


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("JIRA_ACCESS_TOKEN", "access-token")


def test_headers_require_access_token(monkeypatch):
    monkeypatch.delenv("JIRA_ACCESS_TOKEN")

    with pytest.raises(ValueError, match="JIRA_ACCESS_TOKEN"):
        jira._headers()


def test_headers_include_bearer_token():
    headers = jira._headers()
    assert headers["Authorization"] == "Bearer access-token"
    assert headers["Accept"] == "application/json"


def test_request_absolute_raises_with_structured_error_messages(monkeypatch):
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=400,
                json_data={
                    "errorMessages": ["The issue no longer exists."],
                    "errors": {"assignee": "User does not exist"},
                },
            )
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        jira._request_absolute("GET", "https://api.atlassian.com/me")

    assert "no longer exists" in str(excinfo.value)
    assert "assignee: User does not exist" in str(excinfo.value)


def test_request_absolute_truncates_unstructured_error_body(monkeypatch):
    long_body = "x" * 5000
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(return_value=MockResponse(status_code=500, text=long_body)),
    )

    with pytest.raises(RuntimeError) as excinfo:
        jira._request_absolute("GET", "https://api.atlassian.com/me")

    assert "[truncated]" in str(excinfo.value)
    assert len(str(excinfo.value)) < len(long_body)


def test_request_absolute_truncates_large_structured_error_body(monkeypatch):
    long_messages = [f"error {i}" for i in range(500)]
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=400, json_data={"errorMessages": long_messages}
            )
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        jira._request_absolute("GET", "https://api.atlassian.com/me")

    assert "[truncated]" in str(excinfo.value)
    assert len(str(excinfo.value)) < len("; ".join(long_messages))


def test_request_absolute_retries_once_on_429_with_retry_after(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(jira.time, "sleep", lambda s: sleep_calls.append(s))
    mock_request = Mock(
        side_effect=[
            MockResponse(status_code=429, headers={"Retry-After": "2"}),
            MockResponse(json_data={"ok": True}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = jira._request_absolute("GET", "https://api.atlassian.com/me")

    assert result == {"ok": True}
    assert sleep_calls == [2]
    assert mock_request.call_count == 2


def test_request_absolute_does_not_retry_429_twice(monkeypatch):
    monkeypatch.setattr(jira.time, "sleep", lambda s: None)
    mock_request = Mock(
        return_value=MockResponse(status_code=429, headers={"Retry-After": "1"})
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    with pytest.raises(RuntimeError):
        jira._request_absolute("GET", "https://api.atlassian.com/me")

    assert mock_request.call_count == 2


def test_resolve_cloud_id_passes_through_explicit_value(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(jira.requests, "request", mock_request)

    assert jira._resolve_cloud_id("explicit-id") == "explicit-id"
    mock_request.assert_not_called()


def test_resolve_cloud_id_auto_resolves_single_site(monkeypatch):
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(return_value=MockResponse(json_data=[_SITE_A])),
    )

    assert jira._resolve_cloud_id("") == "site-a"


def test_resolve_cloud_id_raises_when_multiple_sites(monkeypatch):
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(return_value=MockResponse(json_data=[_SITE_A, _SITE_B])),
    )

    with pytest.raises(ValueError, match="Multiple Jira sites"):
        jira._resolve_cloud_id("")


def test_resolve_cloud_id_raises_when_no_sites(monkeypatch):
    monkeypatch.setattr(
        jira.requests, "request", Mock(return_value=MockResponse(json_data=[]))
    )

    with pytest.raises(ValueError, match="No accessible Jira sites"):
        jira._resolve_cloud_id("")


def test_accessible_resources_raises_on_non_list_response(monkeypatch):
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"unexpected": "shape"})),
    )

    with pytest.raises(ValueError, match="Unexpected response format"):
        jira._accessible_resources()


def test_resolve_cloud_id_multiple_sites_message_handles_non_dict_entries(monkeypatch):
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(return_value=MockResponse(json_data=["not-a-dict", "also-not-a-dict"])),
    )

    with pytest.raises(ValueError, match="details unavailable"):
        jira._resolve_cloud_id("")


def test_request_builds_url_with_resolved_cloud_id(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"id": "10001", "key": "ENG-1"})
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    jira._request("GET", "site-a", "/rest/api/2/issue/ENG-1")

    assert mock_request.call_args.kwargs["url"] == (
        "https://api.atlassian.com/ex/jira/site-a/rest/api/2/issue/ENG-1"
    )


def test_request_percent_encodes_cloud_id_in_url(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={}))
    monkeypatch.setattr(jira.requests, "request", mock_request)

    jira._request("GET", "site/../a", "/rest/api/2/issue/ENG-1")

    assert mock_request.call_args.kwargs["url"] == (
        "https://api.atlassian.com/ex/jira/site%2F..%2Fa/rest/api/2/issue/ENG-1"
    )


def test_get_issue_percent_encodes_issue_key_in_path(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data={"key": "ENG-1"}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    jira.jira_get_issue("ENG-1/../secrets?x=1")

    issue_call = mock_request.call_args_list[1]
    assert issue_call.kwargs["url"] == (
        "https://api.atlassian.com/ex/jira/site-a/rest/api/2/issue/"
        "ENG-1%2F..%2Fsecrets%3Fx%3D1"
    )


def test_path_segment_rejects_bare_dot_and_dot_dot():
    # "." and ".." are always-unreserved per RFC 3986, so quote() never
    # touches them, and requests/urllib3 normalize dot-segments out of
    # the final URL before sending -- percent-encoding alone can't close
    # this off, so the value must be rejected outright instead.
    with pytest.raises(ValueError, match=r"\.\."):
        jira._path_segment("..")
    with pytest.raises(ValueError, match=r"\."):
        jira._path_segment(".")
    # A value that merely CONTAINS ".." (not equal to it) is a normal,
    # legitimately encodable value -- only an exact match is rejected.
    assert jira._path_segment("ENG-1/../secrets") == "ENG-1%2F..%2Fsecrets"


def test_get_issue_rejects_bare_dot_dot_issue_key(monkeypatch):
    mock_request = Mock(side_effect=[MockResponse(json_data=[_SITE_A])])
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_get_issue(".."))

    assert result["status"] == "error"
    # _issue_path(issue_key) is built (and raises) before _request is
    # ever called, so no network request -- not even the
    # accessible-resources lookup used to resolve cloud_id -- goes out
    # at all.
    mock_request.assert_not_called()


def test_list_accessible_sites_returns_sites(monkeypatch):
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(return_value=MockResponse(json_data=[_SITE_A, _SITE_B])),
    )

    result = json.loads(jira.jira_list_accessible_sites())

    assert result["status"] == "success"
    assert result["sites"] == [_SITE_A, _SITE_B]


def test_get_current_user_returns_profile(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "account_id": "u1",
                "email": "ada@example.com",
                "name": "Ada",
                "picture": "https://example.com/ada.png",
            }
        )
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_get_current_user())

    assert result["status"] == "success"
    assert result["user"]["email"] == "ada@example.com"
    assert mock_request.call_args.kwargs["url"] == "https://api.atlassian.com/me"


def test_get_current_user_returns_error_payload_on_failure(monkeypatch):
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=401, json_data={"errorMessages": ["Unauthorized"]}
            )
        ),
    )

    result = json.loads(jira.jira_get_current_user())

    assert result["status"] == "error"
    assert "Unauthorized" in result["message"]


def test_list_projects_uses_resolved_cloud_id(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(
                json_data={
                    "values": [{"id": "1", "key": "ENG", "name": "Engineering"}],
                    "isLast": True,
                }
            ),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_list_projects())

    assert result["status"] == "success"
    assert result["projects"] == [{"id": "1", "key": "ENG", "name": "Engineering"}]
    assert result["truncated"] is False
    assert result["next_start_at"] is None
    project_call = mock_request.call_args_list[1]
    assert project_call.kwargs["url"] == (
        "https://api.atlassian.com/ex/jira/site-a/rest/api/2/project/search"
    )
    assert project_call.kwargs["params"]["startAt"] == 0


def test_list_projects_reports_next_start_at_when_truncated(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(
                json_data={
                    "values": [{"id": "1", "key": "ENG", "name": "Engineering"}],
                    "isLast": False,
                }
            ),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_list_projects(start_at=5))

    assert result["truncated"] is True
    assert result["next_start_at"] == 6
    project_call = mock_request.call_args_list[1]
    assert project_call.kwargs["params"]["startAt"] == 5


def test_list_projects_empty_page_never_repeats_offset(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data={"values": [], "isLast": False}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_list_projects(start_at=5))

    assert result["truncated"] is False
    assert result["next_start_at"] is None


def test_search_issues_sends_jql_and_reports_next_page_token(monkeypatch):
    # No isLast in the mock on purpose: the enhanced-search endpoint's
    # pagination signal is nextPageToken presence, and isLast is not
    # guaranteed to appear in the response.
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(
                json_data={
                    "issues": [{"key": "ENG-1", "fields": {"summary": "Bug"}}],
                    "nextPageToken": "token-2",
                }
            ),
            MockResponse(json_data={"count": 42}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_search_issues("project = ENG"))

    assert result["status"] == "success"
    assert result["issues"][0]["key"] == "ENG-1"
    assert result["issues"][0]["summary"] == "Bug"
    assert result["returned_count"] == 1
    assert result["total_count"] == 42
    assert result["truncated"] is True
    assert result["next_page_token"] == "token-2"
    search_call = mock_request.call_args_list[1]
    assert search_call.kwargs["url"] == (
        "https://api.atlassian.com/ex/jira/site-a/rest/api/3/search/jql"
    )
    assert search_call.kwargs["params"]["jql"] == "project = ENG"
    count_call = mock_request.call_args_list[2]
    assert count_call.kwargs["url"] == (
        "https://api.atlassian.com/ex/jira/site-a/rest/api/3/search/approximate-count"
    )
    assert count_call.kwargs["json"] == {"jql": "project = ENG"}


def test_search_issues_passes_next_page_token_when_provided(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(
                json_data={
                    "issues": [{"key": "ENG-2", "fields": {"summary": "Bug 2"}}],
                    "isLast": True,
                }
            ),
            MockResponse(json_data={"count": 1}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(
        jira.jira_search_issues("project = ENG", next_page_token="token-2")
    )

    assert result["status"] == "success"
    assert result["truncated"] is False
    assert result["next_page_token"] is None
    search_call = mock_request.call_args_list[1]
    assert search_call.kwargs["params"]["nextPageToken"] == "token-2"


def test_search_issues_summarizes_issue_fields(monkeypatch):
    raw_issue = {
        "expand": "renderedFields,names,schema,operations,editmeta,changelog",
        "id": "34280",
        "self": "https://api.atlassian.com/ex/jira/site-a/rest/api/3/issue/34280",
        "key": "DW-782",
        "fields": {
            "summary": "[Connectify][6thman] Persist customer configuration",
            "status": {
                "name": "待办",
                "id": "10004",
                "iconUrl": "https://api.atlassian.com/.../10004",
                "statusCategory": {"id": 2, "name": "To Do", "key": "new"},
            },
            "assignee": {
                "accountId": "712020:98fdf2ea-b760-410a-ac9e-43b4f4d5038a",
                "displayName": "jiarongling",
                "emailAddress": "jiarongling@example.com",
                "avatarUrls": {"48x48": "https://secure.gravatar.com/avatar/..."},
            },
            "priority": {
                "name": "P2 - High",
                "iconUrl": "https://api.atlassian.com/.../priority",
            },
            "issuetype": {"name": "任务", "avatarId": 10318, "subtask": False},
            "project": {
                "id": "10033",
                "key": "DW",
                "name": "Datapel WMS",
                "avatarUrls": {"48x48": "https://api.atlassian.com/.../avatar"},
            },
            "parent": {"key": "DW-746", "fields": {"summary": "Epic"}},
            "resolution": None,
            "labels": ["connectify", "connectify-uplift"],
            "created": "2026-09-09T10:00:00.000+0000",
            "updated": "2026-09-11T19:57:00.000+0000",
        },
    }
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data={"issues": [raw_issue], "isLast": True}),
            MockResponse(json_data={"count": 1}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_search_issues("project = DW"))

    assert result["issues"] == [
        {
            "key": "DW-782",
            "summary": "[Connectify][6thman] Persist customer configuration",
            "status": "待办",
            "status_category": "To Do",
            "assignee": {
                "account_id": "712020:98fdf2ea-b760-410a-ac9e-43b4f4d5038a",
                "display_name": "jiarongling",
            },
            "priority": "P2 - High",
            "issue_type": "任务",
            "project_key": "DW",
            "labels": ["connectify", "connectify-uplift"],
            "parent_key": "DW-746",
            "resolution": None,
            "created": "2026-09-09T10:00:00.000+0000",
            "updated": "2026-09-11T19:57:00.000+0000",
        }
    ]
    # None of the URL/avatar/icon clutter that bloats a raw issue to
    # ~3-4 KB should survive into the summarized shape.
    assert "avatarUrls" not in json.dumps(result)
    assert "self" not in result["issues"][0]
    assert "expand" not in result["issues"][0]


def test_search_issues_summarizes_resolved_issue(monkeypatch):
    # The comprehensive fixture above only exercises resolution=None
    # (unresolved). _summarize_issue reads resolution.get("name") for the
    # resolved case, which a null-only fixture can never catch a
    # regression in (e.g. reading the wrong key, or always emitting null).
    raw_issue = {
        "key": "DW-785",
        "fields": {
            "summary": "Identify /api/dev consumers",
            "status": {"name": "Done", "statusCategory": {"name": "Done"}},
            "resolution": {"id": "10000", "name": "Fixed"},
        },
    }
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data={"issues": [raw_issue], "isLast": True}),
            MockResponse(json_data={"count": 1}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_search_issues("project = DW"))

    assert result["issues"][0]["resolution"] == "Fixed"


def test_search_issues_total_count_is_none_when_count_endpoint_fails(monkeypatch):
    # Approximate-count is only called when the first page has more
    # results beyond it (a nextPageToken) -- a final page's count is
    # exact for free, so this needs a page with one to actually reach
    # the count endpoint at all.
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(
                json_data={
                    "issues": [{"key": "ENG-1", "fields": {"summary": "ok"}}],
                    "nextPageToken": "next-token",
                }
            ),
            MockResponse(json_data={"errorMessages": ["bad jql"]}, status_code=400),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_search_issues("text ~ Connectify"))

    assert result["status"] == "success"
    assert result["total_count"] is None
    assert result["returned_count"] == 1


def _raw_issue_with_summary(key: str, summary_length: int):
    return {
        "key": key,
        "fields": {"summary": "x" * summary_length, "status": {"name": "To Do"}},
    }


def test_search_issues_retries_with_smaller_page_when_over_output_budget(monkeypatch):
    # 20 issues is the projected page at the default limit (50); 3 issues
    # is what the _RETRY_PAGE_SIZE (10) retry happens to return here.
    large_page = {
        "issues": [_raw_issue_with_summary(f"ENG-{i}", 200) for i in range(20)],
        "nextPageToken": "orig-next-token",
    }
    small_page = {
        "issues": [_raw_issue_with_summary(f"ENG-{i}", 200) for i in range(3)],
        "nextPageToken": "retry-next-token",
    }
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data=large_page),
            MockResponse(json_data=small_page),
            MockResponse(json_data={"count": 500}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)
    # Small enough that the 20-issue page overflows but the 3-issue retry
    # page fits.
    monkeypatch.setattr(jira, "get_tool_max_output_length", lambda: 1500)

    result = json.loads(jira.jira_search_issues("project = ENG"))

    assert result["status"] == "success"
    assert result["returned_count"] == 3
    assert result["total_count"] == 500
    assert result["next_page_token"] == "retry-next-token"
    first_search_call = mock_request.call_args_list[1]
    retry_search_call = mock_request.call_args_list[2]
    assert retry_search_call.kwargs["params"]["maxResults"] == jira._RETRY_PAGE_SIZE
    # Both requests must use the SAME incoming next_page_token (absent
    # here -- the caller passed none), not Jira's nextPageToken from the
    # oversized first page: that token points past all 20 originally
    # fetched issues, so retrying with it would skip the ones dropped
    # here rather than actually returning a smaller, complete page.
    assert "nextPageToken" not in first_search_call.kwargs["params"]
    assert "nextPageToken" not in retry_search_call.kwargs["params"]


def test_search_issues_returns_bounded_error_when_minimal_page_still_too_big(
    monkeypatch,
):
    # One issue whose summary alone is bigger than the budget -- every
    # candidate size in _page_size_candidates (default limit=50 -> 50,
    # 10, 1) gets a page this oversized, so all three attempts must be
    # exhausted (no count call: fit is never reached). The budget (100)
    # is small enough to force the error message itself to be truncated,
    # but big enough to still hold a real (shortened) message -- the
    # fallback error response must itself respect the configured cap,
    # not just the search page it's reporting on.
    page = {
        "issues": [_raw_issue_with_summary("ENG-1", 50)],
        "nextPageToken": "token",
    }
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data=page),
            MockResponse(json_data=page),
            MockResponse(json_data=page),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)
    monkeypatch.setattr(jira, "get_tool_max_output_length", lambda: 100)

    raw_response = jira.jira_search_issues("project = ENG")
    result = json.loads(raw_response)

    assert len(raw_response) <= 100
    assert result["status"] == "error"
    assert "output limit" in result["message"]
    assert mock_request.call_count == 4


def test_search_issues_bounded_error_degrades_to_minimal_envelope_below_overhead(
    monkeypatch,
):
    # A configured cap smaller than even the shortest possible error
    # envelope ({"status": "error"}, 19 chars) can't hold a "message"
    # key at all -- there's no valid JSON this tool could return that's
    # both under budget and carries explanatory text, so the best it can
    # do is the smallest valid envelope rather than crash or emit
    # invalid JSON.
    page = {
        "issues": [_raw_issue_with_summary("ENG-1", 50)],
        "nextPageToken": "token",
    }
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data=page),
            MockResponse(json_data=page),
            MockResponse(json_data=page),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)
    monkeypatch.setattr(jira, "get_tool_max_output_length", lambda: 5)

    raw_response = jira.jira_search_issues("project = ENG")
    result = json.loads(raw_response)

    assert result == {"status": "error"}


def test_search_issues_cursor_stuck_error_is_also_bounded(monkeypatch):
    # The pagination-cursor-stuck error is a second, independent _error
    # call site with the same unmeasured-against-the-cap gap the
    # minimal-page-overflow error had -- must be fixed the same way, not
    # just at the one call site a review happened to point at. An empty
    # page (129 chars serialized) fits the 130-char budget so the "does
    # the page fit" check passes and this error path is actually
    # reached; the full cursor-stuck message (136 chars) doesn't, so the
    # response must be the truncated form instead.
    page = {"issues": [], "nextPageToken": "same-token"}
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data=page),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)
    monkeypatch.setattr(jira, "get_tool_max_output_length", lambda: 130)

    raw_response = jira.jira_search_issues(
        "project = ENG", next_page_token="same-token"
    )
    result = json.loads(raw_response)

    assert len(raw_response) <= 130
    assert result["status"] == "error"
    assert "did not advance" in result["message"]


def test_page_size_candidates_include_a_floor_below_a_small_limit():
    # A caller-requested limit at or below _RETRY_PAGE_SIZE must still
    # get a chance at an even smaller page instead of skipping straight
    # to "even at a minimal page size" without ever trying one.
    assert jira._page_size_candidates(3) == (3, 1)
    assert jira._page_size_candidates(jira._RETRY_PAGE_SIZE) == (
        jira._RETRY_PAGE_SIZE,
        1,
    )
    assert jira._page_size_candidates(50) == (50, jira._RETRY_PAGE_SIZE, 1)
    assert jira._page_size_candidates(1) == (1,)


def test_search_issues_tries_a_size_of_one_when_small_limit_overflows(monkeypatch):
    # limit=3 (below _RETRY_PAGE_SIZE) must still fall back to size=1
    # rather than erroring out after only the size-3 attempt.
    oversized_page = {
        "issues": [_raw_issue_with_summary("ENG-1", 200)],
        "nextPageToken": None,
    }
    fitting_page = {"issues": [{"key": "ENG-2", "fields": {"summary": "ok"}}]}
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data=oversized_page),
            MockResponse(json_data=fitting_page),
            MockResponse(json_data={"count": 1}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)
    monkeypatch.setattr(jira, "get_tool_max_output_length", lambda: 400)

    result = json.loads(jira.jira_search_issues("project = ENG", limit=3))

    assert result["status"] == "success"
    assert result["returned_count"] == 1
    second_search_call = mock_request.call_args_list[2]
    assert second_search_call.kwargs["params"]["maxResults"] == 1


def test_search_issues_fails_on_non_advancing_pagination_cursor(monkeypatch):
    page = {
        "issues": [{"key": "ENG-1", "fields": {"summary": "ok"}}],
        "nextPageToken": "same-token",
    }
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data=page),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(
        jira.jira_search_issues("project = ENG", next_page_token="same-token")
    )

    assert result["status"] == "error"
    assert "did not advance" in result["message"]


def test_search_issues_skips_approximate_count_on_later_pages(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data={"issues": [], "isLast": True}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(
        jira.jira_search_issues("project = ENG", next_page_token="page-2-token")
    )

    assert result["status"] == "success"
    assert result["total_count"] is None
    # Only 2 calls total (sites + search) -- no approximate-count call.
    assert mock_request.call_count == 2


def test_search_issues_final_first_page_uses_exact_count_without_a_network_call(
    monkeypatch,
):
    # A first page with no nextPageToken is the complete result set --
    # len(issues) is already the exact count, so calling the advisory
    # approximate-count endpoint for it would be a pure wasted request.
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(
                json_data={
                    "issues": [
                        {"key": "ENG-1", "fields": {"summary": "a"}},
                        {"key": "ENG-2", "fields": {"summary": "b"}},
                    ]
                }
            ),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_search_issues("project = ENG"))

    assert result["status"] == "success"
    assert result["total_count"] == 2
    assert result["returned_count"] == 2
    # Only 2 calls total (sites + search) -- no approximate-count call.
    assert mock_request.call_count == 2


def test_search_issues_first_page_with_more_results_still_calls_approximate_count(
    monkeypatch,
):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(
                json_data={
                    "issues": [{"key": "ENG-1", "fields": {"summary": "a"}}],
                    "nextPageToken": "next-token",
                }
            ),
            MockResponse(json_data={"count": 42}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_search_issues("project = ENG"))

    assert result["total_count"] == 42
    assert mock_request.call_count == 3


def test_search_issues_raw_fields_returns_the_unslimmed_jira_shape(monkeypatch):
    # An existing integration written against the pre-projection schema
    # (issue["fields"]["status"]["id"], etc.) can opt back into it
    # instead of the compact projection.
    raw_issue = {
        "key": "ENG-1",
        "fields": {
            "summary": "ok",
            "status": {"id": "3", "name": "In Progress"},
            "assignee": {"accountId": "u1", "emailAddress": "a@example.com"},
            "project": {"id": "10", "name": "Engineering"},
        },
    }
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data={"issues": [raw_issue]}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_search_issues("project = ENG", raw_fields=True))

    assert result["issues"] == [raw_issue]
    assert result["issues"][0]["fields"]["status"]["id"] == "3"
    assert result["issues"][0]["fields"]["assignee"]["emailAddress"] == "a@example.com"


def test_search_issues_default_still_returns_the_compact_projection(monkeypatch):
    raw_issue = {
        "key": "ENG-1",
        "fields": {
            "summary": "ok",
            "status": {"id": "3", "name": "In Progress"},
        },
    }
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data={"issues": [raw_issue]}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_search_issues("project = ENG"))

    assert result["issues"] == [
        {
            "key": "ENG-1",
            "summary": "ok",
            "status": "In Progress",
            "status_category": None,
            "assignee": None,
            "priority": None,
            "issue_type": None,
            "project_key": None,
            "labels": [],
            "parent_key": None,
            "resolution": None,
            "created": None,
            "updated": None,
        }
    ]


def test_approximate_count_warns_on_non_integer_count(monkeypatch, caplog):
    mock_request = Mock(
        side_effect=[MockResponse(json_data={"count": "12"})],
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    with caplog.at_level(logging.WARNING, logger="jira-mcp"):
        result = jira._approximate_count("site-a", "project = ENG")

    assert result is None
    assert "non-integer" in caplog.text


def test_approximate_count_treats_bool_count_as_non_integer(monkeypatch, caplog):
    # bool is a subclass of int in Python -- a "count": true shape
    # anomaly must not be silently accepted as if it were a real count.
    mock_request = Mock(side_effect=[MockResponse(json_data={"count": True})])
    monkeypatch.setattr(jira.requests, "request", mock_request)

    with caplog.at_level(logging.WARNING, logger="jira-mcp"):
        result = jira._approximate_count("site-a", "project = ENG")

    assert result is None
    assert "non-integer" in caplog.text


def test_fetch_and_summarize_page_raises_on_non_list_issues(monkeypatch):
    mock_request = Mock(side_effect=[MockResponse(json_data={"issues": "not-a-list"})])
    monkeypatch.setattr(jira.requests, "request", mock_request)

    with pytest.raises(RuntimeError, match="Unexpected 'issues' shape"):
        jira._fetch_and_summarize_page("site-a", "project = ENG", 50, "")


def test_summarize_issue_tolerates_malformed_nested_fields():
    # A single issue with a malformed nested field (e.g. a misconfigured
    # custom field reshaping "assignee" to a string) must degrade
    # gracefully instead of raising AttributeError and failing the
    # entire page over one bad issue.
    raw_issue = {
        "key": "ENG-1",
        "fields": {
            "summary": "ok",
            "assignee": "not-a-dict",
            "status": ["not", "a", "dict"],
            "priority": 42,
        },
    }
    result = jira._summarize_issue(raw_issue)
    assert result["key"] == "ENG-1"
    assert result["assignee"] is None
    assert result["status"] is None
    assert result["priority"] is None


def test_search_issues_fallback_attempts_use_a_shorter_bounded_timeout(monkeypatch):
    # The first attempt (at the caller's requested size) should keep the
    # normal timeout/retry settings; only later fallback attempts get
    # the shorter, no-retry budget, so a rate-limited Jira endpoint can't
    # stack a full 429 retry-sleep across every one of up to 3 attempts.
    large_page = {
        "issues": [_raw_issue_with_summary(f"ENG-{i}", 200) for i in range(20)],
        "nextPageToken": "orig-next-token",
    }
    small_page = {"issues": [{"key": "ENG-1", "fields": {"summary": "ok"}}]}
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data=large_page),
            MockResponse(json_data=small_page),
            MockResponse(json_data={"count": 1}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)
    monkeypatch.setattr(jira, "get_tool_max_output_length", lambda: 1500)

    jira.jira_search_issues("project = ENG")

    first_search_call = mock_request.call_args_list[1]
    fallback_search_call = mock_request.call_args_list[2]
    assert first_search_call.kwargs["timeout"] == jira.DEFAULT_TIMEOUT_SECONDS
    assert fallback_search_call.kwargs["timeout"] == (
        jira._FALLBACK_ATTEMPT_TIMEOUT_SECONDS
    )


_SENTINEL_JQL = 'text ~ "super-secret-customer-name@example.com"'


def test_search_issues_success_log_omits_raw_jql(monkeypatch, caplog):
    # A final first page (no nextPageToken) has an exact count for free
    # (len(issues)) and never calls the approximate-count endpoint.
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data={"issues": [], "isLast": True}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    with caplog.at_level(logging.INFO, logger="jira-mcp"):
        jira.jira_search_issues(_SENTINEL_JQL)

    assert "super-secret-customer-name" not in caplog.text


def test_search_issues_count_failure_warning_omits_raw_jql(monkeypatch, caplog):
    # Approximate-count is only called for a first page that has MORE
    # results beyond it (a nextPageToken) -- a final first page's count
    # is exact for free and never calls the endpoint at all.
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(
                json_data={
                    "issues": [{"key": "ENG-1", "fields": {"summary": "ok"}}],
                    "nextPageToken": "next-token",
                }
            ),
            MockResponse(json_data={"errorMessages": [_SENTINEL_JQL]}, status_code=400),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    with caplog.at_level(logging.WARNING, logger="jira-mcp"):
        jira.jira_search_issues(_SENTINEL_JQL)

    assert "super-secret-customer-name" not in caplog.text


def test_search_issues_error_log_omits_raw_jql(monkeypatch, caplog):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data={"errorMessages": [_SENTINEL_JQL]}, status_code=400),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    with caplog.at_level(logging.ERROR, logger="jira-mcp"):
        jira.jira_search_issues(_SENTINEL_JQL)

    assert "super-secret-customer-name" not in caplog.text


def test_approximate_count_uses_a_short_no_retry_timeout(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data={"count": 3}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = jira._approximate_count("site-a", "project = ENG")

    assert result == 3
    count_call = mock_request.call_args_list[0]
    assert count_call.kwargs["timeout"] == jira._APPROXIMATE_COUNT_TIMEOUT_SECONDS


def test_approximate_count_does_not_retry_on_429(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(jira.time, "sleep", lambda s: sleep_calls.append(s))
    mock_request = Mock(
        return_value=MockResponse(status_code=429, headers={"Retry-After": "1"})
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = jira._approximate_count("site-a", "project = ENG")

    assert result is None
    assert sleep_calls == []
    assert mock_request.call_count == 1


def test_get_issue_returns_issue(monkeypatch):
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(
            side_effect=[
                MockResponse(json_data=[_SITE_A]),
                MockResponse(json_data={"key": "ENG-1", "fields": {"summary": "Bug"}}),
            ]
        ),
    )

    result = json.loads(jira.jira_get_issue("ENG-1"))

    assert result["status"] == "success"
    assert result["issue"]["key"] == "ENG-1"


def test_create_issue_sends_expected_fields(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data={"key": "ENG-2", "id": "10002"}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(
        jira.jira_create_issue(
            project_key="ENG",
            summary="New bug",
            description="Steps to reproduce",
            assignee_account_id="u1",
            priority="High",
        )
    )

    assert result["status"] == "success"
    assert result["issue"]["key"] == "ENG-2"
    create_call = mock_request.call_args_list[1]
    assert create_call.kwargs["json"] == {
        "fields": {
            "project": {"key": "ENG"},
            "summary": "New bug",
            "issuetype": {"name": "Task"},
            "description": "Steps to reproduce",
            "assignee": {"accountId": "u1"},
            "priority": {"name": "High"},
        }
    }


def test_update_issue_requires_at_least_one_field(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_update_issue("ENG-1"))

    assert result["status"] == "error"
    assert "No fields" in result["message"]
    mock_request.assert_not_called()


def test_update_issue_unassigns_on_explicit_empty_assignee_id(monkeypatch):
    mock_request = Mock(
        side_effect=[MockResponse(json_data=[_SITE_A]), MockResponse(json_data={})]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_update_issue("ENG-1", assignee_account_id=""))

    assert result["status"] == "success"
    update_call = mock_request.call_args_list[1]
    assert update_call.kwargs["json"] == {"fields": {"assignee": None}}


def test_create_issue_rejects_empty_summary(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_create_issue(project_key="ENG", summary=""))

    assert result["status"] == "error"
    assert "summary cannot be empty" in result["message"]
    mock_request.assert_not_called()


def test_update_issue_rejects_empty_summary(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_update_issue("ENG-1", summary=""))

    assert result["status"] == "error"
    assert "summary cannot be empty" in result["message"]
    mock_request.assert_not_called()


def test_update_issue_rejects_empty_priority(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_update_issue("ENG-1", priority=""))

    assert result["status"] == "error"
    assert "priority cannot be empty" in result["message"]
    mock_request.assert_not_called()


def test_list_transitions_returns_id_and_name(monkeypatch):
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(
            side_effect=[
                MockResponse(json_data=[_SITE_A]),
                MockResponse(
                    json_data={
                        "transitions": [
                            {"id": "11", "name": "In Progress", "extra": "dropped"},
                            {"id": "21", "name": "Done"},
                        ]
                    }
                ),
            ]
        ),
    )

    result = json.loads(jira.jira_list_transitions("ENG-1"))

    assert result["status"] == "success"
    assert result["transitions"] == [
        {"id": "11", "name": "In Progress"},
        {"id": "21", "name": "Done"},
    ]


def test_transition_issue_matches_name_case_insensitively(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(
                json_data={
                    "transitions": [
                        {"id": "11", "name": "In Progress"},
                        {"id": "21", "name": "Done"},
                    ]
                }
            ),
            MockResponse(json_data={}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_transition_issue("ENG-1", "done"))

    assert result["status"] == "success"
    assert result["transitioned_to"] == "Done"
    transition_call = mock_request.call_args_list[2]
    assert transition_call.kwargs["json"] == {"transition": {"id": "21"}}


def test_transition_issue_reports_available_transitions_when_not_found(monkeypatch):
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(
            side_effect=[
                MockResponse(json_data=[_SITE_A]),
                MockResponse(
                    json_data={"transitions": [{"id": "11", "name": "In Progress"}]}
                ),
            ]
        ),
    )

    result = json.loads(jira.jira_transition_issue("ENG-1", "Nonexistent"))

    assert result["status"] == "error"
    assert "In Progress" in result["message"]


def test_list_comments_reports_truncated(monkeypatch):
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(
            side_effect=[
                MockResponse(json_data=[_SITE_A]),
                MockResponse(
                    json_data={
                        "comments": [{"id": "1", "body": "First"}],
                        "total": 3,
                    }
                ),
            ]
        ),
    )

    result = json.loads(jira.jira_list_comments("ENG-1"))

    assert result["status"] == "success"
    assert result["truncated"] is True
    assert result["next_start_at"] == 1


def test_list_comments_paginates_with_start_at(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(
                json_data={
                    "comments": [{"id": "2", "body": "Second"}],
                    "total": 2,
                }
            ),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_list_comments("ENG-1", start_at=1))

    assert result["truncated"] is False
    assert result["next_start_at"] is None
    comment_call = mock_request.call_args_list[1]
    assert comment_call.kwargs["params"]["startAt"] == 1


def test_list_comments_empty_page_never_repeats_offset(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data={"comments": [], "total": 10}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_list_comments("ENG-1", start_at=5))

    assert result["truncated"] is False
    assert result["next_start_at"] is None


def test_list_comments_ignores_non_int_total(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(
                json_data={"comments": [{"id": "1", "body": "First"}], "total": None}
            ),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_list_comments("ENG-1"))

    assert result["status"] == "success"
    assert result["truncated"] is False
    assert result["next_start_at"] is None


def test_add_comment_sends_body(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data={"id": "1", "body": "Looks good"}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_add_comment("ENG-1", "Looks good"))

    assert result["status"] == "success"
    comment_call = mock_request.call_args_list[1]
    assert comment_call.kwargs["json"] == {"body": "Looks good"}


def test_search_users_maps_fields(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(
                json_data=[
                    {
                        "accountId": "u1",
                        "displayName": "Ada Lovelace",
                        "emailAddress": "ada@example.com",
                    }
                ]
            ),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_search_users("ada"))

    assert result["status"] == "success"
    assert result["users"] == [
        {"account_id": "u1", "display_name": "Ada Lovelace", "email": "ada@example.com"}
    ]
    assert result["truncated"] is False
    user_call = mock_request.call_args_list[1]
    assert user_call.kwargs["params"]["startAt"] == 0


def test_search_users_reports_truncated_on_full_page(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(
                json_data=[
                    {
                        "accountId": f"u{i}",
                        "displayName": f"User {i}",
                        "emailAddress": f"u{i}@example.com",
                    }
                    for i in range(2)
                ]
            ),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_search_users("a", limit=2, start_at=2))

    assert result["truncated"] is True
    assert result["next_start_at"] == 4
    user_call = mock_request.call_args_list[1]
    assert user_call.kwargs["params"]["startAt"] == 2


def test_search_users_returns_error_on_non_list_response(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data={"unexpected": "shape"}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_search_users("ada"))

    assert result["status"] == "error"
    assert "Unexpected response format" in result["message"]


def test_search_users_counts_raw_page_for_pagination(monkeypatch):
    # A malformed (non-dict) element is dropped from users but must still
    # count toward the page size, or pagination would stall or re-read rows.
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(
                json_data=[
                    {"accountId": "u1", "displayName": "Ada", "emailAddress": "a@x.io"},
                    "malformed-entry",
                ]
            ),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_search_users("a", limit=2))

    assert result["status"] == "success"
    assert len(result["users"]) == 1
    assert result["truncated"] is True
    assert result["next_start_at"] == 2


def test_jira_app_registry_includes_offline_access_scope():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    jira_app = next(
        row for row in get_builtin_public_mcp_app_rows() if row["app_id"] == "jira"
    )
    assert "offline_access" in jira_app["oauth_scopes"]
