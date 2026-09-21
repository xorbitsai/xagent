import json
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import jira
from xagent.web.tools.mcp import utils as jira_mcp_utils


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
    assert result["projects"] == [
        {"id": "1", "key": "ENG", "name": "Engineering", "project_type_key": None}
    ]
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


def test_list_projects_clamps_limit_and_offset(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data={"values": [], "isLast": True}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    json.loads(jira.jira_list_projects(limit=9999, start_at=-5))

    project_call = mock_request.call_args_list[1]
    assert project_call.kwargs["params"]["maxResults"] == jira.MAX_LIMIT
    assert project_call.kwargs["params"]["startAt"] == 0


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


def test_list_projects_drops_avatar_urls(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(
                json_data={
                    "values": [
                        {
                            "id": "10033",
                            "key": "DW",
                            "name": "Datapel WMS",
                            "projectTypeKey": "software",
                            "avatarUrls": {
                                "48x48": "https://api.atlassian.com/.../avatar"
                            },
                        }
                    ],
                    "isLast": True,
                }
            ),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_list_projects())

    assert result["projects"] == [
        {
            "id": "10033",
            "key": "DW",
            "name": "Datapel WMS",
            "project_type_key": "software",
        }
    ]
    assert "avatarUrls" not in json.dumps(result)


def test_list_projects_raw_fields_returns_the_unsummarized_jira_shape(monkeypatch):
    raw_project = {
        "id": "10033",
        "key": "DW",
        "name": "Datapel WMS",
        "projectTypeKey": "software",
        "avatarUrls": {"48x48": "https://api.atlassian.com/.../avatar"},
        "lead": {"accountId": "abc123", "displayName": "Alice"},
    }
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data={"values": [raw_project], "isLast": True}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_list_projects(raw_fields=True))

    assert result["projects"] == [raw_project]
    assert result["truncated"] is False
    assert result["next_start_at"] is None


def test_list_projects_raw_fields_shrinks_page_to_fit_and_advances_correctly(
    monkeypatch,
):
    # Unlike the summarized path (small enough per-project that this has
    # never been a practical concern), raw project objects can be large
    # enough that even a page smaller than `limit` needs to shrink to
    # fit -- next_start_at must then point at the first DROPPED project
    # (offset + kept count), not wherever Jira's own isLast/page-size
    # signal said, so a caller resuming from it neither skips nor
    # repeats a project.
    big_projects = [
        {"id": str(i), "key": f"P{i}", "name": "x" * 300, "extra": "y" * 300}
        for i in range(20)
    ]
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data={"values": big_projects, "isLast": True}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)
    monkeypatch.setattr(jira, "get_tool_max_output_length", lambda: 3000)

    raw_response = jira.jira_list_projects(raw_fields=True)
    result = json.loads(raw_response)

    assert len(raw_response) <= 3000
    assert result["status"] == "success"
    assert 0 < len(result["projects"]) < 20
    assert result["truncated"] is True
    assert result["next_start_at"] == len(result["projects"])


def test_list_projects_next_start_at_counts_raw_page_not_filtered(monkeypatch):
    # One malformed (non-dict) entry alongside two real projects. If
    # next_start_at were computed from the filtered `projects` list (2)
    # instead of the raw page (3), the next call would re-fetch an
    # already-seen project instead of skipping past it.
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(
                json_data={
                    "values": [
                        {"id": "1", "key": "ENG", "name": "Engineering"},
                        "not-a-dict",
                        {"id": "2", "key": "DW", "name": "Datapel WMS"},
                    ],
                    "isLast": False,
                }
            ),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_list_projects())

    assert len(result["projects"]) == 2
    assert result["truncated"] is True
    assert result["next_start_at"] == 3


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
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_search_issues("project = ENG"))

    assert result["status"] == "success"
    assert result["issues"][0]["key"] == "ENG-1"
    assert result["truncated"] is True
    assert result["next_page_token"] == "token-2"
    search_call = mock_request.call_args_list[1]
    assert search_call.kwargs["url"] == (
        "https://api.atlassian.com/ex/jira/site-a/rest/api/3/search/jql"
    )
    assert search_call.kwargs["params"]["jql"] == "project = ENG"


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


def test_search_issues_clamps_limit(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data={"issues": []}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    json.loads(jira.jira_search_issues("project = ENG", limit=0))

    search_call = mock_request.call_args_list[1]
    assert search_call.kwargs["params"]["maxResults"] == 1


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


def test_get_issue_raw_fields_returns_the_unflattened_jira_shape(monkeypatch):
    # raw_fields restores access to the pre-summarization "issue.fields"
    # nesting (e.g. status.id, not just the summary's flattened
    # status_category name) for an existing integration written against
    # the old raw shape, or a nested sub-field the summary never exposed.
    raw_issue = {
        "id": "1",
        "key": "ENG-1",
        "fields": {
            "summary": "Bug",
            "status": {"id": "10004", "name": "In Progress"},
        },
    }
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(
            side_effect=[
                MockResponse(json_data=[_SITE_A]),
                MockResponse(json_data=raw_issue),
            ]
        ),
    )

    result = json.loads(jira.jira_get_issue("ENG-1", raw_fields=True))

    assert result["status"] == "success"
    assert result["issue"]["fields"]["status"]["id"] == "10004"
    assert result["issue"]["fields"]["summary"] == "Bug"
    # The summarized-mode-only extras don't apply in raw mode.
    assert "extra_field_values" not in result
    assert "description_truncated" not in result


def test_get_issue_raw_fields_still_fetches_requested_extra_fields(monkeypatch):
    # extra_fields still controls which fields Jira is asked for in raw
    # mode -- it just skips the summarized path's separate
    # extra_field_values envelope, since the raw "fields" object already
    # contains everything requested.
    raw_issue = {
        "id": "1",
        "key": "ENG-1",
        "fields": {"summary": "Bug", "customfield_10099": "custom value"},
    }
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data=raw_issue),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(
        jira.jira_get_issue("ENG-1", extra_fields="customfield_10099", raw_fields=True)
    )

    assert result["issue"]["fields"]["customfield_10099"] == "custom value"
    issue_call = mock_request.call_args_list[1]
    assert "customfield_10099" in issue_call.kwargs["params"]["fields"]


def test_get_issue_raw_fields_bounds_an_oversized_response(monkeypatch):
    # Same output-cap guarantee as the summarized path, just via
    # success_with_capped_dict directly on the raw payload instead of
    # the summary's own bounding machinery.
    raw_issue = {
        "id": "1",
        "key": "ENG-1",
        "fields": {
            "summary": "Bug",
            "issuelinks": [{"outwardIssue": {"key": f"ENG-{i}"}} for i in range(500)],
        },
    }
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(
            side_effect=[
                MockResponse(json_data=[_SITE_A]),
                MockResponse(json_data=raw_issue),
            ]
        ),
    )
    monkeypatch.setattr(jira, "get_tool_max_output_length", lambda: 2000)
    monkeypatch.setattr(jira_mcp_utils, "get_tool_max_output_length", lambda: 2000)

    raw_response = jira.jira_get_issue("ENG-1", raw_fields=True)
    result = json.loads(raw_response)

    assert len(raw_response) <= 2000
    assert result["status"] == "success"
    assert result["truncated"] is True


def test_get_issue_flattens_description_and_surfaces_dependencies(monkeypatch):
    # This fixture's ADF-shaped description is a defensive-path test,
    # not a claim about what this tool's actual v2 request returns in
    # production: _issue_path hardcodes /rest/api/2/..., and per
    # Atlassian's own migration docs, only v3 returns ADF objects by
    # default -- v2 returns wiki-markup strings (the plain-string branch
    # _flatten_adf already handles). Kept as a real self URL, not a v3
    # one, so this fixture doesn't contradict which endpoint is called.
    raw_issue = {
        "expand": "renderedFields,names,schema,operations,editmeta,changelog",
        "id": "34280",
        "self": "https://api.atlassian.com/ex/jira/site-a/rest/api/2/issue/34280",
        "key": "DW-782",
        "fields": {
            "summary": "[Connectify][6thman] Persist customer configuration",
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {"type": "text", "text": "ECS task replacement wipes"},
                            {"type": "hardBreak"},
                            {"type": "text", "text": "the persisted config."},
                        ],
                    }
                ],
            },
            "status": {"name": "待办", "statusCategory": {"name": "To Do"}},
            "assignee": {
                "accountId": "712020:98fdf2ea",
                "displayName": "jiarongling",
                "avatarUrls": {"48x48": "https://secure.gravatar.com/avatar/..."},
            },
            "reporter": {"accountId": "u2", "displayName": "Pete Rocke"},
            "creator": {"accountId": "u2", "displayName": "Pete Rocke"},
            "priority": {"name": "P2 - High"},
            "issuetype": {"name": "任务"},
            "project": {"key": "DW", "name": "Datapel WMS"},
            "parent": {"key": "DW-746"},
            "resolution": None,
            "labels": ["connectify"],
            "components": [{"name": "Connectify"}],
            "fixVersions": [{"name": "2026.09"}],
            "duedate": "2026-11-27",
            "created": "2026-09-09T10:00:00.000+0000",
            "updated": "2026-09-11T19:57:00.000+0000",
            "issuelinks": [
                {
                    "type": {"inward": "is blocked by", "outward": "blocks"},
                    "outwardIssue": {
                        "key": "DW-747",
                        "fields": {
                            "summary": "Confirm config API auth",
                            "status": {"name": "In Progress"},
                        },
                    },
                }
            ],
            "subtasks": [
                {
                    "key": "DW-786",
                    "fields": {
                        "summary": "Secure /api/dev",
                        "status": {"name": "In Progress"},
                    },
                }
            ],
        },
    }
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(
            side_effect=[
                MockResponse(json_data=[_SITE_A]),
                MockResponse(json_data=raw_issue),
            ]
        ),
    )

    result = json.loads(jira.jira_get_issue("DW-782"))

    issue = result["issue"]
    assert issue["id"] == "34280"
    assert issue["description"] == "ECS task replacement wipes\nthe persisted config."
    assert issue["assignee"] == {
        "account_id": "712020:98fdf2ea",
        "display_name": "jiarongling",
    }
    assert issue["components"] == ["Connectify"]
    assert issue["fix_versions"] == ["2026.09"]
    assert issue["due_date"] == "2026-11-27"
    assert issue["issue_links"] == [
        {
            "relationship": "blocks",
            "issue_key": "DW-747",
            "summary": "Confirm config API auth",
            "status": "In Progress",
        }
    ]
    assert issue["subtasks"] == [
        {
            "issue_key": "DW-786",
            "summary": "Secure /api/dev",
            "status": "In Progress",
        }
    ]
    assert "avatarUrls" not in json.dumps(result)
    assert "self" not in issue
    assert "expand" not in issue


def test_get_issue_truncates_an_oversized_description(monkeypatch):
    long_description = {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {
                        "type": "text",
                        "text": "x" * (jira._ISSUE_DESCRIPTION_MAX_CHARS + 500),
                    }
                ],
            }
        ],
    }
    raw_issue = {
        "key": "ENG-1",
        "fields": {"summary": "big", "description": long_description},
    }
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(
            side_effect=[
                MockResponse(json_data=[_SITE_A]),
                MockResponse(json_data=raw_issue),
            ]
        ),
    )

    result = json.loads(jira.jira_get_issue("ENG-1"))

    assert result["description_truncated"] is True
    assert len(
        result["issue"]["description"]
    ) <= jira._ISSUE_DESCRIPTION_MAX_CHARS + len("... [truncated]")
    # The response as a whole must still be valid, parseable JSON --
    # json.loads above already proves that, but assert it isn't merely
    # accidentally small enough to dodge the real framework cap too.
    assert len(json.dumps(result)) < jira._ISSUE_DESCRIPTION_MAX_CHARS + 1000


def test_get_issue_falls_back_to_capped_dict_when_still_too_big(monkeypatch):
    # Even with description capped, an issue with many dependencies can
    # still overflow a sufficiently small configured budget -- this must
    # degrade to success_with_capped_dict's shrink-until-bounded output
    # rather than return invalid (cut-mid-JSON) text.
    raw_issue = {
        "key": "ENG-1",
        "fields": {
            "summary": "many deps",
            "issuelinks": [
                {
                    "type": {"outward": "blocks"},
                    "outwardIssue": {
                        "key": f"ENG-{i}",
                        "fields": {"summary": "x" * 200, "status": {"name": "Open"}},
                    },
                }
                for i in range(50)
            ],
        },
    }
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(
            side_effect=[
                MockResponse(json_data=[_SITE_A]),
                MockResponse(json_data=raw_issue),
            ]
        ),
    )
    # jira.py's own budget check (which decides whether to fall back to
    # success_with_capped_dict at all) and success_with_capped_dict's
    # internal budget check are two independent `from ... import
    # get_tool_max_output_length` bindings (one in jira.py, one in
    # mcp/utils.py) -- both read the same real env var in production, so
    # both must be patched together here to reproduce that.
    monkeypatch.setattr(jira, "get_tool_max_output_length", lambda: 2000)
    monkeypatch.setattr(jira_mcp_utils, "get_tool_max_output_length", lambda: 2000)

    raw_response = jira.jira_get_issue("ENG-1")
    result = json.loads(raw_response)

    assert len(raw_response) <= 2000
    assert result["status"] == "success"
    assert result.get("truncated") is True


def test_get_issue_extra_fields_returns_raw_custom_field_values(monkeypatch):
    raw_issue = {
        "key": "ENG-1",
        "fields": {
            "summary": "ok",
            "customfield_10010": "Sprint 42",
            "customfield_10020": 8,
        },
    }
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data=raw_issue),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(
        jira.jira_get_issue("ENG-1", extra_fields="customfield_10010,customfield_10020")
    )

    assert result["extra_field_values"] == {
        "customfield_10010": "Sprint 42",
        "customfield_10020": 8,
    }
    assert "extra_field_values" not in result["issue"]
    issue_call = mock_request.call_args_list[1]
    assert issue_call.kwargs["params"]["fields"] == (
        f"{jira._GET_ISSUE_FIELDS},customfield_10010,customfield_10020"
    )


def test_get_issue_extra_fields_ignores_a_name_already_in_the_default_set(monkeypatch):
    # duedate is already fetched/summarized by default -- naming it in
    # extra_fields must not re-fetch, re-cap, or duplicate it under
    # extra_field_values.
    raw_issue = {
        "key": "ENG-1",
        "fields": {"summary": "ok", "duedate": "2026-01-01"},
    }
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data=raw_issue),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_get_issue("ENG-1", extra_fields="duedate"))

    assert result["issue"]["due_date"] == "2026-01-01"
    assert "extra_field_values" not in result
    issue_call = mock_request.call_args_list[1]
    assert issue_call.kwargs["params"]["fields"] == jira._GET_ISSUE_FIELDS


def test_get_issue_bounds_the_aggregate_extra_field_values_size(monkeypatch):
    # Two custom fields each individually under _ISSUE_DESCRIPTION_MAX_
    # CHARS can still combine to exceed the output budget -- an entirely
    # ordinary multi-field request, not a pathological one. Without an
    # aggregate cap, extra_field_values (passed as a protected top-level
    # extra, immune to success_with_capped_dict's gradual shrinking)
    # would push the whole response past budget and get dropped
    # entirely by the extreme fallback -- reporting success with no
    # sign the requested fields were ever there.
    #
    # Pinned to a specific max_output_length (both bindings -- jira.py's
    # own and the one success_with_capped_dict reads from utils.py --
    # since they're separate imports) rather than relying on whatever
    # get_tool_max_output_length() resolves to from the ambient
    # environment: this test's assertion bound is calibrated against
    # remaining_budget's exact value (max_output_length // 2), so an
    # unpinned/differently-configured cap could silently make it
    # exercise a different code path (or fail outright) without
    # actually catching a regression. 60_000 keeps remaining_budget at
    # exactly _ISSUE_DESCRIPTION_MAX_CHARS (30_000), matching this
    # test's existing assertion bound.
    monkeypatch.setattr(jira, "get_tool_max_output_length", lambda: 60_000)
    monkeypatch.setattr(jira_mcp_utils, "get_tool_max_output_length", lambda: 60_000)
    near_cap_value = "x" * (jira._ISSUE_DESCRIPTION_MAX_CHARS - 100)
    raw_issue = {
        "key": "ENG-1",
        "fields": {
            "summary": "ok",
            "customfield_a": near_cap_value,
            "customfield_b": near_cap_value,
        },
    }
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(
            side_effect=[
                MockResponse(json_data=[_SITE_A]),
                MockResponse(json_data=raw_issue),
            ]
        ),
    )

    raw_response = jira.jira_get_issue(
        "ENG-1", extra_fields="customfield_a,customfield_b"
    )
    result = json.loads(raw_response)

    assert result["status"] == "success"
    assert "extra_field_values" in result
    assert result["extra_field_values_truncated"] is True
    total_extra_size = len(json.dumps(result["extra_field_values"]))
    assert total_extra_size <= jira._ISSUE_DESCRIPTION_MAX_CHARS + 1000


def test_get_issue_extra_field_values_stays_within_a_small_configured_cap(
    monkeypatch,
):
    # A configured max_output_length well under 60_000 (the point below
    # which an earlier version's aggregate-budget floor exceeded the
    # ENTIRE cap) must not make extra_field_values -- or
    # description_truncated alongside it -- vanish. Since
    # extra_field_values is a protected top-level extra that
    # success_with_capped_dict can't gradually shrink, an oversized
    # aggregate budget here forces that helper's last-resort
    # with_extras=False fallback, which drops every extra at once and
    # reports a plain "success" with no sign any of them were ever
    # requested -- the regression this test guards against.
    monkeypatch.setattr(jira, "get_tool_max_output_length", lambda: 2000)
    monkeypatch.setattr(jira_mcp_utils, "get_tool_max_output_length", lambda: 2000)
    raw_issue = {
        "id": "1",
        "key": "ENG-1",
        "fields": {"summary": "ok", "customfield_big": "x" * 25000},
    }
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(
            side_effect=[
                MockResponse(json_data=[_SITE_A]),
                MockResponse(json_data=raw_issue),
            ]
        ),
    )

    result = json.loads(jira.jira_get_issue("ENG-1", extra_fields="customfield_big"))

    assert result["status"] == "success"
    assert len(json.dumps(result)) <= 2000
    assert "extra_field_values" in result
    assert "description_truncated" in result
    assert result["extra_field_values_truncated"] is True


def test_get_issue_rejects_jira_field_selector_syntax_in_extra_fields(monkeypatch):
    # Jira's `fields` query param treats a leading "-" as "exclude this
    # field" and "*" as a wildcard, not a literal field id -- passed
    # through unfiltered, "-description" could suppress the very field
    # this tool's docstring promises is always returned.
    raw_issue = {
        "key": "ENG-1",
        "fields": {"summary": "ok", "description": "the real description"},
    }
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data=raw_issue),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_get_issue("ENG-1", extra_fields="-description,*all"))

    assert result["issue"]["description"] == "the real description"
    assert "extra_field_values" not in result
    issue_call = mock_request.call_args_list[1]
    assert issue_call.kwargs["params"]["fields"] == jira._GET_ISSUE_FIELDS


def test_cap_text_field_marks_truncation():
    payload = {"body": "x" * 20}
    assert jira._cap_text_field(payload, "body", 10) is True
    assert payload["body"].startswith("xxxxxxxxxx")
    assert payload["body"].endswith("[truncated]")

    short_payload = {"body": "short"}
    assert jira._cap_text_field(short_payload, "body", 10) is False
    assert short_payload["body"] == "short"


def test_flatten_adf_returns_plain_strings_unchanged():
    # Jira Server/Data Center can still return wiki-markup strings for a
    # description/comment body; that must pass through untouched instead
    # of being mistaken for "not ADF, so drop it".
    assert jira._flatten_adf("plain text") == "plain text"
    assert jira._flatten_adf("") is None
    assert jira._flatten_adf(None) is None
    assert jira._flatten_adf(123) is None


def test_flatten_adf_renders_mentions_emojis_and_inline_cards():
    # mention/emoji/inlineCard are leaf inline nodes with no "text" and no
    # "content" -- without explicit handling they silently render as ""
    # and the @-mention/emoji/link vanishes from the flattened text.
    adf = {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": "Hello "},
                    {
                        "type": "mention",
                        "attrs": {"id": "u1", "text": "@Bruce"},
                    },
                    {"type": "text", "text": " "},
                    {
                        "type": "emoji",
                        "attrs": {"shortName": ":smile:", "text": "😃"},
                    },
                    {"type": "text", "text": " see "},
                    {
                        "type": "inlineCard",
                        "attrs": {"url": "https://example.com/DW-782"},
                    },
                ],
            }
        ],
    }
    assert jira._flatten_adf(adf) == "Hello @Bruce 😃 see https://example.com/DW-782"


def test_flatten_adf_emoji_falls_back_to_short_name_without_text():
    adf = {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [{"type": "emoji", "attrs": {"shortName": ":+1:"}}],
            }
        ],
    }
    assert jira._flatten_adf(adf) == ":+1:"


def test_flatten_adf_list_items_have_no_blank_line_between_them():
    adf = {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "bulletList",
                "content": [
                    {
                        "type": "listItem",
                        "content": [
                            {
                                "type": "paragraph",
                                "content": [{"type": "text", "text": "first"}],
                            }
                        ],
                    },
                    {
                        "type": "listItem",
                        "content": [
                            {
                                "type": "paragraph",
                                "content": [{"type": "text", "text": "second"}],
                            }
                        ],
                    },
                ],
            }
        ],
    }
    assert jira._flatten_adf(adf) == "first\nsecond"


def test_flatten_adf_blockquote_has_no_trailing_blank_line():
    adf = {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "blockquote",
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": "quoted"}],
                    }
                ],
            },
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": "after"}],
            },
        ],
    }
    assert jira._flatten_adf(adf) == "quoted\nafter"


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
    assert "summary" in result["message"]
    mock_request.assert_not_called()


def test_create_issue_rejects_whitespace_only_summary(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_create_issue(project_key="ENG", summary="   "))

    assert result["status"] == "error"
    assert "summary" in result["message"]
    mock_request.assert_not_called()


def test_update_issue_rejects_empty_summary(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_update_issue("ENG-1", summary=""))

    assert result["status"] == "error"
    assert "summary" in result["message"]
    mock_request.assert_not_called()


def test_update_issue_rejects_padded_summary(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_update_issue("ENG-1", summary="  padded  "))

    assert result["status"] == "error"
    assert "summary" in result["message"]
    mock_request.assert_not_called()


def test_update_issue_rejects_empty_priority(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_update_issue("ENG-1", priority=""))

    assert result["status"] == "error"
    assert "priority" in result["message"]
    assert "omit the parameter instead" in result["message"]
    mock_request.assert_not_called()


def test_update_issue_rejects_padded_priority(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_update_issue("ENG-1", priority=" High "))

    assert result["status"] == "error"
    assert "priority" in result["message"]
    assert "omit the parameter instead" in result["message"]
    mock_request.assert_not_called()


def test_summary_and_priority_rejection_does_not_log_an_error(monkeypatch, caplog):
    # A caller-input validation rejection is a routine, expected outcome
    # (not a connector/API failure) -- it must not surface as an ERROR log
    # for either field, consistently.
    mock_request = Mock()
    monkeypatch.setattr(jira.requests, "request", mock_request)

    with caplog.at_level("ERROR", logger="jira-mcp"):
        create_result = json.loads(
            jira.jira_create_issue(project_key="ENG", summary="")
        )
        update_summary_result = json.loads(jira.jira_update_issue("ENG-1", summary=""))
        update_priority_result = json.loads(
            jira.jira_update_issue("ENG-1", priority="")
        )

    assert create_result["status"] == "error"
    assert update_summary_result["status"] == "error"
    assert update_priority_result["status"] == "error"
    assert caplog.records == []


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


def test_list_comments_clamps_limit_and_offset(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data={"comments": [], "total": 0}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    json.loads(jira.jira_list_comments("ENG-1", limit=9999, start_at=-5))

    comment_call = mock_request.call_args_list[1]
    assert comment_call.kwargs["params"]["maxResults"] == jira.MAX_LIMIT
    assert comment_call.kwargs["params"]["startAt"] == 0


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


def test_list_comments_flattens_adf_body_and_drops_avatar_urls(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(
                json_data={
                    "comments": [
                        {
                            "id": "10001",
                            "author": {
                                "accountId": "u1",
                                "displayName": "Bruce",
                                "avatarUrls": {
                                    "48x48": "https://secure.gravatar.com/..."
                                },
                            },
                            "body": {
                                "type": "doc",
                                "version": 1,
                                "content": [
                                    {
                                        "type": "paragraph",
                                        "content": [
                                            {"type": "text", "text": "Looks good"}
                                        ],
                                    }
                                ],
                            },
                            "created": "2026-09-11T10:00:00.000+0000",
                            "updated": "2026-09-11T10:00:00.000+0000",
                        }
                    ],
                    "total": 1,
                }
            ),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_list_comments("ENG-1"))

    assert result["total"] == 1
    assert result["returned_count"] == 1
    assert result["comments"] == [
        {
            "id": "10001",
            "author": {"account_id": "u1", "display_name": "Bruce"},
            "body": "Looks good",
            "visibility": None,
            "jsd_public": None,
            "body_truncated": False,
            "created": "2026-09-11T10:00:00.000+0000",
            "updated": "2026-09-11T10:00:00.000+0000",
        }
    ]
    assert "avatarUrls" not in json.dumps(result)


def test_list_comments_raw_fields_returns_the_unflattened_jira_shape(monkeypatch):
    raw_comment = {
        "id": "10001",
        "author": {"accountId": "u1", "displayName": "Bruce"},
        "body": {
            "type": "doc",
            "version": 1,
            "content": [
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": "Looks good"}],
                }
            ],
        },
        "renderedBody": "<p>Looks good</p>",
    }
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data={"comments": [raw_comment], "total": 1}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_list_comments("ENG-1", raw_fields=True))

    assert result["comments"] == [raw_comment]
    assert result["next_start_at"] is None


def test_list_comments_raw_fields_shrinks_page_to_fit_and_advances_correctly(
    monkeypatch,
):
    raw_comments = [{"id": str(i), "body": "x" * 300} for i in range(20)]
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data={"comments": raw_comments, "total": 20}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)
    monkeypatch.setattr(jira, "get_tool_max_output_length", lambda: 3000)

    raw_response = jira.jira_list_comments("ENG-1", raw_fields=True)
    result = json.loads(raw_response)

    assert len(raw_response) <= 3000
    assert result["status"] == "success"
    assert 0 < len(result["comments"]) < 20
    assert result["truncated"] is True
    assert result["next_start_at"] == len(result["comments"])


def test_list_comments_raw_fields_errors_on_an_unfittable_single_comment(monkeypatch):
    # A raw comment's body is an ADF object, not a plain string
    # _cap_text_field can shrink -- unlike the summarized path, a
    # single unfittable raw comment can't be shrunk further, so this
    # must be the bounded-error case, not a silently-skipped page.
    raw_comment = {"id": "1", "body": {"type": "doc", "content": "x" * 5000}}
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data={"comments": [raw_comment], "total": 1}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)
    monkeypatch.setattr(jira, "get_tool_max_output_length", lambda: 300)

    result = json.loads(jira.jira_list_comments("ENG-1", raw_fields=True))

    assert result["status"] == "error"


def _raw_comment_with_body(comment_id: str, body_length: int):
    return {"id": comment_id, "body": "x" * body_length}


def test_list_comments_truncates_an_oversized_single_body(monkeypatch):
    long_body = "x" * (jira._COMMENT_BODY_MAX_CHARS + 500)
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(
                json_data={
                    "comments": [{"id": "1", "body": long_body}],
                    "total": 1,
                }
            ),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_list_comments("ENG-1"))

    assert result["comments"][0]["body_truncated"] is True
    assert len(result["comments"][0]["body"]) <= jira._COMMENT_BODY_MAX_CHARS + len(
        "... [truncated]"
    )


def test_list_comments_returns_fewer_than_requested_when_page_overflows(monkeypatch):
    # 20 body-capped comments together still overflow a small budget --
    # the page must shrink to a whole prefix that fits, with
    # next_start_at pointing at the RAW position of the first comment
    # left out (not the count of comments actually returned).
    raw_comments = [_raw_comment_with_body(str(i), 200) for i in range(20)]
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data={"comments": raw_comments, "total": 20}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)
    monkeypatch.setattr(jira, "get_tool_max_output_length", lambda: 1500)

    result = json.loads(jira.jira_list_comments("ENG-1"))

    assert result["status"] == "success"
    assert 0 < len(result["comments"]) < 20
    assert result["truncated"] is True
    assert result["next_start_at"] == len(result["comments"])


def test_list_comments_returns_bounded_error_when_single_comment_too_big(monkeypatch):
    huge_body = "x" * (jira._COMMENT_BODY_MAX_CHARS + 500)
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(
                json_data={"comments": [{"id": "1", "body": huge_body}], "total": 1}
            ),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)
    # Small enough that even one body-capped comment doesn't fit.
    monkeypatch.setattr(jira, "get_tool_max_output_length", lambda: 5)

    result = json.loads(jira.jira_list_comments("ENG-1"))

    assert result["status"] == "error"
    assert "output limit" in result["message"]


def test_list_comments_shrinks_a_single_oversized_comment_instead_of_dropping_it(
    monkeypatch,
):
    # When a single comment doesn't fit at its default per-comment body
    # cap, it must be shrunk further (a smaller body cap, same
    # body_truncated signal) rather than dropped from the page --
    # next_start_at advances past it normally (it WAS delivered), not
    # skipped as if it never existed.
    raw_comment = {"id": "1", "body": "x" * 500}
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data={"comments": [raw_comment], "total": 1}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)
    monkeypatch.setattr(jira, "get_tool_max_output_length", lambda: 400)

    result = json.loads(jira.jira_list_comments("ENG-1"))

    assert result["status"] == "success"
    assert len(result["comments"]) == 1
    comment = result["comments"][0]
    assert comment["id"] == "1"
    assert comment["body_truncated"] is True
    assert 0 < len(comment["body"]) < 500
    assert len(json.dumps(result)) <= 400
    assert result["next_start_at"] is None  # this WAS the whole (one-comment) page


def test_list_comments_returns_error_when_not_even_an_empty_bodied_comment_fits(
    monkeypatch,
):
    # A budget so small that not even a single comment with its body
    # shrunk to nothing fits at all -- no comment from this raw page can
    # be delivered, so this must be the hard error case, not a silent
    # "success" with the comment skipped and unrecoverable via
    # next_start_at.
    raw_comment = {"id": "1", "body": "x" * 50}
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data={"comments": [raw_comment], "total": 1}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)
    empty_page_size = len(jira._build_comments_response([], 1, 1))
    monkeypatch.setattr(jira, "get_tool_max_output_length", lambda: empty_page_size + 5)

    result = json.loads(jira.jira_list_comments("ENG-1"))

    assert result["status"] == "error"


def test_fit_comments_page_shrinks_a_single_unfittable_comment():
    # A single comment that doesn't fit at the default cap, preceded by
    # a malformed (non-dict) raw entry -- next_start_at_for(1)'s "every
    # raw comment accounted for" branch must still be reached (the
    # malformed entry is filtered, not counted against kept), and the
    # comment's body must be shrunk (not the comment dropped).
    raw_comments = ["not-a-dict", {"id": "1", "body": "x" * 500}]
    offset = 10

    response = jira._fit_comments_page(raw_comments, offset, 12, True, 400)

    assert response is not None
    result = json.loads(response)
    assert len(result["comments"]) == 1
    assert result["comments"][0]["body_truncated"] is True
    assert 0 < len(result["comments"][0]["body"]) < 500
    # has_more_raw=True and every kept (filtered) comment is included --
    # the raw-page-relative "more beyond this page" signal, not a
    # skip-past-the-unfittable-entry one.
    assert result["next_start_at"] == offset + len(raw_comments)


def test_fit_comments_page_errors_when_not_even_an_empty_bodied_comment_fits():
    # Mirrors the size-too-small-for-anything case above but calls
    # _fit_comments_page directly: returns None (the caller's cue to
    # return a bounded error) rather than skipping the comment via
    # next_start_at, since retrying could never succeed here either.
    raw_comments = ["not-a-dict", {"id": "1", "body": "x" * 50}]
    offset = 10
    empty_page_size = len(jira._build_comments_response([], 5, offset + 2))

    response = jira._fit_comments_page(
        raw_comments, offset, 5, False, empty_page_size + 5
    )

    assert response is None


def test_list_comments_next_start_at_counts_raw_page_not_filtered(monkeypatch):
    # Same reasoning as the equivalent jira_list_projects test: one
    # malformed (non-dict) entry alongside two real comments must not
    # shrink next_start_at below the raw page size Jira's startAt is
    # positional over.
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(
                json_data={
                    "comments": [
                        {"id": "1", "body": "First"},
                        "not-a-dict",
                        {"id": "2", "body": "Second"},
                    ],
                    "total": 10,
                }
            ),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_list_comments("ENG-1"))

    assert result["returned_count"] == 2
    assert result["truncated"] is True
    assert result["next_start_at"] == 3


def test_list_comments_surfaces_restricted_visibility(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(
                json_data={
                    "comments": [
                        {
                            "id": "1",
                            "body": "internal only",
                            "visibility": {"type": "role", "value": "Administrators"},
                        }
                    ],
                    "total": 1,
                }
            ),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_list_comments("ENG-1"))

    assert result["comments"][0]["visibility"] == {
        "type": "role",
        "value": "Administrators",
    }


def test_list_comments_surfaces_jsd_internal_flag_with_no_visibility_set(monkeypatch):
    # Jira Service Management marks a comment internal-only via
    # jsdPublic, a mechanism separate from (and not mirrored into)
    # visibility -- a JSM agent can mark a comment internal with no
    # visibility restriction set at all, so visibility alone would
    # silently report it as public.
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(
                json_data={
                    "comments": [
                        {"id": "1", "body": "agent-only note", "jsdPublic": False}
                    ],
                    "total": 1,
                }
            ),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_list_comments("ENG-1"))

    assert result["comments"][0]["visibility"] is None
    assert result["comments"][0]["jsd_public"] is False


def test_get_issue_requests_exact_field_list(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data={"key": "ENG-1", "fields": {}}),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    jira.jira_get_issue("ENG-1")

    issue_call = mock_request.call_args_list[1]
    assert issue_call.kwargs["params"]["fields"] == jira._GET_ISSUE_FIELDS


def test_get_issue_fields_constant_value():
    # test_get_issue_requests_exact_field_list only proves the request
    # uses whatever _GET_ISSUE_FIELDS currently is -- if the constant
    # itself were accidentally emptied or weakened, both sides of that
    # comparison would move together and stay green. Pin the literal
    # text here so a change to the constant is caught on its own.
    assert jira._GET_ISSUE_FIELDS == (
        "summary,description,status,assignee,reporter,creator,"
        "priority,issuetype,project,parent,resolution,labels,"
        "components,fixVersions,duedate,created,updated,"
        "issuelinks,subtasks"
    )


def test_summarize_issue_link_reports_outward_for_empty_but_present_outward_issue():
    # A permission-redacted linked issue can come back as a present but
    # empty outwardIssue ({}) rather than an absent key. Truthiness on
    # outwardIssue would treat that the same as "no outward issue at
    # all" and silently fall through to inwardIssue, inverting the
    # reported relationship direction.
    link = {
        "type": {"inward": "is blocked by", "outward": "blocks"},
        "outwardIssue": {},
    }
    assert jira._summarize_issue_link(link) == {
        "relationship": "blocks",
        "issue_key": None,
        "summary": None,
        "status": None,
    }


def test_summarize_comment_reports_restricted_for_empty_but_present_visibility():
    # Same presence-vs-truthiness bug class: a present-but-empty {}
    # visibility object must still be reported as restricted, not
    # collapsed into None (which this tool's docstring says means "not
    # restricted").
    summarized = jira._summarize_comment({"id": "1", "body": "x", "visibility": {}})
    assert summarized["visibility"] == {"type": None, "value": None}


def test_summarize_person_reports_present_for_empty_but_present_person():
    # Same presence-vs-truthiness bug class: a permission-redacted
    # assignee/reporter/creator/author can come back as a present but
    # empty {} rather than an absent key. Truthiness would collapse
    # that into the same None used for "genuinely unassigned", hiding
    # "assigned to someone you can't see" as "unassigned".
    assert jira._summarize_person({}) == {"account_id": None, "display_name": None}
    assert jira._summarize_person(None) is None


def test_get_issue_caps_an_oversized_extra_field_value(monkeypatch):
    long_value = "x" * (jira._ISSUE_DESCRIPTION_MAX_CHARS + 500)
    raw_issue = {
        "key": "ENG-1",
        "fields": {"summary": "ok", "customfield_10099": long_value},
    }
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(
            side_effect=[
                MockResponse(json_data=[_SITE_A]),
                MockResponse(json_data=raw_issue),
            ]
        ),
    )

    result = json.loads(jira.jira_get_issue("ENG-1", extra_fields="customfield_10099"))

    value = result["extra_field_values"]["customfield_10099"]
    assert len(value) <= jira._ISSUE_DESCRIPTION_MAX_CHARS + len("... [truncated]")
    assert value.endswith("[truncated]")
    assert result["extra_field_values_truncated"] is True


def test_get_issue_flattens_and_caps_an_adf_shaped_extra_field(monkeypatch):
    # A custom "Rich Text" field comes back as an ADF dict, the same
    # shape description does -- it must be flattened to plain text (not
    # returned as a raw JSON tree) and still capped like any other large
    # extra_fields value, since isinstance(value, str) alone would miss
    # it entirely.
    long_adf = {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {
                        "type": "text",
                        "text": "x" * (jira._ISSUE_DESCRIPTION_MAX_CHARS + 500),
                    }
                ],
            }
        ],
    }
    raw_issue = {
        "key": "ENG-1",
        "fields": {"summary": "ok", "customfield_10050": long_adf},
    }
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(
            side_effect=[
                MockResponse(json_data=[_SITE_A]),
                MockResponse(json_data=raw_issue),
            ]
        ),
    )

    result = json.loads(jira.jira_get_issue("ENG-1", extra_fields="customfield_10050"))

    value = result["extra_field_values"]["customfield_10050"]
    assert isinstance(value, str)
    assert len(value) <= jira._ISSUE_DESCRIPTION_MAX_CHARS + len("... [truncated]")
    assert result["extra_field_values_truncated"] is True


def test_get_issue_caps_an_oversized_list_shaped_extra_field(monkeypatch):
    # A multi-value picker/linked-records field can come back as a list
    # rather than a string -- isinstance(value, str) alone would let it
    # through uncapped.
    long_list = ["x" * 200 for _ in range(200)]
    raw_issue = {
        "key": "ENG-1",
        "fields": {"summary": "ok", "customfield_10060": long_list},
    }
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(
            side_effect=[
                MockResponse(json_data=[_SITE_A]),
                MockResponse(json_data=raw_issue),
            ]
        ),
    )

    result = json.loads(jira.jira_get_issue("ENG-1", extra_fields="customfield_10060"))

    value = result["extra_field_values"]["customfield_10060"]
    assert isinstance(value, str)
    assert len(value) <= jira._ISSUE_DESCRIPTION_MAX_CHARS + len("... [truncated]")
    assert result["extra_field_values_truncated"] is True


def test_path_segment_rejects_bare_dot_segments():
    # "." and ".." survive percent-encoding unchanged (always-unreserved
    # per RFC 3986); requests/urllib3 normalize them out of the final
    # URL before sending, silently redirecting to a different endpoint
    # unless rejected outright, matching utils.py's url_path_id.
    with pytest.raises(ValueError):
        jira._path_segment(".")
    with pytest.raises(ValueError):
        jira._path_segment("..")
    assert jira._path_segment("ENG-1") == "ENG-1"


def test_path_segment_rejects_blank_or_padded_values():
    # An empty issue_key (e.g. an unresolved templated variable from an
    # LLM caller) would otherwise silently build /rest/api/2/issue/,
    # hitting the issue-collection endpoint instead of a clear local
    # error naming the actual mistake.
    with pytest.raises(ValueError):
        jira._path_segment("")
    with pytest.raises(ValueError):
        jira._path_segment(" ENG-1")
    with pytest.raises(ValueError):
        jira._path_segment("ENG-1 ")


def test_path_segment_rejects_none():
    # str(None) is "None" -- non-blank, non-padded, so it would
    # otherwise sail through the blank/padded check above and silently
    # become a literal path segment instead of raising.
    with pytest.raises(ValueError):
        jira._path_segment(None)


def test_get_issue_rejects_a_blank_issue_key(monkeypatch):
    # _path_segment raises before _request ever makes a network call --
    # no site-resolution call happens either.
    mock_request = Mock()
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_get_issue(""))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_get_issue_response_always_has_a_truncated_key(monkeypatch):
    # Every other capped-response path in this package guarantees a
    # top-level "truncated" key is always present; the normal (fits
    # comfortably) path must not be the one exception.
    raw_issue = {"key": "ENG-1", "fields": {"summary": "ok"}}
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(
            side_effect=[
                MockResponse(json_data=[_SITE_A]),
                MockResponse(json_data=raw_issue),
            ]
        ),
    )

    result = json.loads(jira.jira_get_issue("ENG-1"))

    assert result["truncated"] is False


def test_get_issue_extra_field_values_survives_capped_dict_fallback(monkeypatch):
    # extra_field_values is a top-level field (a sibling of "issue"),
    # not nested inside it, specifically so success_with_capped_dict's
    # shrink pass -- which only touches `issue` -- can't collapse it.
    # This matters beyond just the *_truncated flag: each value here is
    # already capped before this point, so nesting the dict itself
    # inside `issue` would let the generic pass's dict-shrinking (which
    # halves a dict's KEY COUNT) collapse a single-key extra_field_values
    # straight to {} in one step once `issue` needs to shrink at all --
    # unlike a list, which degrades gradually, a 1-key dict has nowhere
    # to go but empty. It must survive fully even when the fallback
    # shrinks `issue` itself down to just its id.
    long_value = "x" * (jira._ISSUE_DESCRIPTION_MAX_CHARS + 500)
    raw_issue = {
        "key": "ENG-1",
        "fields": {
            "summary": "ok",
            "customfield_10099": long_value,
            "issuelinks": [
                {
                    "type": {"outward": "blocks"},
                    "outwardIssue": {
                        "key": f"ENG-{i}",
                        "fields": {"summary": "x" * 200, "status": {"name": "Open"}},
                    },
                }
                for i in range(50)
            ],
        },
    }
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(
            side_effect=[
                MockResponse(json_data=[_SITE_A]),
                MockResponse(json_data=raw_issue),
            ]
        ),
    )
    monkeypatch.setattr(jira, "get_tool_max_output_length", lambda: 25000)
    monkeypatch.setattr(jira_mcp_utils, "get_tool_max_output_length", lambda: 25000)

    raw_response = jira.jira_get_issue("ENG-1", extra_fields="customfield_10099")
    result = json.loads(raw_response)

    assert len(raw_response) <= 25000
    assert result["status"] == "success"
    assert "extra_field_values" not in result["issue"]
    assert result["extra_field_values"]["customfield_10099"].endswith("[truncated]")
    assert result["extra_field_values_truncated"] is True
    # `issue` (its issuelinks, specifically) is what had to shrink here,
    # not extra_field_values -- confirming the two are protected
    # independently rather than the whole response degrading together.
    assert len(result["issue"].get("issue_links", [])) < 50


def test_flatten_adf_mention_falls_back_to_account_id_without_text():
    adf = {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [{"type": "mention", "attrs": {"id": "u1"}}],
            }
        ],
    }
    assert jira._flatten_adf(adf) == "@u1"


def test_flatten_adf_mention_falls_back_to_at_sign_for_empty_but_present_id():
    # Presence, not truthiness: an empty-but-present id (a plausible
    # broken/deleted-user mention reference) must still render as "@",
    # not silently collapse to "" the same way a genuinely absent id
    # would.
    adf = {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [{"type": "mention", "attrs": {"id": ""}}],
            }
        ],
    }
    assert jira._flatten_adf(adf) == "@"


def test_flatten_adf_inline_card_falls_back_to_data_when_no_url():
    adf = {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {
                        "type": "inlineCard",
                        "attrs": {"data": {"name": "DW-782 config doc"}},
                    }
                ],
            }
        ],
    }
    assert jira._flatten_adf(adf) == "DW-782 config doc"


def test_flatten_adf_tolerates_non_dict_attrs():
    # A malformed node with `attrs` as a non-dict truthy value (e.g. a
    # string) must degrade to an empty rendering, not raise AttributeError.
    adf = {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": "before "},
                    {"type": "mention", "attrs": "not-a-dict"},
                    {"type": "text", "text": " after"},
                ],
            }
        ],
    }
    assert jira._flatten_adf(adf) == "before  after"


def test_flatten_adf_tolerates_non_dict_inline_card_data():
    adf = {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [{"type": "inlineCard", "attrs": {"data": "not-a-dict"}}],
            }
        ],
    }
    assert jira._flatten_adf(adf) is None


def test_flatten_adf_does_not_recurse_past_max_depth():
    # Build a pathologically nested blockquote chain -- well past
    # _ADF_MAX_DEPTH -- and confirm it degrades to a truncation marker
    # instead of raising RecursionError.
    node = {"type": "paragraph", "content": [{"type": "text", "text": "bottom"}]}
    for _ in range(jira._ADF_MAX_DEPTH + 20):
        node = {"type": "blockquote", "content": [node]}
    adf = {"type": "doc", "version": 1, "content": [node]}

    result = jira._flatten_adf(adf)

    assert result is not None
    assert jira._ADF_NESTED_TOO_DEEP_MESSAGE in result


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


def test_search_users_clamps_limit_and_offset(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(json_data=[]),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    json.loads(jira.jira_search_users("ada", limit=9999, start_at=-5))

    user_call = mock_request.call_args_list[1]
    assert user_call.kwargs["params"]["maxResults"] == jira.MAX_LIMIT
    assert user_call.kwargs["params"]["startAt"] == 0


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
