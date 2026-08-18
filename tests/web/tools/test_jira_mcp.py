import json
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import jira


class MockResponse:
    def __init__(
        self, json_data=None, status_code: int = 200, text: str = "", url: str = ""
    ):
        self._json_data = json_data if json_data is not None else {}
        self.status_code = status_code
        self.text = text or (json.dumps(self._json_data) if json_data else "")
        self.content = self.text.encode()
        self.url = url

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


def test_request_builds_url_with_resolved_cloud_id(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"id": "10001", "key": "ENG-1"})
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    jira._request("GET", "site-a", "/rest/api/2/issue/ENG-1")

    assert mock_request.call_args.kwargs["url"] == (
        "https://api.atlassian.com/ex/jira/site-a/rest/api/2/issue/ENG-1"
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
    assert result["projects"] == [{"id": "1", "key": "ENG", "name": "Engineering"}]
    assert result["truncated"] is False
    project_call = mock_request.call_args_list[1]
    assert project_call.kwargs["url"] == (
        "https://api.atlassian.com/ex/jira/site-a/rest/api/2/project/search"
    )


def test_search_issues_sends_jql_and_reports_total(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=[_SITE_A]),
            MockResponse(
                json_data={
                    "issues": [{"key": "ENG-1", "fields": {"summary": "Bug"}}],
                    "total": 5,
                }
            ),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_search_issues("project = ENG"))

    assert result["status"] == "success"
    assert result["issues"][0]["key"] == "ENG-1"
    assert result["total"] == 5
    assert result["truncated"] is True
    search_call = mock_request.call_args_list[1]
    assert search_call.kwargs["params"]["jql"] == "project = ENG"


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
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(
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
        ),
    )

    result = json.loads(jira.jira_search_users("ada"))

    assert result["status"] == "success"
    assert result["users"] == [
        {"account_id": "u1", "display_name": "Ada Lovelace", "email": "ada@example.com"}
    ]


def test_jira_app_registry_includes_offline_access_scope():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    jira_app = next(
        row for row in get_builtin_public_mcp_app_rows() if row["app_id"] == "jira"
    )
    assert "offline_access" in jira_app["oauth_scopes"]
