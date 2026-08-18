import json
from unittest.mock import Mock

import pytest

from xagent.web.tools.mcp import linear


class MockResponse:
    def __init__(self, json_data=None, status_code: int = 200, text: str = ""):
        self._json_data = json_data if json_data is not None else {}
        self.status_code = status_code
        self.text = text or (json.dumps(self._json_data) if json_data else "")

    def json(self):
        return self._json_data


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("LINEAR_ACCESS_TOKEN", "access-token")


def test_headers_require_access_token(monkeypatch):
    monkeypatch.delenv("LINEAR_ACCESS_TOKEN")

    with pytest.raises(ValueError, match="LINEAR_ACCESS_TOKEN"):
        linear._headers()


def test_headers_include_bearer_token_and_json_content_type():
    assert linear._headers() == {
        "Authorization": "Bearer access-token",
        "Content-Type": "application/json",
    }


def test_graphql_sends_query_and_variables_as_json_body(monkeypatch):
    mock_post = Mock(return_value=MockResponse(json_data={"data": {"ok": True}}))
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = linear._graphql("query { ok }", {"a": 1})

    assert result == {"ok": True}
    assert mock_post.call_args.args[0] == linear.LINEAR_GRAPHQL_URL
    assert mock_post.call_args.kwargs["json"] == {
        "query": "query { ok }",
        "variables": {"a": 1},
    }


def test_graphql_raises_on_top_level_errors_with_200_status(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={"errors": [{"message": "Field not found"}]}
            )
        ),
    )

    with pytest.raises(RuntimeError, match="Field not found"):
        linear._graphql("query { bad }")


def test_graphql_raises_with_structured_error_body_on_http_error(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                status_code=400,
                json_data={"errors": [{"message": "Invalid token"}]},
            )
        ),
    )

    with pytest.raises(RuntimeError, match="Invalid token"):
        linear._graphql("query { viewer { id } }")


def test_graphql_truncates_unstructured_error_body_on_http_error(monkeypatch):
    long_body = "x" * 5000
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(return_value=MockResponse(status_code=500, text=long_body)),
    )

    with pytest.raises(RuntimeError) as excinfo:
        linear._graphql("query { viewer { id } }")

    assert "[truncated]" in str(excinfo.value)
    assert len(str(excinfo.value)) < len(long_body)


def test_get_current_user_returns_profile(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "viewer": {
                            "id": "u1",
                            "name": "Ada",
                            "email": "ada@example.com",
                            "displayName": "ada",
                            "admin": False,
                        }
                    }
                }
            )
        ),
    )

    result = json.loads(linear.linear_get_current_user())

    assert result["status"] == "success"
    assert result["user"]["email"] == "ada@example.com"


def test_get_current_user_returns_error_payload_on_failure(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                status_code=401,
                json_data={"errors": [{"message": "Authentication required"}]},
            )
        ),
    )

    result = json.loads(linear.linear_get_current_user())

    assert result["status"] == "error"
    assert "Authentication required" in result["message"]


def test_list_teams_returns_nodes(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "teams": {
                            "nodes": [{"id": "t1", "key": "ENG", "name": "Engineering"}]
                        }
                    }
                }
            )
        ),
    )

    result = json.loads(linear.linear_list_teams())

    assert result["status"] == "success"
    assert result["teams"] == [{"id": "t1", "key": "ENG", "name": "Engineering"}]


def test_search_issues_applies_team_filter(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {
                    "issues": {
                        "nodes": [
                            {
                                "id": "i1",
                                "identifier": "ENG-1",
                                "title": "Bug",
                            }
                        ]
                    }
                }
            }
        )
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_search_issues(team_id="t1"))

    assert result["status"] == "success"
    assert result["issues"][0]["identifier"] == "ENG-1"
    variables = mock_post.call_args.kwargs["json"]["variables"]
    assert variables["teamId"] == "t1"
    query_text = mock_post.call_args.kwargs["json"]["query"]
    assert query_text.count("{") == query_text.count("}")


def test_search_issues_filters_by_title_query_client_side(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "issues": {
                            "nodes": [
                                {
                                    "id": "i1",
                                    "identifier": "ENG-1",
                                    "title": "Login bug",
                                },
                                {
                                    "id": "i2",
                                    "identifier": "ENG-2",
                                    "title": "Docs typo",
                                },
                            ]
                        }
                    }
                }
            )
        ),
    )

    result = json.loads(linear.linear_search_issues(query="login"))

    assert result["status"] == "success"
    assert len(result["issues"]) == 1
    assert result["issues"][0]["identifier"] == "ENG-1"


def test_get_issue_returns_not_found_error_when_missing(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(return_value=MockResponse(json_data={"data": {"issue": None}})),
    )

    result = json.loads(linear.linear_get_issue("ENG-999"))

    assert result["status"] == "error"
    assert "not found" in result["message"]


def test_create_issue_sends_expected_input(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {
                    "issueCreate": {
                        "success": True,
                        "issue": {
                            "id": "i1",
                            "identifier": "ENG-1",
                            "title": "New bug",
                        },
                    }
                }
            }
        )
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(
        linear.linear_create_issue(
            team_id="t1", title="New bug", assignee_id="u1", priority=2
        )
    )

    assert result["status"] == "success"
    assert result["issue"]["identifier"] == "ENG-1"
    sent_input = mock_post.call_args.kwargs["json"]["variables"]["input"]
    assert sent_input == {
        "teamId": "t1",
        "title": "New bug",
        "assigneeId": "u1",
        "priority": 2,
    }


def test_create_issue_reports_error_when_linear_reports_failure(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={"data": {"issueCreate": {"success": False, "issue": None}}}
            )
        ),
    )

    result = json.loads(linear.linear_create_issue(team_id="t1", title="New bug"))

    assert result["status"] == "error"
    assert "not created" in result["message"]


def test_update_issue_requires_at_least_one_field(monkeypatch):
    mock_post = Mock()
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_update_issue("ENG-1"))

    assert result["status"] == "error"
    assert "No fields" in result["message"]
    mock_post.assert_not_called()


def test_update_issue_omits_priority_when_left_default(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {
                    "issueUpdate": {
                        "success": True,
                        "issue": {"id": "i1", "identifier": "ENG-1"},
                    }
                }
            }
        )
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_update_issue("ENG-1", state_id="s1"))

    assert result["status"] == "success"
    sent_input = mock_post.call_args.kwargs["json"]["variables"]["input"]
    assert sent_input == {"stateId": "s1"}


def test_add_comment_returns_created_comment(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "commentCreate": {
                            "success": True,
                            "comment": {"id": "c1", "body": "Looks good"},
                        }
                    }
                }
            )
        ),
    )

    result = json.loads(linear.linear_add_comment("ENG-1", "Looks good"))

    assert result["status"] == "success"
    assert result["comment"]["body"] == "Looks good"


def test_search_users_filters_by_name_or_email(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "users": {
                            "nodes": [
                                {"id": "u1", "name": "Ada", "email": "ada@example.com"},
                                {"id": "u2", "name": "Bob", "email": "bob@example.com"},
                            ]
                        }
                    }
                }
            )
        ),
    )

    result = json.loads(linear.linear_search_users("ada"))

    assert result["status"] == "success"
    assert len(result["users"]) == 1
    assert result["users"][0]["id"] == "u1"


def test_linear_app_registry_requests_read_and_write_scopes():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    linear_app = next(
        row for row in get_builtin_public_mcp_app_rows() if row["app_id"] == "linear"
    )
    assert linear_app["oauth_scopes"] == ["read", "write"]
