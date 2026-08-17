import base64
import json
from unittest.mock import Mock

import pytest

from xagent.web.tools.mcp import github


class MockResponse:
    def __init__(self, json_data=None, status_code: int = 200, content: bytes = b"{}"):
        self._json_data = json_data if json_data is not None else {}
        self.status_code = status_code
        self.content = content if json_data is None else json.dumps(json_data).encode()

    def json(self):
        return self._json_data


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("GITHUB_ACCESS_TOKEN", "access-token")


def test_headers_require_access_token(monkeypatch):
    monkeypatch.delenv("GITHUB_ACCESS_TOKEN")

    with pytest.raises(ValueError, match="GITHUB_ACCESS_TOKEN"):
        github._headers()


def test_headers_include_bearer_token_and_api_version():
    assert github._headers() == {
        "Authorization": "Bearer access-token",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def test_parse_repo_splits_owner_and_name():
    assert github._parse_repo("octocat/Hello-World") == ("octocat", "Hello-World")


def test_parse_repo_strips_slashes_and_whitespace():
    assert github._parse_repo(" /octocat/Hello-World/ ") == ("octocat", "Hello-World")


@pytest.mark.parametrize("value", ["octocat", "", "octocat/", "/Hello-World"])
def test_parse_repo_rejects_malformed_input(value):
    with pytest.raises(ValueError, match="owner/repo"):
        github._parse_repo(value)


def test_request_raises_with_message_on_error(monkeypatch):
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"message": "Not Found"}, status_code=404
            )
        ),
    )

    with pytest.raises(RuntimeError, match="Not Found"):
        github._request("GET", "/repos/octocat/missing")


def test_request_folds_validation_errors_into_message(monkeypatch):
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "message": "Validation Failed",
                    "errors": [{"field": "title", "message": "cannot be blank"}],
                },
                status_code=422,
            )
        ),
    )

    with pytest.raises(RuntimeError, match="cannot be blank"):
        github._request("POST", "/repos/octocat/Hello-World/issues")


def test_request_returns_empty_dict_on_no_content(monkeypatch):
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(return_value=MockResponse(status_code=204, content=b"")),
    )

    assert github._request("DELETE", "/repos/octocat/Hello-World") == {}


def test_search_repositories_returns_summaries(monkeypatch):
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "total_count": 1,
                    "items": [
                        {
                            "full_name": "octocat/Hello-World",
                            "description": "demo",
                            "private": False,
                            "default_branch": "main",
                            "stargazers_count": 5,
                            "open_issues_count": 1,
                            "html_url": "https://github.com/octocat/Hello-World",
                            "language": "Python",
                            "updated_at": "2026-01-01T00:00:00Z",
                        }
                    ],
                }
            )
        ),
    )

    result = json.loads(github.github_search_repositories("Hello-World"))

    assert result["status"] == "success"
    assert result["total_count"] == 1
    assert result["repositories"][0]["full_name"] == "octocat/Hello-World"


def test_get_current_user_returns_profile(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "login": "octocat",
                "id": 1,
                "name": "The Octocat",
                "email": "octocat@github.com",
                "company": "GitHub",
                "bio": "",
                "public_repos": 8,
                "followers": 100,
                "following": 9,
                "html_url": "https://github.com/octocat",
            }
        )
    )
    monkeypatch.setattr(github.requests, "request", mock_request)

    result = json.loads(github.github_get_current_user())

    assert result["status"] == "success"
    assert result["user"]["login"] == "octocat"
    assert result["user"]["email"] == "octocat@github.com"
    assert mock_request.call_args.kwargs["url"].endswith("/user")


def test_get_current_user_returns_error_payload_on_failure(monkeypatch):
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"message": "Bad credentials"}, status_code=401
            )
        ),
    )

    result = json.loads(github.github_get_current_user())

    assert result["status"] == "error"
    assert "Bad credentials" in result["message"]


def test_get_repository_returns_summary(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"full_name": "octocat/Hello-World"})
    )
    monkeypatch.setattr(github.requests, "request", mock_request)

    result = json.loads(github.github_get_repository("octocat/Hello-World"))

    assert result["status"] == "success"
    assert result["repository"]["full_name"] == "octocat/Hello-World"
    assert mock_request.call_args.kwargs["url"].endswith("/repos/octocat/Hello-World")


def test_get_repository_rejects_malformed_repo(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(github.requests, "request", mock_request)

    result = json.loads(github.github_get_repository("not-a-repo"))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_list_issues_excludes_pull_requests(monkeypatch):
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data=[
                    {"number": 1, "title": "a real issue", "labels": []},
                    {
                        "number": 2,
                        "title": "actually a PR",
                        "labels": [],
                        "pull_request": {"url": "https://api.github.com/x"},
                    },
                ]
            )
        ),
    )

    result = json.loads(github.github_list_issues("octocat/Hello-World"))

    assert result["status"] == "success"
    numbers = [issue["number"] for issue in result["issues"]]
    assert numbers == [1]


def test_list_issues_normalizes_dict_labels(monkeypatch):
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data=[
                    {
                        "number": 1,
                        "title": "labeled",
                        "labels": [{"name": "bug"}, "enhancement"],
                    }
                ]
            )
        ),
    )

    result = json.loads(github.github_list_issues("octocat/Hello-World"))

    assert result["issues"][0]["labels"] == ["bug", "enhancement"]


def test_get_issue_flags_pull_request(monkeypatch):
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "number": 5,
                    "title": "a PR",
                    "labels": [],
                    "pull_request": {"url": "https://api.github.com/x"},
                }
            )
        ),
    )

    result = json.loads(github.github_get_issue("octocat/Hello-World", 5))

    assert result["issue"]["is_pull_request"] is True


def test_create_issue_splits_comma_separated_labels(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"number": 10, "title": "new issue", "labels": []}
        )
    )
    monkeypatch.setattr(github.requests, "request", mock_request)

    result = json.loads(
        github.github_create_issue(
            "octocat/Hello-World", "new issue", body="details", labels="bug, urgent"
        )
    )

    assert result["status"] == "success"
    sent = mock_request.call_args.kwargs["json"]
    assert sent["title"] == "new issue"
    assert sent["body"] == "details"
    assert sent["labels"] == ["bug", "urgent"]


def test_comment_on_issue_posts_body(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"id": 99, "html_url": "https://github.com/x/x/issues/1#c99"}
        )
    )
    monkeypatch.setattr(github.requests, "request", mock_request)

    result = json.loads(
        github.github_comment_on_issue("octocat/Hello-World", 1, "looks good")
    )

    assert result["status"] == "success"
    assert result["comment_id"] == 99
    assert mock_request.call_args.kwargs["json"] == {"body": "looks good"}
    assert mock_request.call_args.kwargs["url"].endswith(
        "/repos/octocat/Hello-World/issues/1/comments"
    )


def test_list_pull_requests_returns_summaries(monkeypatch):
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data=[
                    {
                        "number": 3,
                        "title": "fix bug",
                        "head": {"ref": "fix-branch"},
                        "base": {"ref": "main"},
                    }
                ]
            )
        ),
    )

    result = json.loads(github.github_list_pull_requests("octocat/Hello-World"))

    assert result["status"] == "success"
    assert result["pull_requests"][0]["head"] == "fix-branch"
    assert result["pull_requests"][0]["base"] == "main"


def test_get_pull_request_returns_summary(monkeypatch):
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"number": 3, "merged": False})),
    )

    result = json.loads(github.github_get_pull_request("octocat/Hello-World", 3))

    assert result["status"] == "success"
    assert result["pull_request"]["number"] == 3


def test_create_pull_request_sends_head_and_base(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"number": 7, "title": "add feature"})
    )
    monkeypatch.setattr(github.requests, "request", mock_request)

    result = json.loads(
        github.github_create_pull_request(
            "octocat/Hello-World", "add feature", "feature-branch", "main"
        )
    )

    assert result["status"] == "success"
    sent = mock_request.call_args.kwargs["json"]
    assert sent == {"title": "add feature", "head": "feature-branch", "base": "main"}


def test_get_file_contents_decodes_base64_file(monkeypatch):
    encoded = base64.b64encode(b"print('hi')\n").decode()
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "type": "file",
                    "path": "main.py",
                    "sha": "abc123",
                    "size": 12,
                    "encoding": "base64",
                    "content": encoded,
                }
            )
        ),
    )

    result = json.loads(
        github.github_get_file_contents("octocat/Hello-World", "main.py")
    )

    assert result["status"] == "success"
    assert result["type"] == "file"
    assert result["content"] == "print('hi')\n"


def test_get_file_contents_lists_directory(monkeypatch):
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data=[
                    {"name": "main.py", "path": "src/main.py", "type": "file"},
                    {"name": "lib", "path": "src/lib", "type": "dir"},
                ]
            )
        ),
    )

    result = json.loads(github.github_get_file_contents("octocat/Hello-World", "src"))

    assert result["status"] == "success"
    assert result["type"] == "directory"
    assert len(result["entries"]) == 2


def test_get_file_contents_rejects_non_file_type(monkeypatch):
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"type": "symlink"})),
    )

    result = json.loads(github.github_get_file_contents("octocat/Hello-World", "link"))

    assert result["status"] == "error"
    assert "symlink" in result["message"]


def test_list_commits_returns_summaries(monkeypatch):
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data=[
                    {
                        "sha": "deadbeef",
                        "commit": {
                            "message": "fix bug",
                            "author": {"name": "Alice", "date": "2026-01-01T00:00:00Z"},
                        },
                        "html_url": "https://github.com/x/x/commit/deadbeef",
                    }
                ]
            )
        ),
    )

    result = json.loads(github.github_list_commits("octocat/Hello-World"))

    assert result["status"] == "success"
    assert result["commits"][0]["message"] == "fix bug"
    assert result["commits"][0]["author"] == "Alice"


def test_search_code_returns_items(monkeypatch):
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "total_count": 1,
                    "items": [
                        {
                            "name": "main.py",
                            "path": "src/main.py",
                            "repository": {"full_name": "octocat/Hello-World"},
                            "html_url": "https://github.com/x/x/blob/main/src/main.py",
                        }
                    ],
                }
            )
        ),
    )

    result = json.loads(github.github_search_code("def parse"))

    assert result["status"] == "success"
    assert result["items"][0]["repository"] == "octocat/Hello-World"


def test_tool_returns_error_payload_on_missing_token(monkeypatch):
    monkeypatch.delenv("GITHUB_ACCESS_TOKEN")

    result = json.loads(github.github_get_repository("octocat/Hello-World"))

    assert result["status"] == "error"
    assert "GITHUB_ACCESS_TOKEN" in result["message"]
