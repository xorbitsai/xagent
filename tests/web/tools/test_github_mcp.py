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


def test_parse_repo_strips_surrounding_whitespace():
    assert github._parse_repo(" octocat/Hello-World ") == ("octocat", "Hello-World")


@pytest.mark.parametrize(
    "value",
    [
        "octocat",
        "",
        "octocat/",
        "/Hello-World",
        "owner//repo",
        "owner/repo/extra",
        "/octocat/Hello-World/",
    ],
)
def test_parse_repo_rejects_malformed_input(value):
    """Extra/leading/trailing slashes must be rejected outright, not
    silently repaired into a subtly wrong (owner, name) pair."""
    with pytest.raises(ValueError, match="owner/repo"):
        github._parse_repo(value)


@pytest.mark.parametrize(
    "value",
    [
        "owner?x=y/repo",  # query-string injection via the owner segment
        "owner/repo#frag",  # fragment injection via the name segment
        "../owner",  # dot-segment traversal attempt as the owner
        "owner/..",  # dot-segment traversal attempt as the name
    ],
)
def test_parse_repo_rejects_injection_attempts(value):
    """owner/name each pass _parse_repo's single-slash shape check but must
    still be rejected by per-segment validation -- these are exactly one
    "/" apart with non-empty parts, so only the character-level guard in
    _encode_path_component catches them."""
    with pytest.raises(ValueError, match="not allowed"):
        github._parse_repo(value)


@pytest.mark.parametrize("value", ["", ".", "..", "a/b", "a?b", "a#b"])
def test_encode_path_component_rejects_forbidden_values(value):
    with pytest.raises(ValueError, match="not allowed"):
        github._encode_path_component(value, field="test")


def test_encode_path_component_percent_encodes_reserved_characters():
    """File paths (unlike owner/repo names) can legitimately contain spaces
    and other reserved characters; these must be percent-encoded rather
    than sent raw."""
    assert github._encode_path_component("my notes.md", field="path") == (
        "my%20notes.md"
    )


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


def test_search_repositories_sends_query_and_clamps_over_limit(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"total_count": 0, "items": []})
    )
    monkeypatch.setattr(github.requests, "request", mock_request)

    github.github_search_repositories("org:openai stars:>100", limit=500)

    assert mock_request.call_args.kwargs["method"] == "GET"
    assert mock_request.call_args.kwargs["url"] == (
        f"{github.GITHUB_BASE_URL}/search/repositories"
    )
    assert mock_request.call_args.kwargs["params"] == {
        "q": "org:openai stars:>100",
        "per_page": github.MAX_PER_PAGE,
    }
    assert mock_request.call_args.kwargs["timeout"] == github.DEFAULT_TIMEOUT_SECONDS


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


def test_list_issues_sends_non_default_state_and_labels(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data=[]))
    monkeypatch.setattr(github.requests, "request", mock_request)

    github.github_list_issues(
        "octocat/Hello-World", state="closed", labels="bug,urgent", limit=5
    )

    assert mock_request.call_args.kwargs["url"] == (
        f"{github.GITHUB_BASE_URL}/repos/octocat/Hello-World/issues"
    )
    assert mock_request.call_args.kwargs["params"] == {
        "state": "closed",
        "per_page": github.MAX_PER_PAGE,
        "page": 1,
        "labels": "bug,urgent",
    }


def test_list_issues_follows_pages_when_first_page_is_all_pull_requests(monkeypatch):
    """A PR-heavy (or all-PR) first page must not be reported as "no more
    issues" -- github_list_issues has to keep paging until it either fills
    the requested limit or GitHub runs out of pages."""
    first_page = [
        {
            "number": i,
            "title": f"pr {i}",
            "labels": [],
            "pull_request": {"url": "https://api.github.com/x"},
        }
        for i in range(1, github.MAX_PER_PAGE + 1)
    ]
    second_page = [{"number": 200, "title": "a real issue", "labels": []}]
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=first_page),
            MockResponse(json_data=second_page),
        ]
    )
    monkeypatch.setattr(github.requests, "request", mock_request)

    result = json.loads(github.github_list_issues("octocat/Hello-World", limit=5))

    assert result["status"] == "success"
    assert [issue["number"] for issue in result["issues"]] == [200]
    assert result["truncated"] is False
    assert mock_request.call_count == 2
    first_call, second_call = mock_request.call_args_list
    assert first_call.kwargs["params"]["page"] == 1
    assert second_call.kwargs["params"]["page"] == 2


def test_list_issues_reports_truncated_when_limit_reached_mid_page(monkeypatch):
    page = [{"number": i, "title": f"issue {i}", "labels": []} for i in range(1, 11)]
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(return_value=MockResponse(json_data=page)),
    )

    result = json.loads(github.github_list_issues("octocat/Hello-World", limit=3))

    assert result["status"] == "success"
    assert len(result["issues"]) == 3
    assert result["truncated"] is True


def test_list_issues_stops_at_max_pages_and_reports_truncated(monkeypatch):
    """When every page is entirely pull requests, the outer loop must still
    terminate at MAX_ISSUE_PAGES (not loop forever) and report truncated,
    since real issues might exist on pages beyond the bound."""
    pr_only_page = [
        {
            "number": i,
            "title": f"pr {i}",
            "labels": [],
            "pull_request": {"url": "https://api.github.com/x"},
        }
        for i in range(1, github.MAX_PER_PAGE + 1)
    ]
    mock_request = Mock(return_value=MockResponse(json_data=pr_only_page))
    monkeypatch.setattr(github.requests, "request", mock_request)

    result = json.loads(github.github_list_issues("octocat/Hello-World", limit=5))

    assert result["status"] == "success"
    assert result["issues"] == []
    assert result["truncated"] is True


def test_list_issues_preserves_partial_results_on_mid_pagination_failure(monkeypatch):
    """A rate limit (or any transient error) on page 2+ must not discard the
    issues already collected from page 1, matching slack.py's channel
    listing precedent for the same failure mode."""
    # A full (100-item), half-PR page: the requested limit (clamped to 100)
    # isn't satisfied by the 50 real issues it yields, and it isn't short
    # either -- so the loop must actually attempt page 2 (which then fails)
    # instead of stopping after page 1 for either reason.
    first_page = [
        {
            "number": i,
            "title": f"item {i}",
            "labels": [],
            **(
                {"pull_request": {"url": "https://api.github.com/x"}}
                if i % 2 == 0
                else {}
            ),
        }
        for i in range(1, github.MAX_PER_PAGE + 1)
    ]
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data=first_page),
            MockResponse(json_data={"message": "rate limited"}, status_code=429),
        ]
    )
    monkeypatch.setattr(github.requests, "request", mock_request)

    result = json.loads(github.github_list_issues("octocat/Hello-World", limit=100_000))

    assert result["status"] == "success"
    assert len(result["issues"]) == github.MAX_PER_PAGE // 2
    assert result["truncated"] is True
    assert "rate limited" in result["error"]
    assert mock_request.call_count == 2


def test_list_issues_reraises_on_first_page_failure(monkeypatch):
    """A failure with nothing collected yet must still surface as an error
    (not a "successful" empty list), same as before pagination was added."""
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"message": "Not Found"}, status_code=404
            )
        ),
    )

    result = json.loads(github.github_list_issues("octocat/missing-repo"))

    assert result["status"] == "error"
    assert "Not Found" in result["message"]


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


def test_get_issue_builds_exact_url(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"number": 42, "title": "x", "labels": []})
    )
    monkeypatch.setattr(github.requests, "request", mock_request)

    github.github_get_issue("octocat/Hello-World", 42)

    assert mock_request.call_args.kwargs["method"] == "GET"
    assert mock_request.call_args.kwargs["url"] == (
        f"{github.GITHUB_BASE_URL}/repos/octocat/Hello-World/issues/42"
    )


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


def test_list_pull_requests_sends_non_default_state_and_limit(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data=[]))
    monkeypatch.setattr(github.requests, "request", mock_request)

    github.github_list_pull_requests("octocat/Hello-World", state="closed", limit=5)

    assert mock_request.call_args.kwargs["url"] == (
        f"{github.GITHUB_BASE_URL}/repos/octocat/Hello-World/pulls"
    )
    assert mock_request.call_args.kwargs["params"] == {
        "state": "closed",
        "per_page": 5,
    }


def test_get_pull_request_returns_summary(monkeypatch):
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"number": 3, "merged": False})),
    )

    result = json.loads(github.github_get_pull_request("octocat/Hello-World", 3))

    assert result["status"] == "success"
    assert result["pull_request"]["number"] == 3


def test_get_pull_request_builds_exact_url(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"number": 9}))
    monkeypatch.setattr(github.requests, "request", mock_request)

    github.github_get_pull_request("octocat/Hello-World", 9)

    assert mock_request.call_args.kwargs["url"] == (
        f"{github.GITHUB_BASE_URL}/repos/octocat/Hello-World/pulls/9"
    )


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


def test_get_file_contents_sends_ref_param_when_provided(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "type": "file",
                "path": "main.py",
                "encoding": "utf-8",
                "content": "hi",
            }
        )
    )
    monkeypatch.setattr(github.requests, "request", mock_request)

    github.github_get_file_contents(
        "octocat/Hello-World", "main.py", ref="feature-branch"
    )

    assert mock_request.call_args.kwargs["url"] == (
        f"{github.GITHUB_BASE_URL}/repos/octocat/Hello-World/contents/main.py"
    )
    assert mock_request.call_args.kwargs["params"] == {"ref": "feature-branch"}


def test_get_file_contents_omits_ref_param_when_not_provided(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "type": "file",
                "path": "main.py",
                "encoding": "utf-8",
                "content": "hi",
            }
        )
    )
    monkeypatch.setattr(github.requests, "request", mock_request)

    github.github_get_file_contents("octocat/Hello-World", "main.py")

    assert mock_request.call_args.kwargs["params"] == {}


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


def test_get_file_contents_accepts_empty_path_for_repo_root(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data=[{"name": "README.md", "path": "README.md", "type": "file"}]
        )
    )
    monkeypatch.setattr(github.requests, "request", mock_request)

    result = json.loads(github.github_get_file_contents("octocat/Hello-World", ""))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"].endswith("/contents/")


@pytest.mark.parametrize(
    "path",
    [
        "/src/main.py",
        "src/main.py/",
        "src//main.py",
        "/",
        "//",
        "src/../etc",  # dot-segment traversal attempt within a path
        "src/main.py?x=y",  # query-string injection within a path segment
    ],
)
def test_get_file_contents_rejects_malformed_path(path, monkeypatch):
    """Leading/trailing/consecutive slashes, dot-segments, and reserved
    characters must be rejected outright, not silently interpolated into a
    malformed request URL."""
    mock_request = Mock()
    monkeypatch.setattr(github.requests, "request", mock_request)

    result = json.loads(github.github_get_file_contents("octocat/Hello-World", path))

    assert result["status"] == "error"
    assert "not allowed" in result["message"]
    mock_request.assert_not_called()


def test_get_file_contents_percent_encodes_path_segments(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "type": "file",
                "path": "my notes.md",
                "encoding": "utf-8",
                "content": "hi",
            }
        )
    )
    monkeypatch.setattr(github.requests, "request", mock_request)

    result = json.loads(
        github.github_get_file_contents("octocat/Hello-World", "docs/my notes.md")
    )

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"].endswith("/contents/docs/my%20notes.md")


def test_get_file_contents_reports_error_for_oversized_file(monkeypatch):
    """encoding == "none" means GitHub omitted the content because the file
    exceeds the Contents API's size limit -- this must surface as an error,
    not a silent empty-string "success"."""
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"type": "file", "path": "big.bin", "encoding": "none"}
            )
        ),
    )

    result = json.loads(
        github.github_get_file_contents("octocat/Hello-World", "big.bin")
    )

    assert result["status"] == "error"
    assert "too large" in result["message"]


def test_get_file_contents_flags_directory_at_the_1000_entry_cap(monkeypatch):
    entries = [
        {"name": f"file{i}.py", "path": f"src/file{i}.py", "type": "file"}
        for i in range(1000)
    ]
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(return_value=MockResponse(json_data=entries)),
    )

    result = json.loads(github.github_get_file_contents("octocat/Hello-World", "src"))

    assert result["status"] == "success"
    assert result["truncated"] is True


def test_get_file_contents_directory_under_cap_is_not_truncated(monkeypatch):
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data=[{"name": "main.py", "path": "src/main.py", "type": "file"}]
            )
        ),
    )

    result = json.loads(github.github_get_file_contents("octocat/Hello-World", "src"))

    assert result["status"] == "success"
    assert result["truncated"] is False


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


def test_list_commits_sends_path_and_clamps_over_limit(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data=[]))
    monkeypatch.setattr(github.requests, "request", mock_request)

    github.github_list_commits("octocat/Hello-World", path="src/main.py", limit=500)

    assert mock_request.call_args.kwargs["url"] == (
        f"{github.GITHUB_BASE_URL}/repos/octocat/Hello-World/commits"
    )
    assert mock_request.call_args.kwargs["params"] == {
        "per_page": github.MAX_PER_PAGE,
        "path": "src/main.py",
    }


def test_list_commits_omits_path_param_when_not_provided(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data=[]))
    monkeypatch.setattr(github.requests, "request", mock_request)

    github.github_list_commits("octocat/Hello-World")

    assert "path" not in mock_request.call_args.kwargs["params"]


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


def test_search_code_sends_query_and_clamps_over_limit(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"total_count": 0, "items": []})
    )
    monkeypatch.setattr(github.requests, "request", mock_request)

    github.github_search_code("repo:octocat/Hello-World def parse", limit=500)

    assert mock_request.call_args.kwargs["url"] == (
        f"{github.GITHUB_BASE_URL}/search/code"
    )
    assert mock_request.call_args.kwargs["params"] == {
        "q": "repo:octocat/Hello-World def parse",
        "per_page": github.MAX_PER_PAGE,
    }


def test_tool_returns_error_payload_on_missing_token(monkeypatch):
    monkeypatch.delenv("GITHUB_ACCESS_TOKEN")

    result = json.loads(github.github_get_repository("octocat/Hello-World"))

    assert result["status"] == "error"
    assert "GITHUB_ACCESS_TOKEN" in result["message"]
