import base64
import json
from unittest.mock import Mock

import pytest

from xagent.web.tools.mcp import github


class MockResponse:
    def __init__(
        self,
        json_data=None,
        status_code: int = 200,
        content: bytes = b"{}",
        headers: dict | None = None,
    ):
        self._json_data = json_data if json_data is not None else {}
        self.status_code = status_code
        self.content = content if json_data is None else json.dumps(json_data).encode()
        self.headers = headers or {}

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


@pytest.mark.parametrize("value", ["", ".", ".."])
def test_encode_file_path_segment_rejects_forbidden_values(value):
    with pytest.raises(ValueError, match="not allowed"):
        github._encode_file_path_segment(value, field="path")


def test_encode_file_path_segment_allows_and_encodes_question_mark_and_hash():
    """Unlike _encode_path_component (owner/repo), '?' and '#' are legitimate
    filename characters -- they must be percent-encoded rather than rejected,
    since the encoded form can't be reinterpreted as a query/fragment once
    it reaches the URL."""
    assert github._encode_file_path_segment("why?.md", field="path") == "why%3F.md"
    assert github._encode_file_path_segment("issue#1.txt", field="path") == (
        "issue%231.txt"
    )


def test_encode_file_path_segment_percent_encodes_reserved_characters():
    assert github._encode_file_path_segment("my notes.md", field="path") == (
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


def test_request_raw_includes_rate_limit_metadata_in_error(monkeypatch):
    """Retry-After/X-RateLimit-* headers must not be silently dropped -- a
    rate limit or transient server error would otherwise be indistinguishable
    from a validation/permission failure in the raised message."""
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"message": "API rate limit exceeded"},
                status_code=403,
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": "1699999999",
                },
            )
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        github._request("GET", "/repos/octocat/Hello-World/issues")

    assert "retry_after=60s" in str(excinfo.value)
    assert "rate_limit_reset=1699999999" in str(excinfo.value)


def test_request_raw_omits_rate_limit_metadata_when_not_rate_limited(monkeypatch):
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"message": "Not Found"}, status_code=404
            )
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        github._request("GET", "/repos/octocat/missing")

    assert "retry_after" not in str(excinfo.value)
    assert "rate_limit_reset" not in str(excinfo.value)


@pytest.mark.parametrize(
    "limit, expected",
    [
        (0, 1),
        (1, 1),
        (30, 30),
        (100, 100),
        (101, 100),
        (-5, 1),
        ("not-a-number", 30),  # falls back to the default, then clamps
    ],
)
def test_clamp_limit_boundaries(limit, expected):
    assert github._clamp_limit(limit) == expected


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
        "octocat/Hello-World", state="closed", labels="bug,urgent", limit=5, page=2
    )

    assert mock_request.call_args.kwargs["method"] == "GET"
    assert mock_request.call_args.kwargs["url"] == (
        f"{github.GITHUB_BASE_URL}/repos/octocat/Hello-World/issues"
    )
    assert mock_request.call_args.kwargs["timeout"] == github.DEFAULT_TIMEOUT_SECONDS
    assert mock_request.call_args.kwargs["headers"]["Authorization"] == (
        "Bearer access-token"
    )
    assert mock_request.call_args.kwargs["params"] == {
        "state": "closed",
        "per_page": github.MAX_PER_PAGE,
        "page": 2,
        "labels": "bug,urgent",
    }


def test_list_issues_follows_pages_when_first_page_is_all_pull_requests(monkeypatch):
    """A PR-heavy (or all-PR) first page must not be reported as "no more
    issues" -- github_list_issues has to keep paging until it either fills
    the requested limit or GitHub runs out of pages (confirmed here via the
    Link header, since the second page is genuinely the last one)."""
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
            MockResponse(
                json_data=first_page,
                headers={"Link": '<https://api.github.com/x?page=2>; rel="next"'},
            ),
            MockResponse(json_data=second_page),  # no Link header -- last page
        ]
    )
    monkeypatch.setattr(github.requests, "request", mock_request)

    result = json.loads(github.github_list_issues("octocat/Hello-World", limit=5))

    assert result["status"] == "success"
    assert [issue["number"] for issue in result["issues"]] == [200]
    assert result["truncated"] is False
    assert result["next_page"] is None
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
    # A mid-page limit hit resumes the SAME page, skipping the raw items
    # already consumed from it.
    assert result["next_page"] == 1
    assert result["next_skip"] == 3


def test_list_issues_skip_resumes_exactly_where_a_mid_page_cut_left_off(monkeypatch):
    """Round-trip: the next_page/next_skip a truncated call returns must let
    a follow-up call pick up the remaining items from the same page."""
    page = [{"number": i, "title": f"issue {i}", "labels": []} for i in range(1, 11)]
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(return_value=MockResponse(json_data=page)),
    )

    first = json.loads(github.github_list_issues("octocat/Hello-World", limit=3))
    assert [issue["number"] for issue in first["issues"]] == [1, 2, 3]
    assert first["next_page"] == 1
    assert first["next_skip"] == 3

    second = json.loads(
        github.github_list_issues(
            "octocat/Hello-World",
            limit=3,
            page=first["next_page"],
            skip=first["next_skip"],
        )
    )
    assert [issue["number"] for issue in second["issues"]] == [4, 5, 6]
    assert second["next_page"] == 1
    assert second["next_skip"] == 6


def test_list_issues_trailing_prs_after_limit_do_not_count_as_truncation(monkeypatch):
    """Hitting the limit with only pull requests left on the final page is
    not a truncation -- PRs are excluded from the result anyway, so nothing
    the caller asked for was left behind."""
    page = [
        {"number": 1, "title": "issue 1", "labels": []},
        {"number": 2, "title": "issue 2", "labels": []},
        {
            "number": 3,
            "title": "pr 3",
            "labels": [],
            "pull_request": {"url": "https://api.github.com/x"},
        },
    ]
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(return_value=MockResponse(json_data=page)),  # no Link header
    )

    result = json.loads(github.github_list_issues("octocat/Hello-World", limit=2))

    assert result["status"] == "success"
    assert len(result["issues"]) == 2
    assert result["truncated"] is False
    assert result["next_page"] is None


def test_list_issues_trailing_prs_after_limit_still_offer_next_page(monkeypatch):
    """Same trailing-PRs-only shape, but with a next page confirmed by the
    Link header: the page-boundary continuation is lossless (no real issue
    was left behind on this page), so next_page must be offered."""
    page = [
        {"number": 1, "title": "issue 1", "labels": []},
        {
            "number": 2,
            "title": "pr 2",
            "labels": [],
            "pull_request": {"url": "https://api.github.com/x"},
        },
    ]
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data=page,
                headers={"Link": '<https://api.github.com/x?page=2>; rel="next"'},
            )
        ),
    )

    result = json.loads(github.github_list_issues("octocat/Hello-World", limit=1))

    assert result["status"] == "success"
    assert len(result["issues"]) == 1
    assert result["truncated"] is True
    assert result["next_page"] == 2


def test_list_issues_full_final_page_without_next_link_is_not_truncated(monkeypatch):
    """A page that happens to come back exactly MAX_PER_PAGE long is not
    itself proof more pages exist -- only the Link header's absent "next"
    rel proves this genuinely was the last page. Regression test for the
    previous length-based heuristic falsely marking this truncated."""
    page = [
        {"number": i, "title": f"issue {i}", "labels": []}
        for i in range(1, github.MAX_PER_PAGE + 1)
    ]
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(return_value=MockResponse(json_data=page)),  # no Link header
    )

    result = json.loads(
        github.github_list_issues("octocat/Hello-World", limit=github.MAX_PER_PAGE)
    )

    assert result["status"] == "success"
    assert len(result["issues"]) == github.MAX_PER_PAGE
    assert result["truncated"] is False
    assert result["next_page"] is None


def test_list_issues_stops_at_max_pages_and_reports_truncated(monkeypatch):
    """When every page is entirely pull requests and the Link header still
    reports more pages exist, the outer loop must still terminate at
    MAX_ISSUE_PAGES (not loop forever) and report truncated with the page
    to continue from."""
    pr_only_page = [
        {
            "number": i,
            "title": f"pr {i}",
            "labels": [],
            "pull_request": {"url": "https://api.github.com/x"},
        }
        for i in range(1, github.MAX_PER_PAGE + 1)
    ]
    mock_request = Mock(
        return_value=MockResponse(
            json_data=pr_only_page,
            headers={"Link": '<https://api.github.com/x?page=2>; rel="next"'},
        )
    )
    monkeypatch.setattr(github.requests, "request", mock_request)

    result = json.loads(github.github_list_issues("octocat/Hello-World", limit=5))

    assert result["status"] == "success"
    assert result["issues"] == []
    assert result["truncated"] is True
    assert result["next_page"] == github.MAX_ISSUE_PAGES + 1


def test_list_issues_stops_when_aggregate_time_budget_is_exceeded(monkeypatch):
    """MAX_ISSUE_PAGES only bounds request COUNT -- a slow or PR-heavy repo
    could otherwise hold the call open for MAX_ISSUE_PAGES *
    DEFAULT_TIMEOUT_SECONDS (up to 5 minutes). The aggregate wall-clock
    budget must cut it short with a resumable partial result instead."""
    first_page = [
        {
            "number": i,
            "title": f"pr {i}",
            "labels": [],
            "pull_request": {"url": "https://api.github.com/x"},
        }
        for i in range(1, github.MAX_PER_PAGE + 1)
    ]
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data=first_page,
                headers={"Link": '<https://api.github.com/x?page=2>; rel="next"'},
            )
        ),
    )
    # First call computes the deadline; the second (before page 2) finds the
    # budget already exhausted.
    clock = iter([0.0, github.MAX_ISSUE_LIST_SECONDS + 1])
    monkeypatch.setattr(github.time, "monotonic", lambda: next(clock))

    result = json.loads(github.github_list_issues("octocat/Hello-World", limit=5))

    assert result["status"] == "success"
    assert result["issues"] == []
    assert result["truncated"] is True
    assert result["next_page"] == 2
    assert result["next_skip"] == 0


def test_list_issues_starts_from_specified_page(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data=[{"number": 1, "title": "issue", "labels": []}]
        )
    )
    monkeypatch.setattr(github.requests, "request", mock_request)

    github.github_list_issues("octocat/Hello-World", page=3)

    assert mock_request.call_args.kwargs["params"]["page"] == 3


def test_list_issues_returns_next_page_when_more_full_pages_remain(monkeypatch):
    page = [
        {"number": i, "title": f"issue {i}", "labels": []}
        for i in range(1, github.MAX_PER_PAGE + 1)
    ]
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data=page,
                headers={"Link": '<https://api.github.com/x?page=6>; rel="next"'},
            )
        ),
    )

    result = json.loads(
        github.github_list_issues(
            "octocat/Hello-World", limit=github.MAX_PER_PAGE, page=5
        )
    )

    assert result["status"] == "success"
    assert result["truncated"] is True
    assert result["next_page"] == 6


def test_list_issues_preserves_partial_results_on_mid_pagination_failure(monkeypatch):
    """A rate limit (or any transient error) on page 2+ must not discard the
    issues already collected from page 1, matching slack.py's channel
    listing precedent for the same failure mode."""
    # A full (100-item), half-PR page whose Link header reports a next page
    # -- the requested limit (clamped to 100) isn't satisfied by the 50 real
    # issues it yields, so the loop must actually attempt page 2 (which then
    # fails) instead of stopping after page 1.
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
            MockResponse(
                json_data=first_page,
                headers={"Link": '<https://api.github.com/x?page=2>; rel="next"'},
            ),
            MockResponse(json_data={"message": "rate limited"}, status_code=429),
        ]
    )
    monkeypatch.setattr(github.requests, "request", mock_request)

    result = json.loads(github.github_list_issues("octocat/Hello-World", limit=100_000))

    assert result["status"] == "success"
    assert len(result["issues"]) == github.MAX_PER_PAGE // 2
    assert result["truncated"] is True
    # The failed page is the continuation point -- page 1's Link header
    # confirmed page 2 exists, so a retry can resume there. Nothing on that
    # page has been consumed yet, so next_skip is 0.
    assert result["next_page"] == 2
    assert result["next_skip"] == 0
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
    assert result["encoding"] == "utf-8"


def test_get_file_contents_returns_raw_base64_for_non_utf8_content(monkeypatch):
    """A binary (non-UTF-8) file must not be silently corrupted into
    replacement characters while still reporting success -- return the
    original base64 with an explicit encoding marker instead."""
    binary_bytes = b"\x89PNG\r\n\x1a\n\x00\x01\xff\xfe"
    encoded = base64.b64encode(binary_bytes).decode()
    monkeypatch.setattr(
        github.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "type": "file",
                    "path": "image.png",
                    "sha": "def456",
                    "size": len(binary_bytes),
                    "encoding": "base64",
                    "content": encoded,
                }
            )
        ),
    )

    result = json.loads(
        github.github_get_file_contents("octocat/Hello-World", "image.png")
    )

    assert result["status"] == "success"
    assert result["encoding"] == "base64"
    assert result["content"] == encoded
    assert "�" not in result["content"]


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
    ],
)
def test_get_file_contents_rejects_malformed_path(path, monkeypatch):
    """Leading/trailing/consecutive slashes and dot-segments must be
    rejected outright, not silently interpolated into a malformed request
    URL. '?'/'#' are deliberately not tested here -- unlike owner/repo,
    those are legitimate filename characters; see the percent-encoding
    test below."""
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


def test_get_file_contents_percent_encodes_question_mark_and_hash_in_path(monkeypatch):
    """Regression test: '?'/'#' in a filename must be percent-encoded and
    the request allowed to proceed, not rejected as if they were injection
    attempts (they were previously indistinguishable from the owner/repo
    validator's stricter rule)."""
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "type": "file",
                "path": "docs/why?.md",
                "encoding": "utf-8",
                "content": "hi",
            }
        )
    )
    monkeypatch.setattr(github.requests, "request", mock_request)

    result = json.loads(
        github.github_get_file_contents("octocat/Hello-World", "docs/why?.md")
    )

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"].endswith("/contents/docs/why%3F.md")


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
