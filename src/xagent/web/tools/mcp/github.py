import base64
import json
import logging
import os
import re
from typing import Any
from urllib.parse import quote

import requests
from mcp.server.fastmcp import FastMCP

from .utils import setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("github-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("github-mcp")

GITHUB_BASE_URL = "https://api.github.com"
DEFAULT_TIMEOUT_SECONDS = 30
MAX_PER_PAGE = 100
# Bounded multi-page fetch for github_list_issues, mirroring slack.py's
# MAX_PAGES convention -- GitHub's issues endpoint mixes in pull requests,
# so a single page of raw items can be mostly/entirely PRs and undercount
# real issues even when more exist on later pages.
MAX_ISSUE_PAGES = 10

_FORBIDDEN_PATH_CHARS = re.compile(r"[/?#]")


def _encode_path_component(value: str, *, field: str) -> str:
    """Validate and percent-encode one URL path segment.

    Rejects '/', '?', or '#' within the segment (these would change the
    request's path/query/fragment structure regardless of encoding) and a
    bare '.' or '..' segment -- unvalidated, an owner/repo or file path
    value reaching a same-host route via raw string interpolation could
    otherwise let a crafted input traverse to an unintended GitHub API
    route (e.g. "..") or inject a query string (e.g. "owner?x=y"), while
    still carrying this connector's bearer token. The remaining value is
    percent-encoded so segments that legitimately contain spaces or other
    reserved characters (common in file paths, unlike owner/repo names)
    still produce a well-formed request.
    """
    if not value or _FORBIDDEN_PATH_CHARS.search(value) or value in (".", ".."):
        raise ValueError(f"{field} contains characters that are not allowed: {value!r}")
    return quote(value, safe="")


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _headers() -> dict[str, str]:
    access_token = os.environ.get("GITHUB_ACCESS_TOKEN")
    if not access_token:
        raise ValueError("GITHUB_ACCESS_TOKEN environment variable is missing")
    return {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _parse_repo(repo: str) -> tuple[str, str]:
    """Split a "owner/repo" full name into its two, percent-encoded parts.

    Rejects a malformed name (extra/leading/trailing slashes, e.g.
    "owner//repo" or "owner/repo/extra") outright rather than silently
    repairing it — .strip("/") + .partition("/") previously accepted those
    and mangled the extra segment into `name`, which then reached the
    GitHub API as a subtly wrong path instead of a caught bug. Each part is
    further validated/encoded by _encode_path_component, which additionally
    rejects "?"/"#"/dot-segment values that partition() alone would let
    through (e.g. "owner?x=y/repo" has exactly one "/").
    """
    value = repo.strip()
    if value.count("/") != 1:
        raise ValueError(f'repo must be in "owner/repo" format, got: {repo!r}')
    owner, name = value.split("/")
    if not owner or not name:
        raise ValueError(f'repo must be in "owner/repo" format, got: {repo!r}')
    return (
        _encode_path_component(owner, field="repo owner"),
        _encode_path_component(name, field="repo name"),
    )


def _clamp_limit(limit: int, *, default: int = 30) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = default
    return max(1, min(value, MAX_PER_PAGE))


def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_data: dict[str, Any] | None = None,
) -> Any:
    """Call the GitHub REST API. Returns the parsed JSON body (dict or list).

    GitHub answers errors with a JSON body carrying "message" and, for
    validation failures (422), an "errors" list -- both are folded into the
    raised message rather than surfaced as a bare HTTP status.
    """
    response = requests.request(
        method=method,
        url=f"{GITHUB_BASE_URL}{path}",
        headers=_headers(),
        params=params,
        json=json_data,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    if response.status_code >= 400:
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        message = (
            payload.get("message") if isinstance(payload, dict) else None
        ) or f"GitHub API error (status {response.status_code})"
        errors = payload.get("errors") if isinstance(payload, dict) else None
        if errors:
            detail = "; ".join(
                str(item.get("message") or item)
                if isinstance(item, dict)
                else str(item)
                for item in errors
            )
            message = f"{message}: {detail}"
        raise RuntimeError(message)
    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


def _summarize_repo(repo: dict[str, Any]) -> dict[str, Any]:
    return {
        "full_name": repo.get("full_name"),
        "description": repo.get("description"),
        "private": repo.get("private"),
        "default_branch": repo.get("default_branch"),
        "stargazers_count": repo.get("stargazers_count"),
        "open_issues_count": repo.get("open_issues_count"),
        "html_url": repo.get("html_url"),
        "language": repo.get("language"),
        "updated_at": repo.get("updated_at"),
    }


def _summarize_issue(issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "number": issue.get("number"),
        "title": issue.get("title"),
        "state": issue.get("state"),
        "body": issue.get("body"),
        "labels": [
            label.get("name") if isinstance(label, dict) else label
            for label in issue.get("labels") or []
        ],
        "user": (issue.get("user") or {}).get("login"),
        "comments": issue.get("comments"),
        "html_url": issue.get("html_url"),
        "created_at": issue.get("created_at"),
        "updated_at": issue.get("updated_at"),
        # GitHub returns pull requests through the issues endpoints too;
        # this key's presence is the documented way to tell them apart.
        "is_pull_request": "pull_request" in issue,
    }


def _summarize_pull_request(pr: dict[str, Any]) -> dict[str, Any]:
    return {
        "number": pr.get("number"),
        "title": pr.get("title"),
        "state": pr.get("state"),
        "body": pr.get("body"),
        "user": (pr.get("user") or {}).get("login"),
        "head": (pr.get("head") or {}).get("ref"),
        "base": (pr.get("base") or {}).get("ref"),
        "draft": pr.get("draft"),
        "merged": pr.get("merged"),
        "mergeable": pr.get("mergeable"),
        "html_url": pr.get("html_url"),
        "created_at": pr.get("created_at"),
        "updated_at": pr.get("updated_at"),
    }


@mcp.tool()
def github_get_current_user() -> str:
    """
    Get the profile of the GitHub account this connector is authenticated as
    (login, name, email, company, etc.). Use this for "my account" /
    "who am I" requests instead of asking the user for their username.
    """
    try:
        result = _request("GET", "/user")
        return _success(
            user={
                "login": result.get("login"),
                "id": result.get("id"),
                "name": result.get("name"),
                "email": result.get("email"),
                "company": result.get("company"),
                "bio": result.get("bio"),
                "public_repos": result.get("public_repos"),
                "followers": result.get("followers"),
                "following": result.get("following"),
                "html_url": result.get("html_url"),
            }
        )
    except Exception as e:
        logger.error(f"Error fetching authenticated GitHub user: {e}")
        return _error(str(e))


@mcp.tool()
def github_search_repositories(query: str, limit: int = 20) -> str:
    """
    Search GitHub repositories (name, description, topics, etc.).
    query: a GitHub search-syntax query, e.g. "xagent language:python" or
    "org:openai stars:>100".
    limit: max repositories to return (default 20, capped at 100).
    """
    try:
        result = _request(
            "GET",
            "/search/repositories",
            params={"q": query, "per_page": _clamp_limit(limit)},
        )
        repos = [_summarize_repo(repo) for repo in result.get("items") or []]
        return _success(
            repositories=repos, total_count=result.get("total_count", len(repos))
        )
    except Exception as e:
        logger.error(f"Error searching GitHub repositories for {query!r}: {e}")
        return _error(str(e))


@mcp.tool()
def github_get_repository(repo: str) -> str:
    """
    Get metadata for a repository.
    repo: full repository name in "owner/repo" format (e.g. "octocat/Hello-World").
    """
    try:
        owner, name = _parse_repo(repo)
        result = _request("GET", f"/repos/{owner}/{name}")
        return _success(repository=_summarize_repo(result))
    except Exception as e:
        logger.error(f"Error fetching GitHub repository {repo}: {e}")
        return _error(str(e))


@mcp.tool()
def github_list_issues(
    repo: str, state: str = "open", labels: str = "", limit: int = 30
) -> str:
    """
    List issues in a repository. Pull requests are excluded (use
    github_list_pull_requests for those), even though GitHub's underlying
    endpoint returns both -- a PR-heavy page is fetched past (up to
    MAX_ISSUE_PAGES pages) so real issues on later pages aren't missed.
    repo: "owner/repo".
    state: "open", "closed", or "all" (default "open").
    labels: optional comma-separated label names to filter by.
    limit: max issues to return (default 30, capped at 100).
    """
    try:
        owner, name = _parse_repo(repo)
        max_results = _clamp_limit(limit)
        issues: list[dict[str, Any]] = []
        truncated = False
        pages_fetched = 0
        for page in range(1, MAX_ISSUE_PAGES + 1):
            params: dict[str, Any] = {
                "state": state,
                "per_page": MAX_PER_PAGE,
                "page": page,
            }
            if labels:
                params["labels"] = labels
            try:
                raw_page = _request(
                    "GET", f"/repos/{owner}/{name}/issues", params=params
                )
            except Exception as page_exc:
                if not pages_fetched:
                    raise
                # A mid-pagination failure (e.g. a rate limit) must not
                # discard the pages already fetched -- return the partial
                # list with a marker instead, same as slack.py's channel
                # listing.
                logger.warning(
                    f"GitHub issue pagination stopped early for {repo}: {page_exc}"
                )
                return _success(
                    issues=issues[:max_results], truncated=True, error=str(page_exc)
                )
            pages_fetched += 1
            if not raw_page:
                break
            for index, issue in enumerate(raw_page):
                if "pull_request" in issue:
                    continue
                issues.append(_summarize_issue(issue))
                if len(issues) >= max_results:
                    # More raw items remain on this page, or GitHub filled
                    # the page (implying at least one more page may exist) --
                    # either means real issues could remain unfetched.
                    truncated = (index + 1 < len(raw_page)) or (
                        len(raw_page) == MAX_PER_PAGE
                    )
                    break
            if len(issues) >= max_results:
                break
            if len(raw_page) < MAX_PER_PAGE:
                break  # GitHub's own last page
        else:
            # MAX_ISSUE_PAGES exhausted without hitting a short (last) page.
            truncated = True
        return _success(issues=issues[:max_results], truncated=truncated)
    except Exception as e:
        logger.error(f"Error listing GitHub issues for {repo}: {e}")
        return _error(str(e))


@mcp.tool()
def github_get_issue(repo: str, issue_number: int) -> str:
    """
    Get a single issue (or pull request, since PRs share the issue number
    space) by number.
    repo: "owner/repo".
    """
    try:
        owner, name = _parse_repo(repo)
        result = _request("GET", f"/repos/{owner}/{name}/issues/{issue_number}")
        return _success(issue=_summarize_issue(result))
    except Exception as e:
        logger.error(f"Error fetching GitHub issue {repo}#{issue_number}: {e}")
        return _error(str(e))


@mcp.tool()
def github_create_issue(repo: str, title: str, body: str = "", labels: str = "") -> str:
    """
    Create a new issue in a repository.
    repo: "owner/repo".
    title: issue title.
    body: issue description (Markdown supported).
    labels: optional comma-separated label names to apply.
    """
    try:
        owner, name = _parse_repo(repo)
        json_data: dict[str, Any] = {"title": title}
        if body:
            json_data["body"] = body
        if labels:
            json_data["labels"] = [
                label.strip() for label in labels.split(",") if label.strip()
            ]
        result = _request("POST", f"/repos/{owner}/{name}/issues", json_data=json_data)
        return _success(issue=_summarize_issue(result))
    except Exception as e:
        logger.error(f"Error creating GitHub issue in {repo}: {e}")
        return _error(str(e))


@mcp.tool()
def github_comment_on_issue(repo: str, issue_number: int, body: str) -> str:
    """
    Add a comment to an issue or pull request.
    repo: "owner/repo".
    issue_number: the issue or pull request number.
    body: the comment body (Markdown supported).
    """
    try:
        owner, name = _parse_repo(repo)
        result = _request(
            "POST",
            f"/repos/{owner}/{name}/issues/{issue_number}/comments",
            json_data={"body": body},
        )
        return _success(comment_id=result.get("id"), html_url=result.get("html_url"))
    except Exception as e:
        logger.error(f"Error commenting on GitHub issue {repo}#{issue_number}: {e}")
        return _error(str(e))


@mcp.tool()
def github_list_pull_requests(repo: str, state: str = "open", limit: int = 30) -> str:
    """
    List pull requests in a repository.
    repo: "owner/repo".
    state: "open", "closed", or "all" (default "open").
    limit: max pull requests to return (default 30, capped at 100).
    """
    try:
        owner, name = _parse_repo(repo)
        result = _request(
            "GET",
            f"/repos/{owner}/{name}/pulls",
            params={"state": state, "per_page": _clamp_limit(limit)},
        )
        return _success(pull_requests=[_summarize_pull_request(pr) for pr in result])
    except Exception as e:
        logger.error(f"Error listing GitHub pull requests for {repo}: {e}")
        return _error(str(e))


@mcp.tool()
def github_get_pull_request(repo: str, pull_number: int) -> str:
    """
    Get a single pull request by number.
    repo: "owner/repo".
    """
    try:
        owner, name = _parse_repo(repo)
        result = _request("GET", f"/repos/{owner}/{name}/pulls/{pull_number}")
        return _success(pull_request=_summarize_pull_request(result))
    except Exception as e:
        logger.error(f"Error fetching GitHub pull request {repo}#{pull_number}: {e}")
        return _error(str(e))


@mcp.tool()
def github_create_pull_request(
    repo: str, title: str, head: str, base: str, body: str = ""
) -> str:
    """
    Create a new pull request.
    repo: "owner/repo".
    title: pull request title.
    head: the branch containing the changes (e.g. "feature-branch", or
    "other-owner:branch" for a cross-fork PR).
    base: the branch to merge into (e.g. "main").
    body: pull request description (Markdown supported).
    """
    try:
        owner, name = _parse_repo(repo)
        json_data: dict[str, Any] = {"title": title, "head": head, "base": base}
        if body:
            json_data["body"] = body
        result = _request("POST", f"/repos/{owner}/{name}/pulls", json_data=json_data)
        return _success(pull_request=_summarize_pull_request(result))
    except Exception as e:
        logger.error(f"Error creating GitHub pull request in {repo}: {e}")
        return _error(str(e))


@mcp.tool()
def github_get_file_contents(repo: str, path: str, ref: str = "") -> str:
    """
    Read a file's contents, or list a directory, at a path in a repository.
    repo: "owner/repo".
    path: file or directory path relative to the repo root (e.g. "src/main.py",
    or "" for the repo root).
    ref: optional branch, tag, or commit SHA (defaults to the repo's default branch).
    """
    try:
        owner, name = _parse_repo(repo)
        # path is interpolated directly into the request URL below (unlike
        # github_list_commits' path, sent as a query param that requests
        # percent-encodes regardless of content) -- each segment is
        # validated/encoded individually, same as _parse_repo's owner/repo
        # handling above. An empty segment (from a leading/trailing/double
        # slash) is rejected by _encode_path_component's own emptiness check.
        encoded_path = (
            "/".join(
                _encode_path_component(segment, field="path")
                for segment in path.split("/")
            )
            if path
            else ""
        )
        params: dict[str, Any] = {"ref": ref} if ref else {}
        result = _request(
            "GET", f"/repos/{owner}/{name}/contents/{encoded_path}", params=params
        )
        if isinstance(result, list):
            entries = [
                {
                    "name": entry.get("name"),
                    "path": entry.get("path"),
                    "type": entry.get("type"),
                }
                for entry in result
            ]
            # GitHub's Contents API returns at most 1000 entries for a
            # directory with no continuation token in the response body --
            # flag it so a directory at exactly that cap isn't mistaken for
            # a complete listing (use the Git Trees API for larger ones).
            return _success(
                type="directory", entries=entries, truncated=len(entries) >= 1000
            )
        if result.get("type") != "file":
            return _error(f"Path '{path}' is not a file: type={result.get('type')}")
        encoding = result.get("encoding")
        if encoding == "none":
            # GitHub omits file content (encoding: "none") for files above
            # the Contents API's ~1MB size limit -- returning "" here would
            # silently report an empty read as if the file were genuinely
            # empty, rather than surfacing the real "too large" condition.
            return _error(
                f"File '{path}' is too large for the Contents API (no content "
                "returned) -- use github_list_commits or clone the repository "
                "to read it"
            )
        raw_content = result.get("content") or ""
        if encoding == "base64":
            content = base64.b64decode(raw_content).decode("utf-8", errors="replace")
        else:
            content = raw_content
        return _success(
            type="file",
            path=result.get("path"),
            sha=result.get("sha"),
            size=result.get("size"),
            content=content,
        )
    except Exception as e:
        logger.error(f"Error fetching GitHub file contents {repo}:{path}: {e}")
        return _error(str(e))


@mcp.tool()
def github_list_commits(repo: str, path: str = "", limit: int = 30) -> str:
    """
    List recent commits in a repository.
    repo: "owner/repo".
    path: optional file or directory path to restrict history to.
    limit: max commits to return (default 30, capped at 100).
    """
    try:
        owner, name = _parse_repo(repo)
        params: dict[str, Any] = {"per_page": _clamp_limit(limit)}
        if path:
            params["path"] = path
        result = _request("GET", f"/repos/{owner}/{name}/commits", params=params)
        commits = [
            {
                "sha": commit.get("sha"),
                "message": (commit.get("commit") or {}).get("message"),
                "author": ((commit.get("commit") or {}).get("author") or {}).get(
                    "name"
                ),
                "date": ((commit.get("commit") or {}).get("author") or {}).get("date"),
                "html_url": commit.get("html_url"),
            }
            for commit in result
        ]
        return _success(commits=commits)
    except Exception as e:
        logger.error(f"Error listing GitHub commits for {repo}: {e}")
        return _error(str(e))


@mcp.tool()
def github_search_code(query: str, limit: int = 20) -> str:
    """
    Search code across GitHub (or, when scoped with "repo:owner/repo" or
    "org:name" in the query, within a specific repository or organization).
    query: a GitHub code-search query, e.g. "repo:octocat/Hello-World def parse".
    limit: max results to return (default 20, capped at 100).
    """
    try:
        result = _request(
            "GET",
            "/search/code",
            params={"q": query, "per_page": _clamp_limit(limit)},
        )
        items = [
            {
                "name": item.get("name"),
                "path": item.get("path"),
                "repository": (item.get("repository") or {}).get("full_name"),
                "html_url": item.get("html_url"),
            }
            for item in result.get("items") or []
        ]
        return _success(items=items, total_count=result.get("total_count", len(items)))
    except Exception as e:
        logger.error(f"Error searching GitHub code for {query!r}: {e}")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
