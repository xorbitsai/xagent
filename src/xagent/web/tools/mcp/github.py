import base64
import json
import logging
import os
from typing import Any

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
    """Split a "owner/repo" full name into its two parts."""
    value = repo.strip().strip("/")
    owner, _, name = value.partition("/")
    if not owner or not name:
        raise ValueError(f'repo must be in "owner/repo" format, got: {repo!r}')
    return owner, name


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
    endpoint returns both.
    repo: "owner/repo".
    state: "open", "closed", or "all" (default "open").
    labels: optional comma-separated label names to filter by.
    limit: max issues to return (default 30, capped at 100).
    """
    try:
        owner, name = _parse_repo(repo)
        params: dict[str, Any] = {
            "state": state,
            "per_page": _clamp_limit(limit),
        }
        if labels:
            params["labels"] = labels
        result = _request("GET", f"/repos/{owner}/{name}/issues", params=params)
        issues = [
            _summarize_issue(issue) for issue in result if "pull_request" not in issue
        ]
        return _success(issues=issues)
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
        params: dict[str, Any] = {"ref": ref} if ref else {}
        result = _request(
            "GET", f"/repos/{owner}/{name}/contents/{path}", params=params
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
            return _success(type="directory", entries=entries)
        if result.get("type") != "file":
            return _error(f"Path '{path}' is not a file: type={result.get('type')}")
        encoding = result.get("encoding")
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
