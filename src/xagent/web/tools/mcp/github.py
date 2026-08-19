import base64
import json
import logging
import os
import re
import time
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
# Aggregate wall-clock budget across github_list_issues' whole paginated
# fetch. MAX_ISSUE_PAGES alone bounds request COUNT, not TIME -- a slow or
# PR-heavy repo could otherwise hold one tool call for up to
# MAX_ISSUE_PAGES * DEFAULT_TIMEOUT_SECONDS (5 minutes) before returning.
MAX_ISSUE_LIST_SECONDS = 60

_FORBIDDEN_REPO_CHARS = re.compile(r"[/?#]")


def _encode_path_component(value: str, *, field: str) -> str:
    """Validate and percent-encode one owner/repo path segment.

    Rejects '/', '?', or '#' within the segment (these would change the
    request's path/query/fragment structure regardless of encoding) and a
    bare '.' or '..' segment -- unvalidated, an owner/repo value reaching a
    same-host route via raw string interpolation could otherwise let a
    crafted input traverse to an unintended GitHub API route (e.g. "..") or
    inject a query string (e.g. "owner?x=y"), while still carrying this
    connector's bearer token. '?'/'#' are rejected outright here (rather
    than left to percent-encoding) because a legitimate owner or repo name
    never contains them -- unlike file paths, see _encode_file_path_segment.
    """
    if not value or _FORBIDDEN_REPO_CHARS.search(value) or value in (".", ".."):
        raise ValueError(f"{field} contains characters that are not allowed: {value!r}")
    return quote(value, safe="")


def _encode_file_path_segment(value: str, *, field: str) -> str:
    """Validate and percent-encode one file-path segment (already split on
    '/', so this never sees an actual '/').

    Unlike owner/repo names, a real filename can legitimately contain '?'
    or '#' (e.g. "docs/why?.md", "issue#1.txt") -- these are safe to allow
    here because quote(..., safe="") percent-encodes them to "%3F"/"%23"
    before the segment reaches the URL, so they can't be reinterpreted as
    query/fragment delimiters. Only emptiness and dot-segments (path
    traversal) are rejected.
    """
    if not value or value in (".", ".."):
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


_LINK_REL_PATTERN = re.compile(r'rel="([^"]+)"')


def _link_header_rels(link_header: str | None) -> set[str]:
    """Parse a GitHub `Link` response header into the set of `rel` values
    present, e.g. {"next", "last"} -- the authoritative way to tell whether
    another page exists. Inferring it from a page's item count instead
    (e.g. "this page came back full") falsely flags an exactly-full final
    page as truncated.
    """
    if not link_header:
        return set()
    return set(_LINK_REL_PATTERN.findall(link_header))


def _request_raw(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_data: dict[str, Any] | None = None,
) -> requests.Response:
    """Call the GitHub REST API and return the raw response (status/headers
    included) on success -- callers that only need the parsed JSON body
    should use `_request` instead; this exists for callers that also need
    pagination (`Link`) or rate-limit headers.

    GitHub answers errors with a JSON body carrying "message" and, for
    validation failures (422), an "errors" list -- both are folded into the
    raised message, along with any rate-limit/retry headers present, rather
    than surfaced as a bare HTTP status.
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
        # Rate-limit/retry headers would otherwise be silently dropped,
        # leaving the caller unable to distinguish a rate limit or transient
        # server error from a validation/permission failure.
        retry_after = response.headers.get("Retry-After")
        rate_limit_remaining = response.headers.get("X-RateLimit-Remaining")
        rate_limit_reset = response.headers.get("X-RateLimit-Reset")
        rate_limit_bits = []
        if retry_after:
            rate_limit_bits.append(f"retry_after={retry_after}s")
        if rate_limit_remaining == "0" and rate_limit_reset:
            rate_limit_bits.append(f"rate_limit_reset={rate_limit_reset}")
        if rate_limit_bits:
            message = f"{message} ({', '.join(rate_limit_bits)})"
        raise RuntimeError(message)
    return response


def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_data: dict[str, Any] | None = None,
) -> Any:
    """Call the GitHub REST API. Returns the parsed JSON body (dict or list)."""
    response = _request_raw(method, path, params=params, json_data=json_data)
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
    repo: str,
    state: str = "open",
    labels: str = "",
    limit: int = 30,
    page: int = 1,
    skip: int = 0,
) -> str:
    """
    List issues in a repository. Pull requests are excluded (use
    github_list_pull_requests for those), even though GitHub's underlying
    endpoint returns both -- a PR-heavy page is fetched past (up to
    MAX_ISSUE_PAGES pages from the starting page) so real issues on later
    pages aren't missed.
    repo: "owner/repo".
    state: "open", "closed", or "all" (default "open").
    labels: optional comma-separated label names to filter by.
    limit: max issues to return (default 30, capped at 100).
    page: which GitHub results page to start from (default 1).
    skip: raw items to skip at the start of `page` (default 0) -- pass
    both next_page and next_skip from a truncated response's next_page/
    next_skip fields to resume exactly where it left off, including
    partway through a page.
    """
    try:
        owner, name = _parse_repo(repo)
        max_results = _clamp_limit(limit)
        start_page = max(1, int(page))
        start_skip = max(0, int(skip))
        issues: list[dict[str, Any]] = []
        truncated = False
        next_page: int | None = None
        next_skip = 0
        pages_fetched = 0
        has_next_page = False
        current_page = start_page
        deadline = time.monotonic() + MAX_ISSUE_LIST_SECONDS
        for current_page in range(start_page, start_page + MAX_ISSUE_PAGES):
            if pages_fetched and time.monotonic() >= deadline:
                # The aggregate budget (not just the per-page timeout) is
                # exhausted -- return what was collected so far as a
                # resumable partial result rather than holding this call
                # open for the full MAX_ISSUE_PAGES worth of requests.
                logger.warning(
                    f"GitHub issue pagination hit its {MAX_ISSUE_LIST_SECONDS}s "
                    f"budget for {repo} after {pages_fetched} page(s)"
                )
                return _success(
                    issues=issues[:max_results],
                    truncated=True,
                    next_page=current_page,
                    next_skip=0,
                )
            params: dict[str, Any] = {
                "state": state,
                "per_page": MAX_PER_PAGE,
                "page": current_page,
            }
            if labels:
                params["labels"] = labels
            try:
                response = _request_raw(
                    "GET", f"/repos/{owner}/{name}/issues", params=params
                )
            except Exception as page_exc:
                if not pages_fetched:
                    raise
                # A mid-pagination failure (e.g. a rate limit) must not
                # discard the pages already fetched -- return the partial
                # list with a marker instead, same as slack.py's channel
                # listing. The failed page is a valid continuation point:
                # reaching it at all means the previous page's Link header
                # confirmed it exists, and nothing on it has been consumed
                # yet, so next_skip stays 0 (retry the whole page).
                logger.warning(
                    f"GitHub issue pagination stopped early for {repo}: {page_exc}"
                )
                return _success(
                    issues=issues[:max_results],
                    truncated=True,
                    next_page=current_page,
                    next_skip=0,
                    error=str(page_exc),
                )
            pages_fetched += 1
            raw_page = response.json() if response.content else []
            has_next_page = "next" in _link_header_rels(response.headers.get("Link"))
            if not raw_page:
                break
            # The caller's skip only applies to the very first page of this
            # call (a resumed page); every page fetched after that starts
            # fresh at index 0.
            page_skip = start_skip if current_page == start_page else 0
            hit_limit_mid_page = False
            for index, issue in enumerate(raw_page):
                if index < page_skip:
                    continue
                if "pull_request" in issue:
                    continue
                issues.append(_summarize_issue(issue))
                if len(issues) >= max_results:
                    # Only a real issue left behind on this page counts as a
                    # mid-page cut -- trailing PRs are excluded from the
                    # result anyway, so "items remain on the page" alone
                    # would falsely report truncation (and needlessly
                    # withhold a continuation) when everything left is PRs.
                    hit_limit_mid_page = any(
                        "pull_request" not in later for later in raw_page[index + 1 :]
                    )
                    truncated = hit_limit_mid_page or has_next_page
                    if hit_limit_mid_page:
                        # Resume the SAME page, skipping every raw item
                        # already consumed from it -- GitHub's page cursor
                        # can't do this on its own, but the raw index can.
                        next_page = current_page
                        next_skip = index + 1
                    elif has_next_page:
                        next_page = current_page + 1
                    break
            if len(issues) >= max_results:
                break
            if not has_next_page:
                break  # confirmed last page via the Link header
        else:
            # MAX_ISSUE_PAGES exhausted -- only report truncated/next_page
            # if the last page we saw actually indicated more exist.
            truncated = has_next_page
            if has_next_page:
                next_page = current_page + 1
        return _success(
            issues=issues[:max_results],
            truncated=truncated,
            next_page=next_page,
            next_skip=next_skip,
        )
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
        # validated/encoded individually via _encode_file_path_segment,
        # which (unlike _parse_repo's owner/repo validator) allows
        # legitimate filename characters like '?'/'#' since they're
        # percent-encoded away rather than reaching the URL raw. An empty
        # segment (from a leading/trailing/double slash) is rejected by its
        # own emptiness check.
        encoded_path = (
            "/".join(
                _encode_file_path_segment(segment, field="path")
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
            decoded_bytes = base64.b64decode(raw_content)
            try:
                content = decoded_bytes.decode("utf-8")
                content_encoding = "utf-8"
            except UnicodeDecodeError:
                # A binary (non-UTF-8) file decoded with errors="replace"
                # would silently turn into U+FFFD garbage while still
                # reporting success -- return the original base64 instead
                # so the caller can tell it's binary and decode it properly.
                content = raw_content
                content_encoding = "base64"
        else:
            content = raw_content
            content_encoding = "utf-8"
        return _success(
            type="file",
            path=result.get("path"),
            sha=result.get("sha"),
            size=result.get("size"),
            content=content,
            encoding=content_encoding,
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
