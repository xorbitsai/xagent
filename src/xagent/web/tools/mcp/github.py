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
# Default result count for the two search tools -- one constant shared by
# their signatures and their _clamp_limit fallbacks so the values can't
# silently drift apart (the list tools use _clamp_limit's own default 30).
SEARCH_DEFAULT_LIMIT = 20
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
# Bounded rate-limit retry for idempotent reads -- same wait-and-retry shape
# as jira.py/slack.py/intercom.py's own copies (a single wait on a small
# Retry-After), but not an exact mirror: this one is GET-only, opt-out via
# allow_retry, also fires on GitHub's 403-shaped rate limits (not just 429),
# and folds rate-limit headers into the raised message, none of which the
# siblings do. Never retries writes (POST could double-apply a mutation if
# the original request actually reached the server before being rate-limited).
MAX_RETRY_AFTER_SECONDS = 30

# Module-local bindings, rather than calling time.monotonic()/time.sleep()/
# time.time() directly at each call site, so tests can monkeypatch just this
# connector's clock (monkeypatch.setattr(github, "_monotonic"/"_sleep"/
# "_wall_clock", ...)) instead of the singleton stdlib time module, which
# would otherwise leak the fake clock/no-op sleep into unrelated code running
# in the same process for the duration of the test. _wall_clock (epoch
# seconds) is distinct from _monotonic (elapsed seconds, no fixed epoch): it
# exists only to diff against GitHub's X-RateLimit-Reset, itself an epoch
# timestamp.
_monotonic = time.monotonic
_sleep = time.sleep
_wall_clock = time.time

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


def _decode_base64_content(raw_content: str) -> bytes:
    """Strict base64 decode of a Contents API `content` field.

    GitHub wraps the base64 body with a newline every ~60 characters, which
    `validate=True`'s non-alphabet check would otherwise reject as invalid --
    whitespace is stripped first so only genuinely malformed input (bad
    alphabet or padding, e.g. "!!!!") raises, rather than permissively
    decoding to truncated/wrong bytes the way the default validate=False
    does.
    """
    normalized = re.sub(r"\s", "", raw_content)
    return base64.b64decode(normalized, validate=True)


def _require_object_items(items: Any, *, context: str) -> None:
    """Raise if the response isn't a list of objects.

    Every list-returning endpoint here iterates its items with an
    unguarded `.get()` (directly or via a `_summarize_*` helper) --
    without the item check, a non-object entry would surface as an
    unhelpful `'str' object has no attribute 'get'` (or, in
    github_list_issues' pagination, escape the per-page handler entirely
    and discard results already collected) instead of identifying what
    GitHub actually returned. The top-level check matters separately:
    `github_list_pull_requests`/`github_list_commits` fall back to `[]`
    only when the response body is empty, not when it parses to a
    non-list value (e.g. `{}`) -- `all(...)` over an empty dict's
    (zero) keys is vacuously true, so without this check that case would
    silently report a successful empty result instead of the malformed
    response it actually was.
    """
    if not isinstance(items, list):
        # "value", not "body": two call sites pass the search envelope's
        # extracted `items` field rather than the response body itself, so
        # naming the body here would misdiagnose which layer is malformed.
        raise ValueError(
            f"GitHub returned a non-list value in {context} ({type(items).__name__})"
        )
    if not all(isinstance(item, dict) for item in items):
        raise ValueError(f"GitHub returned a non-object item in {context}")


def _require_object(value: Any, *, context: str) -> None:
    """Raise if a single-object response isn't an object.

    The single-object GET/create tools (github_get_current_user,
    github_get_repository, github_get_issue, github_create_issue,
    github_comment_on_issue, github_get_pull_request,
    github_create_pull_request) call `_request()` and immediately `.get()`
    or pass the result to a `_summarize_*` helper -- an unexpected `null`
    or list body (e.g. a malformed or proxy-mangled response) would
    otherwise surface as an unhelpful `'NoneType'`/`'list' object has no
    attribute 'get'` instead of identifying what GitHub actually
    returned, the same class of gap `_require_object_items` closes for
    the list-returning tools. github_search_repositories/github_search_code
    also call this, on the search envelope itself, ahead of the separate
    `_require_object_items` check on its `items` field.
    """
    if not isinstance(value, dict):
        raise ValueError(
            f"GitHub returned a non-object value in {context} ({type(value).__name__})"
        )


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _partial(**payload: Any) -> str:
    """Pagination stopped before collecting everything the caller asked
    for, for a reason beyond the tool's own item/page-count limit (a
    request fault, a malformed item, or the aggregate deadline): the
    collected items and resume cursor are preserved (same payload shape
    as `_success`, plus `error` when there is one), but `status: "partial"`
    keeps a caller that branches only on `status` from mistaking the
    result for a clean, complete page. Only item_limit/more_pages/
    max_pages -- hitting the requested count or GitHub's own page cap,
    with nothing wrong -- stay `_success`.
    """
    return json.dumps({"status": "partial", **payload}, ensure_ascii=False)


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


def _validate_positive_number(value: int, *, field: str) -> int:
    """Reject a non-positive issue/pull-request number before it reaches an
    authenticated route -- FastMCP's plain `int` schema doesn't enforce a
    lower bound, and a direct Python call (as in tests) bypasses FastMCP
    validation entirely, so 0/negative values would otherwise reach
    `/issues/0` or `/pulls/-1` unchecked.
    """
    try:
        number = int(value)
    except (TypeError, ValueError):
        # int(None) raises a bare, unnamed TypeError -- every other
        # rejection in this function names the field, so this must too
        # rather than leaking a raw builtin exception.
        raise ValueError(
            f"{field} must be a positive integer, got: {value!r}"
        ) from None
    if number < 1:
        raise ValueError(f"{field} must be a positive integer, got: {value!r}")
    return number


def _require_nonblank(value: str, *, field: str) -> str:
    """Reject an empty/whitespace-only required string before it reaches
    an authenticated route -- GitHub answers these (e.g. an empty issue
    title or PR head/base) with an opaque 422 rather than a message
    identifying which field was blank.
    """
    if not value or not value.strip():
        raise ValueError(f"{field} must not be empty")
    return value


_ISSUE_OR_PR_STATES = frozenset({"open", "closed", "all"})


def _validate_state(state: str, *, field: str = "state") -> str:
    if state not in _ISSUE_OR_PR_STATES:
        raise ValueError(
            f"{field} must be one of {sorted(_ISSUE_OR_PR_STATES)!r}, got: {state!r}"
        )
    return state


# Splits a Link header into its comma-separated entries structurally --
# each entry is a "<uri>" followed by zero or more ";param" pieces up to
# the next entry's comma. Matching the params ([^,]*) only AFTER the closing
# ">" means the uri itself (group omitted, matched but not captured) is
# never exposed to _LINK_REL_PATTERN below, so "rel=" text inside a target
# URI (e.g. "<https://x/?rel=foo>") can't be mistaken for the parameter.
_LINK_ENTRY_PATTERN = re.compile(r"<[^>]*>((?:\s*;[^,]*)*)")
_LINK_REL_PATTERN = re.compile(r'rel\s*=\s*(?:"([^"]*)"|([^;,\s]+))', re.IGNORECASE)


def _link_header_rels(link_header: str | None) -> set[str]:
    """Parse a GitHub `Link` response header into the set of `rel` values
    present, e.g. {"next", "last"} -- the authoritative way to tell whether
    another page exists. Inferring it from a page's item count instead
    (e.g. "this page came back full") falsely flags an exactly-full final
    page as truncated.

    GitHub always quotes the rel value (rel="next"), but RFC 8288 also
    permits an unquoted token, and a quoted value may itself be a
    whitespace-separated list of multiple relation types (rel="next last")
    -- both forms, and multi-value rel, are handled here rather than only
    a single quoted token, so a spec-compliant but differently-formatted
    response (e.g. from a proxy) isn't silently treated as having no rels
    at all, or as having only one. Rel values are case-insensitive per the
    same RFC, so they're lowercased to match callers' `"next" in ...` checks.
    """
    if not link_header:
        return set()
    rels: set[str] = set()
    for entry_match in _LINK_ENTRY_PATTERN.finditer(link_header):
        rel_match = _LINK_REL_PATTERN.search(entry_match.group(1))
        if not rel_match:
            continue
        quoted, unquoted = rel_match.groups()
        value = quoted if quoted is not None else unquoted
        rels.update(token.lower() for token in value.split())
    return rels


def _request_raw(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_data: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    allow_retry: bool = True,
) -> requests.Response:
    """Call the GitHub REST API and return the raw response (status/headers
    included) on success -- callers that only need the parsed JSON body
    should use `_request` instead; this exists for callers that also need
    pagination (`Link`) or rate-limit headers.

    GitHub answers errors with a JSON body carrying "message" and, for
    validation failures (422), an "errors" list -- both are folded into the
    raised message, along with any rate-limit/retry headers present, rather
    than surfaced as a bare HTTP status.

    allow_retry=False opts out of the bounded rate-limit retry below (429,
    or a 403 carrying a rate-limit header -- see is_rate_limited_403). A
    caller running under its own wall-clock budget (github_list_issues) must
    pass False: the retry's sleep + second attempt happen inside this call,
    so they are invisible to any deadline the caller computed its `timeout`
    from -- a rate limit near that deadline would otherwise hold the call up
    to `timeout + MAX_RETRY_AFTER_SECONDS + timeout` and blow the budget the
    caller's timeout-capping exists to enforce.
    """
    for attempt in (0, 1):
        response = requests.request(
            method=method,
            url=f"{GITHUB_BASE_URL}{path}",
            headers=_headers(),
            params=params,
            json=json_data,
            timeout=timeout,
        )
        # GitHub answers both its primary rate limit (quota exhausted) and
        # secondary/abuse-detection limit with 403, not 429 -- a plain
        # `status_code == 429` check never catches the common real-world
        # rate-limit case. Only treat a 403 as a rate limit, never a retry
        # trigger, when it carries a rate-limit-specific header; a genuine
        # permission-denied 403 carries neither and must keep failing fast
        # rather than being retried into the same denial.
        is_rate_limited_403 = response.status_code == 403 and (
            response.headers.get("Retry-After") is not None
            or response.headers.get("X-RateLimit-Remaining") == "0"
        )
        # Retry once, only for reads: a POST/PATCH/DELETE rejected with a
        # rate limit never reached GitHub's mutation logic, but replaying it
        # anyway risks a double-apply if that assumption is ever wrong for
        # some endpoint -- reads have no such risk.
        if (
            allow_retry
            and (response.status_code == 429 or is_rate_limited_403)
            and attempt == 0
            and method.upper() == "GET"
        ):
            # retry_after_seconds is None (rather than 0) whenever there is
            # no usable wait signal at all -- Retry-After: "0" and an
            # already-past X-RateLimit-Reset are both legitimate "retry
            # right now" signals, not "no information," and must not be
            # conflated with the missing-header case below them.
            retry_after_header = response.headers.get("Retry-After")
            retry_after_seconds = None
            if retry_after_header is not None:
                try:
                    retry_after_seconds = int(retry_after_header)
                except ValueError:
                    retry_after_seconds = None
            if retry_after_seconds is None and is_rate_limited_403:
                # No (usable) Retry-After -- GitHub's primary-limit 403
                # usually omits it, using X-RateLimit-Reset (an epoch
                # timestamp) instead: derive a wait from that rather than
                # skipping the retry outright. Gated on is_rate_limited_403
                # (not just "status is 429 or 403") so this doesn't also
                # silently extend a bare 429's existing, already-tested
                # Retry-After-only behavior to a header 429 was never
                # documented or tested to use here. The
                # MAX_RETRY_AFTER_SECONDS bound below still applies, so a
                # reset that's minutes/hours away (the common case for a
                # fully exhausted primary quota) falls through to the normal
                # error rather than sleeping for it.
                reset_header = response.headers.get("X-RateLimit-Reset")
                if reset_header:
                    try:
                        retry_after_seconds = max(
                            0, int(reset_header) - int(_wall_clock())
                        )
                    except ValueError:
                        retry_after_seconds = None
            if (
                retry_after_seconds is not None
                and 0 <= retry_after_seconds <= MAX_RETRY_AFTER_SECONDS
            ):
                _sleep(retry_after_seconds)
                continue
        break
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
    # The list-PRs endpoint never includes a "merged" key at all (only the
    # single-PR GET does) but does include "merged_at" -- without the
    # fallback, github_list_pull_requests would always report merged: null
    # for genuinely merged PRs instead of using the field it actually has.
    # A payload carrying NEITHER key stays None ("unknown") rather than
    # being reported as a confident false. "mergeable" has no substitute
    # (GitHub only computes it for a single PR); None there means
    # "unknown", which is honest for a list response.
    merged = pr.get("merged")
    if merged is None and "merged_at" in pr:
        merged = pr["merged_at"] is not None
    return {
        "number": pr.get("number"),
        "title": pr.get("title"),
        "state": pr.get("state"),
        "body": pr.get("body"),
        "user": (pr.get("user") or {}).get("login"),
        "head": (pr.get("head") or {}).get("ref"),
        "base": (pr.get("base") or {}).get("ref"),
        "draft": pr.get("draft"),
        "merged": merged,
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
        _require_object(result, context="authenticated user profile")
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
        logger.error(f"Error fetching authenticated GitHub user: {e}", exc_info=True)
        return _error(str(e))


@mcp.tool()
def github_search_repositories(
    query: str, limit: int = SEARCH_DEFAULT_LIMIT, page: int = 1
) -> str:
    """
    Search GitHub repositories (name, description, topics, etc.).
    query: a GitHub search-syntax query, e.g. "xagent language:python" or
    "org:openai stars:>100".
    limit: max repositories to return (default 20, capped at 100).
    page: which GitHub results page to fetch (default 1) -- pass the
    previous response's next_page when truncated is true to continue.
    GitHub's Search API caps combined results at 1000 regardless of page.
    """
    try:
        query = _require_nonblank(query, field="query")
        start_page = max(1, int(page or 1))
        response = _request_raw(
            "GET",
            "/search/repositories",
            params={
                "q": query,
                "per_page": _clamp_limit(limit, default=SEARCH_DEFAULT_LIMIT),
                "page": start_page,
            },
        )
        result = response.json() if response.content else {}
        _require_object(result, context="repository search response")
        raw_items = result.get("items") or []
        _require_object_items(raw_items, context="repository search results")
        repos = [_summarize_repo(repo) for repo in raw_items]
        has_next_page = "next" in _link_header_rels(response.headers.get("Link"))
        return _success(
            repositories=repos,
            total_count=result.get("total_count", len(repos)),
            # GitHub can answer a 200 with incomplete_results=true when its
            # search index times out -- dropping this would let a caller
            # mistake a partial index result for an exhaustive one.
            incomplete_results=bool(result.get("incomplete_results")),
            truncated=has_next_page,
            truncation_reason="more_pages" if has_next_page else None,
            next_page=start_page + 1 if has_next_page else None,
        )
    except Exception as e:
        logger.error(
            f"Error searching GitHub repositories for {query!r}: {e}", exc_info=True
        )
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
        _require_object(result, context=f"repository metadata for '{repo}'")
        return _success(repository=_summarize_repo(result))
    except Exception as e:
        logger.error(f"Error fetching GitHub repository {repo}: {e}", exc_info=True)
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
    next_skip fields to resume from that point, including partway through
    a page. next_skip is a raw index into GitHub's live, created-desc
    result set, not a stable cursor: an issue created or closed/reopened
    (changing its sort position) between calls can still shift the page's
    contents enough to duplicate or skip an item across the resume.

    A response with status "partial" means pagination stopped before
    collecting everything requested -- a request fault or malformed item
    (see its `error` field, when present) or the aggregate time budget
    running out. Either way the returned issues are valid and
    next_page/next_skip resume from where it stopped -- do NOT discard
    them or restart from page 1.
    """
    try:
        owner, name = _parse_repo(repo)
        state = _validate_state(state)
        max_results = _clamp_limit(limit)
        start_page = max(1, int(page or 1))
        start_skip = max(0, int(skip or 0))
        issues: list[dict[str, Any]] = []
        truncated = False
        # Mirrors `truncated` at every exit below so a caller can dispatch
        # on one stable field instead of checking "truncation_reason",
        # "error", or neither depending on which exit was taken.
        truncation_reason: str | None = None
        next_page: int | None = None
        next_skip = 0
        pages_fetched = 0
        has_next_page = False
        current_page = start_page
        deadline = _monotonic() + MAX_ISSUE_LIST_SECONDS
        for current_page in range(start_page, start_page + MAX_ISSUE_PAGES):
            if pages_fetched and _monotonic() >= deadline:
                # The aggregate budget (not just the per-page timeout) is
                # exhausted -- return what was collected so far as a
                # resumable partial result rather than holding this call
                # open for the full MAX_ISSUE_PAGES worth of requests.
                logger.warning(
                    f"GitHub issue pagination hit its {MAX_ISSUE_LIST_SECONDS}s "
                    f"budget for {repo} after {pages_fetched} page(s)"
                )
                return _partial(
                    issues=issues[:max_results],
                    truncated=True,
                    truncation_reason="deadline",
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
                # Cap this request's own timeout to what's left of the
                # aggregate budget -- otherwise a single in-flight request
                # near the deadline could still run the full
                # DEFAULT_TIMEOUT_SECONDS and blow past it before the
                # pre-request check above ever gets to act on it.
                request_timeout = max(
                    1.0, min(DEFAULT_TIMEOUT_SECONDS, deadline - _monotonic())
                )
                # allow_retry=False: the retry's sleep + second attempt run
                # inside _request_raw, invisible to this deadline -- a 429
                # near the budget's end would hold the call ~60s past it.
                # A mid-pagination 429 already has a better path here: the
                # per-page handler below returns the collected pages as a
                # resumable partial ("request_failed") immediately.
                response = _request_raw(
                    "GET",
                    f"/repos/{owner}/{name}/issues",
                    params=params,
                    timeout=request_timeout,
                    allow_retry=False,
                )
                # Parsing is inside this try too: a malformed 200 JSON body
                # must be treated the same as a request failure -- outside
                # it, response.json() raising would escape to the outer
                # handler and discard every page already collected.
                raw_page = response.json() if response.content else []
                if not isinstance(raw_page, list):
                    # A valid-JSON, non-list body (e.g. an error object
                    # returned with a 2xx status) would otherwise reach the
                    # per-item loop below and raise there -- outside this
                    # try, discarding every page already collected.
                    raise ValueError(
                        "GitHub issues endpoint returned a non-list page "
                        f"({type(raw_page).__name__})"
                    )
            except Exception as page_exc:
                if not pages_fetched:
                    raise
                # A mid-pagination failure (e.g. a rate limit) must not
                # discard the pages already fetched -- return them as a
                # resumable partial. (slack.py preserves partial results
                # for the same fault but still labels them status:"success"
                # with an error field; the "partial" status here is
                # deliberately NOT the same -- see _partial.) The failed
                # page is a valid continuation point: reaching it at all
                # means the previous page's Link header confirmed it
                # exists, and nothing on it has been consumed yet, so
                # next_skip stays 0 (retry the whole page).
                logger.warning(
                    f"GitHub issue pagination stopped early for {repo}: {page_exc}"
                )
                return _partial(
                    issues=issues[:max_results],
                    truncated=True,
                    truncation_reason="request_failed",
                    next_page=current_page,
                    next_skip=0,
                    error=str(page_exc),
                )
            pages_fetched += 1
            has_next_page = "next" in _link_header_rels(response.headers.get("Link"))
            if not raw_page:
                # An empty page normally means GitHub's Link header already
                # said so too (no "next" rel) -- but derive truncated/
                # next_page from has_next_page rather than assuming
                # completion, in case a malformed/unusual response still
                # reports one.
                truncated = has_next_page
                truncation_reason = "more_pages" if has_next_page else None
                if has_next_page:
                    next_page = current_page + 1
                break
            # The caller's skip only applies to the very first page of this
            # call (a resumed page); every page fetched after that starts
            # fresh at index 0.
            page_skip = start_skip if current_page == start_page else 0
            hit_limit_mid_page = False
            bad_item_error: str | None = None
            for index, issue in enumerate(raw_page):
                if index < page_skip:
                    continue
                if not isinstance(issue, dict):
                    # A non-object entry (e.g. `[null]`) would otherwise
                    # raise from `"pull_request" in issue` or
                    # `_summarize_issue()`, outside any try, discarding
                    # every page already collected. Stop short and resume
                    # after this item instead of retrying it forever.
                    bad_item_error = (
                        "GitHub issues endpoint returned a non-object item "
                        f"at page {current_page} index {index}: "
                        f"{type(issue).__name__}"
                    )
                    next_page = current_page
                    next_skip = index + 1
                    break
                if "pull_request" in issue:
                    continue
                issues.append(_summarize_issue(issue))
                if len(issues) >= max_results:
                    # Only a real issue left behind on this page counts as a
                    # mid-page cut -- trailing PRs are excluded from the
                    # result anyway, so "items remain on the page" alone
                    # would falsely report truncation (and needlessly
                    # withhold a continuation) when everything left is PRs.
                    # A non-object trailing item (e.g. `[null]`) can't be
                    # inspected for "pull_request" without raising -- treat
                    # it conservatively as "not a pull request" so it counts
                    # toward a mid-page cut instead of crashing and
                    # discarding every issue already collected on this call.
                    hit_limit_mid_page = any(
                        not isinstance(later, dict) or "pull_request" not in later
                        for later in raw_page[index + 1 :]
                    )
                    truncated = hit_limit_mid_page or has_next_page
                    if hit_limit_mid_page:
                        # Resume the SAME page, skipping every raw item
                        # already consumed from it -- GitHub's page cursor
                        # can't do this on its own, but the raw index can.
                        next_page = current_page
                        next_skip = index + 1
                        truncation_reason = "item_limit"
                    elif has_next_page:
                        next_page = current_page + 1
                        truncation_reason = "more_pages"
                    else:
                        truncation_reason = None
                    break
            if bad_item_error:
                logger.warning(
                    f"GitHub issue pagination stopped early for {repo}: "
                    f"{bad_item_error}"
                )
                return _partial(
                    issues=issues[:max_results],
                    truncated=True,
                    truncation_reason="bad_item",
                    next_page=next_page,
                    next_skip=next_skip,
                    error=bad_item_error,
                )
            if len(issues) >= max_results:
                break
            if not has_next_page:
                break  # confirmed last page via the Link header
        else:
            # MAX_ISSUE_PAGES exhausted -- only report truncated/next_page
            # if the last page we saw actually indicated more exist. This
            # is a more specific reason than "more_pages": it's our own
            # page-count safety cap, not just "there happen to be more".
            truncated = has_next_page
            truncation_reason = "max_pages" if has_next_page else None
            if has_next_page:
                next_page = current_page + 1
        return _success(
            issues=issues[:max_results],
            truncated=truncated,
            truncation_reason=truncation_reason,
            next_page=next_page,
            next_skip=next_skip,
        )
    except Exception as e:
        logger.error(f"Error listing GitHub issues for {repo}: {e}", exc_info=True)
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
        issue_number = _validate_positive_number(issue_number, field="issue_number")
        result = _request("GET", f"/repos/{owner}/{name}/issues/{issue_number}")
        _require_object(result, context=f"issue {repo}#{issue_number}")
        return _success(issue=_summarize_issue(result))
    except Exception as e:
        logger.error(
            f"Error fetching GitHub issue {repo}#{issue_number}: {e}", exc_info=True
        )
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
        title = _require_nonblank(title, field="title")
        json_data: dict[str, Any] = {"title": title}
        if body:
            json_data["body"] = body
        if labels:
            json_data["labels"] = [
                label.strip() for label in labels.split(",") if label.strip()
            ]
        result = _request("POST", f"/repos/{owner}/{name}/issues", json_data=json_data)
        _require_object(result, context=f"created issue in '{repo}'")
        return _success(issue=_summarize_issue(result))
    except Exception as e:
        logger.error(f"Error creating GitHub issue in {repo}: {e}", exc_info=True)
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
        issue_number = _validate_positive_number(issue_number, field="issue_number")
        body = _require_nonblank(body, field="body")
        result = _request(
            "POST",
            f"/repos/{owner}/{name}/issues/{issue_number}/comments",
            json_data={"body": body},
        )
        _require_object(result, context=f"comment on {repo}#{issue_number}")
        return _success(comment_id=result.get("id"), html_url=result.get("html_url"))
    except Exception as e:
        logger.error(
            f"Error commenting on GitHub issue {repo}#{issue_number}: {e}",
            exc_info=True,
        )
        return _error(str(e))


@mcp.tool()
def github_list_pull_requests(
    repo: str, state: str = "open", limit: int = 30, page: int = 1
) -> str:
    """
    List pull requests in a repository.
    repo: "owner/repo".
    state: "open", "closed", or "all" (default "open").
    limit: max pull requests to return (default 30, capped at 100).
    page: which GitHub results page to fetch (default 1) -- pass the
    previous response's next_page when truncated is true to continue.
    """
    try:
        owner, name = _parse_repo(repo)
        state = _validate_state(state)
        start_page = max(1, int(page or 1))
        response = _request_raw(
            "GET",
            f"/repos/{owner}/{name}/pulls",
            params={
                "state": state,
                "per_page": _clamp_limit(limit),
                "page": start_page,
            },
        )
        result = response.json() if response.content else []
        _require_object_items(result, context="pull request list")
        has_next_page = "next" in _link_header_rels(response.headers.get("Link"))
        return _success(
            pull_requests=[_summarize_pull_request(pr) for pr in result],
            truncated=has_next_page,
            truncation_reason="more_pages" if has_next_page else None,
            next_page=start_page + 1 if has_next_page else None,
        )
    except Exception as e:
        logger.error(
            f"Error listing GitHub pull requests for {repo}: {e}", exc_info=True
        )
        return _error(str(e))


@mcp.tool()
def github_get_pull_request(repo: str, pull_number: int) -> str:
    """
    Get a single pull request by number.
    repo: "owner/repo".
    """
    try:
        owner, name = _parse_repo(repo)
        pull_number = _validate_positive_number(pull_number, field="pull_number")
        result = _request("GET", f"/repos/{owner}/{name}/pulls/{pull_number}")
        _require_object(result, context=f"pull request {repo}#{pull_number}")
        return _success(pull_request=_summarize_pull_request(result))
    except Exception as e:
        logger.error(
            f"Error fetching GitHub pull request {repo}#{pull_number}: {e}",
            exc_info=True,
        )
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
        title = _require_nonblank(title, field="title")
        head = _require_nonblank(head, field="head")
        base = _require_nonblank(base, field="base")
        json_data: dict[str, Any] = {"title": title, "head": head, "base": base}
        if body:
            json_data["body"] = body
        result = _request("POST", f"/repos/{owner}/{name}/pulls", json_data=json_data)
        _require_object(result, context=f"created pull request in '{repo}'")
        return _success(pull_request=_summarize_pull_request(result))
    except Exception as e:
        logger.error(
            f"Error creating GitHub pull request in {repo}: {e}", exc_info=True
        )
        return _error(str(e))


@mcp.tool()
def github_get_file_contents(repo: str, path: str, ref: str = "") -> str:
    """
    Read a file's contents, or list a directory, at a path in a repository.
    repo: "owner/repo".
    path: file or directory path relative to the repo root (e.g. "src/main.py",
    or "" for the repo root).
    ref: optional branch, tag, or commit SHA (defaults to the repo's default branch).

    A file result's `encoding` is "utf-8" for text content, or "base64"
    for binary/non-UTF-8 content (decode the returned `content` yourself
    in that case). A file over the Contents API's ~1MB size limit, or a
    submodule entry, is reported as an error/a distinct `type` rather than
    empty content. A directory result's `entries` is capped at 1000 (the
    Contents API's own limit, with `truncated=true` and no continuation --
    use the Git Trees API for a larger directory).
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
            _require_object_items(result, context=f"directory listing for '{path}'")
            entries = [
                {
                    "name": entry.get("name"),
                    "path": entry.get("path"),
                    # A submodule entry can carry submodule_git_url with
                    # type left as the legacy "file" -- surface the
                    # authoritative marker instead of the possibly-stale
                    # type field so a submodule isn't mistaken for a
                    # readable file.
                    "type": "submodule"
                    if entry.get("submodule_git_url")
                    else entry.get("type"),
                    **(
                        {"submodule_git_url": entry["submodule_git_url"]}
                        if entry.get("submodule_git_url")
                        else {}
                    ),
                }
                for entry in result
            ]
            # GitHub's Contents API returns at most 1000 entries for a
            # directory with no continuation token in the response body --
            # flag it so a directory at exactly that cap isn't mistaken for
            # a complete listing. Unlike the list tools' pagination, there
            # is genuinely no page/cursor parameter to offer here (the
            # Contents API itself doesn't support one for directories), so
            # the remediation is surfaced as a message instead of a
            # continuation field the caller could act on.
            at_cap = len(entries) >= 1000
            return _success(
                type="directory",
                entries=entries,
                truncated=at_cap,
                truncation_reason="entry_cap" if at_cap else None,
                **(
                    {
                        "message": (
                            "This directory has 1000+ entries; the Contents "
                            "API has no continuation for more -- use the Git "
                            "Trees API (not available as a tool here) to list "
                            "the rest"
                        )
                    }
                    if at_cap
                    else {}
                ),
            )
        submodule_git_url = result.get("submodule_git_url")
        if submodule_git_url:
            # A submodule can be returned with type="file" for backward
            # compatibility (as well as the documented type="submodule")
            # and no real content -- checked ahead of the "not a file"
            # rejection below (which a type="file" submodule would
            # otherwise slip past) so it isn't reported as a successful
            # empty read.
            return _success(
                type="submodule",
                path=result.get("path"),
                sha=result.get("sha"),
                submodule_git_url=submodule_git_url,
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
                "returned) -- clone the repository to read it; this connector "
                "has no raw-blob or Trees API tool for large files"
            )
        raw_content = result.get("content") or ""
        if encoding == "base64":
            try:
                decoded_bytes = _decode_base64_content(raw_content)
            except ValueError as decode_exc:
                # Permissive (validate=False) decoding would otherwise turn
                # non-alphabet input (e.g. "!!!!") into silently-truncated
                # or empty bytes while still reporting success.
                return _error(f"File '{path}' has invalid base64 content: {decode_exc}")
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
        logger.error(
            f"Error fetching GitHub file contents {repo}:{path}: {e}", exc_info=True
        )
        return _error(str(e))


@mcp.tool()
def github_list_commits(
    repo: str, path: str = "", limit: int = 30, page: int = 1
) -> str:
    """
    List recent commits in a repository.
    repo: "owner/repo".
    path: optional file or directory path to restrict history to.
    limit: max commits to return (default 30, capped at 100).
    page: which GitHub results page to fetch (default 1) -- pass the
    previous response's next_page when truncated is true to continue.
    """
    try:
        owner, name = _parse_repo(repo)
        start_page = max(1, int(page or 1))
        params: dict[str, Any] = {"per_page": _clamp_limit(limit), "page": start_page}
        if path:
            # Sent as a query param, so (unlike github_get_file_contents'
            # path) there's no injection risk to percent-encode away --
            # but an empty segment (leading/trailing/double slash) or a
            # bare "."/".." segment is still not a real path, and reached
            # GitHub unvalidated here while the equivalent call to
            # github_get_file_contents already rejected it up front.
            # Validated per-segment only for that consistency; the
            # original (unencoded) string is still what's sent, since
            # requests percent-encodes query params on its own.
            for segment in path.split("/"):
                _encode_file_path_segment(segment, field="path")
            params["path"] = path
        response = _request_raw("GET", f"/repos/{owner}/{name}/commits", params=params)
        result = response.json() if response.content else []
        _require_object_items(result, context="commit list")
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
        has_next_page = "next" in _link_header_rels(response.headers.get("Link"))
        return _success(
            commits=commits,
            truncated=has_next_page,
            truncation_reason="more_pages" if has_next_page else None,
            next_page=start_page + 1 if has_next_page else None,
        )
    except Exception as e:
        logger.error(f"Error listing GitHub commits for {repo}: {e}", exc_info=True)
        return _error(str(e))


@mcp.tool()
def github_search_code(
    query: str, limit: int = SEARCH_DEFAULT_LIMIT, page: int = 1
) -> str:
    """
    Search code across GitHub (or, when scoped with "repo:owner/repo" or
    "org:name" in the query, within a specific repository or organization).
    query: a GitHub code-search query, e.g. "repo:octocat/Hello-World def parse".
    limit: max results to return (default 20, capped at 100).
    page: which GitHub results page to fetch (default 1) -- pass the
    previous response's next_page when truncated is true to continue.
    GitHub's Search API caps combined results at 1000 regardless of page.
    """
    try:
        query = _require_nonblank(query, field="query")
        start_page = max(1, int(page or 1))
        response = _request_raw(
            "GET",
            "/search/code",
            params={
                "q": query,
                "per_page": _clamp_limit(limit, default=SEARCH_DEFAULT_LIMIT),
                "page": start_page,
            },
        )
        result = response.json() if response.content else {}
        _require_object(result, context="code search response")
        raw_items = result.get("items") or []
        _require_object_items(raw_items, context="code search results")
        items = [
            {
                "name": item.get("name"),
                "path": item.get("path"),
                "repository": (item.get("repository") or {}).get("full_name"),
                "html_url": item.get("html_url"),
            }
            for item in raw_items
        ]
        has_next_page = "next" in _link_header_rels(response.headers.get("Link"))
        return _success(
            items=items,
            total_count=result.get("total_count", len(items)),
            incomplete_results=bool(result.get("incomplete_results")),
            truncated=has_next_page,
            truncation_reason="more_pages" if has_next_page else None,
            next_page=start_page + 1 if has_next_page else None,
        )
    except Exception as e:
        logger.error(f"Error searching GitHub code for {query!r}: {e}", exc_info=True)
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
