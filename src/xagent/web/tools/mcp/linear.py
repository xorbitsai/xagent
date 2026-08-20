import json
import logging
import os
import re
import time
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

from .utils import setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("linear-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("linear-mcp")

LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql"
DEFAULT_TIMEOUT_SECONDS = 30
MAX_LIMIT = 100
# Linear's GraphQL rate limit does NOT use HTTP 429 / Retry-After like the
# REST-shaped Slack/Intercom siblings -- per Linear's rate-limiting docs, a
# rate-limited request returns HTTP 400 with a RATELIMITED code inside
# errors[].extensions, and reset info in the X-RateLimit-Requests-Reset
# header (UTC epoch milliseconds). On that signal we wait once and retry
# rather than failing outright -- pagination here can fire up to
# MAX_ISSUE_SEARCH_PAGES/MAX_USER_SEARCH_PAGES sequential requests per tool
# call, making a transient rate limit far more likely to surface mid-fetch
# than in a single-request tool.
MAX_RETRY_AFTER_SECONDS = 30
# Matches zoom.py's convention: an error body that isn't the expected
# {"errors": [...]} GraphQL shape (e.g. an HTML gateway error page) must not
# be forwarded to the LLM/logs verbatim and unbounded.
MAX_ERROR_RESPONSE_TEXT_CHARS = 1000
# Bounded multi-page fetch for a client-side text filter (linear_search_issues'
# title match, since Linear's API has no server-side title filter) --
# mirrors slack.py's/aws.py's MAX_PAGES precedent: a single MAX_LIMIT-sized
# page can undercount real matches that live further out.
MAX_ISSUE_SEARCH_PAGES = 10
# Workspaces typically have far fewer members than a repository has issues,
# so linear_search_users gets a smaller cap for the same bounded-fetch reason.
MAX_USER_SEARCH_PAGES = 5
# Neither issues(...) nor users(...) below sets an explicit orderBy -- per
# Linear's docs, that defaults to createdAt, which is immutable and
# monotonically increasing, so paginating via after: endCursor across
# multiple requests can't duplicate or skip rows even if new ones are
# created mid-scan.

_UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

# Per Linear's own SDK type definitions (WorkflowStateFilter.type's doc
# comment) -- "duplicate" is a real, documented value that a plain reading
# of the Team/WorkflowState object docs (which only list 6) would miss.
_VALID_STATE_TYPES = frozenset(
    {"triage", "backlog", "unstarted", "started", "completed", "canceled", "duplicate"}
)

_VALID_PRIORITIES = frozenset({0, 1, 2, 3, 4})


def _validate_priority(priority: int | None) -> str | None:
    if priority is not None and priority not in _VALID_PRIORITIES:
        return f"priority must be one of {sorted(_VALID_PRIORITIES)}, got: {priority!r}"
    return None


def _validate_title(title: str | None) -> str | None:
    """None means "leave unset" (only meaningful for linear_update_issue)
    and always passes; a provided title must be non-blank."""
    if title is not None and not title.strip():
        return "title must not be empty"
    return None


def _success(*, _errors: list[Any] | None = None, **payload: Any) -> str:
    """`_errors` becomes a `warnings` list entry, not to be confused with
    the unrelated `error` string some pagination-loop callers pass directly
    in `payload` (a page-fetch exception after earlier pages succeeded).
    The two describe different things and can both be present at once: a
    sub-field resolver failure on an already-fetched page (`warnings`) and
    the reason a *later* page could not be fetched at all (`error`) --
    neither takes precedence over the other, both are simply reported.
    """
    body: dict[str, Any] = {"status": "success", **payload}
    if _errors:
        # A genuine partial GraphQL success (one sub-field failed, others
        # resolved) -- surface it in the result instead of only the server
        # log, consistent with this module's own truncated=true contract
        # for "results are incomplete".
        body["warnings"] = [_graphql_errors_message(_errors)]
    return json.dumps(body, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _headers() -> dict[str, str]:
    access_token = os.environ.get("LINEAR_ACCESS_TOKEN")
    if not access_token:
        raise ValueError("LINEAR_ACCESS_TOKEN environment variable is missing")
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }


def _clamp_limit(limit: int) -> int:
    return max(1, min(int(limit), MAX_LIMIT))


def _graphql_errors_message(errors: list[Any]) -> str:
    messages = []
    for entry in errors:
        if isinstance(entry, dict) and entry.get("message"):
            messages.append(str(entry["message"]))
        else:
            messages.append(str(entry))
    return "; ".join(messages) if messages else "Unknown Linear API error"


def _truncate_error_text(text: str) -> str:
    if len(text) > MAX_ERROR_RESPONSE_TEXT_CHARS:
        return text[:MAX_ERROR_RESPONSE_TEXT_CHARS] + "... [truncated]"
    return text


def _is_rate_limited(response: requests.Response) -> bool:
    """Linear signals a rate-limited GraphQL request as HTTP 400 with a
    RATELIMITED code in errors[].extensions -- not HTTP 429."""
    if response.status_code != 400:
        return False
    try:
        payload = response.json()
    except ValueError:
        return False
    if not isinstance(payload, dict):
        return False
    for entry in payload.get("errors") or []:
        if (
            isinstance(entry, dict)
            and (entry.get("extensions") or {}).get("code") == "RATELIMITED"
        ):
            return True
    return False


def _rate_limit_wait_seconds(response: requests.Response) -> float | None:
    """Linear reports the reset time via X-RateLimit-Requests-Reset, a UTC
    epoch-milliseconds timestamp -- not a Retry-After header.

    Returns None when the header is missing/unparsable (nothing to act
    on -- distinct from a valid, already-elapsed reset time). A return of
    0.0 means the reset window has already passed by the time this is
    read, so an immediate retry (not a skipped one) is the likely-to-
    succeed move.
    """
    reset_header = response.headers.get("X-RateLimit-Requests-Reset")
    if not reset_header:
        return None
    try:
        reset_ms = int(reset_header)
    except ValueError:
        return None
    return max(0.0, (reset_ms / 1000.0) - time.time())


def _graphql(
    query: str, variables: dict[str, Any] | None = None
) -> tuple[dict[str, Any], list[Any]]:
    """Run one GraphQL query/mutation against Linear's single API endpoint.

    Linear answers auth failures with a non-200 status (body shape not
    guaranteed) and rate limiting with HTTP 400 + RATELIMITED (see
    _is_rate_limited), but schema/validation errors with HTTP 200 and a
    top-level "errors" array.

    Returns (data, errors). Every query/mutation in this module selects
    exactly one top-level field, so if that field comes back null alongside
    a non-empty "errors" array there is nothing usable to return -- that is
    treated as a hard failure and raises instead. If at least one top-level
    field is non-null, it's a genuine partial success (e.g. a nested
    sub-field's resolver failed): errors is returned alongside data so the
    caller can surface it as a warning rather than only logging it.
    """
    for attempt in (0, 1):
        response = requests.post(
            LINEAR_GRAPHQL_URL,
            headers=_headers(),
            json={"query": query, "variables": variables or {}},
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        if attempt == 0 and _is_rate_limited(response):
            wait_seconds = _rate_limit_wait_seconds(response)
            if wait_seconds is not None and wait_seconds <= MAX_RETRY_AFTER_SECONDS:
                # max(0.0, ...): _rate_limit_wait_seconds already clamps to
                # non-negative, but time.sleep() raises ValueError on a
                # negative argument -- this is belt-and-suspenders against
                # that clamp ever being dropped in a future edit.
                time.sleep(max(0.0, wait_seconds))
                continue
        break

    if response.status_code >= 400:
        detail: str | None = None
        try:
            payload = response.json()
            if isinstance(payload, dict) and payload.get("errors"):
                detail = _graphql_errors_message(payload["errors"])
        except ValueError:
            pass
        if detail is None:
            detail = _truncate_error_text(response.text.strip())
        raise RuntimeError(
            f"Linear API error (status {response.status_code}): {detail}"
        )

    try:
        payload = response.json()
    except ValueError:
        detail = _truncate_error_text(response.text.strip())
        raise RuntimeError(
            f"Linear API returned a non-JSON response: {detail}"
        ) from None
    if not isinstance(payload, dict):
        raise RuntimeError(
            "Linear API returned an unexpected (non-object) response body"
        )

    data = payload.get("data") or {}
    errors = payload.get("errors") or []
    if errors:
        message = _graphql_errors_message(errors)
        if all(value is None for value in data.values()):
            # Every top-level field is null (or data is empty) -- nothing
            # usable to return, e.g. a permission failure on the single
            # requested object. Checking for None specifically (not just
            # falsy) matters because a resolved-but-empty field like {} or
            # [] is a genuine partial success, not a hard failure.
            raise RuntimeError(message)
        # At least one top-level field resolved -- a genuine partial
        # success. Log for operators; the caller surfaces `errors` in its
        # own result via _success's `_errors`.
        logger.warning(f"Linear GraphQL partial error (data still returned): {message}")
    return data, errors


def _resolve_team_uuid(team_id: str) -> str:
    """Resolve team_id to its real UUID.

    Mutation input types (e.g. IssueCreateInput.teamId) and filter inputs
    (e.g. IssueFilter.team.id) require the team's actual UUID, not its
    human-readable key (e.g. "ENG") -- but linear_list_teams hands back
    both, and every tool that takes a team_id here documents accepting
    either. An already-UUID team_id is returned unchanged without a
    round-trip.

    Resolution goes through `teams(filter: { key: { eqIgnoreCase: ... } })`,
    not the top-level `team(id: ...)` field -- Linear's own SDK type
    definitions document TeamFilter.key as the supported way to look up a
    team by its key, but never document `team(id:)` accepting anything but
    a UUID ("Fetches a specific team by its ID"), unlike issue-related id
    fields, which explicitly document accepting either form. Filtering by
    key is the unambiguous, documented path regardless of what team(id:)
    does with a raw key.

    `eqIgnoreCase`, not `eq`, because team keys are conventionally uppercase
    (e.g. "ENG") but nothing in the calling tools' docstrings tells a
    caller that -- an LLM that lowercases user text ("the eng team") must
    still resolve to the same team.

    (No equivalent resolver exists for issue_id: Linear's
    CommentCreateInput.issueId and every other issueId-typed input accept
    either an issue's UUID or its human-readable identifier directly,
    confirmed against Linear's own SDK type definitions -- so passing
    issue_id straight through needs no lookup.)
    """
    if _UUID_PATTERN.match(team_id):
        return team_id
    data, _errors = _graphql(
        "query($key: String!) { teams(filter: { key: { eqIgnoreCase: $key } },"
        " first: 1) { nodes { id } } }",
        {"key": team_id},
    )
    teams = (data.get("teams") or {}).get("nodes") or []
    if not teams:
        raise ValueError(f"Team '{team_id}' not found")
    return str(teams[0]["id"])


@mcp.tool()
def linear_get_current_user() -> str:
    """
    Get the profile of the Linear account this connector is authenticated as
    (id, name, email, display name, whether the account is a workspace
    admin). Use this for "my account" / "who am I" requests instead of
    asking the user for their Linear user id.
    """
    try:
        data, errors = _graphql("query { viewer { id name email displayName admin } }")
        return _success(user=data.get("viewer") or {}, _errors=errors)
    except Exception as e:
        logger.error(f"Error fetching authenticated Linear user: {e}")
        return _error(str(e))


@mcp.tool()
def linear_list_teams(limit: int = 50) -> str:
    """
    List the teams (workspaces' sub-organizations) this account can see —
    id, key (e.g. "ENG"), and name. Use the returned id/key with
    linear_list_workflow_states, linear_list_labels, linear_create_issue, and
    linear_search_issues.
    """
    try:
        data, errors = _graphql(
            "query($first: Int!) { teams(first: $first) { nodes { id key name }"
            " pageInfo { hasNextPage } } }",
            {"first": _clamp_limit(limit)},
        )
        teams_data = data.get("teams") or {}
        teams = teams_data.get("nodes") or []
        truncated = bool((teams_data.get("pageInfo") or {}).get("hasNextPage"))
        return _success(teams=teams, truncated=truncated, _errors=errors)
    except Exception as e:
        logger.error(f"Error listing Linear teams: {e}")
        return _error(str(e))


@mcp.tool()
def linear_list_workflow_states(team_id: str, limit: int = 100) -> str:
    """
    List a team's workflow states (e.g. "Todo", "In Progress", "Done") — id,
    name, and type. Resolve a state name to an id here before passing
    state_id to linear_update_issue.
    team_id: a team id or key from linear_list_teams.
    """
    try:
        resolved_team_id = _resolve_team_uuid(team_id)
        data, errors = _graphql(
            "query($teamId: String!, $first: Int!) { team(id: $teamId) {"
            " states(first: $first) { nodes { id name type position }"
            " pageInfo { hasNextPage } } } }",
            {"teamId": resolved_team_id, "first": _clamp_limit(limit)},
        )
        team = data.get("team")
        if not team or not isinstance(team, dict):
            return _error(f"Team '{team_id}' not found")
        states_data = team.get("states") or {}
        states = states_data.get("nodes") or []
        truncated = bool((states_data.get("pageInfo") or {}).get("hasNextPage"))
        return _success(states=states, truncated=truncated, _errors=errors)
    except Exception as e:
        logger.error(f"Error listing Linear workflow states for team {team_id}: {e}")
        return _error(str(e))


@mcp.tool()
def linear_list_labels(team_id: str, limit: int = 100) -> str:
    """
    List a team's issue labels — id and name. Resolve label names to ids
    here before passing label_ids to linear_create_issue or
    linear_update_issue.
    team_id: a team id or key from linear_list_teams.
    """
    try:
        resolved_team_id = _resolve_team_uuid(team_id)
        data, errors = _graphql(
            "query($teamId: String!, $first: Int!) { team(id: $teamId) {"
            " labels(first: $first) { nodes { id name color }"
            " pageInfo { hasNextPage } } } }",
            {"teamId": resolved_team_id, "first": _clamp_limit(limit)},
        )
        team = data.get("team")
        if not team or not isinstance(team, dict):
            return _error(f"Team '{team_id}' not found")
        labels_data = team.get("labels") or {}
        labels = labels_data.get("nodes") or []
        truncated = bool((labels_data.get("pageInfo") or {}).get("hasNextPage"))
        return _success(labels=labels, truncated=truncated, _errors=errors)
    except Exception as e:
        logger.error(f"Error listing Linear labels for team {team_id}: {e}")
        return _error(str(e))


@mcp.tool()
def linear_search_users(query: str = "", limit: int = 20) -> str:
    """
    Search workspace members by name or email — id, name, email. Resolve a
    person to an id here before passing assignee_id to linear_create_issue
    or linear_update_issue. Leave query empty to list every workspace
    member.
    """
    try:
        needle = query.strip().lower()
        max_matches = _clamp_limit(limit)
        matches: list[dict[str, Any]] = []
        cursor: str | None = None
        has_next_page = False
        all_errors: list[Any] = []
        pages_scanned = 0
        # Linear's API has no server-side name/email filter, so a match
        # beyond the first MAX_LIMIT-sized page would otherwise be silently
        # missed -- keep paging (bounded) until enough matches are found or
        # the server genuinely runs out of pages.
        for _ in range(MAX_USER_SEARCH_PAGES):
            try:
                data, errors = _graphql(
                    "query($first: Int!, $after: String) { users(first: $first,"
                    " after: $after) { nodes { id name email active }"
                    " pageInfo { hasNextPage endCursor } } }",
                    {"first": MAX_LIMIT, "after": cursor},
                )
            except Exception as page_exc:
                if not pages_scanned:
                    raise
                # A mid-pagination failure must not discard matches already
                # collected -- return the partial list with a marker
                # instead, mirroring slack_list_channels.
                logger.warning(
                    f"Linear user search pagination stopped early: {page_exc}"
                )
                return _success(
                    users=matches[:max_matches],
                    truncated=True,
                    error=str(page_exc),
                    _errors=all_errors,
                )
            pages_scanned += 1
            all_errors.extend(errors)
            users_data = data.get("users") or {}
            raw_page = users_data.get("nodes") or []
            page_info = users_data.get("pageInfo") or {}
            has_next_page = bool(page_info.get("hasNextPage"))
            cursor = page_info.get("endCursor")
            if needle:
                matches.extend(
                    user
                    for user in raw_page
                    if needle in str(user.get("name") or "").lower()
                    or needle in str(user.get("email") or "").lower()
                )
            else:
                matches.extend(raw_page)
            if len(matches) >= max_matches or not has_next_page:
                break
            if not cursor:
                # hasNextPage is true but Linear gave no cursor to continue
                # from -- refetching without one would just re-request page
                # 1 and duplicate nodes, so stop instead of looping forever.
                break
        truncated = has_next_page or len(matches) > max_matches
        return _success(
            users=matches[:max_matches], truncated=truncated, _errors=all_errors
        )
    except Exception as e:
        logger.error(f"Error searching Linear users for '{query}': {e}")
        return _error(str(e))


# Used for the list/search path -- excludes the (potentially large,
# unbounded) description body, since up to MAX_ISSUE_SEARCH_PAGES *
# MAX_LIMIT issues can be fetched server-side to answer one search.
# Mirrors hubspot.py's trimmed _FORM_SUMMARY_FIELDS/_EMAIL_SUMMARY_FIELDS
# and intercom.py's bounded conversation "preview" for the same reason:
# don't forward an unbounded body per list row to the LLM.
_ISSUE_SUMMARY_FIELDS = (
    "id identifier title priority url createdAt updatedAt"
    " state { id name type } assignee { id name email }"
    " team { id key name } labels { nodes { id name } }"
)
# Used for single-issue fetches and mutation results, where the caller
# asked about (or just created/updated) exactly one issue and the full
# body is the point. Derived from _ISSUE_SUMMARY_FIELDS (rather than a
# second hand-maintained literal) so the two field sets can't silently
# drift apart on a future edit to the fields they share.
_ISSUE_DETAIL_FIELDS = _ISSUE_SUMMARY_FIELDS.replace("title", "title description", 1)


@mcp.tool()
def linear_search_issues(
    query: str = "",
    team_id: str = "",
    assignee_id: str = "",
    state_type: str = "",
    limit: int = 20,
) -> str:
    """
    Search/list issues, optionally filtered by team, assignee, or workflow
    state type.
    query: optional case-insensitive substring matched against title (Linear
    has no full-text filter on this field via the API, so this is applied
    client-side — up to MAX_ISSUE_SEARCH_PAGES pages are fetched past a
    page with no matches yet, so a match outside the first page isn't
    missed, though a very large result set can still exhaust that bound).
    Returned issues omit the full description body (use linear_get_issue
    for that) to avoid fetching an unbounded amount of text per result.
    team_id: optional team id/key from linear_list_teams.
    assignee_id: optional user id from linear_search_users.
    state_type: optional workflow state type to filter by — one of
    "triage", "backlog", "unstarted", "started", "completed", "canceled",
    "duplicate".
    limit: max issues to return (default 20, capped at 100).
    """
    try:
        if state_type and state_type not in _VALID_STATE_TYPES:
            return _error(
                f"state_type must be one of {sorted(_VALID_STATE_TYPES)}, "
                f"got: {state_type!r}"
            )
        filter_parts = []
        variables: dict[str, Any] = {}
        if team_id:
            filter_parts.append("team: { id: { eq: $teamId } }")
            variables["teamId"] = _resolve_team_uuid(team_id)
        if assignee_id:
            filter_parts.append("assignee: { id: { eq: $assigneeId } }")
            variables["assigneeId"] = assignee_id
        if state_type:
            filter_parts.append("state: { type: { eq: $stateType } }")
            variables["stateType"] = state_type

        filter_clause = ""
        filter_signature = ""
        if filter_parts:
            filter_clause = ", filter: { " + ", ".join(filter_parts) + " }"
            type_signature = {
                "teamId": "$teamId: ID!",
                "assigneeId": "$assigneeId: ID!",
                "stateType": "$stateType: String!",
            }
            filter_signature = ", " + ", ".join(
                type_signature[key] for key in variables
            )

        max_results = _clamp_limit(limit)
        needle = query.strip().lower()

        graphql_query = (
            f"query($first: Int!, $after: String{filter_signature}) {{"
            f" issues(first: $first, after: $after{filter_clause}) {{"
            f" nodes {{ {_ISSUE_SUMMARY_FIELDS} }}"
            " pageInfo { hasNextPage endCursor } } }"
        )

        if not needle:
            variables["first"] = max_results
            variables["after"] = None
            data, errors = _graphql(graphql_query, variables)
            issues_data = data.get("issues") or {}
            issues = issues_data.get("nodes") or []
            has_next_page = bool((issues_data.get("pageInfo") or {}).get("hasNextPage"))
            return _success(
                issues=issues[:max_results], truncated=has_next_page, _errors=errors
            )

        # Linear's API has no server-side title filter, so a single
        # MAX_LIMIT-sized page can undercount real matches that live
        # further out -- keep paging (bounded) until enough matches are
        # found or the server genuinely runs out of pages.
        matches: list[dict[str, Any]] = []
        cursor: str | None = None
        has_next_page = False
        all_errors: list[Any] = []
        pages_scanned = 0
        for _ in range(MAX_ISSUE_SEARCH_PAGES):
            # A fresh dict per page, not a mutate-in-place reuse of
            # `variables` -- each request's payload must stay independent
            # of later loop iterations' state.
            page_variables = {**variables, "first": MAX_LIMIT, "after": cursor}
            try:
                data, errors = _graphql(graphql_query, page_variables)
            except Exception as page_exc:
                if not pages_scanned:
                    raise
                # A mid-pagination failure must not discard matches already
                # collected -- return the partial list with a marker
                # instead, mirroring slack_list_channels.
                logger.warning(
                    f"Linear issue search pagination stopped early: {page_exc}"
                )
                return _success(
                    issues=matches[:max_results],
                    truncated=True,
                    error=str(page_exc),
                    _errors=all_errors,
                )
            pages_scanned += 1
            all_errors.extend(errors)
            issues_data = data.get("issues") or {}
            raw_page = issues_data.get("nodes") or []
            page_info = issues_data.get("pageInfo") or {}
            has_next_page = bool(page_info.get("hasNextPage"))
            cursor = page_info.get("endCursor")
            matches.extend(
                issue
                for issue in raw_page
                if needle in str(issue.get("title") or "").lower()
            )
            if len(matches) >= max_results or not has_next_page:
                break
            if not cursor:
                # hasNextPage is true but Linear gave no cursor to continue
                # from -- refetching without one would just re-request page
                # 1 and duplicate nodes, so stop instead of looping forever.
                break
        truncated = has_next_page or len(matches) > max_results
        return _success(
            issues=matches[:max_results], truncated=truncated, _errors=all_errors
        )
    except Exception as e:
        logger.error(f"Error searching Linear issues: {e}")
        return _error(str(e))


@mcp.tool()
def linear_get_issue(issue_id: str) -> str:
    """
    Get one issue's full details, including labels, assignee, and state.
    issue_id: an issue UUID or its human-readable identifier (e.g. "ENG-123").
    """
    try:
        data, errors = _graphql(
            f"query($id: String!) {{ issue(id: $id) {{ {_ISSUE_DETAIL_FIELDS} }} }}",
            {"id": issue_id},
        )
        issue = data.get("issue")
        if not issue or not isinstance(issue, dict):
            return _error(f"Issue '{issue_id}' not found")
        return _success(issue=issue, _errors=errors)
    except Exception as e:
        logger.error(f"Error fetching Linear issue {issue_id}: {e}")
        return _error(str(e))


@mcp.tool()
def linear_create_issue(
    team_id: str,
    title: str,
    description: str = "",
    assignee_id: str = "",
    priority: int | None = None,
    label_ids: list[str] | None = None,
) -> str:
    """
    Create a new issue.
    team_id: the target team's id or key, from linear_list_teams.
    title: the issue title.
    description: optional body (Markdown supported).
    assignee_id: optional user id from linear_search_users.
    priority: 0 (no priority), 1 (urgent), 2 (high), 3 (normal), 4 (low);
    omit to let Linear apply its own default.
    label_ids: optional label ids from linear_list_labels.
    """
    try:
        if err := _validate_title(title):
            return _error(err)
        if err := _validate_priority(priority):
            return _error(err)
        issue_input: dict[str, Any] = {
            "teamId": _resolve_team_uuid(team_id),
            "title": title,
        }
        if description:
            issue_input["description"] = description
        if assignee_id:
            issue_input["assigneeId"] = assignee_id
        if priority is not None:
            issue_input["priority"] = priority
        if label_ids is not None:
            issue_input["labelIds"] = label_ids

        data, errors = _graphql(
            "mutation($input: IssueCreateInput!) { issueCreate(input: $input)"
            f" {{ success issue {{ {_ISSUE_DETAIL_FIELDS} }} }} }}",
            {"input": issue_input},
        )
        result = data.get("issueCreate") or {}
        if not result.get("success"):
            return _error("Linear reported the issue was not created")
        return _success(issue=result.get("issue"), _errors=errors)
    except Exception as e:
        logger.error(f"Error creating Linear issue in team {team_id}: {e}")
        return _error(str(e))


@mcp.tool()
def linear_update_issue(
    issue_id: str,
    title: str | None = None,
    description: str | None = None,
    state_id: str | None = None,
    assignee_id: str | None = None,
    priority: int | None = None,
    label_ids: list[str] | None = None,
    add_label_ids: list[str] | None = None,
    remove_label_ids: list[str] | None = None,
) -> str:
    """
    Update an existing issue. Only the fields explicitly provided are
    changed; leave a parameter unset (None) to leave that field untouched.
    issue_id: an issue UUID or its human-readable identifier (e.g. "ENG-123").
    state_id: a workflow state id from linear_list_workflow_states, to move
    the issue (e.g. to "Done").
    assignee_id: a user id from linear_search_users, to reassign the issue —
    pass an empty string to unassign it.
    priority: 0 (no priority), 1 (urgent), 2 (high), 3 (normal), 4 (low).
    label_ids: label ids from linear_list_labels, replacing the issue's
    entire label set; pass an empty list to remove all labels. Leave unset
    to leave existing labels untouched, or use add_label_ids/remove_label_ids
    instead to change specific labels without needing to know the full
    existing set. Cannot be combined with add_label_ids/remove_label_ids.
    add_label_ids: label ids from linear_list_labels to add, without
    affecting labels not listed here.
    remove_label_ids: label ids from linear_list_labels to remove, without
    affecting labels not listed here.
    """
    try:
        if err := _validate_title(title):
            return _error(err)
        if err := _validate_priority(priority):
            return _error(err)
        if label_ids is not None and (
            add_label_ids is not None or remove_label_ids is not None
        ):
            return _error(
                "label_ids replaces the full label set and cannot be "
                "combined with add_label_ids/remove_label_ids — use one or "
                "the other"
            )
        if (
            add_label_ids
            and remove_label_ids
            and set(add_label_ids) & set(remove_label_ids)
        ):
            return _error(
                "add_label_ids and remove_label_ids cannot share the same label id"
            )

        issue_input: dict[str, Any] = {}
        if title is not None:
            issue_input["title"] = title
        if description is not None:
            issue_input["description"] = description
        if state_id is not None:
            issue_input["stateId"] = state_id
        if assignee_id is not None:
            issue_input["assigneeId"] = assignee_id or None
        if priority is not None:
            issue_input["priority"] = priority
        if label_ids is not None:
            issue_input["labelIds"] = label_ids
        if add_label_ids is not None:
            issue_input["addedLabelIds"] = add_label_ids
        if remove_label_ids is not None:
            issue_input["removedLabelIds"] = remove_label_ids
        if not issue_input:
            return _error("No fields provided to update")

        data, errors = _graphql(
            "mutation($id: String!, $input: IssueUpdateInput!) {"
            f" issueUpdate(id: $id, input: $input) {{ success issue {{ {_ISSUE_DETAIL_FIELDS} }} }} }}",
            {"id": issue_id, "input": issue_input},
        )
        result = data.get("issueUpdate") or {}
        if not result.get("success"):
            return _error("Linear reported the issue was not updated")
        return _success(issue=result.get("issue"), _errors=errors)
    except Exception as e:
        logger.error(f"Error updating Linear issue {issue_id}: {e}")
        return _error(str(e))


@mcp.tool()
def linear_list_comments(issue_id: str, limit: int = 50) -> str:
    """
    List comments on an issue (body, author, timestamp).
    issue_id: an issue UUID or its human-readable identifier (e.g. "ENG-123").
    """
    try:
        data, errors = _graphql(
            "query($id: String!, $first: Int!) { issue(id: $id) { comments"
            "(first: $first) { nodes { id body createdAt user { id name } }"
            " pageInfo { hasNextPage } } } }",
            {"id": issue_id, "first": _clamp_limit(limit)},
        )
        issue = data.get("issue")
        if not issue or not isinstance(issue, dict):
            return _error(f"Issue '{issue_id}' not found")
        comments_data = issue.get("comments") or {}
        comments = comments_data.get("nodes") or []
        truncated = bool((comments_data.get("pageInfo") or {}).get("hasNextPage"))
        return _success(comments=comments, truncated=truncated, _errors=errors)
    except Exception as e:
        logger.error(f"Error listing comments for Linear issue {issue_id}: {e}")
        return _error(str(e))


@mcp.tool()
def linear_add_comment(issue_id: str, body: str) -> str:
    """
    Add a comment to an issue.
    issue_id: an issue UUID or its human-readable identifier (e.g. "ENG-123").
    body: the comment text (Markdown supported).
    """
    try:
        if not body.strip():
            return _error("body must not be empty")
        data, errors = _graphql(
            "mutation($input: CommentCreateInput!) { commentCreate(input: $input)"
            " { success comment { id body createdAt } } }",
            {"input": {"issueId": issue_id, "body": body}},
        )
        result = data.get("commentCreate") or {}
        if not result.get("success"):
            return _error("Linear reported the comment was not created")
        return _success(comment=result.get("comment"), _errors=errors)
    except Exception as e:
        logger.error(f"Error adding comment to Linear issue {issue_id}: {e}")
        return _error(str(e))


@mcp.tool()
def linear_list_projects(team_id: str = "", limit: int = 50) -> str:
    """
    List projects, optionally scoped to one team — id, name, status, and
    progress.
    team_id: optional team id/key from linear_list_teams; omit to list
    across every team this account can see.
    """
    try:
        max_results = _clamp_limit(limit)
        if team_id:
            resolved_team_id = _resolve_team_uuid(team_id)
            data, errors = _graphql(
                "query($teamId: String!, $first: Int!) { team(id: $teamId) {"
                " projects(first: $first) { nodes { id name"
                " status { id name type } progress url }"
                " pageInfo { hasNextPage } } } }",
                {"teamId": resolved_team_id, "first": max_results},
            )
            team = data.get("team")
            if not team or not isinstance(team, dict):
                return _error(f"Team '{team_id}' not found")
            projects_data = team.get("projects") or {}
        else:
            data, errors = _graphql(
                "query($first: Int!) { projects(first: $first) { nodes { id"
                " name status { id name type } progress url }"
                " pageInfo { hasNextPage } } }",
                {"first": max_results},
            )
            projects_data = data.get("projects") or {}
        projects = projects_data.get("nodes") or []
        truncated = bool((projects_data.get("pageInfo") or {}).get("hasNextPage"))
        return _success(projects=projects, truncated=truncated, _errors=errors)
    except Exception as e:
        logger.error(f"Error listing Linear projects: {e}")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
