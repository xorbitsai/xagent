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
# rather than failing outright.
MAX_RETRY_AFTER_SECONDS = 30
# Matches zoom.py's convention: an error body that isn't the expected
# {"errors": [...]} GraphQL shape (e.g. an HTML gateway error page) must not
# be forwarded to the LLM/logs verbatim and unbounded.
MAX_ERROR_RESPONSE_TEXT_CHARS = 1000

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
    """The result envelope has exactly one outcome field, `status`
    ("success" here, "error" from `_error()`), plus two independent
    incompleteness signals that a caller may need alongside a successful
    result: `truncated` (more results exist past `limit`) and `warnings`
    (this call's own top-level field resolved, but a nested sub-field on it
    failed -- see `_graphql`'s docstring). Neither implies or excludes the
    other; a single response can be truncated, carry warnings, both, or
    neither, all while `status` stays "success".
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


def _mutation_failure_message(base_message: str, errors: list[Any]) -> str:
    """`success: false` on its own gives the caller nothing to act on --
    bad team, missing permission, and invalid label id all look identical.
    Fold in whatever detail the same GraphQL response's `errors` array
    carries (already returned by `_graphql` alongside `data`), when any."""
    if not errors:
        return base_message
    return f"{base_message}: {_graphql_errors_message(errors)}"


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


_RATE_LIMIT_RESET_HEADERS = (
    "X-RateLimit-Requests-Reset",
    "X-RateLimit-Endpoint-Requests-Reset",
    "X-RateLimit-Complexity-Reset",
)


def _rate_limit_wait_seconds(response: requests.Response) -> float | None:
    """Linear enforces three independent rate-limit dimensions -- requests,
    endpoint requests, and complexity -- each with its own UTC
    epoch-milliseconds `X-RateLimit-*-Reset` header, and all three surface
    the same RATELIMITED code. The list/search calls in this module are
    exactly the shape most likely to trip the *complexity* limit (large
    `first`, nested selections), so reading only one dimension's header
    could wait on the wrong window; take the max reset across whichever of
    the three headers are present on the response.

    Returns None when none of the headers are present/parsable (nothing to
    act on -- distinct from a valid, already-elapsed reset time). A return
    of 0.0 means every present reset window has already passed by the time
    this is read, so an immediate retry (not a skipped one) is the likely-
    to-succeed move.
    """
    wait_seconds: float | None = None
    for header_name in _RATE_LIMIT_RESET_HEADERS:
        reset_header = response.headers.get(header_name)
        if not reset_header:
            continue
        try:
            reset_ms = int(reset_header)
        except ValueError:
            continue
        header_wait = max(0.0, (reset_ms / 1000.0) - time.time())
        wait_seconds = (
            header_wait if wait_seconds is None else max(wait_seconds, header_wait)
        )
    return wait_seconds


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
    if len(data) > 1:
        # The hard-failure vs partial-success discriminator below is only
        # correct when every query/mutation in this module selects exactly
        # one top-level field (documented above) -- enforced here so a
        # future multi-root-field query fails loudly instead of silently
        # letting one null field alongside one resolved field be
        # misclassified as a partial success.
        raise RuntimeError(
            f"Linear API response had {len(data)} top-level fields "
            f"({sorted(data)}), but this module's error handling assumes "
            "exactly one"
        )
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


def _team_scoped_list(
    team_id: str, field_name: str, first: int, selection: str
) -> tuple[list[Any], bool, list[Any]]:
    """Shared skeleton behind every `team(id: ...) { <field_name> { ... } }`
    tool below (workflow states, labels, and the team-scoped half of
    projects): resolve team → single-field query → validate the team
    exists → extract nodes/truncated. Raises ValueError with the same
    "Team '...' not found" message each call site returned directly before
    this was extracted, so the caller's existing `except Exception` handling
    needs no change."""
    resolved_team_id = _resolve_team_uuid(team_id)
    data, errors = _graphql(
        f"query($teamId: String!, $first: Int!) {{ team(id: $teamId) {{"
        f" {field_name}(first: $first) {{ nodes {{ {selection} }}"
        " pageInfo { hasNextPage } } } }",
        {"teamId": resolved_team_id, "first": first},
    )
    team = data.get("team")
    if not team or not isinstance(team, dict):
        raise ValueError(f"Team '{team_id}' not found")
    field_data = team.get(field_name) or {}
    nodes = field_data.get("nodes") or []
    truncated = bool((field_data.get("pageInfo") or {}).get("hasNextPage"))
    return nodes, truncated, errors


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
        states, truncated, errors = _team_scoped_list(
            team_id, "states", _clamp_limit(limit), "id name type position"
        )
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
        labels, truncated, errors = _team_scoped_list(
            team_id, "labels", _clamp_limit(limit), "id name color"
        )
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
        needle = query.strip()
        max_matches = _clamp_limit(limit)
        variables: dict[str, Any] = {"first": max_matches}
        filter_clause = ""
        if needle:
            # UserFilter.name/email are StringComparators supporting
            # containsIgnoreCase, and UserFilter.or combines sub-filters --
            # confirmed against Linear's own GraphQL schema. Matching name
            # OR email server-side means the result is complete up to
            # max_matches, no client-side scanning or bounded page cap
            # needed.
            filter_clause = (
                ", filter: { or: [{ name: { containsIgnoreCase: $query } },"
                " { email: { containsIgnoreCase: $query } }] }"
            )
            variables["query"] = needle
        data, errors = _graphql(
            f"query($first: Int!{', $query: String!' if needle else ''}) {{"
            f" users(first: $first{filter_clause}) {{ nodes {{ id name email"
            " active } pageInfo { hasNextPage } } }",
            variables,
        )
        users_data = data.get("users") or {}
        users = users_data.get("nodes") or []
        truncated = bool((users_data.get("pageInfo") or {}).get("hasNextPage"))
        return _success(users=users, truncated=truncated, _errors=errors)
    except Exception as e:
        logger.error(f"Error searching Linear users for '{query}': {e}")
        return _error(str(e))


# Used for the list/search path -- excludes the (potentially large,
# unbounded) description body, since up to MAX_LIMIT issues can be
# returned per call. Mirrors hubspot.py's trimmed
# _FORM_SUMMARY_FIELDS/_EMAIL_SUMMARY_FIELDS and intercom.py's bounded
# conversation "preview" for the same reason: don't forward an unbounded
# body per list row to the LLM.
_ISSUE_SUMMARY_FIELDS = (
    "id identifier title priority url createdAt updatedAt"
    " state { id name type } assignee { id name email }"
    " team { id key name } labels { nodes { id name } }"
)
# Used for single-issue fetches and mutation results, where the caller
# asked about (or just created/updated) exactly one issue and the full
# body is the point. Appends to _ISSUE_SUMMARY_FIELDS (rather than a
# second hand-maintained literal) so the two field sets can't silently
# drift apart on a future edit to the fields they share.
_ISSUE_DETAIL_FIELDS = f"{_ISSUE_SUMMARY_FIELDS} description"


@mcp.tool()
def linear_search_issues(
    query: str = "",
    team_id: str = "",
    assignee_id: str = "",
    state_type: str = "",
    limit: int = 20,
) -> str:
    """
    Search/list issues, optionally filtered by team, assignee, workflow
    state type, or a title substring.
    query: optional case-insensitive substring matched against title,
    filtered server-side (Linear's IssueFilter.title is a StringComparator
    supporting containsIgnoreCase) — a match is never missed regardless of
    how many issues exist beyond `limit`.
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
        # filter_parts/filter_var_types/variables are built together, one
        # entry per active filter, so filter_var_types's GraphQL variable
        # declarations always match variables exactly -- no separate
        # dict keyed by variable name, so there's nothing to fall out of
        # sync with what's actually in `variables` (a prior version built
        # this signature by iterating `variables` itself, which broke if
        # any key was inserted before this block in a future edit).
        filter_parts = []
        filter_var_types = []
        variables: dict[str, Any] = {}
        if team_id:
            filter_parts.append("team: { id: { eq: $teamId } }")
            filter_var_types.append("$teamId: ID!")
            variables["teamId"] = _resolve_team_uuid(team_id)
        if assignee_id:
            filter_parts.append("assignee: { id: { eq: $assigneeId } }")
            filter_var_types.append("$assigneeId: ID!")
            variables["assigneeId"] = assignee_id
        if state_type:
            filter_parts.append("state: { type: { eq: $stateType } }")
            filter_var_types.append("$stateType: String!")
            variables["stateType"] = state_type
        needle = query.strip()
        if needle:
            # IssueFilter.title is a StringComparator supporting
            # containsIgnoreCase -- confirmed against Linear's own GraphQL
            # schema (github.com/linear/linear packages/sdk/src/schema.graphql).
            filter_parts.append("title: { containsIgnoreCase: $query }")
            filter_var_types.append("$query: String!")
            variables["query"] = needle

        filter_clause = ""
        filter_signature = ""
        if filter_parts:
            filter_clause = ", filter: { " + ", ".join(filter_parts) + " }"
            filter_signature = ", " + ", ".join(filter_var_types)

        variables["first"] = _clamp_limit(limit)

        graphql_query = (
            f"query($first: Int!{filter_signature}) {{"
            f" issues(first: $first{filter_clause}) {{"
            f" nodes {{ {_ISSUE_SUMMARY_FIELDS} }}"
            " pageInfo { hasNextPage } } }"
        )
        data, errors = _graphql(graphql_query, variables)
        issues_data = data.get("issues") or {}
        issues = issues_data.get("nodes") or []
        has_next_page = bool((issues_data.get("pageInfo") or {}).get("hasNextPage"))
        return _success(issues=issues, truncated=has_next_page, _errors=errors)
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
            return _error(
                _mutation_failure_message(
                    "Linear reported the issue was not created", errors
                )
            )
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
    the issue (e.g. to "Done") — an empty string is treated the same as
    leaving it unset (a state can't be unassigned the way assignee_id can).
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
        if state_id:
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
            return _error(
                _mutation_failure_message(
                    "Linear reported the issue was not updated", errors
                )
            )
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
            return _error(
                _mutation_failure_message(
                    "Linear reported the comment was not created", errors
                )
            )
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
        project_selection = "id name status { id name type } progress url"
        if team_id:
            projects, truncated, errors = _team_scoped_list(
                team_id, "projects", max_results, project_selection
            )
        else:
            data, errors = _graphql(
                f"query($first: Int!) {{ projects(first: $first) {{ nodes {{ {project_selection} }}"
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
