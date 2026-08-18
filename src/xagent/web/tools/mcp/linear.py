import json
import logging
import os
import re
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
# Matches zoom.py's convention: an error body that isn't the expected
# {"errors": [...]} GraphQL shape (e.g. an HTML gateway error page) must not
# be forwarded to the LLM/logs verbatim and unbounded.
MAX_ERROR_RESPONSE_TEXT_CHARS = 1000

_UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


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


def _graphql(query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run one GraphQL query/mutation against Linear's single API endpoint.

    Linear answers auth/rate-limit failures with a non-200 status (body
    shape not guaranteed), but schema/validation errors with HTTP 200 and a
    top-level "errors" array — both are checked, in that order.
    """
    response = requests.post(
        LINEAR_GRAPHQL_URL,
        headers=_headers(),
        json={"query": query, "variables": variables or {}},
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    if response.status_code >= 400:
        detail: str | None = None
        try:
            payload = response.json()
            if isinstance(payload, dict) and payload.get("errors"):
                detail = _graphql_errors_message(payload["errors"])
        except ValueError:
            pass
        if detail is None:
            detail = response.text.strip()
            if len(detail) > MAX_ERROR_RESPONSE_TEXT_CHARS:
                detail = detail[:MAX_ERROR_RESPONSE_TEXT_CHARS] + "... [truncated]"
        raise RuntimeError(
            f"Linear API error (status {response.status_code}): {detail}"
        )

    payload = response.json()
    errors = payload.get("errors")
    if errors:
        raise RuntimeError(_graphql_errors_message(errors))
    return payload.get("data") or {}


def _resolve_issue_uuid(issue_id: str) -> str:
    """Resolve issue_id to its real UUID.

    The top-level `issue(id: ...)` query field (used by linear_get_issue,
    linear_list_comments, etc.) accepts either an issue's UUID or its
    human-readable identifier (e.g. "ENG-123") — but mutation input types
    that take an issueId (e.g. CommentCreateInput) only accept the UUID.
    Callers of those mutations must resolve a human-readable identifier
    first; an already-UUID issue_id is returned unchanged without a
    round-trip.
    """
    if _UUID_PATTERN.match(issue_id):
        return issue_id
    data = _graphql("query($id: String!) { issue(id: $id) { id } }", {"id": issue_id})
    issue = data.get("issue")
    if not issue:
        raise ValueError(f"Issue '{issue_id}' not found")
    return str(issue["id"])


@mcp.tool()
def linear_get_current_user() -> str:
    """
    Get the profile of the Linear account this connector is authenticated as
    (id, name, email, display name). Use this for "my account" / "who am I"
    requests instead of asking the user for their Linear user id.
    """
    try:
        data = _graphql("query { viewer { id name email displayName admin } }")
        return _success(user=data.get("viewer") or {})
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
        data = _graphql(
            "query($first: Int!) { teams(first: $first) { nodes { id key name } } }",
            {"first": _clamp_limit(limit)},
        )
        teams = (data.get("teams") or {}).get("nodes") or []
        return _success(teams=teams)
    except Exception as e:
        logger.error(f"Error listing Linear teams: {e}")
        return _error(str(e))


@mcp.tool()
def linear_list_workflow_states(team_id: str) -> str:
    """
    List a team's workflow states (e.g. "Todo", "In Progress", "Done") — id,
    name, and type. Resolve a state name to an id here before passing
    state_id to linear_update_issue.
    team_id: a team id or key from linear_list_teams.
    """
    try:
        data = _graphql(
            "query($teamId: String!) { team(id: $teamId) { states(first: 100)"
            " { nodes { id name type position } } } }",
            {"teamId": team_id},
        )
        team = data.get("team") or {}
        states = (team.get("states") or {}).get("nodes") or []
        return _success(states=states)
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
        data = _graphql(
            "query($teamId: String!, $first: Int!) { team(id: $teamId) {"
            " labels(first: $first) { nodes { id name color } } } }",
            {"teamId": team_id, "first": _clamp_limit(limit)},
        )
        team = data.get("team") or {}
        labels = (team.get("labels") or {}).get("nodes") or []
        return _success(labels=labels)
    except Exception as e:
        logger.error(f"Error listing Linear labels for team {team_id}: {e}")
        return _error(str(e))


@mcp.tool()
def linear_search_users(query: str, limit: int = 20) -> str:
    """
    Search workspace members by name or email — id, name, email. Resolve a
    person to an id here before passing assignee_id to linear_create_issue
    or linear_update_issue.
    """
    try:
        data = _graphql(
            "query($first: Int!) { users(first: $first) { nodes { id name"
            " email active } } }",
            {"first": MAX_LIMIT},
        )
        needle = query.strip().lower()
        all_users = (data.get("users") or {}).get("nodes") or []
        if needle:
            matches = [
                user
                for user in all_users
                if needle in str(user.get("name") or "").lower()
                or needle in str(user.get("email") or "").lower()
            ]
        else:
            matches = all_users
        max_matches = _clamp_limit(limit)
        return _success(
            users=matches[:max_matches], truncated=len(matches) > max_matches
        )
    except Exception as e:
        logger.error(f"Error searching Linear users for '{query}': {e}")
        return _error(str(e))


_ISSUE_FIELDS = (
    "id identifier title description priority url createdAt updatedAt"
    " state { id name type } assignee { id name email }"
    " team { id key name } labels { nodes { id name } }"
)


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
    client-side over the fetched page — a query with matches outside the
    first page of results, bounded by limit, will be missed).
    team_id: optional team id/key from linear_list_teams.
    assignee_id: optional user id from linear_search_users.
    state_type: optional workflow state type to filter by — one of
    "triage", "backlog", "unstarted", "started", "completed", "canceled".
    limit: max issues to return (default 20, capped at 100).
    """
    try:
        filter_parts = []
        variables: dict[str, Any] = {}
        if team_id:
            filter_parts.append("team: { id: { eq: $teamId } }")
            variables["teamId"] = team_id
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
        variables["first"] = max_results if not query else MAX_LIMIT

        graphql_query = (
            f"query($first: Int!{filter_signature}) {{ issues(first: $first"
            f"{filter_clause}) {{ nodes {{ {_ISSUE_FIELDS} }}"
            " pageInfo { hasNextPage } } }"
        )
        data = _graphql(graphql_query, variables)
        issues_data = data.get("issues") or {}
        issues = issues_data.get("nodes") or []
        has_next_page = bool((issues_data.get("pageInfo") or {}).get("hasNextPage"))

        needle = query.strip().lower()
        if needle:
            issues = [
                issue
                for issue in issues
                if needle in str(issue.get("title") or "").lower()
            ]
            # The unfiltered fetch (up to MAX_LIMIT) may itself contain more
            # title matches than max_results, independent of whether the
            # server has further pages beyond that fetch.
            truncated = has_next_page or len(issues) > max_results
        else:
            # With no client-side filter, `issues` is bounded exactly to
            # max_results by the `first: max_results` fetch above, so
            # len(issues) > max_results can never be true — hasNextPage from
            # the server is the only real signal of more results.
            truncated = has_next_page
        return _success(issues=issues[:max_results], truncated=truncated)
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
        data = _graphql(
            f"query($id: String!) {{ issue(id: $id) {{ {_ISSUE_FIELDS} }} }}",
            {"id": issue_id},
        )
        issue = data.get("issue")
        if not issue:
            return _error(f"Issue '{issue_id}' not found")
        return _success(issue=issue)
    except Exception as e:
        logger.error(f"Error fetching Linear issue {issue_id}: {e}")
        return _error(str(e))


@mcp.tool()
def linear_create_issue(
    team_id: str,
    title: str,
    description: str = "",
    assignee_id: str = "",
    priority: int = 0,
    label_ids: list[str] | None = None,
) -> str:
    """
    Create a new issue.
    team_id: the target team's id or key, from linear_list_teams.
    title: the issue title.
    description: optional body (Markdown supported).
    assignee_id: optional user id from linear_search_users.
    priority: 0 (no priority), 1 (urgent), 2 (high), 3 (normal), 4 (low).
    label_ids: optional label ids from linear_list_labels.
    """
    try:
        issue_input: dict[str, Any] = {"teamId": team_id, "title": title}
        if description:
            issue_input["description"] = description
        if assignee_id:
            issue_input["assigneeId"] = assignee_id
        if priority:
            issue_input["priority"] = priority
        if label_ids:
            issue_input["labelIds"] = label_ids

        data = _graphql(
            "mutation($input: IssueCreateInput!) { issueCreate(input: $input)"
            f" {{ success issue {{ {_ISSUE_FIELDS} }} }} }}",
            {"input": issue_input},
        )
        result = data.get("issueCreate") or {}
        if not result.get("success"):
            return _error("Linear reported the issue was not created")
        return _success(issue=result.get("issue"))
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
    """
    try:
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
        if not issue_input:
            return _error("No fields provided to update")

        data = _graphql(
            "mutation($id: String!, $input: IssueUpdateInput!) {"
            f" issueUpdate(id: $id, input: $input) {{ success issue {{ {_ISSUE_FIELDS} }} }} }}",
            {"id": issue_id, "input": issue_input},
        )
        result = data.get("issueUpdate") or {}
        if not result.get("success"):
            return _error("Linear reported the issue was not updated")
        return _success(issue=result.get("issue"))
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
        data = _graphql(
            "query($id: String!, $first: Int!) { issue(id: $id) { comments"
            "(first: $first) { nodes { id body createdAt user { id name } } } } }",
            {"id": issue_id, "first": _clamp_limit(limit)},
        )
        issue = data.get("issue")
        if not issue:
            return _error(f"Issue '{issue_id}' not found")
        comments = (issue.get("comments") or {}).get("nodes") or []
        return _success(comments=comments)
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
        issue_uuid = _resolve_issue_uuid(issue_id)
        data = _graphql(
            "mutation($input: CommentCreateInput!) { commentCreate(input: $input)"
            " { success comment { id body createdAt } } }",
            {"input": {"issueId": issue_uuid, "body": body}},
        )
        result = data.get("commentCreate") or {}
        if not result.get("success"):
            return _error("Linear reported the comment was not created")
        return _success(comment=result.get("comment"))
    except Exception as e:
        logger.error(f"Error adding comment to Linear issue {issue_id}: {e}")
        return _error(str(e))


@mcp.tool()
def linear_list_projects(team_id: str = "", limit: int = 50) -> str:
    """
    List projects, optionally scoped to one team — id, name, state, and
    progress.
    team_id: optional team id/key from linear_list_teams; omit to list
    across every team this account can see.
    """
    try:
        max_results = _clamp_limit(limit)
        if team_id:
            data = _graphql(
                "query($teamId: String!, $first: Int!) { team(id: $teamId) {"
                " projects(first: $first) { nodes { id name state progress"
                " url } } } }",
                {"teamId": team_id, "first": max_results},
            )
            team = data.get("team") or {}
            projects = (team.get("projects") or {}).get("nodes") or []
        else:
            data = _graphql(
                "query($first: Int!) { projects(first: $first) { nodes { id"
                " name state progress url } } }",
                {"first": max_results},
            )
            projects = (data.get("projects") or {}).get("nodes") or []
        return _success(projects=projects)
    except Exception as e:
        logger.error(f"Error listing Linear projects: {e}")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
