import json
import logging
import os
import uuid
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

from .utils import setup_proxy_env, url_path_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("planner-mcp")

setup_proxy_env()

mcp = FastMCP("planner-mcp")

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
DEFAULT_TIMEOUT_SECONDS = 30

# Microsoft's own Planner API examples for every "new item, let Planner
# place it" case (create bucket, create task, add an assignment) use this
# exact literal as the orderHint -- it is documented as a minimal valid
# value under "Using order hints in Planner", not an app-specific choice.
_DEFAULT_ORDER_HINT = " !"


class _GraphRequestError(RuntimeError):
    """Graph HTTP failure that retains its status without response parsing."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str, *, details: Any = None) -> str:
    payload: dict[str, Any] = {"status": "error", "message": message}
    if details is not None:
        payload["details"] = details
    return json.dumps(payload, ensure_ascii=False)


def _graph_headers(extra_headers: dict[str, str] | None = None) -> dict[str, str]:
    token = os.environ.get("AUTH_TOKEN")
    if not token:
        raise ValueError("AUTH_TOKEN environment variable is missing")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    return headers


def _graph_request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> Any:
    response = requests.request(
        method=method,
        url=f"{GRAPH_BASE_URL}{path}",
        headers=_graph_headers(extra_headers),
        params=params,
        json=body,
        timeout=timeout,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        response_text = response.text.strip()
        message = str(exc)
        if response_text:
            message = f"{message} - {response_text}"
        raise _GraphRequestError(message, status_code=response.status_code) from exc

    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


def _current_etag(path: str) -> str:
    """Fetch a Planner resource's current @odata.etag for a subsequent
    If-Match-guarded PATCH/DELETE.

    Every Planner write (PATCH/DELETE) requires the caller's last-known
    etag in an If-Match header -- Planner versions all its resources this
    way rather than with a simple last-write-wins model, and Microsoft's
    own docs describe reading the latest etag as the expected recovery
    path for a 409/412 conflict. An MCP tool caller (an LLM agent) has no
    reliable way to carry an etag across separate tool calls the way a
    stateful UI client would, so each update/delete tool here fetches a
    fresh etag immediately before writing rather than requiring the caller
    to supply one. This narrows, but does not eliminate, the race against a
    concurrent edit; a 409/412 from the subsequent write still surfaces to
    the caller as an error rather than being silently retried, since
    retrying automatically could silently overwrite a real concurrent
    change.
    """
    result = _graph_request("GET", path)
    etag = result.get("@odata.etag") if isinstance(result, dict) else None
    if not isinstance(etag, str) or not etag:
        raise RuntimeError(
            f"Planner did not return an @odata.etag for {path}; cannot safely "
            "apply this update"
        )
    return etag


def _etag_guarded_write(
    path: str,
    method: str,
    *,
    body: dict[str, Any] | None = None,
    etag: str | None = None,
) -> Any:
    """PATCH or DELETE a Planner resource with the required If-Match header.

    If the caller already has a recent etag for this exact resource (e.g.
    from a planner_get_task/planner_get_task_details call earlier in the
    same turn), passing it here skips the extra _current_etag GET. A stale
    supplied etag is no less safe than a freshly-fetched one that loses a
    race immediately after being read -- either way Planner rejects the
    write with 412 rather than silently applying it, so this is a pure
    latency optimization, not a weaker safety guarantee. When omitted, a
    fresh etag is fetched immediately before the write, as before.
    """
    resolved_etag = etag or _current_etag(path)
    return _graph_request(
        method, path, body=body, extra_headers={"If-Match": resolved_etag}
    )


def _resolve_list_path(default_path: str, next_link: str | None) -> str:
    """Resolve a list tool's request path: the default first-page path, or
    a caller-supplied next_link to continue a previous page.

    next_link must be validated as actually pointing at Graph before being
    reused as a request path -- accepting an arbitrary caller-supplied URL
    here and handing it to _graph_request would let a forged next_link
    redirect this server-side, bearer-token-carrying request to a
    different host (Graph's own @odata.nextLink is always same-origin with
    GRAPH_BASE_URL, so requiring that exact prefix accepts every
    legitimate value while rejecting a forged one).
    """
    if next_link is None:
        return default_path
    if not isinstance(next_link, str) or not next_link.startswith(f"{GRAPH_BASE_URL}/"):
        raise ValueError(
            "next_link must be a @odata.nextLink value returned by a previous "
            "call to this tool"
        )
    return next_link[len(GRAPH_BASE_URL) :]


def _build_assignments(user_ids: list[str] | None) -> dict[str, Any] | None:
    if not user_ids:
        return None
    return {
        user_id: {
            "@odata.type": "#microsoft.graph.plannerAssignment",
            "orderHint": _DEFAULT_ORDER_HINT,
        }
        for user_id in user_ids
    }


@mcp.tool()
def planner_list_plans(group_id: str, next_link: str | None = None) -> str:
    """List the Planner plans owned by a Microsoft 365 group.

    next_link is optional -- pass the next_link value from a previous
    response to fetch the next page instead of the first."""
    try:
        path = _resolve_list_path(
            f"/groups/{url_path_id(group_id, 'group_id')}/planner/plans", next_link
        )
        result = _graph_request("GET", path)
        return _success(
            plans=result.get("value", []), next_link=result.get("@odata.nextLink")
        )
    except Exception as e:
        logger.error("Error listing Planner plans for group %s: %s", group_id, e)
        return _error(str(e))


@mcp.tool()
def planner_get_plan(plan_id: str) -> str:
    """Get a Planner plan's details by id."""
    try:
        result = _graph_request(
            "GET", f"/planner/plans/{url_path_id(plan_id, 'plan_id')}"
        )
        return _success(plan=result)
    except Exception as e:
        logger.error("Error getting Planner plan %s: %s", plan_id, e)
        return _error(str(e))


@mcp.tool()
def planner_create_plan(group_id: str, title: str) -> str:
    """Create a new Planner plan owned by a Microsoft 365 group.

    The signed-in user must already be a member of the group (Graph
    enforces this; it is not merely a display requirement)."""
    try:
        title = title.strip()
        if not title:
            raise ValueError("title is required")
        body = {
            "container": {
                "url": f"{GRAPH_BASE_URL}/groups/{url_path_id(group_id, 'group_id')}"
            },
            "title": title,
        }
        result = _graph_request("POST", "/planner/plans", body=body)
        return _success(plan=result)
    except Exception as e:
        logger.error("Error creating Planner plan for group %s: %s", group_id, e)
        return _error(str(e))


@mcp.tool()
def planner_list_buckets(plan_id: str, next_link: str | None = None) -> str:
    """List the buckets (task-board columns) in a Planner plan.

    next_link is optional -- pass the next_link value from a previous
    response to fetch the next page instead of the first."""
    try:
        path = _resolve_list_path(
            f"/planner/plans/{url_path_id(plan_id, 'plan_id')}/buckets", next_link
        )
        result = _graph_request("GET", path)
        return _success(
            buckets=result.get("value", []), next_link=result.get("@odata.nextLink")
        )
    except Exception as e:
        logger.error("Error listing Planner buckets for plan %s: %s", plan_id, e)
        return _error(str(e))


@mcp.tool()
def planner_create_bucket(plan_id: str, name: str) -> str:
    """Create a new bucket in a Planner plan, added after the existing buckets."""
    try:
        name = name.strip()
        if not name:
            raise ValueError("name is required")
        body = {"name": name, "planId": plan_id, "orderHint": _DEFAULT_ORDER_HINT}
        result = _graph_request("POST", "/planner/buckets", body=body)
        return _success(bucket=result)
    except Exception as e:
        logger.error("Error creating Planner bucket in plan %s: %s", plan_id, e)
        return _error(str(e))


@mcp.tool()
def planner_list_tasks(plan_id: str, next_link: str | None = None) -> str:
    """List the tasks in a Planner plan.

    next_link is optional -- pass the next_link value from a previous
    response to fetch the next page instead of the first."""
    try:
        path = _resolve_list_path(
            f"/planner/plans/{url_path_id(plan_id, 'plan_id')}/tasks", next_link
        )
        result = _graph_request("GET", path)
        return _success(
            tasks=result.get("value", []), next_link=result.get("@odata.nextLink")
        )
    except Exception as e:
        logger.error("Error listing Planner tasks for plan %s: %s", plan_id, e)
        return _error(str(e))


@mcp.tool()
def planner_list_my_tasks(next_link: str | None = None) -> str:
    """List the Planner tasks assigned to the signed-in user, across all plans.

    next_link is optional -- pass the next_link value from a previous
    response to fetch the next page instead of the first."""
    try:
        path = _resolve_list_path("/me/planner/tasks", next_link)
        result = _graph_request("GET", path)
        return _success(
            tasks=result.get("value", []), next_link=result.get("@odata.nextLink")
        )
    except Exception as e:
        logger.error("Error listing the signed-in user's Planner tasks: %s", e)
        return _error(str(e))


@mcp.tool()
def planner_get_task(task_id: str) -> str:
    """Get a Planner task's basic properties (title, bucket, dates,
    assignments, percent complete) by id. Use planner_get_task_details for
    its description and checklist."""
    try:
        result = _graph_request(
            "GET", f"/planner/tasks/{url_path_id(task_id, 'task_id')}"
        )
        return _success(task=result)
    except Exception as e:
        logger.error("Error getting Planner task %s: %s", task_id, e)
        return _error(str(e))


@mcp.tool()
def planner_create_task(
    plan_id: str,
    title: str,
    bucket_id: str | None = None,
    assignee_user_ids: list[str] | None = None,
    due_date_time: str | None = None,
) -> str:
    """Create a new Planner task in a plan.

    bucket_id, if given, must be a bucket already in this plan (from
    planner_list_buckets); omit it to leave the task unbucketed.
    assignee_user_ids, if given, is a list of Azure AD user ids to assign
    the task to. due_date_time, if given, must be an RFC3339/ISO 8601 UTC
    timestamp, e.g. "2026-09-30T00:00:00Z"."""
    try:
        title = title.strip()
        if not title:
            raise ValueError("title is required")
        body: dict[str, Any] = {"planId": plan_id, "title": title}
        if bucket_id:
            body["bucketId"] = bucket_id
        if due_date_time:
            body["dueDateTime"] = due_date_time
        assignments = _build_assignments(assignee_user_ids)
        if assignments:
            body["assignments"] = assignments
        result = _graph_request("POST", "/planner/tasks", body=body)
        return _success(task=result)
    except Exception as e:
        logger.error("Error creating Planner task in plan %s: %s", plan_id, e)
        return _error(str(e))


@mcp.tool()
def planner_update_task(
    task_id: str,
    title: str | None = None,
    bucket_id: str | None = None,
    percent_complete: int | None = None,
    priority: int | None = None,
    due_date_time: str | None = None,
    start_date_time: str | None = None,
    etag: str | None = None,
) -> str:
    """Update a Planner task's basic properties. Only the fields provided
    are changed; omitted fields keep their current value.

    percent_complete is 0-100 (100 marks the task completed). priority is
    0-10 (Planner maps 0-1 to "urgent", 2-4 "important", 5-7 "medium", 8-10
    "low"). due_date_time/start_date_time must be RFC3339/ISO 8601 UTC
    timestamps, e.g. "2026-09-30T00:00:00Z". etag is optional -- pass the
    @odata.etag from a recent planner_get_task call on this same task to
    skip an extra lookup; omit it to have one fetched automatically."""
    try:
        body: dict[str, Any] = {}
        if title is not None:
            body["title"] = title
        if bucket_id is not None:
            body["bucketId"] = bucket_id
        if percent_complete is not None:
            if not 0 <= percent_complete <= 100:
                raise ValueError("percent_complete must be between 0 and 100")
            body["percentComplete"] = percent_complete
        if priority is not None:
            if not 0 <= priority <= 10:
                raise ValueError("priority must be between 0 and 10")
            body["priority"] = priority
        if due_date_time is not None:
            body["dueDateTime"] = due_date_time
        if start_date_time is not None:
            body["startDateTime"] = start_date_time
        if not body:
            raise ValueError("at least one field must be provided to update the task")

        task_path = f"/planner/tasks/{url_path_id(task_id, 'task_id')}"
        _etag_guarded_write(task_path, "PATCH", body=body, etag=etag)
        return _success(message="Task updated successfully")
    except Exception as e:
        logger.error("Error updating Planner task %s: %s", task_id, e)
        return _error(str(e))


@mcp.tool()
def planner_assign_task(
    task_id: str, user_ids: list[str], etag: str | None = None
) -> str:
    """Assign a Planner task to one or more users, in addition to any
    existing assignees. To unassign a user, use planner_unassign_task.
    etag is optional -- see planner_update_task."""
    try:
        if not user_ids:
            raise ValueError("user_ids is required")
        task_path = f"/planner/tasks/{url_path_id(task_id, 'task_id')}"
        body = {"assignments": _build_assignments(user_ids)}
        _etag_guarded_write(task_path, "PATCH", body=body, etag=etag)
        return _success(message="Task assigned successfully")
    except Exception as e:
        logger.error("Error assigning Planner task %s: %s", task_id, e)
        return _error(str(e))


@mcp.tool()
def planner_unassign_task(
    task_id: str, user_ids: list[str], etag: str | None = None
) -> str:
    """Remove one or more users from a Planner task's assignments. etag is
    optional -- see planner_update_task."""
    try:
        if not user_ids:
            raise ValueError("user_ids is required")
        task_path = f"/planner/tasks/{url_path_id(task_id, 'task_id')}"
        body = {"assignments": dict.fromkeys(user_ids)}
        _etag_guarded_write(task_path, "PATCH", body=body, etag=etag)
        return _success(message="Task unassigned successfully")
    except Exception as e:
        logger.error("Error unassigning Planner task %s: %s", task_id, e)
        return _error(str(e))


@mcp.tool()
def planner_delete_task(task_id: str, etag: str | None = None) -> str:
    """Delete a Planner task. etag is optional -- see planner_update_task."""
    try:
        task_path = f"/planner/tasks/{url_path_id(task_id, 'task_id')}"
        _etag_guarded_write(task_path, "DELETE", etag=etag)
        return _success(message="Task deleted successfully")
    except Exception as e:
        logger.error("Error deleting Planner task %s: %s", task_id, e)
        return _error(str(e))


@mcp.tool()
def planner_get_task_details(task_id: str) -> str:
    """Get a Planner task's description and checklist items."""
    try:
        result = _graph_request(
            "GET", f"/planner/tasks/{url_path_id(task_id, 'task_id')}/details"
        )
        return _success(details=result)
    except Exception as e:
        logger.error("Error getting Planner task details for %s: %s", task_id, e)
        return _error(str(e))


@mcp.tool()
def planner_update_task_description(
    task_id: str, description: str, etag: str | None = None
) -> str:
    """Set a Planner task's description (replaces the existing description).
    etag is optional -- see planner_update_task; note this is the task
    *details* object's own etag, distinct from the task's own etag."""
    try:
        details_path = f"/planner/tasks/{url_path_id(task_id, 'task_id')}/details"
        _etag_guarded_write(
            details_path, "PATCH", body={"description": description}, etag=etag
        )
        return _success(message="Task description updated successfully")
    except Exception as e:
        logger.error("Error updating Planner task description for %s: %s", task_id, e)
        return _error(str(e))


@mcp.tool()
def planner_add_checklist_item(
    task_id: str, title: str, etag: str | None = None
) -> str:
    """Add a checklist item to a Planner task. Returns the new item's id
    (needed by planner_set_checklist_item_checked/planner_delete_checklist_item).
    etag is optional -- see planner_update_task_description."""
    try:
        title = title.strip()
        if not title:
            raise ValueError("title is required")
        details_path = f"/planner/tasks/{url_path_id(task_id, 'task_id')}/details"
        item_id = str(uuid.uuid4())
        body = {
            "checklist": {
                item_id: {
                    "@odata.type": "microsoft.graph.plannerChecklistItem",
                    "title": title,
                    "isChecked": False,
                }
            }
        }
        _etag_guarded_write(details_path, "PATCH", body=body, etag=etag)
        return _success(checklist_item_id=item_id)
    except Exception as e:
        logger.error("Error adding checklist item to Planner task %s: %s", task_id, e)
        return _error(str(e))


@mcp.tool()
def planner_set_checklist_item_checked(
    task_id: str, item_id: str, is_checked: bool, etag: str | None = None
) -> str:
    """Check or uncheck a Planner task's checklist item by id (from
    planner_add_checklist_item or planner_get_task_details). etag is
    optional -- see planner_update_task_description."""
    try:
        details_path = f"/planner/tasks/{url_path_id(task_id, 'task_id')}/details"
        body = {
            "checklist": {
                item_id: {
                    "@odata.type": "microsoft.graph.plannerChecklistItem",
                    "isChecked": is_checked,
                }
            }
        }
        _etag_guarded_write(details_path, "PATCH", body=body, etag=etag)
        return _success(message="Checklist item updated successfully")
    except Exception as e:
        logger.error(
            "Error updating checklist item %s on Planner task %s: %s",
            item_id,
            task_id,
            e,
        )
        return _error(str(e))


@mcp.tool()
def planner_delete_checklist_item(
    task_id: str, item_id: str, etag: str | None = None
) -> str:
    """Remove a checklist item from a Planner task by id. etag is optional
    -- see planner_update_task_description."""
    try:
        details_path = f"/planner/tasks/{url_path_id(task_id, 'task_id')}/details"
        body = {"checklist": {item_id: None}}
        _etag_guarded_write(details_path, "PATCH", body=body, etag=etag)
        return _success(message="Checklist item deleted successfully")
    except Exception as e:
        logger.error(
            "Error deleting checklist item %s from Planner task %s: %s",
            item_id,
            task_id,
            e,
        )
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
