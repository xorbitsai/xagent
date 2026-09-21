import json
import logging
import os
import re
import uuid
from typing import Any
from urllib.parse import unquote

import requests
from mcp.server.fastmcp import FastMCP

from ....config import (
    MIN_TOOL_MAX_OUTPUT_LENGTH,
    TOOL_MAX_OUTPUT_LENGTH,
    get_tool_max_output_length,
)
from ....core.utils.security import redact_sensitive_text
from ...utils.graphql_errors import truncate_error_text
from .utils import (
    require_clean_identifier,
    setup_proxy_env,
    success_with_capped_dict,
    url_path_id,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("planner-mcp")

setup_proxy_env()

mcp = FastMCP("planner-mcp")

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
DEFAULT_TIMEOUT_SECONDS = 30

# Planner assignment writes require an orderHint for every assigned user.
# Microsoft's create/update examples use " !" as the insertion hint; Graph
# resolves that relative hint to a concrete value. Bucket creation can still
# omit orderHint because Graph generates bucket ordering when it is absent.


class _GraphRequestError(RuntimeError):
    """Graph HTTP failure that retains its status without response parsing."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _EtagConflictError(RuntimeError):
    """Raised when an etag-guarded write loses an optimistic-concurrency
    race (Graph responded 412 Precondition Failed)."""


class _MutationOutcomeIndeterminate(RuntimeError):
    """A mutation may have reached Graph but its outcome cannot be observed."""


def _bounded_json(payloads: tuple[dict[str, Any], ...]) -> str:
    max_output_length = max(MIN_TOOL_MAX_OUTPUT_LENGTH, get_tool_max_output_length())
    for payload in payloads:
        response = json.dumps(payload, ensure_ascii=False)
        if len(response) <= max_output_length:
            return response
    return json.dumps({"status": "error"}, ensure_ascii=False)


def _success(**payload: Any) -> str:
    return _bounded_json(
        ({"status": "success", **payload}, {"status": "success", "truncated": True})
    )


def _error(message: str, *, details: Any = None) -> str:
    message = truncate_error_text(redact_sensitive_text(str(message)))
    payload: dict[str, Any] = {"status": "error", "message": message}
    if details is not None:
        payload["details"] = details
    return _bounded_json(
        (payload, {"status": "error", "message": message[:128]}, {"status": "error"})
    )


def _conflict(message: str) -> str:
    message = truncate_error_text(redact_sensitive_text(str(message)))
    return _bounded_json(
        (
            {"status": "conflict_stale_version", "message": message},
            {"status": "conflict_stale_version"},
        )
    )


def _indeterminate(
    message: str, *, reconciliation: dict[str, Any] | None = None
) -> str:
    message = truncate_error_text(redact_sensitive_text(str(message)))
    reconciliation_payload = (
        {"reconciliation": reconciliation} if reconciliation is not None else {}
    )
    return _bounded_json(
        (
            {
                "status": "indeterminate",
                "retryable": False,
                "mutation_may_have_completed": True,
                "message": message,
                **reconciliation_payload,
            },
            {
                "status": "indeterminate",
                "retryable": False,
                "mutation_may_have_completed": True,
                **reconciliation_payload,
            },
            {"status": "indeterminate", "retryable": False},
            {"status": "indeterminate"},
        )
    )


def _graph_object(result: Any, resource_name: str) -> dict[str, Any]:
    """Validate the object shape promised by a singleton Graph endpoint."""
    if not isinstance(result, dict):
        raise RuntimeError(f"Planner returned an invalid {resource_name} object")
    return result


def _created_object(result: Any, resource_name: str) -> dict[str, Any]:
    """Validate a create response without treating an anomalous 2xx as failure.

    Graph may have committed the mutation even when a proxy or service defect
    replaces the documented response object. Without a stable returned id, the
    caller must inspect the relevant collection instead of retrying blindly.
    """
    if (
        not isinstance(result, dict)
        or not isinstance(result.get("id"), str)
        or not result["id"]
    ):
        raise _MutationOutcomeIndeterminate(
            f"Planner accepted the create request but did not return a valid "
            f"{resource_name} object; do not retry automatically. Read the "
            "relevant collection first to determine whether it was created."
        )
    return result


_LIST_PROJECTION_FIELDS: dict[str, tuple[str, ...]] = {
    "plans": ("id", "title", "owner", "createdDateTime"),
    "buckets": ("id", "name", "planId", "orderHint"),
    "tasks": (
        "id",
        "title",
        "planId",
        "bucketId",
        "percentComplete",
        "priority",
        "startDateTime",
        "dueDateTime",
        "completedDateTime",
        "hasDescription",
        "activeChecklistItemCount",
        "checklistItemCount",
        "referenceCount",
        "previewType",
        "createdDateTime",
    ),
}


def _bounded_list_response(
    list_field: str,
    items: Any,
    *,
    next_link: Any,
    retry_next_link: str | None,
) -> str:
    """Return one complete Graph page without letting the output filter cut JSON.

    The unmodified page is preferred. If it is too large, every item is kept
    but projected to the stable summary fields useful in a list view; task
    assignments become a count and can be fetched in full with
    ``planner_get_task``. If even a minimal identity/title projection cannot
    fit, return a valid error that points back to the current page rather than
    advancing to Graph's next page and silently skipping records.
    """
    if not isinstance(items, list):
        raise RuntimeError("Planner returned a non-list value collection")
    if next_link is not None and not isinstance(next_link, str):
        raise RuntimeError("Planner returned an invalid @odata.nextLink")
    if list_field not in _LIST_PROJECTION_FIELDS:
        raise ValueError(f"unsupported Planner list field: {list_field}")

    max_output_length = max(MIN_TOOL_MAX_OUTPUT_LENGTH, get_tool_max_output_length())

    def _serialize(values: list[Any], *, truncated: bool, mode: str | None) -> str:
        payload: dict[str, Any] = {
            "status": "success",
            list_field: values,
            "next_link": next_link,
            "truncated": truncated,
        }
        if mode is not None:
            payload["projection"] = mode
        return json.dumps(payload, ensure_ascii=False)

    response = _serialize(items, truncated=False, mode=None)
    if len(response) <= max_output_length:
        return response

    fields = _LIST_PROJECTION_FIELDS[list_field]
    projected: list[Any] = []
    for item in items:
        if not isinstance(item, dict):
            projected.append(item)
            continue
        summary = {field: item[field] for field in fields if field in item}
        if list_field == "tasks" and isinstance(item.get("assignments"), dict):
            summary["assignee_count"] = len(item["assignments"])
        projected.append(summary)

    response = _serialize(projected, truncated=True, mode="summary_fields")
    if len(response) <= max_output_length:
        return response

    label_field = "name" if list_field == "buckets" else "title"
    compact: list[Any] = []
    for item in projected:
        if not isinstance(item, dict):
            compact.append(item)
            continue
        identity = {"id": item["id"]} if "id" in item else {}
        label = item.get(label_field)
        if isinstance(label, str):
            identity[label_field] = label[:256]
        compact.append(identity)

    response = _serialize(compact, truncated=True, mode="identity_and_label")
    if len(response) <= max_output_length:
        return response

    error_payloads = (
        {
            "status": "error",
            "message": (
                "This Planner page cannot fit the configured output limit even "
                f"after projection; increase {TOOL_MAX_OUTPUT_LENGTH} and retry the "
                "same page. No records from this page were returned."
            ),
            "retry_next_link": retry_next_link,
            "retry_same_page": True,
        },
        {
            "status": "error",
            "message": "Planner page exceeds the output limit; retry the same page.",
            "retry_next_link": retry_next_link,
        },
        {"status": "error"},
    )
    for payload in error_payloads:
        response = json.dumps(payload, ensure_ascii=False)
        if len(response) <= max_output_length:
            return response
    return json.dumps({"status": "error"}, ensure_ascii=False)


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
    mutation: bool = False,
) -> Any:
    try:
        response = requests.request(
            method=method,
            url=f"{GRAPH_BASE_URL}{path}",
            headers=_graph_headers(extra_headers),
            params=params,
            json=body,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        if mutation:
            raise _MutationOutcomeIndeterminate(
                "Planner mutation outcome is unknown after a transport failure; "
                "do not retry automatically. Read the resource first to determine "
                "whether the change was applied."
            ) from exc
        raise
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        response_text = truncate_error_text(
            redact_sensitive_text(response.text.strip())
        )
        message = truncate_error_text(redact_sensitive_text(str(exc)))
        if response_text:
            message = f"{message} - {response_text}"
        if mutation and response.status_code >= 500:
            raise _MutationOutcomeIndeterminate(
                "Planner mutation outcome is unknown after a server failure; "
                "do not retry automatically. Read the resource first to determine "
                "whether the change was applied."
            ) from exc
        raise _GraphRequestError(message, status_code=response.status_code) from exc

    if response.status_code == 204 or not response.content:
        return {}
    try:
        return response.json()
    except ValueError as exc:
        if mutation:
            raise _MutationOutcomeIndeterminate(
                "Planner accepted the mutation request but returned an unreadable "
                "response; do not retry automatically. Read the resource first to "
                "determine whether the change was applied."
            ) from exc
        raise RuntimeError("Planner returned an unreadable JSON response") from exc


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

    A 412 or 409 response is this mechanism's designed outcome (a
    concurrent edit won the race), not a generic failure, so it's raised
    as _EtagConflictError rather than left as an opaque _GraphRequestError
    -- every caller catches it separately to return a structured
    conflict_stale_version response. Microsoft's own "Planner resource
    versioning" docs state both codes must be handled this way ("client
    apps are expected to handle versioning related error codes 409 and
    412 by reading the latest version of the item and resolving the
    conflicting changes"), not just 412 the way outlook.py's single-code
    precedent (for a different resource type) handles it.
    """
    resolved_etag = etag or _current_etag(path)
    try:
        return _graph_request(
            method,
            path,
            body=body,
            extra_headers={"If-Match": resolved_etag},
            mutation=True,
        )
    except _GraphRequestError as exc:
        if exc.status_code in (409, 412):
            raise _EtagConflictError(
                "This item changed before the update could be applied. Read "
                "it again and retry."
            ) from exc
        raise


_PERCENT_ESCAPE_RE = re.compile(r"%[0-9A-Fa-f]{2}")


def _normalize_percent_escape_case(path: str) -> str:
    """Uppercase the hex digits of every %XX escape in ``path``.

    RFC 3986 percent-encoded octets are case-insensitive (%2F and %2f are
    the same character), so this lets a next_link that differs from
    default_path only in hex-digit casing compare equal. Unlike
    ``unquote()``, this never turns an escaped character back into its
    literal form -- %2F stays %2F rather than becoming a real "/" -- so it
    can't blur the boundary between a genuine path separator and a
    percent-encoded one sitting inside a single segment.
    """
    return _PERCENT_ESCAPE_RE.sub(lambda m: m.group(0).upper(), path)


def _resolve_list_path(default_path: str, next_link: str | None) -> str:
    """Resolve a list tool's request path: the default first-page path, or
    a caller-supplied next_link to continue a previous page.

    next_link must be validated as actually pointing at Graph, and at this
    exact tool's own collection, before being reused as a request path.
    Requiring the GRAPH_BASE_URL prefix alone would still let a forged
    next_link redirect this server-side, bearer-token-carrying request to
    an unrelated Graph endpoint (e.g. /me/messages) that this tool was
    never meant to reach -- the AUTH_TOKEN is shared across every connected
    Microsoft connector, so that redirection could expose data well beyond
    this tool's own scope. A genuine @odata.nextLink always targets the
    exact same collection as the request that produced it, differing only
    in its query string ($skip/$skiptoken), so comparing the path portion
    against default_path accepts every legitimate value while rejecting a
    forged one.

    The comparison normalizes percent-escape hex-digit casing only -- it
    deliberately does not fully decode either side. A caller-supplied
    next_link is untrusted, and requests/urllib3 percent-decode some
    escapes (e.g. %2E -> '.') without re-collapsing dot segments that
    decoding reveals, while leaving others (%2F) encoded; comparing on a
    fully-decoded representation would validate a different string than
    the one actually sent, and could accept dot-segments that decoding
    only reveals after this check has already passed. As defense in
    depth, any decoded dot segment in the candidate path is rejected
    outright, mirroring outlook.py's _next_link_path.
    """
    if next_link is None:
        return default_path
    if not isinstance(next_link, str) or not next_link.startswith(f"{GRAPH_BASE_URL}/"):
        raise ValueError(
            "next_link must be a @odata.nextLink value returned by a previous "
            "call to this tool"
        )
    path = next_link[len(GRAPH_BASE_URL) :]
    candidate = path.split("?", 1)[0]
    if _normalize_percent_escape_case(candidate) != _normalize_percent_escape_case(
        default_path
    ):
        raise ValueError(
            "next_link must be a @odata.nextLink value returned by a previous "
            "call to this tool"
        )
    if any(segment in {".", ".."} for segment in unquote(candidate).split("/")):
        raise ValueError(
            "next_link must be a @odata.nextLink value returned by a previous "
            "call to this tool"
        )
    return path


def _validated_user_ids(user_ids: list[str]) -> list[str]:
    if not isinstance(user_ids, list):
        raise TypeError("user_ids must be a list of strings")
    return [require_clean_identifier(user_id, "user_id") for user_id in user_ids]


def _build_assignments(user_ids: list[str] | None) -> dict[str, Any] | None:
    if not user_ids:
        return None
    return {
        user_id: {
            "@odata.type": "#microsoft.graph.plannerAssignment",
            "orderHint": " !",
        }
        for user_id in _validated_user_ids(user_ids)
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
        return _bounded_list_response(
            "plans",
            result.get("value", []),
            next_link=result.get("@odata.nextLink"),
            retry_next_link=next_link,
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
        return success_with_capped_dict("plan", _graph_object(result, "plan"))
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
        result = _graph_request("POST", "/planner/plans", body=body, mutation=True)
        return success_with_capped_dict("plan", _created_object(result, "plan"))
    except _MutationOutcomeIndeterminate as e:
        return _indeterminate(
            str(e),
            reconciliation={
                "read_tool": "planner_list_plans",
                "group_id": group_id,
                "match": {"title": title},
            },
        )
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
        return _bounded_list_response(
            "buckets",
            result.get("value", []),
            next_link=result.get("@odata.nextLink"),
            retry_next_link=next_link,
        )
    except Exception as e:
        logger.error("Error listing Planner buckets for plan %s: %s", plan_id, e)
        return _error(str(e))


@mcp.tool()
def planner_create_bucket(plan_id: str, name: str) -> str:
    """Create a new bucket in a Planner plan. Its position among existing
    buckets is assigned by the service."""
    try:
        name = name.strip()
        if not name:
            raise ValueError("name is required")
        body = {
            "name": name,
            "planId": require_clean_identifier(plan_id, "plan_id"),
        }
        result = _graph_request("POST", "/planner/buckets", body=body, mutation=True)
        return success_with_capped_dict("bucket", _created_object(result, "bucket"))
    except _MutationOutcomeIndeterminate as e:
        return _indeterminate(
            str(e),
            reconciliation={
                "read_tool": "planner_list_buckets",
                "plan_id": plan_id,
                "match": {"name": name},
            },
        )
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
        return _bounded_list_response(
            "tasks",
            result.get("value", []),
            next_link=result.get("@odata.nextLink"),
            retry_next_link=next_link,
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
        return _bounded_list_response(
            "tasks",
            result.get("value", []),
            next_link=result.get("@odata.nextLink"),
            retry_next_link=next_link,
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
        return success_with_capped_dict("task", _graph_object(result, "task"))
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
        body: dict[str, Any] = {
            "planId": require_clean_identifier(plan_id, "plan_id"),
            "title": title,
        }
        if bucket_id is not None:
            body["bucketId"] = require_clean_identifier(bucket_id, "bucket_id")
        if due_date_time is not None:
            due_date_time = due_date_time.strip()
            if not due_date_time:
                raise ValueError("due_date_time cannot be empty")
            body["dueDateTime"] = due_date_time
        if assignee_user_ids is not None and not isinstance(assignee_user_ids, list):
            raise TypeError("assignee_user_ids must be a list of strings")
        assignments = _build_assignments(assignee_user_ids)
        if assignments:
            body["assignments"] = assignments
        result = _graph_request("POST", "/planner/tasks", body=body, mutation=True)
        return success_with_capped_dict("task", _created_object(result, "task"))
    except _MutationOutcomeIndeterminate as e:
        return _indeterminate(
            str(e),
            reconciliation={
                "read_tool": "planner_list_tasks",
                "plan_id": plan_id,
                "match": {
                    "title": title,
                    **({"bucketId": bucket_id} if bucket_id is not None else {}),
                },
            },
        )
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
    timestamps, e.g. "2026-09-30T00:00:00Z"; pass "" for either to clear it.
    Pass "" for bucket_id to unbucket the task. etag is optional -- pass the
    @odata.etag from a recent planner_get_task call on this same task to
    skip an extra lookup; omit it to have one fetched automatically."""
    try:
        body: dict[str, Any] = {}
        if title is not None:
            title = title.strip()
            if not title:
                raise ValueError("title cannot be empty")
            body["title"] = title
        if bucket_id is not None:
            body["bucketId"] = (
                None
                if bucket_id == ""
                else require_clean_identifier(bucket_id, "bucket_id")
            )
        if percent_complete is not None:
            if not 0 <= percent_complete <= 100:
                raise ValueError("percent_complete must be between 0 and 100")
            body["percentComplete"] = percent_complete
        if priority is not None:
            if not 0 <= priority <= 10:
                raise ValueError("priority must be between 0 and 10")
            body["priority"] = priority
        if due_date_time is not None:
            if due_date_time == "":
                body["dueDateTime"] = None
            else:
                due_date_time = due_date_time.strip()
                if not due_date_time:
                    raise ValueError("due_date_time cannot be blank")
                body["dueDateTime"] = due_date_time
        if start_date_time is not None:
            if start_date_time == "":
                body["startDateTime"] = None
            else:
                start_date_time = start_date_time.strip()
                if not start_date_time:
                    raise ValueError("start_date_time cannot be blank")
                body["startDateTime"] = start_date_time
        if not body:
            raise ValueError("at least one field must be provided to update the task")

        task_path = f"/planner/tasks/{url_path_id(task_id, 'task_id')}"
        _etag_guarded_write(task_path, "PATCH", body=body, etag=etag)
        return _success(message="Task updated successfully")
    except _MutationOutcomeIndeterminate as e:
        return _indeterminate(
            str(e),
            reconciliation={"read_tool": "planner_get_task", "task_id": task_id},
        )
    except _EtagConflictError as e:
        return _conflict(str(e))
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
        if not isinstance(user_ids, list):
            raise TypeError("user_ids must be a list of strings")
        if not user_ids:
            raise ValueError("user_ids is required")
        task_path = f"/planner/tasks/{url_path_id(task_id, 'task_id')}"
        body = {"assignments": _build_assignments(user_ids)}
        _etag_guarded_write(task_path, "PATCH", body=body, etag=etag)
        return _success(message="Task assigned successfully")
    except _MutationOutcomeIndeterminate as e:
        return _indeterminate(
            str(e),
            reconciliation={"read_tool": "planner_get_task", "task_id": task_id},
        )
    except _EtagConflictError as e:
        return _conflict(str(e))
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
        if not isinstance(user_ids, list):
            raise TypeError("user_ids must be a list of strings")
        if not user_ids:
            raise ValueError("user_ids is required")
        task_path = f"/planner/tasks/{url_path_id(task_id, 'task_id')}"
        body = {"assignments": dict.fromkeys(_validated_user_ids(user_ids))}
        _etag_guarded_write(task_path, "PATCH", body=body, etag=etag)
        return _success(message="Task unassigned successfully")
    except _MutationOutcomeIndeterminate as e:
        return _indeterminate(
            str(e),
            reconciliation={"read_tool": "planner_get_task", "task_id": task_id},
        )
    except _EtagConflictError as e:
        return _conflict(str(e))
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
    except _MutationOutcomeIndeterminate as e:
        return _indeterminate(
            str(e),
            reconciliation={"read_tool": "planner_get_task", "task_id": task_id},
        )
    except _EtagConflictError as e:
        return _conflict(str(e))
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
        return success_with_capped_dict(
            "details", _graph_object(result, "task details")
        )
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
    except _MutationOutcomeIndeterminate as e:
        return _indeterminate(
            str(e),
            reconciliation={
                "read_tool": "planner_get_task_details",
                "task_id": task_id,
            },
        )
    except _EtagConflictError as e:
        return _conflict(str(e))
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
    item_id: str | None = None
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
    except _MutationOutcomeIndeterminate as e:
        return _indeterminate(
            str(e),
            reconciliation={
                "read_tool": "planner_get_task_details",
                "task_id": task_id,
                "checklist_item_id": item_id,
            },
        )
    except _EtagConflictError as e:
        return _conflict(str(e))
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
                require_clean_identifier(item_id, "item_id"): {
                    "@odata.type": "microsoft.graph.plannerChecklistItem",
                    "isChecked": is_checked,
                }
            }
        }
        _etag_guarded_write(details_path, "PATCH", body=body, etag=etag)
        return _success(message="Checklist item updated successfully")
    except _MutationOutcomeIndeterminate as e:
        return _indeterminate(
            str(e),
            reconciliation={
                "read_tool": "planner_get_task_details",
                "task_id": task_id,
                "checklist_item_id": item_id,
            },
        )
    except _EtagConflictError as e:
        return _conflict(str(e))
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
        body = {"checklist": {require_clean_identifier(item_id, "item_id"): None}}
        _etag_guarded_write(details_path, "PATCH", body=body, etag=etag)
        return _success(message="Checklist item deleted successfully")
    except _MutationOutcomeIndeterminate as e:
        return _indeterminate(
            str(e),
            reconciliation={
                "read_tool": "planner_get_task_details",
                "task_id": task_id,
                "checklist_item_id": item_id,
            },
        )
    except _EtagConflictError as e:
        return _conflict(str(e))
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
