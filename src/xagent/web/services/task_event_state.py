"""Capture task control identity before crossing an event transport."""

from copy import deepcopy
from typing import Any

from .task_execution_controller import task_execution_controller

_VERSIONED_TASK_EVENT_TYPES = {
    "agent_error",
    "error",
    "task_completed",
    "task_error",
    "task_pause_requested",
    "task_paused",
    "task_resumed",
    "task_started",
    "task_waiting_for_user",
}


def _is_versioned_task_event(message: dict[str, Any]) -> bool:
    message_type = str(message.get("type") or "")
    if message_type in _VERSIONED_TASK_EVENT_TYPES:
        return True
    return (
        message_type == "trace_event"
        and str(
            message.get("event_type")
            or (
                message.get("data", {}).get("event_type")
                if isinstance(message.get("data"), dict)
                else ""
            )
        )
        == "task_info"
    )


def _event_task_id(message: dict[str, Any]) -> int | None:
    candidates = [message.get("task_id")]
    task_data = message.get("task")
    if isinstance(task_data, dict):
        candidates.append(task_data.get("id"))
        candidates.append(task_data.get("task_id"))
    data = message.get("data")
    if isinstance(data, dict):
        candidates.append(data.get("id"))
        candidates.append(data.get("task_id"))
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return int(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _event_task_control_state(message: dict[str, Any]) -> dict[str, Any] | None:
    sources = [message]
    task_data = message.get("task")
    if isinstance(task_data, dict):
        sources.append(task_data)
    data = message.get("data")
    if isinstance(data, dict):
        sources.append(data)

    for source in sources:
        version = source.get("state_version")
        control_state = source.get("control_state")
        status = source.get("status")
        if (
            isinstance(version, int)
            and not isinstance(version, bool)
            and version >= 0
            and isinstance(control_state, str)
            and isinstance(status, str)
            and (isinstance(source.get("run_id"), str) or source.get("run_id") is None)
        ):
            return {
                "run_id": source.get("run_id"),
                "state_version": version,
                "control_state": control_state,
                "status": status,
            }
    return None


def _with_task_control_state_snapshot(
    message: dict[str, Any],
    *,
    task_id: int,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Attach one already-loaded control-state tuple without database I/O."""

    if not _is_versioned_task_event(message):
        return deepcopy(message)
    resolved_state = _event_task_control_state(message) or state
    enriched = deepcopy(message)
    enriched.update(resolved_state)
    enriched["task_id"] = task_id

    if enriched.get("type") == "trace_event":
        data = enriched.get("data")
        enriched["data"] = {
            **(data if isinstance(data, dict) else {}),
            **resolved_state,
        }

    task_data = enriched.get("task")
    if isinstance(task_data, dict):
        enriched["task"] = {
            **task_data,
            **resolved_state,
            "id": task_id,
        }
    return enriched


async def _with_current_task_control_state(
    message: dict[str, Any],
    *,
    fallback_task_id: int | None = None,
) -> dict[str, Any]:
    """Attach one canonical DB state tuple to a state-bearing event.

    Event producers can finish out of order. Preserve a producer-captured
    state tuple when present; otherwise attach the current row snapshot.
    Clients compare the resulting ``run_id`` / ``state_version`` before
    applying the event.
    """

    if not _is_versioned_task_event(message):
        return message
    task_id = _event_task_id(message) or fallback_task_id
    if task_id is None:
        return message
    state = _event_task_control_state(message)
    if state is None:
        snapshot = await task_execution_controller.snapshot(task_id)
        if snapshot is None:
            return message
        state = snapshot.as_dict()
    return _with_task_control_state_snapshot(
        message,
        task_id=task_id,
        state=state,
    )
