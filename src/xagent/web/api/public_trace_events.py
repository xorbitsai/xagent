"""Public trace event normalization for websocket stream consumers."""

from __future__ import annotations

from typing import Any

from sqlalchemy import or_

from ...core.tools.adapters.vibe.connector_runtime import (
    redact_runtime_sensitive_payload,
)

TOOL_EVENT_TYPES = frozenset(
    {
        "tool_execution_start",
        "tool_execution_end",
        "tool_execution_failed",
    }
)
WORKFORCE_DELEGATION_EVENT_TYPES = frozenset(
    {
        "workforce_delegation_start",
        "workforce_delegation_end",
        "workforce_delegation_error",
    }
)
WORKFORCE_DELEGATION_INTERNAL_EVENT_TYPE = "task_update_general"
DELEGATED_AGENT_TRACE_SOURCE = "xagent-agent-tool-child"

WORKFORCE_DELEGATION_PUBLIC_FIELDS = (
    "status",
    "agent_id",
    "agent_name",
    "tool_name",
    "workforce_run_id",
    "workforce_id",
    "workforce_name",
    "worker_member_id",
    "worker_alias",
    "worker_task_id",
    "output",
    "output_length",
    "error",
    "file_outputs",
)


def public_task_trace_filter(trace_model: Any) -> Any:
    """Include top-level task traces and delegated child-agent traces."""
    return or_(
        trace_model.build_id.is_(None),
        trace_model.data["source"].as_string() == DELEGATED_AGENT_TRACE_SOURCE,
    )


def is_audit_only_trace_data(data: Any) -> bool:
    """Return True for trace payloads that must stay server-side."""
    return isinstance(data, dict) and data.get("__audit_only__") is True


def is_public_workforce_delegation_summary(event_type: str, data: Any) -> bool:
    """Return True when trace data carries a safe workforce delegation summary."""
    if (
        event_type != WORKFORCE_DELEGATION_INTERNAL_EVENT_TYPE
        or not isinstance(data, dict)
        or is_audit_only_trace_data(data)
    ):
        return False
    payload_event_type = data.get("event_type")
    return (
        isinstance(payload_event_type, str)
        and payload_event_type in WORKFORCE_DELEGATION_EVENT_TYPES
    )


def _public_workforce_delegation_data(data: dict[str, Any]) -> dict[str, Any]:
    public_data = {
        key: data[key] for key in WORKFORCE_DELEGATION_PUBLIC_FIELDS if key in data
    }
    output = public_data.get("output")
    if isinstance(output, str):
        public_data["output"] = output[:2000]
        public_data.setdefault("output_length", len(output))
    return public_data


def normalize_public_trace_event(
    event_type: str,
    data: Any,
) -> tuple[str, Any]:
    """Map internal trace rows to the public websocket event contract.

    Internal workforce delegation summaries are persisted as task updates so
    they do not expand the core trace taxonomy. Public stream consumers should
    still see a top-level workforce_delegation_* event with only safe summary
    fields.
    """
    if event_type in TOOL_EVENT_TYPES:
        data = redact_runtime_sensitive_payload(data)

    if not is_public_workforce_delegation_summary(event_type, data):
        return event_type, data

    public_event_type = str(data["event_type"])
    return public_event_type, _public_workforce_delegation_data(data)
