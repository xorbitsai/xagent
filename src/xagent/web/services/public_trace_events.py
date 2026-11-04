"""Public trace event normalization for websocket stream consumers."""

from __future__ import annotations

from typing import Any

from sqlalchemy import or_

from ...core.tools.adapters.vibe.connector_runtime import (
    redact_runtime_sensitive_payload,
)
from .client_error_messages import (
    CLIENT_SAFE_TASK_FAILURE,
    model_error_client_projection,
)
from .memory_availability import caller_facing_trace_data
from .trace_event_types import GENERAL_ERROR_EVENT_TYPES

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
GENERAL_ERROR_PUBLIC_FIELDS = (
    "status",
    "execution_id",
    "pattern",
    "stream_message_id",
    "success",
    "turn_id",
)
PATTERN_END_EVENT_TYPES = frozenset(
    {"dag_execute_end", "react_task_end", "task_end_react"}
)
FAILED_PATTERN_END_PUBLIC_FIELDS = (
    "status",
    "success",
    "execution_id",
    "pattern",
    "stream_message_id",
    "turn_id",
    "iteration",
    "task_preview",
    "step_id",
)
FAILED_PATTERN_RESULT_PUBLIC_FIELDS = ("success", "status")

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


# What a guest (widget / share) socket keeps from an LLM trace event: metadata
# and model output. The request messages sent to the model are never forwarded.
PUBLIC_AUDIENCE_LLM_FIELDS = frozenset(
    {
        "model_name",
        "selected_model",
        "task_type",
        "step_id",
        "step_name",
        "dag_step_id",
        "phase",
        "iteration",
        "task_id",
        "execution_id",
        "turn_id",
        "stream_message_id",
        "pattern",
        "status",
        "success",
        "context_messages_count",
        "tools_count",
        "context_tokens",
        "context_threshold",
        "context_window",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_input_tokens",
        "finish_reason",
        "usage_missing",
        "response",
        "error",
        "error_message",
        "error_type",
        # Delegated child-agent attribution: the widget routes child LLM
        # events into the delegated step by these.
        "source",
        "worker_task_id",
        "parent_task_id",
        "agent_id",
        "agent_name",
        "workforce_run_id",
        "workforce_id",
        "workforce_name",
        "worker_member_id",
        "worker_alias",
    }
)


def _is_llm_event_type(event_type: Any) -> bool:
    return isinstance(event_type, str) and (
        event_type.startswith("llm_") or event_type.endswith("_llm")
    )


def _public_audience_trace_data(event_type: Any, data: Any) -> Any:
    if not _is_llm_event_type(event_type):
        return data
    if not isinstance(data, dict):
        return {}
    return {
        key: value for key, value in data.items() if key in PUBLIC_AUDIENCE_LLM_FIELDS
    }


def public_audience_frame(frame: Any) -> Any:
    """Project an outbound websocket frame for a widget / share guest socket.

    Never mutates ``frame``: other audiences receive it unchanged.
    """
    if not isinstance(frame, dict) or frame.get("type") != "trace_event":
        return frame
    return {
        **frame,
        "data": _public_audience_trace_data(frame.get("event_type"), frame.get("data")),
    }


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
    # Public trace surfaces include historical checkpoint rows. Older rows may
    # contain a trusted host resolver's arbitrary diagnostic nested inside the
    # snapshot, so fold it recursively before applying event-specific shaping.
    data = caller_facing_trace_data(data)

    if event_type in TOOL_EVENT_TYPES:
        data = redact_runtime_sensitive_payload(data)
    elif event_type in GENERAL_ERROR_EVENT_TYPES:
        public_data = (
            {key: data[key] for key in GENERAL_ERROR_PUBLIC_FIELDS if key in data}
            if isinstance(data, dict)
            else {}
        )
        projection = (
            model_error_client_projection(data.get("model_error"))
            if isinstance(data, dict)
            else None
        )
        if projection is None:
            public_data["error_message"] = CLIENT_SAFE_TASK_FAILURE
        else:
            public_data.update(projection)
        data = public_data
        event_type = "trace_error"
    elif event_type in PATTERN_END_EVENT_TYPES and isinstance(data, dict):
        result = data.get("result")
        result_is_failure = isinstance(result, dict) and (
            result.get("success") is False
            or result.get("status") == "failed"
            or bool(result.get("error"))
        )
        if (
            data.get("success") is False
            or data.get("status") == "failed"
            or result_is_failure
        ):
            public_data = {
                key: data[key]
                for key in FAILED_PATTERN_END_PUBLIC_FIELDS
                if key in data
            }
            if isinstance(result, dict):
                public_data["result"] = {
                    key: result[key]
                    for key in FAILED_PATTERN_RESULT_PUBLIC_FIELDS
                    if key in result
                }
            data = public_data

    if not is_public_workforce_delegation_summary(event_type, data):
        return event_type, data

    public_event_type = str(data["event_type"])
    return public_event_type, _public_workforce_delegation_data(data)
