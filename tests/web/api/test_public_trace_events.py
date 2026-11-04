from __future__ import annotations

import pytest

from xagent.core.tools.adapters.vibe.connector_runtime import REDACTED_RUNTIME_SECRET
from xagent.web.services.public_trace_events import normalize_public_trace_event


def test_normalize_public_trace_event_redacts_tool_runtime_secrets() -> None:
    event_type, data = normalize_public_trace_event(
        "tool_execution_start",
        {
            "tool_name": "shiftcare",
            "tool_args": {
                "headers": {
                    "Authorization": "Bearer public-stream-token",
                    "X-Account": "6185",
                },
                "connector_runtime": {
                    "secrets": {"authorization": "Bearer nested-token"},
                    "auth_selector": {"resource_owner_key": "xagent:user:owner"},
                },
            },
        },
    )

    assert event_type == "tool_execution_start"
    assert "public-stream-token" not in repr(data)
    assert "nested-token" not in repr(data)
    assert "xagent:user:owner" not in repr(data)
    assert data["tool_args"]["headers"]["Authorization"] == REDACTED_RUNTIME_SECRET
    assert data["tool_args"]["headers"]["X-Account"] == "6185"
    assert (
        data["tool_args"]["connector_runtime"]["auth_selector"]["resource_owner_key"]
        == REDACTED_RUNTIME_SECRET
    )


@pytest.mark.parametrize(
    "event_type",
    ["trace_error", "task_error_general", "step_error_general"],
)
def test_normalize_public_trace_event_redacts_general_failure_diagnostics(
    event_type: str,
) -> None:
    raw_error = "provider token=secret host=/srv/private"

    normalized_event_type, data = normalize_public_trace_event(
        event_type,
        {
            "status": "failed",
            "execution_id": "execution-1730",
            "pattern": "ReActPattern",
            "success": False,
            "error_type": "agent_error",
            "error": raw_error,
            "error_message": raw_error,
            "traceback": f"Traceback: {raw_error}",
            "context": {"messages": [{"content": raw_error}]},
        },
    )

    assert normalized_event_type == "trace_error"
    assert raw_error not in repr(data)
    assert data == {
        "status": "failed",
        "execution_id": "execution-1730",
        "pattern": "ReActPattern",
        "success": False,
        "error_message": "Task execution failed.",
    }


_MODEL_FAILURE_RAW = "Model is decommissioned"
_MODEL_FAILURE_PUBLIC_BASE = {
    "status": "failed",
    "execution_id": "execution-1730",
    "pattern": "ReActPattern",
    "success": False,
}


def _model_failure_event_data(model_error: object) -> dict[str, object]:
    raw_error = "OpenAI API error (403): provider_raw=RAW_MARKER"
    return {
        **_MODEL_FAILURE_PUBLIC_BASE,
        "error_type": "agent_error",
        "error": raw_error,
        "error_message": raw_error,
        "provider_raw": "RAW_MARKER",
        "model_error": model_error,
    }


_MODEL_FAILURE_403 = (
    {
        "kind": "access_denied",
        "status_code": 403,
        "provider_code": "provider_code_4204",
        "message": _MODEL_FAILURE_RAW,
    },
    {
        "error_code": "model_error",
        "error_message": (
            "Model provider call failed (403 provider_code_4204): "
            "Model is decommissioned"
        ),
        "kind": "access_denied",
        "status_code": 403,
        "provider_code": "provider_code_4204",
    },
)
_MODEL_FAILURE_400 = (
    {
        "kind": "bad_request",
        "status_code": 400,
        "provider_code": "invalid_request",
        "message": _MODEL_FAILURE_RAW,
    },
    {"error_message": "Task execution failed."},
)
_MODEL_FAILURE_401 = (
    {
        "kind": "authentication_failed",
        "status_code": 401,
        "provider_code": "invalid_api_key",
        "message": "Incorrect API key provided: sk-test-key-echo",
    },
    {
        "error_code": "model_error",
        "error_message": (
            "Model provider call failed (401 invalid_api_key): "
            "the credential was rejected."
        ),
        "kind": "authentication_failed",
        "status_code": 401,
        "provider_code": "invalid_api_key",
    },
)
_MODEL_FAILURE_TIMEOUT = (
    {
        "kind": "timeout",
        "status_code": None,
        "provider_code": None,
        "message": None,
    },
    {
        "error_code": "model_error",
        "error_message": "Model provider call failed: the request timed out.",
        "kind": "timeout",
        "status_code": None,
        "provider_code": None,
    },
)
_MODEL_FAILURE_MALFORMED = (
    "not a dict",
    {"error_message": "Task execution failed."},
)


@pytest.mark.parametrize(
    "event_type, model_error, expected_extra",
    [
        *(
            pytest.param(event_type, *case, id=f"{event_type}-{name}")
            for event_type in (
                "trace_error",
                "task_error_general",
                "step_error_general",
            )
            for name, case in (
                ("403-with-body", _MODEL_FAILURE_403),
                ("400-keeps-the-generic-shape", _MODEL_FAILURE_400),
            )
        ),
        pytest.param(
            "trace_error", *_MODEL_FAILURE_401, id="trace_error-401-body-withheld"
        ),
        pytest.param("trace_error", *_MODEL_FAILURE_TIMEOUT, id="trace_error-timeout"),
        pytest.param(
            "trace_error", *_MODEL_FAILURE_MALFORMED, id="trace_error-malformed"
        ),
    ],
)
def test_normalize_public_trace_event_projects_model_failures(
    event_type: str,
    model_error: object,
    expected_extra: dict[str, object],
) -> None:
    normalized_event_type, data = normalize_public_trace_event(
        event_type, _model_failure_event_data(model_error)
    )

    assert normalized_event_type == "trace_error"
    assert data == {**_MODEL_FAILURE_PUBLIC_BASE, **expected_extra}
    assert "RAW_MARKER" not in repr(data)
    assert "sk-test-key-echo" not in repr(data)
    assert "OpenAI API error" not in repr(data)


@pytest.mark.parametrize("data", [None, "failed", ["model_error"]])
def test_normalize_public_trace_event_without_a_dict_payload_stays_generic(
    data: object,
) -> None:
    normalized_event_type, public = normalize_public_trace_event("trace_error", data)

    assert normalized_event_type == "trace_error"
    assert public == {"error_message": "Task execution failed."}


@pytest.mark.parametrize(
    "trace_event_type",
    ["task", "step"],
)
def test_live_general_failure_trace_uses_redacted_public_event(
    trace_event_type: str,
) -> None:
    from xagent.core.agent.trace import STEP_ERROR, TASK_ERROR, TraceEvent
    from xagent.web.services.task_event_trace_handler import TaskEventTraceHandler

    raw_error = "live provider token=secret"
    event = TraceEvent(
        TASK_ERROR if trace_event_type == "task" else STEP_ERROR,
        task_id="42",
        step_id="step-1" if trace_event_type == "step" else None,
        data={
            "status": "failed",
            "error_message": raw_error,
            "context": {"messages": [{"content": raw_error}]},
        },
    )

    stream_event = TaskEventTraceHandler(42)._convert_trace_event_to_stream_event(event)

    assert stream_event is not None
    assert stream_event["event_type"] == "trace_error"
    assert raw_error not in repr(stream_event)
    assert stream_event["data"] == {
        "status": "failed",
        "error_message": "Task execution failed.",
    }


@pytest.mark.parametrize("event_type", ["dag_execute_end", "react_task_end"])
def test_failed_pattern_end_redacts_nested_diagnostics(event_type: str) -> None:
    raw_error = "provider token=secret host=/srv/private"

    normalized_event_type, data = normalize_public_trace_event(
        event_type,
        {
            "status": "failed",
            "execution_id": "execution-1730",
            "pattern": "ReActPattern",
            "step_id": "step-1",
            "result": {
                "success": False,
                "status": "failed",
                "error": raw_error,
                "context": {"messages": [{"content": raw_error}]},
            },
        },
    )

    assert normalized_event_type == event_type
    assert raw_error not in repr(data)
    assert data == {
        "status": "failed",
        "execution_id": "execution-1730",
        "pattern": "ReActPattern",
        "step_id": "step-1",
        "result": {"success": False, "status": "failed"},
    }


def test_successful_pattern_end_preserves_result() -> None:
    result = {"success": True, "status": "completed", "output": "safe answer"}

    _, data = normalize_public_trace_event(
        "react_task_end",
        {"status": "completed", "result": result},
    )

    assert data == {"status": "completed", "result": result}


def test_public_checkpoint_trace_folds_nested_memory_reason_without_mutation() -> None:
    raw_reason = "host resolver secret shard eu-3"
    payload = {
        "checkpoint_type": "agent_execution_checkpoint",
        "snapshot": {
            "context": {
                "metadata": {
                    "memory_available": False,
                    "memory_availability_reason": raw_reason,
                }
            }
        },
    }

    event_type, data = normalize_public_trace_event("system_update_general", payload)

    assert event_type == "system_update_general"
    assert raw_reason not in repr(data)
    assert (
        data["snapshot"]["context"]["metadata"]["memory_availability_reason"]
        == "unavailable"
    )
    assert (
        payload["snapshot"]["context"]["metadata"]["memory_availability_reason"]
        == raw_reason
    )


def test_public_trace_folds_non_string_memory_reason_without_failing() -> None:
    _, data = normalize_public_trace_event(
        "task_update_general",
        {"memory_availability_reason": {"unexpected": "shape"}},
    )

    assert data == {"memory_availability_reason": "unavailable"}


def test_top_level_failed_pattern_status_survives_public_normalization() -> None:
    from xagent.web.services.task_agent_execution import _derive_agent_execution_status

    event_type, data = normalize_public_trace_event(
        "dag_execute_end",
        {"success": False, "error": "provider token=secret"},
    )

    assert data == {"success": False}
    assert (
        _derive_agent_execution_status(
            [{"event_type": event_type, "data": data}],
        )
        == "failed"
    )


@pytest.mark.parametrize(
    ("trace_event_type", "expected_event_type"),
    [("dag", "dag_execute_end"), ("react", "react_task_end")],
)
def test_live_failed_pattern_end_redacts_nested_diagnostics(
    trace_event_type: str,
    expected_event_type: str,
) -> None:
    from xagent.core.agent.trace import TASK_END_DAG, TASK_END_REACT, TraceEvent
    from xagent.web.services.task_event_trace_handler import TaskEventTraceHandler

    raw_error = "live pattern token=secret"
    event = TraceEvent(
        TASK_END_DAG if trace_event_type == "dag" else TASK_END_REACT,
        task_id="42",
        data={
            "status": "failed",
            "result": {"success": False, "error": raw_error},
        },
    )

    stream_event = TaskEventTraceHandler(42)._convert_trace_event_to_stream_event(event)

    assert stream_event is not None
    assert stream_event["event_type"] == expected_event_type
    assert raw_error not in repr(stream_event)
    assert stream_event["data"] == {
        "status": "failed",
        "result": {"success": False},
    }


def test_mcp_load_summary_audit_event_is_not_fanned_out() -> None:
    from xagent.core.agent.trace import SYSTEM_INFO, TraceEvent
    from xagent.web.services.task_event_trace_handler import TaskEventTraceHandler

    event = TraceEvent(
        event_type=SYSTEM_INFO,
        task_id="42",
        data={
            "__audit_only__": True,
            "event_type": "mcp_load_summary",
            "requested_servers": ["Gmail"],
            "loaded_servers": [],
            "failures": [{"server_name": "Gmail", "reason": "oauth_token_required"}],
            "successful_tool_count": 0,
        },
    )

    assert TaskEventTraceHandler(42)._convert_trace_event_to_stream_event(event) is None


def test_live_model_failure_trace_carries_the_model_error_projection() -> None:
    from xagent.core.agent.trace import TASK_ERROR, TraceEvent
    from xagent.web.services.task_event_trace_handler import TaskEventTraceHandler

    event = TraceEvent(
        TASK_ERROR,
        task_id="42",
        step_id=None,
        data={
            "status": "failed",
            "error_message": "OpenAI API error (403): provider_raw=RAW_MARKER",
            "model_error": {
                "kind": "access_denied",
                "status_code": 403,
                "provider_code": "provider_code_4204",
                "message": _MODEL_FAILURE_RAW,
            },
        },
    )

    stream_event = TaskEventTraceHandler(42)._convert_trace_event_to_stream_event(event)

    assert stream_event is not None
    assert stream_event["event_type"] == "trace_error"
    assert stream_event["data"]["error_code"] == "model_error"
    assert stream_event["data"]["error_message"] == (
        "Model provider call failed (403 provider_code_4204): Model is decommissioned"
    )
    assert "RAW_MARKER" not in repr(stream_event)
