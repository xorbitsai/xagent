from __future__ import annotations

from xagent.core.agent.trace import (
    TraceAction,
    TraceCategory,
    TraceEvent,
    TraceEventType,
    TraceScope,
)
from xagent.core.agent_v2.checkpoint import CHECKPOINT_EVENT_TYPE, CHECKPOINT_TYPE
from xagent.web.api.websocket import _is_agent_v2_checkpoint_data
from xagent.web.api.ws_trace_handlers import (
    WebSocketTraceHandler,
    get_event_type_mapping,
)


def test_agent_v2_checkpoint_is_not_converted_to_websocket_stream_event() -> None:
    event = TraceEvent(
        CHECKPOINT_EVENT_TYPE,
        task_id="365",
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "execution_id": "365",
            "snapshot": {"label": "dag_before_llm"},
        },
    )

    stream_event = WebSocketTraceHandler(365)._convert_trace_event_to_stream_event(
        event
    )

    assert stream_event is None


def test_action_tool_error_maps_to_tool_execution_failed() -> None:
    event = TraceEvent(
        TraceEventType(TraceScope.ACTION, TraceAction.ERROR, TraceCategory.TOOL),
        task_id="365",
        step_id="default",
        data={"tool_name": "execute_python_code", "error_message": "failed"},
    )

    assert get_event_type_mapping(event) == "tool_execution_failed"


def test_action_llm_error_maps_to_llm_call_failed() -> None:
    event = TraceEvent(
        TraceEventType(TraceScope.ACTION, TraceAction.ERROR, TraceCategory.LLM),
        task_id="365",
        step_id="365",
        data={"error_message": "read timed out"},
    )

    assert get_event_type_mapping(event) == "llm_call_failed"


def test_historical_stream_identifies_agent_v2_checkpoint_payload() -> None:
    assert _is_agent_v2_checkpoint_data(
        {
            "checkpoint_type": CHECKPOINT_TYPE,
            "execution_id": "365",
            "snapshot": {"label": "dag_before_llm"},
        }
    )
    assert _is_agent_v2_checkpoint_data(
        {
            "type": "checkpoint",
            "execution_id": "365",
            "pattern_state": {"status": "running"},
            "context": {"messages": []},
        }
    )
    assert not _is_agent_v2_checkpoint_data({"event": "ai_message"})
