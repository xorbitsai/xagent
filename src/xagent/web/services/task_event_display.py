"""Public display projections for V2 facts at one committed task horizon.

Business authorization and file materialization remain at the API boundary.
No chat/Trace projection participates in message ownership or event identity.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ...core.agent.trace import normalize_llm_trace_payload
from ..models.task import Task
from ..models.task_execution_event import TaskExecutionEvent
from ..utils.db_timezone import safe_timestamp_to_unix
from .assistant_history_safety import (
    assistant_history_has_safe_ancillary_payload,
    client_safe_assistant_history_content,
)
from .client_error_messages import CLIENT_SAFE_TASK_FAILURE
from .public_trace_events import is_audit_only_trace_data, normalize_public_trace_event
from .task_event_trace_handler import is_agent_checkpoint_data, serialize_trace_data
from .task_execution_event_store import MAX_EXECUTION_EVENT_PAGE_SIZE

# Control/recovery facts are not display data. In particular, avoid fetching
# checkpoint bodies merely to discard them after deserialization.
_INTERNAL_KINDS = frozenset(
    {
        "recovery_state",
        "input_applied",
        "input_delivery_changed",
        "command_accepted",
        "control_state_changed",
        "interaction_requested",
    }
)
_STREAM_KINDS = frozenset(
    {"final_answer_start", "final_answer_end", "final_answer_error"}
)


@dataclass(frozen=True)
class EventDisplaySnapshot:
    horizon: int
    events: tuple[dict[str, Any], ...]
    messages: tuple[dict[str, Any], ...]
    compactions: tuple[dict[str, Any], ...]


def event_data(event: TaskExecutionEvent) -> dict[str, Any]:
    if event.payload_version != 1 or not isinstance(event.payload, dict):
        raise _invalid(event, "unsupported payload")
    data = event.payload.get("data", event.payload)
    if not isinstance(data, dict):
        raise _invalid(event, "invalid data")
    return data


def _invalid(event: TaskExecutionEvent, reason: str) -> ValueError:
    return ValueError(
        f"Invalid display event (task_id={event.task_id}, event_id={event.event_id}): {reason}"
    )


def _identity(value: Any, event: TaskExecutionEvent, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise _invalid(event, f"missing {field}")
    return value


def display_horizon(db: Session, task_id: int) -> int:
    return int(
        db.execute(
            select(Task.conversation_event_sequence).where(Task.id == task_id)
        ).scalar_one()
    )


def load_display_facts(
    db: Session,
    task_id: int,
    *,
    scope_id: str = "root",
    through_sequence: int,
    after_sequence: int = 0,
    source: str | None = None,
    kinds: tuple[str, ...] | None = None,
) -> list[TaskExecutionEvent]:
    events: list[TaskExecutionEvent] = []
    after = after_sequence
    query = select(TaskExecutionEvent).where(
        TaskExecutionEvent.task_id == task_id,
        TaskExecutionEvent.scope_id == scope_id,
        TaskExecutionEvent.sequence <= through_sequence,
        TaskExecutionEvent.kind.not_in(_INTERNAL_KINDS),
    )
    if source is not None:
        query = query.where(
            TaskExecutionEvent.payload["data"]["source"].as_string() == source
        )
    if kinds is not None:
        query = query.where(TaskExecutionEvent.kind.in_(kinds))
    while True:
        page = list(
            db.scalars(
                query.where(TaskExecutionEvent.sequence > after)
                .order_by(TaskExecutionEvent.sequence)
                .limit(MAX_EXECUTION_EVENT_PAGE_SIZE)
            )
        )
        if not page:
            return events
        events.extend(page)
        after = int(page[-1].sequence)


def _stream_aliases(events: list[TaskExecutionEvent]) -> dict[str, str]:
    """Resolve terminal transcript ownership using explicit run/stream links."""
    aliases: dict[str, str] = {}
    interval: list[TaskExecutionEvent] = []
    for settlement in events:
        interval.append(settlement)
        if settlement.kind != "execution_settled":
            continue
        data = event_data(settlement)
        if data.get("status") == "completed":
            messages = [
                row
                for row in interval
                if row.kind == "assistant_message"
                and row.run_id == settlement.run_id
                and event_data(row).get("message_type") == "assistant_response"
            ]
            completions = [
                row
                for row in interval
                if row.kind == "task_completion" and row.run_id == settlement.run_id
            ]
            linked = [
                row
                for row in completions
                if isinstance(event_data(row).get("result"), dict)
                and event_data(row)["result"].get("stream_message_id")
            ]
            if not linked and any(row.kind == "final_answer_end" for row in interval):
                raise _invalid(settlement, "missing completed stream provenance")
            if linked:
                if not settlement.run_id or len(completions) != 1 or len(messages) != 1:
                    raise _invalid(settlement, "ambiguous stream settlement provenance")
                stream_id = _identity(
                    event_data(linked[0])["result"]["stream_message_id"],
                    linked[0],
                    "stream_message_id",
                )
                ends = [
                    row
                    for row in interval
                    if row.kind == "final_answer_end"
                    and row.run_id == settlement.run_id
                    and event_data(row).get("message_id") == stream_id
                    and row.sequence < linked[0].sequence < messages[0].sequence
                ]
                if len(ends) != 1:
                    raise _invalid(settlement, "missing or ambiguous completed stream")
                aliases[str(messages[0].event_id)] = stream_id
        interval = []
    return aliases


def _message(event: TaskExecutionEvent) -> dict[str, Any] | None:
    data = event_data(event)
    if is_audit_only_trace_data(data) or is_agent_checkpoint_data(data):
        return None
    if event.kind in {"input_accepted", "assistant_message"}:
        role = "user" if event.kind == "input_accepted" else "assistant"
        content = data.get("content")
        message_type = data.get("message_type")
        if (content is not None and not isinstance(content, str)) or (
            message_type is not None and not isinstance(message_type, str)
        ):
            raise _invalid(event, "invalid message content or type")
        content = content or ""
        message_type = message_type or ""
        safe = role == "user" or assistant_history_has_safe_ancillary_payload(
            message_type
        )
        if role == "assistant":
            content = client_safe_assistant_history_content(
                content=content, message_type=message_type
            )
        attachments = data.get("attachments") if safe else None
        interactions = data.get("interactions") if safe else None
    elif (
        event.kind == "agent_message"
        and data.get("expect_response")
        and data.get("visible") is not False
        and data.get("message")
    ):
        role, message_type = "assistant", "question"
        content = data["message"]
        metadata = data.get("metadata")
        if (metadata is not None and not isinstance(metadata, dict)) or not isinstance(
            content, str
        ):
            raise _invalid(event, "invalid question")
        interactions = (metadata or {}).get("interactions")
        attachments = data.get("attachments")
    else:
        return None
    return {
        "id": int(event.sequence),
        "role": role,
        "content": content,
        "message_type": message_type,
        "interactions": interactions if isinstance(interactions, list) else None,
        "attachments": attachments if isinstance(attachments, list) else [],
        "turn_id": event.turn_id,
        "created_at": event.occurred_at,
    }


def convert_display_event(
    event: TaskExecutionEvent,
    *,
    message: dict[str, Any] | None = None,
    stream_alias: str | None = None,
    replay: bool = False,
    stream_interrupted: bool = False,
) -> dict[str, Any] | None:
    """Convert one durable occurrence for both live delivery and history."""
    if event.kind in _INTERNAL_KINDS or event.kind == "execution_settled":
        return None
    data = event_data(event)
    if is_audit_only_trace_data(data) or is_agent_checkpoint_data(data):
        return None
    kind = str(event.kind)
    wire_id = event.payload.get("protocol_event_id")
    if kind in {"input_accepted", "assistant_message"}:
        message = message or _message(event)
        assert message is not None
        wire_id = f"execution_message_{event.event_id}"
        content = message["content"]
        if kind == "input_accepted":
            turn_id = _identity(event.turn_id, event, "turn_id")
            kind = "user_message"
            data = {"message": content, "content": content, "turn_id": turn_id}
            if message["attachments"]:
                data["files"] = data["attachments"] = message["attachments"]
        else:
            kind = "agent_message"
            data = {
                "message": content,
                "content": content,
                "message_id": stream_alias or wire_id,
                "message_type": message["message_type"],
                "display": "chat",
                "role": "assistant",
                "visible": True,
                "expect_response": False,
                "metadata": {"interactions": message["interactions"] or []},
            }
    else:
        wire_id = _identity(wire_id, event, "protocol_event_id")
        data = deepcopy(data)
        if kind in _STREAM_KINDS:
            message_id = _identity(data.get("message_id"), event, "message_id")
            if not message_id.startswith("final_answer_"):
                raise _invalid(event, "invalid stream message_id")
            if kind == "final_answer_error":
                data["error"] = CLIENT_SAFE_TASK_FAILURE
            if stream_interrupted:
                data["status"] = "interrupted"
        elif kind == "user_message":
            # Root bubbles belong to input_accepted. A child inspector can
            # display its own prompt, without the embedded runtime context.
            if event.scope_id == "root":
                return None
            data = {
                key: value
                for key, value in data.items()
                if key
                in {
                    "message",
                    "content",
                    "turn_id",
                    "files",
                    "attachments",
                    "source",
                    "agent_id",
                    "agent_name",
                    "worker_member_id",
                    "worker_alias",
                }
            }
        elif kind in {"task_completion", "ai_message"}:
            data["display"] = "timeline"
        elif kind == "agent_message":
            if message is None:
                _message(event)
            data["message_id"] = f"execution_message_{event.event_id}"
            if replay:
                data["expect_response"] = False
        if kind.startswith("llm_") or kind.endswith("_llm"):
            data = normalize_llm_trace_payload(data)
    kind, data = normalize_public_trace_event(kind, data)
    if kind in _STREAM_KINDS:
        stream_data = serialize_trace_data(data)
        for field in ("type", "event_id", "task_id", "timestamp"):
            stream_data.pop(field, None)
        return {
            **stream_data,
            "type": kind,
            "event_id": wire_id,
            "task_id": int(event.task_id),
            "execution_sequence": int(event.sequence),
            "timestamp": safe_timestamp_to_unix(event.occurred_at),
        }
    return {
        "type": "trace_event",
        "event_id": wire_id,
        "event_type": kind,
        "task_id": int(event.task_id),
        "execution_sequence": int(event.sequence),
        "timestamp": safe_timestamp_to_unix(event.occurred_at),
        "step_id": event.payload.get("step_id"),
        "parent_event_id": event.payload.get("parent_event_id"),
        "data": serialize_trace_data(data),
    }


def load_event_display_snapshot(
    db: Session,
    task_id: int,
    *,
    scope_id: str = "root",
    through_sequence: int | None = None,
    active_run_id: str | None = None,
) -> EventDisplaySnapshot:
    horizon = (
        display_horizon(db, task_id) if through_sequence is None else through_sequence
    )
    facts = load_display_facts(db, task_id, scope_id=scope_id, through_sequence=horizon)
    aliases = _stream_aliases(facts)
    terminal_streams = {
        (row.run_id, event_data(row).get("message_id"))
        for row in facts
        if row.kind in {"final_answer_end", "final_answer_error"}
    }
    messages = []
    events = []
    compactions = []
    for row in facts:
        message = _message(row)
        if message is not None:
            messages.append(message)
        converted = convert_display_event(
            row,
            message=message,
            stream_alias=aliases.get(str(row.event_id)),
            replay=True,
            stream_interrupted=bool(
                row.kind == "final_answer_start"
                and (row.run_id, event_data(row).get("message_id"))
                not in terminal_streams
                and (not row.run_id or row.run_id != active_run_id)
            ),
        )
        if converted is not None:
            events.append(converted)
            if row.kind == "action_end_compact":
                compactions.append(
                    {
                        "sequence": int(row.sequence),
                        "id": f"compact-{row.event_id}",
                        "role": "system",
                        "message_type": "compaction",
                        "content": "",
                        "compaction": {
                            key: converted["data"].get(key)
                            for key in (
                                "original_tokens",
                                "compacted_tokens",
                                "compression_ratio",
                            )
                        },
                        "created_at": row.occurred_at,
                    }
                )
    return EventDisplaySnapshot(
        horizon, tuple(events), tuple(messages), tuple(compactions)
    )


def load_live_display_event(
    db: Session, task_id: int, protocol_event_id: str, *, scope_id: str = "root"
) -> dict[str, Any] | None:
    event = db.scalar(
        select(TaskExecutionEvent).where(
            TaskExecutionEvent.task_id == task_id,
            TaskExecutionEvent.scope_id == scope_id,
            TaskExecutionEvent.payload["protocol_event_id"].as_string()
            == protocol_event_id,
        )
    )
    if event is None:
        raise ValueError(
            f"Missing committed display event (task_id={task_id}, protocol_event_id={protocol_event_id})"
        )
    if event.kind == "user_message":
        event = db.scalar(
            select(TaskExecutionEvent).where(
                TaskExecutionEvent.task_id == task_id,
                TaskExecutionEvent.scope_id == scope_id,
                TaskExecutionEvent.kind == "input_accepted",
                TaskExecutionEvent.turn_id == event.turn_id,
            )
        )
        if event is None:
            raise ValueError(
                f"Missing accepted display turn (task_id={task_id}, protocol_event_id={protocol_event_id})"
            )
    return convert_display_event(event)


async def publish_task_result(message: dict[str, Any], task_id: int) -> None:
    """Project a committed terminal result, then use the host-neutral transport."""
    from .db_runtime import run_db_io_cancellation_safe
    from .task_events import publish_task_event

    # V2 settlement owners supply their exact committed control snapshot.
    # V1 error notices and pre-lease failures have no fact-backed result.
    if "state_version" in message:
        display_events = await run_db_io_cancellation_safe(
            lambda: completion_display_events(task_id, message)
        )
        if display_events is not None:
            for event in display_events:
                await publish_task_event(event, task_id)
            if display_events:
                message = {
                    **message,
                    "message_id": display_events[-1]["data"]["message_id"],
                }
    await publish_task_event(message, task_id)


def completion_display_events(
    task_id: int, message: dict[str, Any]
) -> tuple[dict[str, Any], ...] | None:
    """Read committed transcript ownership for a terminal business notification."""
    from ..models.database import get_session_local

    with get_session_local()() as db:
        if (
            db.scalar(
                select(Task.conversation_storage_version).where(Task.id == task_id)
            )
            != 2
        ):
            return None
        run_id = message.get("run_id")
        state_version = message.get("state_version")
        status = (message.get("task") or {}).get("status")
        settlement = db.scalar(
            select(TaskExecutionEvent).where(
                TaskExecutionEvent.task_id == task_id,
                TaskExecutionEvent.scope_id == "root",
                TaskExecutionEvent.idempotency_key
                == f"result:{run_id}:{state_version}:{status}",
            )
        )
        if settlement is None:
            raise ValueError(
                f"Missing display settlement (task_id={task_id}, run_id={run_id}, state_version={state_version})"
            )
        return settlement_display_events(db, settlement)


def settlement_display_events(
    db: Session, settlement: TaskExecutionEvent
) -> tuple[dict[str, Any], ...]:
    """Return transcript frames owned by this committed settlement interval."""
    task_id, run_id = int(settlement.task_id), settlement.run_id
    previous = (
        db.scalar(
            select(TaskExecutionEvent.sequence)
            .where(
                TaskExecutionEvent.task_id == task_id,
                TaskExecutionEvent.scope_id == "root",
                TaskExecutionEvent.kind == "execution_settled",
                TaskExecutionEvent.sequence < settlement.sequence,
            )
            .order_by(TaskExecutionEvent.sequence.desc())
            .limit(1)
        )
        or 0
    )
    facts = load_display_facts(
        db,
        task_id,
        after_sequence=int(previous),
        through_sequence=int(settlement.sequence),
        kinds=(
            "assistant_message",
            "task_completion",
            "final_answer_end",
            "execution_settled",
        ),
    )
    aliases = _stream_aliases(facts)
    frames = []
    for row in facts:
        if row.kind == "assistant_message" and row.run_id == run_id:
            frame = convert_display_event(
                row, stream_alias=aliases.get(str(row.event_id))
            )
            if frame is not None:
                frames.append(frame)
    return tuple(frames)
