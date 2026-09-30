"""Version-two recovery reads; legacy projections are never content sources."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ...core.agent.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    READABLE_CHECKPOINT_TYPES,
    CheckpointAccessRefusedError,
    CheckpointCorruptError,
    CheckpointUnavailableError,
    UnknownToolEffectError,
    checkpoint_execution_id,
)
from ...core.agent.result import CONTROL_TOOL_NAMES
from ..models.task import Task, TaskStatus
from ..models.task_execution_event import TaskExecutionEvent
from .task_lease_service import current_task_lease, task_lease_attempt_predicate


def event_checkpoint_data(event: TaskExecutionEvent) -> dict[str, Any]:
    """Validate the persisted recovery envelope before trusting its snapshot."""
    payload = event.payload
    data = payload.get("data") if isinstance(payload, dict) else None
    if (
        event.payload_version != 1
        or event.kind != "recovery_state"
        or not isinstance(data, dict)
        or data.get("checkpoint_type") not in READABLE_CHECKPOINT_TYPES
        or data.get("snapshot_schema_version") != CHECKPOINT_SCHEMA_VERSION
        or not isinstance(data.get("snapshot"), dict)
        or not checkpoint_execution_id(data)
        or not isinstance(data["snapshot"].get("context"), dict)
    ):
        raise CheckpointCorruptError("Execution event has an unreadable recovery state")
    return data


def check_recovery_owner(db: Session, task_id: int) -> None:
    """Recheck the current lease; an event snapshot never grants ownership."""
    lease = current_task_lease()
    query = select(Task.status).where(Task.id == task_id)
    if lease is not None:
        if lease.task_id != task_id:
            raise CheckpointAccessRefusedError(
                "Recovery reader is bound to another task", reason="lease_mismatch"
            )
        query = query.where(
            Task.runner_id == lease.runner_id,
            Task.run_id == lease.run_id,
            task_lease_attempt_predicate(lease),
        )
    status = db.scalar(query)
    if status is None:
        raise CheckpointAccessRefusedError(
            "Recovery reader lost its task lease", reason="lease_mismatch"
        )
    # Coordinator cancellation can settle a task without releasing its lease.
    # This live status check also covers settlements committed after horizon H.
    if status in {TaskStatus.COMPLETED, TaskStatus.FAILED}:
        raise CheckpointAccessRefusedError("Recovery task has already ended")


def read_event_checkpoint(
    db: Session,
    *,
    task_id: int,
    scope_id: str,
    execution_id: str,
    run_id: str | None,
    filter_run: bool = True,
) -> dict[str, Any] | None:
    horizon = db.scalar(
        select(Task.conversation_event_sequence).where(Task.id == task_id)
    )
    if horizon is None:
        raise CheckpointUnavailableError("Recovery task no longer exists")
    query = select(TaskExecutionEvent).where(
        TaskExecutionEvent.task_id == task_id,
        TaskExecutionEvent.scope_id == scope_id,
        TaskExecutionEvent.kind == "recovery_state",
        TaskExecutionEvent.sequence <= horizon,
    )
    if filter_run:
        query = query.where(TaskExecutionEvent.run_id == run_id)
    # Keyset pages also bound work when other executions share a build scope.
    before = int(horizon) + 1
    for _ in range(50):
        page = list(
            db.scalars(
                query.where(TaskExecutionEvent.sequence < before)
                .order_by(TaskExecutionEvent.sequence.desc())
                .limit(100)
            )
        )
        if not page:
            return None
        for event in page:
            raw_data = (
                event.payload.get("data") if isinstance(event.payload, dict) else None
            )
            if isinstance(raw_data, dict):
                identity = checkpoint_execution_id(raw_data)
                if identity and identity != execution_id:
                    continue
            data = event_checkpoint_data(event)
            if scope_id == "root":
                settlement = db.scalar(
                    select(TaskExecutionEvent)
                    .where(
                        TaskExecutionEvent.task_id == task_id,
                        TaskExecutionEvent.scope_id == scope_id,
                        TaskExecutionEvent.run_id == event.run_id,
                        TaskExecutionEvent.kind == "execution_settled",
                        TaskExecutionEvent.sequence > event.sequence,
                        TaskExecutionEvent.sequence <= horizon,
                    )
                    .order_by(TaskExecutionEvent.sequence.desc())
                    .limit(1)
                )
                if settlement is not None:
                    if (
                        settlement.payload_version != 1
                        or not isinstance(settlement.payload, dict)
                        or settlement.payload.get("status")
                        not in {status.value for status in TaskStatus}
                    ):
                        raise CheckpointCorruptError("Unreadable execution settlement")
                    if settlement.payload["status"] in {
                        TaskStatus.COMPLETED.value,
                        TaskStatus.FAILED.value,
                    }:
                        raise CheckpointAccessRefusedError(
                            "Execution was already settled; its checkpoint cannot be resumed"
                        )
            snapshot = data["snapshot"]
            state = snapshot.get("pattern_state") or {}
            if snapshot.get("pattern") == "AutoPattern":
                if not isinstance(state, dict):
                    raise CheckpointCorruptError("Unreadable Auto recovery state")
                decision = state.get("decision") or {}
                if not isinstance(decision, dict):
                    raise CheckpointCorruptError("Unreadable Auto decision")
                action = decision.get("action")
                if action not in ("react", "plan_execute", "final_answer") and (
                    state.get("react_state") or state.get("dag_state")
                ):
                    raise CheckpointCorruptError(
                        "Auto child state has no recognized decision"
                    )
                state = (
                    state.get("react_state")
                    if action == "react"
                    else state.get("dag_state")
                    if action == "plan_execute"
                    else None
                ) or {}
            if not isinstance(state, dict) or not isinstance(
                state.get("active_step_pattern_states", {}), dict
            ):
                raise CheckpointCorruptError("Unreadable recovery pattern state")
            states = [state, *state.get("active_step_pattern_states", {}).values()]
            for pattern_state in states:
                if not isinstance(pattern_state, dict) or not isinstance(
                    pattern_state.get("pending_tool_calls", []), list
                ):
                    raise CheckpointCorruptError("Unreadable pending tool state")
                for call in pattern_state.get("pending_tool_calls", []):
                    if not isinstance(call, dict) or not call.get("name"):
                        raise CheckpointCorruptError("Unreadable pending tool call")
                    if call.get("name") in CONTROL_TOOL_NAMES:
                        continue
                    if not call.get("tool_attempt_id") or not call.get(
                        "assistant_message_id"
                    ):
                        raise CheckpointCorruptError(
                            "Pending tool call has no durable attempt identity"
                        )
                    read_committed_tool_outcome(
                        db,
                        task_id=task_id,
                        scope_id=scope_id,
                        tool_call=call,
                        through_sequence=int(horizon),
                    )
            return deepcopy(data)
        before = int(page[-1].sequence)
    raise CheckpointUnavailableError("Recovery event scan exceeded its page limit")


def read_committed_tool_outcome(
    db: Session,
    *,
    task_id: int,
    scope_id: str,
    tool_call: dict[str, Any],
    through_sequence: int | None = None,
) -> dict[str, Any] | None:
    """Return an occurrence's committed outcome, or refuse an unknown effect."""
    query = (
        select(TaskExecutionEvent)
        .where(
            TaskExecutionEvent.task_id == task_id,
            TaskExecutionEvent.scope_id == scope_id,
            TaskExecutionEvent.tool_attempt_id == tool_call["tool_attempt_id"],
            TaskExecutionEvent.kind.in_(
                ["tool_execution_start", "tool_execution_end", "tool_execution_failed"]
            ),
        )
        .order_by(TaskExecutionEvent.sequence)
    )
    if through_sequence is not None:
        query = query.where(TaskExecutionEvent.sequence <= through_sequence)
    rows = list(db.scalars(query))
    if not rows:
        return None
    outcome = None
    for event in rows:
        data = event.payload.get("data") if isinstance(event.payload, dict) else None
        if (
            event.payload_version != 1
            or not isinstance(data, dict)
            or event.assistant_message_id != tool_call.get("assistant_message_id")
            or data.get("tool_name") != tool_call["name"]
            or data.get("tool_call_id") != tool_call.get("id")
        ):
            raise CheckpointCorruptError(
                "Tool attempt facts do not match the saved call"
            )
        if event.kind in {"tool_execution_end", "tool_execution_failed"}:
            if data.get("interrupted") is True:
                raise UnknownToolEffectError(
                    "Interrupted tool attempt has no confirmed result"
                )
            if "result" not in data:
                raise CheckpointCorruptError(
                    "Committed tool outcome has no reusable result"
                )
            outcome = {
                "result": deepcopy(data["result"]),
                "error": data.get("error")
                if event.kind == "tool_execution_failed"
                else None,
            }
    if outcome is None:
        raise UnknownToolEffectError(
            "Tool attempt started without a committed result; automatic replay is unsafe"
        )
    return outcome


def find_event_checkpoint_anchor(
    db: Session,
    *,
    task_id: int,
    protocol_event_id: str,
    execution_id: str,
    run_id: str,
) -> TaskExecutionEvent | None:
    event = db.scalar(
        select(TaskExecutionEvent).where(
            TaskExecutionEvent.task_id == task_id,
            TaskExecutionEvent.scope_id == "root",
            TaskExecutionEvent.kind == "recovery_state",
            TaskExecutionEvent.run_id == run_id,
            TaskExecutionEvent.payload["protocol_event_id"].as_string()
            == protocol_event_id,
        )
    )
    if event is None:
        return None
    data = event_checkpoint_data(event)
    if checkpoint_execution_id(data) != execution_id:
        return None
    return event


def read_event_waiting_question(
    db: Session, task_id: int
) -> tuple[str | None, list[dict[str, Any]] | None]:
    """Read the legacy-shaped waiting projection from V2 outbound facts."""
    run_id = db.scalar(select(Task.run_id).where(Task.id == task_id))
    event = db.scalar(
        select(TaskExecutionEvent)
        .where(
            TaskExecutionEvent.task_id == task_id,
            TaskExecutionEvent.scope_id == "root",
            TaskExecutionEvent.run_id == run_id,
            TaskExecutionEvent.kind == "agent_message",
            TaskExecutionEvent.payload["data"]["expect_response"]
            .as_boolean()
            .is_(True),
        )
        .order_by(TaskExecutionEvent.sequence.desc())
        .limit(1)
    )
    if event is None:
        return None, None
    if event.payload_version != 1:
        raise CheckpointCorruptError(
            "Waiting message uses an unsupported event version"
        )
    data = event.payload["data"]
    metadata = data.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise CheckpointCorruptError("Waiting message has invalid metadata")
    return data.get("message"), (metadata or {}).get("interactions")
