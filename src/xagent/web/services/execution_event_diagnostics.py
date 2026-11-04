"""Read-only, payload-free diagnostics for one task's V2 execution events.

Backs the admin diagnostic endpoint. Every query is filtered by ``task_id``
(indexed) and selects columns or GROUP BY aggregates only: the ``payload``
column of ``task_execution_events`` is never selected, filtered on or
loaded, because payloads hold user messages and tool arguments/results. Free
text on neighbouring rows (``Task.error_message``,
``task_auto_recovery.last_error``, ``task_recovery_events.detail``) is
likewise left out; only whether the error columns are set is reported.

Event sections read a snapshot at the task's committed horizon
(``Task.conversation_event_sequence``), read first: every append advances the
horizon in the same transaction, so all events at or below it are committed
and events appended while the request runs are excluded. The recovery rows
are separate tables and may be slightly newer than that snapshot.

Every list is bounded by an explicit limit and reports whether it was cut.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..models.task import Task
from ..models.task_auto_recovery import TaskAutoRecovery, TaskRecoveryEvent
from ..models.task_execution_event import TaskExecutionEvent
from .ops_signals import active_degradations

# Fact kinds, as written by ``task_execution_event_writer`` and the trace
# handlers and read by ``task_execution_event_recovery``.
SETTLEMENT_KIND = "execution_settled"
RECOVERY_STATE_KIND = "recovery_state"
TOOL_START_KIND = "tool_execution_start"
# ``read_committed_tool_outcome`` treats any of these rows for an attempt as a
# committed outcome. A settlement pair written by a later run
# (``tool:{attempt}:settlement:...``) uses the same kinds and attempt id, so
# it counts the same way.
TOOL_TERMINAL_KINDS = ("tool_execution_end", "tool_execution_failed")

MAX_EVENT_GROUPS = 200
MAX_RUNS = 20
MAX_RECOVERY_STATE_SCOPES = 50
MAX_OPEN_TOOL_ATTEMPTS = 100
MAX_RECOVERY_EVENTS = 20


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _in_snapshot(task_id: int, horizon: int) -> Any:
    event = TaskExecutionEvent
    return (event.task_id == task_id) & (event.sequence <= horizon)


def _event_summary(db: Session, task_id: int, horizon: int) -> dict[str, Any]:
    event = TaskExecutionEvent
    total, min_sequence, max_sequence, latest_occurred_at = db.execute(
        select(
            func.count(event.id),
            func.min(event.sequence),
            func.max(event.sequence),
            func.max(event.occurred_at),
        ).where(_in_snapshot(task_id, horizon))
    ).one()
    rows = db.execute(
        select(event.scope_id, event.kind, func.count(event.id))
        .where(_in_snapshot(task_id, horizon))
        .group_by(event.scope_id, event.kind)
        .order_by(event.scope_id, event.kind)
        .limit(MAX_EVENT_GROUPS + 1)
    ).all()
    return {
        "total": int(total),
        "min_sequence": min_sequence,
        "max_sequence": max_sequence,
        "latest_occurred_at": _iso(latest_occurred_at),
        "by_scope_kind": [
            {"scope_id": scope, "kind": kind, "count": int(count)}
            for scope, kind, count in rows[:MAX_EVENT_GROUPS]
        ],
        "by_scope_kind_truncated": len(rows) > MAX_EVENT_GROUPS,
    }


def _runs(db: Session, task_id: int, horizon: int) -> dict[str, Any]:
    event = TaskExecutionEvent
    max_sequence = func.max(event.sequence)
    rows = db.execute(
        select(
            event.run_id,
            func.count(event.id),
            func.min(event.sequence),
            max_sequence,
            func.count(event.id).filter(event.kind == SETTLEMENT_KIND),
        )
        .where(_in_snapshot(task_id, horizon))
        .group_by(event.run_id)
        .order_by(max_sequence.desc())
        .limit(MAX_RUNS + 1)
    ).all()
    return {
        "items": [
            {
                "run_id": run_id,
                "event_count": int(count),
                "min_sequence": min_seq,
                "max_sequence": max_seq,
                "settled": int(settled) > 0,
            }
            for run_id, count, min_seq, max_seq, settled in rows[:MAX_RUNS]
        ],
        "truncated": len(rows) > MAX_RUNS,
    }


def _recovery_states(db: Session, task_id: int, horizon: int) -> dict[str, Any]:
    event = TaskExecutionEvent
    rows = db.execute(
        select(event.scope_id, func.max(event.sequence))
        .where(_in_snapshot(task_id, horizon), event.kind == RECOVERY_STATE_KIND)
        .group_by(event.scope_id)
        .order_by(event.scope_id)
        .limit(MAX_RECOVERY_STATE_SCOPES + 1)
    ).all()
    return {
        "items": [
            {"scope_id": scope, "max_sequence": max_seq}
            for scope, max_seq in rows[:MAX_RECOVERY_STATE_SCOPES]
        ],
        "truncated": len(rows) > MAX_RECOVERY_STATE_SCOPES,
    }


def _open_tool_attempts(db: Session, task_id: int, horizon: int) -> dict[str, Any]:
    """Attempts with a start fact and no terminal fact, per scope.

    ``read_committed_tool_outcome`` raises ``UnknownToolEffectError`` for an
    attempt whose rows contain a start but no ``tool_execution_end`` or
    ``tool_execution_failed``; this is the same predicate, evaluated with one
    grouped query. The reader additionally refuses a terminal fact whose
    payload carries ``interrupted: true``; that flag lives in the payload and
    is deliberately not inspected here. The reader only evaluates attempts
    the latest checkpoint lists as pending, so an old open attempt listed
    here does not by itself make recovery refuse.
    """
    event = TaskExecutionEvent
    has_start = func.count(event.id).filter(event.kind == TOOL_START_KIND) > 0
    has_terminal = func.count(event.id).filter(event.kind.in_(TOOL_TERMINAL_KINDS)) > 0
    start_sequence = func.min(event.sequence).filter(event.kind == TOOL_START_KIND)
    open_filter = has_start & ~has_terminal

    open_total = db.scalar(
        select(func.count()).select_from(
            select(event.scope_id, event.tool_attempt_id)
            .where(_in_snapshot(task_id, horizon), event.tool_attempt_id.is_not(None))
            .group_by(event.scope_id, event.tool_attempt_id)
            .having(open_filter)
            .subquery()
        )
    )
    attempts = db.execute(
        select(event.scope_id, event.tool_attempt_id, start_sequence)
        .where(_in_snapshot(task_id, horizon), event.tool_attempt_id.is_not(None))
        .group_by(event.scope_id, event.tool_attempt_id)
        .having(open_filter)
        .order_by(start_sequence.desc(), event.tool_attempt_id)
        .limit(MAX_OPEN_TOOL_ATTEMPTS)
    ).all()

    kinds: dict[tuple[str, str], list[str]] = {}
    start_runs: dict[int, str | None] = {}
    if attempts:
        kind_rows = db.execute(
            select(event.scope_id, event.tool_attempt_id, event.kind)
            .where(
                _in_snapshot(task_id, horizon),
                event.tool_attempt_id.in_([attempt for _, attempt, _ in attempts]),
            )
            .group_by(event.scope_id, event.tool_attempt_id, event.kind)
        ).all()
        for scope, attempt, kind in kind_rows:
            kinds.setdefault((scope, attempt), []).append(kind)
        # The run of the first start, read from that row: a later run's
        # settlement start carries the same attempt id under another run.
        start_runs = {
            int(sequence): run_id
            for sequence, run_id in db.execute(
                select(event.sequence, event.run_id).where(
                    event.task_id == task_id,
                    event.sequence.in_([seq for _, _, seq in attempts]),
                )
            ).all()
        }
    return {
        "open_count": int(open_total or 0),
        "truncated": int(open_total or 0) > MAX_OPEN_TOOL_ATTEMPTS,
        "items": [
            {
                "tool_attempt_id": attempt,
                "scope_id": scope,
                "run_id": start_runs.get(start_seq),
                "start_sequence": start_seq,
                "kinds": sorted(kinds.get((scope, attempt), [])),
            }
            for scope, attempt, start_seq in attempts
        ],
    }


def _auto_recovery(db: Session, task_id: int) -> dict[str, Any]:
    row = db.execute(
        select(
            TaskAutoRecovery.run_id,
            TaskAutoRecovery.reason,
            TaskAutoRecovery.state,
            TaskAutoRecovery.state_detail,
            TaskAutoRecovery.paused_state_version,
            TaskAutoRecovery.interrupted_at,
            TaskAutoRecovery.episode_started_at,
            TaskAutoRecovery.no_progress_resumes,
            TaskAutoRecovery.total_resumes,
            TaskAutoRecovery.next_attempt_at,
            TaskAutoRecovery.last_command_id,
            TaskAutoRecovery.last_error,
            TaskAutoRecovery.updated_at,
        ).where(TaskAutoRecovery.task_id == task_id)
    ).one_or_none()
    state: dict[str, Any] | None = None
    if row is not None:
        state = {
            "run_id": row.run_id,
            "reason": row.reason,
            "state": row.state,
            "state_detail": row.state_detail,
            "paused_state_version": row.paused_state_version,
            "interrupted_at": _iso(row.interrupted_at),
            "episode_started_at": _iso(row.episode_started_at),
            "no_progress_resumes": row.no_progress_resumes,
            "total_resumes": row.total_resumes,
            "next_attempt_at": _iso(row.next_attempt_at),
            "last_command_id": row.last_command_id,
            # The run's own error text, the same content as error_message.
            "has_last_error": bool(row.last_error),
            "updated_at": _iso(row.updated_at),
        }
    # ``detail`` is documented as client-safe but is free-form JSON; the
    # diagnostic shape keeps to the typed columns.
    events = db.execute(
        select(
            TaskRecoveryEvent.id,
            TaskRecoveryEvent.run_id,
            TaskRecoveryEvent.created_at,
            TaskRecoveryEvent.event,
            TaskRecoveryEvent.reason,
            TaskRecoveryEvent.attempt,
            TaskRecoveryEvent.next_attempt_at,
        )
        .where(TaskRecoveryEvent.task_id == task_id)
        .order_by(TaskRecoveryEvent.id.desc())
        .limit(MAX_RECOVERY_EVENTS + 1)
    ).all()
    return {
        "state": state,
        "events": [
            {
                "id": e.id,
                "run_id": e.run_id,
                "created_at": _iso(e.created_at),
                "event": e.event,
                "reason": e.reason,
                "attempt": e.attempt,
                "next_attempt_at": _iso(e.next_attempt_at),
            }
            for e in events[:MAX_RECOVERY_EVENTS]
        ],
        "events_truncated": len(events) > MAX_RECOVERY_EVENTS,
    }


def task_execution_event_diagnostics(
    db: Session, task_id: int
) -> dict[str, Any] | None:
    """Payload-free event-log health for one task, or ``None`` if it is absent.

    The ``degradations`` section is ``active_degradations()`` of the process
    serving the request only: a worker process keeps its own registry, so an
    empty section here says nothing about the workers.

    Raises ``SQLAlchemyError`` on a database failure; callers must not
    substitute zeros for it.
    """
    task = db.execute(
        select(
            Task.id,
            Task.conversation_storage_version,
            Task.status,
            Task.control_state,
            Task.run_id,
            Task.state_version,
            Task.conversation_event_sequence,
            Task.runner_id,
            Task.lease_attempt_id,
            Task.lease_expires_at,
            Task.last_heartbeat_at,
            # Only the presence of the free-text column is reported.
            (Task.error_message.is_not(None) & (Task.error_message != "")).label(
                "has_error_message"
            ),
        ).where(Task.id == task_id)
    ).one_or_none()
    if task is None:
        return None
    horizon = int(task.conversation_event_sequence or 0)
    status = task.status
    return {
        "storage": "v2" if task.conversation_storage_version == 2 else "v1",
        "task": {
            "id": task.id,
            "conversation_storage_version": task.conversation_storage_version,
            "status": getattr(status, "value", status),
            "control_state": task.control_state,
            "run_id": task.run_id,
            "state_version": task.state_version,
            "conversation_event_sequence": task.conversation_event_sequence,
            "runner_id": task.runner_id,
            "lease_attempt_id": task.lease_attempt_id,
            "lease_expires_at": _iso(task.lease_expires_at),
            "last_heartbeat_at": _iso(task.last_heartbeat_at),
            "has_error_message": bool(task.has_error_message),
        },
        "events": _event_summary(db, task_id, horizon),
        "runs": _runs(db, task_id, horizon),
        "recovery_states": _recovery_states(db, task_id, horizon),
        "open_tool_attempts": _open_tool_attempts(db, task_id, horizon),
        "auto_recovery": _auto_recovery(db, task_id),
        "degradations": active_degradations(),
    }
