"""Let a host settle a waiting interaction without a client chat message.

A task that waits for user input resumes when a new user message arrives. When
the answer was collected outside the chat (a host saved a connector, the user
skipped a step), the host calls ``settle_task_interaction``: it appends a
server-flagged, not-displayed user message to the task, through the same durable
command a chat frame uses, so the run resumes exactly as it would for a typed
answer. No WebSocket client has to be connected.

A settle only counts when the task waits on that exact interaction. The open
interaction ids are recorded on the waiting message
(``metadata.interaction_ids``, one per request of the waiting batch), and both
the call below and the command's execution check them before anything is
enqueued or injected (``task_waits_on_interaction_sync``). The agent applies
the same rule to its restored state (``is_stale_settle_message``) as a last
line of defence. A wait without recorded ids (a plain question, or one that
predates this record) cannot be settled.

On a match the settle text is handed to the matched tool alone; every other
request of the batch is left unanswered, and the run resumes and replans.

The call is idempotent while a settle for the same interaction is still
queued. A settle that ran to completion without resuming the task (it was
dropped, or failed) does not block a later one: the next call enqueues under a
fresh derived command id.
"""

from __future__ import annotations

import enum
import hashlib
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ...core.agent.result_card import RESULT_CARD_MESSAGE_TYPE
from ..models.chat_message import TaskChatMessage
from ..models.database import get_session_local
from ..models.task import Task, TaskStatus, TraceEvent
from ..models.task_command import TERMINAL_COMMAND_STATUSES, TaskExecutionCommand
from ..models.task_execution_event import TaskExecutionEvent
from ..models.user import User
from .client_error_messages import ClientErrorCode
from .message_metadata import is_valid_interaction_id, is_valid_metadata_kind

DEFAULT_SETTLE_KIND = "interaction_settled"
DEFAULT_SETTLE_TEXT = "[interaction settled]"
MAX_SETTLE_TEXT_CHARS = 200
SETTLE_COMMAND_PREFIX = "settle:"
# Command ids are at most 64 characters, so the interaction id is hashed in.
_COMMAND_ID_DIGEST_CHARS = 24
_MAX_SETTLE_ATTEMPTS = 8
_RECENT_WAITING_EVENT_SCAN = 10


class SettleOutcome(str, enum.Enum):
    """What ``settle_task_interaction`` did."""

    QUEUED = "queued"
    """The settle message was accepted and will resume the task."""
    DUPLICATE = "duplicate"
    """A settle for this interaction is already queued, or already applied."""
    NOT_WAITING = "not_waiting"
    """The task does not wait on this interaction; nothing was queued."""


def _recorded_interaction_ids(raw: Any) -> frozenset[str]:
    data = raw if isinstance(raw, dict) else {}
    if not data.get("expect_response"):
        return frozenset()
    metadata = data.get("metadata")
    ids = metadata.get("interaction_ids") if isinstance(metadata, dict) else None
    if not isinstance(ids, list):
        return frozenset()
    return frozenset(item for item in ids if isinstance(item, str))


def _open_interaction_ids(db: Session, task: Task) -> frozenset[str]:
    """Interaction ids of the task's latest waiting message.

    Reads the same persisted fact the waiting question is shown from, in the
    task's own storage version.
    """
    if task.conversation_storage_version == 2:
        from .task_execution_event_recovery import read_event_waiting_data

        return _recorded_interaction_ids(read_event_waiting_data(db, int(task.id)))
    rows = db.scalars(
        select(TraceEvent)
        .where(
            TraceEvent.task_id == task.id,
            TraceEvent.event_type == "agent_message",
            # Result cards are agent messages too; any number of them can sit
            # between the question and now, so they must not use up the scan.
            func.coalesce(TraceEvent.data["message_type"].as_string(), "")
            != RESULT_CARD_MESSAGE_TYPE,
        )
        .order_by(TraceEvent.id.desc())
        .limit(_RECENT_WAITING_EVENT_SCAN)
    )
    for row in rows:
        if isinstance(row.data, dict) and row.data.get("expect_response"):
            return _recorded_interaction_ids(row.data)
    return frozenset()


def task_waits_on_interaction_sync(task_id: int, interaction_id: str) -> bool:
    """Whether the task waits for user input on exactly ``interaction_id``."""
    with get_session_local()() as db:
        task = db.get(Task, task_id)
        if task is None or task.status != TaskStatus.WAITING_FOR_USER:
            return False
        return interaction_id in _open_interaction_ids(db, task)


def turn_delivery_exists_sync(task_id: int, turn_id: str) -> bool:
    """Whether a user message was already accepted for ``turn_id``.

    An earlier attempt of the same command leaves this row behind when it
    injected the settle and the worker died before marking it dispatched.
    """
    with get_session_local()() as db:
        row = db.scalar(
            select(TaskChatMessage.id).where(
                TaskChatMessage.task_id == task_id,
                TaskChatMessage.role == "user",
                TaskChatMessage.turn_id == turn_id,
            )
        )
        if row is not None:
            return True
        event = db.scalar(
            select(TaskExecutionEvent.id).where(
                TaskExecutionEvent.task_id == task_id,
                TaskExecutionEvent.scope_id == "root",
                TaskExecutionEvent.idempotency_key == f"message:{turn_id}",
                TaskExecutionEvent.kind == "input_accepted",
            )
        )
        return event is not None


def _check_actor_sync(task_id: int, actor_user_id: int, actor_is_admin: bool) -> None:
    """Raise the chat-frame errors for a missing task or a foreign actor."""
    from .task_command_execution import (
        ClientVisiblePermissionError,
        ClientVisibleValidationError,
    )

    with get_session_local()() as db:
        owner_id = db.query(Task.user_id).filter(Task.id == task_id).scalar()
    if owner_id is None:
        raise ClientVisibleValidationError(
            f"Task {task_id} not found",
            error_code=ClientErrorCode.TASK_UNAVAILABLE,
        )
    if not actor_is_admin and int(owner_id) != actor_user_id:
        raise ClientVisiblePermissionError(
            f"Access denied: Task {task_id} does not belong to you",
            error_code=ClientErrorCode.TASK_UNAVAILABLE,
        )


def _attempt_command_ids(interaction_id: str) -> list[str]:
    digest = hashlib.sha256(interaction_id.encode("utf-8")).hexdigest()
    base = f"{SETTLE_COMMAND_PREFIX}{digest[:_COMMAND_ID_DIGEST_CHARS]}"
    return [base] + [f"{base}:{attempt}" for attempt in range(1, _MAX_SETTLE_ATTEMPTS)]


def _settle_commands_sync(
    task_id: int, command_ids: list[str]
) -> dict[str, str | None]:
    with get_session_local()() as db:
        rows = db.query(TaskExecutionCommand.command_id, TaskExecutionCommand.status)
        rows = rows.filter(
            TaskExecutionCommand.task_id == task_id,
            TaskExecutionCommand.command_id.in_(command_ids),
        )
        return {str(command_id): status for command_id, status in rows}


async def settle_task_interaction(
    *,
    task_id: int,
    actor_user: User,
    interaction_id: str,
    text: str = DEFAULT_SETTLE_TEXT,
    kind: str = DEFAULT_SETTLE_KIND,
) -> SettleOutcome:
    """Settle the waiting interaction ``interaction_id`` of ``task_id``.

    ``actor_user`` must own the task (or be an admin), exactly as for a chat
    frame. ``interaction_id`` is the opaque id the waiting tool put in its
    result. ``kind`` is a host-chosen identifier clients can branch on. ``text``
    is what the model and the settled tool see.

    Returns ``NOT_WAITING`` unless the task waits on that interaction;
    ``DUPLICATE`` while an earlier settle for it is still queued (or when the
    task has already moved on after one); ``QUEUED`` otherwise.

    Raises ``ValueError`` for a malformed argument, when a different settle is
    already queued under the same derived command id (a changed ``text`` racing
    an earlier call), and after 8 settle attempts for one interaction. It also
    raises the same client-visible errors as a chat frame (task missing, access
    denied, input rejected).
    """
    from ..api.websocket import _enqueue_websocket_task_command
    from .task_command_execution import (
        execute_durable_task_command,
        run_db_io_cancellation_safe,
    )
    from .task_command_transport import (
        TaskCommandKind,
        dispatch_task_command_promptly,
    )

    if not is_valid_interaction_id(interaction_id):
        raise ValueError("interaction_id must be 1-64 characters of [A-Za-z0-9_-]")
    if not is_valid_metadata_kind(kind):
        raise ValueError("kind must be a lowercase identifier of at most 64 characters")
    if not text.strip() or len(text) > MAX_SETTLE_TEXT_CHARS:
        raise ValueError(
            f"text must be 1-{MAX_SETTLE_TEXT_CHARS} characters of visible text"
        )

    await run_db_io_cancellation_safe(
        lambda: _check_actor_sync(
            task_id, int(actor_user.id), bool(getattr(actor_user, "is_admin", False))
        )
    )
    attempts = _attempt_command_ids(interaction_id)
    seen = await run_db_io_cancellation_safe(
        lambda: _settle_commands_sync(task_id, attempts)
    )
    if not await run_db_io_cancellation_safe(
        lambda: task_waits_on_interaction_sync(task_id, interaction_id)
    ):
        # A repeated call after the first one resumed the task is a replay.
        return SettleOutcome.DUPLICATE if seen else SettleOutcome.NOT_WAITING

    command_id = None
    for candidate in attempts:
        status = seen.get(candidate, "")
        if status == "":
            command_id = candidate
            break
        if status not in TERMINAL_COMMAND_STATUSES:
            return SettleOutcome.DUPLICATE
    if command_id is None:
        raise ValueError("too many settle attempts for this interaction")

    message_data: dict[str, Any] = {"message": text, "user": actor_user}
    enqueued = await _enqueue_websocket_task_command(
        task_id=task_id,
        message_data=message_data,
        kind=TaskCommandKind.MESSAGE,
        command_id=command_id,
        server_message_metadata={
            "display": "none",
            "kind": kind,
            "interaction_id": interaction_id,
        },
    )
    assert enqueued is not None
    if not enqueued.created:
        if not enqueued.payload_matches:
            raise ValueError("a different settle was already queued for this id")
        return SettleOutcome.DUPLICATE
    await dispatch_task_command_promptly(
        execute_durable_task_command, command_db_id=enqueued.command_id
    )
    return SettleOutcome.QUEUED
