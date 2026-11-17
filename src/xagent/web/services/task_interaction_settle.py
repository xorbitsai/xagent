"""Let a host settle a waiting interaction without a client chat message.

A task that waits for user input resumes when a new user message arrives. When
the answer was collected outside the chat (a host saved a connector, the user
skipped a step), the host calls ``settle_task_interaction``: it appends a
hidden, server-flagged user message to the task, through the same durable
command a chat frame uses, so the run resumes exactly as it would for a typed
answer. No WebSocket client has to be connected.

The call is idempotent per ``interaction_id`` (the default command id is
``settle:<interaction_id>``): repeating it never appends a second message. At
execution the command is dropped if the task is no longer waiting, and the
agent ignores it unless a pending request of the waiting batch carries that
``interaction_id`` (``ReActPattern._is_stale_settle_message``), so a settle can
neither start a new turn nor answer a different question.
"""

from __future__ import annotations

import enum
from typing import Any

from sqlalchemy import exists

from ..models.database import get_session_local
from ..models.task import Task, TaskStatus
from ..models.task_command import TaskExecutionCommand
from ..models.user import User
from .client_error_messages import ClientErrorCode
from .message_metadata import (
    is_valid_interaction_id,
    is_valid_metadata_kind,
)

DEFAULT_SETTLE_KIND = "interaction_settled"
DEFAULT_SETTLE_TEXT = "[interaction settled]"
MAX_SETTLE_TEXT_CHARS = 200
_COMMAND_ID_PREFIX = "settle:"


class SettleOutcome(str, enum.Enum):
    """What ``settle_task_interaction`` did."""

    QUEUED = "queued"
    """The settle message was accepted and will resume the task."""
    DUPLICATE = "duplicate"
    """The same ``interaction_id`` was already queued or applied."""
    NOT_WAITING = "not_waiting"
    """The task is not waiting for user input; nothing was queued."""


def _precheck_sync(task_id: int, actor_user_id: int, actor_is_admin: bool) -> str:
    """``waiting``, ``duplicate`` or ``not_waiting`` for the task's current state."""
    from .task_command_execution import (
        ClientVisiblePermissionError,
        ClientVisibleValidationError,
    )

    with get_session_local()() as db:
        task = db.query(Task).filter(Task.id == task_id).first()
        if task is None:
            raise ClientVisibleValidationError(
                f"Task {task_id} not found",
                error_code=ClientErrorCode.TASK_UNAVAILABLE,
            )
        if not actor_is_admin and int(task.user_id) != actor_user_id:
            raise ClientVisiblePermissionError(
                f"Access denied: Task {task_id} does not belong to you",
                error_code=ClientErrorCode.TASK_UNAVAILABLE,
            )
        return "waiting" if task.status == TaskStatus.WAITING_FOR_USER else "other"


def _command_exists_sync(task_id: int, command_id: str) -> bool:
    with get_session_local()() as db:
        return bool(
            db.query(
                exists().where(
                    TaskExecutionCommand.task_id == task_id,
                    TaskExecutionCommand.command_id == command_id,
                )
            ).scalar()
        )


async def settle_task_interaction(
    *,
    task_id: int,
    actor_user: User,
    interaction_id: str,
    text: str = DEFAULT_SETTLE_TEXT,
    kind: str = DEFAULT_SETTLE_KIND,
    command_id: str | None = None,
) -> SettleOutcome:
    """Settle the waiting interaction ``interaction_id`` of ``task_id``.

    ``actor_user`` must own the task (or be an admin), exactly as for a chat
    frame. ``interaction_id`` is the opaque id the waiting tool put in its
    result; it is also the idempotency key. ``kind`` is a host-chosen
    identifier clients can branch on. ``text`` is what the model sees.

    Raises ``ValueError`` for a malformed argument and the same client-visible
    errors as a chat frame (task missing, access denied, input rejected).
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
    resolved_command_id = command_id or f"{_COMMAND_ID_PREFIX}{interaction_id}"

    state = await run_db_io_cancellation_safe(
        lambda: _precheck_sync(
            task_id, int(actor_user.id), bool(getattr(actor_user, "is_admin", False))
        )
    )
    if state != "waiting":
        # A repeated call after the first one resumed the task is a replay, not
        # a "not waiting" error.
        if await run_db_io_cancellation_safe(
            lambda: _command_exists_sync(task_id, resolved_command_id)
        ):
            return SettleOutcome.DUPLICATE
        return SettleOutcome.NOT_WAITING

    message_data: dict[str, Any] = {"message": text, "user": actor_user}
    enqueued = await _enqueue_websocket_task_command(
        task_id=task_id,
        message_data=message_data,
        kind=TaskCommandKind.MESSAGE,
        command_id=resolved_command_id,
        server_message_metadata={
            "hidden": True,
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
