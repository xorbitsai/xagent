"""Atomic persistence for terminal task-command outcomes."""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..models.task import Task
from ..models.task_command import TaskExecutionCommand
from ..models.task_command_terminal_event import TaskCommandTerminalEvent


class TerminalTaskEventMessageCode(str, enum.Enum):
    """Closed vocabulary for rendering terminal outcomes without stored text."""

    TASK_COMMAND_FAILED = "task_command_failed"
    TASK_COMMAND_DEFERRED = "task_command_deferred"
    EXTERNAL_CANCEL_NOT_APPLIED = "external_cancel_not_applied"
    EXTERNAL_TURN_INTERRUPTED = "external_turn_interrupted"


@dataclass(frozen=True)
class TerminalTaskEventDraft:
    """Client-safe terminal outcome staged by a command disposition."""

    message_code: TerminalTaskEventMessageCode | None
    resend_safe: bool
    include_command_identity: bool = True


_DRAFT_ATTRIBUTE = "_xagent_terminal_task_event_draft"
_CANCEL_COMMAND_KIND = "cancel"
_EXTERNAL_COMMAND_SCOPE = "external"


def is_external_cancel_command(*, kind: str, scope: object) -> bool:
    """Return whether strict command kind/scope values name an external cancel.

    The helper accepts normalized primitives so durable ORM rows and live
    command snapshots can share one disclosure-policy classifier without
    importing each other's modules.
    """

    return kind == _CANCEL_COMMAND_KIND and scope == _EXTERNAL_COMMAND_SCOPE


def bind_terminal_event_draft(
    error: BaseException,
    draft: TerminalTaskEventDraft,
) -> None:
    """Attach client-safe presentation metadata without performing delivery."""

    setattr(error, _DRAFT_ATTRIBUTE, draft)


def terminal_event_draft_for_error(
    error: BaseException,
) -> TerminalTaskEventDraft | None:
    """Read presentation metadata previously attached by an executor adapter."""

    draft = getattr(error, _DRAFT_ATTRIBUTE, None)
    return draft if isinstance(draft, TerminalTaskEventDraft) else None


def stage_terminal_event(
    db: Session,
    *,
    command_db_id: int,
    draft: TerminalTaskEventDraft | None = None,
) -> TaskCommandTerminalEvent:
    """Stage one idempotent event without committing the caller's transaction.

    The command must already have a terminal disposition in this transaction.
    The caller owns the commit, which makes disposition and event one recovery
    boundary instead of two best-effort operations. Run correlation is copied
    exclusively from the immutable command-acceptance snapshot; reading the
    task's current run or state version here could associate an old command
    outcome with a newer interaction.

    The insert runs in a savepoint so a concurrent natural-key winner can be
    adopted without poisoning the caller's transaction. Other integrity
    failures still propagate after the savepoint has been rolled back. On
    SQLite, the caller must execute the terminal-disposition DML in this
    transaction before entering this helper so the SAVEPOINT does not become
    pysqlite's first write-adjacent statement.
    """

    snapshot = (
        db.query(TaskExecutionCommand, Task)
        .join(Task, Task.id == TaskExecutionCommand.task_id)
        .filter(TaskExecutionCommand.id == command_db_id)
        .populate_existing()
        .one_or_none()
    )
    if snapshot is None:
        raise ValueError(f"Task command {command_db_id} does not exist")
    command, task = snapshot
    if command.status not in {"completed", "failed"}:
        raise ValueError(
            f"Task command {command_db_id} is not terminal: {command.status}"
        )
    outcome_version = int(command.attempt_count or 0)
    outcome = str(command.status)
    if draft is None:
        failed = command.status == "failed"
        draft = TerminalTaskEventDraft(
            message_code=(
                TerminalTaskEventMessageCode.TASK_COMMAND_FAILED if failed else None
            ),
            resend_safe=False,
        )
    scope = command.payload.get("scope") if isinstance(command.payload, dict) else None

    event = TaskCommandTerminalEvent(
        event_id=str(uuid.uuid4()),
        task_command_id=int(command.id),
        task_id=int(command.task_id),
        task_run_id=command.target_run_id,
        task_state_version=(
            int(command.target_state_version)
            if command.target_state_version is not None
            else None
        ),
        command_id=str(command.command_id),
        command_kind=str(command.kind),
        actor_user_id=(
            int(command.actor_user_id) if command.actor_user_id is not None else None
        ),
        actor_subject=(
            str(command.actor_subject) if command.actor_subject is not None else None
        ),
        task_owner_user_id=(
            int(command.task_owner_user_id)
            if command.task_owner_user_id is not None
            else int(task.user_id)
        ),
        task_owner_subject=(
            str(command.task_owner_subject)
            if command.task_owner_subject is not None
            else None
        ),
        outcome_version=outcome_version,
        outcome=outcome,
        message_code=(draft.message_code.value if draft.message_code else None),
        resend_safe=bool(draft.resend_safe),
        # Any external-scope command, not only the cancel: the audience is
        # anonymous, cannot act on durable command identity, and must not be
        # shown it. Enforced here — the single persistence chokepoint —
        # rather than at each draft-building call site, so a caller that
        # forgets (or a defaulted draft on the success path, which carries
        # include_command_identity=True) cannot leak the identity for an
        # external MESSAGE terminal (#1979 review NEW-1).
        include_command_identity=bool(
            draft.include_command_identity and scope != _EXTERNAL_COMMAND_SCOPE
        ),
    )
    try:
        with db.begin_nested():
            db.add(event)
            db.flush()
    except IntegrityError:
        existing = (
            db.query(TaskCommandTerminalEvent)
            .filter(
                TaskCommandTerminalEvent.task_command_id == command_db_id,
                TaskCommandTerminalEvent.outcome_version == outcome_version,
            )
            .one_or_none()
        )
        if existing is None:
            raise
        return existing
    return event
