"""Strict START inputs for accepted turns and explicit existing execution.

The existing variant executes the stored description without adding a user
transcript row. All variants acquire their execution lease on the worker.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models.task import Task, TaskStatus
from .task_command_transport import (
    ClaimedTaskCommand,
    StagedTaskCommand,
    TaskCommandKind,
    TaskCommandTaskMissing,
    stage_task_command,
)

_Identity = Annotated[
    str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._:-]+$")
]
_FileId = Annotated[str, Field(min_length=1)]


class ExistingExecutionContext(BaseModel):
    """The three persisted legacy execution options used by execute_task."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    execution_mode: str | None = None
    process_description: str | None = None
    examples: list[JsonValue] | None = None


class TaskStartPayload(BaseModel):
    """JSON-only inputs not recoverable from the task configuration alone.

    The command envelope supplies task and actor identities. ``turn_id`` is
    also the command's idempotency key and the persisted transcript identity.
    ``message`` is the transcript text; ``execution_message`` preserves the
    separate, possibly file-enriched Agent input. File chips are already in the
    transcript; only authorized file IDs cross this boundary. The runner must
    resolve them against its own accessible storage.

    Version is required on the wire. Unknown fields are rejected rather than
    silently discarding a caller's runtime option or accepting runtime objects,
    arbitrary request context, connector secrets or a transferred lease.
    """

    # Frozen prevents field rebinding, not mutation of nested lists. Producers
    # must not mutate file_ids after validation.
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Annotated[int, Field(ge=1, le=1)]
    run_id: _Identity
    state_version: Annotated[int, Field(ge=1)]
    turn_id: _Identity
    kind: Literal["create", "append", "existing"]
    message: str
    execution_message: str | None = None
    file_ids: list[_FileId] = Field(default_factory=list)
    before_message_id: Annotated[int, Field(gt=0)] | None = None
    timezone: str | None = None
    force_fresh: bool = False
    runtime_values_ref: _Identity | None = None
    existing_context: ExistingExecutionContext | None = None

    @model_validator(mode="after")
    def validate_turn_kind(self) -> Self:
        if (
            self.runtime_values_ref is not None
            and self.runtime_values_ref != self.turn_id
        ):
            raise ValueError("Runtime values must belong to the accepted turn")
        if self.kind == "existing":
            if (
                self.file_ids
                or self.before_message_id is not None
                or self.force_fresh
                or self.runtime_values_ref is not None
                or self.message != self.execution_message
            ):
                raise ValueError(
                    "Existing execution cannot add a transcript turn or new inputs"
                )
            if self.existing_context is None:
                raise ValueError("Existing execution requires its explicit context")
        elif self.existing_context is not None:
            raise ValueError("Only existing execution accepts legacy context")
        if self.kind == "create" and self.force_fresh:
            raise ValueError("CREATE has no previous execution to discard")
        return self


def stage_task_start_command(
    db: Session,
    *,
    task_id: int,
    actor_user_id: int,
    start: TaskStartPayload,
) -> StagedTaskCommand:
    """Stage START as the last write of the caller's acceptance transaction.

    The caller must have reserved the turn through its task-state CAS in THIS
    transaction, including authorization, message persistence and file binding.
    That write holds SQLite's writer lock and PostgreSQL's task row lock until
    the transaction ends. FOR UPDATE retains explicit locking at this check;
    it is not the only lock in a valid acceptance transaction. This helper
    does not replace business acceptance.

    Do not commit a RUNNING turn and call this in a later transaction. On any
    failure the caller must roll back the entire acceptance transaction, as for
    ``stage_task_command``. A repeated identity returns its payload comparison;
    a mismatch must not be committed as a new acceptance.
    """
    db.flush()
    task = db.execute(
        select(
            Task.status,
            Task.run_id,
            Task.state_version,
            Task.control_state,
            Task.runner_id,
            Task.lease_attempt_id,
            Task.lease_expires_at,
            Task.last_heartbeat_at,
        )
        .where(Task.id == task_id)
        .with_for_update()
    ).one_or_none()
    if task is None:
        raise TaskCommandTaskMissing(f"Task {task_id} not found")
    if (
        task.status != TaskStatus.RUNNING
        or task.run_id != start.run_id
        or task.state_version != start.state_version
        or task.control_state != "running"
    ):
        raise ValueError("START does not match the accepted task turn")
    if any(
        value is not None
        for value in (
            task.runner_id,
            task.lease_attempt_id,
            task.lease_expires_at,
            task.last_heartbeat_at,
        )
    ):
        raise ValueError("START acceptance must not own an execution lease")
    return stage_task_command(
        db,
        task_id=task_id,
        actor_user_id=actor_user_id,
        command_id=start.turn_id,
        kind=TaskCommandKind.START,
        payload=start.model_dump(mode="json"),
    )


def read_task_start_command(command: ClaimedTaskCommand) -> TaskStartPayload:
    """Decode persisted inputs and verify their immutable envelope identity.

    This does not authorize execution, acquire a lease or prove that a prior
    attempt had no effect. Those are responsibilities of the START consumer.
    """
    if command.kind != TaskCommandKind.START:
        raise ValueError("Expected a START command")
    start = TaskStartPayload.model_validate(command.payload)
    if start.run_id != command.target_run_id or start.turn_id != command.command_id:
        raise ValueError("START payload does not match its command identity")
    return start
