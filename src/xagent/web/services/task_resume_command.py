"""Durable SDK/A2A reply admission and exact-attempt worker handoff."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Self, cast
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import and_, exists, func, or_, select
from sqlalchemy.orm import Session

from ...config import get_task_lease_ttl_seconds
from ...core.agent.checkpoint import (
    CheckpointAccessRefusedError,
    CheckpointCorruptError,
    CheckpointReadError,
    CheckpointUnavailableError,
)
from ..models.database import get_session_local
from ..models.task import Task, TaskStatus
from ..models.task_command import TaskExecutionCommand
from ..models.user import User
from .db_runtime import drain_async_task_cancellation_safe, run_db_io_cancellation_safe
from .llm_utils import AutoModelUnavailableError
from .task_command_transport import (
    COMMAND_COMPLETED,
    COMMAND_FAILED,
    COMMAND_PROCESSING,
    ClaimedTaskCommand,
    SettledTaskCommand,
    TaskCommandKind,
    TaskCommandRejected,
    finish_task_command_no_commit,
    notify_task_command_dispatcher,
    stage_task_command,
)
from .task_lease_service import TaskLease, acquire_task_lease_no_commit, get_runner_id
from .task_resume import (
    TaskReplyInput,
    TaskReplyResumeResult,
    TaskResumeBusyError,
    TaskResumeNotResumableError,
    TaskResumeRetryableError,
)

logger = logging.getLogger(__name__)


class ResumeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    version: Annotated[int, Field(ge=1, le=1)]
    source: Literal["sdk", "a2a"]
    run_id: Annotated[str, Field(min_length=1, max_length=64)]
    agent_id: Annotated[int, Field(gt=0)]
    prior_status: Literal["paused", "waiting_for_user"]
    text: str
    message_id: str

    @model_validator(mode="after")
    def validate_source_status(self) -> Self:
        if self.source == "sdk" and self.prior_status != "waiting_for_user":
            raise ValueError("SDK replies require a pending interaction")
        return self


def _admit_reply(
    ctx: TaskReplyInput, source: Literal["sdk", "a2a"], message_id: str, command_id: str
) -> int:
    if ctx.run_id is None:
        raise TaskResumeNotResumableError
    payload = ResumeInput(
        version=1,
        source=source,
        run_id=ctx.run_id,
        agent_id=ctx.agent_id,
        prior_status=cast(Literal["paused", "waiting_for_user"], ctx.status.value),
        text=ctx.text,
        message_id=message_id,
    )
    with get_session_local()() as db:
        task = db.execute(
            select(Task).where(Task.id == ctx.task_id).with_for_update()
        ).scalar_one_or_none()
        if (
            task is None
            or task.user_id != ctx.task_owner_user_id
            or task.agent_id != ctx.agent_id
            or task.source != source
        ):
            raise TaskResumeBusyError
        while True:
            existing = (
                db.query(TaskExecutionCommand)
                .filter_by(task_id=ctx.task_id, command_id=command_id)
                .first()
            )
            if existing is None:
                break
            owner = db.get(User, task.user_id)
            if (
                owner is None
                or existing.actor_user_id != owner.id
                or existing.task_owner_user_id != owner.id
                or existing.actor_subject != owner.actor_subject
                or existing.task_owner_subject != owner.actor_subject
                or existing.payload != payload.model_dump(mode="json")
            ):
                raise TaskResumeBusyError
            if (
                source == "a2a"
                and cast(dict[str, Any], existing.result or {}).get("outcome")
                == "retryable_unavailable"
            ):
                # Keep each command's acceptance snapshot immutable. A
                # retry gets one deterministic successor, while injection
                # still uses the original A2A message ID.
                command_id = uuid5(
                    NAMESPACE_URL, f"a2a-reply-retry:{ctx.task_id}:{existing.id}"
                ).hex
                continue
            return int(existing.id)
        changed = (
            db.query(Task)
            .filter(
                Task.id == ctx.task_id,
                Task.run_id == ctx.run_id,
                Task.status == ctx.status,
                # Wait for live owners to finish. Reclaim an expired resting
                # lease here: the recovery sweep only scans RUNNING tasks.
                or_(
                    and_(
                        Task.runner_id.is_(None),
                        Task.lease_attempt_id.is_(None),
                        Task.lease_expires_at.is_(None),
                    ),
                    Task.lease_expires_at < datetime.now(timezone.utc),
                ),
                Task.control_state.in_(("idle", ctx.status.value)),
            )
            .update(
                {
                    # Invalidate the old attempt in this same transaction so
                    # its heartbeat/finalizer cannot overwrite admission.
                    Task.runner_id: None,
                    Task.lease_attempt_id: None,
                    Task.lease_expires_at: None,
                    Task.control_state: "resume_requested",
                    Task.state_version: func.coalesce(Task.state_version, 0) + 1,
                },
                synchronize_session=False,
            )
        )
        if changed != 1:
            raise TaskResumeBusyError
        staged = stage_task_command(
            db,
            task_id=ctx.task_id,
            actor_user_id=ctx.task_owner_user_id,
            command_id=command_id,
            kind=TaskCommandKind.RESUME_INPUT,
            payload=payload.model_dump(mode="json"),
        )
        command_db_id = staged.staged_db_id
        try:
            db.commit()
        except Exception:
            db.close()
            # Prove the immutable command accepted the reply; do not generate
            # another ID after an uncertain acknowledgement.
            with get_session_local()() as check:
                row = check.get(TaskExecutionCommand, command_db_id)
                if (
                    row is None
                    or row.command_id != command_id
                    or row.payload != payload.model_dump(mode="json")
                ):
                    raise
        return command_db_id


def _read_reply_outcome(command_db_id: int) -> dict | None:
    with get_session_local()() as db:
        row = db.get(TaskExecutionCommand, command_db_id)
        if row is None:
            return {"outcome": "unavailable"}
        result = cast(dict[str, Any], row.result or {})
        if result.get("outcome"):
            return result
        if row.status == COMMAND_FAILED:
            return {"outcome": "unavailable"}
        if row.status == COMMAND_COMPLETED:
            task = db.get(Task, row.task_id)
            completed_at = row.completed_at
            if completed_at is not None and completed_at.tzinfo is None:
                completed_at = completed_at.replace(tzinfo=timezone.utc)
            stale_handoff = (
                completed_at is not None
                and (datetime.now(timezone.utc) - completed_at).total_seconds()
                > get_task_lease_ttl_seconds()
            )
            if (
                task is None
                or task.run_id != row.target_run_id
                or (
                    stale_handoff
                    and task.lease_attempt_id != result.get("lease_attempt_id")
                )
            ):
                # The handoff owner exited before recording an outcome. Never
                # reissue the reply or treat a later run as its result.
                return {"outcome": "unavailable"}
        return None


async def enqueue_resume_input(
    ctx: TaskReplyInput,
    *,
    source: Literal["sdk", "a2a"],
    message_id: str,
    command_id: str | None = None,
) -> TaskReplyResumeResult:
    from .task_event_bridge import get_task_event_bridge

    get_task_event_bridge().require_ready()
    command_db_id = await run_db_io_cancellation_safe(
        lambda: _admit_reply(ctx, source, message_id, command_id or uuid4().hex)
    )
    notify_task_command_dispatcher()
    # These APIs already wait for checkpoint validation and local scheduling.
    # Preserve that response boundary while preparation now runs on a worker.
    while (
        result := await run_db_io_cancellation_safe(
            lambda: _read_reply_outcome(command_db_id)
        )
    ) is None:
        await asyncio.sleep(0.25)
    if result["outcome"] == "busy":
        raise TaskResumeBusyError
    if result["outcome"] == "not_resumable":
        raise TaskResumeNotResumableError
    if result["outcome"] == "auto_model_unavailable":
        raise AutoModelUnavailableError("Configured Auto model is unavailable")
    if result["outcome"] != "accepted":
        raise CheckpointUnavailableError("Reply preparation did not complete")
    return TaskReplyResumeResult(
        run_id=result["run_id"],
        state_version=result["state_version"],
        control_state=result["control_state"],
    )


def _handoff(command: ClaimedTaskCommand) -> tuple[ResumeInput, TaskLease, int, dict]:
    try:
        payload = ResumeInput.model_validate(command.payload)
    except ValueError:
        raise TaskCommandRejected(
            "Invalid reply payload", reason="invalid_payload"
        ) from None
    runner = get_runner_id()
    with get_session_local()() as db:
        task = db.execute(
            select(Task).where(Task.id == command.task_id).with_for_update()
        ).scalar_one_or_none()
        claim_predicates = (
            TaskExecutionCommand.id == command.id,
            TaskExecutionCommand.status == COMMAND_PROCESSING,
            TaskExecutionCommand.claimed_by == runner,
            TaskExecutionCommand.attempt_count == command.attempt_count,
            TaskExecutionCommand.claim_expires_at > datetime.now(timezone.utc),
        )
        row = db.execute(
            select(TaskExecutionCommand).where(*claim_predicates).with_for_update()
        ).scalar_one_or_none()
        if task is None or row is None:
            raise TaskCommandRejected("Reply claim changed", reason="stale_claim")
        owner = db.get(User, task.user_id)
        if (
            owner is None
            or owner.actor_subject != row.task_owner_subject
            or owner.id != row.task_owner_user_id
            or owner.id != row.actor_user_id
            or owner.actor_subject != row.actor_subject
            or task.agent_id != payload.agent_id
            or task.source != payload.source
            or row.target_run_id != payload.run_id
        ):
            raise TaskCommandRejected(
                "Reply identity changed", reason="identity_changed"
            )
        lease = acquire_task_lease_no_commit(
            db,
            int(task.id),
            runner_id=runner,
            expected_run_id=payload.run_id,
            expected_status=TaskStatus(payload.prior_status),
            claim_predicates=(
                Task.control_state == "resume_requested",
                Task.state_version == row.target_state_version,
                exists(select(1).where(*claim_predicates)),
            ),
        )
        if lease is None:
            raise TaskCommandRejected("Reply state changed", reason="state_changed")
        db.refresh(task)
        state = {
            "run_id": lease.run_id,
            "lease_attempt_id": lease.attempt_id,
            "state_version": int(task.state_version),
            "control_state": task.control_state,
        }
        if not finish_task_command_no_commit(
            db,
            command.id,
            runner,
            result=state,
            expected_attempt_count=command.attempt_count,
            require_live_claim=True,
        ):
            raise TaskCommandRejected("Reply claim changed", reason="stale_claim")
        owner_id = int(task.user_id)
        try:
            db.commit()
        except Exception:
            db.close()
            with get_session_local()() as check:
                saved = check.get(TaskExecutionCommand, command.id)
                saved_task = check.get(Task, command.task_id)
                if not (
                    saved is not None
                    and saved.status == COMMAND_COMPLETED
                    and saved.result == state
                    and saved_task is not None
                    and saved_task.run_id == lease.run_id
                    and saved_task.runner_id == runner
                    and saved_task.lease_attempt_id == lease.attempt_id
                ):
                    raise
        return payload, lease, owner_id, state


def _record_outcome(command: ClaimedTaskCommand, state: dict, outcome: str) -> None:
    with get_session_local()() as db:
        # The immutable handoff attempt, not the task's now-possibly-released
        # lease, owns this command's preparation result.
        row = db.get(TaskExecutionCommand, command.id)
        if row is not None and row.status == COMMAND_COMPLETED and row.result == state:
            setattr(row, "result", {**state, "outcome": outcome})
            db.commit()


def settle_failed_reply_no_commit(db: Session, row: TaskExecutionCommand) -> None:
    """Restore this failed admission in the command's terminal transaction."""
    for status in (TaskStatus.PAUSED, TaskStatus.WAITING_FOR_USER):
        db.query(Task).filter(
            Task.id == row.task_id,
            Task.run_id == row.target_run_id,
            Task.state_version == row.target_state_version,
            Task.status == status,
            Task.control_state == "resume_requested",
            exists(
                select(1).where(
                    TaskExecutionCommand.id == row.id,
                    TaskExecutionCommand.status == COMMAND_FAILED,
                )
            ),
        ).update(
            {
                Task.control_state: status.value,
                Task.state_version: Task.state_version + 1,
            },
            synchronize_session=False,
        )


async def execute_resume_input(command: ClaimedTaskCommand) -> SettledTaskCommand:
    # Own handoff through scheduling so cancellation cannot abandon a lease
    # committed by the database thread before its result reaches this caller.
    return await drain_async_task_cancellation_safe(
        asyncio.create_task(_execute_resume_input(command))
    )


async def _execute_resume_input(command: ClaimedTaskCommand) -> SettledTaskCommand:
    from .task_resume import resume_a2a_task, resume_task_reply

    payload, lease, owner_id, state = await run_db_io_cancellation_safe(
        lambda: _handoff(command)
    )
    outcome = "unavailable"
    try:
        if payload.source == "sdk":
            await resume_task_reply(
                TaskReplyInput(
                    task_id=command.task_id,
                    agent_id=payload.agent_id,
                    task_owner_user_id=owner_id,
                    run_id=payload.run_id,
                    status=TaskStatus(payload.prior_status),
                    text=payload.text,
                ),
                preacquired_lease=lease,
                preacquired_state=state,
                turn_id=command.command_id,
            )
        else:
            await resume_a2a_task(
                agent_id=payload.agent_id,
                task_owner_user_id=owner_id,
                task_id=command.task_id,
                previous_run_id=payload.run_id,
                resumable_status=TaskStatus(payload.prior_status),
                text=payload.text,
                message_id=payload.message_id,
                preacquired_lease=lease,
            )
        outcome = "accepted"
    except TaskResumeBusyError:
        outcome = "busy"
    except (TaskResumeNotResumableError, CheckpointCorruptError):
        outcome = "not_resumable"
    except CheckpointAccessRefusedError as exc:
        outcome = "not_resumable" if exc.reason == "superseded_legacy" else "busy"
    except TaskResumeRetryableError:
        outcome = "retryable_unavailable"
    except CheckpointReadError:
        outcome = "unavailable"
    except AutoModelUnavailableError:
        outcome = "auto_model_unavailable"
    except Exception:
        logger.exception("Shared reply preparation failed task_id=%s", command.task_id)
    finally:
        await run_db_io_cancellation_safe(
            lambda: _record_outcome(command, state, outcome)
        )
    return SettledTaskCommand({**state, "outcome": outcome})
