"""Atomically hand one accepted START to one exact execution lease."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, cast

from pydantic import ValidationError
from sqlalchemy import exists, func, select
from sqlalchemy.orm import Session

from ...core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from ..models.chat_message import TaskChatMessage
from ..models.database import get_session_local
from ..models.task import Task, TaskStatus
from ..models.task_command import TaskExecutionCommand
from ..models.uploaded_file import UploadedFile
from ..models.user import User
from .chat_history_service import DELIVERY_FAILED, mark_user_message_delivery
from .db_runtime import drain_async_task_cancellation_safe, run_db_io_cancellation_safe
from .mcp_runtime import (
    MCPActorAuthorizationPolicy,
    MCPBuiltinOAuthActorPolicyRequiredError,
)
from .task_actor_policy import load_shared_actor_policy
from .task_command_transport import (
    COMMAND_COMPLETED,
    COMMAND_FAILED,
    COMMAND_PROCESSING,
    ClaimedTaskCommand,
    SettledTaskCommand,
    TaskCommandRejected,
    command_identity_matches_task,
    finish_task_command_no_commit,
)
from .task_coordinator_service import TaskLease as TaskOwnerLease
from .task_coordinator_service import (
    begin_task_execution_no_commit,
    lock_task_lease_no_commit,
)
from .task_execution_controller import task_control_snapshot
from .task_lease_service import TaskLease
from .task_orchestrator import (
    TaskTurnPayload,
    TurnKind,
    _ClaimedTurn,
    _EnqueuedTurn,
    _retire_turn_session_best_effort,
    _schedule_committed_turn,
    timezone_schedule_context,
)
from .task_runtime_secrets import delete_runtime_values_no_commit, load_runtime_values
from .task_start_protocol import TaskStartPayload, read_task_start_command

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _StartHandoff:
    task_owner_user_id: int
    start: TaskStartPayload
    claimed: _ClaimedTurn
    actor_policy: MCPActorAuthorizationPolicy | None = None


def reconcile_start_acceptance(
    *,
    task_id: int,
    task_owner_user_id: int,
    payload: TaskTurnPayload,
    accepted: _EnqueuedTurn,
) -> bool:
    """Prove acceptance without requiring the worker to remain queued."""
    with get_session_local()() as db:
        command = db.get(TaskExecutionCommand, accepted.command_db_id)
        task = db.get(Task, task_id)
        if command is None or task is None or task.user_id != task_owner_user_id:
            return False
        owner = db.get(User, task_owner_user_id)
        if owner is None or command.task_owner_subject != owner.actor_subject:
            return False
        if (
            command.task_id != task_id
            or command.command_id != payload.turn_id
            or command.target_run_id != accepted.run_id
            or command.payload != accepted.command_payload
        ):
            return False
        body = command.payload
        if (
            body.get("message") != payload.transcript_message
            or body.get("execution_message") != payload.execution_message
            or body.get("file_ids") != list(payload.file_ids)
            or body.get("before_message_id") != accepted.before_message_id
        ):
            return False
        message = db.execute(
            select(TaskChatMessage.id).where(
                TaskChatMessage.task_id == task_id,
                TaskChatMessage.turn_id == payload.turn_id,
                TaskChatMessage.role == "user",
                TaskChatMessage.content == payload.transcript_message.strip(),
            )
        ).first()
        if message is None:
            return False
        bound = (
            db.query(UploadedFile)
            .filter(
                UploadedFile.file_id.in_(payload.file_ids),
                UploadedFile.task_id == task_id,
                UploadedFile.user_id == task_owner_user_id,
            )
            .count()
        )
        if bound != len(set(payload.file_ids)):
            return False
        if command.payload.get("runtime_values_ref") is not None and (
            command.status in ("pending", "processing")
            or (
                task.run_id == accepted.run_id
                and task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            )
        ):
            try:
                load_runtime_values(
                    db,
                    task=task,
                    turn_id=payload.turn_id,
                    required=True,
                    run_id=accepted.run_id,
                )
            except ConnectorRuntimeError:
                return False
        return True


def _reject_start(
    db: Session, row: TaskExecutionCommand, reason: str
) -> SettledTaskCommand:
    """Fence both writes, including on SQLite where FOR UPDATE is ignored."""
    owned = db.query(TaskExecutionCommand).filter(
        TaskExecutionCommand.id == row.id,
        TaskExecutionCommand.status == COMMAND_PROCESSING,
        TaskExecutionCommand.claimed_by == row.claimed_by,
        TaskExecutionCommand.attempt_count == row.attempt_count,
        TaskExecutionCommand.claim_expires_at > datetime.now(timezone.utc),
    )
    result = {"rejection_reason": reason}
    if (
        owned.update(
            {
                TaskExecutionCommand.status: COMMAND_FAILED,
                TaskExecutionCommand.error: reason,
                TaskExecutionCommand.result: result,
                TaskExecutionCommand.claimed_by: None,
                TaskExecutionCommand.claim_expires_at: None,
                TaskExecutionCommand.completed_at: datetime.now(timezone.utc),
            },
            synchronize_session=False,
        )
        != 1
    ):
        db.rollback()
        raise TaskCommandRejected(
            "START rejection lost its claim", reason="stale_claim"
        )
    from .task_command_terminal_events import stage_terminal_event

    settle_failed_start_no_commit(db, row)
    stage_terminal_event(db, command_db_id=int(row.id))
    db.commit()
    return SettledTaskCommand(result)


def settle_failed_start_no_commit(db: Session, row: TaskExecutionCommand) -> None:
    """Reject the accepted input, preserving any previous run and its owner."""
    mark_user_message_delivery(
        db,
        task_id=int(row.task_id),
        turn_id=str(row.command_id),
        status=DELIVERY_FAILED,
    )
    delete_runtime_values_no_commit(
        db, task_id=int(row.task_id), turn_id=str(row.command_id)
    )
    # A never-started new Task needs a terminal result. An append's failure
    # belongs to its command and must not overwrite the previous run's result.
    changed = (
        db.query(Task)
        .filter(
            Task.id == row.task_id,
            Task.run_id == row.payload.get("expected_run_id"),
            Task.state_version == row.target_state_version,
            Task.status == TaskStatus.PENDING,
            exists(
                select(1).where(
                    TaskExecutionCommand.id == row.id,
                    TaskExecutionCommand.status == COMMAND_FAILED,
                )
            ),
        )
        .update(
            {
                Task.status: TaskStatus.FAILED,
                Task.control_state: "failed",
                Task.state_version: func.coalesce(Task.state_version, 0) + 1,
                Task.error_message: "Task could not start.",
            },
            synchronize_session=False,
        )
    )
    if changed:
        from .task_orchestrator import sync_trigger_run_status
        from .workforce_runtime import sync_workforce_run_status

        task = db.get(Task, row.task_id)
        assert task is not None
        db.refresh(task)
        sync_workforce_run_status(db, task, TaskStatus.FAILED)
        sync_trigger_run_status(
            db,
            task,
            TaskStatus.FAILED,
            error_message=cast(str | None, task.error_message),
        )


def _commit_handoff(
    command: ClaimedTaskCommand, owner_lease: TaskOwnerLease
) -> _StartHandoff | SettledTaskCommand:
    db = get_session_local()()
    runner = owner_lease.runner_id
    try:
        if not lock_task_lease_no_commit(db, owner_lease):
            raise TaskCommandRejected("Task owner changed", reason="stale_owner")
        # Same lock order as acceptance. On SQLite the CAS and subsequent
        # command completion are covered by the writer transaction instead.
        task = db.execute(
            select(Task).where(Task.id == command.task_id).with_for_update()
        ).scalar_one_or_none()
        row = db.execute(
            select(TaskExecutionCommand)
            .where(
                TaskExecutionCommand.id == command.id,
                TaskExecutionCommand.status == COMMAND_PROCESSING,
                TaskExecutionCommand.claimed_by == runner,
                TaskExecutionCommand.attempt_count == command.attempt_count,
                TaskExecutionCommand.claim_expires_at > datetime.now(timezone.utc),
            )
            .with_for_update()
        ).scalar_one_or_none()
        if task is None or row is None:
            raise TaskCommandRejected(
                "START claim is no longer owned", reason="stale_claim"
            )
        try:
            start = read_task_start_command(command)
        except (ValidationError, ValueError):
            return _reject_start(db, row, "invalid_start")
        if not command_identity_matches_task(db, task, row):
            return _reject_start(db, row, "start_identity_changed")
        if row.target_state_version != start.state_version:
            return _reject_start(db, row, "invalid_start_version")
        if (
            task.run_id != start.expected_run_id
            or task.state_version != start.state_version
            or task.status == TaskStatus.RUNNING
            or (start.kind == "create" and task.status != TaskStatus.PENDING)
        ):
            return _reject_start(db, row, "start_state_changed")
        if start.channel is not None:
            from .channel_runtime import (
                ChannelAuthorizationError,
                ChannelConfigurationError,
                _load_channel_owner_sync,
            )

            try:
                channel_owner = _load_channel_owner_sync(
                    db,
                    channel_id=start.channel.channel_id,
                    external_user_id=start.channel.external_user_id,
                )
            except (ChannelAuthorizationError, ChannelConfigurationError):
                return _reject_start(db, row, "channel_unavailable")
            if (
                task.channel_id != start.channel.channel_id
                or channel_owner.user_id != task.user_id
            ):
                return _reject_start(db, row, "channel_identity_changed")
        try:
            actor_policy = load_shared_actor_policy(
                task, is_create=start.kind == "create"
            )
        except (MCPBuiltinOAuthActorPolicyRequiredError, ValueError):
            return _reject_start(db, row, "actor_policy_required")
        try:
            if start.runtime_values_ref is not None:
                load_runtime_values(
                    db,
                    task=task,
                    turn_id=start.runtime_values_ref,
                    required=True,
                    run_id=start.run_id,
                )
        except ConnectorRuntimeError:
            # Never expose decryption details or permit missing optional
            # secrets to silently change the accepted execution input.
            return _reject_start(db, row, "runtime_values_unavailable")
        execution = begin_task_execution_no_commit(
            db,
            owner_lease,
            expected=task_control_snapshot(task),
            new_run=True,
            run_id=start.run_id,
        )
        if execution is None:
            return _reject_start(db, row, "start_state_changed")
        db.refresh(task)
        setattr(task, "input", start.message)
        from .workforce_runtime import sync_workforce_run_status

        sync_workforce_run_status(db, task, TaskStatus.RUNNING)
        lease = TaskLease(
            task_id=owner_lease.task_id,
            runner_id=owner_lease.runner_id,
            attempt_id=owner_lease.attempt_id,
            run_id=execution.run_id,
        )
        result = {"run_id": start.run_id, "lease_attempt_id": lease.attempt_id}
        if not finish_task_command_no_commit(
            db,
            command.id,
            runner,
            result=result,
            expected_attempt_count=command.attempt_count,
            require_live_claim=True,
        ):
            db.rollback()
            raise TaskCommandRejected(
                "START claim changed during handoff", reason="stale_claim"
            )
        handoff = _StartHandoff(
            task_owner_user_id=int(task.user_id),
            start=start,
            actor_policy=actor_policy,
            claimed=_ClaimedTurn(
                status=TaskStatus.RUNNING,
                updated_at=cast(datetime | None, task.updated_at),
                before_message_id=start.before_message_id,
                task_source=cast(str | None, task.source),
                run_id=start.run_id,
                state_version=int(task.state_version),
                control_state="running",
                agent_config=cast(dict[str, Any] | None, task.agent_config),
                task_lease=lease,
            ),
        )
        try:
            db.commit()
        except Exception:
            _retire_turn_session_best_effort(db, task_id=int(task.id))
            if not _handoff_was_committed(command.id, lease):
                # No automatic replay after uncertain execution admission.
                # The processing claim or exact lease remains for recovery.
                raise
        return handoff
    finally:
        _retire_turn_session_best_effort(db, task_id=command.task_id)


def _handoff_was_committed(command_id: int, lease: TaskLease) -> bool:
    with get_session_local()() as db:
        row = db.get(TaskExecutionCommand, command_id)
        task = db.get(Task, lease.task_id)
        return bool(
            row is not None
            and task is not None
            and row.status == COMMAND_COMPLETED
            and row.result
            == {"run_id": lease.run_id, "lease_attempt_id": lease.attempt_id}
            and task.run_id == lease.run_id
            and task.runner_id == lease.runner_id
            and task.lease_attempt_id == lease.attempt_id
        )


async def execute_task_start(command: ClaimedTaskCommand) -> SettledTaskCommand:
    # Own handoff through scheduling so cancellation cannot abandon a lease
    # committed by the database thread before its result reaches this caller.
    return await drain_async_task_cancellation_safe(
        asyncio.create_task(_execute_task_start(command))
    )


async def _execute_task_start(command: ClaimedTaskCommand) -> SettledTaskCommand:
    from .task_execution_controller import task_execution_controller

    # Command completion releases the durable queue before registration.
    # Keep the local handoff barrier until the exact execution is registered.
    async with task_execution_controller.command(command.task_id):
        from .task_coordinator_runtime import current_task_coordinator

        coordinator = current_task_coordinator(command.task_id)
        assert coordinator is not None and coordinator.lease is not None
        owner_lease = coordinator.lease
        handoff = await run_db_io_cancellation_safe(
            lambda: _commit_handoff(command, owner_lease)
        )
        if isinstance(handoff, SettledTaskCommand):
            return handoff
        start = handoff.start
        context = timezone_schedule_context(start.timezone) or {}
        config = handoff.claimed.agent_config or {}
        for key in ("trigger_id", "trigger_run_id", "trigger_type", "trigger_test"):
            if key in config:
                context[key] = config[key]
        if start.kind == "channel":
            from .task_orchestrator import _schedule_bg, settle_task_lease_isolated

            try:
                _schedule_bg(
                    task_id=command.task_id,
                    task_owner_user_id=handoff.task_owner_user_id,
                    task_source=handoff.claimed.task_source,
                    run_id=start.run_id,
                    task_lease=handoff.claimed.task_lease,
                    payload=TaskTurnPayload(
                        transcript_message=start.message,
                        execution_message=start.execution_message,
                        file_ids=tuple(start.file_ids),
                        turn_id=start.turn_id,
                    ),
                    force_fresh=False,
                    context=None,
                    before_message_id=start.before_message_id,
                    channel_command=command,
                )
            except BaseException:
                await run_db_io_cancellation_safe(
                    lambda: settle_task_lease_isolated(
                        handoff.claimed.task_lease,
                        error_message="Channel execution scheduling failed",
                    )
                )
                raise
            return SettledTaskCommand({"run_id": start.run_id})
        if start.kind == "existing":
            from .task_orchestrator import _schedule_bg, settle_task_lease_isolated

            assert start.existing_context is not None
            try:
                _schedule_bg(
                    task_id=command.task_id,
                    task_owner_user_id=handoff.task_owner_user_id,
                    task_source=handoff.claimed.task_source,
                    run_id=start.run_id,
                    task_lease=handoff.claimed.task_lease,
                    payload=TaskTurnPayload(
                        transcript_message=start.message,
                        execution_message=start.execution_message,
                        turn_id=start.turn_id,
                    ),
                    force_fresh=False,
                    context=start.existing_context.model_dump(exclude_none=True),
                )
            except BaseException:
                await run_db_io_cancellation_safe(
                    lambda: settle_task_lease_isolated(
                        handoff.claimed.task_lease,
                        error_message="Existing execution scheduling failed",
                    )
                )
                raise
            return SettledTaskCommand({"run_id": start.run_id})
        # This returns after local scheduling, not after Agent execution. The
        # command is already terminal, so subsequent controls can proceed.
        await _schedule_committed_turn(
            task_id=command.task_id,
            task_owner_user_id=handoff.task_owner_user_id,
            payload=TaskTurnPayload(
                transcript_message=start.message,
                execution_message=start.execution_message,
                file_ids=tuple(start.file_ids),
                turn_id=start.turn_id,
            ),
            claimed=handoff.claimed,
            kind=TurnKind(start.kind),
            force_fresh=start.force_fresh,
            context=context,
            mcp_runtime_authorization_policy=handoff.actor_policy,
        )
        return SettledTaskCommand({"run_id": start.run_id})
