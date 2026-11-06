"""Accept legacy execute_task without creating another transcript message."""

import logging
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import select

from ...config import get_shared_task_execution_enabled
from ..models.database import get_session_local
from ..models.task import Task, TaskStatus
from ..models.task_command import TaskExecutionCommand
from ..models.user import User
from .mcp_runtime import MCPBuiltinOAuthActorPolicyRequiredError
from .task_command_transport import (
    ClaimedTaskCommand,
    TaskCommandRejected,
    notify_task_command_dispatcher,
)
from .task_event_bridge import get_task_event_bridge
from .task_orchestrator import TaskTurnError, reserve_task_start_no_commit
from .task_runtime import mcp_runtime_authorization_policy_required
from .task_start_protocol import (
    ExistingExecutionContext,
    TaskStartPayload,
    read_task_start_command,
    stage_task_start_command,
)

logger = logging.getLogger(__name__)


def enqueue_existing_execution(
    *,
    task_id: int,
    task_owner_user_id: int,
    task_description: str,
    context: dict,
    actor_user_id: int,
) -> str:
    run_id, _ = stage_existing_execution(
        task_id=task_id,
        task_owner_user_id=task_owner_user_id,
        task_description=task_description,
        context=context,
        actor_user_id=actor_user_id,
    )
    notify_task_command_dispatcher()
    return run_id


def stage_existing_execution(
    *,
    task_id: int,
    task_owner_user_id: int,
    task_description: str,
    context: dict,
    actor_user_id: int,
) -> tuple[str, int]:
    """Durably accept ``execute_task`` as a START of kind ``existing``.

    Returns the START's run id and its command row id, so the caller can
    dispatch it promptly. Works in both modes: shared execution additionally
    requires the event bridge; combined mode stages the same command and its
    local dispatcher consumes it (``execute_existing_start_local``).
    """
    shared = get_shared_task_execution_enabled()
    if shared:
        get_task_event_bridge().require_ready()
    run_id, turn_id = str(uuid4()), uuid4().hex
    with get_session_local()() as db:
        task = db.execute(
            select(Task).where(Task.id == task_id).with_for_update()
        ).scalar_one_or_none()
        actor = db.get(User, actor_user_id)
        if (
            task is None
            or task.user_id != task_owner_user_id
            or actor is None
            or (actor.id != task_owner_user_id and not actor.is_admin)
        ):
            raise TaskTurnError("task_not_found")
        if mcp_runtime_authorization_policy_required(task.agent_config):
            raise MCPBuiltinOAuthActorPolicyRequiredError(
                "Legacy execution does not support actor-marked tasks"
            )
        changed = reserve_task_start_no_commit(
            db,
            task_id=task_id,
            task_owner_user_id=task_owner_user_id,
            statuses=tuple(
                status for status in TaskStatus if status != TaskStatus.RUNNING
            ),
        )
        if changed != 1:
            raise TaskTurnError("busy")
        db.refresh(task)
        if task.conversation_storage_version == 2:
            from .task_execution_event_writer import append_fact_no_commit

            # V1 executes the stored description without another chat row.
            # V2 still needs an accepted input occurrence before the command
            # can be acknowledged or its runtime user_message can be emitted.
            append_fact_no_commit(
                db,
                task_id=task_id,
                run_id=run_id,
                turn_id=turn_id,
                kind="input_accepted",
                key=f"message:{turn_id}",
                payload={
                    "user_id": task_owner_user_id,
                    "role": "user",
                    "content": task_description,
                    "message_type": "user_message",
                    "attachments": None,
                    "interactions": None,
                    "turn_id": turn_id,
                },
            )
        start = TaskStartPayload(
            version=1,
            run_id=run_id,
            expected_run_id=cast(str | None, task.run_id),
            state_version=int(task.state_version),
            turn_id=turn_id,
            kind="existing",
            message=task_description,
            execution_message=task_description,
            file_ids=[],
            existing_context=ExistingExecutionContext.model_validate(context),
        )
        staged = stage_task_start_command(
            db, task_id=task_id, actor_user_id=actor_user_id, start=start
        )
        try:
            db.commit()
        except Exception:
            db.close()
            with get_session_local()() as check:
                saved = check.get(TaskExecutionCommand, staged.staged_db_id)
                if (
                    saved is None
                    or saved.command_id != turn_id
                    or saved.payload != start.model_dump(mode="json")
                ):
                    raise
    return run_id, int(staged.staged_db_id)


async def execute_existing_start_local(command: ClaimedTaskCommand) -> dict[str, Any]:
    """Consume an accepted ``existing`` START in combined mode.

    Shared execution hands the START to a task coordinator; combined mode has
    none, so the claimed command schedules the same local execution the
    WebSocket used to start inline. The command completes once the execution is
    scheduled, not when the Agent finishes, as in shared mode.
    """
    from .db_runtime import run_db_io_cancellation_safe
    from .task_orchestrator import (
        TaskTurnError,
        TaskTurnOrchestrator,
        TaskTurnPayload,
    )

    try:
        start = read_task_start_command(command)
    except ValueError as exc:
        raise TaskCommandRejected(
            "START payload is invalid", reason="invalid_start"
        ) from exc
    if start.kind != "existing" or start.existing_context is None:
        raise TaskCommandRejected(
            "Only existing execution is consumed without a task coordinator",
            reason="unsupported_start",
        )

    def load_task() -> tuple[int, str | None, TaskStatus] | None:
        with get_session_local()() as db:
            row = db.execute(
                select(Task.user_id, Task.source, Task.status).where(
                    Task.id == command.task_id
                )
            ).one_or_none()
            if row is None:
                return None
            return int(row.user_id), row.source, row.status

    loaded = await run_db_io_cancellation_safe(load_task)
    if loaded is None:
        raise TaskCommandRejected("Task no longer exists", reason="task_missing")
    owner_user_id, source, status = loaded
    if status == TaskStatus.RUNNING:
        raise TaskCommandRejected(
            "Task is already running", reason="start_state_changed"
        )
    try:
        await TaskTurnOrchestrator.schedule_existing_task_execution(
            task_id=command.task_id,
            task_owner_user_id=owner_user_id,
            task_source=source,
            payload=TaskTurnPayload(
                transcript_message=start.message,
                execution_message=start.execution_message,
                turn_id=start.turn_id,
            ),
            context=start.existing_context.model_dump(exclude_none=True),
            actor_user_id=command.actor_user_id,
        )
    except (TaskTurnError, MCPBuiltinOAuthActorPolicyRequiredError) as exc:
        logger.warning(
            "existing execution rejected task=%s command=%s: %s",
            command.task_id,
            command.command_id,
            exc,
        )
        raise TaskCommandRejected(
            "Existing execution could not be scheduled", reason="start_state_changed"
        ) from exc
    return {"run_id": start.run_id}
