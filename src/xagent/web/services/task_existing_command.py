"""Accept legacy execute_task without creating another transcript message."""

from uuid import uuid4

from sqlalchemy import func, select

from ..models.database import get_session_local
from ..models.task import Task, TaskStatus
from ..models.task_command import TaskExecutionCommand
from ..models.user import User
from .mcp_runtime import MCPBuiltinOAuthActorPolicyRequiredError
from .task_command_transport import notify_task_command_dispatcher
from .task_event_bridge import get_task_event_bridge
from .task_orchestrator import TaskTurnError
from .task_runtime import mcp_runtime_authorization_policy_required
from .task_start_protocol import (
    ExistingExecutionContext,
    TaskStartPayload,
    stage_task_start_command,
)


def enqueue_existing_execution(
    *,
    task_id: int,
    task_owner_user_id: int,
    task_description: str,
    context: dict,
    actor_user_id: int,
) -> str:
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
        changed = (
            db.query(Task)
            .filter(Task.id == task_id, Task.status != TaskStatus.RUNNING)
            .update(
                {
                    Task.status: TaskStatus.RUNNING,
                    Task.run_id: run_id,
                    Task.control_state: "running",
                    Task.state_version: func.coalesce(Task.state_version, 0) + 1,
                    Task.runner_id: None,
                    Task.lease_attempt_id: None,
                    Task.lease_expires_at: None,
                    Task.last_heartbeat_at: None,
                    Task.last_checkpoint_event_id: None,
                    Task.last_checkpoint_trace_event_id: None,
                },
                synchronize_session=False,
            )
        )
        if changed != 1:
            raise TaskTurnError("busy")
        db.refresh(task)
        start = TaskStartPayload(
            version=1,
            run_id=run_id,
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
    notify_task_command_dispatcher()
    return run_id
