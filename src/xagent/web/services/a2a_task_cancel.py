"""Execute A2A cancellation against its exact durable-command target."""

from typing import Any

from sqlalchemy import func, update

from ..models.database import get_session_local
from ..models.task import Task, TaskStatus
from .a2a_protocol import A2ATaskSnapshot, a2a_error
from .db_runtime import run_db_io_cancellation_safe
from .task_execution_controller import StaleTaskRunError, TaskControlState

_TERMINAL_STATUSES = {TaskStatus.COMPLETED, TaskStatus.FAILED}


def _task_run_id(task: Task) -> str | None:
    run_id = getattr(task, "run_id", None)
    return str(run_id) if run_id is not None else None


def _load_cancelable_a2a_task_sync(
    *,
    task_id: int,
    agent_id: int,
    expected_run_id: str | None,
    expected_state_version: int,
) -> A2ATaskSnapshot:
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = (
            db.query(Task)
            .filter(
                Task.id == task_id,
                Task.agent_id == agent_id,
                Task.source == "a2a",
            )
            .first()
        )
        if task is None:
            raise a2a_error("task_not_found", "Task not found.", status_code=404)
        if _is_completed_a2a_cancel_target(
            task,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
        ):
            return A2ATaskSnapshot.from_task(task)
        _assert_a2a_cancel_target(
            task,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
        )
        return A2ATaskSnapshot.from_task(task)


def _assert_a2a_cancel_target(
    task: Task,
    *,
    expected_run_id: str | None,
    expected_state_version: int,
) -> None:
    """Reject a cancel command whose immutable task-state target is stale."""

    current_run_id = _task_run_id(task)
    current_state_version = int(task.state_version or 0)
    if (
        current_run_id != expected_run_id
        or current_state_version != expected_state_version
    ):
        raise StaleTaskRunError(
            f"task {task.id} changed from run/version "
            f"{expected_run_id}/{expected_state_version} to "
            f"{current_run_id}/{current_state_version}"
        )


def _is_completed_a2a_cancel_target(
    task: Task,
    *,
    expected_run_id: str | None,
    expected_state_version: int,
) -> bool:
    """Validate an idempotent cancel replay against its immutable target."""

    agent_config: dict[str, Any] = (
        task.agent_config if isinstance(task.agent_config, dict) else {}
    )
    if agent_config.get("a2a_state") != "TASK_STATE_CANCELED":
        return False

    current_run_id = _task_run_id(task)
    current_state_version = int(task.state_version or 0)
    is_exact_completion = (
        current_run_id == expected_run_id
        and current_state_version
        in {expected_state_version, expected_state_version + 1}
        and task.status == TaskStatus.FAILED
        and task.control_state == TaskControlState.FAILED.value
    )
    if not is_exact_completion:
        raise StaleTaskRunError(
            f"task {task.id} has a canceled marker outside command target "
            f"{expected_run_id}/{expected_state_version}; current state is "
            f"{current_run_id}/{current_state_version}/"
            f"{task.status.value}/{task.control_state}"
        )
    return True


def _finalize_a2a_cancel_sync(
    *,
    task_id: int,
    agent_id: int,
    expected_run_id: str | None,
    expected_state_version: int,
    local_cancel_requested: bool,
) -> A2ATaskSnapshot:
    """Atomically persist cancellation for one exact task-state target."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = (
            db.query(Task)
            .filter(
                Task.id == task_id,
                Task.agent_id == agent_id,
                Task.source == "a2a",
            )
            .first()
        )
        if task is None:
            raise a2a_error("task_not_found", "Task not found.", status_code=404)
        agent_config: dict[str, Any] = (
            dict(task.agent_config) if isinstance(task.agent_config, dict) else {}
        )
        if _is_completed_a2a_cancel_target(
            task,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
        ):
            return A2ATaskSnapshot.from_task(task)

        current_run_id = _task_run_id(task)
        current_state_version = int(task.state_version or 0)
        same_run = current_run_id == expected_run_id
        settled_local_cancel = (
            local_cancel_requested
            and same_run
            and current_state_version == expected_state_version + 1
            and task.status == TaskStatus.FAILED
            and task.control_state == TaskControlState.FAILED.value
            and task.runner_id is None
            and task.lease_expires_at is None
        )
        direct_cancel = (
            same_run
            and current_state_version == expected_state_version
            and task.status not in _TERMINAL_STATUSES
        )
        if not settled_local_cancel and not direct_cancel:
            raise StaleTaskRunError(
                f"task {task.id} cannot finalize cancel target "
                f"{expected_run_id}/{expected_state_version}; current state is "
                f"{current_run_id}/{current_state_version}/"
                f"{task.status.value}/{task.control_state}"
            )

        agent_config["a2a_state"] = "TASK_STATE_CANCELED"
        target_state_version = current_state_version
        final_state_version = (
            current_state_version
            if settled_local_cancel
            else expected_state_version + 1
        )
        statement = (
            update(Task)
            .where(
                Task.id == task_id,
                Task.agent_id == agent_id,
                Task.source == "a2a",
                func.coalesce(Task.state_version, 0) == target_state_version,
            )
            .values(
                agent_config=agent_config,
                status=TaskStatus.FAILED,
                control_state=TaskControlState.FAILED.value,
                state_version=final_state_version,
                runner_id=None,
                lease_attempt_id=None,
                lease_expires_at=None,
                last_heartbeat_at=None,
                output=None,
                error_message="Task canceled by A2A client.",
            )
        )
        if expected_run_id is None:
            statement = statement.where(Task.run_id.is_(None))
        else:
            statement = statement.where(Task.run_id == expected_run_id)
        if settled_local_cancel:
            statement = statement.where(
                Task.status == TaskStatus.FAILED,
                Task.control_state == TaskControlState.FAILED.value,
                Task.runner_id.is_(None),
                Task.lease_expires_at.is_(None),
            )
        else:
            statement = statement.where(Task.status.notin_(_TERMINAL_STATUSES))

        updated = db.execute(
            statement.returning(Task).execution_options(synchronize_session=False)
        ).scalar_one_or_none()
        if updated is None:
            db.rollback()
            raise StaleTaskRunError(
                f"task {task_id} changed while finalizing cancel target "
                f"{expected_run_id}/{expected_state_version}"
            )
        snapshot = A2ATaskSnapshot.from_task(updated)
        db.commit()
        return snapshot


async def cancel_a2a_task(
    *,
    task_id: int,
    agent_id: int,
    expected_run_id: str | None,
    expected_state_version: int,
) -> A2ATaskSnapshot:
    """Cancel one exact durable-command target while the caller owns its gate."""

    task = await run_db_io_cancellation_safe(
        lambda: _load_cancelable_a2a_task_sync(
            task_id=task_id,
            agent_id=agent_id,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
        )
    )
    if task.agent_config.get("a2a_state") == "TASK_STATE_CANCELED":
        return task
    if task.status in _TERMINAL_STATUSES:
        raise a2a_error(
            "task_not_cancelable",
            "Task is not in a cancelable state.",
            status_code=400,
            details={"taskId": task.id},
        )

    from .task_execution import background_task_manager

    cancel_outcome = await background_task_manager.cancel_task(task.id)
    finalized = await run_db_io_cancellation_safe(
        lambda: _finalize_a2a_cancel_sync(
            task_id=task_id,
            agent_id=agent_id,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
            local_cancel_requested=cancel_outcome.requested,
        )
    )
    return finalized
