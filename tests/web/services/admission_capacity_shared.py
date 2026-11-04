"""Saturate and release one admission bucket for capacity-wait tests."""

from xagent.web.models.database import get_session_local
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_admission import TaskAdmissionTicket
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.services.task_command_transport import (
    TaskCommandKind,
    stage_task_command,
)


def saturate_bucket(owner_user_id, agent_id, bucket="tenant:batch", *, held=True):
    """Hold the bucket's only slot with another task's live execution.

    With ``held=False`` the other task's START only waits, so a later command
    in the bucket is blocked by an older waiter while no slot is occupied.
    """
    with get_session_local()() as db:
        holder = Task(
            user_id=owner_user_id,
            agent_id=agent_id,
            title="Holder",
            source="sdk",
            status=TaskStatus.RUNNING,
            run_id="run-holder",
            control_state="running",
            runner_id="worker-9",
            lease_attempt_id="attempt-9",
        )
        db.add(holder)
        db.flush()
        staged = stage_task_command(
            db,
            task_id=holder.id,
            actor_user_id=owner_user_id,
            command_id="start-holder",
            kind=TaskCommandKind.START,
            payload={},
        )
        ticket = db.get(TaskAdmissionTicket, staged.staged_db_id)
        assert ticket is not None and ticket.bucket_key == bucket
        if held:
            ticket.runner_id = "worker-9"
            ticket.owner_attempt_id = "attempt-9"
        db.commit()
        return staged.staged_db_id


def release_bucket(holding_command_id):
    """Finish the holder's execution so the slot can be reserved again."""
    with get_session_local()() as db:
        db.delete(db.get(TaskAdmissionTicket, holding_command_id))
        db.get(TaskExecutionCommand, holding_command_id).status = "completed"
        db.commit()
