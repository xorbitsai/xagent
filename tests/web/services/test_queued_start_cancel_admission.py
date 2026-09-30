"""Real controls settle capacity-waiting STARTs at the admission boundary."""

# Pytest fixture imports are intentionally shadowed by test parameters.
# ruff: noqa: F401, F811

from unittest.mock import Mock

import pytest
from cryptography.fernet import Fernet

from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRef
from xagent.web.models.agent import Agent
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_admission import TaskAdmissionTicket
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.task_command_terminal_event import TaskCommandTerminalEvent
from xagent.web.models.task_runtime_secret import TaskRuntimeSecret
from xagent.web.services import (
    a2a_task_cancel,
    external_task_cancel,
    task_command_execution,
)
from xagent.web.services import task_command_transport as transport
from xagent.web.services.chat_history_service import (
    DELIVERY_DISPATCHED,
    DELIVERY_FAILED,
)
from xagent.web.services.task_orchestrator import TaskTurnOrchestrator, TaskTurnPayload
from xagent.web.services.task_runtime_secrets import stage_runtime_values

from .test_task_execution_admission import (
    Execution,
    engine,
    enqueue,
    eventually,
    host,
)


@pytest.mark.parametrize("scope", ["a2a", "external"])
async def test_cancel_settles_capacity_waiting_start(scope, host, monkeypatch):
    """External runtime input is the supported host contract for its START."""
    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setattr(
        task_command_execution, "get_session_local", lambda: host.sessions
    )
    monkeypatch.setattr(a2a_task_cancel, "get_session_local", lambda: host.sessions)
    monkeypatch.setattr(
        external_task_cancel, "get_session_local", lambda: host.sessions
    )
    monkeypatch.setattr(
        "xagent.web.services.task_event_bridge.get_task_event_bridge", lambda: Mock()
    )

    blocker = enqueue(host)
    active = Execution(host)
    assert await transport.dispatch_one_task_command(active)

    try:
        with host.sessions() as db:
            agent = Agent(user_id=host.user, name=f"Queued {scope} cancellation")
            db.add(agent)
            db.flush()
            task = Task(
                user_id=host.user,
                agent_id=agent.id,
                source=scope,
                title=f"Queued {scope} task",
                status=TaskStatus.PENDING,
                control_state="pending",
                state_version=0,
                agent_config=(
                    {"a2a_context_id": "ctx-queued-cancel"} if scope == "a2a" else None
                ),
            )
            db.add(task)
            db.flush()
            task_id = int(task.id)
            agent_id = int(agent.id)
            turn_id = f"turn-queued-{scope}-cancel"
            if scope == "external":
                # The external host may stage connector inputs before using the
                # shared turn-acceptance seam. A2A currently produces no such input.
                stage_runtime_values(
                    db,
                    task_id=task_id,
                    turn_id=turn_id,
                    values_by_ref={
                        ConnectorRef("mcp", 1): {"secrets": {"token": "synthetic"}}
                    },
                )
            accepted = TaskTurnOrchestrator.claim_created_turn_no_commit(
                db,
                task_id=task_id,
                task_owner_user_id=host.user,
                actor_user_id=host.user,
                payload=TaskTurnPayload("Run the queued turn", turn_id=turn_id),
            )
            db.commit()
            start_id = int(accepted.command_db_id)

        with host.sessions() as db:
            assert db.query(TaskRuntimeSecret).filter_by(task_id=task_id).count() == (
                1 if scope == "external" else 0
            )

        executed = []

        async def execute(command):
            executed.append(command.id)
            return {}

        assert not await transport.dispatch_one_task_command(
            execute, command_db_id=start_id
        )
        assert executed == []

        payload = {"agent_id": agent_id, "target_state_version": 0}
        if scope == "external":
            payload.update({"scope": "external", "turn_id": turn_id})
        with host.sessions() as db:
            cancel = transport.enqueue_task_command(
                db,
                task_id=task_id,
                actor_user_id=host.user,
                command_id=f"cancel:{task_id}:0",
                kind=transport.TaskCommandKind.CANCEL,
                payload=payload,
            )

        assert await transport.dispatch_one_task_command(
            task_command_execution.execute_durable_task_command,
            command_db_id=cancel.command_id,
        )

        with host.sessions() as db:
            assert db.query(TaskRuntimeSecret).filter_by(task_id=task_id).count() == 0
            task = db.get(Task, task_id)
            queued_start = db.get(TaskExecutionCommand, start_id)
            cancel_row = db.get(TaskExecutionCommand, cancel.command_id)
            delivery = db.query(TaskChatMessage).filter_by(turn_id=turn_id).one()
            events = {
                event.task_command_id: event.outcome
                for event in db.query(TaskCommandTerminalEvent)
                .filter(
                    TaskCommandTerminalEvent.task_command_id.in_(
                        (start_id, cancel.command_id)
                    )
                )
                .all()
            }
            assert (task.status, task.error_message) == (
                TaskStatus.FAILED,
                (
                    "Stopped by the visitor."
                    if scope == "external"
                    else "Task canceled by A2A client."
                ),
            )
            assert queued_start.status == "failed"
            assert queued_start.error == "Task command stopped by a control request."
            assert queued_start.result == {
                "rejection_reason": "cancelled_before_admission"
            }
            assert cancel_row.status == "completed"
            assert delivery.delivery_status == (
                DELIVERY_DISPATCHED if scope == "external" else DELIVERY_FAILED
            )
            assert events == {start_id: "failed", cancel.command_id: "completed"}
    finally:
        active.finish.set()
        active.cleanup.set()

    def blocker_released() -> bool:
        with host.sessions() as db:
            return db.get(TaskAdmissionTicket, blocker.command_id) is None

    await eventually(blocker_released)
    assert not await transport.dispatch_one_task_command(
        execute, command_db_id=start_id
    )
    assert executed == []
