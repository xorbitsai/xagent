"""Admit commands through a real owner for handoff-focused tests."""

from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_coordinator_runtime as runtime


async def claim_task_command(db, *, runner_id, command_db_id):
    row = db.get(TaskExecutionCommand, command_db_id)
    if row is None or row.status in transport.COMMAND_TERMINAL:
        return None
    coordinator = await runtime.get_task_coordinator_registry().ensure(row.task_id)
    if coordinator is None:
        return None

    # Handoff unit tests arrange a processing row before invoking the handler;
    # dispatcher integration tests exercise serialization through execute_command.
    token = runtime._current_coordinator.set(coordinator)
    try:
        return transport.claim_task_command(
            db, runner_id=runner_id, command_db_id=command_db_id
        )
    finally:
        runtime._current_coordinator.reset(token)


async def settle_command(command, operation):
    coordinator = await runtime.get_task_coordinator_registry().ensure(command.task_id)
    assert coordinator is not None

    async def settle():
        return operation()

    return await coordinator.execute_command(command, settle)


def claim_for_owner(db, owner_lease, command_db_id):
    """Arrange a command under a real lease in transaction-only handoff tests."""
    from types import SimpleNamespace

    token = runtime._current_coordinator.set(
        SimpleNamespace(task_id=owner_lease.task_id, lease=owner_lease)
    )
    try:
        return transport.claim_task_command(
            db, runner_id=owner_lease.runner_id, command_db_id=command_db_id
        )
    finally:
        runtime._current_coordinator.reset(token)
