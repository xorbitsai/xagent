"""Run code as a real lease owner in handoff-focused tests: admit commands,
or call finalizers directly.
"""

from contextlib import contextmanager

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


class _LeaseOwner:
    """The in-process owner of one lease, for finalizers called directly.

    It borrows the real ownership comparison, so a mismatched lease is
    refused exactly as a live coordinator would refuse it.
    """

    owns_execution = runtime.TaskCoordinator.owns_execution

    def __init__(self, lease):
        self.task_id = lease.task_id
        self.lease = lease


@contextmanager
def owning_coordinator(lease):
    """Run the block as the coordinator that owns ``lease``."""
    token = runtime._current_coordinator.set(_LeaseOwner(lease))
    try:
        yield
    finally:
        runtime._current_coordinator.reset(token)
