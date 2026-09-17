"""Real acquisition and heartbeat registration for live-input API tests."""

import pytest

from xagent.web.services import task_lease_service as leases


@pytest.fixture
async def live_task_lease():
    manager = leases._get_task_lease_heartbeat_manager()
    registrations = []

    def acquire(db, task):
        lease = leases.acquire_task_lease(
            db,
            int(task.id),
            runner_id=leases.get_runner_id(),
            expected_run_id=str(task.run_id),
        )
        assert lease is not None
        registrations.append(manager.register(lease))
        db.refresh(task)
        return lease

    try:
        yield acquire
    finally:
        for registration in registrations:
            await registration.close()
        await manager.wait_until_idle()
