"""One admitted slot per active execution, independent of stamped ticket rows."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import literal

from tests.web.services.test_task_execution_admission import (
    Execution,
    dispatch_next,
)
from tests.web.services.test_task_execution_admission import engine as engine_fixture
from tests.web.services.test_task_execution_admission import (
    enqueue,
)
from tests.web.services.test_task_execution_admission import host as host_fixture
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_admission import TaskAdmissionTicket
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.services import task_admission_observation as observation
from xagent.web.services import task_admission_pacing as pacing
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_coordinator_runtime as runtime
from xagent.web.services import task_coordinator_service as ownership
from xagent.web.services import task_execution_admission as admission
from xagent.web.services.task_admission_observation import read_admission_snapshot
from xagent.web.services.task_execution_controller import task_control_snapshot

engine = engine_fixture
host = host_fixture


class RunExecution(Execution):
    """Execution whose settled status and run continuity are chosen per test."""

    def __init__(self, host, *, settled=TaskStatus.COMPLETED, new_run=True):
        super().__init__(host)
        self.settled = settled
        self.new_run = new_run

    async def __call__(self, command):
        owner = runtime.current_task_coordinator(command.task_id)
        with self.host.sessions() as db, db.begin():
            ownership.begin_task_execution_no_commit(
                db,
                owner.lease,
                expected=task_control_snapshot(db.get(Task, command.task_id)),
                new_run=self.new_run,
            )
        self.started.append(command.id)

        async def run():
            try:
                await self.finish.wait()
            finally:
                with self.host.sessions() as db, db.begin():
                    task = db.get(Task, command.task_id)
                    task.status = self.settled
                    task.control_state = self.settled.value
                self.terminal.set()
            await self.cleanup.wait()

        owner.track_execution(asyncio.create_task(run()))
        return {}


def task_of(host, command):
    with host.sessions() as db:
        return db.get(TaskExecutionCommand, command.command_id).task_id


def stamped_tickets(host, task_id):
    with host.sessions() as db:
        return [
            ticket.command_id
            for ticket in db.query(TaskAdmissionTicket)
            .filter(
                TaskAdmissionTicket.task_id == task_id,
                TaskAdmissionTicket.owner_attempt_id.isnot(None),
            )
            .order_by(TaskAdmissionTicket.command_id)
        ]


def snapshot(host, bucket):
    with host.sessions() as db:
        return read_admission_snapshot(db, [bucket])[0]


@pytest.mark.parametrize("capacity", [1, 2])
async def test_resume_during_prior_cleanup_holds_one_slot(host, capacity):
    admission.set_task_admission_hook(
        lambda db, command: admission.AdmissionPolicy("turns", capacity, 20)
    )
    first = enqueue(host)
    task_id = task_of(host, first)
    paused = RunExecution(host, settled=TaskStatus.PAUSED)
    assert await transport.dispatch_one_task_command(paused)
    paused.finish.set()
    await asyncio.wait_for(paused.terminal.wait(), 5)
    # PAUSED is published while the previous execution's cleanup is still held.
    resume = enqueue(host, task_id=task_id, kind=transport.TaskCommandKind.RESUME)
    resumed = RunExecution(host, new_run=False)
    assert await transport.dispatch_one_task_command(
        resumed, command_db_id=resume.command_id
    )
    assert resumed.started == [resume.command_id]
    assert stamped_tickets(host, task_id) == [first.command_id, resume.command_id]
    assert snapshot(host, "turns").active == 1

    other = enqueue(host)
    other_execution = RunExecution(host)
    others = [other]
    assert await transport.dispatch_one_task_command(
        other_execution, command_db_id=other.command_id
    ) == (capacity == 2)
    if capacity == 2:
        third = enqueue(host)
        others.append(third)
        assert not await transport.dispatch_one_task_command(
            other_execution, command_db_id=third.command_id
        )
        assert snapshot(host, "turns").active == 2
    for waiting in others[capacity - 1 :]:
        assert transport.load_task_command(waiting.command_id).attempt_count == 0

    paused.cleanup.set()
    resumed.finish.set()
    resumed.cleanup.set()
    await dispatch_next(other_execution)
    assert other_execution.started == [command.command_id for command in others]


async def test_cross_lane_guidance_joins_running_execution_without_a_slot(host):
    first = enqueue(host)
    original = RunExecution(host)
    assert await transport.dispatch_one_task_command(original)
    task_id = task_of(host, first)
    admission.set_task_admission_hook(
        lambda db, command: admission.AdmissionPolicy("other-lane", 1, 20)
    )
    occupying = enqueue(host)
    occupied = RunExecution(host)
    assert await transport.dispatch_one_task_command(occupied)
    message = enqueue(host, task_id=task_id, kind=transport.TaskCommandKind.MESSAGE)
    delivered = []

    async def inject(command):
        delivered.append(command.id)
        return {}

    # Guidance on the running execution needs no slot from its own saturated lane.
    assert await transport.dispatch_one_task_command(
        inject, command_db_id=message.command_id
    )
    assert delivered == [message.command_id]
    with host.sessions() as db:
        assert db.get(TaskAdmissionTicket, message.command_id) is None
    assert snapshot(host, "other-lane").active == 1
    assert snapshot(host, "tenant:batch").active == 1

    waiting = enqueue(host)
    assert not await transport.dispatch_one_task_command(
        occupied, command_db_id=waiting.command_id
    )
    occupied.finish.set()
    occupied.cleanup.set()
    await dispatch_next(occupied)
    assert occupied.started == [occupying.command_id, waiting.command_id]
    original.finish.set()
    original.cleanup.set()


async def test_guidance_that_becomes_a_new_turn_acquires_its_own_lane(
    host, monkeypatch
):
    from unittest.mock import AsyncMock

    from xagent.web.services import task_command_execution
    from xagent.web.services.task_command_execution import (
        execute_durable_task_command,
    )

    now = [100.0]
    monkeypatch.setattr(pacing, "database_time", lambda: literal(now[0]))
    monkeypatch.setattr(observation, "database_time", lambda: literal(now[0]))
    monkeypatch.setattr(task_command_execution, "publish_task_event", AsyncMock())
    monkeypatch.setattr(
        task_command_execution, "get_session_local", lambda: host.sessions
    )
    first = enqueue(host)
    original = RunExecution(host)
    assert await transport.dispatch_one_task_command(original)
    task_id = task_of(host, first)
    admission.set_task_admission_hook(
        lambda db, command: admission.AdmissionPolicy(
            "other-lane",
            1,
            20,
            pacing=pacing.StartupPacing(interval_seconds=60, burst=1),
        )
    )
    occupying = enqueue(host)
    occupied = RunExecution(host)
    assert await transport.dispatch_one_task_command(occupied)
    with host.sessions() as db:
        message = transport.enqueue_task_command(
            db,
            task_id=task_id,
            actor_user_id=host.user,
            command_id="next-message",
            kind=transport.TaskCommandKind.MESSAGE,
            payload={"message": "Continue", "client_message_id": "next-message"},
        )
    attempts = []

    async def settle_then_route(command):
        # The original run ends between the claim and the production message
        # handler; its new-turn path must then refuse without a held ticket.
        attempts.append(command.id)
        original.finish.set()
        await asyncio.wait_for(original.terminal.wait(), 5)
        original.cleanup.set()
        return await execute_durable_task_command(command)

    async def inject(command):
        attempts.append(command.id)
        return {}

    assert await transport.dispatch_one_task_command(
        settle_then_route, command_db_id=message.command_id
    )
    row = transport.load_task_command(message.command_id)
    assert (row.status, row.attempt_count, row.failure_count, row.defer_count) == (
        "pending",
        1,
        0,
        0,
    )
    assert attempts == [message.command_id]
    with host.sessions() as db, db.begin():
        db.get(TaskExecutionCommand, message.command_id).retry_available_at = (
            datetime.now(timezone.utc) - timedelta(seconds=1)
        )

    # The new turn now waits for its persisted lane's capacity ...
    assert not await transport.dispatch_one_task_command(
        inject, command_db_id=message.command_id
    )
    assert attempts == [message.command_id]
    assert snapshot(host, "other-lane").delay_reason == "capacity"
    occupied.finish.set()
    occupied.cleanup.set()
    await asyncio.wait_for(occupied.terminal.wait(), 5)
    # ... and, once capacity opens, for that lane's startup allowance.
    async with asyncio.timeout(5):
        while snapshot(host, "other-lane").active:
            await asyncio.sleep(0.01)
    assert not await transport.dispatch_one_task_command(
        inject, command_db_id=message.command_id
    )
    assert snapshot(host, "other-lane").delay_reason == "startup_pacing"
    now[0] = 160.0
    await dispatch_next(inject)
    assert attempts == [message.command_id, message.command_id]
    assert transport.load_task_command(message.command_id).status == "completed"
    assert occupied.started == [occupying.command_id]


@pytest.mark.parametrize("control_state", ["pause_requested", "resume_requested"])
async def test_cross_lane_message_joins_only_a_running_control_state(
    host, control_state
):
    first = enqueue(host)
    original = RunExecution(host)
    assert await transport.dispatch_one_task_command(original)
    task_id = task_of(host, first)
    with host.sessions() as db, db.begin():
        task = db.get(Task, task_id)
        assert task.status == TaskStatus.RUNNING
        task.control_state = control_state
    admission.set_task_admission_hook(
        lambda db, command: admission.AdmissionPolicy("other-lane", 1, 20)
    )
    occupying = enqueue(host)
    occupied = RunExecution(host)
    assert await transport.dispatch_one_task_command(occupied)
    message = enqueue(host, task_id=task_id, kind=transport.TaskCommandKind.MESSAGE)
    delivered = []

    async def inject(command):
        delivered.append(command.id)
        return {}

    # Not live guidance: it waits for its own lane instead of joining.
    assert not await transport.dispatch_one_task_command(
        inject, command_db_id=message.command_id
    )
    assert transport.load_task_command(message.command_id).attempt_count == 0
    assert delivered == []
    assert snapshot(host, "other-lane").pending == 1
    occupied.finish.set()
    occupied.cleanup.set()
    await dispatch_next(inject)
    assert delivered == [message.command_id]
    assert occupied.started == [occupying.command_id]
    original.finish.set()
    original.cleanup.set()
