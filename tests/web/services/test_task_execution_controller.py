from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.task_execution_controller import (
    StaleTaskRunError,
    TaskControlState,
    TaskExecutionController,
    apply_task_control_transition,
    control_state_for_status,
    task_control_snapshot,
    transition_task_control_state_sync,
)


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'task-controller.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


def _create_task(db) -> Task:
    user = User(username="controller-user", password_hash="hash", is_admin=False)
    db.add(user)
    db.commit()
    task = Task(
        user_id=user.id,
        title="Controller test",
        description="Controller test",
        status=TaskStatus.PENDING,
        execution_mode="auto",
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def test_snapshot_requires_a_persisted_task_id() -> None:
    task = Task(
        user_id=1,
        title="Transient task",
        description="Transient task",
        status=TaskStatus.PENDING,
        execution_mode="auto",
    )

    with pytest.raises(ValueError, match="task with no ID"):
        task_control_snapshot(task)


def test_unknown_task_status_has_a_clear_error() -> None:
    with pytest.raises(ValueError, match="Unsupported task status"):
        control_state_for_status("cancelled")  # type: ignore[arg-type]


def test_transition_does_not_flush_unrelated_pending_objects(db_session) -> None:
    task = _create_task(db_session)
    unrelated = Task(
        user_id=task.user_id,
        title="Unrelated task",
        description="Unrelated task",
        status=TaskStatus.PENDING,
        execution_mode="auto",
    )
    db_session.add(unrelated)
    db_session.commit()
    unrelated.title = "Still pending"

    snapshot = apply_task_control_transition(
        task,
        TaskControlState.RUNNING,
        status=TaskStatus.RUNNING,
        new_run=True,
    )

    assert snapshot.control_state == TaskControlState.RUNNING
    assert unrelated in db_session.dirty


@pytest.mark.asyncio
async def test_same_task_commands_are_fifo_serialized() -> None:
    controller = TaskExecutionController()
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    order: list[str] = []

    async def first() -> None:
        async with controller.command(7):
            order.append("first-enter")
            first_entered.set()
            await release_first.wait()
            order.append("first-exit")

    async def queued(name: str) -> None:
        await first_entered.wait()
        async with controller.command(7):
            order.append(name)

    tasks = [
        asyncio.create_task(first()),
        asyncio.create_task(queued("second")),
        asyncio.create_task(queued("third")),
    ]
    await first_entered.wait()
    await asyncio.sleep(0)
    release_first.set()
    await asyncio.gather(*tasks)

    assert order == ["first-enter", "first-exit", "second", "third"]
    assert controller._gates == {}


@pytest.mark.asyncio
async def test_gate_is_reentrant_for_nested_transport_and_turn_claim() -> None:
    controller = TaskExecutionController()

    async with controller.command(11):
        async with controller.command(11):
            assert controller._gates[11].depth == 2

    assert controller._gates == {}


@pytest.mark.asyncio
async def test_child_task_reentry_fails_instead_of_deadlocking() -> None:
    controller = TaskExecutionController()

    async def reenter_from_child() -> None:
        async with controller.command(11):
            raise AssertionError("child task must not acquire its parent's gate")

    async with controller.command(11):
        child = asyncio.create_task(reenter_from_child())
        with pytest.raises(RuntimeError, match="reentered from a child asyncio task"):
            await asyncio.wait_for(child, timeout=1)

    assert controller._gates == {}


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_split_or_leak_the_gate() -> None:
    controller = TaskExecutionController()
    owner_entered = asyncio.Event()
    release_owner = asyncio.Event()
    third_entered = asyncio.Event()

    async def owner() -> None:
        async with controller.command(19):
            owner_entered.set()
            await release_owner.wait()

    async def waiter() -> None:
        async with controller.command(19):
            raise AssertionError("cancelled waiter must not enter")

    async def third() -> None:
        async with controller.command(19):
            third_entered.set()

    owner_task = asyncio.create_task(owner())
    await owner_entered.wait()
    waiter_task = asyncio.create_task(waiter())
    await asyncio.sleep(0)
    waiter_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter_task

    third_task = asyncio.create_task(third())
    await asyncio.sleep(0)
    assert not third_entered.is_set()
    release_owner.set()
    await asyncio.gather(owner_task, third_task)
    assert controller._gates == {}


@pytest.mark.asyncio
async def test_different_tasks_can_progress_concurrently() -> None:
    controller = TaskExecutionController()
    both_entered = asyncio.Event()
    entered: set[int] = set()

    async def run(task_id: int) -> None:
        async with controller.command(task_id):
            entered.add(task_id)
            if len(entered) == 2:
                both_entered.set()
            await asyncio.wait_for(both_entered.wait(), timeout=1)

    await asyncio.gather(run(1), run(2))
    assert entered == {1, 2}


def test_transitions_keep_run_identity_and_advance_version(db_session) -> None:
    task = _create_task(db_session)

    running = transition_task_control_state_sync(
        int(task.id),
        TaskControlState.RUNNING,
        status=TaskStatus.RUNNING,
        new_run=True,
    )
    pause_requested = transition_task_control_state_sync(
        int(task.id),
        TaskControlState.PAUSE_REQUESTED,
        expected_run_id=running.run_id,
    )
    paused = transition_task_control_state_sync(
        int(task.id),
        TaskControlState.PAUSED,
        status=TaskStatus.PAUSED,
        expected_run_id=running.run_id,
    )

    assert running.run_id
    assert pause_requested.run_id == running.run_id == paused.run_id
    assert [
        running.state_version,
        pause_requested.state_version,
        paused.state_version,
    ] == [
        1,
        2,
        3,
    ]
    assert pause_requested.status == TaskStatus.RUNNING
    assert pause_requested.control_state == TaskControlState.PAUSE_REQUESTED

    with pytest.raises(StaleTaskRunError):
        transition_task_control_state_sync(
            int(task.id),
            TaskControlState.COMPLETED,
            status=TaskStatus.COMPLETED,
            expected_run_id="superseded-run",
        )


def test_concurrent_transitions_get_distinct_atomic_versions(db_session) -> None:
    task = _create_task(db_session)
    running = transition_task_control_state_sync(
        int(task.id),
        TaskControlState.RUNNING,
        status=TaskStatus.RUNNING,
        new_run=True,
    )

    def transition(state: TaskControlState):
        return transition_task_control_state_sync(
            int(task.id),
            state,
            expected_run_id=running.run_id,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        pause_future = executor.submit(transition, TaskControlState.PAUSE_REQUESTED)
        resume_future = executor.submit(transition, TaskControlState.RESUME_REQUESTED)
        snapshots = [pause_future.result(), resume_future.result()]

    assert {snapshot.state_version for snapshot in snapshots} == {2, 3}
    assert {snapshot.run_id for snapshot in snapshots} == {running.run_id}
