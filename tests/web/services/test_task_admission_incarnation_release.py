"""A drained incarnation releases its own ticket while the task keeps running.

Tickets carry no execution identity: ownership is the task lease, which every
incarnation of one task shares. A RESUME classified into another bucket
therefore kept the previous bucket's slot for the whole resumed run, because
both release paths run only while the task is not RUNNING
(xorbitsai/xagent#2777). The coordinator now releases exactly the ticket of
the execution handle that finished, keeping the idle release as the backstop.
"""

import asyncio
import logging

from tests.web.services.test_task_admission_execution_slots import (
    RunExecution,
    enqueue,
    stamped_tickets,
    task_of,
)
from tests.web.services.test_task_execution_admission import engine as engine_fixture
from tests.web.services.test_task_execution_admission import host as host_fixture
from xagent.web.models.task import Task, TaskStatus
from xagent.web.services import task_admission_execution
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_coordinator_runtime as runtime
from xagent.web.services import task_coordinator_service as ownership
from xagent.web.services import task_execution_admission as admission
from xagent.web.services.task_admission_observation import read_admission_snapshot
from xagent.web.services.task_execution_controller import task_control_snapshot

engine = engine_fixture
host = host_fixture


def classify_by_kind(db, command):
    """START work belongs to ``batch``; every continuation to ``interactive``."""
    bucket = "batch" if command.kind == "start" else "interactive"
    return admission.AdmissionPolicy(bucket, 1, 20)


def active(host, bucket):
    with host.sessions() as db:
        rows = read_admission_snapshot(db, [bucket])
    return rows[0].active if rows else 0


async def eventually_active(host, bucket, expected):
    async with asyncio.timeout(5):
        while active(host, bucket) != expected:
            await asyncio.sleep(0.01)


async def pause_then_resume_across_buckets(host):
    """Task A pauses in ``batch`` and resumes in ``interactive``.

    Returns the paused and resumed executions; the paused cleanup is still held
    when this returns, so both buckets hold A's slot.
    """
    admission.set_task_admission_hook(classify_by_kind)
    first = enqueue(host)
    task_id = task_of(host, first)
    paused = RunExecution(host, settled=TaskStatus.PAUSED)
    assert await transport.dispatch_one_task_command(paused)
    paused.finish.set()
    await asyncio.wait_for(paused.terminal.wait(), 5)
    resume = enqueue(host, task_id=task_id, kind=transport.TaskCommandKind.RESUME)
    resumed = RunExecution(host, new_run=False)
    assert await transport.dispatch_one_task_command(
        resumed, command_db_id=resume.command_id
    )
    assert resumed.started == [resume.command_id]
    assert stamped_tickets(host, task_id) == [first.command_id, resume.command_id]
    assert (active(host, "batch"), active(host, "interactive")) == (1, 1)
    return paused, resumed


async def test_joined_guidance_continuation_keeps_the_original_ticket_charged(host):
    """A MESSAGE that joins the running execution rides the joined ticket.

    C2 (the message) never stamps its own ticket: it joins H1's running
    execution. The resume handle H2 it creates while H1 is still live must
    inherit H1's ticket, not C2's (unstamped) one, and that ticket must stay
    charged until every handle of the continued execution has finished.
    """
    admission.set_task_admission_hook(classify_by_kind)
    first = enqueue(host)
    task_id = task_of(host, first)
    original = RunExecution(host, settled=TaskStatus.PAUSED)
    assert await transport.dispatch_one_task_command(original)

    message = enqueue(host, task_id=task_id, kind=transport.TaskCommandKind.MESSAGE)
    h2_finish = asyncio.Event()
    h2s: list[asyncio.Task] = []

    async def join(command):
        with host.sessions() as db:
            run_id = db.get(Task, task_id).run_id
        task_admission_execution.allow_injected_guidance(task_id, run_id)
        owner = runtime.current_task_coordinator(command.task_id)

        async def continue_run():
            await asyncio.gather(original.handles[0], return_exceptions=True)
            await original.terminal.wait()
            with host.sessions() as db, db.begin():
                task = db.get(Task, task_id)
                ownership.begin_task_execution_no_commit(
                    db,
                    owner.lease,
                    expected=task_control_snapshot(task),
                    new_run=False,
                )
            try:
                await h2_finish.wait()
            finally:
                with host.sessions() as db, db.begin():
                    task = db.get(Task, task_id)
                    task.status = TaskStatus.COMPLETED
                    task.control_state = TaskStatus.COMPLETED.value

        h2 = asyncio.create_task(continue_run())
        h2s.append(h2)
        owner.track_execution(h2)
        return {}

    assert await transport.dispatch_one_task_command(
        join, command_db_id=message.command_id
    )
    # The joined MESSAGE never stamped its own ticket.
    assert stamped_tickets(host, task_id) == [first.command_id]

    original.finish.set()
    await asyncio.wait_for(original.terminal.wait(), 5)
    original.cleanup.set()
    await asyncio.wait_for(original.handles[0], 5)

    # Any release H1's drain scheduled has landed once these are awaited.
    await asyncio.gather(*original.owner._admission_releases, return_exceptions=True)
    assert active(host, "batch") == 1
    assert stamped_tickets(host, task_id) == [first.command_id]
    other = enqueue(host)
    other_execution = RunExecution(host)
    assert not await transport.dispatch_one_task_command(
        other_execution, command_db_id=other.command_id
    )

    h2_finish.set()
    await asyncio.wait_for(h2s[0], 5)
    await eventually_active(host, "batch", 0)
    assert await transport.dispatch_one_task_command(
        other_execution, command_db_id=other.command_id
    )
    other_execution.finish.set()
    other_execution.cleanup.set()


async def test_joined_guidance_registered_after_the_original_drains_keeps_the_ticket(
    host,
):
    """A continuation registered only after the original handle already drained.

    ``allow_injected_guidance`` must pin the continuation's ticket at join
    confirmation time. If instead H2's ticket were derived from whatever is
    still live in ``_child_admissions`` when it registers, draining H1 first
    would leave nothing to inherit and the bucket would read 0 while H2 (the
    continued run) is still executing.
    """
    admission.set_task_admission_hook(classify_by_kind)
    first = enqueue(host)
    task_id = task_of(host, first)
    original = RunExecution(host, settled=TaskStatus.PAUSED)
    assert await transport.dispatch_one_task_command(original)

    message = enqueue(host, task_id=task_id, kind=transport.TaskCommandKind.MESSAGE)
    h2_finish = asyncio.Event()
    h2s: list[asyncio.Task] = []

    async def join(command):
        with host.sessions() as db:
            run_id = db.get(Task, task_id).run_id
        task_admission_execution.allow_injected_guidance(task_id, run_id)
        owner = runtime.current_task_coordinator(command.task_id)

        # Drain the original handle before the continuation ever registers.
        original.finish.set()
        await original.terminal.wait()
        original.cleanup.set()
        await original.handles[0]
        await asyncio.gather(
            *original.owner._admission_releases, return_exceptions=True
        )

        async def continue_run():
            with host.sessions() as db, db.begin():
                task = db.get(Task, task_id)
                ownership.begin_task_execution_no_commit(
                    db,
                    owner.lease,
                    expected=task_control_snapshot(task),
                    new_run=False,
                )
            try:
                await h2_finish.wait()
            finally:
                with host.sessions() as db, db.begin():
                    task = db.get(Task, task_id)
                    task.status = TaskStatus.COMPLETED
                    task.control_state = TaskStatus.COMPLETED.value

        h2 = asyncio.create_task(continue_run())
        h2s.append(h2)
        owner.track_execution(h2)
        return {}

    assert await transport.dispatch_one_task_command(
        join, command_db_id=message.command_id
    )

    assert active(host, "batch") == 1
    assert stamped_tickets(host, task_id) == [first.command_id]
    other = enqueue(host)
    other_execution = RunExecution(host)
    assert not await transport.dispatch_one_task_command(
        other_execution, command_db_id=other.command_id
    )

    h2_finish.set()
    await asyncio.wait_for(h2s[0], 5)
    await eventually_active(host, "batch", 0)
    assert await transport.dispatch_one_task_command(
        other_execution, command_db_id=other.command_id
    )
    other_execution.finish.set()
    other_execution.cleanup.set()


async def test_previous_bucket_slot_is_released_once_its_incarnation_drains(host):
    paused, resumed = await pause_then_resume_across_buckets(host)
    other = enqueue(host)
    other_execution = RunExecution(host)
    # Both slots stay held while the previous incarnation is still draining.
    assert not await transport.dispatch_one_task_command(
        other_execution, command_db_id=other.command_id
    )

    paused.cleanup.set()
    await eventually_active(host, "batch", 0)
    assert active(host, "interactive") == 1
    assert not resumed.terminal.is_set()
    # The resumed run keeps only the slot of the lane it now runs in.
    assert await transport.dispatch_one_task_command(
        other_execution, command_db_id=other.command_id
    )
    assert other_execution.started == [other.command_id]

    resumed.finish.set()
    resumed.cleanup.set()
    other_execution.finish.set()
    other_execution.cleanup.set()
    await eventually_active(host, "interactive", 0)
    await eventually_active(host, "batch", 0)


async def test_cancelled_successor_never_releases_the_predecessor_slot(host):
    paused, resumed = await pause_then_resume_across_buckets(host)
    handle = resumed.handles[0]
    handle.cancel()
    await asyncio.gather(handle, return_exceptions=True)
    await eventually_active(host, "interactive", 0)
    # The previous incarnation still holds its own slot until it drains.
    assert active(host, "batch") == 1
    other = enqueue(host)
    other_execution = RunExecution(host)
    assert not await transport.dispatch_one_task_command(
        other_execution, command_db_id=other.command_id
    )

    paused.cleanup.set()
    await eventually_active(host, "batch", 0)
    assert await transport.dispatch_one_task_command(
        other_execution, command_db_id=other.command_id
    )
    other_execution.finish.set()
    other_execution.cleanup.set()


async def test_same_bucket_resume_keeps_one_slot_after_the_predecessor_drains(host):
    admission.set_task_admission_hook(
        lambda db, command: admission.AdmissionPolicy("turns", 1, 20)
    )
    first = enqueue(host)
    task_id = task_of(host, first)
    paused = RunExecution(host, settled=TaskStatus.PAUSED)
    assert await transport.dispatch_one_task_command(paused)
    paused.finish.set()
    await asyncio.wait_for(paused.terminal.wait(), 5)
    resume = enqueue(host, task_id=task_id, kind=transport.TaskCommandKind.RESUME)
    resumed = RunExecution(host, new_run=False)
    assert await transport.dispatch_one_task_command(
        resumed, command_db_id=resume.command_id
    )
    assert active(host, "turns") == 1

    paused.cleanup.set()
    async with asyncio.timeout(5):
        while stamped_tickets(host, task_id) != [resume.command_id]:
            await asyncio.sleep(0.01)
    # Releasing the drained START ticket leaves the RESUME ticket holding the slot.
    assert active(host, "turns") == 1
    other = enqueue(host)
    other_execution = RunExecution(host)
    assert not await transport.dispatch_one_task_command(
        other_execution, command_db_id=other.command_id
    )

    resumed.finish.set()
    resumed.cleanup.set()
    await eventually_active(host, "turns", 0)
    assert await transport.dispatch_one_task_command(
        other_execution, command_db_id=other.command_id
    )
    other_execution.finish.set()
    other_execution.cleanup.set()


async def test_drained_handle_of_a_superseded_owner_releases_nothing(host):
    """A handle draining after another owner took the task frees no slot."""
    admission.set_task_admission_hook(classify_by_kind)
    first = enqueue(host)
    task_id = task_of(host, first)
    paused = RunExecution(host, settled=TaskStatus.PAUSED)
    assert await transport.dispatch_one_task_command(paused)
    paused.finish.set()
    await asyncio.wait_for(paused.terminal.wait(), 5)
    # Another worker takes the task over and admits its own RESUME while the
    # previous owner's cleanup is still held.
    successor = ownership.TaskLease(
        task_id=task_id, runner_id="worker-2", attempt_id="successor-attempt"
    )
    resume = enqueue(host, task_id=task_id, kind=transport.TaskCommandKind.RESUME)
    with host.sessions() as db, db.begin():
        task = db.get(Task, task_id)
        task.runner_id = successor.runner_id
        task.lease_attempt_id = successor.attempt_id
        db.flush()
        assert admission.reserve_task_admission(db, resume.command_id, successor)
    assert active(host, "interactive") == 1
    assert stamped_tickets(host, task_id) == [first.command_id, resume.command_id]

    paused.cleanup.set()
    await asyncio.gather(paused.handles[0], return_exceptions=True)
    # `_close` drains pending admission releases before finishing, so waiting
    # for the superseded owner to fully close makes the assertion deterministic.
    async with asyncio.timeout(5):
        while paused.owner.state is not runtime.CoordinatorState.CLOSED:
            await asyncio.sleep(0.01)
    assert stamped_tickets(host, task_id) == [first.command_id, resume.command_id]
    assert active(host, "interactive") == 1


class _EndsWithoutSettling:
    """Begins execution, then tracks a handle that returns without settling.

    The task row stays RUNNING; only the ticket's release, not a status flip,
    frees its slot.
    """

    def __init__(self, host):
        self.host = host
        self.owner = None

    async def __call__(self, command):
        owner = runtime.current_task_coordinator(command.task_id)
        self.owner = owner
        with self.host.sessions() as db, db.begin():
            ownership.begin_task_execution_no_commit(
                db,
                owner.lease,
                expected=task_control_snapshot(db.get(Task, command.task_id)),
                new_run=True,
            )

        async def run() -> None:
            return

        owner.track_execution(asyncio.create_task(run()))
        return {}


async def test_handle_that_ends_without_settling_frees_its_slot(host):
    admission.set_task_admission_hook(classify_by_kind)
    first = enqueue(host)
    task_id = task_of(host, first)
    execute = _EndsWithoutSettling(host)
    assert await transport.dispatch_one_task_command(execute)

    await eventually_active(host, "batch", 0)
    assert stamped_tickets(host, task_id) == []
    # The row never settled, so the owner ends in recovery, not a clean release.
    async with asyncio.timeout(5):
        while execute.owner.state is not runtime.CoordinatorState.CLOSED:
            await asyncio.sleep(0.01)
    with host.sessions() as db:
        task = db.get(Task, task_id)
        assert task.status == TaskStatus.RUNNING
        assert task.runner_id is not None
        assert task.lease_attempt_id is not None


async def test_failed_release_is_retried_on_the_next_drained_handle(
    host, monkeypatch, caplog
):
    paused, resumed = await pause_then_resume_across_buckets(host)
    real_release = admission.release_command_admission

    def raise_release(db, lease, command_id):
        raise RuntimeError("simulated release failure")

    monkeypatch.setattr(admission, "release_command_admission", raise_release)
    caplog.set_level(
        logging.ERROR, logger="xagent.web.services.task_coordinator_runtime"
    )

    paused.cleanup.set()
    await asyncio.gather(paused.handles[0], return_exceptions=True)
    async with asyncio.timeout(5):
        while not any(
            "could not release admission" in record.getMessage()
            for record in caplog.records
        ):
            await asyncio.sleep(0.01)
    assert active(host, "batch") == 1

    # Restore the real release, then drain an unrelated ungoverned handle: its
    # _child_done must retry the stashed failed release, not merely leave it
    # for the idle backstop.
    monkeypatch.setattr(admission, "release_command_admission", real_release)
    with task_admission_execution.admission_execution(
        -1, paused.owner.task_id, governed=False
    ):
        extra = asyncio.create_task(asyncio.sleep(0))
        paused.owner.track_execution(extra)
    await asyncio.wait_for(extra, 5)

    # The resumed run is still live, so the idle backstop cannot be what
    # freed the stashed batch ticket.
    assert not resumed.terminal.is_set()
    await eventually_active(host, "batch", 0)
    assert not resumed.terminal.is_set()

    resumed.finish.set()
    resumed.cleanup.set()
    await eventually_active(host, "interactive", 0)


async def test_close_waits_for_a_pending_release(host, monkeypatch):
    """``close()`` completes only after a pending drained-handle release lands.

    The release is parked before it opens any transaction, so nothing else in
    ``close()`` can block on it: if ``_close`` did not drain pending releases,
    the close would finish first.
    """
    paused, resumed = await pause_then_resume_across_buckets(host)
    owner = paused.owner
    real_release = owner._release_drained_admission
    gate = asyncio.Event()
    order: list[str] = []

    async def parked_release(command_ids):
        await gate.wait()
        await real_release(command_ids)
        order.append("release")

    monkeypatch.setattr(owner, "_release_drained_admission", parked_release)

    paused.cleanup.set()
    await asyncio.gather(paused.handles[0], return_exceptions=True)
    assert owner._admission_releases

    closing = asyncio.create_task(owner.close())
    closing.add_done_callback(lambda _: order.append("close"))
    # close() cancels its remaining child before it drains pending releases.
    async with asyncio.timeout(5):
        while not resumed.handles[0].done():
            await asyncio.sleep(0.01)
    # Give a close that skipped the drain every chance to finish first.
    await asyncio.sleep(0.2)
    assert not closing.done()

    gate.set()
    await asyncio.wait_for(closing, 5)
    assert order == ["release", "close"]
    assert active(host, "batch") == 0

    resumed.finish.set()
    resumed.cleanup.set()


async def test_handle_tracked_outside_an_admission_context_is_not_mapped(host):
    """An ungoverned handle neither holds nor releases any ticket."""
    admission.set_task_admission_hook(classify_by_kind)
    first = enqueue(host)
    task_id = task_of(host, first)
    execute = RunExecution(host, settled=TaskStatus.PAUSED)
    extra_finish = asyncio.Event()
    extras: list[asyncio.Task] = []

    async def start(command):
        result = await execute(command)
        owner = runtime.current_task_coordinator(command.task_id)
        with task_admission_execution.admission_execution(
            command.id, command.task_id, governed=False
        ):
            extras.append(asyncio.create_task(extra_finish.wait()))
            owner.track_execution(extras[0])
        return result

    assert await transport.dispatch_one_task_command(start)
    assert stamped_tickets(host, task_id) == [first.command_id]
    assert active(host, "batch") == 1

    # The governed handle's drain frees its slot although the ungoverned
    # handle is still live: that handle holds nothing.
    execute.finish.set()
    execute.cleanup.set()
    await asyncio.wait_for(execute.handles[0], 5)
    await eventually_active(host, "batch", 0)
    assert not extras[0].done()
    extra_finish.set()
    await asyncio.wait_for(extras[0], 5)
    assert stamped_tickets(host, task_id) == []


def test_stale_owner_cannot_release_the_successor_ticket(host):
    """The single-ticket release is fenced by the ticket's current owner."""
    admission.set_task_admission_hook(classify_by_kind)
    first = enqueue(host)
    task_id = task_of(host, first)
    with host.sessions() as db, db.begin():
        current = ownership.acquire_task_lease_no_commit(db, task_id, runner_id="w1")
        assert current is not None
        assert admission.reserve_task_admission(db, first.command_id, current)
    stale = ownership.TaskLease(
        task_id=task_id, runner_id="w0", attempt_id="stale-attempt"
    )
    with host.sessions() as db, db.begin():
        admission.release_command_admission(db, stale, first.command_id)
    assert stamped_tickets(host, task_id) == [first.command_id]
    assert active(host, "batch") == 1
    with host.sessions() as db, db.begin():
        admission.release_command_admission(db, current, first.command_id)
    assert stamped_tickets(host, task_id) == []
    assert active(host, "batch") == 0
