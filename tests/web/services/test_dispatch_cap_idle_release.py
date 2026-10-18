"""A process at its dispatch cap gives up ownership of tasks it will not run."""

# Pytest fixture imports are intentionally shadowed by test parameters.
# ruff: noqa: F811

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from tests.shared.async_waits import DB_PROGRESS_TIMEOUT, eventually
from tests.web.services.test_dispatch_cap import (  # noqa: F401
    CapHost,
    cap_host,
    set_limit,
)
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_admission import TaskAdmissionBucket, TaskAdmissionTicket
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_coordinator_runtime as runtime


def record_idle_outcomes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The outcome of every idle release attempt, in order."""

    outcomes: list[str] = []
    real = runtime.TaskCoordinator._release_if_idle

    def recording(self: Any, **kwargs: Any) -> str:
        outcome = real(self, **kwargs)
        outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(runtime.TaskCoordinator, "_release_if_idle", recording)
    return outcomes


def add_command(
    host: CapHost,
    task_id: int,
    kind: str,
    *,
    status: str = "pending",
    attempts: int = 0,
    retry_available_at: datetime | None = None,
    dispatch_class: str = "batch",
) -> int:
    host._identity += 1
    with host.factory() as db:
        row = TaskExecutionCommand(
            task_id=task_id,
            command_id=f"cmd-{host._identity}",
            kind=kind,
            payload={},
            status=status,
            attempt_count=attempts,
            retry_available_at=retry_available_at,
            dispatch_class=dispatch_class,
        )
        db.add(row)
        db.commit()
        return int(row.id)


def other_runner_candidate(host: CapHost, command_db_id: int | None) -> int | None:
    with host.factory() as db:
        row = transport._claimable_query(
            db, runner_id="other-runner", command_db_id=command_db_id, cap=None
        ).first()
        return int(row.id) if row is not None else None


def lease_runner(host: CapHost, task_id: int) -> str | None:
    with host.factory() as db:
        return db.get(Task, task_id).runner_id


CAP_LOG = "at its dispatch cap"

IDLE_ROWS = [
    # name, limit, task status, rows (kind, status, attempts), active, outcome,
    # held rows the INFO line reports (0: no line), another task holds a row too
    (
        "start",
        1,
        TaskStatus.PENDING,
        [("start", "pending", 0)],
        False,
        "released",
        1,
        False,
    ),
    (
        "resume",
        1,
        TaskStatus.PAUSED,
        [("resume", "pending", 0)],
        False,
        "released",
        1,
        False,
    ),
    (
        "reply",
        1,
        TaskStatus.WAITING_FOR_USER,
        [("resume_input", "pending", 0)],
        False,
        "released",
        1,
        False,
    ),
    (
        "message_completed",
        1,
        TaskStatus.COMPLETED,
        [("message", "pending", 0)],
        False,
        "released",
        1,
        False,
    ),
    (
        "message_failed",
        1,
        TaskStatus.FAILED,
        [("message", "pending", 0)],
        False,
        "released",
        1,
        False,
    ),
    (
        "two_held_beside_other_task",
        1,
        TaskStatus.COMPLETED,
        [("message", "pending", 0), ("message", "pending", 1)],
        False,
        "released",
        2,
        True,
    ),
    ("no_pending", 1, TaskStatus.COMPLETED, [], False, "released", 0, False),
    (
        "cap_open",
        2,
        TaskStatus.PENDING,
        [("start", "pending", 0)],
        False,
        "busy",
        0,
        False,
    ),
    (
        "task_running",
        1,
        TaskStatus.RUNNING,
        [("start", "pending", 0)],
        False,
        "busy",
        0,
        False,
    ),
    (
        "pause_pending",
        1,
        TaskStatus.PENDING,
        [("start", "pending", 0), ("pause", "pending", 0)],
        False,
        "busy",
        0,
        False,
    ),
    (
        "cancel_pending",
        1,
        TaskStatus.PENDING,
        [("start", "pending", 0), ("cancel", "pending", 0)],
        False,
        "busy",
        0,
        False,
    ),
    (
        "processing",
        1,
        TaskStatus.PENDING,
        [("start", "processing", 1)],
        False,
        "busy",
        0,
        False,
    ),
    (
        "task_active",
        1,
        TaskStatus.PENDING,
        [("start", "pending", 0)],
        True,
        "busy",
        0,
        False,
    ),
]


@pytest.mark.parametrize(
    "limit,status,rows,active,expected,logged,other_held",
    [row[1:] for row in IDLE_ROWS],
    ids=[row[0] for row in IDLE_ROWS],
)
async def test_idle_check_releases_lease_when_cap_closed(
    cap_host: CapHost,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    limit: int,
    status: TaskStatus,
    rows: list[tuple[str, str, int]],
    active: bool,
    expected: str,
    logged: int,
    other_held: bool,
) -> None:
    host = cap_host
    set_limit(monkeypatch, limit)
    host.live(host.new_task())
    target = host.new_task(status=status)
    ids = [add_command(host, target, k, status=s, attempts=a) for k, s, a in rows]
    if other_held:
        # Held rows of another task must not be counted for this one.
        add_command(host, host.new_task(), "start")
    if active:
        # Another dispatch attempt holds a slot for this task.
        host.manager._dispatch_reservations[target] += 1
    outcomes = record_idle_outcomes(monkeypatch)
    caplog.set_level(logging.INFO, logger=runtime.logger.name)

    coordinator = await host.registry.ensure(target)
    assert coordinator is not None
    coordinator._ensure_idle_check()
    await eventually(lambda: outcomes)

    assert outcomes[0] == expected
    cap_lines = [r.getMessage() for r in caplog.records if CAP_LOG in r.getMessage()]
    assert cap_lines == (
        [
            f"Task {target} lease released by runner worker at its dispatch cap or batch line "
            f"(limit=1, active=1); {logged} queued execution command(s) left for "
            "a process with a free slot"
        ]
        if logged
        else []
    )
    if expected == "released":
        assert lease_runner(host, target) is None
        if ids:
            assert other_runner_candidate(host, ids[0]) == ids[0]
    else:
        assert lease_runner(host, target) == "worker"
        assert other_runner_candidate(host, ids[0]) is None


async def test_idle_check_releases_handed_back_start(
    cap_host: CapHost,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A start handed back for a retry is held as well; it is claimable once due.

    Both hand-back paths (``defer_task_command`` and
    ``return_to_admission_queue``) leave the row pending with a future
    ``retry_available_at``, which the claim query honours.
    """

    host = cap_host
    set_limit(monkeypatch, 1)
    host.live(host.new_task())
    target = host.new_task()
    start = add_command(
        host,
        target,
        "start",
        attempts=1,
        retry_available_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    outcomes = record_idle_outcomes(monkeypatch)
    caplog.set_level(logging.INFO, logger=runtime.logger.name)

    coordinator = await host.registry.ensure(target)
    assert coordinator is not None
    coordinator._ensure_idle_check()
    await eventually(lambda: outcomes)

    assert outcomes[0] == "released"
    assert lease_runner(host, target) is None
    assert [r.getMessage() for r in caplog.records if CAP_LOG in r.getMessage()] == [
        f"Task {target} lease released by runner worker at its dispatch cap or batch line "
        "(limit=1, active=1); 1 queued execution command(s) left for a process "
        "with a free slot"
    ]
    # No process claims it inside its retry window.
    assert other_runner_candidate(host, start) is None
    with host.factory() as db:
        db.get(TaskExecutionCommand, start).retry_available_at = datetime.now(
            timezone.utc
        ) - timedelta(seconds=1)
        db.commit()
    # Once due, any process with a free slot can claim it.
    assert other_runner_candidate(host, start) == start


async def test_idle_check_releases_lease_once_cap_closes(
    cap_host: CapHost,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An owner that answered busy with a free slot re-reads the cap each pass."""

    host = cap_host
    set_limit(monkeypatch, 2)
    host.live(host.new_task())
    target = host.new_task()
    start = add_command(host, target, "start")
    outcomes = record_idle_outcomes(monkeypatch)
    caplog.set_level(logging.INFO, logger=runtime.logger.name)

    coordinator = await host.registry.ensure(target)
    assert coordinator is not None
    coordinator._ensure_idle_check()
    await eventually(lambda: outcomes)
    assert outcomes[0] == "busy"
    assert lease_runner(host, target) == "worker"

    # Another execution fills the last slot; this process will not run the start.
    host.live(host.new_task())
    coordinator.wake()
    await eventually(lambda: outcomes[-1] == "released")

    assert lease_runner(host, target) is None
    assert other_runner_candidate(host, start) == start
    assert [r.getMessage() for r in caplog.records if CAP_LOG in r.getMessage()] == [
        f"Task {target} lease released by runner worker at its dispatch cap or batch line "
        "(limit=2, active=2); 1 queued execution command(s) left for a process "
        "with a free slot"
    ]


async def test_full_process_hands_held_start_to_other_runners(
    cap_host: CapHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full process applied a pause; a start of that task arrived meanwhile.

    The start is its admission bucket's head, so while this process kept the
    task's lease no process could claim anything from the bucket.
    """

    host = cap_host
    set_limit(monkeypatch, 1)
    host.live(host.new_task())
    target = host.new_task()
    neighbour = host.new_task()
    host.enqueue(target, "pause")
    started: dict[int, int] = {}

    async def handler(command: Any) -> dict[str, Any]:
        # Committed while the full process still owns the task.
        started[target] = host.enqueue(target, "start")
        started[neighbour] = host.enqueue(neighbour, "start")
        with host.factory() as db:
            db.add(TaskAdmissionBucket(key="b", capacity=10, max_pending=10))
            for task_id, command_db_id in started.items():
                db.add(
                    TaskAdmissionTicket(
                        command_id=command_db_id, task_id=task_id, bucket_key="b"
                    )
                )
            db.commit()
        return await host.handler(command)

    outcomes = record_idle_outcomes(monkeypatch)
    assert await asyncio.wait_for(
        transport.dispatch_one_task_command(handler), DB_PROGRESS_TIMEOUT
    )
    await eventually(lambda: outcomes)

    assert outcomes[0] == "released"
    assert lease_runner(host, target) is None
    # Before the release no other runner could claim anything from the
    # bucket; now its head is claimable by any process with a free slot.
    assert other_runner_candidate(host, None) == started[target]
    # Not evidence of the fix (it also holds before it): the neighbour keeps
    # its normal place behind the bucket head.
    assert other_runner_candidate(host, started[neighbour]) is None


async def test_batch_line_releases_an_idle_task_that_this_process_can_still_start(
    cap_host: CapHost,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """On the batch line the idle check releases a task as for a batch command.

    The task's queued command is interactive and this process could still start
    it, so releasing is coarser than necessary; the released command stays
    claimable here and anywhere else with room.
    """

    host = cap_host
    set_limit(monkeypatch, 3)
    monkeypatch.setenv("XAGENT_DISPATCHER_INTERACTIVE_RESERVE_PER_PROCESS", "1")
    for _ in range(2):
        host.live(host.new_task())
    target = host.new_task()
    start = add_command(host, target, "start", dispatch_class="interactive")
    outcomes = record_idle_outcomes(monkeypatch)
    caplog.set_level(logging.INFO, logger=runtime.logger.name)

    coordinator = await host.registry.ensure(target)
    assert coordinator is not None
    coordinator._ensure_idle_check()
    await eventually(lambda: outcomes)

    assert outcomes[0] == "released"
    assert lease_runner(host, target) is None
    assert [r.getMessage() for r in caplog.records if CAP_LOG in r.getMessage()] == [
        f"Task {target} lease released by runner worker at its dispatch cap or "
        "batch line (limit=3, active=2); 1 queued execution command(s) left for "
        "a process with a free slot"
    ]
    assert other_runner_candidate(host, start) == start
    assert await asyncio.wait_for(
        transport.dispatch_one_task_command(host.handler), DB_PROGRESS_TIMEOUT
    )
    assert host.claimed == [(target, transport.TaskCommandKind.START)]
