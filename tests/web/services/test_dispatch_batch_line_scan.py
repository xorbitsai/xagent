"""The dispatch scan on the batch line: what it takes, what it leaves, one time."""

# Pytest fixture imports are intentionally shadowed by test parameters.
# ruff: noqa: F811

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy import event

from tests.web.services.test_dispatch_cap import (  # noqa: F401
    CapHost,
    cap_host,
    set_limit,
)
from tests.web.services.test_dispatch_reserve import (  # noqa: F401
    RESERVE_ENV,
    _state,
    batch_line,
    dispatch,
    set_reserve,
)
from xagent.web.models.task import TaskStatus
from xagent.web.models.task_admission import TaskAdmissionBucket, TaskAdmissionTicket
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.services import task_command_transport as transport
from xagent.web.services.task_command_transport import TaskCommandKind
from xagent.web.services.task_dispatch_cap import DispatchPolicy

KIND = TaskCommandKind
LIMIT, RESERVE = 3, 1


def cap_of(host: CapHost, policy: DispatchPolicy | None = None) -> Any:
    return host.manager.dispatch_cap_snapshot(
        policy or DispatchPolicy(limit=LIMIT, interactive_reserve=RESERVE)
    )


def fill(host: CapHost, active: int) -> None:
    """Run ``active`` tasks of their own, so the process sits on a known line."""
    for _ in range(active):
        host.live(host.new_task())


def found(cap: Any, command_db_id: int | None = None) -> Any:
    return transport._find_command_candidate(command_db_id, (), cap)


# Active tasks out of a limit of 3 with a reserve of 1: open, on the batch
# line, at the limit.
TIERS = {"open": 1, "batch-line": 2, "closed": 3}


# One queued command: (label, kind, dispatch class). Each command belongs to a
# task of its own that this process does not run; an answer's task waits for it.
Spec = tuple[str, str, str]


def enqueue_queue(host: CapHost, queue: list[Spec]) -> dict[str, int]:
    ids = {}
    for label, kind, dispatch_class in queue:
        status = (
            TaskStatus.WAITING_FOR_USER
            if kind == "resume_input"
            else TaskStatus.PENDING
        )
        ids[label] = host.enqueue(
            host.new_task(status=status), kind, dispatch_class=dispatch_class
        )
    return ids


# scenario, expected command per tier (None: no candidate). The expectations
# are written out, not derived.
SCENARIOS = [
    pytest.param(
        [("batch", "start", "batch"), ("interactive", "start", "interactive")],
        {"open": "batch", "batch-line": "interactive", "closed": None},
        id="older-batch-newer-interactive-start",
    ),
    pytest.param(
        [("batch", "start", "batch"), ("answer", "resume_input", "interactive")],
        {"open": "answer", "batch-line": "answer", "closed": None},
        id="older-batch-newer-answer",
    ),
    pytest.param(
        [("batch", "start", "batch")],
        {"open": "batch", "batch-line": None, "closed": None},
        id="only-batch",
    ),
    pytest.param(
        [("batch", "start", "batch"), ("pause", "pause", "batch")],
        {"open": "batch", "batch-line": "pause", "closed": "pause"},
        id="older-batch-newer-pause",
    ),
    pytest.param(
        [("interactive", "start", "interactive"), ("cancel", "cancel", "batch")],
        {"open": "interactive", "batch-line": "interactive", "closed": "cancel"},
        id="older-interactive-newer-cancel",
    ),
    pytest.param(
        [("cancel", "cancel", "batch"), ("interactive", "start", "interactive")],
        {"open": "cancel", "batch-line": "cancel", "closed": "cancel"},
        id="older-cancel-newer-interactive",
    ),
]


@pytest.mark.parametrize("tier", list(TIERS))
@pytest.mark.parametrize(("queue", "expected"), SCENARIOS)
async def test_scan_takes_the_oldest_command_the_tier_allows(
    cap_host: CapHost, queue: list[Spec], expected: dict[str, str | None], tier: str
) -> None:
    host = cap_host
    fill(host, TIERS[tier])
    ids = enqueue_queue(host, queue)
    cap = cap_of(host)

    candidate = found(cap)

    want = expected[tier]
    assert (candidate.id if candidate else None) == (ids[want] if want else None)


@pytest.mark.parametrize("tier", ["batch-line", "closed"])
async def test_a_control_passes_a_command_waiting_for_admission(
    cap_host: CapHost, tier: str
) -> None:
    host = cap_host
    fill(host, TIERS[tier])
    task = host.new_task()
    start = host.enqueue(task, "start")
    cancel = host.enqueue(task, "cancel")
    with host.factory() as db:
        db.add(TaskAdmissionBucket(key="b", capacity=10, max_pending=10))
        db.add(TaskAdmissionTicket(command_id=start, task_id=task, bucket_key="b"))
        db.commit()

    candidate = found(cap_of(host))

    assert candidate is not None and candidate.id == cancel


# ---------------------------------------------------------------- one time


class Clock:
    """A ``_utc_now`` replacement that moves a second per call and counts calls."""

    def __init__(self) -> None:
        self.base = datetime.now(timezone.utc)
        self.calls = 0

    def __call__(self) -> datetime:
        value = self.base + timedelta(seconds=self.calls)
        self.calls += 1
        return value


async def test_every_query_of_a_scan_judges_availability_at_one_time_read_once(
    cap_host: CapHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = cap_host
    fill(host, TIERS["batch-line"])
    clock = Clock()
    start = host.enqueue(host.new_task(), "start", dispatch_class="interactive")
    with host.factory() as db:
        row = db.get(TaskExecutionCommand, start)
        row.retry_available_at = clock.base + timedelta(milliseconds=500)
        db.commit()
    monkeypatch.setattr(transport, "_utc_now", clock)

    # Not yet available at the one time every query uses. With a clock read per
    # query the later queries would see it as due and return it.
    assert found(cap_of(host)) is None
    assert clock.calls == 1


async def test_the_count_reads_the_clock_once(
    cap_host: CapHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = cap_host
    host.enqueue(host.new_task(), "start")
    clock = Clock()
    monkeypatch.setattr(transport, "_utc_now", clock)

    with host.factory() as db:
        transport.count_claimable_task_commands(db, runner_id="worker")

    assert clock.calls == 1


# ---------------------------------------------------------------- replies


async def test_an_interactive_answer_goes_before_an_older_start_on_the_batch_line(
    cap_host: CapHost,
) -> None:
    host = cap_host
    fill(host, TIERS["batch-line"])
    host.enqueue(host.new_task(), "start", dispatch_class="interactive")
    answer = host.enqueue(
        host.new_task(status=TaskStatus.WAITING_FOR_USER),
        "resume_input",
        dispatch_class="interactive",
    )

    candidate = found(cap_of(host))

    assert candidate is not None and candidate.id == answer


# ---------------------------------------------------------------- dispatch


@pytest.mark.parametrize("reserve", [1, 0])
async def test_a_batch_backlog_leaves_the_reserve_to_an_interactive_start(
    cap_host: CapHost, monkeypatch: pytest.MonkeyPatch, reserve: int
) -> None:
    host = cap_host
    set_limit(monkeypatch, LIMIT)
    set_reserve(monkeypatch, reserve)
    batch_tasks = [host.new_task() for _ in range(5)]
    for task in batch_tasks:
        host.enqueue(task, "start")
    interactive_task = host.new_task()
    host.enqueue(interactive_task, "start", dispatch_class="interactive")

    results = [await dispatch(host) for _ in range(6)]

    assert results == [True, True, True, False, False, False]
    if reserve:
        assert host.claimed == [
            (batch_tasks[0], KIND.START),
            (batch_tasks[1], KIND.START),
            (interactive_task, KIND.START),
        ]
    else:
        assert host.claimed == [(task, KIND.START) for task in batch_tasks[:3]]


# ---------------------------------------------------------------- the claim


@pytest.mark.parametrize("tier", ["batch-line", "open"])
async def test_the_claim_applies_the_same_snapshot_as_the_scan(
    cap_host: CapHost, tier: str
) -> None:
    host = cap_host
    fill(host, TIERS[tier])
    task = host.new_task()
    start = host.enqueue(task, "start")
    cap = cap_of(host)
    coordinator = await host.registry.ensure(task)
    assert coordinator is not None

    async def claim() -> Any:
        with host.factory() as db:
            return transport.claim_task_command(
                db, runner_id="worker", command_db_id=start, cap=cap
            )

    claimed = await coordinator.execute_command(
        SimpleNamespace(task_id=task, kind=KIND.START), claim
    )

    if tier == "batch-line":
        assert claimed is None
        assert _state(host, start) == ("pending", 0)
    else:
        assert claimed is not None and claimed.id == start


# ---------------------------------------------------------------- R=0


def where_of(query: Any) -> str:
    clause = query.statement.whereclause
    return str(clause.compile(compile_kwargs={"literal_binds": True}))


NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)


def claimable_where(host: CapHost, cap: Any) -> str:
    with (
        host.factory() as db,
        patch.object(transport, "_utc_now", lambda: NOW),
    ):
        return where_of(
            transport._claimable_query(
                db, runner_id="worker", command_db_id=None, cap=cap
            )
        )


def r0_cap(host: CapHost, active: int, limit: int = 3) -> Any:
    fill(host, active)
    return cap_of(host, DispatchPolicy(limit=limit, interactive_reserve=0))


async def test_with_no_reserve_an_open_snapshot_adds_no_filter(
    cap_host: CapHost,
) -> None:
    host = cap_host
    cap = r0_cap(host, 1)
    assert claimable_where(host, cap) == claimable_where(host, None)
    assert "dispatch_class" not in claimable_where(host, cap)


async def test_with_no_reserve_the_closed_filter_is_the_one_a_reserve_adds(
    cap_host: CapHost,
) -> None:
    host = cap_host
    fill(host, 3)
    no_reserve = cap_of(host, DispatchPolicy(limit=3, interactive_reserve=0))
    with_reserve = cap_of(host, DispatchPolicy(limit=3, interactive_reserve=1))
    assert no_reserve.closed and with_reserve.closed
    assert claimable_where(host, no_reserve) == claimable_where(host, with_reserve)
    assert "dispatch_class" not in claimable_where(host, no_reserve)


def select_count(host: CapHost, cap: Any) -> int:
    engine = host.factory().get_bind()
    statements: list[str] = []

    def record(_conn, _cursor, statement, *_args) -> None:
        if statement.lstrip().upper().startswith("SELECT") and (
            "FROM task_execution_commands" in statement
        ):
            statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        found(cap)
    finally:
        event.remove(engine, "before_cursor_execute", record)
    return len(statements)


async def test_with_no_reserve_a_scan_runs_the_same_number_of_queries(
    cap_host: CapHost,
) -> None:
    host = cap_host
    host.enqueue(host.new_task(), "start")
    fill(host, 1)
    assert (
        select_count(host, cap_of(host, DispatchPolicy(limit=3, interactive_reserve=0)))
        == 2
    )
    fill(host, 2)
    closed = cap_of(host, DispatchPolicy(limit=3, interactive_reserve=0))
    assert closed.closed
    assert select_count(host, closed) == 1


async def test_on_the_batch_line_a_scan_runs_the_open_cap_number_of_queries(
    cap_host: CapHost,
) -> None:
    host = cap_host
    host.enqueue(host.new_task(), "start", dispatch_class="interactive")
    fill(host, 2)
    batch_line = cap_of(host)
    assert batch_line.batch_closed and not batch_line.closed
    # The presence precheck, then the general lookup.
    assert select_count(host, batch_line) == 2
    host.enqueue(
        host.new_task(status=TaskStatus.WAITING_FOR_USER),
        "resume_input",
        dispatch_class="interactive",
    )
    # The presence precheck, then the reply lookup, which finds one and ends
    # the scan.
    assert select_count(host, batch_line) == 2


async def test_a_batch_command_of_a_running_task_is_taken_on_the_batch_line(
    cap_host: CapHost,
) -> None:
    host = cap_host
    running = host.new_task()
    host.live(running)
    fill(host, 1)
    follow_up = host.enqueue(running, "message")
    host.enqueue(host.new_task(), "start", dispatch_class="interactive")

    candidate = found(cap_of(host))

    assert candidate is not None and candidate.id == follow_up
