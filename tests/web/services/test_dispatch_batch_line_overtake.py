"""A cancel or pause passes the batch commands a process holds on its batch
line, and the commands queued behind them, as it does at the cap."""

# Pytest fixture imports are intentionally shadowed by test parameters.
# ruff: noqa: F811

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import aliased
from sqlalchemy.sql import and_, exists, select

from tests.web.services.test_dispatch_cap import (  # noqa: F401
    CapHost,
    cap_host,
    set_limit,
)
from tests.web.services.test_dispatch_cap_cancel_overtake import host  # noqa: F401
from tests.web.services.test_dispatch_cap_cancel_overtake import (  # noqa: F401
    SETTLED,
    _add_waiting_ticket,
    cap_held_task,
    enqueue,
    enqueue_at,
    enqueue_cancel,
    events_of,
    pause_host,
    run,
    state,
    task_state,
)
from tests.web.services.test_dispatch_cap_pause_overtake import (
    NO_EXECUTION,
    assert_pause_not_applicable,
    assert_settled,
    stage_pause,
)
from tests.web.services.test_dispatch_reserve import set_reserve
from xagent.web.models.task import TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.services import task_command_transport as transport
from xagent.web.services.task_command_transport import (
    COMMAND_PENDING,
    TaskCommandKind,
    _unfinished_earlier_command,
)
from xagent.web.services.task_dispatch_cap import (
    CAP_OVERTAKING_CONTROL_KINDS,
    DispatchCapSnapshot,
    DispatchPolicy,
)
from xagent.web.services.task_execution_admission import waiting_admission

CONTROLS = ("cancel", "pause")
INTERACTIVE = {"dispatch_class": "interactive"}


def cap_with(limit: int, reserve: int, *active: int) -> DispatchCapSnapshot:
    return DispatchCapSnapshot(
        policy=DispatchPolicy(limit=limit, interactive_reserve=reserve),
        active_ids=frozenset(active),
    )


async def claim(
    host: CapHost, task_id: int, control: str, command_id: int, cap: Any
) -> int | None:
    """The claim UPDATE as the dispatcher runs it: under the task's lease."""

    coordinator = await host.registry.ensure(task_id)
    assert coordinator is not None

    async def claim_once() -> Any:
        with host.factory() as db:
            claimed = transport.claim_task_command(
                db, runner_id="worker", command_db_id=command_id, cap=cap
            )
            return None if claimed is None else claimed.id

    return await coordinator.execute_command(
        SimpleNamespace(task_id=task_id, kind=TaskCommandKind(control)), claim_once
    )


# ---------------------------------------------------- which earlier commands a control passes

# A shape is what the snapshot makes of the numbers:
# (limit, reserve, other tasks the process holds, own task is active, tier)
# where the tier is "open", "batch-line" or "closed".
SHAPES = {
    "open": (2, 1, 0, False, "open"),
    "batch-line": (2, 1, 1, False, "batch-line"),
    "closed": (2, 1, 2, False, "closed"),
    "task-active": (3, 1, 1, True, "batch-line"),
    "r0-open": (1, 0, 0, False, "open"),
    "r0-closed": (1, 0, 1, False, "closed"),
}
TIER_FLAGS = {
    "open": (False, False),
    "batch-line": (False, True),
    "closed": (True, True),
}
# name -> (shape, earlier commands as (kind, fields), the control passes every
# earlier one, the scan may take the first earlier command)
# The control follows the earlier commands.
# fmt: off
ROWS = {
    "closed-batch-start": ("closed", [("start", {})], True, False),
    "closed-interactive-start": ("closed", [("start", INTERACTIVE)], True, False),
    "batch-line-batch-start": ("batch-line", [("start", {})], True, False),
    "batch-line-interactive-start": ("batch-line", [("start", INTERACTIVE)], False, True),
    "batch-line-batch-start-then-reply": ("batch-line", [("start", {}), ("resume_input", INTERACTIVE)], True, False),
    "batch-line-batch-message-then-reply": ("batch-line", [("message", {}), ("resume_input", INTERACTIVE)], True, False),
    # A command picked up once keeps its place: it may already have had an effect.
    "batch-line-claimed-once-batch-start": ("batch-line", [("start", {"attempt_count": 1})], False, False),
    "batch-line-task-active": ("task-active", [("start", {})], False, True),
    "batch-line-ticketed-start": ("batch-line", [("start", {})], True, False),
    "open-batch-start": ("open", [("start", {})], False, True),
    "R0-closed-batch-start": ("r0-closed", [("start", {})], True, False),
    "R0-open-batch-start": ("r0-open", [("start", {})], False, True),
}
# fmt: on


@pytest.mark.parametrize("control", CONTROLS)
@pytest.mark.parametrize("name", list(ROWS))
async def test_control_overtakes_held_batch_command_below_cap(
    cap_host: CapHost, name: str, control: str
) -> None:
    shape, earlier, passes, takes_first = ROWS[name]
    limit, reserve, others, own_active, tier = SHAPES[shape]
    host = cap_host
    waiting_for_user = any(kind == "resume_input" for kind, _ in earlier)
    task_id = host.new_task(
        status=TaskStatus.WAITING_FOR_USER if waiting_for_user else TaskStatus.PENDING
    )
    earlier_ids = [enqueue(host, task_id, kind, **fields) for kind, fields in earlier]
    if name == "batch-line-ticketed-start":
        _add_waiting_ticket(host, task_id, earlier_ids[0])
    control_id = enqueue(host, task_id, control)
    active = [host.new_task() for _ in range(others)]
    if own_active:
        active.append(task_id)
    cap = cap_with(limit, reserve, *active)
    assert (cap.closed, cap.batch_closed) == TIER_FLAGS[tier]

    layer1 = transport._find_command_candidate(None, (), cap)
    layer2 = transport._find_command_candidate(control_id, (), cap)
    claimed = await claim(host, task_id, control, control_id, cap)

    if passes:
        assert layer1 is not None and layer1.id == control_id
        assert layer2 is not None and layer2.id == control_id
        assert claimed == control_id
    else:
        # The earlier command is the only one the scan may take, or none is.
        assert (layer1.id if layer1 else None) == (
            earlier_ids[0] if takes_first else None
        )
        assert layer2 is None
        assert claimed is None


# ------------------------------------------------- overtake on the batch line, end to end


@pytest.fixture
def batch_line_host(host: CapHost, monkeypatch: pytest.MonkeyPatch) -> CapHost:
    """N=2, R=1: one other running task puts the process on its batch line."""
    set_limit(monkeypatch, 2)
    set_reserve(monkeypatch, 1)
    return host


@pytest.fixture
def batch_line_pause_host(
    pause_host: CapHost, monkeypatch: pytest.MonkeyPatch
) -> CapHost:
    set_limit(monkeypatch, 2)
    set_reserve(monkeypatch, 1)
    return pause_host


def assert_on_the_batch_line(host: CapHost) -> None:
    policy = DispatchPolicy(limit=2, interactive_reserve=1)
    cap = host.manager.dispatch_cap_snapshot(policy)
    assert cap.batch_closed and not cap.closed


async def _end_to_end_waiting_task(
    host: CapHost, first_kind: str
) -> tuple[int, int, int, int]:
    """A waiting task on the batch line with, in order, a held batch command
    (never run) and an interactive answer: ``(agent, task, held, answer)``."""

    agent_id, task_id = await cap_held_task(host, TaskStatus.WAITING_FOR_USER)
    assert_on_the_batch_line(host)
    fields: dict[str, Any] = {}
    if first_kind == "start":
        fields["run_id"] = "run-new"
    held_id = enqueue_at(host, task_id, first_kind, 3, **fields)
    answer_id = enqueue_at(host, task_id, "resume_input", 3, **INTERACTIVE)
    return agent_id, task_id, held_id, answer_id


@pytest.mark.parametrize("first_kind", ["start", "message"])
async def test_cancel_passes_a_held_batch_command_and_the_answer_behind_it(
    batch_line_host: CapHost, first_kind: str
) -> None:
    host = batch_line_host
    agent_id, task_id, held_id, answer_id = await _end_to_end_waiting_task(
        host, first_kind
    )
    cancel_id = enqueue_cancel(host, task_id, agent_id, 3)

    # The control is the first command the scan takes.
    candidate = transport._find_command_candidate(
        None,
        (),
        host.manager.dispatch_cap_snapshot(
            DispatchPolicy(limit=2, interactive_reserve=1)
        ),
    )
    assert candidate is not None and candidate.id == cancel_id

    assert await run() is True

    cancel_row = host.row(cancel_id)
    assert cancel_row.status == "completed"
    assert (cancel_row.result or {}).get("rejection_reason") != "stale_run"
    assert state(host, held_id) == SETTLED
    assert state(host, answer_id) == SETTLED
    assert len(events_of(host, held_id)) == 1
    assert len(events_of(host, answer_id)) == 1
    assert task_state(host, task_id)[0] == TaskStatus.FAILED
    assert task_state(host, task_id)[2] == 4


@pytest.mark.parametrize("first_kind", ["start", "message"])
async def test_pause_passes_a_held_batch_command_and_the_answer_behind_it(
    batch_line_pause_host: CapHost, first_kind: str
) -> None:
    host = batch_line_pause_host
    _, task_id, held_id, answer_id = await _end_to_end_waiting_task(host, first_kind)
    pause_id = stage_pause(host, task_id)

    assert await run() is True

    # Nothing of the task runs: the pause is refused without building an
    # agent, and both commands it passed are settled in the same transaction.
    assert_pause_not_applicable(host, pause_id, NO_EXECUTION)
    assert (host.row(pause_id).result or {}).get("rejection_reason") != "stale_run"
    assert_settled(host, held_id, answer_id)
    assert task_state(host, task_id) == (TaskStatus.WAITING_FOR_USER, None, 3)


# A control passed by id, the way the accepting process dispatches it.
@pytest.mark.parametrize("control", CONTROLS)
async def test_a_control_dispatched_by_id_passes_a_held_batch_start(
    batch_line_pause_host: CapHost, control: str
) -> None:
    host = batch_line_pause_host
    agent_id, task_id = await cap_held_task(host, TaskStatus.WAITING_FOR_USER)
    assert_on_the_batch_line(host)
    start_id = enqueue_at(host, task_id, "start", 3)
    if control == "cancel":
        control_id = enqueue_cancel(host, task_id, agent_id, 3)
    else:
        control_id = stage_pause(host, task_id)

    assert await run(control_id) is True

    if control == "cancel":
        assert host.row(control_id).status == "completed"
        assert task_state(host, task_id)[2] == 4
    else:
        assert_pause_not_applicable(host, control_id, NO_EXECUTION)
    assert host.row(control_id).attempt_count == 1
    assert state(host, start_id) == SETTLED


# ---------------------------------------------------------- the overtake SQL at R = 0


def _closed_only_reference(cap: DispatchCapSnapshot | None) -> Any:
    """The at-cap branch only: what R=0 compiles to."""
    from xagent.web.services.task_dispatch_cap import EXECUTION_OPENING_KINDS

    earlier = aliased(TaskExecutionCommand)
    clauses = [
        earlier.task_id == TaskExecutionCommand.task_id,
        earlier.id < TaskExecutionCommand.id,
        earlier.status.notin_(transport.COMMAND_TERMINAL),
        ~and_(
            TaskExecutionCommand.kind.in_(("cancel", "pause")),
            earlier.status == COMMAND_PENDING,
            waiting_admission(earlier.id),
        ),
    ]
    if cap is not None and cap.closed:
        clauses.append(
            ~and_(
                TaskExecutionCommand.kind.in_(sorted(CAP_OVERTAKING_CONTROL_KINDS)),
                earlier.status == COMMAND_PENDING,
                earlier.kind.in_(sorted(EXECUTION_OPENING_KINDS)),
                earlier.attempt_count == 0,
                TaskExecutionCommand.task_id.notin_(sorted(cap.active_ids)),
            )
        )
    return exists(select(1).where(*clauses))


def _compiled(clause: Any, dialect: Any) -> str:
    return str(clause.compile(dialect=dialect, compile_kwargs={"literal_binds": True}))


R0_CAPS = {
    "no-cap": None,
    "open": cap_with(2, 0),
    "closed": cap_with(1, 0, 7),
}


@pytest.mark.parametrize("dialect", [postgresql.dialect(), sqlite.dialect()])
@pytest.mark.parametrize("name", list(R0_CAPS))
def test_without_a_reserve_the_overtake_clause_is_unchanged(
    name: str, dialect: Any
) -> None:
    cap = R0_CAPS[name]
    assert _compiled(_unfinished_earlier_command(cap), dialect) == _compiled(
        _closed_only_reference(cap), dialect
    )


@pytest.mark.parametrize("dialect", [postgresql.dialect(), sqlite.dialect()])
def test_the_held_below_cap_condition_is_only_built_below_the_cap(
    dialect: Any,
) -> None:
    closed = _compiled(_unfinished_earlier_command(cap_with(2, 1, 7, 8)), dialect)
    batch_line = _compiled(_unfinished_earlier_command(cap_with(2, 1, 7)), dialect)

    assert "dispatch_class" not in closed
    assert "dispatch_class" in batch_line
    assert "EXISTS" in batch_line.upper()
