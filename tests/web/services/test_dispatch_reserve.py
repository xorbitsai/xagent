"""The per-process interactive reserve: policy, snapshot, reservation and scan."""

# Pytest fixture imports are intentionally shadowed by test parameters.
# ruff: noqa: F811

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from tests.web.services.test_dispatch_cap import (  # noqa: F401
    CapHost,
    cap_host,
    set_limit,
)
from xagent import config
from xagent.web.models.task import TaskStatus
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_dispatch_cap, task_execution_admission
from xagent.web.services.task_dispatch_cap import (
    DispatchCapSnapshot,
    DispatchPolicy,
    build_dispatch_cap_snapshot,
    current_dispatch_policy,
)
from xagent.web.services.task_execution import BackgroundTaskManager

LIMIT_ENV = config.DISPATCHER_MAX_INFLIGHT_PER_PROCESS
RESERVE_ENV = config.DISPATCHER_INTERACTIVE_RESERVE_PER_PROCESS


def set_reserve(monkeypatch: pytest.MonkeyPatch, reserve: int) -> None:
    monkeypatch.setenv(RESERVE_ENV, str(reserve))


@pytest.fixture
def fresh_policy_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forget the memoized policy and the already-logged policy."""
    monkeypatch.setattr(task_dispatch_cap, "_policy_memo", None)
    monkeypatch.setattr(task_dispatch_cap, "_logged_policy", None)


# ---------------------------------------------------------------- policy


@pytest.mark.parametrize(
    ("limit", "reserve"),
    [(0, 0), (-1, 0), (1, 1), (1, 2), (4, 4), (4, 5), (4, -1)],
)
def test_policy_rejects_values_outside_its_range(limit: int, reserve: int) -> None:
    with pytest.raises(ValueError):
        DispatchPolicy(limit=limit, interactive_reserve=reserve)


# ---------------------------------------------------------------- I-1

# (limit, reserve, active size, task is active, interactive, admitted)
ADMISSION_ROWS = [
    (1, 0, 0, False, True, True),
    (1, 0, 0, False, False, True),
    (1, 0, 1, False, True, False),
    (1, 0, 1, False, False, False),
    (1, 0, 1, True, False, True),
    (2, 0, 0, False, True, True),
    (2, 0, 0, False, False, True),
    (2, 0, 1, False, True, True),
    (2, 0, 1, False, False, True),
    (2, 0, 2, False, True, False),
    (2, 0, 2, False, False, False),
    (2, 0, 2, True, False, True),
    (2, 1, 0, False, True, True),
    (2, 1, 0, False, False, True),
    (2, 1, 1, False, True, True),
    (2, 1, 1, False, False, False),
    (2, 1, 2, False, True, False),
    (2, 1, 2, False, False, False),
    (2, 1, 2, True, False, True),
    (80, 0, 0, False, True, True),
    (80, 0, 0, False, False, True),
    (80, 0, 79, False, True, True),
    (80, 0, 79, False, False, True),
    (80, 0, 80, False, True, False),
    (80, 0, 80, False, False, False),
    (80, 0, 80, True, False, True),
    (80, 1, 0, False, True, True),
    (80, 1, 0, False, False, True),
    (80, 1, 78, False, True, True),
    (80, 1, 78, False, False, True),
    (80, 1, 79, False, True, True),
    (80, 1, 79, False, False, False),
    (80, 1, 80, False, True, False),
    (80, 1, 80, False, False, False),
    (80, 1, 80, True, False, True),
    (80, 79, 0, False, True, True),
    (80, 79, 0, False, False, True),
    (80, 79, 1, False, True, True),
    (80, 79, 1, False, False, False),
    (80, 79, 79, False, True, True),
    (80, 79, 79, False, False, False),
    (80, 79, 80, False, True, False),
    (80, 79, 80, False, False, False),
    (80, 79, 80, True, False, True),
]


@pytest.mark.parametrize(
    ("limit", "reserve", "active", "task_is_active", "interactive", "admitted"),
    ADMISSION_ROWS,
)
def test_admission_matrix(
    limit: int,
    reserve: int,
    active: int,
    task_is_active: bool,
    interactive: bool,
    admitted: bool,
) -> None:
    snapshot = DispatchCapSnapshot(
        policy=DispatchPolicy(limit=limit, interactive_reserve=reserve),
        active_ids=frozenset(range(100, 100 + active)),
    )
    task_id = 100 if task_is_active else 999
    assert snapshot.admits(task_id, interactive=interactive) is admitted


# ---------------------------------------------------------------- I-2

# (limit env, reserve env, expected policy, config warnings, clamp warnings)
CONFIG_ROWS = [
    pytest.param(None, None, (80, 0), 0, 0, id="unset"),
    pytest.param(None, "0", (80, 0), 0, 0, id="zero"),
    pytest.param(None, "4", (80, 4), 0, 0, id="four"),
    pytest.param(None, " 4 ", (80, 4), 0, 0, id="padded"),
    pytest.param(None, "", (80, 0), 1, 0, id="empty"),
    pytest.param(None, "-1", (80, 0), 1, 0, id="negative"),
    pytest.param(None, "abc", (80, 0), 1, 0, id="not-a-number"),
    pytest.param(None, "80", (80, 79), 0, 1, id="clamped-to-limit-minus-one"),
    pytest.param("1", "4", (1, 0), 0, 1, id="limit-one"),
    pytest.param("2", "4", (2, 1), 0, 1, id="limit-two"),
]


@pytest.mark.parametrize(
    ("limit_env", "reserve_env", "expected", "config_warnings", "clamp_warnings"),
    CONFIG_ROWS,
)
def test_policy_configuration(
    monkeypatch: pytest.MonkeyPatch,
    fresh_policy_state: None,
    limit_env: str | None,
    reserve_env: str | None,
    expected: tuple[int, int],
    config_warnings: int,
    clamp_warnings: int,
) -> None:
    config_logger = Mock()
    cap_logger = Mock()
    monkeypatch.setattr(config, "logger", config_logger)
    monkeypatch.setattr(task_dispatch_cap, "logger", cap_logger)
    monkeypatch.delenv(LIMIT_ENV, raising=False)
    monkeypatch.delenv(RESERVE_ENV, raising=False)
    if limit_env is not None:
        monkeypatch.setenv(LIMIT_ENV, limit_env)
    if reserve_env is not None:
        monkeypatch.setenv(RESERVE_ENV, reserve_env)

    for _ in range(4):
        policy = current_dispatch_policy()
        assert (policy.limit, policy.interactive_reserve) == expected
    assert config_logger.warning.call_count == config_warnings
    assert cap_logger.warning.call_count == clamp_warnings


# ---------------------------------------------------------------- I-7


def _free_slots_taken(limit: int, reserve: int, active: int, interactive: bool) -> int:
    """How many of four new tasks get a slot when ``active`` tasks already run."""
    manager = BackgroundTaskManager()
    manager._cap_clock = lambda: 0.0
    for index in range(active):
        manager._live_executions[10_000 + index] = {object()}  # type: ignore[assignment]
    policy = DispatchPolicy(limit=limit, interactive_reserve=reserve)
    tokens = [
        manager.reserve_dispatch_slot(task_id, policy=policy, interactive=interactive)
        for task_id in range(1, 5)
    ]
    return sum(1 for token in tokens if token is not None)


@pytest.mark.parametrize(
    ("active", "interactive", "expected"),
    [
        pytest.param(10 - 2 - 1, False, 1, id="batch-below-the-batch-line"),
        pytest.param(10 - 2, False, 0, id="batch-at-the-batch-line"),
        pytest.param(10 - 1, True, 1, id="interactive-one-below-the-limit"),
        pytest.param(10 - 2 - 1, True, 3, id="interactive-reserve-plus-one"),
    ],
)
def test_reservations_fill_the_slots_up_to_the_line_of_their_class(
    active: int, interactive: bool, expected: int
) -> None:
    assert _free_slots_taken(10, 2, active, interactive) == expected


# ---------------------------------------------------------------- I-12


def test_gate_held_counts_only_a_full_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = BackgroundTaskManager()
    clock = SimpleNamespace(now=0.0)
    manager._cap_clock = lambda: clock.now
    policy = DispatchPolicy(limit=5, interactive_reserve=2)
    for index in range(3):
        manager._live_executions[index] = {object()}  # type: ignore[assignment]

    manager.dispatch_cap_snapshot(policy)
    clock.now = 10.0
    # Three of five: past the batch line, not at the limit.
    assert manager.drain_gate_held_seconds(10.0) == 0.0

    for index in range(3, 5):
        manager._live_executions[index] = {object()}  # type: ignore[assignment]
    clock.now = 12.0
    manager.dispatch_cap_snapshot(policy)
    assert manager.drain_gate_held_seconds(20.0) == 8.0


# ---------------------------------------------------------------- I-13


async def test_batch_work_stays_at_the_batch_line(
    cap_host: CapHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = cap_host
    set_limit(monkeypatch, 3)
    set_reserve(monkeypatch, 1)
    for _ in range(4):
        host.enqueue(host.new_task(), "start")

    claimed = [
        await asyncio.wait_for(transport.dispatch_one_task_command(host.handler), 10)
        for _ in range(4)
    ]

    assert claimed == [True, True, False, False]
    assert host.registered_active_sizes == [1, 2]


# ---------------------------------------------------------------- I-16


async def test_effective_policy_is_logged_when_it_changes(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fresh_policy_state: None,
) -> None:
    monkeypatch.setattr(
        task_execution_admission, "_installed", task_execution_admission._UPSTREAM
    )
    monkeypatch.setenv(LIMIT_ENV, "10")
    caplog.set_level(logging.INFO, logger=task_dispatch_cap.logger.name)

    monkeypatch.delenv(RESERVE_ENV, raising=False)
    build_dispatch_cap_snapshot()
    monkeypatch.setenv(RESERVE_ENV, "2")
    for _ in range(3):
        build_dispatch_cap_snapshot()
    monkeypatch.setenv(RESERVE_ENV, "3")
    build_dispatch_cap_snapshot()

    lines = [
        r.getMessage() for r in caplog.records if "dispatch policy" in r.getMessage()
    ]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert lines == [
        "dispatch policy limit=10 interactive_reserve=0 "
        "classifier=upstream_command_classifier admission_enabled=False "
        "fair_share=False",
        "dispatch policy limit=10 interactive_reserve=2 "
        "classifier=upstream_command_classifier admission_enabled=False "
        "fair_share=False",
        "dispatch policy limit=10 interactive_reserve=3 "
        "classifier=upstream_command_classifier admission_enabled=False "
        "fair_share=False",
    ]


async def test_installing_a_classifier_after_the_first_snapshot_reports_again(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fresh_policy_state: None,
) -> None:
    monkeypatch.setattr(
        task_execution_admission, "_installed", task_execution_admission._UPSTREAM
    )
    monkeypatch.setenv(LIMIT_ENV, "10")
    monkeypatch.setenv(RESERVE_ENV, "2")
    caplog.set_level(logging.INFO, logger=task_dispatch_cap.logger.name)

    build_dispatch_cap_snapshot()
    build_dispatch_cap_snapshot()
    task_execution_admission.set_command_classifier(
        lambda db, command, view: task_execution_admission.CommandClassification(),
        admission_enabled=True,
    )
    build_dispatch_cap_snapshot()
    build_dispatch_cap_snapshot()

    lines = [
        r.getMessage() for r in caplog.records if "dispatch policy" in r.getMessage()
    ]
    assert len(lines) == 2
    assert "classifier=upstream_command_classifier" in lines[0]
    assert "<lambda>" in lines[1] and "admission_enabled=True" in lines[1]
    assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == [
        "interactive commands of a team are not prioritized inside its admission bucket"
    ]


async def test_admission_with_a_reserve_warns_that_buckets_are_not_prioritized(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fresh_policy_state: None,
) -> None:
    task_execution_admission.set_command_classifier(
        lambda db, command, view: task_execution_admission.CommandClassification(),
        admission_enabled=True,
    )
    monkeypatch.setenv(LIMIT_ENV, "10")
    monkeypatch.setenv(RESERVE_ENV, "2")
    caplog.set_level(logging.INFO, logger=task_dispatch_cap.logger.name)

    build_dispatch_cap_snapshot()
    build_dispatch_cap_snapshot()

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == [
        "interactive commands of a team are not prioritized inside its admission bucket"
    ]
    (policy_line,) = [
        r.getMessage() for r in caplog.records if "dispatch policy" in r.getMessage()
    ]
    assert "admission_enabled=True" in policy_line
    assert "<lambda>" in policy_line


# ---------------------------------------------------------------- I-28


def _reserve(manager, task_id, policy, *, interactive):
    return manager.reserve_dispatch_slot(
        task_id, policy=policy, interactive=interactive
    )


async def test_batch_opened_tasks_never_exceed_the_batch_line() -> None:
    manager = BackgroundTaskManager()
    policy = DispatchPolicy(limit=10, interactive_reserve=2)
    batch_opened: dict[int, Any] = {}

    def start_batch(task_id: int) -> bool:
        token = _reserve(manager, task_id, policy, interactive=False)
        if token is None:
            return False
        batch_opened[task_id] = token
        assert len(batch_opened) <= 8
        return True

    for task_id in range(1, 9):
        assert start_batch(task_id)
    i1 = _reserve(manager, 100, policy, interactive=True)
    i2 = _reserve(manager, 101, policy, interactive=True)
    assert i1 is not None and i2 is not None
    assert len(manager._active_task_ids()) == 10

    # Two batch tasks leave: six batch tasks and the two interactive tasks.
    for task_id in (1, 2):
        manager.release_dispatch_slot(batch_opened.pop(task_id))
    assert len(manager._active_task_ids()) == 8
    # Six batch tasks and two interactive ones: the interactive tasks count too.
    assert not start_batch(150)

    i3 = _reserve(manager, 102, policy, interactive=True)
    i4 = _reserve(manager, 103, policy, interactive=True)
    assert i3 is not None and i4 is not None
    assert len(manager._active_task_ids()) == 10

    # A batch follow-up of each interactive task is admitted.
    for task_id in (100, 101, 102, 103):
        assert _reserve(manager, task_id, policy, interactive=False) is not None
    assert len(manager._active_task_ids()) == 10
    # A new interactive start is refused now, and so is any batch start.
    assert _reserve(manager, 200, policy, interactive=True) is None
    assert not start_batch(201)
    assert len(batch_opened) == 6


# ---------------------------------------------------------------- I-31


@pytest.fixture
async def batch_line(cap_host: CapHost, monkeypatch: pytest.MonkeyPatch) -> CapHost:
    """N=3, R=1, and two running batch tasks: the process sits on the batch line."""
    set_limit(monkeypatch, 3)
    set_reserve(monkeypatch, 1)
    for _ in range(2):
        cap_host.live(cap_host.new_task())
    return cap_host


async def dispatch(host: CapHost, **kwargs: Any) -> bool:
    return await asyncio.wait_for(
        transport.dispatch_one_task_command(host.handler, **kwargs), 10
    )


def _state(host: CapHost, command_id: int) -> tuple[str, int]:
    row = host.row(command_id)
    return row.status, row.attempt_count


async def test_batch_line_scan_takes_interactive_and_control_commands(
    batch_line: CapHost,
) -> None:
    """Between limit-R and limit the scan looks past batch commands it may not
    start and takes controls, commands of active tasks and interactive
    commands, oldest first.
    """
    host = batch_line
    batch_task = host.new_task()
    interactive_task = host.new_task()
    control_task = host.new_task()
    b1 = host.enqueue(batch_task, "start")
    host.enqueue(control_task, "cancel")
    host.enqueue(interactive_task, "start", dispatch_class="interactive")

    # The older cancel needs no slot and is taken on the batch line; the
    # interactive start then takes the last free slot.
    assert [await dispatch(host) for _ in range(3)] == [True, True, False]

    assert host.claimed == [
        (control_task, transport.TaskCommandKind.CANCEL),
        (interactive_task, transport.TaskCommandKind.START),
    ]
    assert _state(host, b1) == ("pending", 0)


async def test_an_interactive_reply_is_taken_on_the_batch_line(
    batch_line: CapHost,
) -> None:
    host = batch_line
    host.enqueue(host.new_task(), "start")
    host.enqueue(host.new_task(), "start", dispatch_class="interactive")
    waiting = host.new_task(status=TaskStatus.WAITING_FOR_USER)
    reply = host.enqueue(waiting, "resume_input", dispatch_class="interactive")

    assert await dispatch(host) is True

    assert host.claimed == [(waiting, transport.TaskCommandKind.RESUME_INPUT)]
    assert host.row(reply).status == "completed"


async def test_dispatch_by_id_takes_an_interactive_start_on_the_batch_line(
    batch_line: CapHost,
) -> None:
    host = batch_line
    host.enqueue(host.new_task(), "start")
    interactive_task = host.new_task()
    i1 = host.enqueue(interactive_task, "start", dispatch_class="interactive")

    assert await dispatch(host, command_db_id=i1) is True

    assert host.claimed == [(interactive_task, transport.TaskCommandKind.START)]


async def test_a_reply_classified_batch_no_longer_blocks_the_scan(
    batch_line: CapHost,
) -> None:
    host = batch_line
    first = host.new_task(status=TaskStatus.WAITING_FOR_USER)
    second = host.new_task(status=TaskStatus.WAITING_FOR_USER)
    held = host.enqueue(first, "resume_input", dispatch_class="batch")
    answered = host.enqueue(second, "resume_input", dispatch_class="interactive")

    assert [await dispatch(host) for _ in range(3)] == [True, False, False]

    assert host.claimed == [(second, transport.TaskCommandKind.RESUME_INPUT)]
    assert host.row(answered).status == "completed"
    assert _state(host, held) == ("pending", 0)


async def test_without_a_reserve_the_same_queue_is_claimed(
    cap_host: CapHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = cap_host
    set_limit(monkeypatch, 3)
    monkeypatch.delenv(RESERVE_ENV, raising=False)
    for _ in range(2):
        host.live(host.new_task())
    batch_task = host.new_task()
    host.enqueue(batch_task, "start")

    assert await dispatch(host) is True

    assert host.claimed == [(batch_task, transport.TaskCommandKind.START)]


# ---------------------------------------------------------------- documentation


def test_example_env_documents_the_reserve() -> None:
    from pathlib import Path

    text = (Path(__file__).resolve().parents[3] / "example.env").read_text()
    assert '# XAGENT_DISPATCHER_INTERACTIVE_RESERVE_PER_PROCESS="0"' in text
    comment = " ".join(
        " ".join(
            line.removeprefix("#").strip()
            for line in text.splitlines()
            if line.startswith("#")
        ).split()
    )
    for phrase in (
        "R (this value)",
        "limit (XAGENT_DISPATCHER_MAX_INFLIGHT_PER_PROCESS)",
        "Between limit-R and limit a cancel or pause also passes the "
        "never-claimed batch commands of its task held at that line, and the "
        "commands queued behind them, and settles them as it does at the cap; "
        "a command already picked up once keeps its place.",
        "cannot use the reserve either; commands of one task still run in order",
        "no interactive command reads the whole claimable backlog",
    ):
        assert phrase in comment
    assert "the queue scan stops at the oldest queued command" not in comment
    assert "Keep this at 0." not in comment
    assert "until the interactive lookup is indexed" not in comment
