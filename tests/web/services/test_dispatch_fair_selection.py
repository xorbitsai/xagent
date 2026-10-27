"""Fair-share dispatch: the switch, the cap snapshot's slot counts and the tenant rotation."""

import logging
import random
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql, sqlite

from tests.web.services.fair_selection_support import (
    QUEUED,
    WORLDS,
    FakeClock,
    RecordingLock,
    World,
    attempt,
    cap_of,
    play,
    random_world,
    scan,
    store_world,
)
from xagent import config
from xagent.web.models import database as database_module
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_coordinator_runtime as runtime
from xagent.web.services import task_coordinator_service as ownership
from xagent.web.services import task_dispatch_cap, task_execution
from xagent.web.services import task_execution_admission as admission
from xagent.web.services.task_dispatch_cap import DispatchCapSnapshot, DispatchPolicy
from xagent.web.services.task_execution import BackgroundTaskManager

SWITCH = "XAGENT_DISPATCHER_FAIR_SHARE"
LARGE = DispatchPolicy(limit=10, interactive_reserve=0)
POLICY = DispatchPolicy(limit=2, interactive_reserve=1)


# --- the cap snapshot's slot counts ---------------------------------------------


def reserve(manager, task_id, key):
    token = manager.reserve_dispatch_slot(
        task_id, policy=LARGE, interactive=False, fairness_key=key
    )
    assert token is not None
    return token


async def test_the_snapshot_counts_the_slots_each_tenant_holds():
    manager = BackgroundTaskManager()
    for task_id, key in ((1, "a"), (2, "a"), (3, "b")):
        reserve(manager, task_id, key)

    snapshot = manager.dispatch_cap_snapshot(LARGE)

    assert dict(snapshot.occupancy) == {"a": 2, "b": 1}
    assert dict(snapshot.occupancy) == dict(
        manager._tenant_occupancy(snapshot.active_ids)
    )
    tenants, top = manager.slot_tenant_counts()
    assert (len(snapshot.occupancy), max(snapshot.occupancy.values())) == (tenants, top)


async def test_the_snapshot_drops_the_tenant_of_a_task_that_left_the_active_set():
    manager = BackgroundTaskManager()
    token = reserve(manager, 1, "a")
    reserve(manager, 2, "b")
    manager.release_dispatch_slot(token)

    snapshot = manager.dispatch_cap_snapshot(LARGE)

    assert dict(snapshot.occupancy) == {"b": 1}


async def test_the_snapshot_that_only_observes_the_cap_state_has_no_counts(monkeypatch):
    manager = BackgroundTaskManager()
    reserve(manager, 1, "a")
    built: list[DispatchCapSnapshot] = []

    def record(**fields):
        built.append(DispatchCapSnapshot(**fields))
        return built[-1]

    monkeypatch.setattr(task_execution, "DispatchCapSnapshot", record)

    manager._observe_current_cap_state(0.0)

    assert [dict(snapshot.occupancy) for snapshot in built] == [{}]


async def test_a_snapshot_can_be_hashed_and_its_counts_do_not_decide_equality():
    manager = BackgroundTaskManager()
    reserve(manager, 1, "a")
    snapshot = manager.dispatch_cap_snapshot(LARGE)
    same_ids = DispatchCapSnapshot(policy=LARGE, active_ids=snapshot.active_ids)

    assert hash(snapshot) == hash(same_ids)
    assert snapshot == same_ids


# --- the switch --------------------------------------------------------------------


@pytest.fixture
def switch(monkeypatch, caplog):
    """The dispatch-side switch with no memory from an earlier test."""
    monkeypatch.setattr(transport, "_fair_share_memo", None)
    monkeypatch.delenv(SWITCH, raising=False)
    caplog.set_level(logging.WARNING, logger="xagent.config")

    def warnings() -> list[str]:
        return [
            record.getMessage()
            for record in caplog.records
            if record.name == "xagent.config" and record.levelno == logging.WARNING
        ]

    return warnings


def test_a_wrong_value_warns_once_however_often_it_is_read(monkeypatch, switch):
    monkeypatch.setenv(SWITCH, "garbage")

    assert [transport._fair_share_enabled() for _ in range(100)] == [False] * 100

    assert len(switch()) == 1 and SWITCH in switch()[0]


def test_another_wrong_value_warns_again_and_a_right_one_takes_effect_at_once(
    monkeypatch, switch
):
    monkeypatch.setenv(SWITCH, "garbage")
    assert transport._fair_share_enabled() is False
    monkeypatch.setenv(SWITCH, "other garbage")
    assert transport._fair_share_enabled() is False
    assert len(switch()) == 2

    monkeypatch.setenv(SWITCH, "true")
    assert transport._fair_share_enabled() is True
    monkeypatch.delenv(SWITCH)
    assert transport._fair_share_enabled() is False
    assert len(switch()) == 2


def test_the_switch_reads_the_configured_value(monkeypatch, switch):
    assert config.DISPATCHER_FAIR_SHARE == SWITCH
    monkeypatch.setenv(SWITCH, "false")
    assert transport._fair_share_enabled() is False
    assert switch() == []


async def test_the_policy_line_reports_the_switch_when_its_value_changes(
    monkeypatch, caplog
):
    monkeypatch.setattr(task_dispatch_cap, "_policy_memo", None)
    monkeypatch.setattr(task_dispatch_cap, "_logged_policy", None)
    monkeypatch.setattr(admission, "_installed", admission._UPSTREAM)
    monkeypatch.setenv(config.DISPATCHER_MAX_INFLIGHT_PER_PROCESS, "10")
    monkeypatch.delenv(config.DISPATCHER_INTERACTIVE_RESERVE_PER_PROCESS, raising=False)
    monkeypatch.delenv(SWITCH, raising=False)
    # The handler keeps the level of the last call, so the lower one goes last.
    caplog.set_level(logging.WARNING, logger="xagent.config")
    caplog.set_level(logging.INFO, logger=task_dispatch_cap.logger.name)

    for raw, snapshots in ((None, 2), ("true", 2), ("garbage", 3), ("YES", 1)):
        if raw is None:
            monkeypatch.delenv(SWITCH, raising=False)
        else:
            monkeypatch.setenv(SWITCH, raw)
        for _ in range(snapshots):
            task_dispatch_cap.build_dispatch_cap_snapshot()

    lines = [
        record.getMessage()
        for record in caplog.records
        if "dispatch policy" in record.getMessage()
    ]
    assert [line.rsplit("fair_share=", 1)[1] for line in lines] == [
        "False",
        "True",
        "False",
        "True",
    ]
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.name == "xagent.config" and record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1 and SWITCH in warnings[0]


# --- a SQLite world of queued commands --------------------------------------------


@pytest.fixture
def world(tmp_path, monkeypatch):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'fair.db'}")
    Base.metadata.create_all(engine)
    built = World(engine)
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(database_module, "_SessionLocal", built.sessions)
    monkeypatch.setattr(transport, "get_runner_id", lambda: "me")
    try:
        yield built
    finally:
        engine.dispose()


def statements_of(engine):
    seen: list[str] = []

    def record(_conn, _cursor, statement, *_args):
        seen.append(" ".join(statement.split()))

    sa.event.listen(engine, "before_cursor_execute", record)
    return seen, lambda: sa.event.remove(engine, "before_cursor_execute", record)


def test_listing_gives_the_oldest_queued_command_of_each_tenant_and_class(world):
    first = world.add("a", "batch")
    world.add("a", "batch", status="processing")
    third = world.add("a", "interactive")
    world.add("b", "batch", status="completed")
    world.add("b", "batch", status="failed")
    sixth = world.add("b", "interactive")
    seen, stop = statements_of(world.engine)
    try:
        with world.sessions() as db:
            tenants = transport._list_queued_tenants(db)
    finally:
        stop()

    assert tenants == {
        "a": {"batch": first, "interactive": third},
        "b": {"interactive": sixth},
    }
    assert len(seen) == 1


def test_listing_counts_a_command_that_cannot_be_claimed_as_queued(world):
    pinned = world.add("a", pinned=True)
    unavailable = world.add("b", unavailable=True)
    with world.sessions() as db:
        assert transport._list_queued_tenants(db) == {
            "a": {"batch": pinned},
            "b": {"batch": unavailable},
        }


def test_between_the_batch_line_and_the_limit_only_interactive_commands_count():
    tenants = {
        "a": {"batch": 1, "interactive": 3},
        "b": {"batch": 2},
        "c": {"interactive": 4},
    }
    open_cap = DispatchCapSnapshot(policy=POLICY, active_ids=frozenset())
    band = DispatchCapSnapshot(policy=POLICY, active_ids=frozenset({90}))

    assert transport._tenants_in_band(tenants, open_cap) == tenants
    assert transport._tenants_in_band(tenants, band) == {
        "a": {"interactive": 3},
        "c": {"interactive": 4},
    }
    assert transport._tenants_in_band(tenants, None) == tenants


def test_the_window_is_a_range_on_the_key_so_that_the_index_serves_it():
    sql = str(
        transport._queued_tenant_window("user:1", "batch").compile(
            dialect=postgresql.dialect()
        )
    )

    assert "fairness_key >=" in sql and "fairness_key <=" in sql
    assert "fairness_key =" not in sql
    assert "ORDER BY task_execution_commands.fairness_key, " in sql
    assert "dispatch_class, task_execution_commands.priority DESC, " in sql
    assert transport.FAIR_SHARE_WINDOW == 16
    assert f"LIMIT {transport.FAIR_SHARE_WINDOW}" in str(
        transport._queued_tenant_window("user:1", "batch").compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )


# --- one scan of the queue ------------------------------------------------------


@pytest.fixture
def clock(monkeypatch):
    """A moveable clock and an empty skip table, counters and miss count."""
    fake = FakeClock()
    monkeypatch.setattr(transport, "_fair_clock", fake)
    monkeypatch.setattr(transport, "_fair_state", transport._FairState())
    return fake


@pytest.fixture
def on(monkeypatch, clock):
    monkeypatch.setenv(SWITCH, "true")
    monkeypatch.setattr(transport, "_fair_share_memo", None)


def recorded(world, action):
    """(result, [(statement, parameters)]) of everything ``action`` sent."""
    seen: list[tuple[str, tuple]] = []

    def record(_conn, _cursor, statement, parameters, *_args):
        seen.append((" ".join(statement.split()), tuple(parameters)))

    sa.event.listen(world.engine, "before_cursor_execute", record)
    try:
        return action(), seen
    finally:
        sa.event.remove(world.engine, "before_cursor_execute", record)


def windows_asked(seen, classes=("batch", "interactive")):
    """The dispatch class of each window query among the recorded statements."""
    return [
        next(value for value in parameters if value in classes)
        for statement, parameters in seen
        if "fairness_key >=" in statement
    ]


def test_a_tenant_whose_oldest_commands_cannot_be_claimed_is_missed_then_skipped(
    world, on, clock
):
    for _ in range(16):
        world.add("a", pinned=True)
    world.add("a")
    first_b = world.add("b")
    for _ in range(19):
        world.add("b")
    cap = cap_of(occupancy={"b": 5})

    chosen, seen = recorded(world, lambda: scan(cap, attempt=attempt()))

    # The sixteen oldest queued commands of a are all pinned elsewhere: a misses.
    assert chosen == first_b
    assert windows_asked(seen) == ["batch", "batch"]
    assert transport.drain_fair_misses() == 1
    _, again = recorded(world, lambda: scan(cap, attempt=attempt()))
    assert windows_asked(again) == ["batch"]
    assert transport.drain_fair_misses() == 0


def test_every_eighth_counted_attempt_takes_the_oldest_claimable_command(
    world, on, clock
):
    for _ in range(16):
        world.add("a", pinned=True)
    oldest_claimable = world.add("a")
    first_b = world.add("b")
    for _ in range(19):
        world.add("b")
    cap = cap_of(occupancy={"b": 5})
    closed = cap_of(active=range(100, 140))
    results = []
    for _ in range(7):
        results.append(scan(cap, attempt=attempt()))
        # Not counted: an explicit id, or a cap with no room.
        scan(cap, attempt=attempt(), command_db_id=first_b)
        scan(closed, attempt=attempt())

    assert results == [first_b] * 7

    assert scan(cap, attempt=attempt()) == oldest_claimable
    assert transport._fair_state.attempts == 8


def test_an_attempt_that_rescans_is_counted_once_and_lists_the_queue_once(
    world, on, clock
):
    world.add("a")
    world.add("b")
    one = attempt()
    cap = cap_of()

    _, seen = recorded(world, lambda: (scan(cap, attempt=one), scan(cap, attempt=one)))

    assert transport._fair_state.attempts == 1
    assert sum("GROUP BY" in statement for statement, _ in seen) == 1


def test_a_skip_lasts_two_seconds_and_is_kept_per_class(world, on, clock):
    world.add("a", "batch", pinned=True)
    world.add("a", "interactive")
    world.add("b", "batch")
    cap = cap_of()

    _, at_start = recorded(world, lambda: scan(cap, attempt=attempt()))
    clock.now += 1.9
    _, before_expiry = recorded(world, lambda: scan(cap, attempt=attempt()))
    clock.now += 0.1
    _, at_expiry = recorded(world, lambda: scan(cap, attempt=attempt()))

    assert windows_asked(at_start) == ["batch", "interactive"]
    assert windows_asked(before_expiry) == ["interactive"]
    assert windows_asked(at_expiry) == ["batch", "interactive"]


def test_a_command_of_a_task_leased_elsewhere_is_not_found_by_the_window(
    world, on, clock
):
    pinned = [world.add("a", pinned=True) for _ in range(16)]
    seventeenth = world.add("a")
    first_b = world.add("b")
    for _ in range(19):
        world.add("b")
    cap = cap_of(occupancy={"b": 5})

    # a holds fewer slots, but its window is full of commands that wait for another
    # process, so b is served; only the round that reads everything reaches a.
    assert [scan(cap, attempt=attempt()) for _ in range(7)] == [first_b] * 7
    assert scan(cap, attempt=attempt()) == seventeenth

    # Once the leases lapse, a's own window finds its oldest command.
    with world.sessions() as db:
        db.execute(
            sa.update(Task)
            .where(Task.runner_id == "other")
            .values(lease_expires_at=datetime.now(timezone.utc) - timedelta(minutes=1))
        )
        db.commit()
    clock.now += 2.0
    assert scan(cap, attempt=attempt()) == pinned[0]


def test_a_reply_is_served_before_any_tenant_is_ranked(world, on, clock):
    world.add("a")
    reply = world.add("b", "interactive", kind="resume_input")
    world.add("c")
    cap = cap_of()
    off, off_seen = recorded(world, lambda: scan(cap))

    chosen, seen = recorded(world, lambda: scan(cap, attempt=attempt()))

    assert chosen == off == reply
    assert [statement for statement, _ in seen] == [s for s, _ in off_seen]
    assert transport._fair_state.attempts == 1


def test_a_scan_for_one_command_sends_what_it_always_sent_and_is_not_counted(
    world, on, clock
):
    target = world.add("a")
    world.add("b")
    cap = cap_of()
    off, off_seen = recorded(world, lambda: scan(cap, command_db_id=target))

    chosen, seen = recorded(
        world, lambda: scan(cap, attempt=attempt(), command_db_id=target)
    )

    assert chosen == off == target
    assert [statement for statement, _ in seen] == [s for s, _ in off_seen]
    assert transport._fair_state.attempts == 0


def test_a_control_command_with_a_stale_key_does_not_make_a_tenant_of_its_own(
    world, on, clock
):
    first = world.add("a")
    world.add("a")
    world.add("b", kind="pause")  # carries a key its task no longer has
    cap = cap_of(active=[900], occupancy={"a": 1})
    off = scan(cap)

    chosen = scan(cap, attempt=attempt())

    assert chosen == off == first
    with world.sessions() as db:
        assert transport._list_queued_tenants(db) == {"a": {"batch": first}}


def test_a_control_command_with_a_stale_key_does_not_hide_the_commands_ahead_of_it(
    world, on, clock
):
    for _ in range(16):
        world.add("a", pinned=True)
    seventeenth = world.add("a")
    world.add("b", kind="pause")
    cap = cap_of()
    off = scan(cap)

    chosen = scan(cap, attempt=attempt())

    # One tenant has work that opens an execution, so the scan reads as it always
    # did: no window is asked, nothing is missed or skipped.
    assert chosen == off == seventeenth
    assert transport.drain_fair_misses() == 0


def test_a_queue_with_no_interactive_command_between_the_lines_reads_as_today(
    world, on, clock, monkeypatch
):
    batch_a = world.add("a", "batch")
    world.add("b", "batch")
    lock = RecordingLock()
    monkeypatch.setattr(transport, "_fair_state_lock", lock)
    band = cap_of(active=[900], policy=POLICY)
    assert band.batch_closed and not band.closed
    off, _ = recorded(world, lambda: scan(band))

    chosen = scan(band, attempt=attempt())

    # The batch commands may not open an execution on the batch line; this scan
    # finds what today's scan finds, and ranks nobody: the lock is taken once to
    # count the attempt and once to read the skip table for the tenant count.
    assert chosen == off
    assert lock.taken == 2
    assert batch_a is not None


def running_tenant_and_newer_task(world):
    """Tenant A runs a task and has a newer task of its own; B has the newest.

    A holds a slot, so B ranks first and its start is the selected command.
    """
    a_start = world.add("A", status="completed")
    a_task = world.task_of(a_start)
    world.add("A")  # a new task of A: older than B's, but it needs a new slot
    return a_task


def test_a_follow_up_of_a_running_task_goes_before_a_newer_task_of_another_tenant(
    world, on, clock
):
    a_task = running_tenant_and_newer_task(world)
    message = world.follow_up(a_task, "A")
    world.add("B")
    cap = cap_of(active={a_task}, occupancy={"A": 1})

    assert scan(cap, attempt=attempt()) == message


def test_a_cancel_goes_before_a_newer_task_of_another_tenant(world, on, clock):
    running_tenant_and_newer_task(world)
    cancel = world.add("A", kind="cancel")  # a task of its own, not a running one
    world.add("B")
    cap = cap_of(active=(), occupancy={"A": 1})

    assert scan(cap, attempt=attempt()) == cancel


def test_the_lookup_among_the_running_tasks_is_sent_only_when_a_task_runs(
    world, on, clock
):
    a_task = running_tenant_and_newer_task(world)
    world.add("B")

    def of_tasks(active):
        _, seen = recorded(
            world,
            lambda: scan(cap_of(active=active, occupancy={"A": 1}), attempt=attempt()),
        )
        return [s for s, _ in seen if "running_task" in s]

    assert of_tasks(()) == []
    assert len(of_tasks({a_task})) == 1


@pytest.mark.parametrize(
    "order", ["first_task_first", "second_task_first"], ids=lambda order: order
)
def test_of_follow_ups_of_two_running_tasks_the_older_is_taken(world, on, clock, order):
    first = world.task_of(world.add("A", status="completed"))
    second = world.task_of(world.add("A", status="completed"))
    world.add("A")  # a new task of A: older than B's, but it needs a new slot
    if order == "first_task_first":
        older = world.follow_up(first, "A")
        newer = world.follow_up(second, "A")
    else:
        older = world.follow_up(second, "A")
        newer = world.follow_up(first, "A")
    assert older < newer
    world.add("B")
    cap = cap_of(active={first, second}, occupancy={"A": 2})

    assert scan(cap, attempt=attempt()) == older


@pytest.mark.parametrize(
    "order", ["control_first", "follow_up_first"], ids=lambda order: order
)
def test_of_two_older_commands_needing_no_slot_the_older_is_taken(
    world, on, clock, order
):
    a_task = running_tenant_and_newer_task(world)
    if order == "control_first":
        cancel = world.add("A", kind="cancel")
        message = world.follow_up(a_task, "A")
    else:
        message = world.follow_up(a_task, "A")
        cancel = world.add("A", kind="cancel")
    world.add("B")
    cap = cap_of(active={a_task}, occupancy={"A": 1})

    assert scan(cap, attempt=attempt()) == min(cancel, message)


def test_a_cancel_of_a_running_task_matches_both_lookups_and_is_taken(world, on, clock):
    a_task = running_tenant_and_newer_task(world)
    cancel = world.follow_up(a_task, "A", kind="cancel")
    world.add("B")
    cap = cap_of(active={a_task}, occupancy={"A": 1})

    assert scan(cap, attempt=attempt()) == cancel


@pytest.mark.parametrize(
    "raw", [None, "false", "garbage"], ids=["unset", "false", "garbage"]
)
@pytest.mark.parametrize("between_the_lines", [False, True], ids=["room", "band"])
def test_with_the_switch_off_a_scan_lists_no_tenant_and_takes_the_oldest(
    world, monkeypatch, clock, raw, between_the_lines
):
    ids = [world.add(*row[:2], kind=row[2]) for row in WORLDS["three"]]
    if between_the_lines:
        reply = world.add("user:2", "interactive", kind="resume_input")
        cap = cap_of(active=[900], occupancy={"user:1": 1}, policy=POLICY)
        assert cap.in_reserve_band
    else:
        cap = cap_of(occupancy={"user:1": 1})
    if raw is None:
        monkeypatch.delenv(SWITCH, raising=False)
    else:
        monkeypatch.setenv(SWITCH, raw)
    monkeypatch.setattr(transport, "_fair_share_memo", None)
    assert transport._new_fair_attempt() is None

    chosen, seen = recorded(world, lambda: scan(cap, attempt=None))

    assert chosen == (reply if between_the_lines else ids[0])
    assert not any("RECURSIVE" in s or "GROUP BY" in s for s, _ in seen)
    assert windows_asked(seen) == []
    assert transport._fair_state.attempts == 0
    # The same world with the switch on is served differently, so the checks above
    # are not a scan that would have found the oldest command anyway.
    if between_the_lines:
        return
    monkeypatch.setenv(SWITCH, "true")
    monkeypatch.setattr(transport, "_fair_share_memo", None)
    assert scan(cap, attempt=attempt()) == ids[1]


async def test_the_dispatch_loop_counts_one_attempt_and_lists_the_queue_once_over_a_rescan(
    world, on, clock, monkeypatch
):
    world.add("a")
    world.add("b")
    manager = BackgroundTaskManager()
    monkeypatch.setattr(task_execution, "background_task_manager", manager)
    monkeypatch.setattr(transport, "build_dispatch_cap_snapshot", lambda: cap_of())
    ensured: list[int] = []

    async def ensure(task_id):
        ensured.append(task_id)
        if len(ensured) == 1:
            # A coordinator that still runs a command: the loop sets the task
            # aside and scans again.
            return SimpleNamespace(_command_tasks={object()})
        return SimpleNamespace(
            _command_tasks=set(),
            execute_command=AsyncMock(return_value=False),
            require_recovery=lambda: None,
        )

    registry = SimpleNamespace(busy_command_tasks=lambda: set(), ensure=ensure)
    monkeypatch.setattr(runtime, "get_task_coordinator_registry", lambda: registry)

    async def executor(command):
        raise AssertionError("no command should be applied")

    seen, stop = statements_of(world.engine)
    try:
        await transport.dispatch_one_task_command(executor)
    finally:
        stop()

    assert transport._fair_state.attempts == 1
    assert sum("GROUP BY" in statement for statement in seen) == 1
    assert len(ensured) == 2 and ensured[0] != ensured[1]
    assert not manager._dispatch_reservations


def test_follow_ups_of_one_task_that_fill_a_tenants_window_make_it_miss(
    world, on, clock
):
    """A tenant whose oldest FAIR_SHARE_WINDOW (16) queued commands are all follow-ups
    of one task that cannot be claimed this time counts as missed, although it has
    newer tasks behind them: the window is a known limit of the selection."""
    running = world.add("A", status="processing")
    task = world.task_of(running)
    for _ in range(16):
        world.follow_up(task, "A")
    world.add("A")  # behind the window
    b_new = world.add("B")
    cap = cap_of(active={task}, occupancy={"B": 1})

    chosen, seen = recorded(world, lambda: scan(cap, attempt=attempt(), busy=(task,)))

    assert chosen == b_new
    assert transport.drain_fair_misses() == 1
    assert windows_asked(seen) == ["batch", "batch"]
    again, seen = recorded(world, lambda: scan(cap, attempt=attempt(), busy=(task,)))
    assert again == b_new
    assert windows_asked(seen) == ["batch"]


def test_the_state_is_used_only_under_the_lock_and_no_statement_runs_under_it(
    world, on, clock, monkeypatch
):
    for _ in range(5):
        world.add("c", pinned=True)
    for tenant in ("a", "b"):
        for _ in range(5):
            world.add(tenant)
    lock = RecordingLock()
    watched = {"attempts", "skips", "misses"}

    class Guarded(transport._FairState):
        def __getattribute__(self, name):
            if name in watched:
                assert lock.held_by_me(), f"{name} read without the lock"
            return super().__getattribute__(name)

        def __setattr__(self, name, value):
            if name in watched:
                assert lock.held_by_me(), f"{name} written without the lock"
            super().__setattr__(name, value)

    state = Guarded.__new__(Guarded)
    object.__setattr__(state, "attempts", 0)
    object.__setattr__(state, "skips", {})
    object.__setattr__(state, "misses", 0)
    monkeypatch.setattr(transport, "_fair_state", state)
    monkeypatch.setattr(transport, "_fair_state_lock", lock)
    problems: list[str] = []
    fallbacks: list[bool] = []

    def hook(_conn, _cursor, statement, *_args):
        if lock.held_by_me():
            problems.append(f"a statement ran under the lock: {statement[:60]}")

    sa.event.listen(world.engine, "before_cursor_execute", hook)

    def worker():
        try:
            for _ in range(200):
                one = attempt()
                scan(cap_of(), attempt=one)
                fallbacks.append(one.fallback)
                # Let a skip lapse so that misses keep being recorded.
                clock.now += 2.5
        except BaseException as exc:  # noqa: BLE001
            problems.append(repr(exc))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        sa.event.remove(world.engine, "before_cursor_execute", hook)

    assert problems == []
    assert object.__getattribute__(state, "attempts") == 800
    assert sum(fallbacks) == 100
    assert object.__getattribute__(state, "misses") > 0


def test_a_tenant_whose_commands_all_wait_for_admission_misses_and_is_skipped(
    world, on, clock
):
    tenant_of: dict[int, str] = {}

    def classify(db, command, view):
        key = tenant_of[command.task_id]
        return admission.CommandClassification(
            admission=admission.AdmissionPolicy(f"tenant:{key}", 1, 20),
            fairness_key=key,
        )

    admission.set_command_classifier(classify, admission_enabled=True)

    def stage(tenant):
        with world.sessions() as db:
            task = Task(user_id=world.user, title="t", status=TaskStatus.PENDING)
            db.add(task)
            db.flush()
            tenant_of[task.id] = tenant
            staged = transport.stage_task_command(
                db,
                task_id=task.id,
                actor_user_id=world.user,
                command_id=f"s{task.id}",
                kind=transport.TaskCommandKind.START,
                payload={},
            )
            db.commit()
            return task.id, staged.staged_db_id

    holder_task, holder_command = stage("a")
    with world.sessions() as db:
        lease = ownership.acquire_task_lease_no_commit(
            db, holder_task, runner_id="other"
        )
        assert admission.reserve_task_admission(db, holder_command, lease)
        db.commit()
    for _ in range(3):
        stage("a")  # the one slot of a's bucket is held, so these wait
    first_b = stage("b")[1]
    stage("b")
    cap = cap_of(occupancy={"b": 3})

    chosen, seen = recorded(world, lambda: scan(cap, attempt=attempt()))

    assert chosen == first_b
    assert windows_asked(seen) == ["batch", "batch"]
    assert transport.drain_fair_misses() == 1
    _, within_two_seconds = recorded(world, lambda: scan(cap, attempt=attempt()))
    assert windows_asked(within_two_seconds) == ["batch"]


# --- priority inside one tenant key and class -----------------------------------

BATCH_CLOSED = cap_of(active=[900], policy=POLICY)


def settle(world, command_id):
    with world.sessions() as db:
        db.execute(
            sa.update(TaskExecutionCommand)
            .where(TaskExecutionCommand.id == command_id)
            .values(status="completed")
        )
        db.commit()


# Queued oldest first as (name, class, priority), then the picks, each settled before
# the next. A second tenant holding more slots puts the first one ahead.
B0, B5, I0, TOP = ("b0", "b", 0), ("b5", "b", 5), ("i", "i", 0), ("top", "b", 5)
PRIORITY_CASES = {
    "higher first": ([B0, B5], ["b5", "b0"], None),
    "older interactive first": ([I0, B5], ["i", "b5"], None),
    "interactive between batch": ([B0, I0, B5], ["i", "b5", "b0"], None),
    "interactive that outranks the oldest one": (
        [("i0", "i", 0), B0, ("w10", "i", 10)],
        ["w10", "i0", "b0"],
        None,
    ),
    "newest of twenty": ([("n", "b", 0)] * 19 + [TOP], ["top"], None),
    "newest of forty": ([("n", "b", 0)] * 39 + [TOP], ["top"], None),
    "equal priority by id": ([("x", "b", 5), ("y", "b", 5)], ["x", "y"], None),
    "negative last": ([("low", "b", -3), ("zero", "b", 0)], ["zero", "low"], None),
    "between batch line and limit": ([B0, I0, B5], ["i"], BATCH_CLOSED),
}


@pytest.mark.parametrize("name", PRIORITY_CASES)
def test_a_tenant_class_is_read_by_priority_and_the_classes_by_age(
    world, on, clock, name
):
    spec, expected, cap = PRIORITY_CASES[name]
    ids = {
        label: world.add("a", "interactive" if kind == "i" else "batch", priority=p)
        for label, kind, p in spec
    }
    world.add("b", "interactive")
    cap = cap or cap_of()
    cap = DispatchCapSnapshot(
        policy=cap.policy, active_ids=cap.active_ids, occupancy={"b": 5}
    )

    picks = []
    for _ in expected:
        chosen = scan(cap, attempt=attempt())
        picks.append(next(label for label, number in ids.items() if number == chosen))
        settle(world, chosen)

    assert picks == expected


def test_an_interactive_command_that_waits_elsewhere_does_not_hold_back_batch_work(
    world, on, clock
):
    blocked = world.add("a", "interactive", pinned=True)
    batch = world.add("a", "batch")
    interactive = world.add("a", "interactive")
    world.add("b", "interactive")
    cap = cap_of(occupancy={"b": 5})

    assert blocked < batch < interactive
    assert scan(cap, attempt=attempt()) == batch


def test_one_tenant_alone_is_read_by_age_and_not_by_priority(world, on, clock):
    oldest = world.add("a", priority=0)
    world.add("a", priority=5)

    assert scan(cap_of(), attempt=attempt()) == oldest


def test_a_window_full_of_commands_that_wait_elsewhere_is_missed_until_the_fallback_round(
    world, on, clock
):
    oldest = world.add("a", priority=0)
    first = world.add("a", pinned=True, priority=5)
    for _ in range(15):
        world.follow_up(world.task_of(first), "a", priority=5)
    other = world.add("b")

    cap = cap_of()
    picks = [scan(cap, attempt=attempt()) for _ in range(7)]

    # The sixteen highest-priority commands of a wait for another process, so its
    # window finds nothing and a is missed; the round that reads the queue as it
    # always did takes the oldest command, not the highest priority.
    assert picks == [other] * 7
    assert transport.drain_fair_misses() == 1
    assert scan(cap, attempt=attempt()) == oldest


def test_a_follow_up_of_a_running_task_is_taken_before_a_higher_priority_command(
    world, on, clock
):
    running = world.add("a", kind="message", priority=-10)
    world.add("a", priority=5)
    world.add("b")

    cap = cap_of(active=[world.task_of(running)], occupancy={"a": 1, "b": 5})

    assert scan(cap, attempt=attempt()) == running


@pytest.mark.parametrize("dialect", [sqlite.dialect(), postgresql.dialect()])
def test_the_dispatch_statements_do_not_select_the_priority_columns(world, dialect):
    with world.sessions() as db:
        statements = [
            db.query(TaskExecutionCommand).statement,
            db.query(Task).statement,
            transport._dispatcher_candidate_query(
                db,
                runner_id="me",
                command_db_id=None,
                shared=True,
                cap=cap_of(),
                now=datetime.now(timezone.utc),
            ).statement,
        ]

    for statement in statements:
        select_list = str(statement.compile(dialect=dialect)).split(" FROM ")[0]
        assert (
            "dispatch_class" in select_list or "requested_dispatch_class" in select_list
        )
        assert "priority" not in select_list


# --- random worlds against a brute-force reference --------------------------------


def test_the_dispatcher_follows_the_reference_over_random_worlds(world, on, clock):
    stats: dict[str, int] = {}
    for seed in range(300):
        play(
            world,
            clock,
            seed,
            tenants=random.Random(seed).randint(2, 7),
            reserves=lambda limit: range(limit),
            stats=stats,
        )
    # The worlds hold a tenant back, and holding it back changes some choice.
    assert stats.get("capped", 0) > 0
    assert stats.get("changed", 0) > 0


def test_without_an_interactive_reserve_no_command_is_held_for_its_class(
    world, on, clock
):
    for seed in range(100):
        play(
            world,
            clock,
            10_000 + seed,
            tenants=random.Random(seed).randint(2, 7),
            reserves=lambda limit: [0],
        )


def test_a_queue_of_one_tenant_is_scanned_as_before_plus_one_listing(world, on, clock):
    for seed in range(200):
        rng = random.Random(20_000 + seed)
        spec = random_world(
            rng,
            tenants=1,
            size=rng.randint(4, 30),
            answers=False,
            admission=False,
            heavy=False,
        )
        world.reset()
        store_world(world, spec)
        transport._fair_state = transport._FairState()
        limit = rng.randint(3, 8)
        queued = [c for c in spec.commands if c.status in QUEUED]
        active = {
            c.task
            for c in rng.sample(queued, k=min(len(queued), rng.randint(0, limit - 1)))
        }
        cap = cap_of(
            active=active,
            policy=DispatchPolicy(
                limit=limit, interactive_reserve=rng.randint(0, limit - 1)
            ),
        )
        assert not cap.closed

        off, off_seen = recorded(world, lambda: scan(cap))
        on_choice, on_seen = recorded(world, lambda: scan(cap, attempt=attempt()))

        assert on_choice == off, f"seed {seed}: {spec.commands}"
        assert len(on_seen) == len(off_seen) + 1, f"seed {seed}: {spec.commands}"


def test_a_scan_that_finds_no_open_command_is_not_counted(world, on, clock):
    record = attempt()

    chosen, seen = recorded(world, lambda: scan(cap_of(), attempt=record))

    assert chosen is None
    assert len(seen) == 1
    assert transport._fair_state.attempts == 0
    assert not record.decided


def test_a_held_cap_ranks_no_tenant_and_is_not_counted(world, on, clock):
    world.add("a")
    world.add("b")
    held = DispatchCapSnapshot(
        policy=cap_of().policy, active_ids=frozenset(), held=True
    )
    off, off_seen = recorded(world, lambda: scan(held))

    chosen, seen = recorded(world, lambda: scan(held, attempt=attempt()))

    assert chosen == off
    assert [statement for statement, _ in seen] == [s for s, _ in off_seen]
    assert transport._fair_state.attempts == 0
