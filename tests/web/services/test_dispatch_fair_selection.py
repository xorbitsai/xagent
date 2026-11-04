"""Fair-share dispatch: the switch, the cap snapshot's slot counts and the tenant rotation."""

import asyncio
import logging
import random
import threading
from collections import Counter
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql, sqlite

from tests.web.services.fair_selection_support import (
    CONTROL_KINDS,
    OPENING_KINDS,
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
from xagent.web.services.task_dispatch_cap import (
    DispatchCapSnapshot,
    DispatchPolicy,
    TenantListingRequired,
)
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
    # b supplied the command both times, so no scan took the plain lookup.
    assert transport.drain_fair_bypasses() == (0, 0)


def test_scans_that_every_listed_tenant_misses_are_counted_until_drained(
    world, on, clock
):
    for tenant in ("a", "b"):
        for _ in range(16):
            world.add(tenant, pinned=True)
    oldest_claimable = world.add("a")
    cap = cap_of()

    # Both windows hold only commands leased elsewhere, so both tenants miss and
    # the plain lookup decides. The next scan asks no tenant, both being skipped.
    assert scan(cap, attempt=attempt()) == oldest_claimable
    assert scan(cap, attempt=attempt()) == oldest_claimable

    assert transport.drain_fair_misses() == 2
    assert transport.drain_fair_bypasses() == (0, 2)
    assert transport.drain_fair_bypasses() == (0, 0)


def test_with_the_switch_off_no_scan_is_counted(world, clock, monkeypatch):
    for tenant in ("a", "b"):
        for _ in range(16):
            world.add(tenant, pinned=True)
    world.add("a")
    monkeypatch.setenv(SWITCH, "false")
    monkeypatch.setattr(transport, "_fair_share_memo", None)
    transport._fair_state.since_fallback = EVERY - 1

    scan(cap_of(), attempt=transport._new_fair_attempt())

    assert transport.drain_fair_bypasses() == (0, 0)


def test_a_queue_of_one_tenant_is_not_counted_as_all_missed(world, on, clock):
    for _ in range(16):
        world.add("a", pinned=True)
    oldest_claimable = world.add("a")

    assert scan(cap_of(), attempt=attempt()) == oldest_claimable
    assert transport.drain_fair_bypasses() == (0, 0)


@pytest.mark.parametrize("line", ["balance", "attempt count"])
def test_each_attempt_decided_as_a_fallback_round_is_counted_once(
    line, world, on, clock
):
    world.add("a")
    world.add("b")
    cap = cap_of()
    if line == "balance":
        transport._fair_state.since_fallback = EVERY - 1
    else:
        transport._fair_state.attempts_since_fallback = AFTER

    one = attempt()
    scan(cap, attempt=one)
    scan(cap, attempt=one)
    scan(cap, attempt=attempt(), command_db_id=1)

    assert one.fallback
    assert transport.drain_fair_bypasses() == (1, 0)
    transport._fair_state.since_fallback = 0
    transport._fair_state.attempts_since_fallback = 0
    scan(cap, attempt=attempt())
    assert transport.drain_fair_bypasses() == (0, 0)


EVERY = transport.FAIR_SHARE_FALLBACK_EVERY
AFTER = transport.FAIR_SHARE_FALLBACK_AFTER_ATTEMPTS


@pytest.mark.parametrize("line", ["balance", "attempt count"])
def test_a_scan_only_reads_the_balance_and_only_adds_to_the_attempt_count(
    line, world, on, clock
):
    for _ in range(16):
        world.add("a", pinned=True)
    oldest_claimable = world.add("a")
    first_b = world.add("b")
    for _ in range(19):
        world.add("b")
    cap = cap_of(occupancy={"b": 5})
    state = transport._fair_state
    if line == "balance":
        state.since_fallback = EVERY - 2
    else:
        state.attempts_since_fallback = AFTER - 20

    # A scan that finds a command leaves the balance alone and adds one to the
    # attempt count: an attempt is a fair round until the balance has reached its
    # line or the attempt count has reached its own.
    rounds = []
    for step in range(40):
        if line == "balance" and step == 20:
            state.since_fallback = EVERY - 1
        one = attempt()
        rounds.append((scan(cap, attempt=one), one.counted, one.fallback))
    assert (
        rounds == [(first_b, True, False)] * 20 + [(oldest_claimable, True, True)] * 20
    )
    if line == "balance":
        assert (state.since_fallback, state.attempts_since_fallback) == (EVERY - 1, 40)
    else:
        assert (state.since_fallback, state.attempts_since_fallback) == (0, AFTER + 20)


@pytest.mark.parametrize("together", [2, 3])
def test_attempts_that_decided_a_fallback_together_each_take_their_debt_off(
    together, world, on, clock
):
    world.add("a")
    world.add("b")
    cap = cap_of()
    transport._fair_state.since_fallback = EVERY - 1
    attempts = [attempt() for _ in range(together)]
    for one in attempts:
        scan(cap, attempt=one)
    assert all(one.fallback for one in attempts)

    for one in attempts:
        transport._fair_count_dispatch(one, True)

    # The debt is not cut at one period, so the slots these rounds opened together
    # are paid back.
    assert transport._fair_state.since_fallback == -(EVERY - 1) * (together - 1)
    later = attempt()
    scan(cap, attempt=later)
    assert not later.fallback


def test_an_attempt_that_rescans_keeps_its_decision_and_lists_the_queue_once(
    world, on, clock
):
    world.add("a")
    world.add("b")
    one = attempt()
    cap = cap_of()

    _, seen = recorded(world, lambda: (scan(cap, attempt=one), scan(cap, attempt=one)))

    assert one.counted and transport._fair_state.since_fallback == 0
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


@pytest.mark.parametrize(
    "b_joins, fair_rounds",
    [(False, EVERY - 1), (True, AFTER)],
    ids=["b opens slots", "b only joins running tasks"],
)
def test_a_command_of_a_task_leased_elsewhere_is_not_found_by_the_window(
    b_joins, fair_rounds, world, on, clock
):
    pinned = [world.add("a", pinned=True) for _ in range(16)]
    seventeenth = world.add("a")
    if b_joins:
        running = world.task_of(world.add("b", status="completed"))
        first_b = world.follow_up(running, "b")
        for _ in range(19):
            world.follow_up(running, "b")
        cap = cap_of(active=[running], occupancy={"b": 5})
    else:
        first_b = world.add("b")
        for _ in range(19):
            world.add("b")
        cap = cap_of(occupancy={"b": 5})

    # a holds fewer slots, but its window is full of commands that wait for another
    # process, so b is served; only the round that reads everything reaches a. When
    # b's command joins a task that already runs here it opens no slot, and only the
    # attempt count brings that round.
    for _ in range(fair_rounds):
        clock.now += transport.FAIR_SHARE_SKIP_SECONDS
        one = attempt()
        assert scan(cap, attempt=one) == first_b
        transport._fair_count_dispatch(one, not b_joins)
    fallback = attempt()
    assert scan(cap, attempt=fallback) == seventeenth
    transport._fair_count_dispatch(fallback, True)
    assert scan(cap, attempt=attempt()) == first_b

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

    one = attempt()

    chosen, seen = recorded(world, lambda: scan(cap, attempt=one))

    assert chosen == off == reply
    assert [statement for statement, _ in seen] == [s for s, _ in off_seen]
    assert one.counted


def test_a_scan_for_one_command_sends_what_it_always_sent_and_is_not_counted(
    world, on, clock
):
    target = world.add("a")
    world.add("b")
    cap = cap_of()
    off, off_seen = recorded(world, lambda: scan(cap, command_db_id=target))

    one = attempt()

    chosen, seen = recorded(world, lambda: scan(cap, attempt=one, command_db_id=target))

    assert chosen == off == target
    assert [statement for statement, _ in seen] == [s for s, _ in off_seen]
    assert not one.counted


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

    # The pause makes no tenant of its own, so a is ranked alone. Its window holds
    # the sixteen commands that wait elsewhere and misses, without a skip, and the
    # plain lookup takes the seventeenth, the row it takes with the switch off.
    assert chosen == off == seventeenth
    assert transport.drain_fair_misses() == 1
    assert transport._fair_state.skips == {}


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
    assert transport._fair_state.since_fallback == 0
    # The same world with the switch on is served differently, so the checks above
    # are not a scan that would have found the oldest command anyway.
    if between_the_lines:
        return
    monkeypatch.setenv(SWITCH, "true")
    monkeypatch.setattr(transport, "_fair_share_memo", None)
    assert scan(cap, attempt=attempt()) == ids[1]


@pytest.fixture
def loop(world, on, monkeypatch):
    """The dispatch loop over a real slot manager and a coordinator that is told what to do."""
    manager = BackgroundTaskManager()
    monkeypatch.setattr(task_execution, "background_task_manager", manager)
    monkeypatch.setattr(transport, "build_dispatch_cap_snapshot", lambda: cap_of())
    plan = SimpleNamespace(busy=0, ensure_raises=False, ensured=[])

    async def ensure(task_id):
        plan.ensured.append(task_id)
        if plan.ensure_raises:
            raise RuntimeError("no coordinator")
        if plan.busy:
            plan.busy -= 1
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

    async def run(**kwargs):
        state = transport._fair_state
        before = state.since_fallback
        await transport.dispatch_one_task_command(executor, **kwargs)
        return state.since_fallback - before, state.attempts_since_fallback

    return SimpleNamespace(manager=manager, plan=plan, run=run, world=world)


# The steps a row of LOOP_ROWS takes; each gets the loop and monkeypatch, and one
# that returns a dict gives the arguments of the attempt.
def add(*args, **kwargs):
    return lambda loop, mp: setattr(loop, "last", loop.world.add(*args, **kwargs))


def hold_a_slot_for_the_last_task(loop, mp):
    """Another attempt's token, and no execution."""
    token = loop.manager.reserve_dispatch_slot(
        loop.world.task_of(loop.last), policy=LARGE, interactive=False
    )
    assert token is not None


def run_the_last_task(loop, mp):
    execution = asyncio.get_running_loop().create_future()
    loop.manager._track_live_execution(loop.world.task_of(loop.last), execution)


def reach_the_attempt_line(loop, mp):
    transport._fair_state.attempts_since_fallback = AFTER


def ask_for_the_last_by_id(loop, mp):
    return {"command_db_id": loop.last}


def ask_for_a_listing(times):
    def step(loop, mp):
        real = loop.manager.reserve_dispatch_slot
        asked = []

        def reserve(*args, **kwargs):
            asked.append(1)
            if len(asked) <= times:
                raise TenantListingRequired
            return real(*args, **kwargs)

        mp.setattr(loop.manager, "reserve_dispatch_slot", reserve)

    return step


def refuse_slots(loop, mp):
    mp.setattr(loop.manager, "reserve_dispatch_slot", lambda *a, **k: None)


def keep_the_coordinator_busy_once(loop, mp):
    loop.plan.busy = 1


def switch_off(loop, mp):
    mp.setenv(SWITCH, "false")
    mp.setattr(transport, "_fair_share_memo", None)


def sit_between_the_lines(loop, mp):
    """One batch slot of two taken, so only interactive commands can be claimed."""
    band = cap_of(active=[900], policy=POLICY)
    assert band.in_reserve_band
    mp.setattr(transport, "build_dispatch_cap_snapshot", lambda: band)


# The attempt count every row starts from, unless a step sets it.
COUNT = 5
# row: (steps, the balance before, the change in the balance, the attempt count after)
LOOP_ROWS = {
    "a fair round that opens a slot": ([add("a")], 0, 1, COUNT + 1),
    "a fallback round that opens a slot": ([add("a")], EVERY - 1, -(EVERY - 1), 0),
    "a fallback round the attempt count forced": (
        [add("a"), reach_the_attempt_line],
        0,
        0,
        0,
    ),
    "a fallback round both lines brought": (
        [add("a"), reach_the_attempt_line],
        EVERY - 1,
        -(EVERY - 1),
        0,
    ),
    "a slot opened after one more listing": (
        [add("a"), ask_for_a_listing(1)],
        0,
        1,
        COUNT + 1,
    ),
    "a second missing listing": ([add("a"), ask_for_a_listing(2)], 0, 0, COUNT + 1),
    "a slot opened after a busy coordinator and a rescan": (
        [add("a"), add("b"), keep_the_coordinator_busy_once],
        0,
        1,
        COUNT + 1,
    ),
    "a refused slot": ([add("a"), refuse_slots], 0, 0, COUNT + 1),
    "a forced fallback round whose slot is refused": (
        [add("a"), reach_the_attempt_line, refuse_slots],
        0,
        0,
        AFTER + 1,
    ),
    "nothing queued": ([], 0, 0, COUNT),
    "nothing claimable: a task leased elsewhere": (
        [add("a", pinned=True), add("b", pinned=True)],
        0,
        0,
        COUNT,
    ),
    "nothing claimable: a retry not yet due": (
        [add("a", unavailable=True), add("b", unavailable=True)],
        0,
        0,
        COUNT,
    ),
    "nothing claimable: batch commands between the lines": (
        [add("a"), add("b"), sit_between_the_lines],
        0,
        0,
        COUNT,
    ),
    "a control command": ([add("a", kind="pause")], 0, 0, COUNT + 1),
    "a fallback round that takes a control command": (
        [add("a", kind="pause")],
        EVERY - 1,
        0,
        COUNT + 1,
    ),
    "a command asked for by id": ([add("a"), ask_for_the_last_by_id], 0, 0, COUNT),
    "a follow-up of a running task": (
        [add("a", kind="message"), run_the_last_task],
        0,
        0,
        COUNT + 1,
    ),
    "a fallback round that picks a running task": (
        [add("a", kind="message"), run_the_last_task],
        EVERY - 1,
        0,
        0,
    ),
    "a task that holds only another attempt's token": (
        [add("a"), hold_a_slot_for_the_last_task],
        0,
        1,
        COUNT + 1,
    ),
    "the switch off": ([add("a"), switch_off], 0, 0, COUNT),
}


@pytest.mark.parametrize("row", list(LOOP_ROWS))
async def test_what_each_dispatch_attempt_does_to_the_balance_and_the_attempt_count(
    row, loop, monkeypatch
):
    steps, before, change, count = LOOP_ROWS[row]
    transport._fair_state.attempts_since_fallback = COUNT
    kwargs: dict = {}
    for step in steps:
        kwargs.update(step(loop, monkeypatch) or {})
    transport._fair_state.since_fallback = before

    assert await loop.run(**kwargs) == (change, count)


async def test_a_rescan_after_a_busy_coordinator_lists_the_queue_once_and_takes_another_task(
    loop, world
):
    world.add("a")
    world.add("b")
    loop.plan.busy = 1
    seen, stop = statements_of(world.engine)
    try:
        await loop.run()
    finally:
        stop()

    assert sum("GROUP BY" in statement for statement in seen) == 1
    assert len(loop.plan.ensured) == 2 and loop.plan.ensured[0] != loop.plan.ensured[1]
    assert not loop.manager._dispatch_reservations


async def test_a_failing_ownership_check_leaves_the_balance_alone(loop):
    loop.world.add("a")
    loop.plan.ensure_raises = True

    with pytest.raises(RuntimeError):
        await loop.run()

    state = transport._fair_state
    assert (state.since_fallback, state.attempts_since_fallback) == (0, 1)
    assert not loop.manager._dispatch_reservations


def test_a_fallback_round_that_finds_nothing_leaves_the_attempt_count_on_its_line(
    world, on, clock
):
    world.add("a", pinned=True)
    world.add("b", pinned=True)
    transport._fair_state.attempts_since_fallback = AFTER
    one = attempt()

    assert scan(cap_of(), attempt=one) is None

    assert one.fallback
    assert transport._fair_state.attempts_since_fallback == AFTER


async def test_polling_on_the_batch_line_does_not_bring_a_fallback_round(
    loop, monkeypatch
):
    """A holds both batch slots for longer than the attempt line takes to poll by.
    Once one frees, A holds one and B none, so B's younger command goes first."""
    world = loop.world
    for _ in range(3):
        world.add("a")
    b = world.add("b")
    policy = DispatchPolicy(limit=3, interactive_reserve=1)
    band = cap_of(active=[900, 901], occupancy={"a": 2}, policy=policy)
    freed = cap_of(active=[900], occupancy={"a": 1}, policy=policy)
    assert band.in_reserve_band and not freed.in_reserve_band

    monkeypatch.setattr(transport, "build_dispatch_cap_snapshot", lambda: band)
    for _ in range(AFTER + 10):
        assert await loop.run() == (0, 0)
    monkeypatch.setattr(transport, "build_dispatch_cap_snapshot", lambda: freed)

    assert await loop.run() == (1, 1)
    assert loop.plan.ensured == [world.task_of(b)]


def test_a_fair_round_finds_nothing_exactly_when_a_fallback_round_would(
    world, on, clock
):
    """Attempts that find nothing are left out of the attempt count. That passes no
    command over only while a fair round and a fallback round agree on it."""
    found_nothing = 0
    for seed in range(80):
        rng = random.Random(30_000 + seed)
        spec = random_world(rng, tenants=rng.randint(2, 6), size=rng.randint(4, 30))
        if seed % 4 == 0:
            # Every command open and none claimable here.
            for command in spec.commands:
                command.pinned = True
        world.reset()
        store_world(world, spec)
        limit = rng.randint(2, 6)
        policy = DispatchPolicy(limit=limit, interactive_reserve=rng.randrange(limit))
        queued = [c for c in spec.commands if c.status in QUEUED]
        for step in range(5):
            holds = rng.sample(queued, k=rng.randint(0, min(limit - 1, len(queued))))
            cap = DispatchCapSnapshot(
                policy=policy,
                active_ids=frozenset(c.task for c in holds),
                occupancy=dict(Counter(c.tenant for c in holds)),
            )
            busy = tuple(c.task for c in spec.commands if rng.random() < 0.05)
            forced = transport._FairAttempt(decided=True, counted=True, fallback=True)
            oldest = scan(cap, attempt=forced, busy=busy)
            transport._fair_state.attempts_since_fallback = 0
            fair = attempt()
            chosen = scan(cap, attempt=fair, busy=busy)
            assert not fair.fallback
            assert (chosen is None) == (oldest is None), f"seed {seed} step {step}"
            found_nothing += chosen is None
            clock.now += rng.uniform(0, 3)
    assert found_nothing >= 40


@pytest.mark.parametrize(
    "b_offers", ["pauses", "starts whose slot is refused", "pauses, a refused once"]
)
async def test_a_start_behind_a_missed_window_gets_the_fallback_round_after_the_attempt_line(
    b_offers, loop, clock, monkeypatch
):
    """A's 16 oldest commands are leased elsewhere and its START is behind them. B is
    listed through an opening command leased elsewhere and offers every attempt a new
    command that opens no slot, so no attempt moves the balance."""
    world = loop.world
    for _ in range(16):
        world.add("a", pinned=True)
    a_task = world.task_of(world.add("a"))
    world.add("b", pinned=True)
    kind, b_dispatched = "pause", "b"
    refused = {"b": 0, "a": 0}
    if b_offers == "starts whose slot is refused":
        kind, b_dispatched = "start", None
        refused["b"] = AFTER + 2
    elif b_offers == "pauses, a refused once":
        refused["a"] = 1
    real = loop.manager.reserve_dispatch_slot

    def reserve(task_id, **kwargs):
        key = kwargs["fairness_key"]
        if refused.get(key, 0) > 0:
            refused[key] -= 1
            return None
        return real(task_id, **kwargs)

    monkeypatch.setattr(loop.manager, "reserve_dispatch_slot", reserve)
    # A fallback round whose slot is refused does not reset the count, so the next
    # attempt is a fallback round too.
    expected = [b_dispatched] * AFTER + (
        [None, "a"] if b_offers == "pauses, a refused once" else ["a"]
    )

    dispatched = []
    for _ in range(len(expected)):
        clock.now += transport.FAIR_SHARE_SKIP_SECONDS
        world.add("b", kind=kind)
        before = len(loop.plan.ensured)
        await transport.dispatch_one_task_command(None)
        tasks = loop.plan.ensured[before:]
        dispatched.append("a" if a_task in tasks else "b" if tasks else None)
        with world.sessions() as db:
            db.execute(
                sa.update(TaskExecutionCommand)
                .where(TaskExecutionCommand.task_id.in_(tasks))
                .values(status="completed")
            )
            db.commit()

    # Every attempt misses A's window and is counted; the attempt after AFTER of
    # them is a fallback round, which reads past the window to A's START. The
    # attempt count brought it, so it leaves the balance alone.
    assert dispatched == expected
    state = transport._fair_state
    assert (state.since_fallback, state.attempts_since_fallback) == (0, 0)


class Wakeups:
    """Dispatch attempts woken together over a slot manager that frees ``free`` slots.

    Every attempt picks before any of them takes a slot, as when one wake-up
    starts them all at once, and the attempts whose pick belongs to tenant a go
    on first, or last when ``fallback_first`` is false. A task already holding a
    token gets a token that takes no slot; a task already executing has a busy
    coordinator, so a later attempt that picked it sets it aside and scans again.
    """

    ATTEMPTS = transport.DISPATCHER_CONCURRENCY

    def __init__(self, world, monkeypatch, fallback_first=True) -> None:
        self.world = world
        self.a_tasks: set[int] = set()
        self.free = 0
        self.held: set[int] = set()
        self.executing: list[int] = []
        self.opened: list[str] = []
        self.waiting: list[tuple[int, asyncio.Future]] = []
        self.picked: set[asyncio.Task] = set()
        self.fallback_first = fallback_first
        monkeypatch.setattr(task_execution, "background_task_manager", self)
        monkeypatch.setattr(
            transport, "build_dispatch_cap_snapshot", lambda: cap_of(occupancy={"a": 5})
        )
        monkeypatch.setattr(transport, "run_db_io_cancellation_safe", self.pick)
        registry = SimpleNamespace(busy_command_tasks=lambda: set(), ensure=self.ensure)
        monkeypatch.setattr(runtime, "get_task_coordinator_registry", lambda: registry)

    def queue(self, tenant, count):
        for _ in range(count):
            command_id = self.world.add(tenant)
            if tenant == "a":
                self.a_tasks.add(self.world.task_of(command_id))

    # What the dispatch loop asks of the slot manager and the coordinator.
    def reserve_dispatch_slot(self, task_id, **_kwargs):
        if task_id in self.held:
            return SimpleNamespace(task_id=task_id, took=False)
        if self.free <= 0:
            return None
        self.free -= 1
        self.held.add(task_id)
        return SimpleNamespace(task_id=task_id, took=True)

    def release_dispatch_slot(self, token):
        if token.took and token.task_id not in self.executing:
            self.free += 1
            self.held.discard(token.task_id)

    def holds_execution(self, task_id):
        return task_id in self.executing

    async def ensure(self, task_id):
        if task_id in self.executing:
            return SimpleNamespace(_command_tasks={object()})

        async def execute_command(command, apply):
            # Registered before the first await, as the real coordinator does.
            self.executing.append(task_id)
            return True

        return SimpleNamespace(_command_tasks=set(), execute_command=execute_command)

    async def pick(self, operation):
        found = operation()
        task = asyncio.current_task()
        if task in self.picked:
            return found
        self.picked.add(task)
        gate = asyncio.get_running_loop().create_future()
        self.waiting.append((found, gate))
        if len(self.waiting) == self.ATTEMPTS:
            ordered = sorted(
                self.waiting,
                key=lambda entry: (
                    (entry[0] is None or entry[0].task_id not in self.a_tasks)
                    == self.fallback_first
                ),
            )
            self.waiting = []

            def release():
                for _, waiting_gate in ordered:
                    waiting_gate.set_result(None)

            # Released from a callback, so that the attempt that completes the set
            # waits too and every attempt goes on in the order above.
            asyncio.get_running_loop().call_soon(release)
        await gate
        return found

    async def wake(self, free):
        self.free = free
        await asyncio.gather(
            *(transport.dispatch_one_task_command(None) for _ in range(self.ATTEMPTS))
        )
        self.opened += ["a" if task in self.a_tasks else "b" for task in self.executing]
        with self.world.sessions() as db:
            db.execute(
                sa.update(TaskExecutionCommand)
                .where(TaskExecutionCommand.task_id.in_(self.executing))
                .values(status="completed")
            )
            db.commit()
        self.held.clear()
        self.executing.clear()
        self.picked.clear()


@pytest.mark.parametrize(
    "fallback_first", [True, False], ids=["fallback first", "fair first"]
)
@pytest.mark.parametrize("freed", [1, 2])
async def test_fallback_rounds_open_one_new_slot_in_every_fallback_period_however_many_a_wakeup_frees(
    freed, fallback_first, world, on, clock, monkeypatch
):
    """Tenant a holds more slots and queued first, so the fair rounds serve b and
    only a fallback round opens a slot for a. The attempts of one wake-up share one
    decision, so who goes on first does not matter: with the fallback round first,
    counting every attempt gets the share wrong, and with the fair rounds first,
    reading the attempt count modulo the period does."""
    wakeups = Wakeups(world, monkeypatch, fallback_first)
    wakeups.queue("a", 40)
    wakeups.queue("b", 80)
    slots = 64

    for _ in range(slots // freed):
        await wakeups.wake(freed)

    assert len(wakeups.opened) == slots
    # With the fallback round first, counting every attempt gives 32 for freed = 1
    # and clearing the balance on a fallback round instead of taking its share off
    # gives 12 for freed = 2; with the fair rounds first, a fallback round on every
    # eighth attempt gives 0.
    assert wakeups.opened.count("a") == slots // EVERY


@pytest.mark.parametrize(
    "fallback_first", [True, False], ids=["fallback first", "fair first"]
)
async def test_after_a_stretch_of_fair_rounds_that_open_no_slot_the_share_is_back_at_once(
    fallback_first, world, on, clock, monkeypatch
):
    """For twenty attempt periods the fair rounds dispatch commands that open no
    slot, as when they join running tasks, and only the fallback rounds the attempt
    count brings open one. Then every wake-up frees one slot, as in the test above:
    the fallback rounds open one in every FAIR_SHARE_FALLBACK_EVERY new slots from
    the first period on, instead of after the fair rounds have paid a debt back."""
    state = transport._fair_state
    for _ in range(20):
        one = attempt()
        while not one.fallback:
            one = attempt()
            transport._ranks_tenants(one, None, None)
            transport._fair_count_attempt(one)
            transport._fair_count_dispatch(one, one.fallback)
    assert (state.since_fallback, state.attempts_since_fallback) == (0, 0)

    wakeups = Wakeups(world, monkeypatch, fallback_first)
    wakeups.queue("a", 40)
    wakeups.queue("b", 160)
    slots = 128
    for _ in range(slots):
        await wakeups.wake(1)

    assert len(wakeups.opened) == slots
    # Taking a period's share off for each forced round leaves a debt of 140 here,
    # and only the attempt count brings fallback rounds until it is paid: 7.
    # Keeping the balance at or above one period's debt gives 15.
    assert wakeups.opened.count("a") == slots // EVERY


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
    watched = {
        "since_fallback",
        "attempts_since_fallback",
        "skips",
        "misses",
        "fallbacks",
        "all_missed",
    }

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
    object.__setattr__(state, "since_fallback", 0)
    object.__setattr__(state, "attempts_since_fallback", 0)
    object.__setattr__(state, "skips", {})
    object.__setattr__(state, "misses", 0)
    object.__setattr__(state, "fallbacks", 0)
    object.__setattr__(state, "all_missed", 0)
    monkeypatch.setattr(transport, "_fair_state", state)
    monkeypatch.setattr(transport, "_fair_state_lock", lock)
    problems: list[str] = []
    fallbacks: list[bool] = []
    THREADS, ROUNDS = 4, 200

    def hook(_conn, _cursor, statement, *_args):
        if lock.held_by_me():
            problems.append(f"a statement ran under the lock: {statement[:60]}")

    sa.event.listen(world.engine, "before_cursor_execute", hook)

    def worker():
        try:
            for _ in range(ROUNDS):
                one = attempt()
                scan(cap_of(), attempt=one)
                transport._fair_count_dispatch(one, True)
                fallbacks.append(one.fallback)
                # Let a skip lapse so that misses keep being recorded.
                clock.now += 2.5
        except BaseException as exc:  # noqa: BLE001
            problems.append(repr(exc))

    threads = [threading.Thread(target=worker) for _ in range(THREADS)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        sa.event.remove(world.engine, "before_cursor_execute", hook)

    assert problems == []
    # Every slot counted once: one for a fair round, EVERY - 1 off for a fallback.
    assert sum(fallbacks) > 0
    balance = object.__getattribute__(state, "since_fallback")
    assert balance == THREADS * ROUNDS - EVERY * sum(fallbacks)
    # The balance stays within what the attempts in flight can move it.
    assert (EVERY - 1) * (1 - THREADS) <= balance <= EVERY - 2 + THREADS
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


def test_one_tenant_alone_is_read_by_priority(world, on, clock):
    world.add("a", priority=0)
    higher = world.add("a", priority=5)

    assert scan(cap_of(), attempt=attempt()) == higher


def test_the_only_queued_tenant_takes_its_highest_priority_interactive_work_first(
    world, on, clock
):
    ids = {
        name: world.add("a", "interactive", priority=p)
        for name, p in (("low", -5), ("five", 5), ("five again", 5), ("top", 10))
    }

    picks = []
    for _ in ids:
        chosen = scan(cap_of(), attempt=attempt())
        picks.append(next(name for name, number in ids.items() if number == chosen))
        settle(world, chosen)

    assert picks == ["top", "five", "five again", "low"]


def test_between_the_lines_a_tenant_with_only_batch_work_does_not_turn_priority_off(
    world, on, clock
):
    world.add("a", "interactive", priority=0)
    higher = world.add("a", "interactive", priority=9)
    world.add("b", "batch")

    # Only a has interactive work, so a is the one tenant ranked.
    assert BATCH_CLOSED.batch_closed
    assert scan(BATCH_CLOSED, attempt=attempt()) == higher


def test_the_only_ranked_tenant_whose_window_misses_takes_the_oldest_row_unskipped(
    world, on, clock
):
    oldest = world.add("a", priority=0)
    first = world.add("a", pinned=True, priority=5)
    for _ in range(15):
        world.follow_up(world.task_of(first), "a", priority=5)

    # No other tenant could take the turn, so the plain lookup decides in the same
    # scan and the pair is not skipped: the next scan reads the window again.
    assert scan(cap_of(), attempt=attempt()) == oldest
    assert transport.drain_fair_misses() == 1
    assert transport._fair_state.skips == {}


def test_a_class_of_the_only_ranked_tenant_that_missed_is_asked_again_next_scan(
    world, on, clock
):
    world.add("a", "interactive", pinned=True)
    assert scan(cap_of(), attempt=attempt()) is None

    widget = world.add("a", "interactive", priority=10)
    world.add("a", "batch")

    # Had the interactive class been skipped, the newer batch command would win.
    assert scan(cap_of(), attempt=attempt()) == widget


def test_a_miss_while_ranked_alone_keeps_the_tenant_counted_for_the_cap(
    world, on, clock
):
    held = world.add("a", "interactive")
    band = cap_of(
        active=[901, 902],
        occupancy={"a": 2},
        policy=DispatchPolicy(limit=3, interactive_reserve=1),
    )
    assert band.in_reserve_band

    # a alone, its window missing because the attempt set its task aside.
    assert scan(band, attempt=attempt(), busy=(world.task_of(held),)) is None
    world.add("b", "interactive", pinned=True)

    # a holds the most slots and b is queued, so a may not take the reserve slot.
    assert scan(band, attempt=attempt()) is None


def test_a_skip_left_from_two_ranked_tenants_is_not_read_once_one_is_left(
    world, on, clock
):
    older = world.add("a", "interactive", priority=0)
    other = world.add("b", "batch")
    cap = cap_of(occupancy={"b": 1})

    # a ranks first; its window misses because the attempt set its task aside, and
    # with b ranked too the pair is skipped.
    assert scan(cap, attempt=attempt(), busy=(world.task_of(older),)) == other
    assert ("a", "interactive") in transport._fair_state.skips
    settle(world, other)
    widget = world.add("a", "interactive", priority=10)

    assert scan(cap, attempt=attempt()) == widget


def test_a_window_full_of_commands_that_wait_elsewhere_is_missed_until_the_fallback_round(
    world, on, clock
):
    oldest = world.add("a", priority=0)
    first = world.add("a", pinned=True, priority=5)
    for _ in range(15):
        world.follow_up(world.task_of(first), "a", priority=5)
    other = world.add("b")

    cap = cap_of()
    fair_rounds = transport.FAIR_SHARE_FALLBACK_EVERY - 1
    picks = []
    for _ in range(fair_rounds):
        one = attempt()
        picks.append(scan(cap, attempt=one))
        transport._fair_count_dispatch(one, True)

    # The sixteen highest-priority commands of a wait for another process, so its
    # window finds nothing and a is missed; once the fair rounds have opened their
    # slots, the round that reads the queue as it always did takes the oldest
    # command, not the highest priority.
    assert picks == [other] * fair_rounds
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
    # The order of the commands of one task keeps commands back, and a control
    # passes an earlier opening command at the batch line or at the limit.
    assert stats.get("held_back", 0) > 0
    assert stats.get("passed", 0) > 0


def test_the_reference_kinds_are_the_dispatchers():
    assert OPENING_KINDS == task_dispatch_cap.EXECUTION_OPENING_KINDS
    assert CONTROL_KINDS == task_dispatch_cap.CAP_OVERTAKING_CONTROL_KINDS


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


def test_a_queue_of_one_tenant_follows_the_reference(world, on, clock):
    for seed in range(200):
        play(
            world,
            clock,
            20_000 + seed,
            tenants=1,
            reserves=lambda limit: range(limit),
        )


def test_a_shallow_queue_of_one_tenant_at_one_priority_is_scanned_as_before(
    world, on, clock
):
    for seed in range(200):
        rng = random.Random(40_000 + seed)
        spec = random_world(
            rng,
            tenants=1,
            size=rng.randint(4, 16),
            answers=False,
            admission=False,
            heavy=False,
        )
        for cmd in spec.commands:
            cmd.priority = 0
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

        off = scan(cap)
        on_choice = scan(cap, attempt=attempt())

        assert on_choice == off, f"seed {seed}: {spec.commands}"


def test_a_scan_that_finds_no_open_command_is_not_counted(world, on, clock):
    record = attempt()

    chosen, seen = recorded(world, lambda: scan(cap_of(), attempt=record))

    assert chosen is None
    assert len(seen) == 1
    assert not record.counted
    assert not record.decided


def test_a_held_cap_ranks_no_tenant_and_is_not_counted(world, on, clock):
    world.add("a")
    world.add("b")
    held = DispatchCapSnapshot(
        policy=cap_of().policy, active_ids=frozenset(), held=True
    )
    off, off_seen = recorded(world, lambda: scan(held))

    one = attempt()

    chosen, seen = recorded(world, lambda: scan(held, attempt=one))

    assert chosen == off
    assert [statement for statement, _ in seen] == [s for s, _ in off_seen]
    assert not one.counted
