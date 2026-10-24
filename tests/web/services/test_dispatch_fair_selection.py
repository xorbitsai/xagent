"""Fair-share dispatch: the switch, the cap snapshot's slot counts and the tenant rotation."""

import logging
import random
import threading
from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from tests.web.services.fair_selection_support import (
    CAP_STATES,
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
    seed_world,
    store_world,
)
from xagent import config
from xagent.web.models import database as database_module
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_coordinator_service as ownership
from xagent.web.services import task_execution
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
    # finds what today's scan finds, and ranks nobody: the lock is taken once, to
    # count the attempt.
    assert chosen == off
    assert lock.taken == 1
    assert batch_a is not None


@pytest.mark.parametrize(
    "raw", [None, "false", "garbage"], ids=["unset", "false", "garbage"]
)
@pytest.mark.parametrize("cap_state", sorted(CAP_STATES))
@pytest.mark.parametrize("name", sorted(WORLDS))
def test_with_the_switch_off_a_scan_sends_what_it_sent_before(
    world, monkeypatch, clock, name, cap_state, raw
):
    if raw is None:
        monkeypatch.delenv(SWITCH, raising=False)
    else:
        monkeypatch.setenv(SWITCH, raw)
    monkeypatch.setattr(transport, "_fair_share_memo", None)
    first = seed_world(world, name)
    cap = DispatchCapSnapshot(
        policy=POLICY, active_ids=CAP_STATES[cap_state](first), occupancy={}
    )
    off = transport._new_fair_attempt()
    assert off is None

    chosen, seen = recorded(world, lambda: scan(cap, attempt=off))
    plain, plain_seen = recorded(world, lambda: scan(cap))

    assert chosen == plain
    assert [s for s, _ in seen] == [s for s, _ in plain_seen]
    assert not any("RECURSIVE" in s or "GROUP BY" in s for s, _ in seen)
    assert windows_asked(seen) == []
    assert transport._fair_state.attempts == 0


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


# --- random worlds against a brute-force reference --------------------------------


def test_the_dispatcher_follows_the_reference_over_random_worlds(world, on, clock):
    for seed in range(300):
        play(
            world,
            clock,
            seed,
            tenants=random.Random(seed).randint(2, 7),
            reserves=lambda limit: range(limit),
        )


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
