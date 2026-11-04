"""The tenant cap of fair-share dispatch: which tenant is held back, and where."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tests.web.services import test_dispatch_fair_selection as fair
from tests.web.services.fair_selection_support import attempt, cap_of, scan
from xagent import config
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_coordinator_runtime as runtime
from xagent.web.services import task_execution
from xagent.web.services.task_dispatch_cap import (
    DispatchCapSnapshot,
    DispatchPolicy,
    TenantListingRequired,
)
from xagent.web.services.task_execution import BackgroundTaskManager

KINDS = transport.TaskCommandKind
BAND = DispatchPolicy(limit=4, interactive_reserve=2)

clock, on, world = fair.clock, fair.on, fair.world
recorded, statements_of, windows_asked = (
    fair.recorded,
    fair.statements_of,
    fair.windows_asked,
)


def band_cap(occupancy, *, active=2, held=False, policy=BAND):
    return DispatchCapSnapshot(
        policy=policy,
        active_ids=frozenset(range(900, 900 + active)),
        held=held,
        occupancy=occupancy,
    )


@pytest.mark.parametrize(
    ("occupancy", "queued", "expected"),
    [
        ({"x": 2, "y": 2}, {}, None),
        ({"a": 2, "b": 1}, {"a": 1}, None),
        ({}, {"a": 1, "b": 2}, None),
        ({"x": 3, "y": 1}, {"y": 5, "z": 6}, None),
        ({"x": 2, "y": 2}, {"x": 7, "y": 3}, "y"),
        ({"x": 2, "y": 2}, {"x": 3, "y": 7}, "x"),
        ({"x": 2, "y": 2}, {"x": 9, "z": 1}, "x"),
        ({"a": 2, "b": 1}, {"a": 9, "b": 1}, "a"),
        ({"": 3, "b": 1}, {"": 4, "b": 5}, ""),
        ({"a": 2, "b": 1}, None, None),
    ],
)
def test_the_capped_tenant_rule(occupancy, queued, expected):
    assert band_cap(occupancy).capped_tenant(queued) == expected


@pytest.mark.parametrize(
    ("active", "held", "band", "expected"),
    [(3, False, True, "a"), (2, True, False, None)],
)
def test_the_cap_applies_only_between_the_batch_line_and_the_limit(
    active, held, band, expected
):
    cap = band_cap({"a": 2, "b": 1}, active=active, held=held)

    assert cap.in_reserve_band is band
    assert cap.capped_tenant({"a": 1, "b": 2}) == expected


# --- the reservation decides --------------------------------------------------------

BOTH = {"a": 10, "b": 20}
AB = [(1, "a"), (2, "b")]
NO_RESERVE = DispatchPolicy(limit=4, interactive_reserve=0)
# row: (policy, slots taken first, [(task, tenant, listing, require_listing, outcome)])
RESERVATION_ROWS = {
    "a-stale-listing-cannot-overshoot": (
        BAND,
        AB,
        [(3, "b", BOTH, False, "token"), (4, "b", BOTH, False, None)],
    ),
    "b-no-listing-no-cap": (
        BAND,
        AB,
        [(3, "b", None, False, "token"), (4, "b", None, False, "token")],
    ),
    "c-one-tenant-queued": (BAND, AB, [(3, "b", {"b": 20}, False, "token")]),
    "d-an-active-task-passes": (BAND, AB, [(1, "a", BOTH, False, "token")]),
    "e-band-without-listing": (BAND, AB, [(3, "b", None, True, "raises")]),
    "f-below-the-batch-line": (BAND, AB[:1], [(2, "b", None, True, "token")]),
    "g-full": (
        BAND,
        [(1, "a"), (2, "b"), (3, "a"), (4, "b")],
        [(5, "a", None, True, None)],
    ),
    "h-held": (BAND, AB, [(3, "b", None, True, None)]),
    "i-no-reserve": (NO_RESERVE, AB + AB[:1], [(4, "b", None, True, "token")]),
}


def slot_state(manager):
    return dict(manager._dispatch_reservations), dict(manager._slot_keys)


@pytest.mark.parametrize("row", list(RESERVATION_ROWS))
async def test_the_reservation_decision(row, monkeypatch):
    policy, seeded, calls = RESERVATION_ROWS[row]
    manager = BackgroundTaskManager()
    for task_id, tenant in seeded:
        assert manager.reserve_dispatch_slot(
            task_id, policy=policy, interactive=True, fairness_key=tenant
        )
    if row == "h-held":
        monkeypatch.setattr(
            task_execution, "release_hold", SimpleNamespace(closed=True)
        )

    for task_id, tenant, listing, require, outcome in calls:
        before = slot_state(manager)

        def reserve():
            return manager.reserve_dispatch_slot(
                task_id,
                policy=policy,
                interactive=True,
                fairness_key=tenant,
                queued_tenants=listing,
                require_listing=require,
            )

        if outcome == "raises":
            with pytest.raises(TenantListingRequired):
                reserve()
            assert slot_state(manager) == before
        elif outcome is None:
            assert reserve() is None
            assert slot_state(manager) == before
        else:
            assert reserve() is not None
            reservations, keys = slot_state(manager)
            assert reservations[task_id] == before[0].get(task_id, 0) + 1
            assert keys == {**before[1], task_id: keys[task_id]}


@pytest.fixture
def dispatcher(monkeypatch):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setenv(config.DISPATCHER_MAX_INFLIGHT_PER_PROCESS, "4")
    monkeypatch.setenv(config.DISPATCHER_INTERACTIVE_RESERVE_PER_PROCESS, "2")
    manager = BackgroundTaskManager()
    monkeypatch.setattr(task_execution, "background_task_manager", manager)
    ensured: list[int] = []
    coordinator = SimpleNamespace(
        _command_tasks=set(),
        execute_command=AsyncMock(return_value=False),
        require_recovery=Mock(),
    )

    async def ensure(task_id):
        ensured.append(task_id)
        return coordinator

    registry = SimpleNamespace(busy_command_tasks=lambda: set(), ensure=ensure)
    monkeypatch.setattr(runtime, "get_task_coordinator_registry", lambda: registry)

    async def executor(command):
        raise AssertionError("no command should be applied")

    return SimpleNamespace(manager=manager, ensured=ensured, executor=executor)


def seed_slots(manager, *slots, policy=BAND):
    for task_id, tenant in slots:
        assert manager.reserve_dispatch_slot(
            task_id, policy=policy, interactive=True, fairness_key=tenant
        )


async def test_a_second_missing_listing_ends_the_attempt(
    world, on, dispatcher, monkeypatch
):
    # The candidate carries no listing and the process is between the lines when
    # its slot is asked for. A selection after a refusal reads the live snapshot,
    # which is then between the lines; faking one below the batch line keeps the
    # bound of one rescan independent of that.
    world.add("a", "interactive", kind="resume_input")
    seed_slots(dispatcher.manager, (900, "a"), (901, "a"))
    below = DispatchCapSnapshot(
        policy=BAND, active_ids=frozenset({900}), occupancy={"a": 1}
    )
    monkeypatch.setattr(transport, "build_dispatch_cap_snapshot", lambda: below)
    real_find = transport._find_command_candidate
    scans: list = []

    def spy(*args, **kwargs):
        assert len(scans) < 2, "a third selection was started"
        scans.append(real_find(*args, **kwargs))
        return scans[-1]

    monkeypatch.setattr(transport, "_find_command_candidate", spy)
    held = slot_state(dispatcher.manager)

    assert await transport.dispatch_one_task_command(dispatcher.executor) is False

    assert len(scans) == 2 and scans[0] == scans[1] and scans[0] is not None
    assert slot_state(dispatcher.manager) == held
    assert dispatcher.ensured == []


@pytest.mark.parametrize(
    ("switch", "directed", "listing", "require"),
    [("true", None, {"team:3": 11, "team:4": 12}, True), ("true", 11, None, False)]
    + [("false", None, None, False)],
    ids=["on", "directed", "off"],
)
async def test_dispatch_passes_the_listing_and_the_listing_requirement(
    switch, directed, listing, require, dispatcher, monkeypatch
):
    monkeypatch.setenv("XAGENT_DISPATCHER_FAIR_SHARE", switch)
    monkeypatch.setattr(transport, "_fair_share_memo", None)
    candidate = transport.CommandCandidate(11, 5, KINDS.START, True, "team:3", listing)
    monkeypatch.setattr(
        transport, "_find_command_candidate", lambda *_a, **_k: candidate
    )
    seen: list[dict] = []

    def reserve(task_id, **kwargs):
        seen.append(kwargs)
        return None

    monkeypatch.setattr(dispatcher.manager, "reserve_dispatch_slot", reserve)

    assert (
        await transport.dispatch_one_task_command(
            dispatcher.executor, command_db_id=directed
        )
        is False
    )

    assert len(seen) == 1
    assert seen[0]["require_listing"] is require
    assert seen[0]["queued_tenants"] == listing


async def test_a_cap_refusal_ends_the_attempt_without_a_miss(
    clock, dispatcher, monkeypatch
):
    seed_slots(dispatcher.manager, (1, "a"), (2, "b"))
    listing = {"a": 10, "b": 20}
    scans: list[int] = []

    def find(*_args, **_kwargs):
        scans.append(1)
        return transport.CommandCandidate(30, 3, KINDS.START, True, "a", listing)

    monkeypatch.setattr(transport, "_find_command_candidate", find)
    before = slot_state(dispatcher.manager)

    assert await transport.dispatch_one_task_command(dispatcher.executor) is False

    assert len(scans) == 1
    assert slot_state(dispatcher.manager) == before
    assert 3 not in dispatcher.manager._slot_keys
    assert transport._fair_state.skips == {} and transport._fair_state.misses == 0
    assert dispatcher.ensured == []


BI = {"batch": 3, "interactive": 9}
SKIP_I, SKIP_BI = {("b", "interactive")}, {("b", "interactive"), ("b", "batch")}


@pytest.mark.parametrize(
    ("tenants", "skipped", "expected"),
    [
        ({"a": {"batch": 1}, "b": {"interactive": 9}}, SKIP_I, {"a": 1}),
        ({"a": {"batch": 1}, "b": BI}, SKIP_I, {"a": 1, "b": 3}),
        ({"a": {"batch": 1}, "b": BI}, SKIP_BI, {"a": 1}),
        ({"b": BI}, set(), {"b": 3}),
    ],
)
def test_the_count_leaves_out_a_tenant_whose_every_queued_class_is_skipped(
    tenants, skipped, expected
):
    counted = transport._counted_tenants(tenants, frozenset(skipped))

    assert dict(counted) == expected


def test_a_skipped_tenant_does_not_make_the_top_holder_capped(world, on, clock):
    own = world.add("a", "interactive")
    world.add("b", "interactive")
    transport._fair_state.skips[("b", "interactive")] = clock.now + 2
    cap = cap_of(active=[900, 901], occupancy={"a": 2, "b": 1}, policy=BAND)

    assert scan(cap, attempt=attempt()) == own


def complete(world, command_id):
    with world.sessions() as db:
        db.get(TaskExecutionCommand, command_id).status = "completed"
        db.commit()


def tracked(monkeypatch):
    found: list = []
    real = transport._find_command_candidate

    def spy(*args, **kwargs):
        found.append(real(*args, **kwargs))
        return found[-1]

    monkeypatch.setattr(transport, "_find_command_candidate", spy)
    return found


async def test_a_reservation_that_finds_the_band_without_a_listing_rescans_once_and_caps(
    world, on, dispatcher, monkeypatch
):
    manager = dispatcher.manager
    seed_slots(manager, (900, "a"))
    reply = world.add("a", "interactive", kind="resume_input")
    reply_task = world.task_of(reply)
    start_b = world.add("b", "interactive")
    found = tracked(monkeypatch)
    real_reserve = manager.reserve_dispatch_slot
    asked: list[tuple] = []

    def reserve(task_id, **kwargs):
        if not asked:
            # Another attempt takes a second slot of a: the process is between
            # the lines when this one asks.
            assert real_reserve(901, policy=BAND, interactive=True, fairness_key="a")
        try:
            token = real_reserve(task_id, **kwargs)
        except TenantListingRequired as raised:
            asked.append((task_id, kwargs, raised))
            raise
        asked.append((task_id, kwargs, token))
        return token

    monkeypatch.setattr(manager, "reserve_dispatch_slot", reserve)
    seen, stop = statements_of(world.engine)
    try:
        assert await transport.dispatch_one_task_command(dispatcher.executor) is False
    finally:
        stop()

    assert len(asked) == 2
    first, second = asked
    assert first[0] == reply_task and first[1]["queued_tenants"] is None
    assert first[1]["require_listing"] is True
    assert isinstance(first[2], TenantListingRequired)
    assert found[0].queued_tenants is None
    assert reply_task not in manager._slot_keys
    assert reply_task not in manager._dispatch_reservations
    assert second[0] == world.task_of(start_b)
    assert set(second[1]["queued_tenants"]) == {"a", "b"}
    assert second[2] is not None
    assert len(found) == 2
    assert transport._fair_state.since_fallback == 1
    assert sum("GROUP BY" in statement for statement in seen) == 1
    with world.sessions() as db:
        assert db.get(TaskExecutionCommand, reply).status == "pending"
    assert transport._fair_state.skips == {} and transport._fair_state.misses == 0


async def test_the_capped_tenant_does_not_hold_back_a_cancel_or_a_running_task(
    world, on, dispatcher, monkeypatch
):
    monkeypatch.setenv(config.DISPATCHER_MAX_INFLIGHT_PER_PROCESS, "5")
    policy = DispatchPolicy(limit=5, interactive_reserve=2)
    running = world.task_of(world.add("c", status="completed"))
    seed_slots(
        dispatcher.manager, (901, "a"), (902, "a"), (running, "c"), policy=policy
    )
    starts_of_a = [world.add("a", "interactive") for _ in range(5)]
    for _ in range(3):
        world.add("c")
    cancel = world.add("c", kind="cancel")
    message = world.follow_up(running, "c")
    found = tracked(monkeypatch)
    real_reserve = dispatcher.manager.reserve_dispatch_slot
    reserved: list[int] = []

    def reserve(task_id, **kwargs):
        reserved.append(task_id)
        return real_reserve(task_id, **kwargs)

    monkeypatch.setattr(dispatcher.manager, "reserve_dispatch_slot", reserve)
    for expected in (cancel, message):
        assert await transport.dispatch_one_task_command(dispatcher.executor) is False
        assert found[-1].id == expected
        complete(world, expected)

    assert reserved == [running]
    assert not set(starts_of_a) & {candidate.id for candidate in found}


def test_a_cancel_behind_the_capped_tenants_start_waits_until_the_cap_lifts(
    world, on, clock
):
    batch_of_y = world.add("y", "batch")
    start = world.add("x", "interactive")
    cancel = world.follow_up(world.task_of(start), "x", kind="cancel")
    held = cap_of(active=[900, 901], occupancy={"x": 2}, policy=BAND)
    lifted = cap_of(active=[900], occupancy={"x": 1}, policy=BAND)

    assert scan(held, attempt=attempt()) is None
    assert (
        transport._find_command_candidate(cancel, (), held, attempt=attempt()) is None
    )
    order = []
    for _ in range(3):
        order.append(scan(lifted, attempt=attempt()))
        complete(world, order[-1])

    assert order == [batch_of_y, start, cancel]


def queue_reply_first(world):
    return world.add("a", "interactive", kind="resume_input"), world.add(
        "b", "interactive", kind="resume_input"
    )


def queue_two(world):
    return world.add("a", "interactive"), world.add("b", "interactive")


def queue_one_in_band(world):
    own = world.add("a", "interactive")
    world.add("b", "batch")
    return own, None


def queue_nothing_found(world):
    for tenant in ("a", "b"):
        for _ in range(16):
            world.add(tenant, "interactive", pinned=True)
        if tenant == "a":
            own = world.add("a", "interactive")
    return own, None


def queue_empty_key(world):
    own = world.add("", "interactive")
    world.add("b", "batch")
    return own, None


PATHS = {
    "reply_first": (queue_reply_first, "a"),
    "fallback_round": (queue_two, "a"),
    "one_in_band": (queue_one_in_band, "a"),
    "nothing_found": (queue_nothing_found, "a"),
    "window": (queue_two, "a"),
    "one_in_band_empty_key": (queue_empty_key, ""),
}


@pytest.mark.parametrize("path", list(PATHS))
def test_every_listed_path_leaves_out_the_capped_tenants_new_work(
    path, world, on, clock
):
    build, tenant = PATHS[path]
    own, other = build(world)
    since_fallback = transport.FAIR_SHARE_FALLBACK_EVERY - 1
    if path == "fallback_round":
        transport._fair_state.since_fallback = since_fallback
    held = {tenant: 2, "b": 2 if path == "window" else 1}
    cap = cap_of(active=[900, 901], occupancy=held, policy=BAND)
    nobody = cap_of(active=[900, 901], policy=BAND)

    chosen, seen = recorded(world, lambda: scan(cap, attempt=attempt()))
    transport._fair_state.since_fallback = (
        since_fallback if path == "fallback_round" else 0
    )
    unheld = scan(nobody, attempt=attempt())

    assert unheld == own
    assert chosen == other
    if path == "window":
        assert len(windows_asked(seen)) == 1
        assert transport._fair_state.skips == {}
    if path == "reply_first":
        listing = next(i for i, (s, _) in enumerate(seen) if "GROUP BY" in s)
        reply = next(
            i for i, (s, _) in enumerate(seen) if "commands.kind = ?" in s and i > 0
        )
        assert listing < reply


def test_a_fallback_round_below_the_batch_line_lists_no_tenant(world, on, clock):
    world.add("a", "interactive")
    world.add("b", "interactive")
    below = cap_of(active=[900], occupancy={"a": 1}, policy=BAND)
    transport._fair_state.since_fallback = transport.FAIR_SHARE_FALLBACK_EVERY - 1

    _, fallback = recorded(world, lambda: scan(below, attempt=attempt()))
    transport._fair_state.since_fallback = 0
    _, ranked = recorded(world, lambda: scan(below, attempt=attempt()))

    assert not any("GROUP BY" in s for s, _ in fallback)
    assert any("GROUP BY" in s for s, _ in ranked)


def test_nobody_capped_sends_the_fair_selection_statements(world, on, clock):
    world.add("a", "interactive", kind="resume_input")
    world.add("b", "interactive")
    cap = cap_of(active=[900, 901], policy=BAND)

    _, seen = recorded(world, lambda: scan(cap, attempt=attempt()))

    assert not any("fairness_key = ?" in s for s, _ in seen)
