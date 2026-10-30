"""Seal, whole-attempt tracking and sticky unsettled evidence for pooled workers.

A pooled host (one web process plus N shared-execution workers) stops a
worker only after every worker has *applied* a seal and then reported nothing
in flight and nothing unsettled. These tests pin the Engine half of that:

- a seal refuses new command attempts at the dispatch entry, synchronously;
- an attempt counts from that entry, so the whole window from the candidate
  query through ``registry.ensure`` (including waiting for an old owner's
  close) to the apply is visible -- a seal cannot slip in between;
- ownership that ends without a proven release leaves a sticky
  ``unsettled_total`` that neither ``reopen()`` nor diagnostic truncation
  clears, while proven releases never set it;
- the held cap in shared mode keeps new execution pending while controls and
  active-task input still flow (the shared+held regression).

Run against SQLite and, with ``XAGENT_TEST_POSTGRES_URL``, PostgreSQL.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest
from sqlalchemy.orm import sessionmaker

from tests.shared.async_waits import DB_PROGRESS_TIMEOUT, eventually
from tests.web.services.task_database_shared import engine as engine_fixture
from tests.web.services.task_database_shared import task_id as task_id_fixture
from tests.web.services.test_task_coordinator_service import (
    database as database_fixture,
)
from xagent.web.models import database as database_module
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand

# Imported up front: modules that bind ``get_session_local`` at import time must
# not first load while a test has replaced it (see test_dispatch_cap).
from xagent.web.services import release_hold as release_hold_module
from xagent.web.services import (  # noqa: F401
    task_command_execution,
)
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_coordinator_runtime as runtime
from xagent.web.services import task_coordinator_service as ownership
from xagent.web.services import (  # noqa: F401
    task_execution,
    task_orchestrator,
    task_resume,
    task_resume_command,
    task_setup_snapshot,
    task_start,
    task_start_consumer,
)
from xagent.web.services.release_hold import (
    UNSETTLED_DIAGNOSTICS_LIMIT,
    ReleaseHold,
    execution_status,
)
from xagent.web.services.task_dispatch_cap import EXECUTION_OPENING_KINDS
from xagent.web.services.task_execution import BackgroundTaskManager
from xagent.web.services.task_execution_controller import task_control_snapshot

database = database_fixture
engine = engine_fixture
task_id = task_id_fixture


@pytest.fixture
def hold(monkeypatch: pytest.MonkeyPatch) -> ReleaseHold:
    """A private hold installed as the process singleton for this test."""

    fresh = ReleaseHold()
    for module in (release_hold_module, task_execution, transport, runtime):
        monkeypatch.setattr(module, "release_hold", fresh)
    return fresh


# --- hold state (no database) -------------------------------------------------


def test_seal_implies_closed_refuses_attempts_and_only_reopen_clears_it(
    hold: ReleaseHold,
) -> None:
    open_attempt = hold.try_begin_attempt("command_attempt", "next")
    assert open_attempt is not None
    hold.seal("release r1")
    hold.seal("release r1")  # idempotent
    assert hold.closed and hold.sealed
    # Already-counted work keeps its token; only new attempts are refused.
    assert hold.try_begin_attempt("command_attempt", "next") is None
    assert [r["kind"] for r in hold.snapshot()["responsibilities"]] == [
        "command_attempt"
    ]
    open_attempt.release()
    # ``close`` never lifts a seal; only ``reopen`` does.
    hold.close("release r1")
    assert hold.sealed
    hold.reopen()
    assert not hold.closed and not hold.sealed
    assert hold.try_begin_attempt("command_attempt", "next") is not None


def test_unsettled_total_is_sticky_across_reopen_and_truncation(
    hold: ReleaseHold,
) -> None:
    hold.record_unsettled("task_lease", "task:1", "release raised")
    hold.close("release r1")
    hold.reopen()
    assert hold.unsettled_total == 1
    for index in range(UNSETTLED_DIAGNOSTICS_LIMIT + 10):
        hold.record_unsettled("command_claim", f"command:{index}", "interrupted")
    snapshot = hold.snapshot()
    assert hold.unsettled_total == UNSETTLED_DIAGNOSTICS_LIMIT + 11
    assert snapshot["unsettled_total"] == UNSETTLED_DIAGNOSTICS_LIMIT + 11
    # Diagnostics are bounded; the newest entries are kept.
    assert len(snapshot["unsettled"]) == UNSETTLED_DIAGNOSTICS_LIMIT
    assert snapshot["unsettled"][-1]["identity"] == (
        f"command:{UNSETTLED_DIAGNOSTICS_LIMIT + 9}"
    )


async def test_execution_status_reports_seal_and_unsettled_without_changing_busy(
    hold: ReleaseHold, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        task_execution, "background_task_manager", BackgroundTaskManager()
    )
    hold.seal("release r1")
    hold.record_unsettled("task_lease", "task:9", "lease lost")
    status = execution_status()
    # ``busy`` keeps its meaning; unsettled evidence is reported beside it.
    assert status["busy"] is False
    assert status["counts"]["task_coordinators"] == 0
    assert status["sealed"] is True
    assert status["unsettled_total"] == 1
    assert status["unsettled"] == [
        {"kind": "task_lease", "identity": "task:9", "reason": "lease lost"}
    ]


# --- shared-mode dispatcher over a real database ---------------------------------


class PoolHost:
    """One shared-execution worker's dispatcher over the parametrized database."""

    def __init__(
        self,
        factory: sessionmaker,
        registry: runtime.TaskCoordinatorRegistry,
        manager: BackgroundTaskManager,
        user_id: int,
    ) -> None:
        self.factory = factory
        self.registry = registry
        self.manager = manager
        self.user_id = user_id
        self.claimed: list[tuple[int, str]] = []
        self.held: list[asyncio.Task] = []
        self.block_handler: asyncio.Event | None = None
        self.in_handler = asyncio.Event()
        # Ends every execution ``handler`` registered.
        self.finish_executions = asyncio.Event()
        self._identity = 0

    def new_task(self) -> int:
        with self.factory() as db:
            task = Task(user_id=self.user_id, title="t", description="t")
            db.add(task)
            db.commit()
            return int(task.id)

    def enqueue(self, task_id: int, kind: str) -> int:
        self._identity += 1
        with self.factory() as db:
            row = TaskExecutionCommand(
                task_id=task_id,
                command_id=f"seal-cmd-{self._identity}",
                kind=kind,
                payload={},
            )
            db.add(row)
            db.commit()
            return int(row.id)

    def status_of(self, command_id: int) -> str:
        with self.factory() as db:
            return str(db.get(TaskExecutionCommand, command_id).status)

    def live(self, task_id: int) -> asyncio.Event:
        finish = asyncio.Event()
        handle = asyncio.get_running_loop().create_task(finish.wait())
        self.held.append(handle)
        self.manager.register_task(task_id, handle)
        return finish

    async def handler(self, command: Any) -> dict[str, Any]:
        self.claimed.append((command.task_id, command.kind.value))
        if self.block_handler is not None:
            self.in_handler.set()
            await self.block_handler.wait()
        if command.kind.value in EXECUTION_OPENING_KINDS:
            handle = asyncio.get_running_loop().create_task(
                self.finish_executions.wait()
            )
            self.held.append(handle)
            self.manager.register_task(command.task_id, handle)
        return {"ok": True}


@pytest.fixture
async def pool(database, hold: ReleaseHold, monkeypatch: pytest.MonkeyPatch):
    factory, existing_task = database
    with factory() as db:
        user_id = int(db.get(Task, existing_task).user_id)
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("XAGENT_TASK_EXECUTION_ROLE", "worker")
    monkeypatch.setattr(database_module, "get_session_local", lambda: factory)
    monkeypatch.setattr(transport, "get_runner_id", lambda: "worker-1")
    monkeypatch.setattr(runtime, "get_runner_id", lambda: "worker-1")
    monkeypatch.setattr(runtime, "get_task_lease_heartbeat_seconds", lambda: 0.5)
    registry = runtime.TaskCoordinatorRegistry(factory)
    monkeypatch.setattr(runtime, "get_task_coordinator_registry", lambda: registry)
    monkeypatch.setattr(runtime, "_registry", registry)
    manager = BackgroundTaskManager()
    monkeypatch.setattr(task_execution, "background_task_manager", manager)
    host = PoolHost(factory, registry, manager, user_id)
    try:
        yield host
    finally:
        if host.block_handler is not None:
            host.block_handler.set()
        for handle in host.held:
            handle.cancel()
        await asyncio.gather(*host.held, return_exceptions=True)
        await registry.close()


async def dispatch(host: PoolHost, command_db_id: int | None = None) -> bool:
    return await asyncio.wait_for(
        transport.dispatch_one_task_command(host.handler, command_db_id=command_db_id),
        DB_PROGRESS_TIMEOUT,
    )


async def wait_thread_event(event: threading.Event) -> None:
    assert await asyncio.to_thread(event.wait, DB_PROGRESS_TIMEOUT)


def idle(hold: ReleaseHold) -> bool:
    status = execution_status()
    return not status["busy"] and status["counts"]["task_coordinators"] == 0


async def test_seal_cannot_slip_between_candidate_query_ensure_and_apply(
    pool: PoolHost, hold: ReleaseHold, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2: an attempt waiting on an old owner's close is counted through it.

    The old coordinator's ``_release`` is blocked; a CANCEL attempt's
    candidate query returns only after that close began, so the attempt waits
    in ``registry.ensure`` for the old owner to leave. Sealing then must not
    report idle, must refuse any new attempt at the entry, and the successor
    coordinator may only be created inside the already-counted attempt.
    """

    host = pool
    task = host.new_task()
    old = await host.registry.ensure(task)
    assert old is not None

    releasing, unblock_release = threading.Event(), threading.Event()
    real_release = old._release

    def blocked_release() -> bool:
        result = real_release()
        releasing.set()
        assert unblock_release.wait(DB_PROGRESS_TIMEOUT)
        return result

    monkeypatch.setattr(old, "_release", blocked_release)

    querying, unblock_query = threading.Event(), threading.Event()
    real_find = transport._find_command_candidate
    queries: list[int] = []

    def gated_find(*args: Any, **kwargs: Any) -> Any:
        queries.append(1)
        if len(queries) == 1:
            querying.set()
            assert unblock_query.wait(DB_PROGRESS_TIMEOUT)
        return real_find(*args, **kwargs)

    monkeypatch.setattr(transport, "_find_command_candidate", gated_find)

    entered_ensure = asyncio.Event()
    real_ensure = host.registry.ensure

    async def observed_ensure(task_id: int) -> Any:
        entered_ensure.set()
        return await real_ensure(task_id)

    monkeypatch.setattr(host.registry, "ensure", observed_ensure)

    created_while: list[list[str]] = []
    real_init = runtime.TaskCoordinator.__init__

    def recording_init(self: Any, registry: Any, task_id: int) -> None:
        created_while.append(
            [item["kind"] for item in hold.snapshot()["responsibilities"]]
        )
        real_init(self, registry, task_id)

    monkeypatch.setattr(runtime.TaskCoordinator, "__init__", recording_init)

    cancel = host.enqueue(task, "cancel")
    attempt = asyncio.create_task(dispatch(host))
    await wait_thread_event(querying)
    # The candidate query is already part of the counted attempt.
    status = execution_status()
    assert status["busy"] is True
    assert [r["kind"] for r in status["hold"]["responsibilities"]] == [
        "command_attempt"
    ]

    closer = asyncio.create_task(old.close())
    await wait_thread_event(releasing)
    unblock_query.set()
    await asyncio.wait_for(entered_ensure.wait(), DB_PROGRESS_TIMEOUT)
    # The attempt now waits for the old owner's close inside ``ensure``.
    assert not attempt.done()

    hold.seal("release r1")
    assert execution_status()["busy"] is True
    # A new attempt is refused at the entry: no query, no token.
    assert await dispatch(host) is False
    assert len(queries) == 1
    assert [r["kind"] for r in hold.snapshot()["responsibilities"]] == [
        "command_attempt"
    ]

    unblock_release.set()
    await closer
    assert await attempt is True
    assert host.claimed == [(task, "cancel")]
    assert host.status_of(cancel) == "completed"
    # The successor owner was created only inside the counted attempt.
    assert created_while == [["command_attempt"]]
    # The sealed worker settles: the successor releases on its idle check.
    await eventually(lambda: idle(hold))
    assert hold.unsettled_total == 0


async def test_idle_poll_is_busy_before_seal_and_absent_after(
    pool: PoolHost, hold: ReleaseHold, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = pool
    querying, unblock = threading.Event(), threading.Event()
    real_find = transport._find_command_candidate
    calls: list[int] = []

    def gated_find(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        querying.set()
        assert unblock.wait(DB_PROGRESS_TIMEOUT)
        return real_find(*args, **kwargs)

    monkeypatch.setattr(transport, "_find_command_candidate", gated_find)
    poll = asyncio.create_task(dispatch(host))
    await wait_thread_event(querying)
    assert execution_status()["busy"] is True
    unblock.set()
    assert await poll is False
    assert idle(hold)

    hold.seal("release r1")
    assert await dispatch(host) is False
    assert calls == [1]
    assert idle(hold)


def lease_admit(db: Any, lease: Any) -> Any:
    return ownership.begin_task_execution_no_commit(
        db,
        lease,
        expected=task_control_snapshot(db.get(Task, lease.task_id)),
        new_run=True,
    )


def settle_completed(db: Any, context: Any, error: BaseException | None) -> None:
    task = db.get(Task, context.lease.task_id)
    task.status = TaskStatus.COMPLETED
    task.control_state = "completed"
    task.state_version += 1


def settle_fails(db: Any, context: Any, error: BaseException | None) -> None:
    raise RuntimeError("settlement commit failed")


@pytest.mark.parametrize(
    "fault",
    ["release_raises", "release_false", "settlement_fails", "unhealthy"],
)
async def test_unproven_owner_release_leaves_sticky_unsettled_evidence(
    pool: PoolHost, hold: ReleaseHold, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    host = pool
    task = host.new_task()
    owner = await host.registry.ensure(task)
    assert owner is not None
    if fault == "release_raises":

        def raising() -> bool:
            raise RuntimeError("release commit failed")

        monkeypatch.setattr(owner, "_release", raising)
        await owner.close()
    elif fault == "release_false":
        monkeypatch.setattr(owner, "_release", lambda: False)
        await owner.close()
    elif fault == "settlement_fails":

        async def execute(_context: Any) -> None:
            return None

        handle = owner.submit_execution(
            admit=lease_admit, execute=execute, settle=settle_fails
        )
        assert handle is not None
        with pytest.raises(RuntimeError):
            await handle
        await owner.close()
    else:
        owner._healthy = False
        await owner.close()

    # Every in-process count is back to zero ...
    status = execution_status()
    assert status["busy"] is False
    assert status["counts"]["task_coordinators"] == 0
    assert status["counts"]["live_executions"] == 0
    # ... but the release was not proven, so the evidence stays.
    assert status["unsettled_total"] == 1
    assert status["unsettled"][0]["kind"] == "task_lease"
    assert status["unsettled"][0]["identity"] == f"task:{task}"
    hold.reopen()
    for index in range(UNSETTLED_DIAGNOSTICS_LIMIT):
        hold.record_unsettled("probe", str(index), "filler")
    assert execution_status()["unsettled_total"] == UNSETTLED_DIAGNOSTICS_LIMIT + 1


async def test_interrupted_command_application_records_the_claim_left_for_recovery(
    pool: PoolHost, hold: ReleaseHold
) -> None:
    host = pool
    task = host.new_task()
    host.block_handler = asyncio.Event()
    host.enqueue(task, "pause")
    attempt = asyncio.create_task(dispatch(host))
    await asyncio.wait_for(host.in_handler.wait(), DB_PROGRESS_TIMEOUT)
    owner = host.registry._coordinators[task]
    # The owner closes under the command (heartbeat loss, shutdown drain).
    await owner.close()
    assert await attempt is False
    kinds = {item["kind"] for item in hold.snapshot()["unsettled"]}
    assert "command_claim" in kinds
    assert hold.unsettled_total >= 1
    assert execution_status()["busy"] is False


async def test_proven_releases_never_record_unsettled(
    pool: PoolHost, hold: ReleaseHold
) -> None:
    host = pool
    # Explicit close with a confirmed release.
    first = await host.registry.ensure(host.new_task())
    assert first is not None
    await first.close()
    # A settled execution followed by the owner's own idle release.
    second_task = host.new_task()
    second = await host.registry.ensure(second_task)
    assert second is not None

    async def execute(_context: Any) -> None:
        return None

    handle = second.submit_execution(
        admit=lease_admit, execute=execute, settle=settle_completed
    )
    assert handle is not None
    await handle
    # As after any command or child: the owner checks whether it is idle.
    second._ensure_idle_check()
    await eventually(lambda: idle(hold))
    # A dispatched control through a fresh owner.
    control_task = host.new_task()
    host.enqueue(control_task, "pause")
    assert await dispatch(host) is True
    await eventually(lambda: idle(hold))
    assert hold.unsettled_total == 0
    assert hold.snapshot()["unsettled"] == []


async def test_shared_held_cap_keeps_new_execution_pending_and_lets_controls_through(
    pool: PoolHost, hold: ReleaseHold
) -> None:
    """Shared+held regression: the order the pooled drain depends on."""

    host = pool
    idle_task = host.new_task()
    active_task = host.new_task()
    controlled = host.new_task()
    finish_active = host.live(active_task)

    hold.close("release r1")
    start = host.enqueue(idle_task, "start")
    queued_start = host.enqueue(controlled, "start")
    active_input = host.enqueue(active_task, "message")
    pause = host.enqueue(controlled, "pause")
    cancel = host.enqueue(controlled, "cancel")

    # Drain whatever the held cap lets through.
    while await dispatch(host):
        pass
    claimed = set(host.claimed)
    assert (active_task, "message") in claimed
    assert (controlled, "pause") in claimed
    assert (controlled, "cancel") in claimed
    assert (idle_task, "start") not in claimed
    assert (controlled, "start") not in claimed
    assert host.status_of(start) == "pending"
    assert host.status_of(active_input) == "completed"
    assert host.status_of(pause) == "completed"
    assert host.status_of(cancel) == "completed"
    # The completed cancel settles the queued START it overtook.
    assert host.status_of(queued_start) != "pending"

    finish_active.set()
    host.finish_executions.set()
    # Coordinators release their leases; nothing unsettled on this path.
    await eventually(lambda: idle(hold))
    assert hold.unsettled_total == 0

    # Reopen (abort): the held START runs exactly once.
    hold.reopen()
    while await dispatch(host):
        pass
    assert host.claimed.count((idle_task, "start")) == 1


async def test_sealed_owner_leaves_a_queued_control_for_the_successor(
    pool: PoolHost, hold: ReleaseHold
) -> None:
    """A control queued behind a running one cannot pin a sealed worker.

    Closed but unsealed, a pending control keeps its task's owner (it will
    claim it next). Once sealed this worker starts no attempt, so the owner
    releases its lease on the idle check and leaves the control pending in
    the database for the successor -- the only way the drain can settle.
    """
    host = pool
    task = host.new_task()
    host.block_handler = asyncio.Event()
    host.enqueue(task, "pause")
    first = asyncio.create_task(dispatch(host))
    await asyncio.wait_for(host.in_handler.wait(), DB_PROGRESS_TIMEOUT)
    queued_cancel = host.enqueue(task, "cancel")
    hold.close("release r1")
    host.block_handler.set()
    assert await first is True

    # Closed only: the queued control still holds the owner across idle checks.
    await asyncio.sleep(1.5)
    assert task in host.registry._coordinators
    assert host.status_of(queued_cancel) == "pending"

    hold.seal("release r1")
    assert await dispatch(host) is False
    await eventually(lambda: idle(hold))
    assert host.status_of(queued_cancel) == "pending"
    assert hold.unsettled_total == 0

    # Abort: the successor attempt (here this worker reopened) applies it.
    hold.reopen()
    assert await dispatch(host) is True
    assert host.status_of(queued_cancel) == "completed"


@pytest.mark.parametrize("stage", ["ensure", "apply"])
async def test_cancelled_dispatch_attempt_releases_its_token(
    pool: PoolHost, hold: ReleaseHold, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    """A dispatcher cancelled mid-attempt leaves no command_attempt behind."""
    host = pool
    task = host.new_task()
    host.enqueue(task, "pause")
    reached = asyncio.Event()
    if stage == "ensure":
        real_ensure = host.registry.ensure

        async def stalled_ensure(task_id: int) -> Any:
            reached.set()
            await asyncio.Event().wait()
            return await real_ensure(task_id)

        monkeypatch.setattr(host.registry, "ensure", stalled_ensure)
    else:
        host.block_handler = asyncio.Event()
        host.in_handler = reached

    attempt = asyncio.create_task(dispatch(host))
    await asyncio.wait_for(reached.wait(), DB_PROGRESS_TIMEOUT)
    # Applying also holds the inner ``command_dispatch`` token.
    assert "command_attempt" in {r["kind"] for r in hold.snapshot()["responsibilities"]}
    attempt.cancel()
    with pytest.raises(asyncio.CancelledError):
        await attempt
    assert hold.snapshot()["responsibilities"] == []


async def test_combined_mode_refuses_command_attempts_once_sealed(
    pool: PoolHost, hold: ReleaseHold, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``sealed`` reported by ``execution_status`` is enforced in combined mode too."""
    host = pool
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "false")
    monkeypatch.delenv("XAGENT_TASK_EXECUTION_ROLE", raising=False)
    task = host.new_task()
    pause = host.enqueue(task, "pause")
    hold.seal("release r1")
    assert await dispatch(host) is False
    assert host.claimed == []
    assert host.status_of(pause) == "pending"
    hold.reopen()
    assert await dispatch(host) is True
    assert host.claimed == [(task, "pause")]
