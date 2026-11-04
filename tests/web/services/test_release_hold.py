"""Planned-restart execution hold: ledger, combined-mode dispatch cap, tick pause.

The hold is the host seam a thin ingress Gate drives. These tests pin what the
Engine side guarantees on its own: refusal before side effects, a ledger that
covers hand-off windows, and the combined dispatcher applying the existing
closed-cap rules (controls and active-task input pass, new execution waits).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from tests.shared.async_waits import DB_PROGRESS_TIMEOUT, eventually
from xagent.web.models import database
from xagent.web.models.database import Base
from xagent.web.models.task import Task
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User

# Imported up front: modules that bind ``get_session_local`` at import time must
# not first load while a test has replaced it (see test_dispatch_cap).
from xagent.web.services import release_hold as release_hold_module
from xagent.web.services import (  # noqa: F401
    task_command_execution,
)
from xagent.web.services import task_command_transport as transport
from xagent.web.services import (  # noqa: F401
    task_coordinator_runtime,
    task_execution,
    task_orchestrator,
    task_resume,
    task_resume_command,
    task_runtime_secrets,
    task_setup_snapshot,
    task_start,
    task_start_consumer,
)
from xagent.web.services.release_hold import ReleaseHold, execution_status
from xagent.web.services.task_execution import BackgroundTaskManager
from xagent.web.services.task_execution_admission import (
    AdmissionQueueFull,
    ExecutionHeld,
)


@pytest.fixture
def hold(monkeypatch: pytest.MonkeyPatch) -> ReleaseHold:
    """A private hold installed as the process singleton for this test."""

    fresh = ReleaseHold()
    monkeypatch.setattr(release_hold_module, "release_hold", fresh)
    monkeypatch.setattr(task_execution, "release_hold", fresh)
    monkeypatch.setattr(transport, "release_hold", fresh)
    return fresh


# --- ledger ------------------------------------------------------------------


def test_admit_refuses_before_the_block_runs_while_held(hold: ReleaseHold) -> None:
    ran: list[str] = []
    hold.close("release r1")
    with pytest.raises(ExecutionHeld) as refused:
        with hold.admit("sdk_create", "agent:1"):
            ran.append("body")
    assert ran == []
    # Every ingress that already projects queue-full handles the refusal.
    assert isinstance(refused.value, AdmissionQueueFull)
    assert hold.try_admit("sdk_create") is None
    assert hold.snapshot()["refused"] == {"sdk_create": 2}
    assert hold.snapshot()["responsibilities"] == []


def test_ledger_counts_admitted_tracked_and_adopted_work(hold: ReleaseHold) -> None:
    async def scenario() -> None:
        admitted = hold.try_admit("external_create", "ca:1")
        assert admitted is not None
        hold.close("release r1")
        # Tracking is allowed while held: it never opens new execution.
        tracked = hold.track("command_dispatch", "7")
        outer = asyncio.get_running_loop().create_future()
        hold.adopt("slack_interaction", "click", outer)
        kinds = {item["kind"] for item in hold.snapshot()["responsibilities"]}
        assert kinds == {"external_create", "command_dispatch", "slack_interaction"}

        admitted.release()
        admitted.release()  # idempotent
        with tracked:
            pass
        outer.set_result(None)
        await asyncio.sleep(0)
        assert hold.snapshot()["responsibilities"] == []

        hold.reopen()
        assert hold.try_admit("external_create") is not None

    asyncio.run(scenario())


async def test_execution_status_reports_each_owner(
    hold: ReleaseHold, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = BackgroundTaskManager()
    monkeypatch.setattr(task_execution, "background_task_manager", manager)
    assert execution_status()["busy"] is False

    finish = asyncio.Event()
    handle = asyncio.get_running_loop().create_task(finish.wait())
    manager.register_task(41, handle)
    token = hold.track("ws_execute_task", "task:41")
    status = execution_status()
    assert status["busy"] is True
    assert status["counts"]["live_executions"] == 1
    assert status["counts"]["responsibilities"] == 1

    token.release()
    finish.set()
    await handle
    await eventually(lambda: not execution_status()["busy"])


# --- combined-mode dispatcher --------------------------------------------------


class CombinedHost:
    """A combined (non-shared) dispatcher over a SQLite command queue."""

    def __init__(self, factory: sessionmaker, user_id: int) -> None:
        self.factory = factory
        self.user_id = user_id
        self.claimed: list[tuple[int, str]] = []
        self.in_handler = asyncio.Event()
        self.release_handler = asyncio.Event()
        self.block = False
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
                command_id=f"cmd-{self._identity}",
                kind=kind,
                payload={},
            )
            db.add(row)
            db.commit()
            return int(row.id)

    def status_of(self, command_id: int) -> tuple[str, int]:
        with self.factory() as db:
            row = db.get(TaskExecutionCommand, command_id)
            return str(row.status), int(row.attempt_count or 0)

    async def handler(self, command: Any) -> dict[str, Any]:
        self.claimed.append((command.task_id, command.kind.value))
        if self.block:
            self.in_handler.set()
            await self.release_handler.wait()
        return {"ok": True}


@pytest.fixture
async def combined(tmp_path, monkeypatch: pytest.MonkeyPatch, hold: ReleaseHold):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'combined.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as db:
        user = User(username="combined-user", password_hash="unused")
        db.add(user)
        db.commit()
        user_id = int(user.id)
    monkeypatch.delenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", raising=False)
    monkeypatch.delenv("XAGENT_WORKER_COUNT", raising=False)
    monkeypatch.setattr(database, "get_session_local", lambda: factory)
    monkeypatch.setattr(transport, "get_runner_id", lambda: "combined")
    manager = BackgroundTaskManager()
    monkeypatch.setattr(task_execution, "background_task_manager", manager)
    held: list[asyncio.Task] = []
    host = CombinedHost(factory, user_id)
    host.manager = manager  # type: ignore[attr-defined]
    host.held = held  # type: ignore[attr-defined]
    try:
        yield host
    finally:
        for handle in held:
            handle.cancel()
        await asyncio.gather(*held, return_exceptions=True)
        engine.dispose()


async def dispatch(host: CombinedHost, command_db_id: int | None = None) -> bool:
    return await asyncio.wait_for(
        transport.dispatch_one_task_command(host.handler, command_db_id=command_db_id),
        DB_PROGRESS_TIMEOUT,
    )


async def test_combined_dispatch_holds_new_execution_but_applies_controls_and_active_input(
    combined: CombinedHost, hold: ReleaseHold
) -> None:
    host = combined
    idle_task = host.new_task()
    active_task = host.new_task()
    paused_target = host.new_task()

    finish = asyncio.Event()
    live = asyncio.get_running_loop().create_task(finish.wait())
    host.held.append(live)  # type: ignore[attr-defined]
    host.manager.register_task(active_task, live)  # type: ignore[attr-defined]

    new_message = host.enqueue(idle_task, "message")
    resume = host.enqueue(paused_target, "resume")
    hold.close("release r1")

    # The immediate path (``dispatch_task_command_promptly`` names the row):
    # a command that would open an execution is not claimed and stays pending
    # with no attempt counted.
    assert await dispatch(host, new_message) is False
    assert await dispatch(host, resume) is False
    assert host.status_of(new_message) == ("pending", 0)
    assert host.status_of(resume) == ("pending", 0)

    # Input for a task that is already active keeps the cap's existing pass.
    active_input = host.enqueue(active_task, "message")
    assert await dispatch(host) is True
    assert host.claimed == [(active_task, "message")]
    assert host.status_of(active_input)[0] == "completed"

    # A control of an idle task is still applied.
    pause = host.enqueue(paused_target, "pause")
    assert await dispatch(host, pause) is True
    assert host.claimed[-1] == (paused_target, "pause")

    # Reopening (an aborted release) lets the held command run, exactly once.
    hold.reopen()
    assert await dispatch(host, new_message) is True
    assert host.claimed.count((idle_task, "message")) == 1
    assert host.status_of(new_message)[0] == "completed"
    finish.set()


async def test_dispatch_attempt_is_a_responsibility_until_it_returns(
    combined: CombinedHost, hold: ReleaseHold
) -> None:
    host = combined
    host.block = True
    command = host.enqueue(host.new_task(), "message")
    attempt = asyncio.get_running_loop().create_task(dispatch(host, command))
    await asyncio.wait_for(host.in_handler.wait(), DB_PROGRESS_TIMEOUT)
    # Claimed, not yet registered anywhere else: the ledger still shows it.
    hold.close("release r1")
    status = execution_status()
    assert status["busy"] is True
    assert [item["kind"] for item in status["hold"]["responsibilities"]] == [
        "command_dispatch"
    ]
    host.release_handler.set()
    assert await attempt is True
    assert execution_status()["busy"] is False


# --- trigger tick ------------------------------------------------------------------


async def test_trigger_tick_pauses_at_its_boundary_and_resumes(
    hold: ReleaseHold, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xagent.web import app as app_module
    from xagent.web.services import triggers as triggers_module
    from xagent.web.services import workforce_runtime

    scans: list[int] = []
    dispatches: list[int] = []
    tick_started = asyncio.Event()
    tick_release = asyncio.Event()

    def scan(_db: Any) -> list[Any]:
        scans.append(1)
        return []

    async def dispatch_pending(_db: Any, *, limit: int) -> int:
        dispatches.append(limit)
        if len(dispatches) == 1:
            # The first tick is in progress when the hold closes.
            tick_started.set()
            await tick_release.wait()
        return 0

    monkeypatch.setattr(triggers_module, "scan_due_scheduled_triggers", scan)
    monkeypatch.setattr(
        triggers_module, "dispatch_pending_trigger_runs", dispatch_pending
    )
    monkeypatch.setattr(
        workforce_runtime, "reap_stale_preview_workforce_runs", lambda _db: []
    )
    monkeypatch.setattr(app_module, "get_gmail_watch_enabled", lambda: False)
    # Each tick opens (and closes) a session it never queries here.
    monkeypatch.setattr(database, "get_session_local", lambda: sessionmaker())

    loop_task = asyncio.get_running_loop().create_task(
        app_module._run_trigger_dispatcher(poll_interval_seconds=0, batch_size=3)
    )
    try:
        await asyncio.wait_for(tick_started.wait(), DB_PROGRESS_TIMEOUT)
        hold.close("release r1")
        # The in-progress tick is a responsibility until it finishes.
        assert any(
            item["kind"] == "trigger_tick"
            for item in hold.snapshot()["responsibilities"]
        )
        tick_release.set()
        await eventually(lambda: not hold.snapshot()["responsibilities"])
        seen = (len(scans), len(dispatches))
        await asyncio.sleep(0.05)  # negative probe: paused ticks do nothing
        assert (len(scans), len(dispatches)) == seen
        hold.reopen()
        await eventually(lambda: len(dispatches) > seen[1])
    finally:
        loop_task.cancel()
        await asyncio.gather(loop_task, return_exceptions=True)
