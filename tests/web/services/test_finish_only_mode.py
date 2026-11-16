"""Finish-only mode: a generation that only finishes the runs it owns."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy.orm import sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from tests.web.services.task_database_shared import task_id as task_id_fixture
from xagent.web import app as app_module
from xagent.web.models import database
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.services import release_hold as release_hold_module
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_event_bridge
from xagent.web.services.release_hold import (
    finishing_status,
    finishing_status_async,
    register_finishing_counter,
    release_hold,
)

engine = engine_fixture
task_id = task_id_fixture

OLD = "blue"
NEW = "green"


@pytest.fixture(autouse=True)
def _fresh_hold(monkeypatch):
    monkeypatch.setattr(release_hold, "_closed", False)
    monkeypatch.setattr(release_hold, "_sealed", False)
    monkeypatch.setattr(release_hold, "_finishing", False)
    monkeypatch.setattr(release_hold_module, "_finishing_counters", {})


@pytest.fixture
def factory(engine, monkeypatch):
    monkeypatch.setattr(transport, "get_shared_task_execution_enabled", lambda: True)
    result = sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(database, "_SessionLocal", result)
    return result


def seed(factory, owner_task_id):
    """One pause and one message per ownership case; {case: {kind: id}}."""

    now = datetime.now(timezone.utc)
    expired, live = now - timedelta(seconds=5), now + timedelta(minutes=5)
    with factory() as db, db.begin():
        user_id = db.get(Task, owner_task_id).user_id
        ids: dict[str, dict[str, int]] = {}

        def case(name, **fields):
            task = Task(user_id=user_id, title=name, **fields)
            db.add(task)
            db.flush()
            ids[name] = {}
            for kind in ("pause", "message"):
                row = TaskExecutionCommand(
                    task_id=task.id,
                    command_id=f"{name}-{kind}",
                    kind=kind,
                    payload={},
                )
                db.add(row)
                db.flush()
                ids[name][kind] = row.id
                # Commands of a task run in order: one command per task here.
                if kind == "pause":
                    break
            return task.id

        case("unowned")
        case(
            "old_live",
            status=TaskStatus.RUNNING,
            runner_id=OLD,
            lease_attempt_id="o",
            lease_expires_at=live,
        )
        case(
            "old_expired",
            status=TaskStatus.RUNNING,
            runner_id=OLD,
            lease_attempt_id="e",
            lease_expires_at=expired,
        )
        case(
            "new_live",
            status=TaskStatus.RUNNING,
            runner_id=NEW,
            lease_attempt_id="n",
            lease_expires_at=live,
        )
    return {name: kinds["pause"] for name, kinds in ids.items()}


def claimable(factory, runner, *, finishing):
    release_hold._finishing = finishing
    try:
        with factory() as db:
            rows = transport._claimable_query(db, runner_id=runner, command_db_id=None)
            return {int(row.id) for row in rows.all()}
    finally:
        release_hold._finishing = False


def test_finishing_claims_only_tasks_whose_lease_it_holds(factory, task_id):
    ids = seed(factory, task_id)
    old = claimable(factory, OLD, finishing=True)
    # A pause of a task nobody owns, of another process's task and of a task
    # whose lease lapsed is not this process's to apply any more.
    assert old == {ids["old_live"]}


def test_serving_generation_claims_everything_new_and_not_the_live_owner(
    factory, task_id
):
    ids = seed(factory, task_id)
    new = claimable(factory, NEW, finishing=False)
    assert new == {ids["unowned"], ids["old_expired"], ids["new_live"]}
    # The serving one never takes a task the finishing one still heartbeats.
    assert ids["old_live"] not in new
    # Not finishing keeps today's rule for the lapsed lease of the same runner.
    assert ids["old_expired"] in claimable(factory, OLD, finishing=False)


def test_a_closed_but_not_finishing_process_still_claims_foreign_controls(
    factory, task_id, monkeypatch
):
    """The gap finish-only mode closes: a closed hold alone lets a control of
    an unowned task through."""
    ids = seed(factory, task_id)
    monkeypatch.setattr(release_hold, "_closed", True)
    assert ids["unowned"] in claimable(factory, OLD, finishing=False)
    assert ids["unowned"] not in claimable(factory, OLD, finishing=True)


def test_claim_applies_the_restriction_in_the_conditional_update(
    factory, task_id, monkeypatch
):
    ids = seed(factory, task_id)
    monkeypatch.setattr(transport, "consumes_task_commands", lambda: True)
    # Non-shared claim path: the shared one returns before the UPDATE when no
    # task coordinator is current, which would leave the UPDATE unexercised.
    monkeypatch.setattr(transport, "get_shared_task_execution_enabled", lambda: False)
    release_hold._finishing = True
    # The lookup finds the owned pause; flip ownership before the UPDATE runs.
    real_query = transport._claimable_query

    def stale_lookup(db, **kw):
        release_hold._finishing = False
        try:
            return real_query(db, **kw)
        finally:
            release_hold._finishing = True

    monkeypatch.setattr(transport, "_claimable_query", stale_lookup)
    with factory() as db:
        claimed = transport.claim_task_command(
            db, runner_id=OLD, command_db_id=ids["unowned"]
        )
    assert claimed is None
    with factory() as db:
        row = db.get(TaskExecutionCommand, ids["unowned"])
        assert row.status == "pending" and row.attempt_count == 0
        assert row.claimed_by is None


def test_hold_begin_finishing_is_one_way():
    release_hold.begin_finishing()
    assert release_hold.finishing and release_hold.closed and not release_hold.sealed
    release_hold.reopen()
    assert release_hold.finishing and release_hold.closed


@pytest.mark.asyncio
async def test_status_reports_idle_only_when_finishing_and_nothing_is_owed(
    factory, task_id, monkeypatch
):
    monkeypatch.setattr(task_event_bridge, "_bridge", SimpleNamespace(host_id="H"))
    status = await finishing_status_async()
    assert status["active"] is False and status["idle"] is False

    release_hold.begin_finishing()
    status = await finishing_status_async()
    assert status == {
        "active": True,
        "owned_runs": 0,
        "responsibilities": 0,
        "socket_writes": 0,
        "prompt_dispatches": 0,
        "tracked_workers": 0,
        "pending_replies": 0,
        "idle": True,
    }

    register_finishing_counter("channel_turns", lambda: 2)
    status = await finishing_status_async()
    assert status["channel_turns"] == 2 and status["idle"] is False

    def broken() -> int:
        raise RuntimeError("cannot count")

    register_finishing_counter("channel_turns", broken)
    status = await finishing_status_async()
    assert status["channel_turns"] is None and status["idle"] is False


@pytest.mark.asyncio
async def test_status_counts_owned_runs_and_reply_routes(factory, task_id, monkeypatch):
    monkeypatch.setattr(task_event_bridge, "_bridge", SimpleNamespace(host_id="H"))
    release_hold.begin_finishing()
    with factory() as db, db.begin():
        db.add(
            TaskExecutionCommand(
                task_id=task_id,
                command_id="start-1",
                kind="start",
                payload={},
                reply_host_id="H",
                reply_origin="o",
            )
        )
        db.add(
            TaskExecutionCommand(
                task_id=task_id,
                command_id="start-2",
                kind="start",
                payload={},
                reply_host_id="OTHER",
                reply_origin="o",
            )
        )
    status = await finishing_status_async()
    assert status["pending_replies"] == 1 and status["idle"] is False

    with factory() as db, db.begin():
        db.query(TaskExecutionCommand).filter_by(command_id="start-1").update(
            {"status": "completed"}
        )
    assert (await finishing_status_async())["idle"] is True

    monkeypatch.setattr(
        release_hold_module.background_task_manager
        if hasattr(release_hold_module, "background_task_manager")
        else __import__(
            "xagent.web.services.task_execution", fromlist=["background_task_manager"]
        ).background_task_manager,
        "release_counts",
        lambda: {"live_executions": 1},
    )
    status = await finishing_status_async()
    assert status["owned_runs"] == 1 and status["idle"] is False


@pytest.mark.asyncio
async def test_begin_finishing_keeps_the_command_dispatcher(monkeypatch):
    stopped: list[str] = []
    dispatcher = asyncio.create_task(asyncio.sleep(60))

    async def stop_dispatcher() -> None:
        stopped.append("dispatcher")

    async def stop_loop(_app) -> None:
        stopped.append("loop")

    async def stop_channels(*, raise_errors: bool) -> None:
        stopped.append("channels")

    monkeypatch.setattr(app_module, "get_shared_task_execution_enabled", lambda: True)
    monkeypatch.setattr(app_module, "_task_command_dispatcher_task", dispatcher)
    monkeypatch.setattr(transport, "stop_task_command_dispatcher", stop_dispatcher)
    for name in (
        "stop_orphan_upload_gc_task",
        "stop_composio_reconcile_task",
        "stop_composio_missing_account_task",
        "stop_retention_purge_task",
        "stop_task_cleanup_retry_task",
        "stop_uploaded_file_recovery_task",
        "stop_task_auto_resume_task",
        "stop_task_lease_recovery_task",
    ):
        monkeypatch.setattr(app_module, name, stop_loop)
    monkeypatch.setattr(app_module, "_stop_chat_channels", stop_channels)
    try:
        await app_module.begin_finishing()
        await app_module.begin_finishing()  # idempotent
        assert release_hold.finishing
        assert "dispatcher" not in stopped
        assert "channels" in stopped and "loop" in stopped
        assert not dispatcher.done()
        assert app_module._task_command_dispatcher_task is dispatcher
    finally:
        dispatcher.cancel()
        await asyncio.gather(dispatcher, return_exceptions=True)


@pytest.mark.asyncio
async def test_finishing_closes_the_dispatch_cap_so_idle_leases_are_released():
    """The closed cap is what lets a finished run's lease go: queued execution
    commands stop keeping it, so the serving generation claims them."""
    from xagent.web.services.task_dispatch_cap import build_dispatch_cap_snapshot

    assert not build_dispatch_cap_snapshot().held
    release_hold.begin_finishing()
    cap = build_dispatch_cap_snapshot()
    assert cap.held and cap.closed
    assert not cap.admits(123456, interactive=True)


def _compiled(expression) -> str:
    from sqlalchemy.dialects import postgresql

    return str(
        expression.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )


@pytest.mark.parametrize("shared", [True, False])
def test_routing_predicate_is_unchanged_without_finishing(monkeypatch, shared):
    """The normal claim path: exactly the pre-existing predicate."""
    from sqlalchemy import or_

    now = datetime.now(timezone.utc)
    monkeypatch.setattr(transport, "get_shared_task_execution_enabled", lambda: shared)
    assert release_hold.finishing is False
    legacy = [
        Task.runner_id == "r",
        Task.runner_id.is_(None),
        Task.lease_expires_at.is_(None),
        Task.lease_expires_at < now,
    ]
    if not shared:
        legacy = [
            TaskExecutionCommand.target_runner_id.is_(None),
            TaskExecutionCommand.target_runner_id == "r",
            *legacy,
        ]
    assert _compiled(transport._command_routing_predicate("r", now)) == _compiled(
        or_(*legacy)
    )


def test_nothing_changes_until_begin_finishing_is_called():
    assert release_hold.finishing is False
    assert release_hold.closed is False
    status = finishing_status()
    assert status == {"active": False, "pending_replies": 0, "idle": False}
    assert "finishing" not in release_hold.snapshot()
    assert release_hold_module.execution_status()["finishing"]["active"] is False


def test_status_outside_finishing_reads_nothing(monkeypatch):
    def forbidden():
        raise AssertionError("must not read the database outside finish-only mode")

    monkeypatch.setattr(release_hold_module, "_pending_reply_count", forbidden)
    register_finishing_counter("x", forbidden)
    assert finishing_status()["active"] is False


@pytest.mark.parametrize("kind", ["message", "resume_input", "start"])
def test_finishing_still_claims_every_kind_of_an_owned_active_task(
    factory, task_id, kind
):
    """Pinned: follow-ups of an owned active task keep flowing while finishing.

    A run waiting for user input could never finish otherwise; the overlap a
    chatty task causes is bounded by the host action's deadline.
    """
    from xagent.web.services.task_dispatch_cap import (
        DispatchCapSnapshot,
        DispatchPolicy,
    )

    now = datetime.now(timezone.utc)
    with factory() as db, db.begin():
        user_id = db.get(Task, task_id).user_id
        owned = Task(
            user_id=user_id,
            title="owned",
            status=TaskStatus.RUNNING,
            runner_id=OLD,
            lease_attempt_id="o",
            lease_expires_at=now + timedelta(minutes=5),
        )
        db.add(owned)
        db.flush()
        row = TaskExecutionCommand(
            task_id=owned.id, command_id=f"owned-{kind}", kind=kind, payload={}
        )
        db.add(row)
        db.flush()
        row_id, owned_id = row.id, owned.id
    cap = DispatchCapSnapshot(
        policy=DispatchPolicy(limit=4, interactive_reserve=1),
        active_ids=frozenset({owned_id}),
        held=True,
    )
    release_hold._finishing = True
    try:
        with factory() as db:
            claimed = transport._claimable_query(
                db, runner_id=OLD, command_db_id=None, cap=cap
            )
            assert row_id in {int(r.id) for r in claimed.all()}
    finally:
        release_hold._finishing = False
