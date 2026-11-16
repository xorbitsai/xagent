"""Finish-only fence gaps found after the first merge.

Each test holds the real code path paused across the fence (or the probe) and
checks what happens when it resumes.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.orm import sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from tests.web.services.task_database_shared import task_id as task_id_fixture
from xagent.web import app as app_module
from xagent.web.models import database
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.services import release_hold as rh_module
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_coordinator_runtime as runtime
from xagent.web.services import task_event_bridge, writer_workers
from xagent.web.services.release_hold import finishing_status_async, release_hold
from xagent.web.services.standby import standby

engine = engine_fixture
task_id = task_id_fixture


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(release_hold, "_closed", False)
    monkeypatch.setattr(release_hold, "_sealed", False)
    monkeypatch.setattr(release_hold, "_finishing", False)
    monkeypatch.setattr(rh_module, "_finishing_counters", {})


@pytest.fixture
def factory(engine, monkeypatch):
    monkeypatch.setattr(transport, "get_shared_task_execution_enabled", lambda: True)
    result = sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(database, "_SessionLocal", result)
    return result


# -- 1. the cap is stale after the waits of a dispatch attempt -----------------


async def _dispatch_start_of_owned_idle_task(
    factory, task_id, monkeypatch, *, fence: str, holds_execution: bool = False
) -> tuple[list[int], dict[int, int]]:
    """A START of a task this process leases but does not run, dispatched."""
    from xagent.web.services import task_coordinator_service as service
    from xagent.web.services import task_execution_admission as admission
    from xagent.web.services.task_execution import background_task_manager as manager

    if fence == "before":
        release_hold.begin_finishing()
    now = datetime.now(timezone.utc)
    with factory() as db, db.begin():
        user_id = db.get(Task, task_id).user_id
        task = Task(
            user_id=user_id,
            title="owned",
            status=TaskStatus.PAUSED,
            runner_id="R",
            lease_attempt_id="a",
            lease_expires_at=now + timedelta(minutes=5),
        )
        db.add(task)
        db.flush()
        row = TaskExecutionCommand(
            task_id=task.id, command_id="s1", kind="start", payload={}
        )
        db.add(row)
        db.flush()
        row_id = row.id
    monkeypatch.setattr(transport, "get_runner_id", lambda: "R")
    monkeypatch.setattr(transport, "consumes_task_commands", lambda: True)
    monkeypatch.setattr(
        runtime,
        "current_task_coordinator",
        lambda _t: SimpleNamespace(lease=SimpleNamespace(runner_id="R")),
    )
    monkeypatch.setattr(service, "lock_task_lease_no_commit", lambda *_a, **_k: True)
    monkeypatch.setattr(admission, "reserve_task_admission", lambda *_a, **_k: True)
    if holds_execution:
        monkeypatch.setattr(manager, "holds_execution", lambda _t: True)

    class Coordinator:
        _command_tasks: set[int] = set()

        async def execute_command(self, _task: Any, apply: Any) -> Any:
            return await apply()

        def require_recovery(self) -> None:
            return None

    class Registry:
        def busy_command_tasks(self) -> list[int]:
            return []

        async def ensure(self, _task_id: int) -> Coordinator:
            if fence == "during":
                release_hold.begin_finishing()  # lands while ensure waits
            return Coordinator()

    monkeypatch.setattr(runtime, "get_task_coordinator_registry", lambda: Registry())
    applied: list[int] = []

    async def executor(command: Any) -> Any:
        applied.append(command.id)
        return None

    try:
        await transport._dispatch_shared_task_command(executor, command_db_id=row_id)
    except AttributeError:
        pass  # the stub executor returns no result; only the claim matters
    return applied, dict(manager._dispatch_reservations)


@pytest.mark.asyncio
@pytest.mark.parametrize("fence", ["during", "before"])
async def test_start_of_an_idle_owned_task_is_not_claimed_after_the_fence(
    factory, task_id, monkeypatch, fence
) -> None:
    applied, reservations = await _dispatch_start_of_owned_idle_task(
        factory, task_id, monkeypatch, fence=fence
    )
    assert applied == [] and reservations == {}


@pytest.mark.asyncio
async def test_start_is_still_claimed_without_finish_only_mode(
    factory, task_id, monkeypatch
) -> None:
    """Pinned: nothing changes until begin_finishing is called."""
    applied, _ = await _dispatch_start_of_owned_idle_task(
        factory, task_id, monkeypatch, fence="never"
    )
    assert len(applied) == 1


@pytest.mark.asyncio
async def test_follow_up_of_a_task_this_process_still_runs_is_claimed(
    factory, task_id, monkeypatch
) -> None:
    applied, _ = await _dispatch_start_of_owned_idle_task(
        factory, task_id, monkeypatch, fence="during", holds_execution=True
    )
    assert len(applied) == 1


# -- 2. work that came and went inside the probe window ------------------------


async def _probe_paused_across(monkeypatch, during):
    real = rh_module._pending_reply_count
    in_probe, proceed = threading.Event(), threading.Event()

    def probe() -> int:
        count = real()  # the snapshot: nothing owed yet
        in_probe.set()
        proceed.wait(10)
        return count

    monkeypatch.setattr(rh_module, "_pending_reply_count", probe)
    status = asyncio.create_task(finishing_status_async())
    await asyncio.to_thread(in_probe.wait, 5)
    during()
    proceed.set()
    return await asyncio.wait_for(status, 5)


@pytest.mark.asyncio
async def test_idle_is_false_when_a_reply_was_bound_and_released_inside_the_probe(
    factory, task_id, monkeypatch
) -> None:
    monkeypatch.setattr(task_event_bridge, "_bridge", SimpleNamespace(host_id="H"))
    release_hold.begin_finishing()

    def during() -> None:
        write = release_hold.acquire_socket_write(None)
        with factory() as db, db.begin():  # commits after the probe's snapshot
            db.add(
                TaskExecutionCommand(
                    task_id=task_id,
                    command_id="p1",
                    kind="pause",
                    payload={},
                    reply_host_id="H",
                    reply_origin="o",
                )
            )
        write.release()

    status = await _probe_paused_across(monkeypatch, during)
    assert status["pending_replies"] == 0 and status["socket_writes"] == 0
    assert status["idle"] is False


@pytest.mark.asyncio
async def test_idle_is_false_when_a_tracked_worker_ran_and_ended_inside_the_probe(
    factory, task_id, monkeypatch
) -> None:
    monkeypatch.setattr(task_event_bridge, "_bridge", SimpleNamespace(host_id="H"))
    release_hold.begin_finishing()
    before = writer_workers.worker_starts()

    def during() -> None:
        writer_workers.start_tracked_thread(lambda: None).join(5)

    status = await _probe_paused_across(monkeypatch, during)
    assert writer_workers.worker_starts() == before + 1
    assert status["tracked_workers"] == 0 and status["idle"] is False


@pytest.mark.asyncio
async def test_idle_is_true_when_nothing_moved_during_the_probe(
    factory, task_id, monkeypatch
) -> None:
    monkeypatch.setattr(task_event_bridge, "_bridge", SimpleNamespace(host_id="H"))
    release_hold.begin_finishing()
    status = await _probe_paused_across(monkeypatch, lambda: None)
    assert status["idle"] is True


# -- 3. a standby does not begin finishing -------------------------------------


@pytest.mark.asyncio
async def test_begin_finishing_is_refused_on_a_standby_and_changes_nothing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(app_module, "get_shared_task_execution_enabled", lambda: True)
    standby.reset()
    standby.begin()
    try:
        with pytest.raises(RuntimeError, match="standby"):
            await app_module.begin_finishing()
        assert release_hold.finishing is False and release_hold.closed is False
    finally:
        standby.reset()


@pytest.mark.asyncio
async def test_the_dispatchers_polling_tokens_do_not_make_idle_flap(
    factory, task_id, monkeypatch
) -> None:
    monkeypatch.setattr(task_event_bridge, "_bridge", SimpleNamespace(host_id="H"))
    release_hold.begin_finishing()

    def during() -> None:
        for kind in ("command_attempt", "command_dispatch"):
            token = release_hold.try_begin_attempt(kind, "n")
            assert token is not None
            token.release()

    status = await _probe_paused_across(monkeypatch, during)
    assert status["idle"] is True
