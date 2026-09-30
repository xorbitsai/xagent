"""Semantic completion survives settlement, reconnect, and later turns."""

import pytest
from sqlalchemy.orm import sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from tests.web.services.task_database_shared import task_id as task_id_fixture
from xagent.web.models.task import Task, TaskStatus
from xagent.web.services import task_execution as execution
from xagent.web.services import task_lease_service as leases
from xagent.web.services import task_stream_snapshot as snapshots
from xagent.web.services.task_execution_controller import (
    TaskControlState,
    apply_task_control_transition,
)

engine = engine_fixture
task_id = task_id_fixture


@pytest.mark.parametrize("route", ["initial", "resume"])
@pytest.mark.parametrize("storage_version", [1, 2])
@pytest.mark.parametrize(
    "outcome", ["completed", "partial", "blocked", None, "invalid"]
)
def test_settled_outcome_and_next_run(
    engine, task_id, monkeypatch, route, storage_version, outcome
):
    factory = sessionmaker(engine)
    monkeypatch.setattr(execution, "get_session_local", lambda: factory)
    monkeypatch.setattr(snapshots, "get_session_local", lambda: factory)
    with factory() as db:
        task = db.get(Task, task_id)
        task.conversation_storage_version = storage_version
        uid = task.user_id
        db.commit()
        lease = leases.acquire_task_lease(
            db, task_id, runner_id="outcome-test", new_run=True
        )
    result = {
        "status": "completed",
        "success": True,
        "output": "Delivered answer",
        "completion_outcome": outcome,
    }
    empty = execution._PreparedTaskFileOutputs((), (), ())
    if route == "initial":
        finalized = execution._finalize_task_execution_result_isolated(
            task_id=task_id,
            task_user_id=uid,
            pre_run_status=TaskStatus.RUNNING,
            result=result,
            expected_run_id=lease.run_id,
            task_lease=lease,
            resolved_scope_segments=(),
            prepared_outputs=empty,
        )
        transported = finalized.broadcast_meta["completion_outcome"]
    else:
        finalized = execution._finalize_resumed_task(
            task_id,
            status="completed",
            success=True,
            output=result["output"],
            task_owner_user_id=uid,
            result=result,
            task_lease=lease,
            prepared_outputs=empty,
        )
        transported = finalized["completion_outcome"]
    expected = None if outcome == "invalid" else outcome
    assert transported == expected
    with factory() as db:
        task = db.get(Task, task_id)
        assert task.status == TaskStatus.COMPLETED
        assert task.completion_outcome == expected
        assert task.output == "Delivered answer"
    assert (
        snapshots.load_task_stream_snapshots([task_id])[0]["completion_outcome"]
        == expected
    )
    with factory() as db:
        task = db.get(Task, task_id)
        apply_task_control_transition(
            task, TaskControlState.RUNNING, status=TaskStatus.RUNNING, new_run=True
        )
        db.commit()
    assert (
        snapshots.load_task_stream_snapshots([task_id])[0]["completion_outcome"] is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["completed", "partial", "blocked", None])
async def test_history_and_rest_read_same_outcome(
    engine, task_id, monkeypatch, outcome
):
    from xagent.web.api import chat, websocket
    from xagent.web.models import database
    from xagent.web.models.user import User

    factory = sessionmaker(engine)

    def get_db():
        with factory() as db:
            yield db

    monkeypatch.setattr(database, "get_db", get_db)
    # No cross-test response cache; the production cache is keyed by updated_at.
    monkeypatch.setattr(chat, "cache_get", lambda key: None)
    monkeypatch.setattr(websocket, "cache_get", lambda key: None)
    with factory() as db:
        task = db.get(Task, task_id)
        task.status = TaskStatus.COMPLETED
        task.completion_outcome = outcome
        uid = task.user_id
        db.commit()
        owner = db.get(User, uid)
        assert (await chat.get_task(task_id, db, owner))[
            "completion_outcome"
        ] == outcome
        assert (await chat.get_task_status(task_id, db, owner))[
            "completion_outcome"
        ] == outcome
    history = websocket._load_historical_stream_snapshot_sync(
        task_id, actor_user_id=uid, actor_is_admin=False
    )
    info = next(
        event for event in history.events if event.get("event_type") == "task_info"
    )
    assert info["data"]["completion_outcome"] == outcome


@pytest.mark.parametrize(
    "status",
    [
        TaskStatus.RUNNING,
        TaskStatus.PAUSED,
        TaskStatus.WAITING_FOR_USER,
        TaskStatus.FAILED,
    ],
)
def test_noncompleted_transition_clears_outcome(status):
    task = Task(id=1, status=TaskStatus.COMPLETED, completion_outcome="partial")
    apply_task_control_transition(task, TaskControlState.RUNNING, status=status)
    assert task.completion_outcome is None
