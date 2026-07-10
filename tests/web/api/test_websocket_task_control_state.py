from __future__ import annotations

import pytest

from xagent.web.api.websocket import _with_current_task_control_state
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User


@pytest.fixture()
def current_task(tmp_path) -> Task:
    init_db(db_url=f"sqlite:///{tmp_path / 'task-state-events.db'}")
    db = next(get_db())
    try:
        user = User(username="event-user", password_hash="hash", is_admin=False)
        db.add(user)
        db.commit()
        task = Task(
            user_id=user.id,
            title="Event state",
            description="Event state",
            status=TaskStatus.RUNNING,
            execution_mode="auto",
            run_id="run-current",
            state_version=7,
            control_state="running",
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        db.expunge(task)
        yield task
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


@pytest.mark.asyncio
async def test_late_state_event_is_rewritten_to_current_snapshot(current_task) -> None:
    event = await _with_current_task_control_state(
        {
            "type": "task_paused",
            "task_id": int(current_task.id),
            "status": "paused",
        }
    )

    assert event["type"] == "task_paused"
    assert event["run_id"] == "run-current"
    assert event["state_version"] == 7
    assert event["control_state"] == "running"
    assert event["status"] == "running"


@pytest.mark.asyncio
async def test_task_info_trace_gets_versioned_state_tuple(current_task) -> None:
    event = await _with_current_task_control_state(
        {
            "type": "trace_event",
            "event_type": "task_info",
            "task_id": int(current_task.id),
            "data": {"id": int(current_task.id), "status": "paused"},
        }
    )

    assert event["state_version"] == 7
    assert event["data"] == {
        "id": int(current_task.id),
        "status": "running",
        "run_id": "run-current",
        "state_version": 7,
        "control_state": "running",
    }


@pytest.mark.asyncio
async def test_producer_snapshot_is_not_relabelled_as_a_newer_run(current_task) -> None:
    event = await _with_current_task_control_state(
        {
            "type": "task_completed",
            "task_id": int(current_task.id),
            "run_id": "run-old",
            "state_version": 5,
            "control_state": "completed",
            "status": "completed",
            "result": "old result",
        }
    )

    assert event["run_id"] == "run-old"
    assert event["state_version"] == 5
    assert event["control_state"] == "completed"
    assert event["status"] == "completed"


@pytest.mark.asyncio
async def test_boolean_state_version_is_replaced_with_current_snapshot(
    current_task,
) -> None:
    event = await _with_current_task_control_state(
        {
            "type": "task_completed",
            "task_id": int(current_task.id),
            "run_id": "run-old",
            "state_version": True,
            "control_state": "completed",
            "status": "completed",
        }
    )

    assert event["run_id"] == "run-current"
    assert event["state_version"] == 7
    assert event["control_state"] == "running"
    assert event["status"] == "running"
