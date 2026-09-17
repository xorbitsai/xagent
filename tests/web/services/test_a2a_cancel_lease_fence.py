"""An old A2A cancellation cannot settle a replacement lease attempt."""

from types import SimpleNamespace

import pytest

from xagent.web.models.agent import Agent
from xagent.web.models.database import get_session_local, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services import a2a_task_cancel, task_coordinator_service
from xagent.web.services.task_execution_controller import StaleTaskRunError


@pytest.mark.parametrize("fence", ["lock", "update"])
def test_replaced_attempt_rejects_cancel_and_preserves_task(
    tmp_path, monkeypatch, fence
):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    init_db(db_url=f"sqlite:///{tmp_path / 'cancel.db'}")
    with get_session_local()() as db:
        user = User(username="owner", password_hash="unused")
        db.add(user)
        db.flush()
        agent = Agent(user_id=user.id, name="Cancel")
        db.add(agent)
        db.flush()
        task = Task(
            user_id=user.id,
            agent_id=agent.id,
            source="a2a",
            title="Cancel",
            status=TaskStatus.PAUSED,
            control_state="paused",
            run_id="run",
            output="retained output",
        )
        db.add(task)
        db.commit()
        lease = task_coordinator_service.acquire_task_lease_no_commit(
            db, task.id, runner_id="worker"
        )
        db.commit()
        task_id, agent_id, version = task.id, agent.id, task.state_version
        task.lease_attempt_id = "replacement-attempt"
        db.commit()
    monkeypatch.setattr(
        a2a_task_cancel,
        "current_task_coordinator",
        lambda task_id: SimpleNamespace(lease=lease),
    )
    if fence == "update":
        # Isolate the UPDATE predicate even if the earlier lock check succeeded.
        monkeypatch.setattr(
            a2a_task_cancel, "lock_task_lease_no_commit", lambda *a: True
        )
    with pytest.raises(StaleTaskRunError):
        a2a_task_cancel._finalize_a2a_cancel_sync(
            task_id=task_id,
            agent_id=agent_id,
            expected_run_id="run",
            expected_state_version=version,
            local_cancel_requested=False,
        )
    with get_session_local()() as db:
        task = db.get(Task, task_id)
        assert task.status == TaskStatus.PAUSED
        assert task.control_state == "paused"
        assert task.lease_attempt_id == "replacement-attempt"
        assert task.state_version == version
        assert task.output == "retained output"
