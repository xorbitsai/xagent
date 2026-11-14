"""Cancelling a paused SDK run stops the automatic resume it waits for.

Uses the automatic-resume fixtures: a paused task resting on ``RUN`` with
its recovery row recorded at ``PV``.
"""

# Pytest fixture imports are intentionally shadowed by test parameters.
# ruff: noqa: F401, F811

from __future__ import annotations

import pytest

from tests.web.services.task_database_shared import engine
from tests.web.services.test_channel_reply_hold import PV, RUN, _seed, factory
from xagent.web.models.agent import Agent
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_auto_recovery import TaskAutoRecovery
from xagent.web.services import sdk_task_cancel, task_auto_resume
from xagent.web.services.sdk_task_cancel import SDK_CANCEL_ERROR_MESSAGE
from xagent.web.services.task_auto_resume import AutoResumeCancel
from xagent.web.services.task_execution_controller import StaleTaskRunError


@pytest.fixture(autouse=True)
def _own_sessions(factory, monkeypatch):
    monkeypatch.setattr(sdk_task_cancel, "get_session_local", lambda: factory)
    monkeypatch.setattr(
        task_auto_resume, "notify_task_command_dispatcher", lambda: None
    )


def _paused_sdk_task(factory, **seed) -> tuple[int, int]:
    with factory() as db:
        task_id = _seed(db, **seed)
        task = db.get(Task, task_id)
        agent = Agent(user_id=task.user_id, name="Paused cancel")
        db.add(agent)
        db.flush()
        task.agent_id = agent.id
        task.source = "sdk"
        db.commit()
        return task_id, int(agent.id)


def _finalize(task_id: int, agent_id: int) -> bool:
    return sdk_task_cancel._finalize_sdk_cancel_sync(
        task_id=task_id,
        agent_id=agent_id,
        expected_run_id=RUN,
        expected_state_version=PV,
        expected_status=TaskStatus.PAUSED,
        terminal_event_state={},
    )


@pytest.mark.parametrize(
    ("state", "final_version", "row_after"),
    [
        # The pending resume is stopped first, which moves the version once.
        ("scheduled", PV + 2, ("manual", "user_cancelled")),
        ("dispatched", PV + 2, ("manual", "user_cancelled")),
        # Nothing is pending: the row is left as it is.
        ("exhausted", PV + 1, ("exhausted", None)),
        (None, PV + 1, None),
    ],
    ids=["scheduled", "dispatched", "gave_up", "no_recovery_row"],
)
def test_paused_run_is_cancelled_and_its_automatic_resume_stopped(
    factory, state, final_version, row_after
):
    task_id, agent_id = _paused_sdk_task(factory, state=state)

    assert _finalize(task_id, agent_id)

    with factory() as db:
        task = db.get(Task, task_id)
        row = db.get(TaskAutoRecovery, task_id)
        assert (task.status, task.control_state) == (TaskStatus.FAILED, "failed")
        assert task.error_message == SDK_CANCEL_ERROR_MESSAGE
        assert (task.run_id, int(task.state_version)) == (RUN, final_version)
        assert (None if row is None else (row.state, row.state_detail)) == row_after
        if row is not None and state in ("scheduled", "dispatched"):
            # Nothing schedules the stopped row again.
            assert row.next_attempt_at is None


def test_a_stop_that_lost_its_race_writes_nothing(factory, monkeypatch):
    task_id, agent_id = _paused_sdk_task(factory, state="scheduled")
    monkeypatch.setattr(
        task_auto_resume,
        "cancel_pending_auto_resume_no_commit",
        lambda *_args, **_kwargs: AutoResumeCancel(cancelled=False, raced=True),
    )

    with pytest.raises(StaleTaskRunError):
        _finalize(task_id, agent_id)

    with factory() as db:
        task = db.get(Task, task_id)
        assert (task.status, int(task.state_version)) == (TaskStatus.PAUSED, PV)
        assert task.error_message is None


def test_a_replay_after_the_resume_was_stopped_finds_its_own_cancel(factory):
    """Stopping the resume moved the version once more, so the cancelled row
    is two past the command's version; a replay still reports it settled
    instead of rejecting a changed run."""
    task_id, agent_id = _paused_sdk_task(factory, state="scheduled")
    assert _finalize(task_id, agent_id)

    assert not _finalize(task_id, agent_id)

    with factory() as db:
        task = db.get(Task, task_id)
        assert (task.error_message, int(task.state_version)) == (
            SDK_CANCEL_ERROR_MESSAGE,
            PV + 2,
        )
