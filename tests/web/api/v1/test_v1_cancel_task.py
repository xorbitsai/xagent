"""Integration tests for ``POST /v1/chat/tasks/{task_id}/cancel``.

The route's contract: which task states it cancels, refuses for good, or
asks the caller to retry; that a refusal writes nothing; and that the
``error_code`` it reports is the one the task reads report.
"""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.services import sdk_task_cancel, task_auto_resume
from xagent.web.services import task_command_execution as command_execution_service
from xagent.web.services.sdk_task_cancel import SDK_CANCEL_ERROR_MESSAGE
from xagent.web.services.task_command_transport import (
    COMMAND_COMPLETED,
    TaskCommandKind,
    TaskCommandRejected,
    TaskCommandTaskMissing,
    stage_task_command,
)
from xagent.web.services.task_execution_controller import TaskControlState
from xagent.web.services.task_start import SdkTaskScope
from xagent.web.services.task_start_protocol import TaskStartPayload

from ..conftest import (
    _admin_headers,
    _direct_db_session,
    _register_second_user,
    client,
)
from .test_workforces import _create_active_workforce, _create_workforce_key

pytestmark = pytest.mark.usefixtures("_test_db")

_agent_name_counter = 0

_ROW_COLUMNS = ("status", "control_state", "run_id", "state_version", "error_message")


def _bearer(full_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {full_key}"}


@pytest.fixture(autouse=True)
def no_background_run():
    """Tasks are accepted and left RUNNING with no coroutine behind them."""
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ):
        yield


@pytest.fixture(autouse=True)
def no_live_broadcast(monkeypatch: pytest.MonkeyPatch):
    from xagent.web.api import websocket as websocket_api

    manager = MagicMock()
    manager.broadcast_to_task = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", manager)


def _create_agent_with_key(headers: dict[str, str] | None = None) -> tuple[int, str]:
    global _agent_name_counter
    _agent_name_counter += 1
    headers = headers or _admin_headers()
    agent_resp = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": f"v1 cancel test agent {_agent_name_counter}",
            "description": "test",
            "instructions": "you are a test agent",
            "execution_mode": "balanced",
        },
    )
    assert agent_resp.status_code == 200, agent_resp.text
    agent_id = agent_resp.json()["id"]
    key_resp = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)
    assert key_resp.status_code == 200, key_resp.text
    return agent_id, key_resp.json()["full_key"]


def _create_task(full_key: str, agent_id: int) -> int:
    response = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={"agent_id": agent_id, "message": {"role": "user", "content": "hello"}},
    )
    assert response.status_code == 202, response.text
    return response.json()["task_id"]


def _running_task() -> tuple[int, int, str]:
    agent_id, key = _create_agent_with_key()
    task_id = _create_task(key, agent_id)
    assert _row(task_id)["status"] == TaskStatus.RUNNING
    return task_id, agent_id, key


def _update(task_id: int, **values: Any) -> None:
    db = _direct_db_session()
    try:
        db.query(Task).filter(Task.id == task_id).update(values)
        db.commit()
    finally:
        db.close()


def _row(task_id: int) -> dict[str, Any]:
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        return {column: getattr(task, column) for column in _ROW_COLUMNS}
    finally:
        db.close()


def _cancel_commands(task_id: int) -> list[TaskExecutionCommand]:
    db = _direct_db_session()
    try:
        return (
            db.query(TaskExecutionCommand)
            .filter(
                TaskExecutionCommand.task_id == task_id,
                TaskExecutionCommand.kind == TaskCommandKind.CANCEL.value,
            )
            .all()
        )
    finally:
        db.close()


QUEUED_RUN = "run-queued"


def _queue_start(
    task_id: int, *, run_id: str = QUEUED_RUN, turn_id: str = "turn-queued"
) -> None:
    """Leave an accepted START that no executor has handed off yet.

    Its payload is the one the acceptance writes: the run and version the row
    holds now, which is what the START consumer compares the row with.
    """
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        start = TaskStartPayload(
            version=1,
            run_id=run_id,
            state_version=int(task.state_version or 0),
            expected_run_id=task.run_id,
            turn_id=turn_id,
            kind="create" if task.status == TaskStatus.PENDING else "append",
            message="hello",
        )
        stage_task_command(
            db,
            task_id=task_id,
            actor_user_id=int(task.user_id),
            command_id=start.turn_id,
            kind=TaskCommandKind.START,
            payload=start.model_dump(mode="json"),
            target_run_id=start.run_id,
        )
        db.commit()
    finally:
        db.close()


def _stop_queued_start(task_id: int) -> None:
    db = _direct_db_session()
    try:
        db.query(TaskExecutionCommand).filter(
            TaskExecutionCommand.task_id == task_id,
            TaskExecutionCommand.kind == TaskCommandKind.START.value,
        ).update(
            {
                "status": "failed",
                "result": {"rejection_reason": "cancelled_before_admission"},
            }
        )
        db.commit()
    finally:
        db.close()


def _cancel(task_id: int, key: str):
    return client.post(f"/v1/chat/tasks/{task_id}/cancel", headers=_bearer(key))


def _get(task_id: int, key: str) -> dict[str, Any]:
    response = client.get(f"/v1/chat/tasks/{task_id}", headers=_bearer(key))
    assert response.status_code == 200, response.text
    return response.json()


def _error(response) -> str:
    return response.json()["error"]["code"]


def _completed_frame(task_id: int, key: str) -> dict[str, Any]:
    response = client.get(f"/v1/chat/tasks/{task_id}/events", headers=_bearer(key))
    assert response.status_code == 200, response.text
    blocks = [block for block in response.text.split("\n\n") if block.strip()]
    for block in blocks:
        lines = block.splitlines()
        if "event: task.completed" in lines:
            data = next(line for line in lines if line.startswith("data: "))
            return json.loads(data[len("data: ") :])
    raise AssertionError(response.text)


def test_cancel_running_task_then_repeat_then_append():
    """The whole caller-visible arc of one cancellation."""
    task_id, agent_id, key = _running_task()
    before = _row(task_id)

    first = _cancel(task_id, key)

    assert first.status_code == 200, first.text
    assert first.json() == {
        "task_id": task_id,
        "status": "failed",
        "run_id": before["run_id"],
        "state_version": before["state_version"] + 1,
        "cancelled_run_id": before["run_id"],
        "error_code": "task_cancelled",
    }
    read = _get(task_id, key)
    assert read["status"] == "failed"
    assert read["error"] == SDK_CANCEL_ERROR_MESSAGE
    assert read["error_code"] == "task_cancelled"
    assert read["output"] is None
    assert _completed_frame(task_id, key)["error_code"] == "task_cancelled"

    # A repeat reports the same outcome and adds no second command.
    again = _cancel(task_id, key)
    assert again.status_code == 200, again.text
    assert again.json() == first.json()
    assert len(_cancel_commands(task_id)) == 1

    # A cancelled task takes a new turn like any failed one, and the code
    # goes with the run it described.
    appended = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(key),
        json={"agent_id": agent_id, "message": {"role": "user", "content": "again"}},
    )
    assert appended.status_code == 202, appended.text
    read = _get(task_id, key)
    assert read["status"] == "running"
    assert read["error_code"] is None


@pytest.mark.parametrize(
    ("values", "expected_code"),
    [
        (
            # No START is queued for it, so there is no turn to aim at.
            {
                "status": TaskStatus.COMPLETED,
                "control_state": TaskControlState.COMPLETED.value,
                "output": "done",
            },
            "task_not_cancellable",
        ),
        (
            {
                "status": TaskStatus.FAILED,
                "control_state": TaskControlState.FAILED.value,
                "error_message": "boom",
            },
            "task_not_cancellable",
        ),
        (
            {"status": TaskStatus.PENDING, "control_state": "idle"},
            "task_busy",
        ),
        (
            {
                "status": TaskStatus.WAITING_FOR_USER,
                "control_state": TaskControlState.RESUME_REQUESTED.value,
            },
            "task_busy",
        ),
    ],
    ids=["completed", "failed_itself", "pending", "answer_in_flight"],
)
def test_states_with_nothing_to_cancel_are_refused_without_a_write(
    values, expected_code
):
    task_id, _agent_id, key = _running_task()
    _update(task_id, runner_id=None, lease_expires_at=None, **values)
    before = _row(task_id)

    response = _cancel(task_id, key)

    assert response.status_code == 409, response.text
    assert _error(response) == expected_code
    assert _row(task_id) == before
    assert _cancel_commands(task_id) == []


@pytest.mark.parametrize(
    "previous",
    [
        {
            "status": TaskStatus.FAILED,
            "control_state": TaskControlState.FAILED.value,
            "error_message": SDK_CANCEL_ERROR_MESSAGE,
        },
        {
            "status": TaskStatus.COMPLETED,
            "control_state": TaskControlState.COMPLETED.value,
            "output": "done",
        },
        {
            "status": TaskStatus.PAUSED,
            "control_state": TaskControlState.PAUSED.value,
        },
    ],
    ids=["previous_run_cancelled", "previous_run_completed", "previous_run_paused"],
)
def test_a_queued_turn_is_the_target_not_the_run_the_row_still_shows(
    monkeypatch: pytest.MonkeyPatch, previous
):
    """While an appended START is queued the row still shows the run before
    it. The request aims at the queued run; it is never answered from the
    row, so a cancelled previous run does not read as "already cancelled".
    The status and code read as the task read shows them: ``pending`` after
    a completed or failed run, ``paused`` after a paused one."""
    task_id, _agent_id, key = _running_task()
    _update(task_id, runner_id=None, lease_expires_at=None, **previous)
    _queue_start(task_id)
    # No executor takes the queued START here, so the command behind it waits.
    monkeypatch.setattr(sdk_task_cancel, "dispatch_one_task_command", AsyncMock())
    monkeypatch.setattr(sdk_task_cancel, "SDK_CANCEL_REQUEST_WAIT_SECONDS", 0.1)
    before = _row(task_id)

    response = _cancel(task_id, key)

    assert response.status_code == 202, response.text
    task_read = _get(task_id, key)
    assert response.json() == {
        "task_id": task_id,
        "status": task_read["status"],
        "run_id": before["run_id"],
        "state_version": before["state_version"],
        "cancelled_run_id": None,
        "error_code": task_read["error_code"],
    }
    assert task_read["status"] == (
        "paused" if before["status"] == TaskStatus.PAUSED else "pending"
    )
    assert _row(task_id) == before
    (command,) = _cancel_commands(task_id)
    assert command.payload["queued_run_id"] == QUEUED_RUN
    assert command.target_run_id == QUEUED_RUN
    assert command.command_id.endswith(f":{QUEUED_RUN}")

    # Once the core has stopped the START, the same request reports that
    # turn as cancelled while the task still reads as the run before it.
    _stop_queued_start(task_id)
    monkeypatch.setattr(sdk_task_cancel, "_queued_start", lambda _db, _task_id: None)
    result = sdk_task_cancel._load_result_sync(
        task_id=task_id,
        scope=SdkTaskScope(agent_id=_agent_id, workforce_id=None),
        aimed_run_id=QUEUED_RUN,
        aimed_at_queued_turn=True,
        command_completed=False,
    )
    assert result.cancelled and result.cancelled_run_id == QUEUED_RUN
    assert result.status == before["status"].value
    assert result.run_id == before["run_id"]
    assert result.error_code == (
        "task_cancelled"
        if before["error_message"] == SDK_CANCEL_ERROR_MESSAGE
        else None
    )
    assert _row(task_id) == before


def test_a_queued_new_task_cancelled_by_its_command_answers_for_the_queued_run(
    monkeypatch: pytest.MonkeyPatch,
):
    """The command completed against a new task that never ran: the answer
    names the queued run as cancelled while the task has no run of its own,
    and every other field reads as the task read does."""
    task_id, _agent_id, key = _running_task()
    _update(
        task_id,
        status=TaskStatus.PENDING,
        control_state="idle",
        run_id=None,
        runner_id=None,
        lease_expires_at=None,
    )
    _queue_start(task_id)
    before = _row(task_id)

    # A cancel is dispatched past a START only while that START waits for
    # admission; this stands in for that dispatch. The core's write is the
    # terminal row, and the dispatcher's settlement of the cancelled
    # admissions then fails the START.
    async def cancelled_as_a_row(_executor: Any, *, command_db_id: int) -> bool:
        _update(
            task_id,
            state_version=before["state_version"] + 1,
            **_cancelled_row_values(),
        )
        _stop_queued_start(task_id)
        _settle_cancel_command(command_db_id, status="completed")
        return True

    monkeypatch.setattr(
        sdk_task_cancel, "dispatch_one_task_command", cancelled_as_a_row
    )

    response = _cancel(task_id, key)

    assert response.status_code == 200, response.text
    assert response.json() == {
        "task_id": task_id,
        "status": "failed",
        "run_id": None,
        "state_version": before["state_version"] + 1,
        "cancelled_run_id": QUEUED_RUN,
        "error_code": "task_cancelled",
    }
    task_read = _get(task_id, key)
    assert (task_read["status"], task_read["error_code"]) == (
        "failed",
        "task_cancelled",
    )
    assert task_read["run_id"] is None


def test_a_queued_turn_whose_cancel_went_stale_while_it_waits_is_busy(
    monkeypatch: pytest.MonkeyPatch,
):
    """The command was rejected because the task moved, and the aimed START
    is still waiting: nothing was cancelled and a retry can succeed."""
    task_id, _agent_id, key = _running_task()
    _update(
        task_id,
        status=TaskStatus.COMPLETED,
        control_state=TaskControlState.COMPLETED.value,
        runner_id=None,
        lease_expires_at=None,
    )
    _queue_start(task_id)
    before = _row(task_id)

    async def went_stale(_executor: Any, *, command_db_id: int) -> bool:
        _settle_cancel_command(
            command_db_id, status="failed", rejection_reason="stale_run"
        )
        return True

    monkeypatch.setattr(sdk_task_cancel, "dispatch_one_task_command", went_stale)

    response = _cancel(task_id, key)

    assert response.status_code == 409, response.text
    assert _error(response) == "task_busy"
    assert _row(task_id) == before


def test_a_queued_turn_that_failed_to_start_is_not_reported_from_the_previous_run(
    monkeypatch: pytest.MonkeyPatch,
):
    """The aimed turn never started and was not stopped by this cancel; the
    row is still the cancelled run before it, which is not the answer. The
    aimed turn is over, so a retry -- which would report that earlier run --
    is not invited."""
    task_id, _agent_id, key = _running_task()
    _update(
        task_id,
        status=TaskStatus.FAILED,
        control_state=TaskControlState.FAILED.value,
        error_message=SDK_CANCEL_ERROR_MESSAGE,
        runner_id=None,
        lease_expires_at=None,
    )
    _queue_start(task_id)

    async def start_fails_and_the_cancel_goes_stale(
        _executor: Any, *, command_db_id: int
    ) -> bool:
        db = _direct_db_session()
        try:
            commands = db.query(TaskExecutionCommand).filter(
                TaskExecutionCommand.task_id == task_id
            )
            commands.filter(
                TaskExecutionCommand.kind == TaskCommandKind.START.value
            ).update(
                {"status": "failed", "result": {"rejection_reason": "invalid_start"}}
            )
            commands.filter(TaskExecutionCommand.id == command_db_id).update(
                {"status": "failed", "result": {"rejection_reason": "stale_run"}}
            )
            db.commit()
        finally:
            db.close()
        return True

    monkeypatch.setattr(
        sdk_task_cancel,
        "dispatch_one_task_command",
        start_fails_and_the_cancel_goes_stale,
    )

    response = _cancel(task_id, key)

    assert response.status_code == 409, response.text
    assert _error(response) == "task_not_cancellable"
    assert (
        response.json()["error"]["message"]
        == "The run this cancel was aimed at has already ended."
    )


def test_a_task_that_reads_failed_after_automatic_resume_gave_up_is_not_cancellable(
    monkeypatch: pytest.MonkeyPatch,
):
    task_id, _agent_id, key = _running_task()
    _update(
        task_id,
        status=TaskStatus.PAUSED,
        control_state=TaskControlState.PAUSED.value,
        runner_id=None,
        lease_expires_at=None,
    )
    monkeypatch.setattr(
        task_auto_resume,
        "status_only_terminal_view",
        lambda _db, _task_id: SimpleNamespace(state="exhausted"),
    )
    before = _row(task_id)

    response = _cancel(task_id, key)

    assert response.status_code == 409, response.text
    assert _error(response) == "task_not_cancellable"
    assert _row(task_id) == before
    assert _cancel_commands(task_id) == []


@pytest.mark.parametrize(
    "ended", ["automatic_resume_gave_up", "gave_up_after_the_row_moved", "completed"]
)
def test_a_target_that_ends_while_the_cancel_is_still_unapplied_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch, ended: str
):
    """The command is recorded against a paused run and still pending at the
    deadline. A paused run whose automatic resume gave up meanwhile reads
    ``failed`` but stays at the version the command is fenced on, and the
    command still cancels it, so the answer is 202 and calling again rejoins
    the same command. Once the row has moved past that version the command
    no longer applies, and a run that completed on its own is over: both
    answer 409."""
    task_id, _agent_id, key = _running_task()
    _update(
        task_id,
        status=TaskStatus.PAUSED,
        control_state=TaskControlState.PAUSED.value,
        runner_id=None,
        lease_expires_at=None,
    )
    before = _row(task_id)
    gave_up: list[bool] = []
    monkeypatch.setattr(
        task_auto_resume,
        "status_only_terminal_view",
        lambda _db, _task_id: SimpleNamespace(state="exhausted") if gave_up else None,
    )

    async def ends_without_applying(_executor: Any, *, command_db_id: int) -> bool:
        if gave_up:
            return False
        if ended == "completed":
            _update(
                task_id,
                status=TaskStatus.COMPLETED,
                control_state=TaskControlState.COMPLETED.value,
                output="done",
            )
            return False
        gave_up.append(True)
        if ended == "gave_up_after_the_row_moved":
            _update(task_id, state_version=before["state_version"] + 1)
        return False

    dispatched = AsyncMock(side_effect=ends_without_applying)
    monkeypatch.setattr(sdk_task_cancel, "dispatch_one_task_command", dispatched)
    monkeypatch.setattr(sdk_task_cancel, "SDK_CANCEL_REQUEST_WAIT_SECONDS", 0.1)

    response = _cancel(task_id, key)

    (command,) = _cancel_commands(task_id)
    assert command.status == "pending"
    if ended != "automatic_resume_gave_up":
        assert response.status_code == 409, response.text
        assert _error(response) == "task_not_cancellable"
        return
    assert response.status_code == 202, response.text
    assert response.json() == {
        "task_id": task_id,
        "status": "failed",
        "run_id": before["run_id"],
        "state_version": before["state_version"],
        "cancelled_run_id": None,
        "error_code": None,
    }
    assert _row(task_id) == before

    again = _cancel(task_id, key)

    assert again.status_code == 202, again.text
    assert again.json() == response.json()
    assert dispatched.await_args.kwargs["command_db_id"] == command.id
    assert [c.id for c in _cancel_commands(task_id)] == [command.id]
    assert _row(task_id) == before
    # The command is then applied as the core applies it.
    _update(
        task_id,
        status=TaskStatus.FAILED,
        control_state=TaskControlState.FAILED.value,
        state_version=before["state_version"] + 1,
        error_message=SDK_CANCEL_ERROR_MESSAGE,
    )
    _settle_cancel_command(command.id, status="completed")

    applied = _cancel(task_id, key)

    assert applied.status_code == 200, applied.text
    assert applied.json()["cancelled_run_id"] == before["run_id"]
    assert applied.json()["error_code"] == "task_cancelled"
    assert len(_cancel_commands(task_id)) == 1


@pytest.mark.parametrize("case", ["other_agent", "other_user", "missing", "non_sdk"])
def test_tasks_outside_the_key_scope_are_404_and_untouched(case):
    task_id, _agent_id, key = _running_task()
    target = task_id
    if case == "other_agent":
        _other_agent, key = _create_agent_with_key()
    elif case == "other_user":
        headers = _register_second_user("mallory", "mallorypass1")
        _other_agent, key = _create_agent_with_key(headers)
    elif case == "missing":
        target = task_id + 10_000
    else:
        _update(task_id, source="internal")
    before = _row(task_id)

    response = _cancel(target, key)

    assert response.status_code == 404, response.text
    assert _error(response) == "task_not_found"
    assert _row(task_id) == before
    assert _cancel_commands(task_id) == []


def test_workforce_key_cancels_its_own_run():
    headers = _admin_headers()
    workforce_id = _create_active_workforce(headers, name="Cancel WF")
    key = _create_workforce_key(headers, workforce_id)
    run = client.post(
        f"/v1/workforces/{workforce_id}/runs",
        headers=_bearer(key),
        json={"message": {"role": "user", "content": "go"}},
    )
    assert run.status_code == 202, run.text
    task_id = run.json()["task_id"]

    response = _cancel(task_id, key)

    assert response.status_code == 200, response.text
    assert response.json()["error_code"] == "task_cancelled"
    assert _get(task_id, key)["error_code"] == "task_cancelled"


def test_a_failure_that_was_not_a_cancellation_carries_no_code():
    task_id, _agent_id, key = _running_task()
    _update(
        task_id,
        status=TaskStatus.FAILED,
        control_state=TaskControlState.FAILED.value,
        error_message="task execution cancelled",
        runner_id=None,
        lease_expires_at=None,
    )

    assert _get(task_id, key)["error_code"] is None
    assert _completed_frame(task_id, key)["error_code"] is None


def test_an_unapplied_cancel_is_accepted_and_repeats_reach_the_same_command(
    monkeypatch: pytest.MonkeyPatch,
):
    task_id, _agent_id, key = _running_task()
    monkeypatch.setattr(sdk_task_cancel, "dispatch_one_task_command", AsyncMock())
    monkeypatch.setattr(sdk_task_cancel, "SDK_CANCEL_REQUEST_WAIT_SECONDS", 0.1)
    before = _row(task_id)

    first = _cancel(task_id, key)
    second = _cancel(task_id, key)

    for response in (first, second):
        assert response.status_code == 202, response.text
        assert response.json() == {
            "task_id": task_id,
            "status": "running",
            "run_id": before["run_id"],
            "state_version": before["state_version"],
            "cancelled_run_id": None,
            "error_code": None,
        }
    assert _row(task_id) == before
    assert len(_cancel_commands(task_id)) == 1


@pytest.mark.parametrize(
    ("reason", "status_code", "code"),
    [
        ("stale_run", 409, "task_busy"),
        ("unsupported_scope", 503, "internal_error"),
        (None, 503, "internal_error"),
    ],
)
def test_a_rejected_command_tells_the_caller_whether_to_retry(
    monkeypatch: pytest.MonkeyPatch, reason, status_code, code
):
    task_id, _agent_id, key = _running_task()
    before = _row(task_id)
    rejecting = AsyncMock(side_effect=TaskCommandRejected("rejected", reason=reason))
    monkeypatch.setattr(
        command_execution_service, "cancel_sdk_task_unserialized", rejecting
    )

    response = _cancel(task_id, key)

    assert response.status_code == status_code, response.text
    assert _error(response) == code
    # The command row's own error text stays on the server.
    assert "rejected" not in response.text
    assert _row(task_id) == before

    # The failed command is put back in the queue by the next call, which
    # then applies it.
    monkeypatch.undo()
    retried = _cancel(task_id, key)
    assert retried.status_code == 200, retried.text
    assert len(_cancel_commands(task_id)) == 1


def test_a_task_deleted_before_the_command_is_stored_is_404(
    monkeypatch: pytest.MonkeyPatch,
):
    task_id, _agent_id, key = _running_task()
    monkeypatch.setattr(
        sdk_task_cancel,
        "enqueue_task_command",
        MagicMock(side_effect=TaskCommandTaskMissing(f"Task {task_id} not found")),
    )

    response = _cancel(task_id, key)

    assert response.status_code == 404, response.text
    assert _error(response) == "task_not_found"


_ENDED_ROWS = {
    "completed": {
        "status": TaskStatus.COMPLETED,
        "control_state": TaskControlState.COMPLETED.value,
        "output": "done",
    },
    "failed_itself": {
        "status": TaskStatus.FAILED,
        "control_state": TaskControlState.FAILED.value,
        "error_message": "boom",
    },
}


def _cancelled_row_values() -> dict[str, Any]:
    return {
        "status": TaskStatus.FAILED,
        "control_state": TaskControlState.FAILED.value,
        "error_message": SDK_CANCEL_ERROR_MESSAGE,
    }


def _settle_cancel_command(
    command_db_id: int, *, status: str, rejection_reason: str | None = None
) -> None:
    """End a CANCEL command the way the dispatcher records its outcome."""
    db = _direct_db_session()
    try:
        db.query(TaskExecutionCommand).filter(
            TaskExecutionCommand.id == command_db_id
        ).update(
            {
                "status": status,
                "result": (
                    {"rejection_reason": rejection_reason}
                    if rejection_reason is not None
                    else {}
                ),
            }
        )
        db.commit()
    finally:
        db.close()


def _replace_cancel_core(monkeypatch: pytest.MonkeyPatch, core: Any) -> None:
    monkeypatch.setattr(command_execution_service, "cancel_sdk_task_unserialized", core)


@pytest.mark.parametrize("ended", list(_ENDED_ROWS))
@pytest.mark.parametrize("exit_path", ["stale_run", "deadline"])
def test_a_run_that_ended_on_its_own_is_never_reported_as_busy_or_accepted(
    monkeypatch: pytest.MonkeyPatch, ended: str, exit_path: str
):
    """Whichever way the wait ends, a run that ended without a cancel
    answers what the request would have answered on arrival."""
    task_id, _agent_id, key = _running_task()
    values = {**_ENDED_ROWS[ended], "runner_id": None, "lease_expires_at": None}
    after_the_run_ended: list[dict[str, Any]] = []

    def end_the_run() -> None:
        _update(task_id, **values)
        after_the_run_ended.append(_row(task_id))

    if exit_path == "stale_run":

        async def ended_before_it_was_applied(**_kwargs: Any) -> None:
            end_the_run()
            raise TaskCommandRejected("moved", reason="stale_run")

        _replace_cancel_core(monkeypatch, ended_before_it_was_applied)
    else:

        async def ended_while_the_command_waits(*_args: Any, **_kwargs: Any) -> bool:
            end_the_run()
            return False

        monkeypatch.setattr(
            sdk_task_cancel,
            "dispatch_one_task_command",
            AsyncMock(side_effect=ended_while_the_command_waits),
        )
        monkeypatch.setattr(sdk_task_cancel, "SDK_CANCEL_REQUEST_WAIT_SECONDS", 0.1)

    response = _cancel(task_id, key)

    assert response.status_code == 409, response.text
    assert _error(response) == "task_not_cancellable"
    assert _row(task_id) == after_the_run_ended[0]
    assert len(_cancel_commands(task_id)) == 1


def test_a_run_that_ended_on_its_own_with_a_turn_queued_behind_it_is_not_cancellable(
    monkeypatch: pytest.MonkeyPatch,
):
    """A turn queued after the aimed run ended does not hide that the aimed
    run ended without a cancel."""
    task_id, _agent_id, key = _running_task()

    async def ended_then_a_turn_was_queued(**_kwargs: Any) -> None:
        _update(
            task_id,
            **_ENDED_ROWS["completed"],
            runner_id=None,
            lease_expires_at=None,
        )
        _queue_start(task_id)
        raise TaskCommandRejected("moved", reason="stale_run")

    _replace_cancel_core(monkeypatch, ended_then_a_turn_was_queued)

    response = _cancel(task_id, key)

    assert response.status_code == 409, response.text
    assert _error(response) == "task_not_cancellable"
    assert len(_cancel_commands(task_id)) == 1


def test_a_cancel_that_settled_as_paused_is_still_reported_busy(
    monkeypatch: pytest.MonkeyPatch,
):
    task_id, _agent_id, key = _running_task()
    after_pause: list[dict[str, Any]] = []

    async def paused_before_it_was_applied(**_kwargs: Any) -> None:
        _update(
            task_id,
            status=TaskStatus.PAUSED,
            control_state=TaskControlState.PAUSED.value,
            runner_id=None,
            lease_expires_at=None,
        )
        after_pause.append(_row(task_id))
        raise TaskCommandRejected("moved", reason="stale_run")

    _replace_cancel_core(monkeypatch, paused_before_it_was_applied)

    response = _cancel(task_id, key)

    assert response.status_code == 409, response.text
    assert _error(response) == "task_busy"
    assert _row(task_id) == after_pause[0]


@pytest.mark.parametrize("replacement_ended", [False, True], ids=["running", "ended"])
def test_a_cancel_aimed_at_a_run_that_was_replaced_is_not_cancellable(
    monkeypatch: pytest.MonkeyPatch, replacement_ended: bool
):
    """The aimed run is over and the row shows another one, whatever state
    that one is in: retrying would cancel it, so the caller is told no retry
    can cancel this run."""
    task_id, _agent_id, key = _running_task()
    after_replacement: list[dict[str, Any]] = []
    replacement: dict[str, Any] = {"run_id": "run-next", "state_version": 50}
    if replacement_ended:
        replacement.update(_ENDED_ROWS["completed"], runner_id=None)

    async def replaced_before_it_was_applied(**_kwargs: Any) -> None:
        _update(task_id, **replacement)
        after_replacement.append(_row(task_id))
        raise TaskCommandRejected("moved", reason="stale_run")

    _replace_cancel_core(monkeypatch, replaced_before_it_was_applied)

    response = _cancel(task_id, key)

    assert response.status_code == 409, response.text
    assert _error(response) == "task_not_cancellable"
    assert (
        response.json()["error"]["message"]
        == "The run this cancel was aimed at has already ended."
    )
    assert _row(task_id) == after_replacement[0]
    assert len(_cancel_commands(task_id)) == 1


def test_a_cancel_still_pending_when_its_run_was_replaced_is_not_cancellable(
    monkeypatch: pytest.MonkeyPatch,
):
    """The wait ends with the command still queued and the row already on
    another run: the aimed run is over, so the answer is not an accepted 202
    that a retry would turn into a cancel of the new run."""
    task_id, _agent_id, key = _running_task()

    async def replaced_while_nothing_applies(*_args: Any, **_kwargs: Any) -> bool:
        _update(task_id, run_id="run-next", state_version=50)
        return False

    monkeypatch.setattr(
        sdk_task_cancel, "dispatch_one_task_command", replaced_while_nothing_applies
    )
    monkeypatch.setattr(sdk_task_cancel, "SDK_CANCEL_REQUEST_WAIT_SECONDS", 0.1)

    response = _cancel(task_id, key)

    assert response.status_code == 409, response.text
    assert _error(response) == "task_not_cancellable"
    assert (
        response.json()["error"]["message"]
        == "The run this cancel was aimed at has already ended."
    )
    assert _row(task_id)["run_id"] == "run-next"


def test_a_turn_queued_while_the_cancel_was_applied_still_reports_the_cancel(
    monkeypatch: pytest.MonkeyPatch,
):
    """The command ended, the task was cancelled by another request and a
    new turn was queued before this request read the outcome: the run this
    call aimed at is cancelled, so the answer says so, while the other
    fields read as the task read does with a turn queued."""
    task_id, _agent_id, key = _running_task()
    before = _row(task_id)

    async def cancelled_elsewhere_then_appended(**_kwargs: Any) -> None:
        _update(
            task_id,
            runner_id=None,
            lease_expires_at=None,
            **_cancelled_row_values(),
        )
        _queue_start(task_id)
        raise TaskCommandRejected("moved", reason="stale_run")

    _replace_cancel_core(monkeypatch, cancelled_elsewhere_then_appended)

    response = _cancel(task_id, key)

    assert response.status_code == 200, response.text
    assert response.json() == {
        "task_id": task_id,
        "status": "pending",
        "run_id": before["run_id"],
        "state_version": before["state_version"],
        "cancelled_run_id": before["run_id"],
        "error_code": None,
    }


def test_a_completed_command_is_a_cancel_of_its_aimed_run_even_if_a_turn_is_queued(
    monkeypatch: pytest.MonkeyPatch,
):
    task_id, _agent_id, key = _running_task()
    before = _row(task_id)

    async def cancelled_then_appended(**_kwargs: Any) -> None:
        _update(
            task_id,
            state_version=before["state_version"] + 1,
            runner_id=None,
            lease_expires_at=None,
            **_cancelled_row_values(),
        )
        _queue_start(task_id)

    _replace_cancel_core(monkeypatch, cancelled_then_appended)

    response = _cancel(task_id, key)

    assert response.status_code == 200, response.text
    assert response.json() == {
        "task_id": task_id,
        "status": "pending",
        "run_id": before["run_id"],
        "state_version": before["state_version"] + 1,
        "cancelled_run_id": before["run_id"],
        "error_code": None,
    }


def test_a_completed_command_names_its_own_run_when_a_new_run_already_runs(
    monkeypatch: pytest.MonkeyPatch,
):
    """The answer describes the row as it is now; ``cancelled_run_id`` is the
    run the command cancelled, and nothing of the new run is touched."""
    task_id, _agent_id, key = _running_task()
    before = _row(task_id)
    after_the_new_run_started: list[dict[str, Any]] = []

    async def cancelled_then_next_run_started(**_kwargs: Any) -> None:
        _update(
            task_id,
            status=TaskStatus.RUNNING,
            control_state=TaskControlState.RUNNING.value,
            run_id="run-next",
            state_version=before["state_version"] + 2,
            error_message=None,
        )
        after_the_new_run_started.append(_row(task_id))

    _replace_cancel_core(monkeypatch, cancelled_then_next_run_started)

    response = _cancel(task_id, key)

    assert response.status_code == 200, response.text
    assert response.json() == {
        "task_id": task_id,
        "status": "running",
        "run_id": "run-next",
        "state_version": before["state_version"] + 2,
        "cancelled_run_id": before["run_id"],
        "error_code": None,
    }
    assert _row(task_id) == after_the_new_run_started[0]
    assert len(_cancel_commands(task_id)) == 1


def test_a_command_applied_by_another_process_while_polling_is_reported(
    monkeypatch: pytest.MonkeyPatch,
):
    task_id, _agent_id, key = _running_task()
    before = _row(task_id)

    def apply_elsewhere() -> None:
        _update(
            task_id,
            state_version=before["state_version"] + 1,
            runner_id=None,
            lease_expires_at=None,
            **_cancelled_row_values(),
        )
        db = _direct_db_session()
        try:
            db.query(TaskExecutionCommand).filter(
                TaskExecutionCommand.task_id == task_id,
                TaskExecutionCommand.kind == TaskCommandKind.CANCEL.value,
            ).update({"status": COMMAND_COMPLETED})
            db.commit()
        finally:
            db.close()

    async def left_to_the_owning_process(*_args: Any, **_kwargs: Any) -> bool:
        threading.Timer(0.2, apply_elsewhere).start()
        return False

    monkeypatch.setattr(
        sdk_task_cancel,
        "dispatch_one_task_command",
        AsyncMock(side_effect=left_to_the_owning_process),
    )

    response = _cancel(task_id, key)

    assert response.status_code == 200, response.text
    assert response.json()["cancelled_run_id"] == before["run_id"]
    assert response.json()["error_code"] == "task_cancelled"


def test_the_wait_is_timed_from_before_the_command_is_dispatched(
    monkeypatch: pytest.MonkeyPatch,
):
    """A dispatch that outlasts the wait leaves nothing to wait for: the
    first poll after it already finds the deadline passed."""
    task_id, _agent_id, key = _running_task()
    polls: list[int] = []
    real_load = sdk_task_cancel.load_task_command

    def counting_load(command_db_id: int):
        polls.append(command_db_id)
        return real_load(command_db_id)

    async def slow_dispatch_that_applies_nothing(*_args: Any, **_kwargs: Any) -> bool:
        await asyncio.sleep(0.4)
        return False

    monkeypatch.setattr(
        sdk_task_cancel,
        "dispatch_one_task_command",
        AsyncMock(side_effect=slow_dispatch_that_applies_nothing),
    )
    monkeypatch.setattr(sdk_task_cancel, "load_task_command", counting_load)
    monkeypatch.setattr(sdk_task_cancel, "SDK_CANCEL_REQUEST_WAIT_SECONDS", 0.1)

    response = _cancel(task_id, key)

    assert response.status_code == 202, response.text
    assert len(polls) == 1


# ----- a cancel repeated after it withdrew a queued turn -----

_PREVIOUS_RUNS = {
    "previous_run_completed": {
        "status": TaskStatus.COMPLETED,
        "control_state": TaskControlState.COMPLETED.value,
        "output": "done",
    },
    "previous_run_cancelled": {
        "status": TaskStatus.FAILED,
        "control_state": TaskControlState.FAILED.value,
        "error_message": SDK_CANCEL_ERROR_MESSAGE,
    },
    "previous_run_paused": {
        "status": TaskStatus.PAUSED,
        "control_state": TaskControlState.PAUSED.value,
    },
    "new_task": {"status": TaskStatus.PENDING, "control_state": "idle", "run_id": None},
}


def _dispatch_through_the_core(monkeypatch: pytest.MonkeyPatch, agent_id: int) -> None:
    """Apply each cancel command with the real core, as its owner would.

    A cancel is dispatched past a queued START only while that START waits for
    admission, which these tests do not set up; the core and the settlement it
    leaves behind are the real ones.
    """

    async def apply(_executor: Any, *, command_db_id: int) -> bool:
        db = _direct_db_session()
        try:
            command = db.get(TaskExecutionCommand, command_db_id)
            task_id = int(command.task_id)
            run_id = command.target_run_id
            version = int(command.target_state_version)
            queued_run_id = command.payload.get("queued_run_id")
        finally:
            db.close()
        await sdk_task_cancel.cancel_sdk_task_unserialized(
            task_id=task_id,
            agent_id=agent_id,
            expected_run_id=run_id,
            expected_state_version=version,
            queued_run_id=queued_run_id,
        )
        # A new task's START is settled by the dispatcher once the cancel
        # has cancelled the row; an appended turn's was stopped by the core.
        db = _direct_db_session()
        try:
            db.query(TaskExecutionCommand).filter(
                TaskExecutionCommand.task_id == task_id,
                TaskExecutionCommand.kind == TaskCommandKind.START.value,
                TaskExecutionCommand.status == "pending",
            ).update(
                {
                    "status": "failed",
                    "result": {"rejection_reason": "cancelled_before_admission"},
                }
            )
            db.commit()
        finally:
            db.close()
        _settle_cancel_command(command_db_id, status="completed")
        return True

    monkeypatch.setattr(sdk_task_cancel, "dispatch_one_task_command", apply)


@pytest.mark.parametrize("previous", list(_PREVIOUS_RUNS))
def test_a_repeated_cancel_answers_for_the_turn_the_first_one_withdrew(
    monkeypatch: pytest.MonkeyPatch, previous: str
):
    """The withdrawn turn leaves the row on the turn before it, so the repeat
    cannot read the turn from the row: it is answered from the record of the
    withdrawal, identically, and adds no second command."""
    task_id, agent_id, key = _running_task()
    _update(task_id, runner_id=None, lease_expires_at=None, **_PREVIOUS_RUNS[previous])
    _queue_start(task_id)
    _dispatch_through_the_core(monkeypatch, agent_id)

    first = _cancel(task_id, key)
    again = _cancel(task_id, key)

    assert first.status_code == 200, first.text
    assert first.json()["cancelled_run_id"] == QUEUED_RUN
    if previous != "new_task":
        # The row still shows the turn before the withdrawn one.
        assert first.json()["run_id"] not in (None, QUEUED_RUN)
    assert again.status_code == 200, again.text
    assert again.json() == first.json()
    assert len(_cancel_commands(task_id)) == 1


def _apply_without_completing(monkeypatch: pytest.MonkeyPatch, agent_id: int) -> None:
    """Apply the cancel with the real core and stop before it is completed.

    The command is claimed as the dispatcher claims it and the core commits
    its effect -- the withdrawn START, or a new task's cancelled row -- but
    the separate transaction that marks the command completed (and, for a
    new task, settles its START) has not run yet.
    """

    async def apply(_executor: Any, *, command_db_id: int) -> bool:
        db = _direct_db_session()
        try:
            command = db.get(TaskExecutionCommand, command_db_id)
            command.status = "processing"
            command.attempt_count = 1
            db.commit()
            task_id = int(command.task_id)
            run_id = command.target_run_id
            version = int(command.target_state_version)
            queued_run_id = command.payload.get("queued_run_id")
        finally:
            db.close()
        await sdk_task_cancel.cancel_sdk_task_unserialized(
            task_id=task_id,
            agent_id=agent_id,
            expected_run_id=run_id,
            expected_state_version=version,
            queued_run_id=queued_run_id,
        )
        return True

    monkeypatch.setattr(sdk_task_cancel, "dispatch_one_task_command", apply)


def _complete_applied_cancel(task_id: int, command_db_id: int) -> None:
    """The step ``_apply_without_completing`` left out: a new task's START is
    settled and the command is marked completed."""
    db = _direct_db_session()
    try:
        db.query(TaskExecutionCommand).filter(
            TaskExecutionCommand.task_id == task_id,
            TaskExecutionCommand.kind == TaskCommandKind.START.value,
            TaskExecutionCommand.status == "pending",
        ).update(
            {
                "status": "failed",
                "result": {"rejection_reason": "cancelled_before_admission"},
            }
        )
        db.commit()
    finally:
        db.close()
    _settle_cancel_command(command_db_id, status="completed")


def _withdrawn_before_completion(
    monkeypatch: pytest.MonkeyPatch, previous: str
) -> tuple[int, str, Any, TaskExecutionCommand]:
    """Queue a turn, cancel it, and leave the cancel applied but processing."""
    task_id, agent_id, key = _running_task()
    _update(task_id, runner_id=None, lease_expires_at=None, **_PREVIOUS_RUNS[previous])
    _queue_start(task_id)
    _apply_without_completing(monkeypatch, agent_id)
    monkeypatch.setattr(sdk_task_cancel, "SDK_CANCEL_REQUEST_WAIT_SECONDS", 0.2)
    first = _cancel(task_id, key)
    (command,) = _cancel_commands(task_id)
    assert command.status == "processing"
    return task_id, key, first, command


@pytest.mark.parametrize("previous", list(_PREVIOUS_RUNS))
def test_a_repeat_while_the_withdrawing_cancel_is_still_processing_rejoins_it(
    monkeypatch: pytest.MonkeyPatch, previous: str
):
    """The cancel withdrew the queued turn but is not marked completed yet.
    A repeat in that interval finds the same command, adds none, and answers
    exactly as the first call did at the same point: an appended turn reads
    as withdrawn, a new task as still being cancelled; once the command
    completes the new task reads as cancelled for the queued run."""
    task_id, key, first, command = _withdrawn_before_completion(monkeypatch, previous)
    dispatched = AsyncMock()
    monkeypatch.setattr(sdk_task_cancel, "dispatch_one_task_command", dispatched)

    again = _cancel(task_id, key)

    assert dispatched.await_args.kwargs["command_db_id"] == command.id
    assert [c.id for c in _cancel_commands(task_id)] == [command.id]
    assert again.status_code == first.status_code, again.text
    assert again.json() == first.json()
    if previous != "new_task":
        assert first.status_code == 200, first.text
        assert first.json()["cancelled_run_id"] == QUEUED_RUN
        # The row still shows the turn before the withdrawn one.
        assert first.json()["run_id"] not in (None, QUEUED_RUN)
        return
    assert first.status_code == 202, first.text
    assert first.json()["cancelled_run_id"] is None
    _complete_applied_cancel(task_id, command.id)

    after_completion = _cancel(task_id, key)

    assert after_completion.status_code == 200, after_completion.text
    assert after_completion.json()["run_id"] is None
    assert after_completion.json()["cancelled_run_id"] == QUEUED_RUN
    assert after_completion.json()["error_code"] == "task_cancelled"
    assert len(_cancel_commands(task_id)) == 1


def test_a_repeat_while_the_withdrawal_is_processing_does_not_cancel_the_paused_turn(
    monkeypatch: pytest.MonkeyPatch,
):
    """The turn before the withdrawn one is paused, and a paused task can be
    cancelled. The withdrawing cancel has stopped the queued START but is not
    marked completed yet, so no START is queued and the row is paused: the
    repeat must still rejoin that command rather than aim a new one at the
    paused turn, and the paused turn is left as it is."""
    task_id, key, first, command = _withdrawn_before_completion(
        monkeypatch, "previous_run_paused"
    )
    before = _row(task_id)
    dispatched = AsyncMock()
    monkeypatch.setattr(sdk_task_cancel, "dispatch_one_task_command", dispatched)

    again = _cancel(task_id, key)

    assert first.status_code == 200, first.text
    assert again.status_code == 200, again.text
    assert again.json() == first.json()
    assert again.json()["status"] == "paused"
    assert again.json()["cancelled_run_id"] == QUEUED_RUN
    assert dispatched.await_args.kwargs["command_db_id"] == command.id
    assert [c.id for c in _cancel_commands(task_id)] == [command.id]
    assert _row(task_id) == before


def test_a_repeat_while_processing_after_the_turn_before_resumed_cancels_that_turn(
    monkeypatch: pytest.MonkeyPatch,
):
    """The withdrawing cancel is still processing when the paused turn before
    it is resumed on a newer version. That command no longer describes the
    row, so the repeat is not answered with the withdrawal: it aims a new
    command at the running turn."""
    task_id, key, _first, command = _withdrawn_before_completion(
        monkeypatch, "previous_run_paused"
    )
    resumed = _row(task_id)
    _update(
        task_id,
        status=TaskStatus.RUNNING,
        control_state=TaskControlState.RUNNING.value,
        state_version=resumed["state_version"] + 1,
    )
    monkeypatch.setattr(sdk_task_cancel, "dispatch_one_task_command", AsyncMock())
    monkeypatch.setattr(sdk_task_cancel, "SDK_CANCEL_REQUEST_WAIT_SECONDS", 0.1)

    response = _cancel(task_id, key)

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "running"
    assert response.json()["run_id"] == resumed["run_id"]
    assert response.json()["cancelled_run_id"] is None
    commands = sorted(_cancel_commands(task_id), key=lambda c: c.id)
    assert len(commands) == 2
    assert commands[0].id == command.id
    assert commands[-1].target_run_id == resumed["run_id"]


def test_a_withdrawal_does_not_answer_once_the_turn_before_runs_again(
    monkeypatch: pytest.MonkeyPatch,
):
    """After the withdrawal the turn before it was resumed and runs again on
    a newer version. The record of the withdrawal no longer describes the
    row, so the cancel aims at the running turn instead of answering that
    the withdrawn one is cancelled while the running one goes on."""
    task_id, agent_id, key = _running_task()
    _update(
        task_id,
        runner_id=None,
        lease_expires_at=None,
        status=TaskStatus.PAUSED,
        control_state=TaskControlState.PAUSED.value,
    )
    _queue_start(task_id)
    _dispatch_through_the_core(monkeypatch, agent_id)
    assert _cancel(task_id, key).status_code == 200
    resumed = _row(task_id)
    _update(
        task_id,
        status=TaskStatus.RUNNING,
        control_state=TaskControlState.RUNNING.value,
        state_version=resumed["state_version"] + 1,
    )
    monkeypatch.setattr(sdk_task_cancel, "dispatch_one_task_command", AsyncMock())
    monkeypatch.setattr(sdk_task_cancel, "SDK_CANCEL_REQUEST_WAIT_SECONDS", 0.1)

    response = _cancel(task_id, key)

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "running"
    assert response.json()["run_id"] == resumed["run_id"]
    assert response.json()["cancelled_run_id"] is None
    commands = _cancel_commands(task_id)
    assert len(commands) == 2
    assert commands[-1].target_run_id == resumed["run_id"]


def _fail_newest_start(task_id: int, outcome: dict[str, Any]) -> None:
    db = _direct_db_session()
    try:
        db.query(TaskExecutionCommand).filter(
            TaskExecutionCommand.task_id == task_id,
            TaskExecutionCommand.kind == TaskCommandKind.START.value,
        ).update(outcome)
        db.commit()
    finally:
        db.close()


@pytest.mark.parametrize(
    "start_outcome",
    [
        {"status": "failed", "result": {"rejection_reason": "invalid_start"}},
        # Stopped by a control other than this endpoint's cancel.
        {
            "status": "failed",
            "result": {"rejection_reason": "cancelled_before_admission"},
        },
    ],
    ids=["failed_itself", "stopped_by_another_control"],
)
def test_a_start_that_no_cancel_of_this_endpoint_withdrew_is_not_a_cancellation(
    start_outcome,
):
    """The newest START failed and the row shows the completed turn before
    it. With no SDK cancel aimed at that START's run the failure is not a
    cancellation, so the call is refused as it would be on arrival."""
    task_id, _agent_id, key = _running_task()
    _update(
        task_id,
        runner_id=None,
        lease_expires_at=None,
        **_PREVIOUS_RUNS["previous_run_completed"],
    )
    _queue_start(task_id)
    _fail_newest_start(task_id, start_outcome)
    before = _row(task_id)

    response = _cancel(task_id, key)

    assert response.status_code == 409, response.text
    assert _error(response) == "task_not_cancellable"
    assert _row(task_id) == before
    assert _cancel_commands(task_id) == []


def test_a_cancel_not_completed_yet_for_a_stopped_start_is_rejoined(
    monkeypatch: pytest.MonkeyPatch,
):
    """The newest START was stopped and this endpoint's cancel aimed at its
    run is still pending. The call rejoins that command instead of judging
    the row or staging another one, and answers as the command's own wait
    does: at the deadline a START stopped before it began is that turn
    cancelled."""
    task_id, _agent_id, key = _running_task()
    _update(
        task_id,
        runner_id=None,
        lease_expires_at=None,
        **_PREVIOUS_RUNS["previous_run_completed"],
    )
    _queue_start(task_id)
    db = _direct_db_session()
    try:
        task = db.get(Task, task_id)
        pending = sdk_task_cancel._enqueue_cancel(
            db, task, int(task.state_version), queued_run_id=QUEUED_RUN
        )
    finally:
        db.close()
    _fail_newest_start(
        task_id,
        {
            "status": "failed",
            "result": {"rejection_reason": "cancelled_before_admission"},
        },
    )
    dispatched = AsyncMock()
    monkeypatch.setattr(sdk_task_cancel, "dispatch_one_task_command", dispatched)
    monkeypatch.setattr(sdk_task_cancel, "SDK_CANCEL_REQUEST_WAIT_SECONDS", 0.1)
    before = _row(task_id)

    response = _cancel(task_id, key)

    assert response.status_code == 200, response.text
    assert response.json()["run_id"] == before["run_id"]
    assert response.json()["cancelled_run_id"] == QUEUED_RUN
    assert dispatched.await_args.kwargs["command_db_id"] == pending.command_id
    assert [c.id for c in _cancel_commands(task_id)] == [pending.command_id]
    assert _row(task_id) == before


def test_a_cancel_after_a_withdrawn_turn_and_a_newer_started_one_answers_for_the_row(
    monkeypatch: pytest.MonkeyPatch,
):
    """A turn was withdrawn, then another was appended and is running: the
    newest START is the row's own run, so the cancel aims at the running
    turn and is not answered with the withdrawal."""
    task_id, agent_id, key = _running_task()
    _update(
        task_id,
        runner_id=None,
        lease_expires_at=None,
        **_PREVIOUS_RUNS["previous_run_completed"],
    )
    _queue_start(task_id)
    _dispatch_through_the_core(monkeypatch, agent_id)
    assert _cancel(task_id, key).status_code == 200
    _queue_start(task_id, run_id="run-second", turn_id="turn-second")
    _update(
        task_id,
        status=TaskStatus.RUNNING,
        control_state=TaskControlState.RUNNING.value,
        run_id="run-second",
    )
    db = _direct_db_session()
    try:
        db.query(TaskExecutionCommand).filter(
            TaskExecutionCommand.task_id == task_id,
            TaskExecutionCommand.command_id == "turn-second",
        ).update({"status": "completed"})
        db.commit()
    finally:
        db.close()
    monkeypatch.setattr(sdk_task_cancel, "dispatch_one_task_command", AsyncMock())
    monkeypatch.setattr(sdk_task_cancel, "SDK_CANCEL_REQUEST_WAIT_SECONDS", 0.1)

    response = _cancel(task_id, key)

    assert response.status_code == 202, response.text
    assert response.json()["run_id"] == "run-second"
    assert response.json()["cancelled_run_id"] is None
    commands = _cancel_commands(task_id)
    assert len(commands) == 2
    assert commands[-1].target_run_id == "run-second"


# ----- a queued START that can no longer hand off is not the target -----


@pytest.mark.parametrize(
    "moved",
    ["running_on_another_run", "waiting_on_another_run"],
)
def test_a_queued_start_that_can_no_longer_hand_off_is_not_the_cancel_target(
    monkeypatch: pytest.MonkeyPatch, moved: str
):
    """The row was moved to another run after the START was accepted (an
    automatic resume won the race), so the START never opens its run. The
    row is cancelled as the run it holds, running or waiting for an answer,
    as it would be with no START at all."""
    task_id, _agent_id, key = _running_task()
    _update(
        task_id,
        runner_id=None,
        lease_expires_at=None,
        **_PREVIOUS_RUNS["previous_run_completed"],
    )
    _queue_start(task_id)
    row_values: dict[str, Any] = {
        "run_id": "run-first",
        "state_version": _row(task_id)["state_version"] + 1,
    }
    if moved == "running_on_another_run":
        row_values.update(
            status=TaskStatus.RUNNING, control_state=TaskControlState.RUNNING.value
        )
    else:
        row_values.update(
            status=TaskStatus.WAITING_FOR_USER,
            control_state=TaskControlState.WAITING_FOR_USER.value,
        )
    _update(task_id, **row_values)
    monkeypatch.setattr(sdk_task_cancel, "dispatch_one_task_command", AsyncMock())
    monkeypatch.setattr(sdk_task_cancel, "SDK_CANCEL_REQUEST_WAIT_SECONDS", 0.1)
    before = _row(task_id)

    response = _cancel(task_id, key)

    assert _row(task_id) == before
    assert response.status_code == 202, response.text
    assert response.json()["run_id"] == "run-first"
    assert response.json()["status"] == row_values["status"].value
    (command,) = _cancel_commands(task_id)
    assert "queued_run_id" not in command.payload
    assert command.target_run_id == "run-first"


# ----- how the wait ends for a queued turn -----


def test_a_cancel_whose_queued_turn_started_before_the_deadline_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
):
    """At the deadline the START has been handed off and the row shows the
    turn running on the aimed run: the command is not applied yet, so the
    answer is 202 for the running turn, not "already ended"."""
    task_id, _agent_id, key = _running_task()
    _update(
        task_id,
        runner_id=None,
        lease_expires_at=None,
        **_PREVIOUS_RUNS["previous_run_completed"],
    )
    _queue_start(task_id)

    async def handed_off(_executor: Any, *, command_db_id: int) -> bool:
        _update(
            task_id,
            status=TaskStatus.RUNNING,
            control_state=TaskControlState.RUNNING.value,
            run_id=QUEUED_RUN,
            state_version=_row(task_id)["state_version"] + 1,
        )
        db = _direct_db_session()
        try:
            db.query(TaskExecutionCommand).filter(
                TaskExecutionCommand.task_id == task_id,
                TaskExecutionCommand.kind == TaskCommandKind.START.value,
            ).update({"status": "completed"})
            db.commit()
        finally:
            db.close()
        return False

    monkeypatch.setattr(sdk_task_cancel, "dispatch_one_task_command", handed_off)
    monkeypatch.setattr(sdk_task_cancel, "SDK_CANCEL_REQUEST_WAIT_SECONDS", 0.1)

    response = _cancel(task_id, key)

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "running"
    assert response.json()["run_id"] == QUEUED_RUN
    assert response.json()["cancelled_run_id"] is None


@pytest.mark.parametrize(
    "values",
    [
        {
            "status": TaskStatus.WAITING_FOR_USER,
            "control_state": TaskControlState.WAITING_FOR_USER.value,
        },
        {"status": TaskStatus.PAUSED, "control_state": TaskControlState.PAUSED.value},
    ],
    ids=["waiting", "paused"],
)
def test_waiting_and_paused_tasks_are_cancelled(values):
    task_id, agent_id, key = _running_task()
    _update(task_id, runner_id=None, lease_expires_at=None, **values)
    before = _row(task_id)

    response = _cancel(task_id, key)

    assert response.status_code == 200, response.text
    assert response.json() == {
        "task_id": task_id,
        "status": "failed",
        "run_id": before["run_id"],
        "state_version": before["state_version"] + 1,
        "cancelled_run_id": before["run_id"],
        "error_code": "task_cancelled",
    }
    read = _get(task_id, key)
    assert (read["status"], read["error_code"]) == ("failed", "task_cancelled")
    assert read["pending_interaction"] is None
    # An answer to the cancelled question has nothing to resume.
    reply = client.post(
        f"/v1/chat/tasks/{task_id}/reply",
        headers=_bearer(key),
        json={
            "agent_id": agent_id,
            "command_id": "reply-after-cancel",
            "message": {"role": "user", "content": "yes"},
        },
    )
    assert reply.status_code == 409, reply.text
    assert _error(reply) == "no_pending_interaction"


_WAITING_OR_PAUSED = {
    "waiting": {
        "status": TaskStatus.WAITING_FOR_USER,
        "control_state": TaskControlState.WAITING_FOR_USER.value,
    },
    "paused": {
        "status": TaskStatus.PAUSED,
        "control_state": TaskControlState.PAUSED.value,
    },
}


def _waiting_or_paused_task(state: str) -> tuple[int, int, str]:
    task_id, agent_id, key = _running_task()
    _update(task_id, runner_id=None, lease_expires_at=None, **_WAITING_OR_PAUSED[state])
    return task_id, agent_id, key


@pytest.mark.parametrize("state", list(_WAITING_OR_PAUSED))
def test_an_unapplied_cancel_of_a_waiting_or_paused_run_is_accepted(
    monkeypatch: pytest.MonkeyPatch, state: str
):
    task_id, _agent_id, key = _waiting_or_paused_task(state)
    monkeypatch.setattr(sdk_task_cancel, "dispatch_one_task_command", AsyncMock())
    monkeypatch.setattr(sdk_task_cancel, "SDK_CANCEL_REQUEST_WAIT_SECONDS", 0.1)
    before = _row(task_id)

    response = _cancel(task_id, key)

    assert response.status_code == 202, response.text
    assert response.json() == {
        "task_id": task_id,
        "status": before["status"].value,
        "run_id": before["run_id"],
        "state_version": before["state_version"],
        "cancelled_run_id": None,
        "error_code": None,
    }
    assert _row(task_id) == before
    assert len(_cancel_commands(task_id)) == 1


@pytest.mark.parametrize("state", list(_WAITING_OR_PAUSED))
def test_a_waiting_or_paused_run_resumed_before_the_cancel_was_applied_is_busy(
    monkeypatch: pytest.MonkeyPatch, state: str
):
    """An answer or an automatic resume took the run first: the resumed run
    is still the aimed one and can be cancelled by retrying."""
    task_id, _agent_id, key = _waiting_or_paused_task(state)
    before = _row(task_id)
    after_resume: list[dict[str, Any]] = []

    async def resumed_before_it_was_applied(**_kwargs: Any) -> None:
        _update(
            task_id,
            status=TaskStatus.RUNNING,
            control_state=TaskControlState.RUNNING.value,
            state_version=before["state_version"] + 1,
        )
        after_resume.append(_row(task_id))
        raise TaskCommandRejected("moved", reason="stale_run")

    _replace_cancel_core(monkeypatch, resumed_before_it_was_applied)

    response = _cancel(task_id, key)

    assert response.status_code == 409, response.text
    assert _error(response) == "task_busy"
    assert _row(task_id) == after_resume[0]


@pytest.mark.parametrize("state", list(_WAITING_OR_PAUSED))
@pytest.mark.parametrize("exit_path", ["stale_run", "deadline"])
def test_a_waiting_or_paused_run_that_ended_on_its_own_is_not_cancellable(
    monkeypatch: pytest.MonkeyPatch, state: str, exit_path: str
):
    """The run resumed and completed before the command was applied: it is
    the aimed run and it ended without a cancel, whichever way the wait
    ends."""
    task_id, _agent_id, key = _waiting_or_paused_task(state)
    after_the_run_ended: list[dict[str, Any]] = []

    def end_the_run() -> None:
        _update(task_id, **_ENDED_ROWS["completed"])
        after_the_run_ended.append(_row(task_id))

    if exit_path == "stale_run":

        async def ended_before_it_was_applied(**_kwargs: Any) -> None:
            end_the_run()
            raise TaskCommandRejected("moved", reason="stale_run")

        _replace_cancel_core(monkeypatch, ended_before_it_was_applied)
    else:

        async def ended_while_the_command_waits(*_args: Any, **_kwargs: Any) -> bool:
            end_the_run()
            return False

        monkeypatch.setattr(
            sdk_task_cancel,
            "dispatch_one_task_command",
            AsyncMock(side_effect=ended_while_the_command_waits),
        )
        monkeypatch.setattr(sdk_task_cancel, "SDK_CANCEL_REQUEST_WAIT_SECONDS", 0.1)

    response = _cancel(task_id, key)

    assert response.status_code == 409, response.text
    assert _error(response) == "task_not_cancellable"
    assert _row(task_id) == after_the_run_ended[0]
    assert len(_cancel_commands(task_id)) == 1


def test_a_waiting_run_with_an_answer_queued_is_refused_without_a_write():
    task_id, _agent_id, key = _waiting_or_paused_task("waiting")
    db = _direct_db_session()
    try:
        stage_task_command(
            db,
            task_id=task_id,
            actor_user_id=None,
            command_id="reply-queued",
            kind=TaskCommandKind.RESUME_INPUT,
            payload={},
        )
        db.commit()
    finally:
        db.close()
    before = _row(task_id)

    response = _cancel(task_id, key)

    assert response.status_code == 409, response.text
    assert _error(response) == "task_busy"
    assert _row(task_id) == before
    assert _cancel_commands(task_id) == []
