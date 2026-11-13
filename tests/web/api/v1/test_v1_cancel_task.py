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
    enqueue_task_command,
)
from xagent.web.services.task_execution_controller import TaskControlState

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


def _queue_start(task_id: int) -> None:
    """Leave an accepted START that no executor has handed off yet."""
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        enqueue_task_command(
            db,
            task_id=task_id,
            actor_user_id=int(task.user_id),
            command_id="turn-queued",
            kind=TaskCommandKind.START,
            payload={},
        )
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
                "status": TaskStatus.PAUSED,
                "control_state": TaskControlState.PAUSED.value,
            },
            "task_busy",
        ),
        (
            {
                "status": TaskStatus.WAITING_FOR_USER,
                "control_state": TaskControlState.WAITING_FOR_USER.value,
            },
            "task_busy",
        ),
    ],
    ids=["completed", "failed_itself", "pending", "paused", "waiting"],
)
def test_states_that_are_not_running_are_refused_without_a_write(values, expected_code):
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
    ],
    ids=["previous_run_cancelled", "previous_run_completed"],
)
def test_a_queued_turn_is_never_answered_from_the_previous_run(previous):
    """While an appended START is queued the row still shows the run before
    it, so answering from the row would report on the wrong turn: a
    cancelled previous run must not read as "already cancelled"."""
    task_id, _agent_id, key = _running_task()
    _update(task_id, runner_id=None, lease_expires_at=None, **previous)
    _queue_start(task_id)
    before = _row(task_id)

    response = _cancel(task_id, key)

    assert response.status_code == 409, response.text
    assert _error(response) == "task_busy"
    assert _row(task_id) == before
    assert _cancel_commands(task_id) == []
    # The read surface agrees: the queued turn is pending and carries no code.
    read = _get(task_id, key)
    assert read["status"] == "pending"
    assert read["error_code"] is None


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


def test_a_turn_queued_while_the_cancel_was_applied_still_reports_the_cancel(
    monkeypatch: pytest.MonkeyPatch,
):
    """The command ended, the task was cancelled by another request and a
    new turn was queued before this request read the outcome: the run this
    call aimed at is cancelled, so the answer says so."""
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
        "status": "failed",
        "run_id": before["run_id"],
        "state_version": before["state_version"],
        "cancelled_run_id": before["run_id"],
        "error_code": "task_cancelled",
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
        "status": "failed",
        "run_id": before["run_id"],
        "state_version": before["state_version"] + 1,
        "cancelled_run_id": before["run_id"],
        "error_code": "task_cancelled",
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
