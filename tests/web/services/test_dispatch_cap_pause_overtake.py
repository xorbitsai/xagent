"""A pause claimed by a process at its dispatch cap passes the never-claimed
commands that process holds for its task. A pause that finds nothing of its
task running settles them, without building an agent."""

from __future__ import annotations

import contextlib
import functools
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import OperationalError

from tests.web.services import test_dispatch_cap, test_dispatch_cap_cancel_overtake
from tests.web.services.test_dispatch_cap import CapHost
from tests.web.services.test_dispatch_cap_cancel_overtake import (
    EXTERNAL_COMMAND_ID,
    RUN,
    SETTLE_LOGGER,
    SETTLED,
    _add_waiting_ticket,
    add_user_row,
    cap_held_task,
    delivery_rows,
    enqueue,
    enqueue_at,
    events_of,
    hold_cap,
    run,
    state,
    task_state,
)
from xagent.web.models.task import TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services import (
    task_command_execution,
)
from xagent.web.services import task_command_transport as transport
from xagent.web.services import (
    task_execution_admission,
    task_orchestrator,
    task_resume_command,
    task_setup_snapshot,
)
from xagent.web.services.task_command_transport import TaskCommandKind

# Fixtures shared with the cancel tests; ``host`` needs ``cap_host`` beside it.
cap_host = test_dispatch_cap.cap_host
host = test_dispatch_cap_cancel_overtake.host
pause_host = test_dispatch_cap_cancel_overtake.pause_host

NO_EXECUTION = "No live execution found to pause"
ALREADY_PAUSED = "Task is already paused"
STOPPED_BEFORE_START = "Task stopped before execution started."
UNTOUCHED = ("pending", 0, None, None)


def stage_pause(
    host: CapHost,
    task_id: int,
    target: str | None = None,
    payload: dict[str, Any] | None = None,
    actor: int | None = None,
) -> int:
    """Stage a pause through the producers' own primitive, so its identity
    columns and its default target (the task's run) are the real ones."""

    host._identity += 1
    with host.factory() as db:
        staged = transport.stage_task_command(
            db,
            task_id=task_id,
            actor_user_id=host.user_id if actor is None else actor,
            command_id=f"pause-{host._identity}",
            kind=TaskCommandKind.PAUSE,
            payload={} if payload is None else payload,
            target_run_id=target,
        )
        db.commit()
        return staged.staged_db_id


def accept_start(host: CapHost, task_id: int) -> tuple[int, str, str]:
    """Accept the first turn of a new task: ``(command id, run id, turn id)``."""

    accepted = task_orchestrator._begin_turn_atomic_sync(
        task_id,
        host.user_id,
        payload=task_orchestrator.TaskTurnPayload(transcript_message="hello"),
        kind=task_orchestrator.TurnKind.CREATE,
    )
    return (
        accepted.command_db_id,
        accepted.command_payload["run_id"],
        accepted.command_payload["turn_id"],
    )


def assert_settled(host: CapHost, *command_ids: int) -> None:
    for command_id in command_ids:
        assert state(host, command_id) == SETTLED
        assert len(events_of(host, command_id)) == 1


def assert_pause_not_applicable(
    host: CapHost, pause_id: int, error: str, builds: int = 0
) -> None:
    row = host.row(pause_id)
    assert (row.status, row.error, row.result) == ("failed", error, None)
    assert len(events_of(host, pause_id)) == 1
    assert host.agent_builds.await_count == builds


def assert_untouched(host: CapHost, command_id: int) -> None:
    assert state(host, command_id) == UNTOUCHED
    assert events_of(host, command_id) == []


async def _new_task_start(host: CapHost) -> None:
    await hold_cap(host)
    task_id = host.new_task(status=TaskStatus.PENDING, run_id=None, state_version=0)
    start_id, _, turn_id = accept_start(host, task_id)
    pause_id = stage_pause(host, task_id)

    assert await run() is True

    assert_pause_not_applicable(host, pause_id, NO_EXECUTION)
    assert_settled(host, start_id)
    assert task_state(host, task_id) == (TaskStatus.FAILED, STOPPED_BEFORE_START, 1)
    assert delivery_rows(host, task_id) == [(turn_id, "failed")]
    (event,) = events_of(host, start_id)
    assert event.include_command_identity is True


async def _appended_start(host: CapHost) -> None:
    await hold_cap(host)
    task_id = host.new_task(status=TaskStatus.COMPLETED, run_id=RUN, state_version=4)
    start_id = enqueue(
        host, task_id, "start", target_run_id="run-new", target_state_version=4
    )
    pause_id = stage_pause(host, task_id)

    assert await run() is True

    assert_pause_not_applicable(host, pause_id, NO_EXECUTION)
    assert_settled(host, start_id)
    assert task_state(host, task_id) == (TaskStatus.COMPLETED, None, 4)


async def _paused_user_resume(host: CapHost) -> None:
    _, task_id = await cap_held_task(host, TaskStatus.PAUSED, control_state="paused")
    resume_id = enqueue_at(host, task_id, "resume", 3, command_id="resume-by-owner")
    pause_id = stage_pause(host, task_id)

    assert await run() is True

    assert_pause_not_applicable(host, pause_id, ALREADY_PAUSED)
    assert_settled(host, resume_id)
    assert task_state(host, task_id) == (TaskStatus.PAUSED, None, 3)


async def _waiting_reply(host: CapHost) -> None:
    _, task_id = await cap_held_task(host, TaskStatus.WAITING_FOR_USER)
    reply_id = enqueue_at(host, task_id, "resume_input", 3)
    pause_id = stage_pause(host, task_id)

    assert await run() is True

    assert_pause_not_applicable(host, pause_id, NO_EXECUTION)
    assert_settled(host, reply_id)
    # The client waiting on its reply is told to retry with a new id.
    assert task_resume_command._read_reply_outcome(reply_id) == {
        "outcome": "busy",
        "retry_with_new_id": True,
    }
    assert task_state(host, task_id) == (TaskStatus.WAITING_FOR_USER, None, 3)


async def _new_turn_message(host: CapHost) -> None:
    _, task_id = await cap_held_task(host, TaskStatus.PAUSED, control_state="paused")
    add_user_row(host, task_id, "turn-earlier", "dispatched")
    add_user_row(host, task_id, "turn-open", "pending")
    rows_before = delivery_rows(host, task_id)
    message_id = enqueue_at(host, task_id, "message", 3, command_id="web-message-1")
    pause_id = stage_pause(host, task_id)

    assert await run() is True

    assert_pause_not_applicable(host, pause_id, ALREADY_PAUSED)
    assert_settled(host, message_id)
    assert delivery_rows(host, task_id) == rows_before
    assert task_state(host, task_id) == (TaskStatus.PAUSED, None, 3)


async def _external_message(host: CapHost) -> None:
    _, task_id = await cap_held_task(
        host, TaskStatus.WAITING_FOR_USER, 5, source="external"
    )
    message_id = enqueue_at(
        host,
        task_id,
        "message",
        5,
        command_id=EXTERNAL_COMMAND_ID,
        payload={"scope": "external", "turn_id": "delivery-1"},
    )
    pause_id = stage_pause(host, task_id)

    assert await run() is True

    assert_pause_not_applicable(host, pause_id, NO_EXECUTION)
    assert_settled(host, message_id)
    (event,) = events_of(host, message_id)
    assert event.include_command_identity is False
    assert delivery_rows(host, task_id) == []
    assert task_state(host, task_id) == (TaskStatus.WAITING_FOR_USER, None, 5)


async def _two_held_rows(host: CapHost) -> None:
    _, task_id = await cap_held_task(host, TaskStatus.WAITING_FOR_USER)
    reply_id = enqueue_at(host, task_id, "resume_input", 3)
    message_id = enqueue_at(host, task_id, "message", 3)
    pause_id = stage_pause(host, task_id)

    assert await run() is True

    assert_pause_not_applicable(host, pause_id, NO_EXECUTION)
    assert_settled(host, reply_id, message_id)
    assert task_state(host, task_id) == (TaskStatus.WAITING_FOR_USER, None, 3)


async def _shared_channel_stop(host: CapHost) -> None:
    await hold_cap(host)
    task_id = host.new_task(status=TaskStatus.PENDING, run_id=None, state_version=0)
    start_id, start_run, _ = accept_start(host, task_id)
    message_id = enqueue(host, task_id, "message", target_state_version=0)
    pause_id = stage_pause(host, task_id, start_run)

    assert await run() is True

    assert_pause_not_applicable(host, pause_id, NO_EXECUTION)
    assert_settled(host, start_id, message_id)
    assert task_state(host, task_id) == (TaskStatus.FAILED, STOPPED_BEFORE_START, 1)


async def _ticketed_start(host: CapHost) -> None:
    await hold_cap(host)
    task_id = host.new_task(status=TaskStatus.PENDING, run_id=None, state_version=0)
    start_id, _, _ = accept_start(host, task_id)
    _add_waiting_ticket(host, task_id, start_id)
    pause_id = stage_pause(host, task_id)

    assert await run() is True

    assert_pause_not_applicable(host, pause_id, NO_EXECUTION, builds=1)
    assert_untouched(host, start_id)


async def _ticketed_resume_beside_held_message(host: CapHost) -> None:
    # Two rows of one task, one with a ticket and one without, exist only when
    # admission was switched on between the two enqueues.
    _, task_id = await cap_held_task(host, TaskStatus.PAUSED, control_state="paused")
    ticketed_id = enqueue_at(host, task_id, "resume", 3, command_id="resume-ticketed")
    _add_waiting_ticket(host, task_id, ticketed_id)
    message_id = enqueue_at(host, task_id, "message", 3)
    pause_id = stage_pause(host, task_id)

    assert await run() is True

    assert_pause_not_applicable(host, pause_id, ALREADY_PAUSED)
    assert_settled(host, message_id)
    assert_untouched(host, ticketed_id)


async def _other_users_pause(host: CapHost) -> None:
    _, task_id = await cap_held_task(host, TaskStatus.PAUSED, control_state="paused")
    resume_id = enqueue_at(host, task_id, "resume", 3)
    with host.factory() as db:
        other = User(username="other-user", password_hash="unused")
        db.add(other)
        db.commit()
        other_id = int(other.id)
    # A payload cannot carry the not-applicable mark in.
    pause_id = stage_pause(
        host,
        task_id,
        payload={"_durable_pause_not_applicable": True},
        actor=other_id,
    )

    assert await run() is True

    row = host.row(pause_id)
    assert row.status == "failed"
    assert row.error.startswith("Access denied")
    assert_untouched(host, resume_id)
    assert task_state(host, task_id) == (TaskStatus.PAUSED, None, 3)
    assert host.agent_builds.await_count == 0


async def _running_row_nothing_live(host: CapHost) -> None:
    _, task_id = await cap_held_task(host, TaskStatus.RUNNING)
    message_id = enqueue_at(host, task_id, "message", 3)
    pause_id = stage_pause(host, task_id)

    assert await run() is True

    assert_pause_not_applicable(host, pause_id, NO_EXECUTION, builds=1)
    assert_settled(host, message_id)
    assert task_state(host, task_id) == (TaskStatus.RUNNING, None, 3)


async def _error_payload_read_fails(host: CapHost) -> None:
    _, task_id = await cap_held_task(host, TaskStatus.RUNNING)
    message_id = enqueue_at(host, task_id, "message", 3)
    pause_id = stage_pause(host, task_id)

    with patch.object(
        task_command_execution,
        "_read_task_error_payload_offloop",
        AsyncMock(side_effect=ValueError("payload data invalid")),
    ):
        assert await run() is True

    # The pause ends as a data error, not as not applicable: nothing it passed
    # is settled.
    row = host.row(pause_id)
    assert (row.status, row.error) == ("failed", "payload data invalid")
    assert_untouched(host, message_id)


async def _precheck_reply_fails(host: CapHost) -> None:
    _, task_id = await cap_held_task(host, TaskStatus.PAUSED, control_state="paused")
    resume_id = enqueue_at(host, task_id, "resume", 3)
    pause_id = stage_pause(host, task_id)

    with patch.object(
        task_command_execution,
        "_read_task_error_payload_offloop",
        AsyncMock(side_effect=ValueError("payload data invalid")),
    ):
        assert await run() is True

    # Unlike the same failure inside ``pause_task``, the pause is retried.
    row = host.row(pause_id)
    assert (row.status, row.failure_count) == ("pending", 1)
    assert_untouched(host, resume_id)
    assert host.agent_builds.await_count == 0


async def _pause_task_other_rejection(host: CapHost, *, unsupported: bool) -> None:
    _, task_id = await cap_held_task(host, TaskStatus.RUNNING)
    message_id = enqueue_at(host, task_id, "message", 3)
    pause_id = stage_pause(host, task_id)
    if unsupported:
        host.agent_builds.return_value = SimpleNamespace()
        error = "Current agent does not support pause functionality"
    else:
        host.agent_builds.return_value.pause_execution.return_value = True
        error = "Task finished before the pause request was applied"

    with patch.object(
        task_command_execution,
        "_apply_pause_requested_isolated",
        return_value=task_command_execution.PauseWriteOutcome.NOT_APPLIED,
    ):
        assert await run() is True

    row = host.row(pause_id)
    assert (row.status, row.error) == ("failed", error)
    assert_untouched(host, message_id)
    assert host.agent_builds.await_count == 1


async def _retry_finds_run_paused(host: CapHost) -> None:
    _, task_id = await cap_held_task(host, TaskStatus.PAUSED, control_state="paused")
    resume_id = enqueue_at(host, task_id, "resume", 3)
    pause_id = stage_pause(host, task_id)
    with host.factory() as db:
        db.query(TaskExecutionCommand).filter(
            TaskExecutionCommand.id == pause_id
        ).update({TaskExecutionCommand.attempt_count: 1})
        db.commit()

    assert await run() is True

    assert host.row(pause_id).status == "completed"
    assert len(events_of(host, pause_id)) == 1
    assert_settled(host, resume_id)
    assert task_state(host, task_id) == (TaskStatus.PAUSED, None, 3)
    assert host.agent_builds.await_count == 1


async def _settlement_failure(host: CapHost) -> None:
    _, task_id = await cap_held_task(host, TaskStatus.PAUSED, control_state="paused")
    resume_id = enqueue_at(host, task_id, "resume", 3)
    pause_id = stage_pause(host, task_id)
    engine = host.factory.kw["bind"]

    def fail_settlement(conn, cursor, statement, parameters, context, executemany):
        head, _, tail = statement.partition("WHERE")
        if (
            statement.startswith("UPDATE task_execution_commands")
            and "completed_at" in head
            and "kind IN" in tail
        ):
            raise OperationalError(statement, parameters, Exception("injected"))

    sa.event.listen(engine, "before_cursor_execute", fail_settlement)
    try:
        # The failure surfaces; what it leaves behind is checked.
        with contextlib.suppress(Exception):
            await run()
    finally:
        sa.event.remove(engine, "before_cursor_execute", fail_settlement)

    # Nothing is half written. The sender may already have been told that the
    # pause did not apply, because the reply goes out before the write.
    assert host.row(pause_id).status == "processing"
    assert_untouched(host, resume_id)
    assert task_state(host, task_id) == (TaskStatus.PAUSED, None, 3)


async def _held_row_read_fails(host: CapHost) -> None:
    _, task_id = await cap_held_task(host, TaskStatus.PAUSED, control_state="paused")
    resume_id = enqueue_at(host, task_id, "resume", 3)
    pause_id = stage_pause(host, task_id)
    engine = host.factory.kw["bind"]
    failures: list[str] = []

    def fail_held_row_read(conn, cursor, statement, parameters, context, many):
        if statement.startswith("SELECT EXISTS") and not failures:
            failures.append(statement)
            raise OperationalError(statement, parameters, Exception("injected"))

    sa.event.listen(engine, "before_cursor_execute", fail_held_row_read)
    try:
        assert await run() is True
    finally:
        sa.event.remove(engine, "before_cursor_execute", fail_held_row_read)

    assert len(failures) == 1
    row = host.row(pause_id)
    assert (row.status, row.failure_count) == ("pending", 1)
    assert_untouched(host, resume_id)
    assert host.agent_builds.await_count == 0


async def _snapshot_data_invalid(host: CapHost, *, with_held_row: bool) -> None:
    _, task_id = await cap_held_task(host, TaskStatus.PAUSED, control_state="paused")
    resume_id = enqueue_at(host, task_id, "resume", 3) if with_held_row else None
    pause_id = stage_pause(host, task_id)

    with patch.object(
        task_setup_snapshot,
        "load_task_setup_snapshot_sync",
        side_effect=ValueError("snapshot data invalid"),
    ):
        assert await run() is True

    row = host.row(pause_id)
    assert (row.status, row.failure_count, row.error) == (
        "failed",
        1,
        "snapshot data invalid",
    )
    if resume_id is not None:
        assert_untouched(host, resume_id)


async def _stale_web_pause(host: CapHost, *, held: str, held_run: str) -> None:
    await hold_cap(host)
    task_id = host.new_task(
        status=TaskStatus.COMPLETED, run_id="run-new", state_version=4
    )
    held_id = enqueue(
        host, task_id, held, target_run_id=held_run, target_state_version=4
    )
    pause_id = stage_pause(host, task_id, "run-old")

    assert await run() is True

    row = host.row(pause_id)
    assert (row.status, row.result) == ("failed", {"rejection_reason": "stale_run"})
    assert len(events_of(host, pause_id)) == 1
    assert_untouched(host, held_id)
    assert host.agent_builds.await_count == 0


async def _ticketed_start_stopped_by_channel(host: CapHost) -> None:
    await hold_cap(host)
    task_id = host.new_task(status=TaskStatus.PENDING, run_id=None, state_version=0)
    start_id, start_run, _ = accept_start(host, task_id)
    _add_waiting_ticket(host, task_id, start_id)
    # No ticket on this one: admission was switched on after it was enqueued.
    message_id = enqueue(host, task_id, "message", target_state_version=0)
    pause_id = stage_pause(host, task_id, start_run)

    assert await run() is True

    assert host.row(pause_id).status == "completed"
    assert state(host, start_id) == (
        "failed",
        0,
        STOPPED_BEFORE_START,
        {"rejection_reason": "cancelled_before_admission"},
    )
    assert_settled(host, message_id)
    for command_id in (start_id, message_id, pause_id):
        assert len(events_of(host, command_id)) == 1
    assert task_state(host, task_id) == (TaskStatus.FAILED, STOPPED_BEFORE_START, 1)


PAUSE_SCENARIOS: dict[str, Any] = {
    "S1-new-task-start": _new_task_start,
    "S2-appended-start": _appended_start,
    "S3-paused-user-resume": _paused_user_resume,
    "S4-waiting-reply": _waiting_reply,
    "S5-new-turn-message": _new_turn_message,
    "S6-external-message": _external_message,
    "S7-two-held-rows": _two_held_rows,
    "S8-shared-channel-stop": _shared_channel_stop,
    "S9-ticketed-start": _ticketed_start,
    "S10-ticketed-resume-beside-held-message": _ticketed_resume_beside_held_message,
    "S11-other-users-pause": _other_users_pause,
    "S12-running-row-nothing-live": _running_row_nothing_live,
    "S12b-error-payload-read-fails": _error_payload_read_fails,
    "S12c-precheck-reply-fails": _precheck_reply_fails,
    "S12d-pause-not-supported": functools.partial(
        _pause_task_other_rejection, unsupported=True
    ),
    "S12e-pause-write-not-applied": functools.partial(
        _pause_task_other_rejection, unsupported=False
    ),
    "S13-retry-finds-run-paused": _retry_finds_run_paused,
    "S14-settlement-failure": _settlement_failure,
    "S15-held-row-read-fails": _held_row_read_fails,
    "S15b-snapshot-data-invalid-held": functools.partial(
        _snapshot_data_invalid, with_held_row=True
    ),
    "S15b-snapshot-data-invalid-no-held": functools.partial(
        _snapshot_data_invalid, with_held_row=False
    ),
    "S16-stale-web-pause": functools.partial(
        _stale_web_pause, held="message", held_run="run-old"
    ),
    # A held start for a run other than the pause's target.
    "S16b-stale-web-pause-other-start": functools.partial(
        _stale_web_pause, held="start", held_run="run-next"
    ),
    "S17-ticketed-start-stopped-by-channel": _ticketed_start_stopped_by_channel,
}


@pytest.mark.parametrize("scenario", list(PAUSE_SCENARIOS))
async def test_pause_settles_overtaken_commands(
    pause_host: CapHost, caplog: pytest.LogCaptureFixture, scenario: str
) -> None:
    with caplog.at_level(logging.INFO, logger=SETTLE_LOGGER):
        await PAUSE_SCENARIOS[scenario](pause_host)

    records = [r for r in caplog.records if "dispatch-cap" in r.getMessage()]
    if scenario == "S7-two-held-rows":
        (record,) = records
        assert record.levelno == logging.INFO
        message = record.getMessage()
        assert "control=pause" in message and "settled_commands=2" in message
        assert "'resume_input': 1" in message and "'message': 1" in message
    elif scenario in (
        "S9-ticketed-start",
        "S11-other-users-pause",
        "S12c-precheck-reply-fails",
        "S12d-pause-not-supported",
        "S12e-pause-write-not-applied",
        "S16-stale-web-pause",
        "S16b-stale-web-pause-other-start",
    ):
        assert records == []


@pytest.mark.parametrize("case", ["legacy", "task_active"])
async def test_pause_precheck_leaves_other_cases_to_pause_task(
    pause_host: CapHost, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    host = pause_host
    if case == "legacy":
        monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "false")
    _, task_id = await cap_held_task(host, TaskStatus.PAUSED, control_state="paused")
    enqueue_at(host, task_id, "message", 3)
    pause_id = stage_pause(host, task_id)
    if case == "task_active":
        # The pause and a message queued just before it were committed together.
        host.live(task_id)
    reply = AsyncMock()
    message_data: dict[str, Any] = {
        "user": SimpleNamespace(id=host.user_id, is_admin=False),
        "_durable_pause_not_applicable": False,
    }

    handled = await task_command_execution._pause_cap_held_commands_without_agent(
        reply, SimpleNamespace(id=pause_id, task_id=task_id), message_data
    )

    assert handled is False
    assert "_durable_command_error" not in message_data
    assert message_data["_durable_pause_not_applicable"] is False
    reply.assert_not_awaited()


# Case -> (control kind, control status, whether the held row is settled).
SELECTION_CASES = {
    "pause_version_equal_row": ("pause", "completed", True),
    "cancel_version_equal_row": ("cancel", "completed", False),
    "pause_ticket_row": ("pause", "failed", False),
    "pause_claimed_once": ("pause", "failed", False),
    "pause_later_row": ("pause", "failed", False),
    "pause_earlier_cancel_row": ("pause", "failed", False),
    "pause_legacy": ("pause", "failed", False),
}


@pytest.mark.parametrize("case", list(SELECTION_CASES))
async def test_settle_selection_by_control_kind(
    host: CapHost, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    kind, control_status, settled = SELECTION_CASES[case]
    if case == "pause_legacy":
        monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "false")
    task_id = host.new_task(status=TaskStatus.FAILED, state_version=1)
    done: dict[str, Any] = {"status": control_status, "attempt_count": 1}
    # The held rows are recorded at the task's current state version.
    if case == "pause_later_row":
        control_id = enqueue(host, task_id, kind, **done)
        held_id = enqueue(host, task_id, "message", target_state_version=1)
    else:
        if case == "pause_earlier_cancel_row":
            held_id = enqueue(host, task_id, "cancel", target_state_version=1)
        elif case == "pause_claimed_once":
            held_id = enqueue(
                host, task_id, "resume", target_state_version=1, attempt_count=1
            )
        else:
            held_id = enqueue(host, task_id, "resume", target_state_version=1)
        control_id = enqueue(host, task_id, kind, **done)
    if case == "pause_ticket_row":
        _add_waiting_ticket(host, task_id, held_id)

    with host.factory() as db:
        task_execution_admission.settle_cap_held_commands_no_commit(db, control_id)
        db.commit()

    if settled:
        assert_settled(host, held_id)
    elif case == "pause_claimed_once":
        assert state(host, held_id) == ("pending", 1, None, None)
        assert events_of(host, held_id) == []
    else:
        assert_untouched(host, held_id)
