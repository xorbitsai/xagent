"""A cancel claimed by a process at its dispatch cap passes the never-claimed
commands that process holds for its task, and settles them when it completes."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import itertools
import logging
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import OperationalError

from tests.web.services.test_dispatch_cap import (  # noqa: F401
    CapHost,
    cap_host,
    set_limit,
)
from xagent.web.models.agent import Agent
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_admission import TaskAdmissionBucket, TaskAdmissionTicket
from xagent.web.models.task_auto_recovery import TaskAutoRecovery
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.task_command_terminal_event import TaskCommandTerminalEvent

# Imported up front: modules that bind ``get_session_local`` at import time must
# not first load while a test has replaced it.
from xagent.web.services import (  # noqa: F401
    a2a_task_cancel,
    external_task_cancel,
    task_auto_resume,
    task_command_execution,
)
from xagent.web.services import task_command_transport as transport
from xagent.web.services import (  # noqa: F401
    task_dispatch_cap,
    task_event_bridge,
    task_execution_admission,
    task_orchestrator,
    task_resume_command,
)
from xagent.web.services.task_auto_resume import auto_resume_command_id
from xagent.web.services.task_command_transport import TaskCommandKind
from xagent.web.services.task_dispatch_cap import (
    CONTROL_KINDS,
    EXECUTION_OPENING_KINDS,
)
from xagent.web.services.task_execution_admission import settle_cancelled_admissions
from xagent.web.services.task_lease_service import utc_now

HELD_KINDS = ("start", "resume", "resume_input", "message")
STOPPED = "Task command stopped by a control request."
SETTLED = ("failed", 0, STOPPED, {"rejection_reason": "cancelled_before_admission"})
RUN = "run-1"
SETTLE_LOGGER = "xagent.web.services.task_execution_admission"
EXTERNAL_COMMAND_ID = "ext-" + uuid.UUID(int=7).hex + uuid.UUID(int=8).hex[:24]


@pytest.fixture
def host(
    cap_host: CapHost,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> CapHost:
    """The shared-mode dispatcher host of ``test_dispatch_cap`` with a limit of
    one execution, and the real cancel cores pointed at its database."""

    for module in (
        task_command_execution,
        a2a_task_cancel,
        external_task_cancel,
        task_resume_command,
    ):
        monkeypatch.setattr(module, "get_session_local", lambda: cap_host.factory)
    monkeypatch.setattr(task_event_bridge, "_bridge", Mock())
    monkeypatch.setattr(
        task_auto_resume, "notify_task_command_dispatcher", lambda: None
    )
    set_limit(monkeypatch, 1)
    return cap_host


def make_agent(host: CapHost) -> int:
    with host.factory() as db:
        row = Agent(user_id=host.user_id, name="agent")
        db.add(row)
        db.commit()
        return int(row.id)


def enqueue(host: CapHost, task_id: int, kind: str, **fields: Any) -> int:
    """Insert one command row the way the queue holds it: ``pending``, never claimed."""

    host._identity += 1
    command_id = fields.pop("command_id", f"cmd-{host._identity}")
    payload = fields.pop("payload", {})
    fields.setdefault("target_state_version", 0)
    with host.factory() as db:
        row = TaskExecutionCommand(
            task_id=task_id,
            command_id=command_id,
            kind=kind,
            payload=payload,
            actor_user_id=host.user_id,
            **fields,
        )
        db.add(row)
        db.commit()
        return int(row.id)


async def hold_cap(host: CapHost) -> None:
    """Fill the one slot with an execution of another task."""

    host.live(host.new_task())


async def cap_held_task(
    host: CapHost,
    status: TaskStatus,
    version: int = 3,
    *,
    source: str = "a2a",
    run_id: str | None = RUN,
    **fields: Any,
) -> tuple[int, int]:
    """A task of a new agent while the one slot is held: ``(agent_id, task_id)``."""

    agent_id = make_agent(host)
    await hold_cap(host)
    task_id = host.new_task(
        source=source,
        agent_id=agent_id,
        status=status,
        run_id=run_id,
        state_version=version,
        **fields,
    )
    return agent_id, task_id


def enqueue_at(
    host: CapHost, task_id: int, kind: str, version: int, **fields: Any
) -> int:
    """Enqueue a command recorded against the task's run and ``version``."""

    return enqueue(
        host,
        task_id,
        kind,
        target_run_id=fields.pop("run_id", RUN),
        target_state_version=version,
        **fields,
    )


def enqueue_cancel(
    host: CapHost, task_id: int, agent_id: int, version: int, **extra: Any
) -> int:
    run_id = extra.pop("run_id", RUN)
    return enqueue_at(
        host,
        task_id,
        "cancel",
        version,
        run_id=run_id,
        payload={"agent_id": agent_id, "target_state_version": version, **extra},
    )


def state(host: CapHost, command_id: int) -> tuple[Any, ...]:
    row = host.row(command_id)
    return (row.status, row.attempt_count, row.error, row.result)


def task_state(host: CapHost, task_id: int) -> tuple[Any, ...]:
    with host.factory() as db:
        task = db.get(Task, task_id)
        return (task.status, task.error_message, task.state_version)


def events_of(host: CapHost, command_id: int) -> list[TaskCommandTerminalEvent]:
    with host.factory() as db:
        rows = (
            db.query(TaskCommandTerminalEvent)
            .filter(TaskCommandTerminalEvent.task_command_id == command_id)
            .all()
        )
        db.expunge_all()
        return rows


def delivery_rows(host: CapHost, task_id: int) -> list[tuple[str, str]]:
    with host.factory() as db:
        return sorted(
            (str(turn_id), str(status))
            for turn_id, status in db.query(
                TaskChatMessage.turn_id, TaskChatMessage.delivery_status
            ).filter(TaskChatMessage.task_id == task_id, TaskChatMessage.role == "user")
        )


def add_user_row(host: CapHost, task_id: int, turn_id: str, status: str) -> None:
    with host.factory() as db:
        db.add(
            TaskChatMessage(
                task_id=task_id,
                user_id=host.user_id,
                role="user",
                content="hello",
                message_type="text",
                turn_id=turn_id,
                delivery_status=status,
            )
        )
        db.commit()


async def handler(command: Any) -> dict[str, Any]:
    return await task_command_execution.execute_durable_task_command(command)


async def run(command_db_id: int | None = None) -> bool:
    return await asyncio.wait_for(
        transport.dispatch_one_task_command(handler, command_db_id=command_db_id), 10
    )


def settle_directly(host: CapHost, cancel_id: int) -> None:
    """Run the terminal-write settlement of a completed cancel in one transaction."""

    with host.factory() as db:
        settle_cancelled_admissions(db, cancel_id)
        task_execution_admission.settle_cap_held_commands_no_commit(db, cancel_id)
        db.commit()


def _matrix_cases() -> list[Any]:
    held_first = [
        pytest.param(
            kind, control, cap_state, "held_first", id=f"{kind}-{control}-{cap_state}"
        )
        for kind, control, cap_state in itertools.product(
            HELD_KINDS,
            ("cancel", "pause"),
            ("open", "closed_task_inactive", "closed_task_active"),
        )
    ]
    control_first = [
        pytest.param(
            "start",
            control,
            cap_state,
            "control_first",
            id=f"arrival-control-first-{control}-{cap_state}",
        )
        for control, cap_state in itertools.product(
            ("cancel", "pause"), ("open", "closed_task_inactive")
        )
    ]
    pause_between = pytest.param(
        "start",
        "cancel",
        "closed_task_inactive",
        "pause_between",
        id="earlier-pause-blocks-cancel",
    )
    return [*held_first, *control_first, pause_between]


@pytest.mark.parametrize("held_kind,control,cap_state,order", _matrix_cases())
async def test_cancel_overtake_matrix(
    host: CapHost, held_kind: str, control: str, cap_state: str, order: str
) -> None:
    task_id = host.new_task(status=TaskStatus.PENDING)
    if order == "control_first":
        control_id = enqueue(host, task_id, control)
        held_id = enqueue(host, task_id, held_kind)
    elif order == "pause_between":
        held_id = enqueue(host, task_id, held_kind)
        enqueue(host, task_id, "pause")
        control_id = enqueue(host, task_id, control)
    else:
        held_id = enqueue(host, task_id, held_kind)
        control_id = enqueue(host, task_id, control)
    if cap_state == "closed_task_inactive":
        await hold_cap(host)
    elif cap_state == "closed_task_active":
        host.live(task_id)
    cap = host.manager.dispatch_cap_snapshot(1)
    assert cap.closed is (cap_state != "open")

    layer1 = transport._find_command_candidate(None, (), cap)
    layer2 = transport._find_command_candidate(control_id, (), cap)

    control_candidate = (control_id, task_id, TaskCommandKind(control))
    held_candidate = (held_id, task_id, TaskCommandKind(held_kind))
    if order == "control_first":
        assert layer1 == control_candidate
        assert layer2 == control_candidate
    elif order == "pause_between":
        assert layer1 is None
        assert layer2 is None
    elif cap_state == "closed_task_inactive":
        expected = control_candidate if control == "cancel" else None
        assert layer1 == expected
        assert layer2 == expected
    else:
        assert layer1 == held_candidate
        assert layer2 is None


@pytest.mark.parametrize(
    "variant", ["overtakes", "claimed_between", "cap_reopened", "task_became_active"]
)
async def test_claim_update_honours_cancel_overtake(
    host: CapHost, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    other = host.new_task()
    _, finish = host.live(other)
    task_id = host.new_task(status=TaskStatus.PENDING)
    start_id = enqueue(host, task_id, "start")
    cancel_id = enqueue(host, task_id, "cancel")
    cap = host.manager.dispatch_cap_snapshot(1)
    assert cap.closed
    coordinator = await host.registry.ensure(task_id)
    assert coordinator is not None
    if variant == "cap_reopened":
        finish.set()
        await host.held[-1]
        assert not host.manager.dispatch_cap_snapshot(1).closed
    if variant == "task_became_active":
        # Started outside the queue after the snapshot; the claim still
        # decides by the snapshot it was given.
        host.live(task_id)
        assert task_id in host.manager.dispatch_cap_snapshot(1).active_ids
    if variant == "claimed_between":
        real_query = transport._claimable_query

        def claimable_then_claimed_elsewhere(*args: Any, **kwargs: Any) -> Any:
            query = real_query(*args, **kwargs)
            real_first = query.first

            def first() -> Any:
                found = real_first()
                with host.factory() as other_db:
                    other_db.query(TaskExecutionCommand).filter(
                        TaskExecutionCommand.id == start_id
                    ).update(
                        {
                            TaskExecutionCommand.status: "processing",
                            TaskExecutionCommand.attempt_count: 1,
                        }
                    )
                    other_db.commit()
                return found

            query.first = first
            return query

        monkeypatch.setattr(
            transport, "_claimable_query", claimable_then_claimed_elsewhere
        )

    async def claim() -> Any:
        with host.factory() as db:
            return transport.claim_task_command(
                db, runner_id="worker", command_db_id=cancel_id, cap=cap
            )

    claimed = await coordinator.execute_command(
        SimpleNamespace(task_id=task_id, kind=TaskCommandKind.CANCEL), claim
    )

    if variant == "claimed_between":
        assert claimed is None
        assert host.row(cancel_id).status == "pending"
        assert host.row(cancel_id).attempt_count == 0
        assert host.row(start_id).status == "processing"
    else:
        assert claimed is not None and claimed.id == cancel_id
        assert host.row(cancel_id).status == "processing"
        assert host.row(start_id).status == "pending"


def _claimed_rows() -> dict[str, dict[str, Any]]:
    return {
        "message_outcome_unknown": {
            "kind": "message",
            "attempt_count": 1,
            "result": {"delivery_outcome": "outcome_unknown"},
        },
        "resume_input_deferred": {
            "kind": "resume_input",
            "attempt_count": 1,
            "retry_available_at": datetime.now(timezone.utc) + timedelta(hours=1),
        },
        "start_failed_back_to_pending": {
            "kind": "start",
            "attempt_count": 1,
            "error": "transient failure",
        },
        "start_processing": {
            "kind": "start",
            "status": "processing",
            "attempt_count": 1,
        },
    }


@pytest.mark.parametrize("name", list(_claimed_rows()))
async def test_cancel_does_not_overtake_claimed_rows(host: CapHost, name: str) -> None:
    fields = dict(_claimed_rows()[name])
    kind = fields.pop("kind")
    await hold_cap(host)
    task_id = host.new_task(status=TaskStatus.PENDING)
    held_id = enqueue(host, task_id, kind, **fields)
    cancel_id = enqueue(host, task_id, "cancel")
    before = state(host, held_id)
    cap = host.manager.dispatch_cap_snapshot(1)

    assert transport._find_command_candidate(cancel_id, (), cap) is None
    assert await run(cancel_id) is False

    assert host.claimed == []
    assert state(host, held_id) == before
    assert host.row(cancel_id).status == "pending"
    assert host.row(cancel_id).attempt_count == 0


async def _a2a_new_task_start(host: CapHost) -> None:
    agent_id, task_id = await cap_held_task(host, TaskStatus.PENDING, 0, run_id=None)
    accepted = task_orchestrator._begin_turn_atomic_sync(
        task_id,
        host.user_id,
        payload=task_orchestrator.TaskTurnPayload(transcript_message="hello"),
        kind=task_orchestrator.TurnKind.CREATE,
    )
    start_id = accepted.command_db_id
    turn_id = accepted.command_payload["turn_id"]
    assert delivery_rows(host, task_id) == [(turn_id, "pending")]
    cancel_id = enqueue_cancel(host, task_id, agent_id, 0, run_id=None)

    assert await run(cancel_id) is True

    assert host.row(cancel_id).status == "completed"
    assert state(host, start_id) == SETTLED
    assert task_state(host, task_id) == (
        TaskStatus.FAILED,
        "Task canceled by A2A client.",
        1,
    )
    assert delivery_rows(host, task_id) == [(turn_id, "failed")]
    (event,) = events_of(host, start_id)
    assert event.include_command_identity is True
    assert len(events_of(host, cancel_id)) == 1


async def _a2a_waiting_reply(host: CapHost) -> None:
    agent_id, task_id = await cap_held_task(host, TaskStatus.WAITING_FOR_USER)
    reply_id = enqueue_at(host, task_id, "resume_input", 3)
    cancel_id = enqueue_cancel(host, task_id, agent_id, 3)

    assert await run() is True

    assert state(host, reply_id) == SETTLED
    assert host.row(cancel_id).status == "completed"
    assert task_state(host, task_id)[0] == TaskStatus.FAILED
    assert task_state(host, task_id)[2] == 4
    (event,) = events_of(host, reply_id)
    assert event.include_command_identity is True
    # The client waiting on its reply is told to retry with a new id.
    assert task_resume_command._read_reply_outcome(reply_id) == {
        "outcome": "busy",
        "retry_with_new_id": True,
    }


async def _a2a_paused_user_resume(host: CapHost) -> None:
    agent_id, task_id = await cap_held_task(
        host, TaskStatus.PAUSED, control_state="paused"
    )
    resume_id = enqueue_at(host, task_id, "resume", 3, command_id="resume-by-owner")
    enqueue_cancel(host, task_id, agent_id, 3)

    assert await run() is True

    assert state(host, resume_id) == SETTLED
    assert task_state(host, task_id)[0] == TaskStatus.FAILED


async def _a2a_paused_auto_resume(host: CapHost) -> None:
    agent_id, task_id = await cap_held_task(
        host, TaskStatus.PAUSED, control_state="paused"
    )
    now = utc_now()
    command_id = auto_resume_command_id(3, 1, RUN)
    with host.factory() as db:
        db.add(
            TaskAutoRecovery(
                task_id=task_id,
                run_id=RUN,
                reason="lease_expired",
                state="dispatched",
                paused_state_version=3,
                interrupted_at=now - timedelta(seconds=60),
                episode_started_at=now - timedelta(seconds=60),
                no_progress_resumes=1,
                total_resumes=1,
                next_attempt_at=now - timedelta(seconds=1),
                last_command_id=command_id,
            )
        )
        db.commit()
    resume_id = enqueue_at(
        host,
        task_id,
        "resume",
        3,
        command_id=command_id,
        payload={"type": "resume_task"},
    )
    enqueue_cancel(host, task_id, agent_id, 3)

    assert await run() is True

    assert state(host, resume_id) == SETTLED
    with host.factory() as db:
        result = task_auto_resume._housekeep_candidate_no_commit(db, task_id, now=now)
        db.commit()
    assert result.outcome is task_auto_resume._Outcome.CONFIRMED
    with host.factory() as db:
        row = db.get(TaskAutoRecovery, task_id)
        assert row.state == "dispatched"
        assert row.next_attempt_at is None


async def _a2a_new_turn_message(host: CapHost) -> None:
    agent_id, task_id = await cap_held_task(
        host, TaskStatus.PAUSED, control_state="paused"
    )
    add_user_row(host, task_id, "turn-earlier", "dispatched")
    add_user_row(host, task_id, "turn-open", "pending")
    rows_before = delivery_rows(host, task_id)
    message_id = enqueue_at(host, task_id, "message", 3, command_id="web-message-1")
    enqueue_cancel(host, task_id, agent_id, 3)

    assert await run() is True

    assert state(host, message_id) == SETTLED
    assert delivery_rows(host, task_id) == rows_before
    assert len(events_of(host, message_id)) == 1


async def _external_message(host: CapHost, *, with_user_row: bool) -> None:
    agent_id, task_id = await cap_held_task(
        host, TaskStatus.WAITING_FOR_USER, 5, source="external"
    )
    turn_id = "delivery-1"
    if with_user_row:
        add_user_row(host, task_id, turn_id, "pending")
    message_id = enqueue_at(
        host,
        task_id,
        "message",
        5,
        command_id=EXTERNAL_COMMAND_ID,
        payload={"scope": "external", "turn_id": turn_id},
    )
    enqueue_cancel(host, task_id, agent_id, 5, scope="external", end_user_id=9)

    assert await run() is True

    assert state(host, message_id) == SETTLED
    assert task_state(host, task_id)[:2] == (
        TaskStatus.FAILED,
        "Stopped by the visitor.",
    )
    (event,) = events_of(host, message_id)
    assert event.include_command_identity is False
    expected = [(turn_id, "dispatched")] if with_user_row else []
    assert delivery_rows(host, task_id) == expected


async def _two_held_rows(host: CapHost) -> None:
    agent_id, task_id = await cap_held_task(host, TaskStatus.WAITING_FOR_USER)
    reply_id = enqueue_at(host, task_id, "resume_input", 3)
    message_id = enqueue_at(host, task_id, "message", 3)
    enqueue_cancel(host, task_id, agent_id, 3)

    assert await run() is True

    for command_id in (reply_id, message_id):
        assert state(host, command_id) == SETTLED
        assert len(events_of(host, command_id)) == 1


async def _cancel_without_version_change(host: CapHost) -> None:
    agent_id, task_id = await cap_held_task(
        host,
        TaskStatus.FAILED,
        control_state="failed",
        agent_config={"a2a_state": "TASK_STATE_CANCELED"},
    )
    resume_id = enqueue_at(host, task_id, "resume", 3)
    cancel_id = enqueue_cancel(host, task_id, agent_id, 3)

    assert await run() is True

    assert host.row(cancel_id).status == "completed"
    assert task_state(host, task_id)[2] == 3
    assert state(host, resume_id) == ("pending", 0, None, None)
    assert events_of(host, resume_id) == []


async def _rejected_cancel(host: CapHost) -> None:
    agent_id, task_id = await cap_held_task(
        host, TaskStatus.WAITING_FOR_USER, 5, source="external"
    )
    message_id = enqueue_at(
        host,
        task_id,
        "message",
        4,
        command_id=EXTERNAL_COMMAND_ID,
        payload={"scope": "external", "turn_id": "delivery-1"},
    )
    cancel_id = enqueue_cancel(
        host, task_id, agent_id, 4, scope="external", end_user_id=9
    )

    assert await run() is True

    assert host.row(cancel_id).status == "failed"
    assert host.row(cancel_id).result["rejection_reason"] == "stale_run"
    assert state(host, message_id) == ("pending", 0, None, None)
    assert events_of(host, message_id) == []


async def _settlement_failure(host: CapHost) -> None:
    agent_id, task_id = await cap_held_task(host, TaskStatus.WAITING_FOR_USER)
    reply_id = enqueue_at(host, task_id, "resume_input", 3)
    cancel_id = enqueue_cancel(host, task_id, agent_id, 3)
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

    assert host.row(cancel_id).status == "processing"
    assert state(host, reply_id) == ("pending", 0, None, None)
    assert events_of(host, reply_id) == []
    assert task_state(host, task_id)[:2] == (
        TaskStatus.FAILED,
        "Task canceled by A2A client.",
    )


SETTLE_SCENARIOS: dict[str, Any] = {
    "1-a2a-new-task-start": _a2a_new_task_start,
    "2-a2a-waiting-reply": _a2a_waiting_reply,
    "3-a2a-paused-user-resume": _a2a_paused_user_resume,
    "4-a2a-paused-auto-resume": _a2a_paused_auto_resume,
    "5-a2a-new-turn-message": _a2a_new_turn_message,
    "6-external-message-no-user-row": functools.partial(
        _external_message, with_user_row=False
    ),
    "6b-external-message-with-user-row": functools.partial(
        _external_message, with_user_row=True
    ),
    "7-two-held-rows": _two_held_rows,
    "8-cancel-without-version-change": _cancel_without_version_change,
    "9-rejected-cancel": _rejected_cancel,
    "10-settlement-failure": _settlement_failure,
}


@pytest.mark.parametrize("scenario", list(SETTLE_SCENARIOS))
async def test_cancel_settles_overtaken_commands(
    host: CapHost, caplog: pytest.LogCaptureFixture, scenario: str
) -> None:
    with caplog.at_level(logging.INFO, logger=SETTLE_LOGGER):
        await SETTLE_SCENARIOS[scenario](host)

    records = [r for r in caplog.records if "dispatch-cap" in r.getMessage()]
    if scenario == "7-two-held-rows":
        (record,) = records
        assert record.levelno == logging.INFO
        message = record.getMessage()
        assert "settled_commands=2" in message
        assert "'resume_input': 1" in message and "'message': 1" in message
    elif scenario in ("8-cancel-without-version-change", "9-rejected-cancel"):
        assert records == []


@pytest.mark.parametrize(
    "case", ["legacy", "later_row", "pause_control", "earlier_pause_row"]
)
async def test_completed_control_settles_nothing(
    host: CapHost, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """Settlement leaves a command alone unless a shared-mode cancel completed
    after it was queued and the command is an execution-opening one."""

    if case == "legacy":
        monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "false")
    task_id = host.new_task(status=TaskStatus.FAILED, state_version=1)
    done: dict[str, Any] = {"status": "completed", "attempt_count": 1}
    if case == "later_row":
        control_id = enqueue(host, task_id, "cancel", **done)
        untouched_id = enqueue(host, task_id, "message")
    elif case == "pause_control":
        untouched_id = enqueue(host, task_id, "resume")
        control_id = enqueue(host, task_id, "pause", **done)
    elif case == "earlier_pause_row":
        untouched_id = enqueue(host, task_id, "pause")
        control_id = enqueue(host, task_id, "cancel", **done)
    else:
        untouched_id = enqueue(host, task_id, "resume")
        control_id = enqueue(host, task_id, "cancel", **done)

    settle_directly(host, control_id)

    assert state(host, untouched_id) == ("pending", 0, None, None)
    assert events_of(host, untouched_id) == []


def _add_waiting_ticket(host: CapHost, task_id: int, command_id: int) -> None:
    with host.factory() as db:
        if db.get(TaskAdmissionBucket, "bucket") is None:
            db.add(TaskAdmissionBucket(key="bucket", capacity=1, max_pending=10))
            db.flush()
        db.add(
            TaskAdmissionTicket(
                command_id=command_id,
                bucket_key="bucket",
                task_id=task_id,
                owner_attempt_id=None,
            )
        )
        db.commit()


@pytest.mark.parametrize("with_cap_held_row", [False, True])
async def test_ticketed_rows_keep_existing_settle_under_cancel(
    host: CapHost, caplog: pytest.LogCaptureFixture, with_cap_held_row: bool
) -> None:
    task_id = host.new_task(status=TaskStatus.FAILED, state_version=1)
    ticketed_id = enqueue(host, task_id, "resume")
    _add_waiting_ticket(host, task_id, ticketed_id)
    held_id = enqueue(host, task_id, "message") if with_cap_held_row else None
    cancel_id = enqueue(host, task_id, "cancel", status="completed", attempt_count=1)

    with host.factory() as db:
        settle_cancelled_admissions(db, cancel_id)
        db.commit()
    assert host.row(ticketed_id).status == "failed"
    if held_id is not None:
        assert host.row(held_id).status == "pending"
    assert len(events_of(host, ticketed_id)) == 1

    with caplog.at_level(logging.INFO, logger=SETTLE_LOGGER):
        with host.factory() as db:
            task_execution_admission.settle_cap_held_commands_no_commit(db, cancel_id)
            db.commit()

    assert len(events_of(host, ticketed_id)) == 1
    if held_id is None:
        assert "settled_commands" not in caplog.text
    else:
        assert "settled_commands=1" in caplog.text
        assert host.row(held_id).status == "failed"
        assert len(events_of(host, held_id)) == 1
        # Both settlements write the same error text.
        assert host.row(ticketed_id).error == host.row(held_id).error == STOPPED


def test_cap_overtaking_kinds() -> None:
    overtaking = task_dispatch_cap.CAP_OVERTAKING_CONTROL_KINDS
    assert overtaking == {"cancel"}
    assert overtaking <= CONTROL_KINDS
    assert EXECUTION_OPENING_KINDS == {"start", "resume", "resume_input", "message"}, (
        "a kind that opens an execution was added: decide in "
        "settle_cap_held_commands_no_commit what happens to it once settled"
    )
