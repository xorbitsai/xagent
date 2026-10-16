"""A status-only caller's append supersedes the automatic resume its task waits for.

In shared execution an append is accepted by staging a START at the version
the acceptance read; an automatic RESUME queued for the same paused run moves
that version first and the START would be rejected, losing the message.
``_accept_turn_no_commit`` therefore replaces a pending automatic resume of a
status-only (SDK) task in the acceptance transaction. Every other source, and
combined execution, accepts exactly as before. Runs on SQLite and, when
``XAGENT_TEST_POSTGRES_URL`` is set, on PostgreSQL.
"""

from __future__ import annotations

import ast
import pathlib
import uuid
from datetime import timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import sqlalchemy as sa

from tests.web.services.coordinator_command_shared import claim_for_owner
from tests.web.services.task_database_shared import engine as engine_fixture
from tests.web.services.test_channel_reply_hold import (
    CID,
    PV,
    RUN,
    _auto_resume,
    _limit,
    _recovery_row,
)
from tests.web.services.test_channel_reply_hold import factory as factory_fixture
from tests.web.services.test_channel_reply_hold import handler as handler_fixture
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import get_engine
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_auto_recovery import TaskAutoRecovery, TaskRecoveryEvent
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services import (
    task_auto_resume,
    task_coordinator_service,
    task_event_bridge,
    task_orchestrator,
    task_start_consumer,
)
from xagent.web.services.task_auto_resume import (
    check_auto_resume_claim_sync,
    run_auto_resume_tick,
    status_only_terminal_view,
)
from xagent.web.services.task_command_execution import ResumeCommandOutcome
from xagent.web.services.task_command_transport import SettledTaskCommand
from xagent.web.services.task_execution import _acquire_resume_task_lease
from xagent.web.services.task_execution_controller import TaskControlState
from xagent.web.services.task_interaction_close import ActiveInteractionAbsent
from xagent.web.services.task_lease_service import utc_now
from xagent.web.services.task_orchestrator import (
    TaskTurnError,
    TaskTurnOrchestrator,
    TaskTurnPayload,
)

engine = engine_fixture
factory = factory_fixture
handler = handler_fixture

SERVICES = pathlib.Path(task_auto_resume.__file__).parent


@pytest.fixture(autouse=True)
def _shared_acceptance(monkeypatch):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(task_event_bridge, "get_task_event_bridge", lambda: Mock())
    monkeypatch.setattr(
        task_auto_resume, "notify_task_command_dispatcher", lambda: None
    )


def _seed(
    factory,
    *,
    source: str,
    row: str | None = "scheduled",
    command_status: str | None = None,
    version: int = PV,
    control_state: str = "paused",
    status: TaskStatus = TaskStatus.PAUSED,
    row_version: int = PV,
) -> int:
    """A task of ``source`` resting on ``RUN`` at ``version`` and, unless
    ``row`` is None, its recovery row fenced at ``row_version``."""
    with factory() as db:
        user = User(username=f"append-{uuid.uuid4().hex[:8]}", password_hash="x")
        db.add(user)
        db.flush()
        task = Task(
            user_id=user.id,
            title="Append",
            description="append test",
            execution_mode="balanced",
            source=source,
            status=status,
            control_state=control_state,
            run_id=RUN,
            state_version=version,
        )
        db.add(task)
        db.flush()
        if row is not None:
            _recovery_row(
                db,
                int(task.id),
                RUN,
                row_version,
                state=row,
                interrupted_ago=60,
                command_status=command_status,
            )
        db.commit()
        return int(task.id)


def _accept(factory, tid: int, *, turn_id: str = "append-turn") -> Any:
    """The caller's transaction: accept an append, commit; roll back on error."""
    with factory() as db:
        owner = db.get(Task, tid).user_id
        try:
            prepared = TaskTurnOrchestrator.claim_append_turn_no_commit(
                db,
                task_id=tid,
                task_owner_user_id=owner,
                payload=TaskTurnPayload("message M", turn_id=turn_id),
            )
        except BaseException:
            db.rollback()
            raise
        db.commit()
        return prepared


def _snapshot(factory, tid: int) -> dict[str, Any]:
    """Everything the acceptance can change, without ids and timestamps."""
    with factory() as db:
        task = db.get(Task, tid)
        row = db.get(TaskAutoRecovery, tid)
        events = db.scalars(
            sa.select(TaskRecoveryEvent)
            .where(TaskRecoveryEvent.task_id == tid)
            .order_by(TaskRecoveryEvent.id)
        ).all()
        commands = db.scalars(
            sa.select(TaskExecutionCommand)
            .where(TaskExecutionCommand.task_id == tid)
            .order_by(TaskExecutionCommand.id)
        ).all()
        messages = db.scalars(
            sa.select(TaskChatMessage.turn_id).where(TaskChatMessage.task_id == tid)
        ).all()
        return {
            "task": (
                task.status,
                task.control_state,
                task.run_id if task.run_id == RUN else "<new run>",
                int(task.state_version),
            ),
            "row": None
            if row is None
            else (
                row.state,
                row.state_detail,
                int(row.paused_state_version),
                row.last_command_id,
            ),
            "events": [(e.event, e.detail) for e in events],
            "commands": [
                (c.kind, c.status, c.command_id if c.kind == "resume" else None)
                for c in commands
            ],
            "messages": list(messages),
        }


def _count_statements(factory, tid: int, fn) -> int:
    statements: list[str] = []

    def count(_conn, _cursor, statement, *_args):
        statements.append(statement)

    target = get_engine()
    sa.event.listen(target, "before_cursor_execute", count)
    try:
        fn()
    finally:
        sa.event.remove(target, "before_cursor_execute", count)
    return len(statements)


def _hand_off_start(factory, tid: int, command_db_id: int) -> Any:
    """Claim the START as the task's owner and run the consumer's handoff."""
    with factory() as db:
        lease = task_coordinator_service.acquire_task_lease_no_commit(
            db, tid, runner_id="worker-1"
        )
        assert lease is not None
        db.commit()
        command = claim_for_owner(db, lease, command_db_id)
    return task_start_consumer._commit_handoff(command, lease)


def _complete_resume_commands(factory, tid: int) -> None:
    """End the RESUME commands of the task, as its handler returning does."""
    with factory() as db:
        db.execute(
            sa.update(TaskExecutionCommand)
            .where(
                TaskExecutionCommand.task_id == tid,
                TaskExecutionCommand.kind == "resume",
            )
            .values(status="completed")
        )
        db.commit()


PENDING_ROWS = {
    "scheduled": dict(row="scheduled"),
    "dispatched_pending": dict(row="dispatched", command_status="pending"),
    "dispatched_rolled_back": dict(
        row="dispatched", command_status="processing", version=PV + 2
    ),
}


# --------------------------------------------------------------------------
# A pending automatic resume is replaced
# --------------------------------------------------------------------------


@pytest.mark.parametrize("row_state", sorted(PENDING_ROWS))
def test_append_supersedes_queued_auto_resume(factory, row_state):
    seed = PENDING_ROWS[row_state]
    tid = _seed(factory, source="sdk", **seed)
    version = seed.get("version", PV)

    prepared = _accept(factory, tid)

    after = _snapshot(factory, tid)
    assert after["task"] == (TaskStatus.PAUSED, "paused", RUN, version + 1)
    assert after["row"][:3] == ("manual", "new_turn", version + 1)
    # The event stays ``stale``, as a channel turn's supersede writes it, so
    # one timeline query reads both; the row says why it rests in ``manual``.
    superseded = [d for e, d in after["events"] if e == "stale"]
    assert len(superseded) == 1
    assert (
        superseded[0].items()
        >= {
            "at": "sdk_append",
            "state": "manual",
            "state_detail": "new_turn",
            "mismatch": "new_turn",
        }.items()
    )
    assert after["messages"] == ["append-turn"]
    start = [c for c in after["commands"] if c[0] == "start"]
    assert start == [("start", "pending", None)]
    # The START is staged at the version that moved, against the same run.
    assert prepared.command_payload["state_version"] == version + 1
    assert prepared.command_payload["expected_run_id"] == RUN

    if seed.get("command_status"):
        # The RESUME already queued is skipped by its claim guard.
        decision = check_auto_resume_claim_sync(
            task_id=tid,
            command_id=CID if version == PV else after["row"][3],
            auto_resume={"expected_run_id": RUN, "expected_state_version": PV},
            status=TaskStatus.PAUSED,
            control_state=TaskControlState.PAUSED.value,
            run_id=RUN,
            state_version=version + 1,
            attempt_count=1,
        )
        assert decision.proceed is False
    else:
        # Nothing is left for the sweeper to dispatch.
        assert run_auto_resume_tick().dispatched == 0
        assert [
            c for c in _snapshot(factory, tid)["commands"] if c[0] == "resume"
        ] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("row_state", sorted(PENDING_ROWS))
async def test_superseded_resume_cannot_reject_the_start(factory, handler, row_state):
    """Message M runs: the RESUME already queued, fenced on the version it
    admitted from, can no longer move the task, so the START is accepted by
    the consumer and the task moves to a new run. A ``scheduled`` row has no
    RESUME command yet: the sweeper finds nothing to dispatch. The others
    drive the real ``resume_task``: a pending RESUME has the append accepted
    between its claim guard and its RESUME_REQUESTED transition; a rolled-back
    one is skipped by its guard."""
    seed = PENDING_ROWS[row_state]
    tid = _seed(factory, source="sdk", **seed)
    version = seed.get("version", PV)

    ch = SimpleNamespace(task_id=tid, run_id=RUN, version=PV)
    if row_state == "scheduled":
        prepared = _accept(factory, tid)
        assert run_auto_resume_tick().dispatched == 0
    elif row_state == "dispatched_rolled_back":
        # The task is already past the RESUME's fence: its claim guard skips
        # it before the transition, with or without the supersede.
        prepared = _accept(factory, tid)
        retry = await _auto_resume(ch, CID, attempt_count=2)
        assert retry.outcome is ResumeCommandOutcome.AUTO_SKIPPED
        handler.background.assert_not_called()
        _complete_resume_commands(factory, tid)
    else:
        accepted: list[Any] = []

        def accept_between_guard_and_transition(_task_id):
            accepted.append(_accept(factory, tid))
            return ActiveInteractionAbsent()

        handler.interaction.side_effect = accept_between_guard_and_transition
        first = await _auto_resume(ch, CID, attempt_count=1)
        # The transition is fenced on the version it admitted from.
        assert first.outcome is ResumeCommandOutcome.DEFERRED
        handler.interaction.side_effect = None
        retry = await _auto_resume(ch, CID, attempt_count=2)
        # The claim guard skips the retry.
        assert retry.outcome is ResumeCommandOutcome.AUTO_SKIPPED
        handler.background.assert_not_called()
        prepared = accepted[0]
        assert _snapshot(factory, tid)["task"] == (
            TaskStatus.PAUSED,
            "paused",
            RUN,
            version + 1,
        )
        # The skipped RESUME ends and lets the START be claimed.
        _complete_resume_commands(factory, tid)

    outcome = _hand_off_start(factory, tid, prepared.command_db_id)
    assert isinstance(outcome, task_start_consumer._StartHandoff)
    assert outcome.start.message == "message M"
    with factory() as db:
        task = db.get(Task, tid)
        assert (task.status, task.run_id) == (TaskStatus.RUNNING, outcome.start.run_id)
        assert task.run_id != RUN


def test_rejected_start_after_supersede_leaves_the_task_paused(factory, monkeypatch):
    """A START that replaced the pending resume and is then rejected for an
    unrelated reason leaves the task PAUSED with automatic resume stopped; the
    message is marked delivery-failed. This pins that outcome: changing it
    (rescheduling the resume, failing the task) must change this test too."""
    tid = _seed(factory, source="sdk", row="scheduled")
    prepared = _accept(factory, tid)
    monkeypatch.setattr(
        task_start_consumer, "command_identity_matches_task", lambda *_a: False
    )

    outcome = _hand_off_start(factory, tid, prepared.command_db_id)

    assert isinstance(outcome, SettledTaskCommand)
    assert outcome.result == {"rejection_reason": "start_identity_changed"}
    after = _snapshot(factory, tid)
    assert after["task"] == (TaskStatus.PAUSED, "paused", RUN, PV + 1)
    assert after["row"][:3] == ("manual", "new_turn", PV + 1)
    with factory() as db:
        delivery = db.scalars(
            sa.select(TaskChatMessage.delivery_status).where(
                TaskChatMessage.task_id == tid
            )
        ).all()
    assert delivery == ["failed"]
    assert run_auto_resume_tick().dispatched == 0
    assert [c for c in _snapshot(factory, tid)["commands"] if c[0] == "resume"] == []


@pytest.mark.parametrize("outcome", ["start_pending", "start_runs", "start_rejected"])
@pytest.mark.parametrize(
    "row", ["scheduled", "exhausted", "dispatch_failed", "manual_disabled"]
)
def test_append_outcomes_project(factory, monkeypatch, row, outcome):
    """What a status-only caller reads after its append: ``paused`` while the
    START is queued, the turn's own status once it runs, and ``failed`` if the
    START is rejected (nothing will resume the task and the message was not
    delivered). The same on a row automatic resume had already stopped, where
    a rejected START reads the new-message text with the row's own state."""
    tid = _seed(
        factory,
        source="sdk",
        row="manual" if row == "manual_disabled" else row,
    )
    if row == "manual_disabled":
        with factory() as db:
            db.get(TaskAutoRecovery, tid).state_detail = "auto_resume_disabled"
            db.commit()
    prepared = _accept(factory, tid)

    def read():
        with factory() as db:
            return status_only_terminal_view(db, tid)

    if outcome == "start_pending":
        assert read() is None
        assert _snapshot(factory, tid)["task"][0] == TaskStatus.PAUSED
        return
    if outcome == "start_runs":
        _hand_off_start(factory, tid, prepared.command_db_id)
        assert read() is None
        assert _snapshot(factory, tid)["task"][0] == TaskStatus.RUNNING
        return
    monkeypatch.setattr(
        task_start_consumer, "command_identity_matches_task", lambda *_a: False
    )
    _hand_off_start(factory, tid, prepared.command_db_id)

    view = read()
    assert view is not None
    assert (view.state, view.error) == (
        # A pending resume is replaced by a ``manual`` row; a stopped row
        # keeps its own state.
        "manual" if row in {"scheduled", "manual_disabled"} else row,
        "Task failed: the new message could not be started.",
    )
    assert _snapshot(factory, tid)["task"][0] == TaskStatus.PAUSED


# --------------------------------------------------------------------------
# Scope: only a pending resume of a status-only source, in shared mode
# --------------------------------------------------------------------------

_SCOPE_ROWS = {
    "none": dict(row=None),
    "scheduled": dict(row="scheduled"),
    "dispatched": dict(row="dispatched", command_status="pending"),
    "exhausted": dict(row="exhausted"),
    # A user pause moved the task three versions past the row's fence.
    "scheduled_at_older_fence": dict(row="scheduled", version=PV + 3),
}


@pytest.mark.parametrize("mode", ["shared", "combined"])
@pytest.mark.parametrize("row_state", sorted(_SCOPE_ROWS))
@pytest.mark.parametrize("source", ["sdk", "a2a", "external"])
def test_supersede_scope(factory, monkeypatch, source, row_state, mode):
    if mode == "combined":
        monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "false")
    writes = (
        source == "sdk"
        and mode == "shared"
        and row_state
        in {
            "scheduled",
            "dispatched",
        }
    )

    def accept_with_count() -> tuple[int, dict[str, Any], dict[str, Any]]:
        tid = _seed(factory, source=source, **_SCOPE_ROWS[row_state])
        before = _snapshot(factory, tid)
        used = _count_statements(factory, tid, lambda: _accept(factory, tid))
        return used, before, _snapshot(factory, tid)

    # The same acceptance with the supersede switched off is the baseline.
    with monkeypatch.context() as m:
        m.setattr(
            task_orchestrator, "_lock_status_only_append_no_commit", lambda *_a: None
        )
        baseline_count, baseline_before, baseline_after = accept_with_count()
    count, before, after = accept_with_count()

    if writes:
        assert after["row"][:2] == ("manual", "new_turn")
        assert after["task"][3] == before["task"][3] + 1
        assert after != baseline_after
        return
    assert after == baseline_after
    if source != "sdk":
        # Not the status-only caller: the very statements main issues.
        assert count == baseline_count


# --------------------------------------------------------------------------
# Busy and zero-row paths leave nothing behind
# --------------------------------------------------------------------------


def _zero_rowcount_on(monkeypatch, call: int) -> None:
    real = task_auto_resume._rowcount
    seen = {"n": 0}

    def rowcount(result):
        seen["n"] += 1
        return 0 if seen["n"] == call else real(result)

    monkeypatch.setattr(task_auto_resume, "_rowcount", rowcount)


@pytest.mark.parametrize(
    "case",
    [
        "resume_requested_with_live_resume",
        "version_cas_loses",
        "refence_loses",
        "give_up_cas_loses",
    ],
)
def test_append_busy(factory, monkeypatch, case):
    if case == "resume_requested_with_live_resume":
        tid = _seed(
            factory,
            source="sdk",
            row="dispatched",
            command_status="processing",
            version=PV + 1,
            control_state="resume_requested",
        )
    else:
        tid = _seed(factory, source="sdk")
        _zero_rowcount_on(
            monkeypatch,
            {"version_cas_loses": 1, "refence_loses": 2, "give_up_cas_loses": 3}[case],
        )
    before = _snapshot(factory, tid)

    with pytest.raises(TaskTurnError, match="busy"):
        _accept(factory, tid)

    assert _snapshot(factory, tid) == before


async def _resume_until_hand_off(factory, handler) -> int:
    """A task whose automatic RESUME ran through the real ``resume_task``: the
    task is ``resume_requested``, the resumed run has not taken its lease, and
    the RESUME command has ended."""
    tid = _seed(factory, source="sdk", row="dispatched", command_status="processing")
    ch = SimpleNamespace(task_id=tid, run_id=RUN, version=PV)
    result = await _auto_resume(ch, CID, attempt_count=1)
    assert result.outcome is ResumeCommandOutcome.SCHEDULED
    handler.background.assert_called_once()
    assert _snapshot(factory, tid)["task"] == (
        TaskStatus.PAUSED,
        "resume_requested",
        RUN,
        PV + 1,
    )
    _complete_resume_commands(factory, tid)
    return tid


@pytest.mark.asyncio
async def test_append_is_busy_until_the_hand_off_takes_its_lease(factory, handler):
    """The RESUME command ends before the resumed run claims its lease; an
    append accepted in between would stage its START at a version the claim
    then moves, and lose the message."""
    tid = await _resume_until_hand_off(factory, handler)
    before = _snapshot(factory, tid)

    with pytest.raises(TaskTurnError, match="busy"):
        _accept(factory, tid)

    assert _snapshot(factory, tid) == before
    with factory() as db:
        owner = db.get(Task, tid).user_id
    assert _acquire_resume_task_lease(tid, owner, RUN) is not None
    with factory() as db:
        task = db.get(Task, tid)
        assert (task.status, task.run_id) == (TaskStatus.RUNNING, RUN)


@pytest.mark.asyncio
async def test_a_stranded_hand_off_accepts_past_the_window(factory, handler):
    """A hand-off whose automatic resume is past the channel window and its
    grace is treated as stranded: the append is accepted and its START takes
    over the task."""
    tid = await _resume_until_hand_off(factory, handler)
    with factory() as db:
        db.get(TaskAutoRecovery, tid).interrupted_at = utc_now() - timedelta(
            seconds=_limit() + 5
        )
        db.commit()

    prepared = _accept(factory, tid)

    assert _snapshot(factory, tid)["messages"] == ["append-turn"]
    outcome = _hand_off_start(factory, tid, prepared.command_db_id)
    assert isinstance(outcome, task_start_consumer._StartHandoff)
    with factory() as db:
        task = db.get(Task, tid)
        assert (task.status, task.run_id) == (TaskStatus.RUNNING, outcome.start.run_id)
        assert task.run_id != RUN


# --------------------------------------------------------------------------
# Writers and exits stay enumerable
# --------------------------------------------------------------------------


def test_give_up_call_sites():
    source = (SERVICES / "task_auto_resume.py").read_text()
    # The definition plus one call per exit.
    assert source.count("_give_up_no_commit(") - 1 == 15


def _is_state_writer_call(node: ast.Call) -> bool:
    if not (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "values"
        and any(keyword.arg == "state" for keyword in node.keywords)
    ):
        return False
    receiver: ast.AST = node.func.value
    while isinstance(receiver, (ast.Call, ast.Attribute)):
        if (
            isinstance(receiver, ast.Call)
            and isinstance(receiver.func, ast.Name)
            and receiver.func.id == "update"
            and receiver.args
            and isinstance(receiver.args[0], ast.Name)
            and receiver.args[0].id == "TaskAutoRecovery"
        ):
            return True
        receiver = receiver.func if isinstance(receiver, ast.Call) else receiver.value
        if isinstance(receiver, ast.Attribute):
            receiver = receiver.value
    return False


def test_recovery_state_writers():
    writers: set[str] = set()
    for path in sorted(SERVICES.glob("*.py")):
        for function in ast.walk(ast.parse(path.read_text())):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(function):
                direct = isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Attribute)
                    and target.attr == "state"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "row"
                    for target in node.targets
                )
                if direct or (
                    isinstance(node, ast.Call) and _is_state_writer_call(node)
                ):
                    writers.add(function.name)
    assert writers == {
        "_give_up_no_commit",
        "_process_due_candidate_no_commit",
        "_record_status_only_fallback_no_commit",
        "_reschedule_dispatched_no_commit",
        "record_interruption_no_commit",
    }
