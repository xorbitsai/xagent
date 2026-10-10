"""``resume_task``'s claim-time guard for automatic RESUME commands.

The decision function is driven directly; the handler cases run the real
``resume_task`` (and, end to end, the durable dispatcher) against a real
database, with only the agent build and the background execution stubbed.
Runs on SQLite and, when ``XAGENT_TEST_POSTGRES_URL`` is set, PostgreSQL.
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from xagent.web.models.agent import Agent
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_auto_recovery import TaskAutoRecovery, TaskRecoveryEvent
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.trigger import (
    AgentTrigger,
    TriggerRun,
    TriggerRunStatus,
    TriggerType,
)
from xagent.web.models.user import User
from xagent.web.services import agent_service_manager, task_auto_resume
from xagent.web.services import task_command_execution as commands
from xagent.web.services import task_execution as task_execution_service
from xagent.web.services.task_auto_resume import (
    TASK_AUTO_RESUME_DISABLED_TRIGGER_ERROR,
    auto_resume_command_id,
    check_auto_resume_claim_sync,
    run_auto_resume_tick,
)
from xagent.web.services.task_command_execution import (
    ResumeCommandOutcome,
    execute_durable_task_command,
)
from xagent.web.services.task_command_transport import dispatch_one_task_command
from xagent.web.services.task_execution import ResumeReservationOutcome
from xagent.web.services.task_interaction_close import ActiveInteractionAbsent
from xagent.web.services.task_lease_service import utc_now

engine = engine_fixture

RUN = "run-guard"
PV = 7
CID = auto_resume_command_id(PV, 1, RUN)


@pytest.fixture
def factory(engine, monkeypatch) -> sessionmaker:
    Base.metadata.create_all(engine)
    result = sessionmaker(engine, autoflush=False)
    # Some callers bound ``get_session_local`` at import; set the globals.
    monkeypatch.setattr("xagent.web.models.database._SessionLocal", result)
    monkeypatch.setattr("xagent.web.models.database._engine", engine)
    monkeypatch.delenv("XAGENT_TASK_AUTO_RESUME_ENABLED", raising=False)
    monkeypatch.delenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", raising=False)
    return result


def _seed(
    db: Session,
    *,
    state: str = "dispatched",
    last_command_id: str | None = CID,
    trigger: bool = False,
    **task_fields: Any,
) -> tuple[int, int | None]:
    user = User(username=f"guard-{uuid.uuid4().hex[:8]}", password_hash="x")
    db.add(user)
    db.flush()
    fields: dict[str, Any] = {
        "status": TaskStatus.PAUSED,
        "control_state": "paused",
        "run_id": RUN,
        "state_version": PV,
    }
    fields.update(task_fields)
    task = Task(
        user_id=user.id,
        title="Guard",
        description="guard test",
        execution_mode="balanced",
        **fields,
    )
    db.add(task)
    db.flush()
    now = utc_now()
    db.add(
        TaskAutoRecovery(
            task_id=task.id,
            run_id=RUN,
            reason="lease_expired",
            state=state,
            paused_state_version=PV,
            interrupted_at=now - timedelta(minutes=1),
            episode_started_at=now - timedelta(minutes=1),
            no_progress_resumes=1 if state == "dispatched" else 0,
            total_resumes=1 if state == "dispatched" else 0,
            next_attempt_at=now + timedelta(seconds=60),
            last_command_id=last_command_id,
        )
    )
    trigger_run_id = None
    if trigger:
        agent = Agent(user_id=user.id, name="guard agent")
        db.add(agent)
        db.flush()
        trigger_row = AgentTrigger(
            user_id=user.id,
            agent_id=agent.id,
            type=TriggerType.SCHEDULED.value,
            name="tick",
            config={},
        )
        db.add(trigger_row)
        db.flush()
        run = TriggerRun(
            trigger_id=trigger_row.id,
            task_id=task.id,
            status=TriggerRunStatus.RUNNING.value,
            idempotency_key=f"guard-{task.id}",
        )
        db.add(run)
        db.flush()
        trigger_run_id = int(run.id)
    db.flush()
    return int(task.id), trigger_run_id


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "expected_run_id": RUN,
        "expected_state_version": PV,
        "reason": "lease_expired",
        "attempt": 1,
    }
    payload.update(overrides)
    return payload


def _check(
    task_id: int,
    *,
    status: Any = TaskStatus.PAUSED,
    control_state: str | None = "paused",
    run_id: str | None = RUN,
    state_version: int | None = PV,
    attempt_count: int = 1,
    command_id: Any = CID,
    auto_resume: Any = None,
):
    return check_auto_resume_claim_sync(
        task_id=task_id,
        command_id=command_id,
        auto_resume=_payload() if auto_resume is None else auto_resume,
        status=status,
        control_state=control_state,
        run_id=run_id,
        state_version=state_version,
        attempt_count=attempt_count,
    )


def _row(factory: sessionmaker, task_id: int) -> TaskAutoRecovery:
    with factory() as db:
        row = db.get(TaskAutoRecovery, task_id)
        db.expunge_all()
        return row


def _events(factory: sessionmaker, task_id: int) -> list[str]:
    with factory() as db:
        return list(
            db.scalars(
                sa.select(TaskRecoveryEvent.event)
                .where(TaskRecoveryEvent.task_id == task_id)
                .order_by(TaskRecoveryEvent.id)
            )
        )


def _task(factory: sessionmaker, task_id: int) -> Task:
    with factory() as db:
        task = db.get(Task, task_id)
        db.expunge_all()
        return task


# --------------------------------------------------------------------------
# Decision
# --------------------------------------------------------------------------


def test_at_the_fence_with_the_switch_on_proceeds_and_writes_nothing(factory):
    with factory() as db:
        task_id, _ = _seed(db)
        db.commit()

    decision = _check(task_id)

    assert decision.proceed
    assert _row(factory, task_id).state == "dispatched"
    assert _events(factory, task_id) == []


@pytest.mark.parametrize(
    ("snapshot", "fields"),
    [
        (
            {
                "status": TaskStatus.WAITING_FOR_USER,
                "control_state": "waiting_for_user",
            },
            {
                "status": TaskStatus.WAITING_FOR_USER,
                "control_state": "waiting_for_user",
            },
        ),
        ({"control_state": "pause_requested"}, {"control_state": "pause_requested"}),
        (
            {"status": TaskStatus.RUNNING, "control_state": "running"},
            {"status": TaskStatus.RUNNING, "control_state": "running"},
        ),
        ({"run_id": "run-new"}, {"run_id": "run-new"}),
        ({"state_version": PV + 1}, {"state_version": PV + 1}),
        # A different command's retry: RESUME_REQUESTED on a first attempt.
        (
            {"control_state": "resume_requested", "state_version": PV + 1},
            {"control_state": "resume_requested", "state_version": PV + 1},
        ),
    ],
)
def test_a_moved_task_is_skipped_as_stale(factory, snapshot, fields):
    with factory() as db:
        task_id, trigger_run_id = _seed(db, trigger=True, **fields)
        db.commit()

    decision = _check(task_id, **snapshot)

    assert (decision.proceed, decision.why) == (False, "stale")
    assert decision.notices == ()
    row = _row(factory, task_id)
    assert row.state == "stale"
    assert row.next_attempt_at is None
    assert _events(factory, task_id) == ["stale"]
    with factory() as db:
        # Not ended, so the TriggerRun waits for the run a person resumed.
        assert db.get(TriggerRun, trigger_run_id).status == "running"


def test_a_stale_claim_mirrors_an_ended_task_to_its_trigger_run(factory):
    with factory() as db:
        task_id, trigger_run_id = _seed(
            db, trigger=True, status=TaskStatus.COMPLETED, control_state="idle"
        )
        db.commit()

    _check(task_id, status=TaskStatus.COMPLETED, control_state="idle")

    with factory() as db:
        assert db.get(TriggerRun, trigger_run_id).status == "completed"


def test_switch_off_at_the_fence_stops_the_row_and_fails_the_trigger_run(
    factory, monkeypatch
):
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_ENABLED", "false")
    with factory() as db:
        task_id, trigger_run_id = _seed(db, trigger=True)
        db.commit()

    decision = _check(task_id)

    assert (decision.proceed, decision.why) == (False, "auto_resume_disabled")
    row = _row(factory, task_id)
    assert (row.state, row.state_detail) == ("manual", "auto_resume_disabled")
    assert _events(factory, task_id) == ["skipped_disabled"]
    with factory() as db:
        run = db.get(TriggerRun, trigger_run_id)
        assert run.status == TriggerRunStatus.FAILED.value
        assert run.error_message == TASK_AUTO_RESUME_DISABLED_TRIGGER_ERROR
    (notice,) = decision.notices
    assert notice.auto_resume["stop_reason"] == "disabled"


@pytest.mark.parametrize("enabled", ["true", "false"])
def test_own_retry_after_resume_requested_proceeds_whatever_the_switch(
    factory, monkeypatch, enabled
):
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_ENABLED", enabled)
    with factory() as db:
        task_id, _ = _seed(db, control_state="resume_requested", state_version=PV + 1)
        db.commit()

    decision = _check(
        task_id,
        control_state="resume_requested",
        state_version=PV + 1,
        attempt_count=2,
    )

    assert decision.proceed
    assert _row(factory, task_id).state == "dispatched"


@pytest.mark.parametrize(
    "snapshot",
    [
        # Only this command's own transition: exactly one version past.
        {"state_version": PV + 2},
        {"run_id": "run-new"},
        {"status": TaskStatus.WAITING_FOR_USER},
    ],
)
def test_own_retry_needs_the_exact_transition_it_wrote(factory, snapshot):
    with factory() as db:
        task_id, _ = _seed(db, control_state="resume_requested")
        db.commit()
    arguments: dict[str, Any] = {
        "control_state": "resume_requested",
        "state_version": PV + 1,
        "attempt_count": 3,
    }
    arguments.update(snapshot)

    assert _check(task_id, **arguments).why == "stale"


def test_skip_leaves_a_row_that_names_another_command(factory):
    with factory() as db:
        task_id, _ = _seed(db, last_command_id=auto_resume_command_id(PV, 2, RUN))
        db.commit()

    decision = _check(task_id, state_version=PV + 1)

    assert decision.why == "stale"
    assert _row(factory, task_id).state == "dispatched"
    assert _events(factory, task_id) == []


@pytest.mark.parametrize(
    ("command_id", "auto_resume"),
    [
        ("resume:by-hand", None),
        (None, None),
        (CID, {"expected_state_version": PV}),
        (CID, _payload(expected_run_id="")),
        (CID, _payload(expected_state_version=True)),
        (CID, _payload(expected_state_version="7")),
        (CID, ["not", "a", "dict"]),
    ],
)
def test_malformed_auto_resume_claims_raise(factory, command_id, auto_resume):
    with factory() as db:
        task_id, _ = _seed(db)
        db.commit()
    with pytest.raises(ValueError, match="invalid auto_resume payload"):
        _check(task_id, command_id=command_id, auto_resume=auto_resume)
    assert _row(factory, task_id).state == "dispatched"


# --------------------------------------------------------------------------
# resume_task
# --------------------------------------------------------------------------


@pytest.fixture
def handler(factory, monkeypatch):
    """Stub everything past the admission decisions of ``resume_task``."""

    def snapshot(task_id, _owner, *, actor_user_id=None, actor_is_admin=False):
        with factory() as db:
            task = db.get(Task, task_id)
            if task is None:
                return None
            return SimpleNamespace(
                task=SimpleNamespace(
                    user_id=task.user_id,
                    status=task.status,
                    control_state=task.control_state,
                    run_id=task.run_id,
                    state_version=task.state_version,
                    source=task.source,
                ),
                runtime_user=SimpleNamespace(id=task.user_id, is_admin=False),
            )

    monkeypatch.setattr(
        "xagent.web.services.task_setup_snapshot.load_task_setup_snapshot_sync",
        snapshot,
    )
    interaction = MagicMock(return_value=ActiveInteractionAbsent())
    monkeypatch.setattr(commands, "active_interaction_id_sync", interaction)
    monkeypatch.setattr(
        commands, "resolve_execution_scope_off_turn", lambda _task_id: None
    )
    manager = MagicMock()
    manager.running_tasks = {}
    manager.resume_admission_state.return_value = None
    manager.try_reserve_resume.return_value = ResumeReservationOutcome.RESERVED
    monkeypatch.setattr(task_execution_service, "background_task_manager", manager)
    background = AsyncMock()
    monkeypatch.setattr(task_execution_service, "execute_resume_background", background)
    agent_service = MagicMock()
    agent_service.supports_live_control = MagicMock(return_value=True)
    agents = MagicMock()
    agents.get_agent_for_task = AsyncMock(return_value=agent_service)
    monkeypatch.setattr(agent_service_manager, "get_agent_manager", lambda: agents)
    return SimpleNamespace(
        interaction=interaction, manager=manager, background=background
    )


async def _resume(
    factory: sessionmaker,
    task_id: int,
    *,
    command_id: Any = CID,
    attempt_count: int = 1,
    auto_resume: Any = None,
):
    reply = AsyncMock()
    with factory() as db:
        user_id = db.get(Task, task_id).user_id
    result = await commands.resume_task(
        reply,
        task_id,
        {
            "type": "resume_task",
            "auto_resume": _payload() if auto_resume is None else auto_resume,
            "user": SimpleNamespace(id=user_id, is_admin=False),
            "_durable_command_id": command_id,
            "_durable_attempt_count": attempt_count,
        },
    )
    return result, reply


@pytest.mark.asyncio
async def test_resume_task_proceeds_at_the_fence(factory, handler):
    with factory() as db:
        task_id, _ = _seed(db)
        db.commit()

    result, reply = await _resume(factory, task_id)

    assert result.outcome is ResumeCommandOutcome.SCHEDULED
    task = _task(factory, task_id)
    assert (task.control_state, task.state_version) == ("resume_requested", PV + 1)
    reply.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields",
    [
        {"status": TaskStatus.WAITING_FOR_USER, "control_state": "waiting_for_user"},
        {"control_state": "pause_requested"},
        {"state_version": PV + 1},
    ],
)
async def test_resume_task_skips_a_moved_task_without_an_error(
    factory, handler, fields
):
    with factory() as db:
        task_id, _ = _seed(db, **fields)
        db.commit()

    result, reply = await _resume(factory, task_id)

    assert result.outcome is ResumeCommandOutcome.AUTO_SKIPPED
    assert result.reason_code == "stale"
    reply.assert_not_awaited()
    handler.manager.try_reserve_resume.assert_not_called()
    assert _row(factory, task_id).state == "stale"
    task = _task(factory, task_id)
    assert task.state_version == fields.get("state_version", PV)


@pytest.mark.asyncio
async def test_resume_task_rejects_auto_resume_on_an_unreserved_command_id(
    factory, handler
):
    with factory() as db:
        task_id, _ = _seed(db)
        db.commit()

    result, reply = await _resume(factory, task_id, command_id="resume:forged")

    assert result.outcome is ResumeCommandOutcome.REJECTED
    assert result.reason_code == "invalid_command_payload"
    handler.manager.try_reserve_resume.assert_not_called()
    task = _task(factory, task_id)
    assert (task.control_state, task.state_version) == ("paused", PV)


@pytest.mark.asyncio
async def test_resume_task_completes_its_own_interrupted_attempt(factory, handler):
    with factory() as db:
        task_id, _ = _seed(db, control_state="resume_requested", state_version=PV + 1)
        db.commit()

    result, _reply = await _resume(factory, task_id, attempt_count=2)

    assert result.outcome is ResumeCommandOutcome.SCHEDULED
    task = _task(factory, task_id)
    assert (task.control_state, task.state_version) == ("resume_requested", PV + 2)


@pytest.mark.asyncio
async def test_row_moved_after_the_guard_defers_and_the_retry_skips(factory, handler):
    with factory() as db:
        task_id, _ = _seed(db)
        db.commit()

    def concurrent_writer(_task_id):
        # A writer outside the command queue (a v1 reply prelease) lands
        # between the guard and the RESUME_REQUESTED transition.
        with factory() as db:
            db.get(Task, task_id).state_version = PV + 1
            db.commit()
        return ActiveInteractionAbsent()

    handler.interaction.side_effect = concurrent_writer
    first, _reply = await _resume(factory, task_id)

    assert first.outcome is ResumeCommandOutcome.DEFERRED
    assert _task(factory, task_id).control_state == "paused"
    assert _row(factory, task_id).state == "dispatched"

    handler.interaction.side_effect = None
    retry, reply = await _resume(factory, task_id, attempt_count=2)

    assert retry.outcome is ResumeCommandOutcome.AUTO_SKIPPED
    reply.assert_not_awaited()
    assert _row(factory, task_id).state == "stale"


# --------------------------------------------------------------------------
# End to end: sweeper -> durable command -> guard
# --------------------------------------------------------------------------


async def _dispatch_auto_command(factory: sessionmaker, task_id: int) -> Any:
    with factory() as db:
        command_db_id = db.scalars(
            sa.select(TaskExecutionCommand.id).where(
                TaskExecutionCommand.task_id == task_id
            )
        ).one()
    assert await dispatch_one_task_command(
        execute_durable_task_command, command_db_id=command_db_id
    )
    with factory() as db:
        command = db.get(TaskExecutionCommand, command_db_id)
        db.expunge_all()
        return command


def _scheduled(factory: sessionmaker) -> int:
    with factory() as db:
        task_id, _ = _seed(db, state="scheduled", last_command_id=None)
        db.get(TaskAutoRecovery, task_id).next_attempt_at = utc_now() - timedelta(
            seconds=1
        )
        db.commit()
    return task_id


@pytest.mark.asyncio
async def test_dispatched_command_resumes_the_task_and_housekeeping_confirms(
    factory, handler, monkeypatch
):
    monkeypatch.setattr(
        task_auto_resume, "notify_task_command_dispatcher", lambda: None
    )
    task_id = _scheduled(factory)
    now = utc_now()
    assert run_auto_resume_tick(now=now).dispatched == 1

    command = await _dispatch_auto_command(factory, task_id)

    assert command.status == "completed"
    assert command.result["resume_outcome"] == "scheduled"
    handler.background.assert_called_once()
    # The resumed run moved the fence: the next check confirms the dispatch.
    report = run_auto_resume_tick(now=now + timedelta(seconds=61))
    assert report.confirmed == 1
    row = _row(factory, task_id)
    assert (row.state, row.next_attempt_at) == ("dispatched", None)


@pytest.mark.asyncio
async def test_dispatched_command_for_a_moved_task_completes_as_auto_skipped(
    factory, handler, monkeypatch
):
    monkeypatch.setattr(
        task_auto_resume, "notify_task_command_dispatcher", lambda: None
    )
    task_id = _scheduled(factory)
    assert run_auto_resume_tick(now=utc_now()).dispatched == 1
    # The user resumed and the run is now waiting on them.
    with factory() as db:
        task = db.get(Task, task_id)
        task.status = TaskStatus.WAITING_FOR_USER
        task.control_state = "waiting_for_user"
        task.state_version = PV + 3
        db.commit()
    broadcasts = AsyncMock()
    monkeypatch.setattr(commands, "_broadcast_terminal_command_error", broadcasts)

    command = await _dispatch_auto_command(factory, task_id)

    assert command.status == "completed"
    assert command.result["resume_outcome"] == "auto_skipped"
    assert command.error is None
    broadcasts.assert_not_awaited()
    handler.background.assert_not_called()
    assert _row(factory, task_id).state == "stale"


# --------------------------------------------------------------------------
# The recovery row decides too
# --------------------------------------------------------------------------


@pytest.mark.parametrize("row_case", ["other_command", "scheduled", "stale", "missing"])
@pytest.mark.parametrize("retry", [False, True])
def test_a_claim_the_row_no_longer_names_is_skipped_and_the_row_left_alone(
    factory, row_case, retry
):
    with factory() as db:
        task_id, _ = _seed(
            db,
            state="scheduled"
            if row_case == "scheduled"
            else "stale"
            if row_case == "stale"
            else "dispatched",
            last_command_id=(
                auto_resume_command_id(PV, 2, RUN)
                if row_case == "other_command"
                else CID
            ),
            **(
                {"control_state": "resume_requested", "state_version": PV + 1}
                if retry
                else {}
            ),
        )
        if row_case == "missing":
            db.delete(db.get(TaskAutoRecovery, task_id))
        db.commit()
    before = _row(factory, task_id)

    decision = _check(
        task_id,
        **(
            {
                "control_state": "resume_requested",
                "state_version": PV + 1,
                "attempt_count": 2,
            }
            if retry
            else {}
        ),
    )

    assert (decision.proceed, decision.why, decision.notices) == (
        False,
        "not_current",
        (),
    )
    after = _row(factory, task_id)
    if before is None:
        assert after is None
    else:
        assert (after.state, after.last_command_id, after.next_attempt_at) == (
            before.state,
            before.last_command_id,
            before.next_attempt_at,
        )
    assert _events(factory, task_id) == []


# --------------------------------------------------------------------------
# A resume this command started and then rolled back
# --------------------------------------------------------------------------


def test_retry_after_its_own_rolled_back_resume_is_scheduled_again(
    factory, monkeypatch
):
    monkeypatch.setattr(
        task_auto_resume, "_AUTO_RESUME_RNG", __import__("random").Random(3)
    )
    with factory() as db:
        task_id, trigger_run_id = _seed(db, trigger=True, state_version=PV + 2)
        db.commit()
    now = utc_now()

    decision = _check(task_id, state_version=PV + 2, attempt_count=2)

    assert (decision.proceed, decision.why) == (False, "rolled_back")
    row = _row(factory, task_id)
    assert row.state == "scheduled"
    assert row.paused_state_version == PV + 2
    assert row.next_attempt_at is not None
    assert row.next_attempt_at.replace(tzinfo=None) >= now.replace(tzinfo=None)
    assert _events(factory, task_id) == ["scheduled"]
    (notice,) = decision.notices
    assert notice.auto_resume["status"] == "scheduled"
    assert notice.state_version == PV + 2
    with factory() as db:
        assert db.get(TriggerRun, trigger_run_id).status == "running"


def test_rolled_back_retry_at_its_limit_stops_with_a_notice(factory):
    with factory() as db:
        task_id, trigger_run_id = _seed(db, trigger=True, state_version=PV + 2)
        db.get(TaskAutoRecovery, task_id).no_progress_resumes = 3
        db.commit()

    decision = _check(task_id, state_version=PV + 2, attempt_count=2)

    assert decision.why == "rolled_back"
    assert _row(factory, task_id).state == "exhausted"
    assert [n.auto_resume["stop_reason"] for n in decision.notices] == ["limit_reached"]
    with factory() as db:
        assert db.get(TriggerRun, trigger_run_id).status == "failed"


@pytest.mark.parametrize("case", ["first_attempt", "user_command", "three_versions"])
def test_a_rollback_lookalike_stays_stale(factory, case):
    with factory() as db:
        task_id, _ = _seed(
            db, state_version=PV + (3 if case == "three_versions" else 2)
        )
        task = db.get(Task, task_id)
        for command_id, kind in [(CID, "resume")] + (
            [("message-1", "message")] if case == "user_command" else []
        ):
            db.add(
                TaskExecutionCommand(
                    task_id=task_id,
                    actor_user_id=task.user_id,
                    command_id=command_id,
                    kind=kind,
                    payload={},
                    status="pending",
                )
            )
        db.commit()

    decision = _check(
        task_id,
        state_version=PV + (3 if case == "three_versions" else 2),
        attempt_count=1 if case == "first_attempt" else 2,
    )

    assert decision.why == "stale"
    assert _row(factory, task_id).state == "stale"


@pytest.mark.asyncio
async def test_resume_task_reschedules_after_its_own_rollback(factory, handler):
    with factory() as db:
        task_id, _ = _seed(db)
        db.commit()
    handler.manager.register_reserved_resume.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await _resume(factory, task_id)

    task = _task(factory, task_id)
    assert (task.control_state, task.state_version) == ("paused", PV + 2)
    handler.manager.register_reserved_resume.side_effect = None
    retry, reply = await _resume(factory, task_id, attempt_count=2)

    assert retry.outcome is ResumeCommandOutcome.AUTO_SKIPPED
    assert retry.reason_code == "rolled_back"
    reply.assert_not_awaited()
    row = _row(factory, task_id)
    assert (row.state, row.paused_state_version) == ("scheduled", PV + 2)


# --------------------------------------------------------------------------
# resume_task contract
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_task_with_the_switch_off_publishes_and_never_replies(
    factory, handler, monkeypatch
):
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_ENABLED", "false")
    published: list[Any] = []

    async def publish(notices):
        published.extend(notices)

    monkeypatch.setattr(task_auto_resume, "publish_recovery_notices", publish)
    with factory() as db:
        task_id, _ = _seed(db)
        db.commit()

    result, reply = await _resume(factory, task_id)

    assert result.outcome is ResumeCommandOutcome.AUTO_SKIPPED
    assert result.reason_code == "auto_resume_disabled"
    reply.assert_not_awaited()
    assert [n.auto_resume["stop_reason"] for n in published] == ["disabled"]
    assert _row(factory, task_id).state == "manual"
    assert _task(factory, task_id).state_version == PV


@pytest.mark.asyncio
async def test_resume_task_skips_an_unanswered_question_without_an_error(
    factory, handler, monkeypatch
):
    from xagent.web.services.task_interaction_close import ActiveInteractionFound

    with factory() as db:
        task_id, _ = _seed(db)
        db.commit()
    handler.interaction.return_value = ActiveInteractionFound(interaction_id=9)

    result, reply = await _resume(factory, task_id)

    assert result.outcome is ResumeCommandOutcome.AUTO_SKIPPED
    assert result.reason_code == "interaction_pending"
    reply.assert_not_awaited()
    handler.manager.try_reserve_resume.assert_not_called()


@pytest.mark.asyncio
async def test_foreign_lease_defers_at_the_fence_and_the_retry_proceeds(
    factory, handler, monkeypatch
):
    with factory() as db:
        task_id, _ = _seed(db)
        db.commit()
    foreign = MagicMock(side_effect=[True, False])
    monkeypatch.setattr(commands, "task_has_live_foreign_runner", foreign)

    first, _reply = await _resume(factory, task_id)
    assert first.outcome is ResumeCommandOutcome.DEFERRED
    assert _row(factory, task_id).state == "dispatched"

    retry, _reply = await _resume(factory, task_id, attempt_count=2)
    assert retry.outcome is ResumeCommandOutcome.SCHEDULED
    assert _task(factory, task_id).control_state == "resume_requested"


def test_a_database_error_while_closing_a_skipped_claim_propagates(
    factory, monkeypatch
):
    with factory() as db:
        task_id, _ = _seed(db, state_version=PV + 1)
        db.commit()

    def broken(*_args, **_kwargs):
        raise sa.exc.OperationalError("UPDATE", {}, Exception("database is down"))

    monkeypatch.setattr(task_auto_resume, "_give_up_no_commit", broken)

    with pytest.raises(sa.exc.OperationalError):
        _check(task_id, state_version=PV + 1)
    assert _row(factory, task_id).state == "dispatched"
