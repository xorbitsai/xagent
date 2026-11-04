"""A channel reply held while an automatic resume of its run is due.

The hold is derived: ``_read_channel_result`` asks
``channel_reply_waits_for_auto_resume`` when the task rests PAUSED, the
shared channel leaf leaves the START without a ``channel_result`` only when
it recorded ``scheduled``, and the resumed run stores its answer on the
START. A new channel turn accepted before the resume hands off supersedes
it, and one sent while it hands off is refused as busy. Most cases seed the
recovery row (or force the recorded state) directly; the shared channel
cases at the end record and dispatch it for real. Runs on SQLite and, when
``XAGENT_TEST_POSTGRES_URL`` is set, on PostgreSQL; the lock-order cases are
PostgreSQL only.
"""

from __future__ import annotations

import dataclasses
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from tests.web.services.channel_delivery_shared import (
    database_url as database_url_fixture,
)
from tests.web.services.channel_delivery_shared import (
    expire_claim,
)
from tests.web.services.channel_delivery_shared import selected as selected_fixture
from tests.web.services.task_database_shared import engine as engine_fixture
from tests.web.services.task_database_shared import task_id as task_id_fixture
from tests.web.services.test_persistence_failure_pause import (
    _checkpoint,
)
from tests.web.services.test_persistence_failure_pause import (
    _late_bound_sessions as _late_bound_sessions_fixture,
)
from tests.web.services.test_persistence_failure_pause import (
    _prepare_run,
    _projected_task,
)
from tests.web.services.test_provider_failure_pause import LLM_RESULT, _settle_channel
from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from tests.web.services.test_task_lease_expiry_interruption import _fail_recording
from xagent.core.agent.interruption import InterruptionReason
from xagent.web.models import database
from xagent.web.models.database import Base, get_engine, get_session_local
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_auto_recovery import TaskAutoRecovery, TaskRecoveryEvent
from xagent.web.models.task_channel_delivery import TaskChannelDelivery
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.models.user_channel import UserChannel

# Imported here, not lazily: a module first imported while ``canonical``
# patches ``database.get_session_local`` would keep that patched factory.
from xagent.web.services import agent_service_manager, channel_delivery
from xagent.web.services import shared_channel_execution as shared
from xagent.web.services import task_auto_recovery, task_auto_resume
from xagent.web.services import task_command_execution as commands
from xagent.web.services import task_execution as task_execution_service
from xagent.web.services.client_error_messages import CLIENT_SAFE_TASK_FAILURE
from xagent.web.services.execution_result_projection import interrupted_channel_result
from xagent.web.services.managed_task_lease import finalize_managed_task_lease_result
from xagent.web.services.task_auto_recovery_policy import staleness_window_seconds
from xagent.web.services.task_auto_resume import (
    CHANNEL_HOLD_CLOCK_GRACE_SECONDS,
    auto_resume_command_id,
    channel_reply_waits_for_auto_resume,
    check_auto_resume_claim_sync,
    select_next_candidate_for_update,
    supersede_pending_auto_resume_no_commit,
)
from xagent.web.services.task_command_execution import ResumeCommandOutcome
from xagent.web.services.task_command_transport import _claimable_query
from xagent.web.services.task_execution import (
    ResumeReservationOutcome,
    _finalize_resumed_task,
    _PreparedTaskFileOutputs,
)
from xagent.web.services.task_interaction_close import (
    ActiveInteractionAbsent,
    ActiveInteractionFound,
)
from xagent.web.services.task_lease_service import acquire_task_lease, utc_now
from xagent.web.services.task_orchestrator import (
    TaskTurnError,
    TaskTurnPayload,
    settle_task_lease_isolated,
)
from xagent.web.services.task_settlement_report import SettlementReport

engine = engine_fixture
task_id = task_id_fixture
canonical = canonical_fixture
database_url = database_url_fixture
selected = selected_fixture
_late_bound_sessions = _late_bound_sessions_fixture

RUN = "run-channel-hold"
PV = 9
CID = auto_resume_command_id(PV, 1, RUN)
AUTO_RESUME_SWITCH = "XAGENT_TASK_AUTO_RESUME_ENABLED"
PAUSE_SWITCH = "XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED"
NO_OUTPUTS = _PreparedTaskFileOutputs((), (), ())


@pytest.fixture(autouse=True)
def _default_window(monkeypatch):
    monkeypatch.delenv("XAGENT_TASK_AUTO_RESUME_CHANNEL_WINDOW_SECONDS", raising=False)


def _limit() -> int:
    """Seconds after an interruption the hold ends."""
    return staleness_window_seconds("channel") + CHANNEL_HOLD_CLOCK_GRACE_SECONDS


@pytest.fixture
def factory(engine, monkeypatch) -> sessionmaker:
    Base.metadata.create_all(engine)
    result = sessionmaker(engine, autoflush=False)
    # Set the globals behind the accessors, as the auto-resume suites do.
    monkeypatch.setattr(database, "_SessionLocal", result)
    monkeypatch.setattr(database, "_engine", engine)
    monkeypatch.delenv(AUTO_RESUME_SWITCH, raising=False)
    return result


def _recovery_row(
    db: Session,
    task_id: int,
    run_id: str,
    pv: int,
    *,
    state: str,
    interrupted_ago: float,
    command_status: str | None = None,
) -> str | None:
    """Add ``task_id``'s recovery row; a ``dispatched`` one names the
    sweeper's RESUME, staged with ``command_status`` when given."""
    now = utc_now()
    dispatched = state == "dispatched"
    command_id = auto_resume_command_id(pv, 1, run_id) if dispatched else None
    db.add(
        TaskAutoRecovery(
            task_id=task_id,
            run_id=run_id,
            reason="lease_expired",
            state=state,
            paused_state_version=pv,
            interrupted_at=now - timedelta(seconds=interrupted_ago),
            episode_started_at=now - timedelta(seconds=interrupted_ago),
            total_resumes=int(dispatched),
            no_progress_resumes=int(dispatched),
            next_attempt_at=now - timedelta(seconds=1),
            last_command_id=command_id,
        )
    )
    if command_status is not None:
        db.add(
            TaskExecutionCommand(
                task_id=task_id,
                actor_user_id=db.get(Task, task_id).user_id,
                command_id=command_id,
                kind="resume",
                payload={
                    "type": "resume_task",
                    "auto_resume": {
                        "expected_run_id": run_id,
                        "expected_state_version": pv,
                        "reason": "lease_expired",
                        "attempt": 1,
                    },
                },
                target_run_id=run_id,
                status=command_status,
            )
        )
    db.flush()
    return command_id


def _seed(
    db: Session,
    *,
    state: str | None = "scheduled",
    status: TaskStatus = TaskStatus.PAUSED,
    control_state: str = "paused",
    version: int = PV,
    command_status: str | None = None,
    interrupted_ago: float = 60,
    row_run: str = RUN,
) -> int:
    """A task resting on ``RUN`` and, unless ``state`` is None, its row
    recorded at ``PV``."""
    user = User(username=f"hold-{uuid.uuid4().hex[:8]}", password_hash="x")
    db.add(user)
    db.flush()
    task = Task(
        user_id=user.id,
        title="Hold",
        description="hold test",
        execution_mode="balanced",
        status=status,
        control_state=control_state,
        run_id=RUN,
        state_version=version,
    )
    db.add(task)
    db.flush()
    if state is not None:
        _recovery_row(
            db,
            int(task.id),
            row_run,
            PV,
            state=state,
            interrupted_ago=interrupted_ago,
            command_status=command_status,
        )
    return int(task.id)


# --------------------------------------------------------------------------
# 1. The decision table
# --------------------------------------------------------------------------

_PAST = "past"
_NEAR = "near"

_TABLE = [
    # (id, seed kwargs, expected)
    ("no_row", {"state": None}, False),
    ("other_run", {"row_run": "older-run"}, False),
    *[
        (state, {"state": state}, False)
        for state in (
            "manual",
            "exhausted",
            "expired",
            "ineligible",
            "dispatch_failed",
            "superseded",
            "disabled",
        )
    ],
    ("scheduled", {}, True),
    ("scheduled_inside_the_grace", {"interrupted_ago": _NEAR}, True),
    ("scheduled_past_the_deadline", {"interrupted_ago": _PAST}, False),
    ("scheduled_version_moved", {"version": PV + 1}, False),
    (
        "scheduled_handing_off",
        {"control_state": "resume_requested", "version": PV + 1},
        True,
    ),
    (
        "stale_handing_off",
        {"state": "stale", "control_state": "resume_requested", "version": PV + 1},
        True,
    ),
    (
        "handing_off_past_the_deadline",
        {
            "control_state": "resume_requested",
            "version": PV + 1,
            "interrupted_ago": _PAST,
        },
        False,
    ),
    (
        "dispatched_queued_after_a_rollback",
        {"state": "dispatched", "version": PV + 2, "command_status": "pending"},
        True,
    ),
    (
        "dispatched_processing_past_the_deadline",
        {
            "state": "dispatched",
            "version": PV + 2,
            "command_status": "processing",
            "interrupted_ago": _PAST,
        },
        True,
    ),
    (
        "dispatched_completed_at_the_fence",
        {"state": "dispatched", "command_status": "completed"},
        True,
    ),
    # Housekeeping still has to close it (dispatch_failed) or reschedule it.
    (
        "dispatched_failed_at_the_fence",
        {"state": "dispatched", "command_status": "failed"},
        True,
    ),
    ("dispatched_command_missing", {"state": "dispatched"}, True),
    (
        "dispatched_completed_past_the_deadline",
        {
            "state": "dispatched",
            "command_status": "completed",
            "interrupted_ago": _PAST,
        },
        False,
    ),
    (
        "resumed_run_paused_by_its_user",
        {"state": "dispatched", "version": PV + 3, "command_status": "completed"},
        False,
    ),
    (
        "resume_failed",
        {"state": "dispatched", "version": PV + 2, "command_status": "failed"},
        False,
    ),
    ("stale_paused", {"state": "stale", "version": PV + 1}, False),
]


def _seed_case(factory, seed: dict[str, Any]) -> int:
    seed = dict(seed)
    ago = seed.get("interrupted_ago")
    if ago == _PAST:
        seed["interrupted_ago"] = _limit() + 5
    elif ago == _NEAR:
        # Past the window itself, inside the clock grace.
        seed["interrupted_ago"] = _limit() - 5
    with factory() as db:
        tid = _seed(db, **seed)
        db.commit()
    return tid


@pytest.mark.parametrize(
    ("seed", "expected"),
    [pytest.param(seed, expected, id=name) for name, seed, expected in _TABLE],
)
def test_hold_decision_table(factory, seed, expected):
    tid = _seed_case(factory, seed)
    with factory() as db:
        assert (
            channel_reply_waits_for_auto_resume(
                db, task_id=tid, run_id=RUN, now=utc_now()
            )
            is expected
        )


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        pytest.param(
            {"status": TaskStatus.RUNNING, "control_state": "running"},
            True,
            id="running",
        ),
        pytest.param({"run_id": "rotated"}, True, id="run_rotated"),
        pytest.param(None, False, id="deleted"),
    ],
)
def test_a_task_that_moved_since_the_caller_read_it(factory, change, expected):
    """Anything but PAUSED on the run reads again (True); a deleted task is
    False, so its reader replies "interrupted"."""
    tid = _seed_case(factory, {"state": "manual"})
    with factory() as db:
        if change is None:
            db.delete(db.get(Task, tid))
        else:
            for name, value in change.items():
                setattr(db.get(Task, tid), name, value)
        db.commit()
    with factory() as db:
        assert (
            channel_reply_waits_for_auto_resume(
                db, task_id=tid, run_id=RUN, now=utc_now()
            )
            is expected
        )


# --------------------------------------------------------------------------
# 2. One consistent read
# --------------------------------------------------------------------------


def test_the_hold_snapshot_is_one_statement(factory, engine):
    """The proxy for the single-snapshot requirement: task, recovery row and
    command status must come from one statement, or READ COMMITTED could
    pair versions from two different interruptions."""
    tid = _seed_case(factory, {"state": "dispatched", "command_status": "pending"})
    statements: list[str] = []

    def count(_conn, _cursor, statement, *_args):
        statements.append(statement)

    with factory() as db:
        db.connection()  # BEGIN outside the count
        sa.event.listen(engine, "before_cursor_execute", count)
        try:
            assert channel_reply_waits_for_auto_resume(
                db, task_id=tid, run_id=RUN, now=utc_now()
            )
        finally:
            sa.event.remove(engine, "before_cursor_execute", count)
    assert len(statements) == 1, statements


# --------------------------------------------------------------------------
# 7. The supersede helper
# --------------------------------------------------------------------------


def _supersede(factory, tid: int) -> bool:
    with factory() as db:
        task = db.execute(
            sa.select(Task).where(Task.id == tid).with_for_update()
        ).scalar_one()
        superseded = supersede_pending_auto_resume_no_commit(db, task, now=utc_now())
        db.commit()
    return superseded


@pytest.mark.parametrize(
    ("seed", "version"),
    [
        pytest.param({}, PV, id="scheduled"),
        pytest.param(
            {"state": "dispatched", "command_status": "pending"},
            PV,
            id="dispatched_queued",
        ),
        pytest.param(
            {"state": "dispatched", "command_status": "processing", "version": PV + 2},
            PV + 2,
            id="retrying_after_a_rollback",
        ),
    ],
)
def test_a_new_turn_supersedes_a_resume_that_has_not_handed_off(factory, seed, version):
    tid = _seed_case(factory, seed)

    assert _supersede(factory, tid)

    with factory() as db:
        task = db.get(Task, tid)
        row = db.get(TaskAutoRecovery, tid)
        events = db.scalars(
            sa.select(TaskRecoveryEvent).where(TaskRecoveryEvent.task_id == tid)
        ).all()
        assert int(task.state_version) == version + 1
        assert (task.status, task.control_state) == (TaskStatus.PAUSED, "paused")
        assert (row.state, row.state_detail, row.next_attempt_at) == (
            "stale",
            "new_turn",
            None,
        )
        assert [(e.event, e.detail["at"], e.detail["mismatch"]) for e in events] == [
            ("stale", "new_turn", "new_turn")
        ]
        assert not channel_reply_waits_for_auto_resume(
            db, task_id=tid, run_id=RUN, now=utc_now()
        )


@pytest.mark.parametrize(
    "seed",
    [
        pytest.param(
            {"control_state": "resume_requested", "version": PV + 1},
            id="handing_off",
        ),
        pytest.param(
            {
                "state": "dispatched",
                "command_status": "processing",
                "control_state": "resume_requested",
                "version": PV + 1,
            },
            id="dispatched_handing_off",
        ),
        pytest.param(
            {
                "status": TaskStatus.RUNNING,
                "control_state": "running",
                "version": PV + 2,
            },
            id="running",
        ),
    ],
)
def test_nothing_to_supersede_leaves_the_task_alone(factory, seed):
    """Released rows are the decision table's; these hold, but have begun
    their hand-off."""
    tid = _seed_case(factory, seed)
    with factory() as db:
        before = (
            int(db.get(Task, tid).state_version),
            db.get(TaskAutoRecovery, tid).state,
        )

    assert not _supersede(factory, tid)

    with factory() as db:
        assert (
            int(db.get(Task, tid).state_version),
            db.get(TaskAutoRecovery, tid).state,
        ) == before
        assert not db.scalars(
            sa.select(TaskRecoveryEvent).where(TaskRecoveryEvent.task_id == tid)
        ).all()


# --------------------------------------------------------------------------
# Channel fixtures: a real accepted channel turn whose run rests PAUSED
# --------------------------------------------------------------------------


def _accept_first_turn(selected) -> int:
    selected.delivery_destination = {"chat_id": "conversation"}
    return shared._accept_channel_turn(selected, TaskTurnPayload("hello"), "ingress")


@pytest.fixture
def channel(selected):
    """M1 accepted; its run started and then rested PAUSED on its run."""
    command_id = _accept_first_turn(selected)
    run_id = selected.run_id
    with get_session_local()() as db:
        command = db.get(TaskExecutionCommand, command_id)
        command.status = "completed"
        command.result = {"run_id": run_id, "lease_attempt_id": "attempt"}
        task = db.get(Task, command.task_id)
        task.status = TaskStatus.PAUSED
        task.control_state = "paused"
        task.run_id = run_id
        task.runner_id = None
        task.state_version = int(task.state_version or 0) + 2
        version = int(task.state_version)
        db.commit()
        tid = int(task.id)
    return SimpleNamespace(
        turn=selected,
        command_id=command_id,
        task_id=tid,
        run_id=run_id,
        version=version,
        channel_id=selected.selection.channel_id,
    )


def _record(
    ch, *, state: str = "scheduled", command_status: str | None = None
) -> str | None:
    with get_session_local()() as db:
        command_id = _recovery_row(
            db,
            ch.task_id,
            ch.run_id,
            ch.version,
            state=state,
            interrupted_ago=30,
            command_status=command_status,
        )
        db.commit()
    return command_id


def _delivery(command_id: int) -> TaskChannelDelivery:
    with get_session_local()() as db:
        row = db.get(TaskChannelDelivery, command_id)
        db.expunge_all()
        return row


def _second_turn(ch) -> shared.SharedChannelTurn:
    selection = dataclasses.replace(
        ch.turn.selection,
        is_new_task=False,
        previous_run_id=ch.run_id,
        state_version=ch.version,
    )
    turn = shared.SharedChannelTurn(selection, workspace=SimpleNamespace())
    turn.origin = "origin-token"
    turn.delivery_destination = {"chat_id": "conversation"}
    return turn


# --------------------------------------------------------------------------
# 3. The read path
# --------------------------------------------------------------------------


def test_read_holds_while_a_resume_is_due(channel):
    _record(channel)
    assert shared._read_channel_result(channel.command_id, channel.run_id) is None


def test_read_is_interrupted_once_automatic_resume_stops(channel):
    _record(channel, state="exhausted")
    assert (
        shared._read_channel_result(channel.command_id, channel.run_id)
        == interrupted_channel_result()
    )


def test_a_stored_channel_result_wins_over_the_hold(channel):
    _record(channel)
    answer = {"success": True, "status": "completed", "output": "the answer"}
    with get_session_local()() as db:
        command = db.get(TaskExecutionCommand, channel.command_id)
        command.result = {**command.result, "channel_result": answer}
        db.commit()
    assert shared._read_channel_result(channel.command_id, channel.run_id) == answer


# --------------------------------------------------------------------------
# 6. Delivery under the hold
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("via", ["deliver", "recover"])
async def test_a_held_delivery_stays_pending_without_a_failure(channel, via):
    _record(channel)
    sender = AsyncMock()
    before = datetime.now(timezone.utc)

    if via == "deliver":
        assert not await channel_delivery.deliver_channel_result(
            channel.command_id, sender
        )
    else:
        await channel_delivery.recover_channel_results(channel.channel_id, sender)

    sender.assert_not_awaited()
    row = _delivery(channel.command_id)
    assert (row.status, row.failure_count, row.claim_token) == ("pending", 0, None)
    available_at = row.available_at
    if available_at.tzinfo is None:
        available_at = available_at.replace(tzinfo=timezone.utc)
    assert before + timedelta(seconds=25) <= available_at
    assert available_at <= datetime.now(timezone.utc) + timedelta(seconds=35)


# --------------------------------------------------------------------------
# 8. Acceptance supersedes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("command_status", [None, "pending"])
def test_a_turn_accepted_during_the_hold_supersedes_the_resume(channel, command_status):
    _record(
        channel,
        state="dispatched" if command_status else "scheduled",
        command_status=command_status,
    )

    second = shared._accept_channel_turn(
        _second_turn(channel), TaskTurnPayload("again"), "ingress"
    )

    with get_session_local()() as db:
        start = db.get(TaskExecutionCommand, second)
        task = db.get(Task, channel.task_id)
        row = db.get(TaskAutoRecovery, channel.task_id)
        assert int(task.state_version) == channel.version + 1
        assert start.payload["state_version"] == channel.version + 1
        assert start.target_state_version == channel.version + 1
        assert (row.state, row.state_detail) == ("stale", "new_turn")
    assert (
        shared._read_channel_result(channel.command_id, channel.run_id)
        == interrupted_channel_result()
    )


# --------------------------------------------------------------------------
# 9. A claimed automatic RESUME loses to the accepted turn
# --------------------------------------------------------------------------


@pytest.fixture
def handler(monkeypatch):
    """Stub everything past the admission decisions of ``resume_task``."""

    def snapshot(task_id, _owner, *, actor_user_id=None, actor_is_admin=False):
        with get_session_local()() as db:
            task = db.get(Task, task_id)
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
    return SimpleNamespace(interaction=interaction, background=background)


async def _auto_resume(ch, command_id: str, *, attempt_count: int):
    with get_session_local()() as db:
        user_id = db.get(Task, ch.task_id).user_id
    return await commands.resume_task(
        AsyncMock(),
        ch.task_id,
        {
            "type": "resume_task",
            "auto_resume": {
                "expected_run_id": ch.run_id,
                "expected_state_version": ch.version,
                "reason": "lease_expired",
                "attempt": 1,
            },
            "user": SimpleNamespace(id=user_id, is_admin=False),
            "_durable_command_id": command_id,
            "_durable_attempt_count": attempt_count,
        },
    )


def _claimable(db: Session) -> list[int]:
    return [c.id for c in _claimable_query(db, runner_id="worker", command_db_id=None)]


@pytest.mark.asyncio
async def test_a_turn_accepted_after_the_guard_defers_the_resume_and_runs(
    channel, handler
):
    auto_command_id = _record(channel, state="dispatched", command_status="processing")
    accepted: list[int] = []

    def accept_between_guard_and_transition(_task_id):
        # M2 lands after the claim guard let the RESUME through, before its
        # RESUME_REQUESTED transition.
        accepted.append(
            shared._accept_channel_turn(
                _second_turn(channel), TaskTurnPayload("again"), "ingress"
            )
        )
        return ActiveInteractionAbsent()

    handler.interaction.side_effect = accept_between_guard_and_transition
    first = await _auto_resume(channel, auto_command_id, attempt_count=1)

    assert first.outcome is ResumeCommandOutcome.DEFERRED
    with get_session_local()() as db:
        task = db.get(Task, channel.task_id)
        assert (task.status, task.control_state) == (TaskStatus.PAUSED, "paused")
        assert int(task.state_version) == channel.version + 1

    handler.interaction.side_effect = None
    retry = await _auto_resume(channel, auto_command_id, attempt_count=2)

    assert retry.outcome is ResumeCommandOutcome.AUTO_SKIPPED
    handler.background.assert_not_called()
    with get_session_local()() as db:
        row = db.get(TaskAutoRecovery, channel.task_id)
        assert (row.state, row.state_detail) == ("stale", "new_turn")
        # Serialized behind the RESUME until it completes (auto_skipped).
        assert accepted[0] not in _claimable(db)
        auto = db.scalars(
            sa.select(TaskExecutionCommand).where(
                TaskExecutionCommand.command_id == auto_command_id
            )
        ).one()
        auto.status = "completed"
        auto.result = {"resume_outcome": "auto_skipped"}
        db.commit()
    with get_session_local()() as db:
        assert _claimable(db) == [accepted[0]]
        # START's own admission checks (task_start_consumer) pass.
        task = db.get(Task, channel.task_id)
        start = db.get(TaskExecutionCommand, accepted[0])
        assert start.target_state_version == start.payload["state_version"]
        assert task.run_id == start.payload["expected_run_id"] == channel.run_id
        assert int(task.state_version) == start.payload["state_version"]
        assert task.status != TaskStatus.RUNNING
    assert (
        shared._read_channel_result(channel.command_id, channel.run_id)
        == interrupted_channel_result()
    )


# --------------------------------------------------------------------------
# 4. The shared channel leaf (S4)
# --------------------------------------------------------------------------


def _force_recorded(monkeypatch, recorded: str) -> None:
    """Settle with the pause switches on, recording ``recorded``."""
    monkeypatch.setenv(PAUSE_SWITCH, "true")
    monkeypatch.setenv(AUTO_RESUME_SWITCH, "true")
    if recorded == "raises":
        _fail_recording(monkeypatch, "python")
        return
    original = task_auto_recovery.record_interruption_no_commit

    def force(db, **kwargs):
        row = original(db, **kwargs)
        row.state = recorded
        if recorded != "scheduled":
            row.next_attempt_at = None
        db.flush()
        return row

    monkeypatch.setattr(task_auto_recovery, "record_interruption_no_commit", force)


@pytest.mark.parametrize("recorded", ["scheduled", "exhausted", "manual", "raises"])
def test_the_leaf_holds_its_reply_only_for_a_scheduled_resume(
    canonical, monkeypatch, recorded
):
    _force_recorded(monkeypatch, recorded)
    factory, _tid = canonical
    ids = _projected_task(factory)
    lease = _prepare_run(factory, ids)
    report = SettlementReport()

    _settle_channel(factory, ids, lease, LLM_RESULT, report=report)

    with factory() as db:
        task = db.get(Task, ids["task"])
        result = db.get(TaskExecutionCommand, ids["command"]).result
        assert task.status == TaskStatus.PAUSED
        assert task.output == "previous answer"
    assert report.paused
    if recorded == "scheduled":
        assert report.recovery.state == "scheduled"
        assert result == {"run_id": lease.run_id, "lease_attempt_id": lease.attempt_id}
    else:
        assert result["channel_result"] == interrupted_channel_result()
        assert (report.recovery is None) is (recorded == "raises")


# --------------------------------------------------------------------------
# 5. The resumed run stores the channel answer
# --------------------------------------------------------------------------

_INTERACTIONS = [
    {"type": "select_one", "field": "choice", "label": "Pick", "options": ["a", "b"]}
]
_COMPLETED = {
    "success": True,
    "status": "completed",
    "output": "The resumed answer",
    "completion_outcome": "completed",
}


def _channel_task_run(
    factory, *, channel_payload: bool = True, start: bool = True, stored=None
):
    ids = _projected_task(factory)
    with factory() as db:
        user_channel = UserChannel(
            user_id=ids["user"],
            channel_type="feishu",
            channel_name="hold",
            config={},
            is_active=True,
        )
        db.add(user_channel)
        db.flush()
        db.get(Task, ids["task"]).channel_id = user_channel.id
        db.commit()
        channel_id = int(user_channel.id)
    lease = _prepare_run(factory, ids)
    if start:
        result: dict[str, Any] = {"run_id": lease.run_id, "lease_attempt_id": "first"}
        if stored is not None:
            result["channel_result"] = stored
        with factory() as db:
            command = TaskExecutionCommand(
                task_id=ids["task"],
                command_id=f"start-{ids['turn_id']}",
                kind="start",
                payload={
                    "channel": (
                        {"channel_id": channel_id, "external_user_id": "sender"}
                        if channel_payload
                        else None
                    )
                },
                target_run_id=lease.run_id,
                status="completed",
                result=result,
            )
            db.add(command)
            db.commit()
            ids["command"] = int(command.id)
    return ids, lease


def _finalize(ids, lease, result: dict[str, Any]) -> dict[str, Any]:
    return _finalize_resumed_task(
        ids["task"],
        status=str(result.get("status") or ""),
        success=bool(result.get("success")),
        output=str(result.get("output") or result.get("error") or ""),
        task_owner_user_id=ids["user"],
        result=dict(result),
        task_lease=lease,
        prepared_outputs=NO_OUTPUTS,
    )


def _start_result(factory, ids) -> dict[str, Any]:
    with factory() as db:
        return dict(db.get(TaskExecutionCommand, ids["command"]).result)


@pytest.mark.parametrize(
    ("result", "stored"),
    [
        pytest.param(
            {
                **_COMPLETED,
                # The transcript stores ``output``; the leaf would show this.
                "chat_response": {"message": "unrewritten text", "interactions": []},
            },
            {
                "success": True,
                "status": "completed",
                "output": "The resumed answer",
                "completion_outcome": "completed",
                "chat_response": {"message": "The resumed answer", "interactions": []},
            },
            id="completed_keeps_the_stored_transcript",
        ),
        pytest.param(
            {
                "success": False,
                "status": "failed",
                "output": "workbook locked",
                "error": "workbook locked",
                "failure_code": "excel_workbook_locked",
            },
            {
                "success": False,
                "status": "failed",
                "output": CLIENT_SAFE_TASK_FAILURE,
                "completion_outcome": None,
                "chat_response": {
                    "message": CLIENT_SAFE_TASK_FAILURE,
                    "interactions": [],
                },
                "failure_code": "excel_workbook_locked",
            },
            id="failed_coded",
        ),
        pytest.param(
            {
                "success": True,
                "status": "waiting_for_user",
                "output": "Which one?",
                "chat_response": {
                    "message": "Which one?",
                    "interactions": _INTERACTIONS,
                },
            },
            {
                "success": True,
                "status": "waiting_for_user",
                "output": "Which one?",
                "completion_outcome": None,
                "chat_response": {
                    "message": "Which one?",
                    "interactions": _INTERACTIONS,
                },
            },
            id="waiting_for_user",
        ),
    ],
)
def test_the_resumed_run_stores_its_channel_result(canonical, result, stored):
    factory, _tid = canonical
    ids, lease = _channel_task_run(factory)

    assert _finalize(ids, lease, result)["lease_released"]

    assert _start_result(factory, ids)["channel_result"] == stored
    if result["status"] == "completed":
        with factory() as db:
            assert db.get(Task, ids["task"]).output == stored["output"]


@pytest.mark.parametrize("result", ["interrupted", "paused_again"])
def test_a_resumed_run_that_pauses_stores_nothing(canonical, monkeypatch, result):
    monkeypatch.setenv(PAUSE_SWITCH, "true")
    monkeypatch.setenv(AUTO_RESUME_SWITCH, "true")
    factory, _tid = canonical
    ids, lease = _channel_task_run(factory)
    payload = (
        {"success": False, "status": "interrupted", "output": ""}
        if result == "interrupted"
        else LLM_RESULT
    )

    assert _finalize(ids, lease, payload)["final_status"] == "paused"
    assert "channel_result" not in _start_result(factory, ids)


@pytest.mark.parametrize("case", ["already_stored", "not_a_channel_turn", "no_start"])
def test_a_start_with_nothing_to_stamp_is_left_alone(canonical, case):
    factory, _tid = canonical
    ids, lease = _channel_task_run(
        factory,
        stored=interrupted_channel_result() if case == "already_stored" else None,
        channel_payload=case != "not_a_channel_turn",
        start=case != "no_start",
    )

    finalized = _finalize(ids, lease, _COMPLETED)

    assert finalized["final_status"] == "completed"
    if case != "no_start":
        assert _start_result(factory, ids).get("channel_result") == (
            interrupted_channel_result() if case == "already_stored" else None
        )


@pytest.mark.parametrize("failure", ["raises", "update_fails"])
def test_a_failed_stamp_never_undoes_the_settlement(canonical, monkeypatch, failure):
    """``raises``: the stamp itself fails. ``update_fails``: the stamp only
    assigns the START's result, and the UPDATE fails when it is flushed."""
    factory, _tid = canonical
    ids, lease = _channel_task_run(factory)
    original = shared.stamp_channel_result_no_commit

    def stamp_then_fail(db, **kwargs):
        if failure == "update_fails":
            # Not JSON-serializable: only the flush of the UPDATE fails.
            kwargs["channel_result"] = {**kwargs["channel_result"], "x": object()}
            return original(db, **kwargs)
        assert original(db, **kwargs)
        db.flush()
        raise RuntimeError("stamp failed")

    monkeypatch.setattr(shared, "stamp_channel_result_no_commit", stamp_then_fail)

    finalized = _finalize(ids, lease, _COMPLETED)

    assert finalized["lease_released"]
    with factory() as db:
        task = db.get(Task, ids["task"])
        assert (task.status, task.output) == (
            TaskStatus.COMPLETED,
            "The resumed answer",
        )
    assert "channel_result" not in _start_result(factory, ids)


def test_a_rolled_back_settlement_stores_nothing(canonical, monkeypatch):
    factory, _tid = canonical
    ids, lease = _channel_task_run(factory)
    monkeypatch.setattr(
        task_execution_service,
        "release_task_lease_no_commit",
        lambda *_args, **_kwargs: False,
    )

    assert _finalize(ids, lease, _COMPLETED)["late_result"]
    assert "channel_result" not in _start_result(factory, ids)


# --------------------------------------------------------------------------
# The whole chain: leaf holds, delivery waits, resumed run answers once
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_held_reply_is_delivered_once_with_the_resumed_answer(
    selected, monkeypatch
):
    _force_recorded(monkeypatch, "scheduled")
    command_id = _accept_first_turn(selected)
    with get_session_local()() as db:
        tid = int(db.get(TaskExecutionCommand, command_id).task_id)
        db.get(Task, tid).conversation_storage_version = 2
        db.commit()
        lease = acquire_task_lease(db, tid, new_run=True)
    # The START hand-off this run came from.
    with get_session_local()() as db:
        start = db.get(TaskExecutionCommand, command_id)
        start.target_run_id = lease.run_id
        start.status = "completed"
        start.result = {"run_id": lease.run_id, "lease_attempt_id": lease.attempt_id}
        db.commit()
    _checkpoint(get_session_local(), tid, lease.run_id)

    with get_session_local()() as db:
        assert finalize_managed_task_lease_result(
            db,
            lease,
            status=TaskStatus.FAILED,
            execution_result=dict(LLM_RESULT),
            completion=(command_id, {"success": False, "status": "failed"}),
            settle_interruption=True,
        )
    with get_session_local()() as db:
        assert db.get(Task, tid).status == TaskStatus.PAUSED
        assert db.get(TaskAutoRecovery, tid).state == "scheduled"

    sender = AsyncMock()
    assert not await channel_delivery.deliver_channel_result(command_id, sender)
    sender.assert_not_awaited()

    # The automatic resume runs the same run to completion.
    with get_session_local()() as db:
        resumed = acquire_task_lease(db, tid, expected_run_id=lease.run_id)
    assert resumed is not None and resumed.run_id == lease.run_id
    finalized = _finalize_resumed_task(
        tid,
        status="completed",
        success=True,
        output="The resumed answer",
        task_owner_user_id=selected.selection.user_id,
        result=dict(_COMPLETED),
        task_lease=resumed,
        prepared_outputs=NO_OUTPUTS,
    )
    assert finalized["final_status"] == "completed"

    expire_claim(command_id)
    await channel_delivery.recover_channel_results(
        selected.selection.channel_id, sender
    )
    assert not await channel_delivery.deliver_channel_result(command_id, sender)

    sender.assert_awaited_once()
    sent = sender.await_args.args[1]
    assert (sent["status"], sent["output"]) == ("completed", "The resumed answer")
    assert _delivery(command_id).status == "delivered"


# --------------------------------------------------------------------------
# 16-17. PostgreSQL: lock order with the sweeper and the claim guard
# --------------------------------------------------------------------------


def _postgresql_only() -> None:
    if get_engine().dialect.name != "postgresql":
        pytest.skip("needs two real PostgreSQL transactions")


@pytest.fixture
def short_lock_timeout():
    """Every new connection waits at most 5s on a lock: a deadlock or a
    wrong lock order fails instead of hanging."""
    engine = get_engine()
    engine.dispose()

    def apply(connection, _record):
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute("SET lock_timeout = '5s'")
        connection.autocommit = False

    sa.event.listen(engine, "connect", apply)
    yield
    sa.event.remove(engine, "connect", apply)
    engine.dispose()


def test_postgresql_sweeper_skips_a_task_an_acceptance_holds(
    channel, short_lock_timeout
):
    _postgresql_only()
    _record(channel)
    scheduled = task_auto_resume.TaskAutoRecoveryState.SCHEDULED
    now = utc_now()
    with get_session_local()() as db:
        assert select_next_candidate_for_update(db, scheduled, now=now) is not None
        db.rollback()

    accepting = get_session_local()()
    try:
        shared.accept_channel_turn_no_commit(
            accepting, _second_turn(channel), TaskTurnPayload("again"), "ingress"
        )
        with get_session_local()() as sweeper:
            # SKIP LOCKED: the acceptance holds the task row.
            assert select_next_candidate_for_update(sweeper, scheduled, now=now) is None
            sweeper.rollback()
        accepting.commit()
    finally:
        accepting.close()

    with get_session_local()() as db:
        # Superseded: nothing left for the next tick.
        assert select_next_candidate_for_update(db, scheduled, now=now) is None
        assert db.get(TaskAutoRecovery, channel.task_id).state == "stale"


def _wait_for_a_lock_waiter(monitor: Session, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        waiting = monitor.execute(
            sa.text(
                "SELECT count(*) FROM pg_stat_activity "
                "WHERE datname = current_database() AND wait_event_type = 'Lock' "
                "AND pid <> pg_backend_pid()"
            )
        ).scalar_one()
        monitor.rollback()
        if waiting:
            return
        time.sleep(0.05)
    raise AssertionError("the guard never waited on the acceptance's locks")


@pytest.mark.parametrize("snapshot", ["at_fence", "moved"])
def test_postgresql_claim_guard_waits_for_an_acceptance_without_deadlock(
    channel, short_lock_timeout, snapshot
):
    """At the fence the guard locks the recovery row; past it, the task row
    first. Either way it waits for the acceptance (which holds both, tasks
    first) and then finds the row superseded."""
    _postgresql_only()
    command_id = _record(channel, state="dispatched", command_status="processing")
    outcome: dict[str, Any] = {}

    def guard() -> None:
        try:
            outcome["decision"] = check_auto_resume_claim_sync(
                task_id=channel.task_id,
                command_id=command_id,
                auto_resume={
                    "expected_run_id": channel.run_id,
                    "expected_state_version": channel.version,
                },
                status=TaskStatus.PAUSED,
                control_state="paused",
                run_id=channel.run_id,
                state_version=(
                    channel.version if snapshot == "at_fence" else channel.version + 5
                ),
                attempt_count=1,
            )
        except BaseException as exc:  # pragma: no cover - reported below
            outcome["error"] = exc

    accepting = get_session_local()()
    thread = threading.Thread(target=guard)
    try:
        shared.accept_channel_turn_no_commit(
            accepting, _second_turn(channel), TaskTurnPayload("again"), "ingress"
        )
        thread.start()
        with get_session_local()() as monitor:
            _wait_for_a_lock_waiter(monitor)
        accepting.commit()
    finally:
        accepting.close()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert "error" not in outcome, outcome.get("error")
    assert outcome["decision"].proceed is False
    with get_session_local()() as db:
        row = db.get(TaskAutoRecovery, channel.task_id)
        assert (row.state, row.state_detail) == ("stale", "new_turn")


# --------------------------------------------------------------------------
# 10-14. Shared channel tasks are scheduled (9b)
# --------------------------------------------------------------------------


def _paused_channel_turn(selected, path: str) -> SimpleNamespace:
    """M1 accepted, its run started and stopped for an unavailable model,
    settled by the channel leaf (S4) or the exception path (S2)."""
    command_id = _accept_first_turn(selected)
    with get_session_local()() as db:
        tid = int(db.get(TaskExecutionCommand, command_id).task_id)
        db.get(Task, tid).conversation_storage_version = 2
        db.commit()
        lease = acquire_task_lease(db, tid, new_run=True)
    with get_session_local()() as db:
        start = db.get(TaskExecutionCommand, command_id)
        start.target_run_id = lease.run_id
        start.status = "completed"
        start.result = {"run_id": lease.run_id, "lease_attempt_id": lease.attempt_id}
        db.commit()
    _checkpoint(get_session_local(), tid, lease.run_id)
    if path == "leaf":
        with get_session_local()() as db:
            assert finalize_managed_task_lease_result(
                db,
                lease,
                status=TaskStatus.FAILED,
                execution_result=dict(LLM_RESULT),
                completion=(command_id, {"success": False, "status": "failed"}),
                settle_interruption=True,
            )
    else:
        assert settle_task_lease_isolated(
            lease,
            error_message="setup/run error: LLMTimeoutError: provider timed out",
            interruption=InterruptionReason.LLM_UNAVAILABLE,
        )
    with get_session_local()() as db:
        task = db.get(Task, tid)
        assert (task.status, task.control_state) == (TaskStatus.PAUSED, "paused")
        version = int(task.state_version)
    return SimpleNamespace(
        turn=selected,
        command_id=command_id,
        task_id=tid,
        run_id=lease.run_id,
        version=version,
        channel_id=selected.selection.channel_id,
    )


def _row_state(tid: int) -> tuple[str, str | None]:
    with get_session_local()() as db:
        row = db.get(TaskAutoRecovery, tid)
        return row.state, row.state_detail


@pytest.mark.parametrize("path", ["leaf", "exception"])
@pytest.mark.parametrize("auto_resume", ["true", "false"])
def test_a_shared_channel_interruption_is_scheduled_and_held(
    selected, monkeypatch, path, auto_resume
):
    """S4 and S2 (S5: ``test_eligible_tasks_record_their_kind``)."""
    monkeypatch.setenv(PAUSE_SWITCH, "true")
    monkeypatch.setenv(AUTO_RESUME_SWITCH, auto_resume)

    ch = _paused_channel_turn(selected, path)

    stored = _start_result_of(ch.command_id).get("channel_result")
    read = shared._read_channel_result(ch.command_id, ch.run_id)
    if auto_resume == "true":
        assert _row_state(ch.task_id) == ("scheduled", None)
        assert stored is None
        assert read is None
    else:
        assert _row_state(ch.task_id) == ("manual", "auto_resume_disabled")
        # The leaf stores it; after S2 the reader derives the same reply.
        assert stored == (interrupted_channel_result() if path == "leaf" else None)
        assert read == interrupted_channel_result()


def _start_result_of(command_id: int) -> dict[str, Any]:
    with get_session_local()() as db:
        return dict(db.get(TaskExecutionCommand, command_id).result or {})


def _due_tick(tid: int) -> task_auto_resume.AutoResumeTickReport:
    with get_session_local()() as db:
        due = db.get(TaskAutoRecovery, tid).next_attempt_at
    if due.tzinfo is None:
        due = due.replace(tzinfo=timezone.utc)
    return task_auto_resume.run_auto_resume_tick(now=due + timedelta(seconds=1))


@pytest.mark.parametrize("switch", ["on", "drained"])
def test_the_sweeper_dispatches_a_channel_row_or_drains_it(
    selected, monkeypatch, switch
):
    monkeypatch.setenv(PAUSE_SWITCH, "true")
    monkeypatch.setenv(AUTO_RESUME_SWITCH, "true")
    ch = _paused_channel_turn(selected, "leaf")
    if switch == "drained":
        monkeypatch.setenv(AUTO_RESUME_SWITCH, "false")

    _due_tick(ch.task_id)

    read = shared._read_channel_result(ch.command_id, ch.run_id)
    if switch == "on":
        assert _row_state(ch.task_id) == ("dispatched", None)
        with get_session_local()() as db:
            (auto,) = db.scalars(
                sa.select(TaskExecutionCommand).where(
                    TaskExecutionCommand.task_id == ch.task_id,
                    TaskExecutionCommand.kind == "resume",
                )
            ).all()
            assert auto.status == "pending"
        assert read is None
    else:
        assert _row_state(ch.task_id) == ("manual", "auto_resume_disabled")
        assert read == interrupted_channel_result()


@pytest.mark.parametrize("state", ["handing_off", "running"])
def test_a_turn_sent_while_the_resume_hands_off_or_runs_is_busy(channel, state):
    auto_command_id = _record(channel, state="dispatched", command_status="processing")
    with get_session_local()() as db:
        task = db.get(Task, channel.task_id)
        if state == "handing_off":
            task.control_state = "resume_requested"
        else:
            task.status = TaskStatus.RUNNING
            task.control_state = "running"
        db.commit()

    with pytest.raises(TaskTurnError) as refused:
        shared._accept_channel_turn(
            _second_turn(channel), TaskTurnPayload("again"), "ingress"
        )

    assert refused.value.reason == "busy"
    with get_session_local()() as db:
        assert int(db.get(Task, channel.task_id).state_version) == channel.version
        assert _row_state(channel.task_id) == ("dispatched", None)
        assert [
            c.command_id
            for c in db.scalars(
                sa.select(TaskExecutionCommand).where(
                    TaskExecutionCommand.task_id == channel.task_id
                )
            )
            if c.id != channel.command_id
        ] == [auto_command_id]
    # M1 still waits for the resumed run.
    assert shared._read_channel_result(channel.command_id, channel.run_id) is None


def _hand_off(ch, *, row: dict[str, Any] | None, manual_resume: str | None) -> None:
    """The task rests PAUSED, ``resume_requested``, on its run: optionally a
    recovery row (``_recovery_row`` kwargs) and a person's RESUME command
    with status ``manual_resume``."""
    with get_session_local()() as db:
        if row is not None:
            _recovery_row(db, ch.task_id, ch.run_id, ch.version, **row)
        if manual_resume is not None:
            db.add(
                TaskExecutionCommand(
                    task_id=ch.task_id,
                    actor_user_id=db.get(Task, ch.task_id).user_id,
                    command_id=f"manual-resume-{uuid.uuid4().hex[:8]}",
                    kind="resume",
                    payload={"type": "resume_task"},
                    target_run_id=ch.run_id,
                    status=manual_resume,
                )
            )
        db.get(Task, ch.task_id).control_state = "resume_requested"
        db.commit()


@pytest.mark.parametrize(
    ("row", "manual_resume", "busy"),
    [
        pytest.param(None, "pending", True, id="manual_resume_queued"),
        pytest.param(
            {"state": "manual", "interrupted_ago": 30},
            "processing",
            True,
            id="manual_resume_running_on_a_manual_row",
        ),
        # Its command ended, but an automatic resume within the window still
        # holds the reply: its run may yet take the lease.
        pytest.param(
            {
                "state": "dispatched",
                "interrupted_ago": 30,
                "command_status": "completed",
            },
            None,
            True,
            id="automatic_within_the_window",
        ),
        # Stranded: nothing can still start the resumed run.
        pytest.param(None, "completed", False, id="stranded_manual"),
        pytest.param(
            {
                "state": "dispatched",
                "interrupted_ago": "past",
                "command_status": "completed",
            },
            None,
            False,
            id="stranded_automatic_past_the_window",
        ),
    ],
)
def test_a_turn_is_busy_only_while_the_hand_off_can_complete(
    channel, row, manual_resume, busy
):
    if row is not None and row["interrupted_ago"] == "past":
        row = {**row, "interrupted_ago": _limit() + 5}
    _hand_off(channel, row=row, manual_resume=manual_resume)
    with get_session_local()() as db:
        before = {
            c.id
            for c in db.scalars(
                sa.select(TaskExecutionCommand).where(
                    TaskExecutionCommand.task_id == channel.task_id
                )
            )
        }

    def accept() -> int:
        return shared._accept_channel_turn(
            _second_turn(channel), TaskTurnPayload("again"), "ingress"
        )

    if busy:
        with pytest.raises(TaskTurnError) as refused:
            accept()
        assert refused.value.reason == "busy"
    else:
        accepted = accept()
    with get_session_local()() as db:
        assert int(db.get(Task, channel.task_id).state_version) == channel.version
        added = [
            c
            for c in db.scalars(
                sa.select(TaskExecutionCommand).where(
                    TaskExecutionCommand.task_id == channel.task_id
                )
            )
            if c.id not in before
        ]
        if busy:
            assert added == []
        else:
            (start,) = added
            assert (start.id, start.kind, start.status) == (
                accepted,
                "start",
                "pending",
            )
            assert start.payload["state_version"] == channel.version


@pytest.mark.asyncio
async def test_an_unanswered_question_ends_the_hold_within_the_limit(
    channel, handler, monkeypatch
):
    """The interaction gate is unreachable today; made reachable, the
    automatic resume is skipped, rescheduled and finally exhausted, and the
    held reply is released once."""
    monkeypatch.setenv(AUTO_RESUME_SWITCH, "true")
    auto_command_id = _record(channel, state="dispatched", command_status="processing")
    handler.interaction.return_value = ActiveInteractionFound(interaction_id=1)

    skipped = await _auto_resume(channel, auto_command_id, attempt_count=1)

    assert skipped.outcome is ResumeCommandOutcome.AUTO_SKIPPED
    assert skipped.reason_code == "interaction_pending"
    handler.background.assert_not_called()
    assert shared._read_channel_result(channel.command_id, channel.run_id) is None
    with get_session_local()() as db:
        auto = db.scalars(
            sa.select(TaskExecutionCommand).where(
                TaskExecutionCommand.command_id == auto_command_id
            )
        ).one()
        auto.status = "completed"
        auto.result = {"resume_outcome": "auto_skipped"}
        db.commit()

    _due_tick(channel.task_id)  # housekeeping: the resume did not happen
    assert _row_state(channel.task_id)[0] == "scheduled"
    assert shared._read_channel_result(channel.command_id, channel.run_id) is None

    with get_session_local()() as db:
        row = db.get(TaskAutoRecovery, channel.task_id)
        row.no_progress_resumes = row.total_resumes = 3
        db.commit()
    _due_tick(channel.task_id)

    assert _row_state(channel.task_id)[0] == "exhausted"
    assert (
        shared._read_channel_result(channel.command_id, channel.run_id)
        == interrupted_channel_result()
    )
