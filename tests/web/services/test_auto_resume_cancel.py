"""A person cancels the automatic resume a paused run waits for.

``cancel_pending_auto_resume_no_commit`` moves ``state_version`` and the
recovery row to the new fence, in ``manual(user_cancelled)``: the notice it
returns and a task read at that fence agree on ``stopped`` / ``cancelled``,
and a RESUME the sweeper already queued (or even let past its claim guard)
completes as ``auto_skipped`` without resuming the run. Nothing calls the
hook yet. Runs on SQLite and, when ``XAGENT_TEST_POSTGRES_URL`` is set, on
PostgreSQL; the lock cases are PostgreSQL only and the compare-and-swap
cases SQLite only.
"""

from __future__ import annotations

import threading
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest
import sqlalchemy as sa

from tests.web.services.task_database_shared import engine as engine_fixture
from tests.web.services.task_database_shared import task_id as task_id_fixture
from tests.web.services.test_channel_reply_hold import (
    CID,
    PV,
    RUN,
    _auto_resume,
    _postgresql_only,
    _recovery_row,
    _seed,
    _wait_for_a_lock_waiter,
)
from tests.web.services.test_channel_reply_hold import factory as factory_fixture
from tests.web.services.test_channel_reply_hold import handler as handler_fixture
from tests.web.services.test_channel_reply_hold import (
    short_lock_timeout as short_lock_timeout_fixture,
)
from tests.web.services.test_resumed_trigger_run_projection import (
    _finalize,
    _resume,
    _trigger_run,
    _trigger_run_state,
)
from tests.web.services.test_task_auto_resume import _trigger
from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from xagent.core.agent.interruption import InterruptionReason
from xagent.web.models.database import get_engine, get_session_local
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_auto_recovery import (
    TaskAutoRecovery,
    TaskAutoRecoveryState,
    TaskRecoveryEvent,
)
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.trigger import TriggerRun, TriggerRunStatus
from xagent.web.models.user import User
from xagent.web.services import task_auto_resume
from xagent.web.services.task_auto_recovery import (
    current_auto_recovery_view,
    record_interruption_no_commit,
)
from xagent.web.services.task_auto_resume import (
    AutoResumeCancel,
    cancel_pending_auto_resume_no_commit,
    check_auto_resume_claim_sync,
    run_auto_resume_tick,
    select_next_candidate_for_update,
    supersede_pending_auto_resume_no_commit,
)
from xagent.web.services.task_command_execution import ResumeCommandOutcome
from xagent.web.services.task_execution_controller import TaskControlState
from xagent.web.services.task_interaction_close import ActiveInteractionAbsent
from xagent.web.services.task_lease_service import utc_now
from xagent.web.services.task_settlement_report import RecordedInterruption

engine = engine_fixture
task_id = task_id_fixture
canonical = canonical_fixture
factory = factory_fixture
handler = handler_fixture
short_lock_timeout = short_lock_timeout_fixture

NOTHING = AutoResumeCancel(cancelled=False)
AT = "channel_stop"


@pytest.fixture(autouse=True)
def _quiet_dispatcher(monkeypatch):
    monkeypatch.setattr(
        task_auto_resume, "notify_task_command_dispatcher", lambda: None
    )


def _seeded(
    factory,
    *,
    state_detail: str | None = None,
    user_command_after: bool = False,
    **seed: Any,
) -> int:
    with factory() as db:
        tid = _seed(db, **seed)
        if state_detail is not None:
            db.get(TaskAutoRecovery, tid).state_detail = state_detail
        if user_command_after:
            # A person's command reached the task after the dispatch.
            db.add(
                TaskExecutionCommand(
                    task_id=tid,
                    actor_user_id=db.get(Task, tid).user_id,
                    command_id="user-pause-1",
                    kind="pause",
                    payload={},
                    target_run_id=RUN,
                    status="completed",
                )
            )
        db.commit()
    return tid


def _locked(db, tid: int) -> Task:
    return db.execute(
        sa.select(Task).where(Task.id == tid).with_for_update()
    ).scalar_one()


def _cancel(factory, tid: int) -> AutoResumeCancel:
    """The caller's transaction: lock the task, cancel, commit or roll back."""
    with factory() as db:
        result = cancel_pending_auto_resume_no_commit(db, _locked(db, tid), at=AT)
        if result.cancelled:
            db.commit()
        else:
            db.rollback()
    return result


def _state(factory, tid: int) -> tuple[Any, ...]:
    with factory() as db:
        task = db.get(Task, tid)
        row = db.get(TaskAutoRecovery, tid)
        events = db.scalars(
            sa.select(TaskRecoveryEvent.event).where(TaskRecoveryEvent.task_id == tid)
        ).all()
        return (
            int(task.state_version),
            None if row is None else (row.state, row.state_detail),
            None if row is None else int(row.paused_state_version),
            list(events),
        )


def _command_status(factory, tid: int) -> str:
    with factory() as db:
        return db.scalars(
            sa.select(TaskExecutionCommand.status).where(
                TaskExecutionCommand.task_id == tid,
                TaskExecutionCommand.command_id == CID,
            )
        ).one()


# --------------------------------------------------------------------------
# C1. What is still pending
# --------------------------------------------------------------------------

_DISPATCHED = {"state": "dispatched"}
_PENDING = [
    pytest.param({}, PV, id="scheduled"),
    *[
        pytest.param({**_DISPATCHED, "command_status": s}, PV, id=f"dispatched_{s}")
        for s in ("pending", "processing", "completed", "failed")
    ],
    pytest.param(_DISPATCHED, PV, id="dispatched_command_missing"),
    *[
        pytest.param(
            {**_DISPATCHED, "command_status": s, "version": PV + 2},
            PV + 2,
            id=f"retry_after_a_rollback_{s}",
        )
        for s in ("pending", "processing")
    ],
]
_NOT_PENDING = [
    pytest.param({"version": PV + 1}, id="scheduled_off_the_fence"),
    *[
        pytest.param(
            {**_DISPATCHED, "command_status": s, "version": PV + 2},
            id=f"rolled_back_{s}",
        )
        for s in ("completed", "failed")
    ],
    # Off the fence a live command still pending is only a retry after the
    # resume it rolled back: two versions on, nothing else since.
    *[
        pytest.param(
            {**_DISPATCHED, "command_status": "pending", "version": PV + gap},
            id=f"live_command_{gap}_versions_on",
        )
        for gap in (1, 3)
    ],
    pytest.param(
        {
            **_DISPATCHED,
            "command_status": "pending",
            "version": PV + 2,
            "user_command_after": True,
        },
        id="live_command_after_a_user_command",
    ),
    pytest.param(
        {
            **_DISPATCHED,
            "command_status": "processing",
            "control_state": "resume_requested",
            "version": PV + 1,
        },
        id="handing_off",
    ),
    *[
        pytest.param({"status": status, "control_state": status.value}, id=status.value)
        for status in (
            TaskStatus.RUNNING,
            TaskStatus.WAITING_FOR_USER,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
        )
    ],
    pytest.param({"state": None}, id="no_row"),
    pytest.param({"row_run": "older-run"}, id="other_run"),
    pytest.param(
        {"state": "manual", "state_detail": "user_cancelled"}, id="already_cancelled"
    ),
    *[
        pytest.param({"state": state}, id=state)
        for state in (
            "manual",
            "expired",
            "exhausted",
            "superseded",
            "dispatch_failed",
            "ineligible",
            "stale",
            "disabled",
        )
    ],
]


@pytest.mark.parametrize(("seed", "fence"), _PENDING)
def test_a_pending_resume_is_cancelled(factory, seed, fence):
    tid = _seeded(factory, **seed)

    result = _cancel(factory, tid)

    assert (result.cancelled, result.raced) == (True, False)
    assert (result.run_id, result.state_version) == (RUN, fence + 1)
    assert _state(factory, tid) == (
        fence + 1,
        ("manual", "user_cancelled"),
        fence + 1,
        ["skipped_user_intent"],
    )


def _unwritten(db, tid: int, version: int) -> None:
    """Nothing staged in ``db``: no pending ORM change, nothing flushed."""
    assert not db.dirty and not db.new
    assert db.scalar(sa.select(Task.state_version).where(Task.id == tid)) == version
    assert not db.scalars(
        sa.select(TaskRecoveryEvent.id).where(TaskRecoveryEvent.task_id == tid)
    ).all()


@pytest.mark.parametrize("seed", _NOT_PENDING)
def test_nothing_pending_is_left_alone(factory, seed):
    tid = _seeded(factory, **seed)
    before = _state(factory, tid)

    with factory() as db:
        result = cancel_pending_auto_resume_no_commit(db, _locked(db, tid), at=AT)
        # Not raced: a hand-off is not cancellable, rather than a lost CAS.
        assert result == NOTHING
        _unwritten(db, tid, before[0])
        db.commit()

    assert _state(factory, tid) == before


# --------------------------------------------------------------------------
# C2 / C10. What a cancel writes; the read agrees with the notice
# --------------------------------------------------------------------------


def test_a_cancel_counts_as_a_cancel_not_a_give_up(factory, monkeypatch):
    counted: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        task_auto_resume,
        "increment_counter",
        lambda name, attributes=None: counted.append((name, attributes or {})),
    )
    tid = _seeded(factory)

    assert _cancel(factory, tid).cancelled

    assert counted == [
        (
            "xagent.task.auto_resume.cancelled",
            {"outcome": "manual", "operation": AT},
        )
    ]


@pytest.mark.parametrize(
    ("seed", "fence"),
    [
        pytest.param({}, PV, id="scheduled"),
        pytest.param({**_DISPATCHED, "command_status": "pending"}, PV, id="dispatched"),
        pytest.param(
            {**_DISPATCHED, "command_status": "pending", "version": PV + 2},
            PV + 2,
            id="retry_after_a_rollback",
        ),
    ],
)
def test_a_cancel_stops_the_row_and_leaves_the_trigger_run(factory, seed, fence):
    with factory() as db:
        tid = _seed(db, **seed)
        row = db.get(TaskAutoRecovery, tid)
        row.total_resumes = 2
        run_id, _ = _trigger(db, db.get(User, db.get(Task, tid).user_id), tid)
        ended = TriggerRun(
            trigger_id=db.get(TriggerRun, run_id).trigger_id,
            task_id=tid,
            status=TriggerRunStatus.FAILED.value,
            idempotency_key=f"ended-{tid}",
            error_message="earlier",
        )
        db.add(ended)
        db.commit()
        ended_id, from_state = int(ended.id), row.state

    result = _cancel(factory, tid)

    assert result.cancelled
    with factory() as db:
        task = db.get(Task, tid)
        row = db.get(TaskAutoRecovery, tid)
        assert (task.status, task.control_state) == (TaskStatus.PAUSED, "paused")
        assert int(task.state_version) == fence + 1
        assert (row.state, row.state_detail, row.next_attempt_at) == (
            "manual",
            "user_cancelled",
            None,
        )
        assert int(row.paused_state_version) == fence + 1
        (event,) = db.scalars(
            sa.select(TaskRecoveryEvent).where(TaskRecoveryEvent.task_id == tid)
        ).all()
        assert (event.event, event.attempt, event.run_id) == (
            "skipped_user_intent",
            2,
            RUN,
        )
        assert event.detail == {
            "at": AT,
            "state": "manual",
            "state_detail": "user_cancelled",
            "from": from_state,
            "task_version": fence,
            "paused_state_version": PV,
        }
        # As after a person's pause (#2177): the run resumed by hand ends it.
        run = db.get(TriggerRun, run_id)
        assert (run.status, run.error_message, run.finished_at) == (
            TriggerRunStatus.RUNNING.value,
            None,
            None,
        )
        ended = db.get(TriggerRun, ended_id)
        assert (ended.status, ended.error_message) == (
            TriggerRunStatus.FAILED.value,
            "earlier",
        )

        view = current_auto_recovery_view(db, task)
    notice = result.notice
    assert notice is not None
    assert (notice.task_id, notice.run_id, notice.state_version) == (
        tid,
        RUN,
        fence + 1,
    )
    assert notice.auto_resume == {
        "status": "stopped",
        "stop_reason": "cancelled",
        "reason": "lease_expired",
        "attempt": 2,
        "next_attempt_at": None,
    }
    assert view is not None
    assert (view["run_id"], view["state_version"], view["auto_resume"]) == (
        notice.run_id,
        notice.state_version,
        notice.auto_resume,
    )


def test_a_resume_by_hand_after_a_cancel_settles_the_trigger_run(
    canonical, monkeypatch
):
    monkeypatch.setattr(
        "xagent.web.services.task_execution.get_session_local",
        lambda: get_session_local(),
    )
    factory, tid = canonical
    with factory() as db:
        task = db.get(Task, tid)
        task.status, task.control_state = TaskStatus.PAUSED, "paused"
        task.run_id, task.state_version = RUN, PV
        _recovery_row(db, tid, RUN, PV, state="scheduled", interrupted_ago=60)
        db.commit()
    run_id = _trigger_run(factory, tid, TriggerRunStatus.RUNNING)

    assert _cancel(factory, tid).cancelled
    assert _trigger_run_state(factory, run_id) == (
        TriggerRunStatus.RUNNING.value,
        None,
        False,
    )

    lease = _resume(factory, tid)
    _finalize(
        factory,
        tid,
        lease,
        status="completed",
        success=True,
        output="the answer",
        result={"status": "completed", "success": True, "output": "the answer"},
    )

    assert _trigger_run_state(factory, run_id) == (
        TriggerRunStatus.COMPLETED.value,
        None,
        True,
    )


# --------------------------------------------------------------------------
# C3-C5. A queued or claimed RESUME is skipped, never resumes the run
# --------------------------------------------------------------------------


def _ch(tid: int) -> SimpleNamespace:
    return SimpleNamespace(task_id=tid, run_id=RUN, version=PV)


@pytest.mark.asyncio
async def test_a_queued_resume_is_skipped_after_a_cancel(factory, handler):
    tid = _seeded(factory, **_DISPATCHED, command_status="pending")

    assert _cancel(factory, tid).cancelled
    # Left queued, not failed: the claim decides its outcome.
    assert _command_status(factory, tid) == "pending"

    skipped = await _auto_resume(_ch(tid), CID, attempt_count=1)

    assert skipped.outcome is ResumeCommandOutcome.AUTO_SKIPPED
    handler.background.assert_not_called()
    assert _state(factory, tid) == (
        PV + 1,
        ("manual", "user_cancelled"),
        PV + 1,
        ["skipped_user_intent"],
    )


@pytest.mark.asyncio
async def test_a_cancel_after_the_guard_defers_the_resume_then_skips_it(
    factory, handler
):
    tid = _seeded(factory, **_DISPATCHED, command_status="processing")
    cancelled: list[AutoResumeCancel] = []

    def cancel_between_guard_and_transition(_task_id):
        cancelled.append(_cancel(factory, tid))
        return ActiveInteractionAbsent()

    handler.interaction.side_effect = cancel_between_guard_and_transition
    first = await _auto_resume(_ch(tid), CID, attempt_count=1)

    assert cancelled[0].cancelled
    assert first.outcome is ResumeCommandOutcome.DEFERRED
    handler.interaction.side_effect = None
    retry = await _auto_resume(_ch(tid), CID, attempt_count=2)

    assert retry.outcome is ResumeCommandOutcome.AUTO_SKIPPED
    handler.background.assert_not_called()
    with factory() as db:
        task = db.get(Task, tid)
        # Never handed off: RESUME_REQUESTED would have moved it to PV + 2.
        assert (task.status, task.control_state, int(task.state_version)) == (
            TaskStatus.PAUSED,
            "paused",
            PV + 1,
        )


@pytest.mark.asyncio
async def test_a_retry_after_a_rollback_is_skipped_not_rescheduled(factory, handler):
    tid = _seeded(factory, **_DISPATCHED, command_status="pending", version=PV + 2)

    assert _cancel(factory, tid).state_version == PV + 3

    retry = await _auto_resume(_ch(tid), CID, attempt_count=2)

    assert retry.outcome is ResumeCommandOutcome.AUTO_SKIPPED
    handler.background.assert_not_called()
    assert _state(factory, tid) == (
        PV + 3,
        ("manual", "user_cancelled"),
        PV + 3,
        ["skipped_user_intent"],
    )


# --------------------------------------------------------------------------
# C6 / C9. Nothing schedules the row again at its fence
# --------------------------------------------------------------------------


@pytest.mark.parametrize("switch", ["on", "off"])
def test_the_sweeper_leaves_a_cancelled_row_alone(factory, monkeypatch, switch):
    tid = _seeded(factory, **_DISPATCHED, command_status="pending")
    assert _cancel(factory, tid).cancelled
    after = _state(factory, tid)
    if switch == "off":
        monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_ENABLED", "false")

    report = run_auto_resume_tick(now=utc_now() + timedelta(days=30))

    assert (report.dispatched, report.gave_up, report.rescheduled, report.stale) == (
        0,
        0,
        0,
        0,
    )
    assert _state(factory, tid) == after
    with factory() as db:
        assert db.scalar(sa.select(sa.func.count(TaskExecutionCommand.id))) == 1


def test_a_later_interruption_is_recorded_at_a_later_fence(factory):
    tid = _seeded(factory)
    assert _cancel(factory, tid).cancelled
    with factory() as db:
        task = db.get(Task, tid)
        # A person resumed by hand; the run was interrupted again.
        task.state_version = PV + 3
        db.flush()
        row = record_interruption_no_commit(
            db,
            task=task,
            reason=InterruptionReason.LEASE_EXPIRED,
            task_status=TaskStatus.PAUSED,
            interrupted_at=utc_now(),
            progress_marker=None,
            gated_by_settlement_switches=False,
        )
        assert row is not None
        recorded = RecordedInterruption.from_row(row)
        db.commit()
    assert (recorded.state, recorded.paused_state_version) == ("scheduled", PV + 3)


# --------------------------------------------------------------------------
# C7. SQLite: a lost compare-and-swap is raced and writes nothing
# --------------------------------------------------------------------------


def _sqlite_only() -> None:
    if get_engine().dialect.name != "sqlite":
        pytest.skip("another writer cannot interleave under the row locks")


def _interleave(monkeypatch, write) -> None:
    """Another transaction commits ``write`` right after the hook's read."""
    original = task_auto_resume._auto_resume_pending

    def pending_then_written(db, task, row):
        result = original(db, task, row)
        with get_session_local()() as other:
            write(other)
            other.commit()
        return result

    monkeypatch.setattr(task_auto_resume, "_auto_resume_pending", pending_then_written)


def test_a_dispatch_that_commits_first_is_raced_then_cancelled(factory, monkeypatch):
    _sqlite_only()
    tid = _seeded(factory)

    def dispatch(other):
        row = other.get(TaskAutoRecovery, tid)
        row.state, row.last_command_id = "dispatched", CID

    with monkeypatch.context() as patch:
        _interleave(patch, dispatch)
        with factory() as db:
            raced = cancel_pending_auto_resume_no_commit(db, _locked(db, tid), at=AT)
            assert (raced.cancelled, raced.raced) == (False, True)
            # Raced means nothing was written: the caller may retry as is.
            _unwritten(db, tid, PV)
            db.commit()
    assert _state(factory, tid) == (PV, ("dispatched", None), PV, [])

    result = _cancel(factory, tid)

    assert result.cancelled
    with factory() as db:
        event = db.scalars(sa.select(TaskRecoveryEvent)).one()
        assert event.detail["from"] == "dispatched"


def test_a_version_moved_first_is_raced_without_a_write(factory, monkeypatch):
    _sqlite_only()
    tid = _seeded(factory)

    def move(other):
        other.get(Task, tid).state_version = PV + 1

    _interleave(monkeypatch, move)
    with factory() as db:
        raced = cancel_pending_auto_resume_no_commit(db, _locked(db, tid), at=AT)
        assert raced.raced
        _unwritten(db, tid, PV + 1)
        db.commit()

    assert _state(factory, tid) == (PV + 1, ("scheduled", None), PV, [])


# --------------------------------------------------------------------------
# C8. A superseding turn and a cancel exclude each other
# --------------------------------------------------------------------------


def _supersede(factory, tid: int) -> bool:
    with factory() as db:
        done = supersede_pending_auto_resume_no_commit(
            db, _locked(db, tid), now=utc_now()
        )
        db.commit()
    return done


def test_a_cancelled_resume_is_not_superseded(factory):
    tid = _seeded(factory)
    assert _cancel(factory, tid).cancelled
    after = _state(factory, tid)

    assert not _supersede(factory, tid)
    assert _state(factory, tid) == after


def test_a_superseded_resume_is_not_cancelled(factory):
    tid = _seeded(factory)
    assert _supersede(factory, tid)
    after = _state(factory, tid)

    assert _cancel(factory, tid) == NOTHING
    assert _state(factory, tid) == after


# --------------------------------------------------------------------------
# P1-P3. PostgreSQL: lock order with the sweeper and the claim guard
# --------------------------------------------------------------------------


def _cancel_in_thread(tid: int, outcome: dict[str, Any]) -> threading.Thread:
    def run() -> None:
        try:
            with get_session_local()() as db:
                outcome["result"] = cancel_pending_auto_resume_no_commit(
                    db, _locked(db, tid), at=AT
                )
                db.commit()
        except BaseException as exc:  # pragma: no cover - reported below
            outcome["error"] = exc

    return threading.Thread(target=run)


def test_postgresql_cancel_waits_for_a_dispatch_then_cancels_it(
    factory, short_lock_timeout
):
    _postgresql_only()
    tid = _seeded(factory)
    outcome: dict[str, Any] = {}
    thread = _cancel_in_thread(tid, outcome)
    sweeper = get_session_local()()
    try:
        now = utc_now()
        assert select_next_candidate_for_update(
            sweeper, TaskAutoRecoveryState.SCHEDULED, now=now
        ) == (sweeper.get(TaskAutoRecovery, tid).next_attempt_at, tid)
        dispatched = task_auto_resume._process_due_candidate_no_commit(
            sweeper, tid, now=now, can_dispatch=True
        )
        assert dispatched.staged
        thread.start()
        with get_session_local()() as monitor:
            _wait_for_a_lock_waiter(monitor)
        sweeper.commit()
    finally:
        sweeper.close()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert "error" not in outcome, outcome.get("error")
    assert outcome["result"].cancelled
    with factory() as db:
        event = db.scalars(
            sa.select(TaskRecoveryEvent).where(
                TaskRecoveryEvent.event == "skipped_user_intent"
            )
        ).one()
        assert event.detail["from"] == "dispatched"


def test_postgresql_sweeper_skips_a_task_a_cancel_holds(factory, short_lock_timeout):
    _postgresql_only()
    tid = _seeded(factory)
    scheduled = TaskAutoRecoveryState.SCHEDULED
    cancelling = get_session_local()()
    try:
        assert cancel_pending_auto_resume_no_commit(
            cancelling, _locked(cancelling, tid), at=AT
        ).cancelled
        with get_session_local()() as sweeper:
            assert (
                select_next_candidate_for_update(sweeper, scheduled, now=utc_now())
                is None
            )
            sweeper.rollback()
        cancelling.commit()
    finally:
        cancelling.close()

    report = run_auto_resume_tick(now=utc_now() + timedelta(days=30))

    assert report.dispatched == 0
    assert _state(factory, tid)[1] == ("manual", "user_cancelled")


@pytest.mark.parametrize("snapshot", ["at_fence", "moved"])
def test_postgresql_claim_guard_waits_for_a_cancel_without_deadlock(
    factory, short_lock_timeout, snapshot
):
    """At the fence the guard locks the recovery row; past it, the task row
    first. Either way it waits for the cancel (which holds both, tasks
    first) and then finds the row no longer naming its command."""
    _postgresql_only()
    tid = _seeded(factory, **_DISPATCHED, command_status="processing")
    outcome: dict[str, Any] = {}

    def guard() -> None:
        try:
            outcome["decision"] = check_auto_resume_claim_sync(
                task_id=tid,
                command_id=CID,
                auto_resume={"expected_run_id": RUN, "expected_state_version": PV},
                status=TaskStatus.PAUSED,
                control_state=TaskControlState.PAUSED.value,
                run_id=RUN,
                state_version=PV if snapshot == "at_fence" else PV + 5,
                attempt_count=1,
            )
        except BaseException as exc:  # pragma: no cover - reported below
            outcome["error"] = exc

    cancelling = get_session_local()()
    thread = threading.Thread(target=guard)
    try:
        assert cancel_pending_auto_resume_no_commit(
            cancelling, _locked(cancelling, tid), at=AT
        ).cancelled
        thread.start()
        with get_session_local()() as monitor:
            _wait_for_a_lock_waiter(monitor)
        cancelling.commit()
    finally:
        cancelling.close()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert "error" not in outcome, outcome.get("error")
    assert outcome["decision"].proceed is False
    assert _state(factory, tid) == (
        PV + 1,
        ("manual", "user_cancelled"),
        PV + 1,
        ["skipped_user_intent"],
    )
