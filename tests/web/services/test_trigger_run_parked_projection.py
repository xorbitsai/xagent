"""Regression tests for #2177: a parked trigger task must not strand its run.

A ``TriggerRun`` is a projection of its ``Task``. When the task parks at
``PAUSED`` / ``WAITING_FOR_USER`` its turn still finalizes through
``task_orchestrator.finish_turn`` -- whose park tail releases that turn's lease --
so the projection has to carry the parked state too:

  - ``finish_turn``'s park tail projects the parked state before releasing the
    lease (``sync_trigger_run_status``)
  - ``sync_trigger_run_status`` maps parked task states onto
    ``TriggerRunStatus.PAUSED`` and clears ``finished_at``
  - the row selector takes every non-terminal run, so the same run is still
    picked up when the task resumes and terminates

Before the fix none of that existed: the park tail only called
``commit_terminal(status)``, the projector could only write ``completed`` /
``failed``, and the lease release clears ``lease_expires_at`` -- so the parked
task was never an expired-lease recovery candidate either, and the run stayed
``running`` with ``finished_at IS NULL`` until the task was resumed and
terminated (or forever).

These tests drive the real production functions (``finish_turn``,
``sync_trigger_run_status``, the recovery candidate query) rather than
hand-written SQL, so a failure here reports the product gap instead of a fixture
artefact. Three kinds of test live here:

  - ``REPRO``: the reproduction. Every one of them failed before the fix.
  - ``GUARD``: the monotonicity rules the projection must keep, so that
    narrowing the row selector or resurrecting a finished run fails loudly.
  - ``TRAP``: the structural facts that explain why the run cannot be repaired
    later. These must keep holding, or the fix has changed the recovery
    contract without saying so.
"""

from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import Enum as SAEnum
from sqlalchemy import String, create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.models.agent import Agent
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.trigger import (
    AgentTrigger,
    TriggerRun,
    TriggerRunStatus,
    TriggerType,
)
from xagent.web.models.user import User
from xagent.web.services import triggers as triggers_module
from xagent.web.services.task_execution_controller import control_state_for_status
from xagent.web.services.task_lease_service import (
    TaskLease,
    get_expired_task_lease_candidates,
    utc_now,
)
from xagent.web.services.task_orchestrator import (
    _trigger_run_status_for_task_status,
    finish_turn,
    sync_trigger_run_status,
)

RUNNER_ID = "parked-runner"
RUN_ID = "parked-run"
ATTEMPT_ID = "parked-attempt"

# The value a parked run is projected onto. Parked task states collapse onto it:
# the distinction between "paused" and "waiting on the user" belongs to the task.
PARKED_RUN_STATUS = TriggerRunStatus.PAUSED.value

PARKED_TASK_STATUSES = (TaskStatus.WAITING_FOR_USER, TaskStatus.PAUSED)
_parked_ids = pytest.mark.parametrize(
    "parked_status", PARKED_TASK_STATUSES, ids=lambda status: status.value
)


@pytest.fixture()
def db_session():
    """A private in-memory database: these tests need no filesystem fixtures."""

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


def _create_user(db, *, suffix: str = "") -> User:
    user = User(
        username=f"parked-trigger-user{suffix}",
        password_hash="hash",
        is_admin=False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _seed_task_with_run(
    db,
    *,
    task_status: TaskStatus,
    run_status: str = TriggerRunStatus.RUNNING.value,
    task_error: str | None = None,
    run_error: str | None = None,
    with_lease: bool = False,
    suffix: str = "",
) -> tuple[Task, TriggerRun, TaskLease | None]:
    """Seed a task with its trigger run at an explicit pair of states.

    With ``with_lease`` this mirrors production ordering: the park branch in
    ``task_execution`` commits the task's parked status while the turn's lease is
    still held, and ``finish_turn`` then runs from ``_schedule_bg._runner``'s
    finally with that same lease.
    """

    user = _create_user(db, suffix=suffix)
    agent = Agent(user_id=int(user.id), name=f"parked trigger agent{suffix}")
    db.add(agent)
    db.flush()
    trigger = AgentTrigger(
        user_id=int(user.id),
        agent_id=int(agent.id),
        type=TriggerType.WEBHOOK.value,
        name=f"parked trigger{suffix}",
        config={},
    )
    db.add(trigger)
    db.flush()
    task = Task(
        user_id=int(user.id),
        title="Parked trigger task",
        description="parked trigger projection reproduction",
        status=task_status,
        control_state=control_state_for_status(task_status).value,
        execution_mode="auto",
        source="trigger",
        error_message=task_error,
        runner_id=RUNNER_ID if with_lease else None,
        run_id=RUN_ID if with_lease else None,
        lease_attempt_id=ATTEMPT_ID if with_lease else None,
        lease_expires_at=utc_now() + timedelta(minutes=5) if with_lease else None,
        state_version=1,
    )
    db.add(task)
    db.flush()
    run = TriggerRun(
        trigger_id=int(trigger.id),
        task_id=int(task.id),
        status=run_status,
        error_message=run_error,
        idempotency_key=f"parked-run-{task_status.value}-{run_status}{suffix}",
    )
    db.add(run)
    db.commit()
    db.refresh(task)
    db.refresh(run)

    lease = (
        TaskLease(
            task_id=int(task.id),
            runner_id=RUNNER_ID,
            run_id=RUN_ID,
            attempt_id=ATTEMPT_ID,
        )
        if with_lease
        else None
    )
    return task, run, lease


def _seed_parked_run(
    db, *, parked_status: TaskStatus
) -> tuple[Task, TriggerRun, TaskLease]:
    """Seed the exact row state a parked turn leaves behind."""

    task, run, lease = _seed_task_with_run(
        db,
        task_status=parked_status,
        with_lease=True,
        suffix=f"-{parked_status.value}",
    )
    assert lease is not None
    return task, run, lease


# ---------------------------------------------------------------------------
# REPRO: the parked turn must stop presenting the run as actively running
# ---------------------------------------------------------------------------


@_parked_ids
def test_parked_turn_keeps_trigger_run_running_reproduction(db_session, parked_status):
    """REPRO (#2177): a parked turn leaves its trigger run reading "running"."""

    task, run, lease = _seed_parked_run(db_session, parked_status=parked_status)
    task_id = int(task.id)
    run_id = int(run.id)

    assert finish_turn(db_session, task_id, task_lease=lease) is True

    db_session.expire_all()
    task_after = db_session.get(Task, task_id)
    run_after = db_session.get(TriggerRun, run_id)

    # Sanity: the park tail really ran -- the lease was released and the parked
    # control status was preserved. Without this the assertions below could pass
    # for the wrong reason (e.g. a fence mismatch returning early).
    assert task_after.status == parked_status
    assert task_after.runner_id is None
    assert task_after.lease_expires_at is None

    assert run_after.status != TriggerRunStatus.RUNNING.value, (
        f"a TriggerRun whose task parked at {parked_status.value} still reads as "
        "'running' after the turn finalized; the lease is already released and "
        "the task is not a recovery candidate, so nothing repairs it (#2177)"
    )
    assert run_after.finished_at is None, (
        "a parked run must not carry finished_at: the task is not finished"
    )
    assert run_after.error_message is None, "waiting for the user is not an error"


@_parked_ids
def test_parked_turn_projects_the_paused_run_status(db_session, parked_status):
    """REPRO (#2177): the parked projection is the value the fix introduces."""

    task, run, lease = _seed_parked_run(db_session, parked_status=parked_status)
    run_id = int(run.id)

    assert finish_turn(db_session, int(task.id), task_lease=lease) is True

    db_session.expire_all()
    run_after = db_session.get(TriggerRun, run_id)
    assert run_after.status == PARKED_RUN_STATUS


# ---------------------------------------------------------------------------
# REPRO: the projector must not fabricate a terminal failure for a parked task
# ---------------------------------------------------------------------------


@_parked_ids
def test_projector_must_not_report_terminal_failure_for_a_parked_task(
    db_session, parked_status
):
    """REPRO (#2177): projecting a parked task must not create a false failure.

    ``sync_trigger_run_status`` collapses every non-COMPLETED status into
    ``failed`` + ``finished_at``. Lease recovery depends on that for an
    unrecoverable crash, but applied to a task that is merely waiting it makes a
    resumable run terminal -- and because the row selector skips terminal rows,
    a later successful resume can never correct it.
    """

    task, run, _lease = _seed_parked_run(db_session, parked_status=parked_status)
    run_id = int(run.id)

    # The projector only stages its writes; every production caller commits in
    # the same transaction as the status transition it mirrors.
    assert sync_trigger_run_status(db_session, task, parked_status) is True
    db_session.commit()

    db_session.expire_all()
    run_after = db_session.get(TriggerRun, run_id)
    assert run_after.status != TriggerRunStatus.FAILED.value, (
        f"a task parked at {parked_status.value} was projected onto its run as a "
        "terminal failure"
    )
    assert run_after.finished_at is None


# ---------------------------------------------------------------------------
# TRAP: why nothing repairs the run while the task stays parked
# ---------------------------------------------------------------------------


def test_parked_task_is_not_a_recovery_candidate_but_a_crashed_one_is(db_session):
    """TRAP (#2177): lease recovery structurally cannot see a parked task.

    The candidate query requires ``status == RUNNING`` *and* a non-NULL
    ``lease_expires_at`` older than the cutoff. The park release NULLs
    ``lease_expires_at``, so a parked task is never re-selected -- which is why
    the run must be projected at park time rather than relying on a later sweep.
    A crashed RUNNING task with an expired lease stays the control case.
    """

    parked, _run, _lease = _seed_parked_run(
        db_session, parked_status=TaskStatus.WAITING_FOR_USER
    )
    parked.runner_id = None
    parked.lease_attempt_id = None
    parked.lease_expires_at = None
    db_session.add(parked)

    crashed = Task(
        user_id=int(parked.user_id),
        title="Crashed task",
        description="expired lease control case",
        status=TaskStatus.RUNNING,
        control_state=control_state_for_status(TaskStatus.RUNNING).value,
        execution_mode="auto",
        source="trigger",
        runner_id="dead-runner",
        run_id="dead-run",
        lease_attempt_id="dead-attempt",
        lease_expires_at=utc_now() - timedelta(minutes=5),
        state_version=1,
    )
    db_session.add(crashed)
    db_session.commit()

    candidates = get_expired_task_lease_candidates(
        db_session, cutoff=utc_now(), limit=100
    )
    candidate_ids = {candidate.task_id for candidate in candidates}

    assert int(crashed.id) in candidate_ids, (
        "control case failed: an expired RUNNING lease is no longer selected, so "
        "this TRAP test no longer describes the recovery contract"
    )
    assert int(parked.id) not in candidate_ids, (
        "a parked task became a recovery candidate; if that is intentional the "
        "fix must explain what the sweep now does with it (#2177)"
    )


# ---------------------------------------------------------------------------
# GUARD: the monotonicity rules the projection must keep
# ---------------------------------------------------------------------------


@_parked_ids
def test_parked_run_is_still_finalized_after_the_task_resumes(
    db_session, parked_status
):
    """GUARD: the row selector must keep covering the parked value.

    This is the trap that makes the parked value dangerous: if the selector goes
    back to listing only ``pending`` / ``running``, a run parked here is never
    selected again and stays ``paused`` forever -- the same lie as staying
    ``running``, just a different value.
    """

    task, run, lease = _seed_parked_run(db_session, parked_status=parked_status)
    task_id = int(task.id)
    run_id = int(run.id)

    assert finish_turn(db_session, task_id, task_lease=lease) is True
    db_session.expire_all()
    assert db_session.get(TriggerRun, run_id).status == PARKED_RUN_STATUS

    # The resume: the task is claimed onto a new run and then terminates.
    task_after = db_session.get(Task, task_id)
    assert sync_trigger_run_status(db_session, task_after, TaskStatus.RUNNING) is True
    db_session.commit()
    db_session.expire_all()
    assert db_session.get(TriggerRun, run_id).status == TriggerRunStatus.RUNNING.value

    task_after = db_session.get(Task, task_id)
    assert sync_trigger_run_status(db_session, task_after, TaskStatus.COMPLETED) is True
    db_session.commit()

    db_session.expire_all()
    final_run = db_session.get(TriggerRun, run_id)
    assert final_run.status == TriggerRunStatus.COMPLETED.value
    assert final_run.finished_at is not None
    assert final_run.error_message is None


def test_terminal_run_is_never_reopened_by_a_late_projection(db_session):
    """GUARD: a finished run must not be resurrected into a parked state.

    Terminal rows are excluded by the selector rather than by a caller check, so
    this holds for a stale projection arriving after the task finished -- for
    example a late settle from a session that was fenced out of the row.
    """

    task, run, _lease = _seed_parked_run(
        db_session, parked_status=TaskStatus.WAITING_FOR_USER
    )
    run_id = int(run.id)

    assert sync_trigger_run_status(db_session, task, TaskStatus.COMPLETED) is True
    db_session.commit()
    db_session.expire_all()
    finished_run = db_session.get(TriggerRun, run_id)
    assert finished_run.status == TriggerRunStatus.COMPLETED.value
    finished_at = finished_run.finished_at
    assert finished_at is not None

    # A late parked projection must be a no-op, not a regression.
    assert (
        sync_trigger_run_status(db_session, task, TaskStatus.WAITING_FOR_USER) is False
    )
    db_session.commit()

    db_session.expire_all()
    reopened = db_session.get(TriggerRun, run_id)
    assert reopened.status == TriggerRunStatus.COMPLETED.value
    assert reopened.finished_at == finished_at


def test_replayed_delivery_does_not_reset_an_advanced_run(db_session):
    """GUARD: a duplicate delivery must not re-arm a run that already advanced.

    ``prepare_trigger_run`` is the entry point a webhook replay hits. Once a run
    has its task attached it must come back untouched -- in particular a parked
    run must not be pushed back to ``pending``, which would re-enter the dispatch
    path while its task is still parked.
    """

    task, run, _lease = _seed_parked_run(
        db_session, parked_status=TaskStatus.WAITING_FOR_USER
    )
    trigger = db_session.get(AgentTrigger, int(run.trigger_id))
    assert trigger is not None

    source_event_id = "redelivered-event"
    run.idempotency_key = triggers_module._trigger_run_idempotency_key(
        trigger, event_payload={}, source_event_id=source_event_id, test=False
    )
    run.status = TriggerRunStatus.PAUSED.value
    db_session.add(run)
    db_session.commit()
    run_id = int(run.id)

    replayed, created = triggers_module.prepare_trigger_run(
        db_session,
        trigger=trigger,
        event_payload={},
        source_event_id=source_event_id,
    )

    assert created is False
    assert int(replayed.id) == run_id
    db_session.expire_all()
    stored = db_session.get(TriggerRun, run_id)
    assert stored.status == TriggerRunStatus.PAUSED.value
    assert int(stored.task_id) == int(task.id)


# ---------------------------------------------------------------------------
# COMPAT: the change stays additive, and the mapping stays total
# ---------------------------------------------------------------------------

LEGACY_RUN_STATUS_VALUES = frozenset({"pending", "running", "completed", "failed"})


def test_legacy_run_status_values_are_unchanged_and_only_paused_was_added():
    """Existing rows carry these strings: no value may move or disappear."""

    values = {member.value for member in TriggerRunStatus}
    assert LEGACY_RUN_STATUS_VALUES <= values
    assert values - LEGACY_RUN_STATUS_VALUES == {TriggerRunStatus.PAUSED.value}, (
        "the parked value must be the only addition, and the only one needed"
    )


def test_run_status_column_is_plain_text_so_the_new_value_needs_no_migration():
    """A native enum column would need ``ALTER TYPE ... ADD VALUE`` instead."""

    column = TriggerRun.__table__.c.status
    assert isinstance(column.type, String)
    assert not isinstance(column.type, SAEnum)
    assert column.type.length is not None
    for member in TriggerRunStatus:
        assert len(member.value) <= column.type.length


def test_the_parked_value_is_not_terminal():
    assert TriggerRunStatus.terminal_values() == frozenset(
        {TriggerRunStatus.COMPLETED.value, TriggerRunStatus.FAILED.value}
    )
    assert TriggerRunStatus.PAUSED.value not in TriggerRunStatus.terminal_values()


@pytest.mark.parametrize(
    ("task_status", "explicit_error", "expected_error"),
    [
        (TaskStatus.COMPLETED, None, None),
        (TaskStatus.FAILED, None, "task level failure"),
        (TaskStatus.FAILED, "explicit failure", "explicit failure"),
    ],
)
def test_terminal_projection_still_writes_what_it_always_wrote(
    db_session, task_status, explicit_error, expected_error
):
    """COMPLETED clears the error, FAILED carries one, an override still wins.

    Lease recovery passes an explicit message and depends on that precedence.
    """

    task, run, _lease = _seed_task_with_run(
        db_session,
        task_status=task_status,
        task_error="task level failure",
        run_error="stale error",
        suffix=f"-{task_status.value}-{explicit_error}",
    )
    run_id = int(run.id)

    assert (
        sync_trigger_run_status(
            db_session, task, task_status, error_message=explicit_error
        )
        is True
    )
    db_session.commit()
    db_session.expire_all()

    stored = db_session.get(TriggerRun, run_id)
    assert stored.status == task_status.value
    assert stored.finished_at is not None
    assert stored.error_message == expected_error


@pytest.mark.parametrize(
    "run_status",
    [TriggerRunStatus.PENDING.value, TriggerRunStatus.RUNNING.value],
)
def test_rows_the_previous_selector_covered_are_still_finalized(db_session, run_status):
    """pending / running were the old selector's whole world: keep finalizing."""

    task, run, _lease = _seed_task_with_run(
        db_session, task_status=TaskStatus.COMPLETED, run_status=run_status
    )
    run_id = int(run.id)

    assert sync_trigger_run_status(db_session, task, TaskStatus.COMPLETED) is True
    db_session.commit()
    db_session.expire_all()
    assert db_session.get(TriggerRun, run_id).status == TriggerRunStatus.COMPLETED.value


def test_a_stale_pending_projection_cannot_regress_a_live_run(db_session):
    """PENDING belongs to preparation/dispatch, so it projects onto nothing."""

    task, run, _lease = _seed_task_with_run(
        db_session,
        task_status=TaskStatus.PENDING,
        run_status=TriggerRunStatus.RUNNING.value,
    )
    run_id = int(run.id)

    assert sync_trigger_run_status(db_session, task, TaskStatus.PENDING) is False
    db_session.commit()
    db_session.expire_all()

    stored = db_session.get(TriggerRun, run_id)
    assert stored.status == TriggerRunStatus.RUNNING.value
    assert stored.finished_at is None


@pytest.mark.parametrize("task_status", list(TaskStatus), ids=lambda item: item.value)
def test_every_task_status_has_an_explicit_projection_answer(task_status):
    """The guard against a repeat of #2177: no task state may lack an answer."""

    target = _trigger_run_status_for_task_status(task_status)
    if task_status is TaskStatus.PENDING:
        assert target is None
        return
    assert target is not None, (
        f"TaskStatus.{task_status.name} has no trigger run projection; a task "
        "settling in that state would leave its run at its previous value"
    )
    assert target in {member.value for member in TriggerRunStatus}


def test_frontend_status_vocabulary_matches_the_backend_enum():
    """The UI list is a contract: every value the backend writes needs a label."""

    source_path = (
        Path(__file__).resolve().parents[3]
        / "frontend"
        / "src"
        / "lib"
        / "agent-triggers-api.ts"
    )
    # A checkout without the frontend tree cannot check a cross-layer contract, so
    # skip instead of failing -- the same guard
    # tests/core/tools/adapters/vibe/test_interaction_type_aliases.py uses. The
    # assertions below stay strict: a present-but-drifted list is exactly what
    # this test exists to catch.
    if not source_path.exists():
        pytest.skip(f"frontend source not present: {source_path}")

    source = source_path.read_text(encoding="utf-8")
    match = re.search(
        r"AGENT_TRIGGER_RUN_STATUSES\s*=\s*\[(.*?)\]\s*as const",
        source,
        re.DOTALL,
    )
    assert match is not None, (
        "AGENT_TRIGGER_RUN_STATUSES is gone from frontend/src/lib/agent-triggers-api.ts"
    )

    frontend_values = re.findall(r'"([a-z_]+)"', match.group(1))
    assert set(frontend_values) == {member.value for member in TriggerRunStatus}, (
        "the frontend run status list and the backend TriggerRunStatus enum disagree"
    )
