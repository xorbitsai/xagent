"""Tests for automatic recovery of expired task execution leases."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from threading import Barrier

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import Query, Session, sessionmaker

from tests.shared.db_teardown import drop_all_tables
from tests.web.services.checkpoint_anchor_shared import build_upgraded_sqlite_engine
from xagent.core.agent.checkpoint import CHECKPOINT_TYPE
from xagent.web.models.agent import Agent
from xagent.web.models.database import (
    get_db,
    get_engine,
    get_session_local,
    init_db,
)
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.trigger import (
    AgentTrigger,
    TriggerRun,
    TriggerRunStatus,
    TriggerType,
)
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce, WorkforceRun
from xagent.web.services import task_lease_recovery, task_lease_service
from xagent.web.services.task_lease_recovery import (
    TASK_LEASE_EXPIRED_ERROR,
    TASK_LEASE_PAUSED_TRIGGER_ERROR,
    recover_expired_task_leases_until_cutoff,
    recover_task_lease_candidate_isolated,
    recover_task_lease_candidate_no_commit,
    run_task_lease_recovery_loop,
)
from xagent.web.services.task_lease_service import (
    TASK_RUN_ID_TRACE_FIELD,
    CheckpointRecoveryVerdict,
    TaskLeaseRecoveryCandidate,
    get_expired_task_lease_candidates,
    resolve_checkpoint_recovery,
    utc_now,
)


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'task-lease-recovery.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        drop_all_tables(get_engine())


def _create_user(db, *, suffix: str) -> User:
    user = User(
        username=f"lease-recovery-{suffix}",
        password_hash="hash",
        is_admin=False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _create_expired_task(
    db,
    *,
    user_id: int,
    suffix: str,
    with_checkpoint: bool = False,
) -> Task:
    task = Task(
        user_id=user_id,
        title=f"Expired lease {suffix}",
        description="lease recovery test",
        status=TaskStatus.RUNNING,
        execution_mode="auto",
        runner_id=f"dead-runner-{suffix}",
        run_id=f"run-{suffix}",
        lease_expires_at=utc_now() - timedelta(seconds=5),
        last_heartbeat_at=utc_now() - timedelta(seconds=10),
        state_version=3,
        control_state="running",
        output="stale output",
    )
    db.add(task)
    db.flush()
    if with_checkpoint:
        event_id = f"checkpoint-{suffix}"
        db.add(
            TraceEvent(
                task_id=task.id,
                event_id=event_id,
                event_type="system_update_general",
                timestamp=utc_now(),
                step_id=None,
                parent_event_id=None,
                data={
                    "checkpoint_type": CHECKPOINT_TYPE,
                    "snapshot": {"type": "checkpoint"},
                    TASK_RUN_ID_TRACE_FIELD: task.run_id,
                },
            )
        )
        task.last_checkpoint_event_id = event_id
    db.commit()
    db.refresh(task)
    return task


def _recover_expired_task(db, task: Task) -> TaskStatus | None:
    candidate = get_expired_task_lease_candidates(
        db,
        cutoff=utc_now(),
        limit=1,
    )[0]
    db.rollback()
    return recover_task_lease_candidate_isolated(
        candidate,
        recovered_at=utc_now(),
    )


def _candidate_for_task(task: Task) -> TaskLeaseRecoveryCandidate:
    """Build the recovery candidate snapshot resolve_checkpoint_recovery
    consumes, directly from a task's current column values -- for tests
    that exercise checkpoint pointer resolution without going through the
    full expired-lease scan query.
    """
    return TaskLeaseRecoveryCandidate(
        task_id=int(task.id),
        runner_id=task.runner_id,
        run_id=task.run_id,
        lease_expires_at=task.lease_expires_at,
        state_version=int(task.state_version or 0),
        last_checkpoint_event_id=task.last_checkpoint_event_id,
        last_checkpoint_trace_event_id=task.last_checkpoint_trace_event_id,
    )


@pytest.fixture()
def sqlite_no_anchor_fk_session(tmp_path):
    """A SQLite database shaped like the checkpoint-anchor migration's ADD
    COLUMN path applied to a table that already existed: full schema via
    create_all, then ``tasks`` rebuilt without the checkpoint pointer's FK
    clause. Alembic's SQLite batch mode cannot add that FK without a full
    table rebuild, so this asymmetric shape -- not the fresh create_all
    schema every other fixture in this module uses -- is the only one on
    which ``last_checkpoint_trace_event_id`` can point at a row that no
    longer exists. The rebuild trick lives in checkpoint_anchor_shared.py,
    shared with
    tests/web/services/test_task_deletion_checkpoint_pointer.py's
    sqlite_upgraded_session, which needs the identical shape.
    """
    engine = build_upgraded_sqlite_engine(tmp_path / "lease-recovery-no-fk.db")
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()
    yield session
    session.close()
    engine.dispose()


def _attach_workforce_and_trigger(db, *, task: Task, user: User) -> tuple:
    manager = Agent(user_id=user.id, name="lease recovery manager")
    db.add(manager)
    db.flush()
    workforce = Workforce(
        owner_user_id=user.id,
        scope_type="user",
        scope_id=str(user.id),
        name=f"Recovery workforce {task.id}",
        manager_agent_id=manager.id,
        status="published",
    )
    db.add(workforce)
    db.flush()
    workforce_run = WorkforceRun(
        workforce_id=workforce.id,
        task_id=task.id,
        user_id=user.id,
        status="running",
        snapshot={},
    )
    db.add(workforce_run)
    db.flush()
    task.agent_config = {"workforce_run_id": int(workforce_run.id)}

    trigger = AgentTrigger(
        user_id=user.id,
        workforce_id=workforce.id,
        type=TriggerType.SCHEDULED.value,
        name=f"Recovery trigger {task.id}",
        config={},
    )
    db.add(trigger)
    db.flush()
    trigger_run = TriggerRun(
        trigger_id=trigger.id,
        task_id=task.id,
        status=TriggerRunStatus.RUNNING.value,
        idempotency_key=f"lease-recovery-{task.id}",
    )
    db.add(trigger_run)
    db.commit()
    return workforce_run, trigger_run


def test_expired_lease_with_checkpoint_pauses_all_lifecycle_projections(
    db_session,
) -> None:
    user = _create_user(db_session, suffix="checkpoint")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="checkpoint",
        with_checkpoint=True,
    )
    workforce_run, trigger_run = _attach_workforce_and_trigger(
        db_session,
        task=task,
        user=user,
    )

    assert _recover_expired_task(db_session, task) == TaskStatus.PAUSED

    db_session.refresh(task)
    db_session.refresh(workforce_run)
    db_session.refresh(trigger_run)
    assert task.status == TaskStatus.PAUSED
    assert task.control_state == "paused"
    assert task.state_version == 4
    assert task.runner_id is None
    assert task.lease_expires_at is None
    assert task.run_id == "run-checkpoint"
    assert task.error_message is None
    assert workforce_run.status == "paused"
    assert workforce_run.completed_at is None
    assert trigger_run.status == TriggerRunStatus.FAILED.value
    assert trigger_run.error_message == TASK_LEASE_PAUSED_TRIGGER_ERROR
    assert trigger_run.finished_at is not None


def test_expired_lease_without_checkpoint_fails_and_clears_stale_output(
    db_session,
) -> None:
    user = _create_user(db_session, suffix="failed")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="failed",
    )
    workforce_run, trigger_run = _attach_workforce_and_trigger(
        db_session,
        task=task,
        user=user,
    )

    assert _recover_expired_task(db_session, task) == TaskStatus.FAILED

    db_session.refresh(task)
    db_session.refresh(workforce_run)
    db_session.refresh(trigger_run)
    assert task.status == TaskStatus.FAILED
    assert task.control_state == "failed"
    assert task.state_version == 4
    assert task.runner_id is None
    assert task.lease_expires_at is None
    assert task.output is None
    assert task.error_message == TASK_LEASE_EXPIRED_ERROR
    assert workforce_run.status == "failed"
    assert workforce_run.completed_at is not None
    assert trigger_run.status == TriggerRunStatus.FAILED.value
    assert trigger_run.error_message == TASK_LEASE_EXPIRED_ERROR


@pytest.mark.parametrize("checkpoint_run_id", [None, "previous-run"])
def test_recovery_rejects_checkpoint_without_current_run_provenance(
    db_session,
    checkpoint_run_id: str | None,
) -> None:
    user = _create_user(db_session, suffix=f"provenance-{checkpoint_run_id}")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix=f"provenance-{checkpoint_run_id}",
        with_checkpoint=True,
    )
    checkpoint = (
        db_session.query(TraceEvent)
        .filter(TraceEvent.event_id == task.last_checkpoint_event_id)
        .one()
    )
    data = dict(checkpoint.data)
    if checkpoint_run_id is None:
        data.pop(TASK_RUN_ID_TRACE_FIELD, None)
    else:
        data[TASK_RUN_ID_TRACE_FIELD] = checkpoint_run_id
    checkpoint.data = data
    # Also anchor the PK pointer at the same row: the PK-first resolution
    # path must reach the identical FAILED verdict the legacy scan does,
    # not just the legacy path exercised when this column is unset.
    task.last_checkpoint_trace_event_id = checkpoint.id
    db_session.commit()

    assert _recover_expired_task(db_session, task) == TaskStatus.FAILED

    db_session.expire_all()
    persisted = db_session.get(Task, int(task.id))
    assert persisted.status == TaskStatus.FAILED
    assert persisted.output is None


def test_exact_checkpoint_pointer_is_not_limited_to_latest_one_hundred_events(
    db_session,
) -> None:
    user = _create_user(db_session, suffix="exact-pointer")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="exact-pointer",
        with_checkpoint=True,
    )
    for index in range(110):
        db_session.add(
            TraceEvent(
                task_id=task.id,
                event_id=f"noise-{index}",
                event_type="system_update_general",
                timestamp=utc_now() + timedelta(microseconds=index + 1),
                step_id=None,
                parent_event_id=None,
                data={"message": f"noise-{index}"},
            )
        )
    db_session.commit()

    assert _recover_expired_task(db_session, task) == TaskStatus.PAUSED


def test_pk_anchor_resolves_without_a_usable_legacy_pointer(db_session) -> None:
    """The exact-row pointer is resolved before the legacy scan runs at
    all -- proven here by leaving the legacy event_id column unset, a
    state a legacy-only scan could never resolve to anything.
    """
    user = _create_user(db_session, suffix="pk-priority")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="pk-priority",
    )
    checkpoint = TraceEvent(
        task_id=task.id,
        event_id="pk-priority-checkpoint",
        event_type="system_update_general",
        timestamp=utc_now(),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "snapshot": {"type": "checkpoint"},
            TASK_RUN_ID_TRACE_FIELD: task.run_id,
        },
    )
    db_session.add(checkpoint)
    db_session.flush()
    task.last_checkpoint_trace_event_id = checkpoint.id
    assert task.last_checkpoint_event_id is None
    db_session.commit()

    candidate = _candidate_for_task(task)
    assert (
        resolve_checkpoint_recovery(db_session, candidate)
        is CheckpointRecoveryVerdict.RECOVERABLE
    )
    assert _recover_expired_task(db_session, task) == TaskStatus.PAUSED


def test_pk_anchor_validation_failure_does_not_fall_back_to_the_legacy_scan(
    db_session,
) -> None:
    """A pointer that resolves to a row failing validation is corruption of
    that exact row, not a cue to go search other rows -- even when the
    legacy event_id column names a different, genuinely valid checkpoint
    that a legacy-only scan would have accepted.

    One shape is deliberately excluded and is not this test's subject: a row
    whose *only* failed condition is the run partition, and only because the
    field is absent, is a pre-existing row rather than a failed validation,
    and does defer to the legacy scan (see
    test_pk_anchor_missing_run_partition_defers_to_the_legacy_scan). This
    test's row fails on ``build_id``, which no reclassification covers.
    """
    user = _create_user(db_session, suffix="pk-no-fallback")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="pk-no-fallback",
    )
    invalid_anchor = TraceEvent(
        task_id=task.id,
        build_id="agent_child",  # fails the build_id IS NULL check
        event_id="pk-no-fallback-invalid",
        event_type="system_update_general",
        timestamp=utc_now(),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "snapshot": {"type": "checkpoint"},
            TASK_RUN_ID_TRACE_FIELD: task.run_id,
        },
    )
    valid_legacy_row = TraceEvent(
        task_id=task.id,
        event_id="pk-no-fallback-valid",
        event_type="system_update_general",
        timestamp=utc_now(),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "snapshot": {"type": "checkpoint"},
            TASK_RUN_ID_TRACE_FIELD: task.run_id,
        },
    )
    db_session.add_all([invalid_anchor, valid_legacy_row])
    db_session.flush()
    task.last_checkpoint_trace_event_id = invalid_anchor.id
    task.last_checkpoint_event_id = valid_legacy_row.event_id
    db_session.commit()

    candidate = _candidate_for_task(task)
    assert (
        resolve_checkpoint_recovery(db_session, candidate)
        is CheckpointRecoveryVerdict.NOT_RECOVERABLE
    )
    assert _recover_expired_task(db_session, task) == TaskStatus.FAILED


def test_pk_anchor_missing_run_partition_defers_to_the_legacy_scan(
    db_session, caplog: pytest.LogCaptureFixture
) -> None:
    """A pointer row whose only failed condition is an *absent* run-partition
    field is a pre-existing row -- the shape the 20260804 backfill produces
    from a trace_events row written before that field existed -- not a
    mismatched one. It defers to the legacy event_id scan the same way a
    dangling pointer does, instead of failing the candidate on the spot.

    The legacy pointer here names a different, genuinely valid checkpoint, so
    the deferral has somewhere to land and the candidate recovers. That is
    what separates this cell from
    test_recovery_rejects_checkpoint_without_current_run_provenance[None],
    where both pointers name the same field-less row and the deferral
    correctly still ends in FAILED.
    """
    caplog.set_level(logging.INFO, logger="xagent.web.services.task_lease_service")
    user = _create_user(db_session, suffix="pk-absent-partition")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="pk-absent-partition",
    )
    pre_existing_anchor = TraceEvent(
        task_id=task.id,
        event_id="pk-absent-partition-preexisting",
        event_type="system_update_general",
        timestamp=utc_now(),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "snapshot": {"type": "checkpoint"},
            # No TASK_RUN_ID_TRACE_FIELD at all -- absent, not wrong.
        },
    )
    valid_legacy_row = TraceEvent(
        task_id=task.id,
        event_id="pk-absent-partition-valid",
        event_type="system_update_general",
        timestamp=utc_now(),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "snapshot": {"type": "checkpoint"},
            TASK_RUN_ID_TRACE_FIELD: task.run_id,
        },
    )
    db_session.add_all([pre_existing_anchor, valid_legacy_row])
    db_session.flush()
    task.last_checkpoint_trace_event_id = pre_existing_anchor.id
    task.last_checkpoint_event_id = valid_legacy_row.event_id
    db_session.commit()

    candidate = _candidate_for_task(task)
    assert (
        resolve_checkpoint_recovery(db_session, candidate)
        is CheckpointRecoveryVerdict.RECOVERABLE
    )
    # resolve_checkpoint_recovery is pure and was just called directly above
    # to check the verdict; clear its log line so the assertion below pins
    # the recovery path's own call, not a second echo of the first.
    caplog.clear()
    assert _recover_expired_task(db_session, task) == TaskStatus.PAUSED

    infos = [
        r
        for r in caplog.records
        if r.levelno == logging.INFO
        and r.name == "xagent.web.services.task_lease_service"
    ]
    assert len(infos) == 1, caplog.records
    assert infos[0].msg == (
        "Task %s's checkpoint pointer %s is missing its "
        "run-partition field; deferring to the legacy event_id scan "
        "rather than treating the row as a mismatch"
    )


def test_pk_anchor_wrong_run_partition_still_fails_without_a_legacy_retry(
    db_session,
) -> None:
    """The reclassification above requires the run-partition field to be
    *absent*. A field that is present and holds another run's id is a real
    mismatch: it fails the candidate on the spot and never reaches the legacy
    scan, even when the legacy pointer names a genuinely valid checkpoint
    that scan would have accepted.
    """
    user = _create_user(db_session, suffix="pk-wrong-partition")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="pk-wrong-partition",
    )
    mismatched_anchor = TraceEvent(
        task_id=task.id,
        event_id="pk-wrong-partition-mismatched",
        event_type="system_update_general",
        timestamp=utc_now(),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "snapshot": {"type": "checkpoint"},
            TASK_RUN_ID_TRACE_FIELD: "some-other-run",
        },
    )
    valid_legacy_row = TraceEvent(
        task_id=task.id,
        event_id="pk-wrong-partition-valid",
        event_type="system_update_general",
        timestamp=utc_now(),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "snapshot": {"type": "checkpoint"},
            TASK_RUN_ID_TRACE_FIELD: task.run_id,
        },
    )
    db_session.add_all([mismatched_anchor, valid_legacy_row])
    db_session.flush()
    task.last_checkpoint_trace_event_id = mismatched_anchor.id
    task.last_checkpoint_event_id = valid_legacy_row.event_id
    db_session.commit()

    candidate = _candidate_for_task(task)
    assert (
        resolve_checkpoint_recovery(db_session, candidate)
        is CheckpointRecoveryVerdict.NOT_RECOVERABLE
    )
    assert _recover_expired_task(db_session, task) == TaskStatus.FAILED


def test_pk_anchor_no_run_id_candidate_fails_closed_before_the_shared_predicate(
    db_session,
) -> None:
    """A candidate with no run_id (a run claim that never wrote provenance)
    fails closed before the shared predicate is even consulted. The shared
    predicate treats a null run_id as a legitimate partition, matched by a
    row whose own run field is also absent -- but that answer belongs to the
    root-checkpoint read path the predicate was written for, not to lease
    recovery, which cannot prove an exact pointer belongs to the expired run
    without provenance. This is the guard the read path's shared predicate
    delegates back to its callers; this test pins that lease recovery
    actually applies it.
    """
    user = _create_user(db_session, suffix="pk-no-run-id")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="pk-no-run-id",
    )
    task.run_id = None
    row = TraceEvent(
        task_id=task.id,
        event_id="pk-no-run-id-row",
        event_type="system_update_general",
        timestamp=utc_now(),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "snapshot": {"type": "checkpoint"},
            # No TASK_RUN_ID_TRACE_FIELD -- matches a null run_id under the
            # shared predicate's own rule, which is exactly why this
            # candidate must fail closed before reaching it.
        },
    )
    db_session.add(row)
    db_session.flush()
    task.last_checkpoint_trace_event_id = row.id
    db_session.commit()

    candidate = _candidate_for_task(task)
    assert (
        resolve_checkpoint_recovery(db_session, candidate)
        is CheckpointRecoveryVerdict.NOT_RECOVERABLE
    )


def test_pk_anchor_execution_identity_mismatch_is_not_recoverable(
    db_session,
) -> None:
    """The shared predicate's execution-identity condition is evaluated for
    lease recovery too, with execution_id set to str(candidate.task_id) --
    web's execution id is the task id, the same value the other two
    by-primary-key callers pass. A row whose payload carries a different
    execution_id fails that condition and is not recoverable, proving the
    sixth condition is actually wired up rather than silently skipped."""
    user = _create_user(db_session, suffix="pk-wrong-execution-id")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="pk-wrong-execution-id",
    )
    row = TraceEvent(
        task_id=task.id,
        event_id="pk-wrong-execution-id-row",
        event_type="system_update_general",
        timestamp=utc_now(),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "snapshot": {"type": "checkpoint"},
            TASK_RUN_ID_TRACE_FIELD: task.run_id,
            "execution_id": "not-this-task-id",
        },
    )
    db_session.add(row)
    db_session.flush()
    task.last_checkpoint_trace_event_id = row.id
    db_session.commit()

    candidate = _candidate_for_task(task)
    assert (
        resolve_checkpoint_recovery(db_session, candidate)
        is CheckpointRecoveryVerdict.NOT_RECOVERABLE
    )


def test_pk_anchor_execution_identity_match_is_recoverable(db_session) -> None:
    """The positive side of the identity check above: a row whose payload
    carries the *correct* execution_id -- str(task.id), matching what
    _candidate_row_failures passes -- is recoverable.

    This is not redundant with the happy-path checkpoints built by
    _create_expired_task, which never set an execution_id field at all;
    checkpoint_execution_id then reads as empty and the identity condition
    is vacuously satisfied regardless of what value is passed in. Only a
    row that actually carries the right id proves the right id is the one
    being passed, rather than merely that nothing is being compared.
    """
    user = _create_user(db_session, suffix="pk-right-execution-id")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="pk-right-execution-id",
    )
    row = TraceEvent(
        task_id=task.id,
        event_id="pk-right-execution-id-row",
        event_type="system_update_general",
        timestamp=utc_now(),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "snapshot": {"type": "checkpoint"},
            TASK_RUN_ID_TRACE_FIELD: task.run_id,
            "execution_id": str(task.id),
        },
    )
    db_session.add(row)
    db_session.flush()
    task.last_checkpoint_trace_event_id = row.id
    db_session.commit()

    candidate = _candidate_for_task(task)
    assert (
        resolve_checkpoint_recovery(db_session, candidate)
        is CheckpointRecoveryVerdict.RECOVERABLE
    )


_RECOVERY_SINGLE_FAULT_FIELDS = [
    "task_id",
    "event_type",
    "build_id",
    "checkpoint_type",
    "run_id",
]


def _mutate_recovery_row_field(
    field: str,
    row_kwargs: dict,
    data: dict,
    other_task_id: int,
) -> None:
    """Corrupt exactly one field _checkpoint_row_matches_candidate checks
    on the PK-anchored recovery path, leaving every other field --
    including the run field -- valid."""
    if field == "task_id":
        row_kwargs["task_id"] = other_task_id
    elif field == "event_type":
        row_kwargs["event_type"] = "agent_progress"
    elif field == "build_id":
        row_kwargs["build_id"] = "agent_child"
    elif field == "checkpoint_type":
        data["checkpoint_type"] = "not_a_readable_checkpoint_type"
    elif field == "run_id":
        data[TASK_RUN_ID_TRACE_FIELD] = "some-other-run"
    else:
        raise AssertionError(f"unknown single-fault field {field}")


@pytest.mark.parametrize("field", _RECOVERY_SINGLE_FAULT_FIELDS)
def test_pk_anchor_single_fault_is_not_recoverable(db_session, field: str) -> None:
    """Each conjunct _checkpoint_row_matches_candidate checks must
    independently be load-bearing on the PK-anchored recovery path: a row
    wrong in exactly one field (every other field, including the run
    field, valid) must still resolve NOT_RECOVERABLE (the candidate fails
    the lease). test_pk_anchor_validation_failure_does_not_fall_back_to_
    the_legacy_scan above only proves build_id does real work; task_id and
    event_type have no such proof, because the legacy scan's own query
    already filters both -- they are redundant there and only load-bearing
    on this PK path, which loads a row by raw primary key with no such
    filter. This test kills each conjunct independently.
    """
    user = _create_user(db_session, suffix=f"pk-single-fault-{field}")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix=f"pk-single-fault-{field}",
    )
    other_task = Task(
        user_id=int(user.id),
        title="Other task",
        description="lease recovery test",
        status=TaskStatus.PENDING,
    )
    db_session.add(other_task)
    db_session.flush()
    other_task_id = int(other_task.id)

    row_kwargs: dict = dict(
        task_id=task.id,
        event_id=f"pk-single-fault-{field}",
        event_type="system_update_general",
        timestamp=utc_now(),
    )
    data: dict = {
        "checkpoint_type": CHECKPOINT_TYPE,
        "snapshot": {"type": "checkpoint"},
        TASK_RUN_ID_TRACE_FIELD: task.run_id,
    }
    _mutate_recovery_row_field(field, row_kwargs, data, other_task_id)

    row = TraceEvent(data=data, **row_kwargs)
    db_session.add(row)
    db_session.flush()
    task.last_checkpoint_trace_event_id = row.id
    db_session.commit()

    candidate = _candidate_for_task(task)
    assert (
        resolve_checkpoint_recovery(db_session, candidate)
        is CheckpointRecoveryVerdict.NOT_RECOVERABLE
    )
    assert _recover_expired_task(db_session, task) == TaskStatus.FAILED


def test_dangling_pk_pointer_falls_back_to_the_legacy_scan(
    sqlite_no_anchor_fk_session,
) -> None:
    """A pointer whose row is gone is only reachable on a database upgraded
    without this column's FK (the D1 asymmetry -- see the migration and
    test_task_deletion_checkpoint_pointer.py). It is a compatibility-window
    state, not corruption: recovery must fall back to the legacy scan
    instead of failing the candidate outright.
    """
    session = sqlite_no_anchor_fk_session
    user = User(
        username="lease-recovery-dangling",
        password_hash="hash",
        is_admin=False,
    )
    session.add(user)
    session.flush()
    task = Task(
        user_id=int(user.id),
        title="dangling pointer",
        description="lease recovery test",
        status=TaskStatus.RUNNING,
        execution_mode="auto",
        runner_id="dead-runner-dangling",
        run_id="run-dangling",
        lease_expires_at=utc_now() - timedelta(seconds=5),
        last_heartbeat_at=utc_now() - timedelta(seconds=10),
        state_version=3,
        control_state="running",
    )
    session.add(task)
    session.flush()
    legacy_checkpoint = TraceEvent(
        task_id=task.id,
        event_id="dangling-legacy-checkpoint",
        event_type="system_update_general",
        timestamp=utc_now(),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "snapshot": {"type": "checkpoint"},
            TASK_RUN_ID_TRACE_FIELD: "run-dangling",
        },
    )
    session.add(legacy_checkpoint)
    session.flush()
    task.last_checkpoint_event_id = legacy_checkpoint.event_id
    # No DB-level FK on this schema form -- this id names no row.
    task.last_checkpoint_trace_event_id = legacy_checkpoint.id + 999
    session.commit()

    candidate = _candidate_for_task(task)
    assert (
        resolve_checkpoint_recovery(session, candidate)
        is CheckpointRecoveryVerdict.RECOVERABLE
    )


def test_ambiguous_legacy_checkpoint_skips_the_candidate_for_the_next_sweep(
    db_session,
) -> None:
    """Two trace_events rows share the same legacy event_id within one
    task's root partition -- the row's identity itself cannot be
    determined from that string alone. This is not folded into FAILED:
    the candidate is left untouched (lease and status unchanged, no
    recovery statement executes) so a later sweep gets another chance to
    resolve it, instead of permanently failing a task whose checkpoint may
    in fact be readable once the ambiguity is gone.
    """
    user = _create_user(db_session, suffix="ambiguous")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="ambiguous",
        with_checkpoint=True,
    )
    duplicate_checkpoint = TraceEvent(
        task_id=task.id,
        event_id=task.last_checkpoint_event_id,
        event_type="system_update_general",
        timestamp=utc_now(),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "snapshot": {"type": "checkpoint"},
            TASK_RUN_ID_TRACE_FIELD: task.run_id,
        },
    )
    db_session.add(duplicate_checkpoint)
    db_session.commit()

    candidate = _candidate_for_task(task)
    assert (
        resolve_checkpoint_recovery(db_session, candidate)
        is CheckpointRecoveryVerdict.INDETERMINATE
    )
    assert _recover_expired_task(db_session, task) is None

    db_session.expire_all()
    persisted = db_session.get(Task, int(task.id))
    assert persisted.status == TaskStatus.RUNNING
    assert persisted.runner_id is not None
    assert persisted.lease_expires_at is not None

    # Nothing about the skip removed the candidate from the expired-lease
    # scan -- a later sweep still selects it.
    rescanned = get_expired_task_lease_candidates(
        db_session,
        cutoff=utc_now(),
        limit=10,
    )
    assert any(c.task_id == int(task.id) for c in rescanned)


def test_ambiguous_legacy_checkpoint_registers_a_degradation_signal(
    db_session,
) -> None:
    """#1071 requires zero-or-multiple matches to fail closed *and* emit
    telemetry. The verdict is only half of that: an ambiguity nothing can
    resolve has to be visible to monitoring, not just to the log."""
    from xagent.web.services.ops_signals import (
        CHECKPOINT_LEGACY_POINTER_AMBIGUOUS,
        active_degradations,
        clear_degradation,
    )

    user = _create_user(db_session, suffix="ambiguous-signal")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="ambiguous-signal",
        with_checkpoint=True,
    )
    db_session.add(
        TraceEvent(
            task_id=task.id,
            event_id=task.last_checkpoint_event_id,
            event_type="system_update_general",
            timestamp=utc_now(),
            data={
                "checkpoint_type": CHECKPOINT_TYPE,
                "snapshot": {"type": "checkpoint"},
                TASK_RUN_ID_TRACE_FIELD: task.run_id,
            },
        )
    )
    db_session.commit()

    clear_degradation(CHECKPOINT_LEGACY_POINTER_AMBIGUOUS)
    try:
        candidate = _candidate_for_task(task)
        assert (
            resolve_checkpoint_recovery(db_session, candidate)
            is CheckpointRecoveryVerdict.INDETERMINATE
        )
        assert CHECKPOINT_LEGACY_POINTER_AMBIGUOUS in active_degradations()
    finally:
        clear_degradation(CHECKPOINT_LEGACY_POINTER_AMBIGUOUS)


def test_pk_anchor_missing_run_partition_then_ambiguous_legacy_id_is_indeterminate(
    db_session,
) -> None:
    """The two deferral conditions meeting at once: a pointer row missing only
    its run-partition field defers to the legacy scan, and that scan finds the
    legacy event_id on more than one row.

    Pinned because it is a change of kind, not of degree. Before the pointer
    row was reclassified, a field-less pointer row resolved NOT_RECOVERABLE on
    the spot and the task was failed once. The same candidate now reaches the
    legacy scan's ambiguity branch, so the verdict is INDETERMINATE: the task
    and its lease are left untouched and every later sweep selects the
    candidate again, with no terminal exit of its own. Leaving it untouched is
    deliberate -- an ambiguity a later sweep may resolve must not fail a
    readable checkpoint -- but the repetition is unbounded, which is why the
    degradation signal asserted below has to fire. Whether an INDETERMINATE
    candidate should reach a terminal state after a bounded number of sweeps
    is #2118, not decided here.

    Both conditions are needed to reach this path. Either alone is already
    covered: test_pk_anchor_missing_run_partition_defers_to_the_legacy_scan
    has the deferral landing on a valid row, and
    test_ambiguous_legacy_checkpoint_skips_the_candidate_for_the_next_sweep
    reaches the ambiguity through a pointer that was never set.
    """
    from xagent.web.services.ops_signals import (
        CHECKPOINT_LEGACY_POINTER_AMBIGUOUS,
        active_degradations,
        clear_degradation,
    )

    user = _create_user(db_session, suffix="absent-partition-ambiguous")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="absent-partition-ambiguous",
    )
    legacy_event_id = "absent-partition-ambiguous-legacy"
    pre_existing_anchor = TraceEvent(
        task_id=task.id,
        event_id="absent-partition-ambiguous-preexisting",
        event_type="system_update_general",
        timestamp=utc_now(),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "snapshot": {"type": "checkpoint"},
            # No TASK_RUN_ID_TRACE_FIELD at all -- absent, not wrong, so the
            # pointer defers instead of failing the candidate on the spot.
        },
    )
    duplicate_legacy_rows = [
        TraceEvent(
            task_id=task.id,
            event_id=legacy_event_id,
            event_type="system_update_general",
            timestamp=utc_now(),
            data={
                "checkpoint_type": CHECKPOINT_TYPE,
                "snapshot": {"type": "checkpoint"},
                TASK_RUN_ID_TRACE_FIELD: task.run_id,
            },
        )
        for _ in range(2)
    ]
    db_session.add_all([pre_existing_anchor, *duplicate_legacy_rows])
    db_session.flush()
    task.last_checkpoint_trace_event_id = pre_existing_anchor.id
    task.last_checkpoint_event_id = legacy_event_id
    db_session.commit()

    clear_degradation(CHECKPOINT_LEGACY_POINTER_AMBIGUOUS)
    try:
        candidate = _candidate_for_task(task)
        assert (
            resolve_checkpoint_recovery(db_session, candidate)
            is CheckpointRecoveryVerdict.INDETERMINATE
        )
        assert CHECKPOINT_LEGACY_POINTER_AMBIGUOUS in active_degradations()

        assert _recover_expired_task(db_session, task) is None

        db_session.expire_all()
        persisted = db_session.get(Task, int(task.id))
        assert persisted.status == TaskStatus.RUNNING
        assert persisted.runner_id is not None
        assert persisted.lease_expires_at is not None

        # No terminal exit: the candidate is still on the expired-lease scan,
        # so the next sweep resolves the same two conditions the same way.
        rescanned = get_expired_task_lease_candidates(
            db_session,
            cutoff=utc_now(),
            limit=10,
        )
        assert any(c.task_id == int(task.id) for c in rescanned)
    finally:
        clear_degradation(CHECKPOINT_LEGACY_POINTER_AMBIGUOUS)


@pytest.mark.asyncio
async def test_a_clean_recovery_sweep_clears_the_ambiguity_signal(
    db_session,
) -> None:
    """One completed drain is the reporting unit: a sweep that finds no
    ambiguity clears a signal an earlier sweep left set, so the signal
    cannot latch on a resolved condition."""
    from xagent.web.services.ops_signals import (
        CHECKPOINT_LEGACY_POINTER_AMBIGUOUS,
        active_degradations,
        clear_degradation,
        register_degradation,
    )

    user = _create_user(db_session, suffix="clean-sweep")
    _create_expired_task(db_session, user_id=int(user.id), suffix="clean-sweep")
    db_session.commit()

    register_degradation(
        CHECKPOINT_LEGACY_POINTER_AMBIGUOUS, "left over from an earlier sweep"
    )
    try:
        await recover_expired_task_leases_until_cutoff(
            cutoff=utc_now(),
            batch_size=10,
        )
        assert CHECKPOINT_LEGACY_POINTER_AMBIGUOUS not in active_degradations()
    finally:
        clear_degradation(CHECKPOINT_LEGACY_POINTER_AMBIGUOUS)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.asyncio
async def test_ambiguity_signal_survives_a_later_clean_candidate_in_the_same_sweep(
    db_session,
    batch_size: int,
) -> None:
    """A drain that hits an ambiguity and then a resolvable candidate still
    reports the ambiguity once it finishes.

    The clear belongs to the whole drain, not to a candidate or a page. At
    either finer grain the clean candidate processed after the ambiguous one
    would erase the signal before the sweep ends, and an ambiguity every
    sweep re-hits could never stay visible. The two batch sizes separate
    those grains: 10 puts both candidates in one page, 1 puts the clean one
    in the page after the ambiguous one.
    """
    from xagent.web.services.ops_signals import (
        CHECKPOINT_LEGACY_POINTER_AMBIGUOUS,
        active_degradations,
        clear_degradation,
    )

    user = _create_user(db_session, suffix=f"ambiguous-then-clean-{batch_size}")
    ambiguous = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix=f"ambiguous-then-clean-first-{batch_size}",
        with_checkpoint=True,
    )
    db_session.add(
        TraceEvent(
            task_id=ambiguous.id,
            event_id=ambiguous.last_checkpoint_event_id,
            event_type="system_update_general",
            timestamp=utc_now(),
            data={
                "checkpoint_type": CHECKPOINT_TYPE,
                "snapshot": {"type": "checkpoint"},
                TASK_RUN_ID_TRACE_FIELD: ambiguous.run_id,
            },
        )
    )
    clean = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix=f"ambiguous-then-clean-second-{batch_size}",
        with_checkpoint=True,
    )
    db_session.commit()

    cutoff = utc_now()
    # Pin the order the drain will see them in: this test is only meaningful
    # for a clean candidate that comes *after* an ambiguous one.
    scan_order = [
        candidate.task_id
        for candidate in get_expired_task_lease_candidates(
            db_session, cutoff=cutoff, limit=10
        )
    ]
    assert scan_order == [int(ambiguous.id), int(clean.id)]

    clear_degradation(CHECKPOINT_LEGACY_POINTER_AMBIGUOUS)
    try:
        await recover_expired_task_leases_until_cutoff(
            cutoff=cutoff, batch_size=batch_size
        )

        assert CHECKPOINT_LEGACY_POINTER_AMBIGUOUS in active_degradations()
        db_session.expire_all()
        # The clean candidate really was processed after the ambiguous one,
        # so the surviving signal is not merely an untouched sweep.
        assert db_session.get(Task, int(clean.id)).status is TaskStatus.PAUSED
        assert db_session.get(Task, int(ambiguous.id)).status is TaskStatus.RUNNING
    finally:
        clear_degradation(CHECKPOINT_LEGACY_POINTER_AMBIGUOUS)


@pytest.mark.parametrize(
    "replacement",
    ["heartbeat", "runner", "run", "state_version", "checkpoint", "checkpoint_pk"],
)
def test_recovery_candidate_cannot_overwrite_newer_task_state(
    db_session,
    replacement: str,
) -> None:
    user = _create_user(db_session, suffix=f"race-{replacement}")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix=f"race-{replacement}",
    )
    candidates = get_expired_task_lease_candidates(
        db_session,
        cutoff=utc_now(),
        limit=10,
    )
    assert len(candidates) == 1

    if replacement == "heartbeat":
        task.lease_expires_at = utc_now() + timedelta(minutes=1)
    elif replacement == "runner":
        task.runner_id = "replacement-runner"
    elif replacement == "run":
        task.run_id = "replacement-run"
    elif replacement == "state_version":
        task.state_version += 1
    elif replacement == "checkpoint":
        task.last_checkpoint_event_id = "new-checkpoint"
    else:
        # checkpoint_pk: only the exact-row pointer moves, the legacy string
        # column is left alone -- a single-column drift the fence's
        # conjunction over both pointer columns must still catch.
        new_event = TraceEvent(
            task_id=task.id,
            event_id="new-checkpoint-pk",
            event_type="system_update_general",
            timestamp=utc_now(),
            step_id=None,
            parent_event_id=None,
            data={
                "checkpoint_type": CHECKPOINT_TYPE,
                "snapshot": {"type": "checkpoint"},
                TASK_RUN_ID_TRACE_FIELD: task.run_id,
            },
        )
        db_session.add(new_event)
        db_session.flush()
        task.last_checkpoint_trace_event_id = new_event.id
    db_session.commit()

    assert (
        recover_task_lease_candidate_isolated(
            candidates[0],
            recovered_at=utc_now(),
        )
        is None
    )

    db_session.refresh(task)
    assert task.status == TaskStatus.RUNNING
    assert task.runner_id is not None


def test_candidate_fence_allows_only_one_recovery(db_session) -> None:
    user = _create_user(db_session, suffix="single-winner")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="single-winner",
    )
    candidate = get_expired_task_lease_candidates(
        db_session,
        cutoff=utc_now(),
        limit=1,
    )[0]

    assert (
        recover_task_lease_candidate_isolated(
            candidate,
            recovered_at=utc_now(),
        )
        == TaskStatus.FAILED
    )
    assert (
        recover_task_lease_candidate_isolated(
            candidate,
            recovered_at=utc_now(),
        )
        is None
    )
    db_session.refresh(task)
    assert task.state_version == 4


def test_two_recovery_workers_have_one_atomic_winner(db_session) -> None:
    user = _create_user(db_session, suffix="concurrent-winner")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="concurrent-winner",
    )
    candidate = get_expired_task_lease_candidates(
        db_session,
        cutoff=utc_now(),
        limit=1,
    )[0]
    db_session.rollback()
    barrier = Barrier(2)

    def recover() -> TaskStatus | None:
        barrier.wait()
        return recover_task_lease_candidate_isolated(
            candidate,
            recovered_at=utc_now(),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = [
            future.result() for future in [executor.submit(recover) for _ in range(2)]
        ]

    assert results.count(TaskStatus.FAILED) == 1
    assert results.count(None) == 1
    db_session.expire_all()
    assert db_session.get(Task, int(task.id)).state_version == 4


def test_postgresql_candidate_query_partitions_workers_with_skip_locked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_sql: list[str] = []

    def capture_first(query: Query):
        captured_sql.append(
            str(
                query.limit(1).statement.compile(
                    dialect=postgresql.dialect(),
                    compile_kwargs={"literal_binds": True},
                )
            )
        )
        return None

    monkeypatch.setattr(Query, "first", capture_first)
    db = Session()
    try:
        candidate = task_lease_service.get_next_expired_task_lease_candidate_for_update(
            db,
            cutoff=utc_now(),
            after=None,
        )
    finally:
        db.close()

    assert candidate is None
    assert len(captured_sql) == 1
    normalized_sql = " ".join(captured_sql[0].split())
    assert "ORDER BY tasks.lease_expires_at ASC, tasks.id ASC" in normalized_sql
    assert "LIMIT 1" in normalized_sql
    assert "FOR UPDATE OF tasks SKIP LOCKED" in normalized_sql


def test_postgresql_batch_uses_one_short_locked_transaction_per_candidate(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = _create_user(db_session, suffix="postgresql-partition")
    expired_ids = [
        int(
            _create_expired_task(
                db_session,
                user_id=int(user.id),
                suffix=f"postgresql-partition-{index}",
            ).id
        )
        for index in range(3)
    ]
    selected_sessions: list[Session] = []

    def select_next_candidate(
        db: Session,
        *,
        cutoff: datetime,
        after: tuple[datetime, int] | None,
    ):
        selected_sessions.append(db)
        candidates = task_lease_service.get_expired_task_lease_candidates(
            db,
            cutoff=cutoff,
            limit=1,
            after=after,
        )
        return candidates[0] if candidates else None

    def reject_legacy_page_scan(*args, **kwargs):
        raise AssertionError("PostgreSQL recovery must not scan an unlocked page")

    monkeypatch.setattr(
        task_lease_recovery,
        "_use_postgresql_recovery_partitioning",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        task_lease_recovery,
        "get_next_expired_task_lease_candidate_for_update",
        select_next_candidate,
        raising=False,
    )
    monkeypatch.setattr(
        task_lease_recovery,
        "get_expired_task_lease_candidates",
        reject_legacy_page_scan,
    )

    first = task_lease_recovery.recover_expired_task_leases_batch_isolated(
        cutoff=utc_now(),
        batch_size=2,
        after=None,
    )
    second = task_lease_recovery.recover_expired_task_leases_batch_isolated(
        cutoff=utc_now(),
        batch_size=2,
        after=first.next_cursor,
    )

    assert first.scanned == 2
    assert first.recovered == 2
    assert second.scanned == 1
    assert second.recovered == 1
    assert len(selected_sessions) == 4
    assert len({id(db) for db in selected_sessions}) == 4
    db_session.expire_all()
    assert {db_session.get(Task, task_id).status for task_id in expired_ids} == {
        TaskStatus.FAILED
    }


def test_postgresql_batch_does_not_count_a_failed_candidate_transaction(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = _create_user(db_session, suffix="postgresql-commit-failure")
    first_task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="postgresql-commit-failure-first",
    )
    second_task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="postgresql-commit-failure-second",
    )
    commit_state = {"must_fail": True}

    class CommitFailingSession(Session):
        def commit(self) -> None:
            if commit_state["must_fail"]:
                commit_state["must_fail"] = False
                raise RuntimeError("simulated commit failure")
            super().commit()

    TestSessionLocal = sessionmaker(
        bind=db_session.get_bind(),
        class_=CommitFailingSession,
        autocommit=False,
        autoflush=False,
    )

    def select_next_candidate(
        db: Session,
        *,
        cutoff: datetime,
        after: tuple[datetime, int] | None,
    ):
        candidates = task_lease_service.get_expired_task_lease_candidates(
            db,
            cutoff=cutoff,
            limit=1,
            after=after,
        )
        return candidates[0] if candidates else None

    monkeypatch.setattr(
        task_lease_recovery,
        "get_next_expired_task_lease_candidate_for_update",
        select_next_candidate,
    )
    monkeypatch.setattr(
        "xagent.web.models.database.get_session_local",
        lambda: TestSessionLocal,
    )

    batch = (
        task_lease_recovery._recover_expired_task_leases_batch_with_postgresql_locks(
            cutoff=utc_now(),
            batch_size=2,
            after=None,
        )
    )

    assert batch.scanned == 2
    assert batch.recovered == 1
    db_session.expire_all()
    assert db_session.get(Task, int(first_task.id)).status == TaskStatus.RUNNING
    assert db_session.get(Task, int(second_task.id)).status == TaskStatus.FAILED


def test_projection_failure_rolls_back_the_task_recovery_transaction(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = _create_user(db_session, suffix="projection-rollback")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="projection-rollback",
    )

    def fail_projection(*args, **kwargs):
        raise RuntimeError("projection failed")

    monkeypatch.setattr(
        task_lease_recovery,
        "sync_workforce_run_status",
        fail_projection,
    )

    candidate = get_expired_task_lease_candidates(
        db_session,
        cutoff=utc_now(),
        limit=1,
    )[0]
    db_session.rollback()

    with pytest.raises(RuntimeError, match="projection failed"):
        recover_task_lease_candidate_isolated(
            candidate,
            recovered_at=utc_now(),
        )

    db_session.expire_all()
    persisted = db_session.get(Task, int(task.id))
    assert persisted.status == TaskStatus.RUNNING
    assert persisted.runner_id == "dead-runner-projection-rollback"


def test_recovery_core_leaves_commit_to_the_session_owner(db_session) -> None:
    user = _create_user(db_session, suffix="no-commit")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="no-commit",
    )
    candidate = get_expired_task_lease_candidates(
        db_session,
        cutoff=utc_now(),
        limit=1,
    )[0]
    pending_user = User(
        username="unrelated-pending-user",
        password_hash="hash",
        is_admin=False,
    )
    db_session.add(pending_user)

    assert (
        recover_task_lease_candidate_no_commit(
            db_session,
            candidate,
            recovered_at=utc_now(),
        )
        == TaskStatus.FAILED
    )

    SessionLocal = get_session_local()
    with SessionLocal() as observer:
        assert observer.get(Task, int(task.id)).status == TaskStatus.RUNNING
        assert (
            observer.query(User)
            .filter(User.username == "unrelated-pending-user")
            .first()
            is None
        )

    db_session.rollback()
    db_session.expire_all()
    assert db_session.get(Task, int(task.id)).status == TaskStatus.RUNNING


@pytest.mark.asyncio
async def test_periodic_recovery_drains_more_than_one_batch_at_fixed_cutoff(
    db_session,
) -> None:
    user = _create_user(db_session, suffix="batch")
    expired_ids = [
        int(
            _create_expired_task(
                db_session,
                user_id=int(user.id),
                suffix=f"batch-{index}",
            ).id
        )
        for index in range(5)
    ]
    live = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="live",
    )
    live.lease_expires_at = utc_now() + timedelta(minutes=1)
    no_expiry = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="no-expiry",
    )
    no_expiry.lease_expires_at = None
    db_session.commit()

    recovered = await recover_expired_task_leases_until_cutoff(
        cutoff=utc_now(),
        batch_size=2,
    )

    assert recovered == 5
    db_session.expire_all()
    statuses = {
        int(task.id): task.status
        for task in db_session.query(Task).filter(Task.id.in_(expired_ids)).all()
    }
    assert statuses == {task_id: TaskStatus.FAILED for task_id in expired_ids}
    assert db_session.get(Task, int(live.id)).status == TaskStatus.RUNNING
    assert db_session.get(Task, int(no_expiry.id)).status == TaskStatus.RUNNING


@pytest.mark.asyncio
async def test_recovery_loop_survives_pool_timeout_and_waits_for_next_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    sleeps: list[float] = []

    async def fake_recover(*, cutoff: datetime, batch_size: int) -> int:
        nonlocal calls
        calls += 1
        assert cutoff.tzinfo is not None
        assert batch_size == 7
        if calls == 1:
            raise SQLAlchemyTimeoutError("pool checkout timed out")
        raise asyncio.CancelledError

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(
        task_lease_recovery,
        "recover_expired_task_leases_until_cutoff",
        fake_recover,
    )
    monkeypatch.setattr(task_lease_recovery.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await run_task_lease_recovery_loop(
            poll_interval_seconds=11,
            batch_size=7,
        )

    assert calls == 2
    assert sleeps == [11]
