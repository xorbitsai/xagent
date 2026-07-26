"""Tests for automatic recovery of expired task execution leases."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from threading import Barrier

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import Query, Session, sessionmaker

from xagent.core.agent.checkpoint import CHECKPOINT_TYPE
from xagent.web.models.agent import Agent
from xagent.web.models.database import (
    Base,
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
    get_expired_task_lease_candidates,
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
        Base.metadata.drop_all(bind=get_engine())


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


@pytest.mark.parametrize(
    "replacement",
    ["heartbeat", "runner", "run", "state_version", "checkpoint"],
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
    else:
        task.last_checkpoint_event_id = "new-checkpoint"
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
