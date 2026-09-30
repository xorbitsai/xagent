from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from xagent.sandbox.base import ExactGenerationProbe
from xagent.web.models.sandbox import DurableSandboxLifecycle
from xagent.web.services.durable_sandbox_lifecycle import (
    DurableLifecycleConflict,
    DurableSandboxDeleteCoordinator,
    DurableSandboxLifecycleRepository,
    DurableSandboxLifecycleService,
    LifecycleFence,
    RegisterLifecycle,
    backend_lifecycle_digest,
    digest_turn_identifier,
)

NOW = datetime(2026, 9, 14, 12, tzinfo=timezone.utc)


def _schema(engine: sa.Engine) -> None:
    metadata = sa.MetaData()
    sa.Table(
        "tasks",
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("runner_id", sa.String(255)),
        sa.Column("run_id", sa.String(64)),
        sa.Column("lease_attempt_id", sa.String(64)),
        sa.Column("status", sa.String(32)),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
    )
    metadata.create_all(engine)
    DurableSandboxLifecycle.__table__.create(engine)


@pytest.fixture
def sessions(tmp_path):
    engine = sa.create_engine(
        f"sqlite:///{tmp_path / 'lifecycles.db'}",
        connect_args={"check_same_thread": False, "timeout": 10},
    )
    with engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA journal_mode=WAL")
    _schema(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        engine.dispose()


def _request(scope: str = "a" * 64) -> RegisterLifecycle:
    return RegisterLifecycle(
        scope_digest=scope,
        task_id=7,
        run_id="run-1",
        lease_attempt_id="attempt-1",
        turn_digest=digest_turn_identifier("raw-turn-1"),
        eligible_at=NOW - timedelta(minutes=2),
        owner_lease_expires_at=NOW - timedelta(minutes=1),
    )


def _set_task(db, *, active: bool, attempt: str = "attempt-1") -> None:
    db.execute(sa.text("DELETE FROM tasks"))
    if active:
        db.execute(
            sa.text(
                "INSERT INTO tasks "
                "(id, runner_id, run_id, lease_attempt_id, status, lease_expires_at) "
                "VALUES (:id, :runner, :run, :attempt, :status, :expiry)"
            ),
            {
                "id": 7,
                "runner": "runner-1",
                "run": "run-1",
                "attempt": attempt,
                "status": "RUNNING",
                "expiry": NOW + timedelta(minutes=1),
            },
        )


def test_register_is_durable_before_create_and_contains_only_opaque_identity(
    sessions,
) -> None:
    with sessions() as db:
        fence = DurableSandboxLifecycleRepository(db).register(_request())
        db.commit()

    assert fence.state == "registered"
    assert fence.active_scope_digest == fence.scope_digest
    assert fence.create_phase == "not_started"
    assert fence.create_terminal_outcome is None
    assert len(fence.create_operation_token) == 64
    assert fence.version == 1
    assert fence.backend_lifecycle_digest == backend_lifecycle_digest(
        fence.scope_digest, fence.lifecycle_token
    )
    assert fence.backend_lifecycle_digest not in {
        fence.scope_digest,
        fence.lifecycle_token,
    }
    columns = {column.name for column in DurableSandboxLifecycle.__table__.columns}
    forbidden = {
        "resource_owner_key",
        "user",
        "user_id",
        "credentials",
        "env",
        "actor_id",
        "session_identity",
        "turn_id",
    }
    assert columns.isdisjoint(forbidden)
    with sessions() as db:
        row = db.execute(sa.select(DurableSandboxLifecycle)).scalar_one()
        persisted = " ".join(str(value) for value in row.__dict__.values())
    assert "raw-turn-1" not in persisted


def test_duplicate_scope_registration_fails_closed(sessions) -> None:
    with sessions() as db:
        DurableSandboxLifecycleRepository(db).register(_request())
        db.commit()
    with sessions() as db:
        with pytest.raises(DurableLifecycleConflict):
            DurableSandboxLifecycleRepository(db).register(_request())


def test_service_commits_registration_before_returning(sessions) -> None:
    service = DurableSandboxLifecycleService(sessions)
    fence = service.register(_request())
    with sessions() as independent_session:
        persisted = DurableSandboxLifecycleRepository(independent_session).get_by_scope(
            fence.scope_digest
        )
    assert persisted is not None
    assert persisted.lifecycle_token == fence.lifecycle_token


def test_ready_and_renew_require_the_exact_active_attempt(sessions) -> None:
    with sessions() as db:
        repo = DurableSandboxLifecycleRepository(db)
        registered = repo.register(_request())
        _set_task(db, active=True, attempt="successor-attempt")
        assert (
            repo.mark_ready(
                registered,
                now=NOW,
                owner_lease_expires_at=NOW + timedelta(minutes=1),
            )
            is None
        )
        _set_task(db, active=True)
        begun = repo.begin_create(registered, now=NOW)
        assert begun is not None
        completed = repo.complete_create(begun, now=NOW, outcome="success")
        assert completed is not None
        ready = repo.mark_ready(
            completed,
            now=NOW,
            owner_lease_expires_at=NOW + timedelta(minutes=1),
        )
        assert ready is not None
        assert ready.state == "ready"
        assert ready.version == completed.version + 1
        _set_task(db, active=True, attempt="successor-attempt")
        assert (
            repo.renew(
                ready,
                now=NOW,
                owner_lease_expires_at=NOW + timedelta(minutes=2),
            )
            is None
        )


def test_begin_and_complete_create_are_exact_monotonic_cas_transitions(
    sessions,
) -> None:
    with sessions() as db:
        repo = DurableSandboxLifecycleRepository(db)
        registered = repo.register(_request())
        _set_task(db, active=True, attempt="successor-attempt")
        assert repo.begin_create(registered, now=NOW) is None
        _set_task(db, active=True)
        begun = repo.begin_create(registered, now=NOW)
        assert begun is not None
        assert begun.create_phase == "may_publish"
        assert begun.version == registered.version + 1
        assert repo.begin_create(registered, now=NOW) is None

        wrong_operation = LifecycleFence(
            **{
                **begun.__dict__,
                "create_operation_token": "f" * 64,
            }
        )
        assert repo.complete_create(wrong_operation, now=NOW, outcome="success") is None
        completed = repo.complete_create(begun, now=NOW, outcome="success")
        assert completed is not None
        assert completed.create_phase == "terminal"
        assert completed.create_terminal_outcome == "success"
        assert repo.complete_create(begun, now=NOW, outcome="success") is None


def test_ready_requires_terminal_create_phase(sessions) -> None:
    with sessions() as db:
        repo = DurableSandboxLifecycleRepository(db)
        registered = repo.register(_request())
        _set_task(db, active=True)
        assert (
            repo.mark_ready(
                registered,
                now=NOW,
                owner_lease_expires_at=NOW + timedelta(minutes=1),
            )
            is None
        )


def test_terminal_absent_create_cannot_be_marked_ready(sessions) -> None:
    with sessions() as db:
        repo = DurableSandboxLifecycleRepository(db)
        registered = repo.register(_request())
        _set_task(db, active=True)
        begun = repo.begin_create(registered, now=NOW)
        assert begun is not None
        absent = repo.complete_create(begun, now=NOW, outcome="terminal_absent")
        assert absent is not None
        assert absent.create_terminal_outcome == "terminal_absent"
        assert (
            repo.mark_ready(
                absent,
                now=NOW,
                owner_lease_expires_at=NOW + timedelta(minutes=1),
            )
            is None
        )


def test_create_phase_and_state_guards_reject_current_version_fences(sessions) -> None:
    with sessions() as db:
        repo = DurableSandboxLifecycleRepository(db)
        registered = repo.register(_request())
        _set_task(db, active=True)

        # Removing quarantine_create's may_publish guard makes this succeed.
        assert (
            repo.quarantine_create(registered, now=NOW, claim_ttl=timedelta(minutes=1))
            is None
        )

        begun = repo.begin_create(registered, now=NOW)
        assert begun is not None
        db.execute(
            sa.update(DurableSandboxLifecycle)
            .where(DurableSandboxLifecycle.id == begun.id)
            .values(
                state="deleting",
                active_scope_digest=None,
                deleting_at=NOW,
                delete_claim_expires_at=NOW + timedelta(minutes=1),
            )
        )
        # The fence still has the current owner/version; only state rejects it.
        assert repo.complete_create(begun, now=NOW, outcome="success") is None

        db.execute(
            sa.update(DurableSandboxLifecycle)
            .where(DurableSandboxLifecycle.id == begun.id)
            .values(
                create_phase="terminal",
                create_terminal_outcome="success",
            )
        )
        current = repo.list_by_scope(begun.scope_digest)[0]
        # Removing observe_create's may_publish guard makes this succeed.
        assert repo.observe_create(current, now=NOW) is None


def test_age_only_never_reclaims_an_exact_active_attempt(sessions) -> None:
    with sessions() as db:
        repo = DurableSandboxLifecycleRepository(db)
        fence = repo.register(_request())
        _set_task(db, active=True)
        db.commit()
    with sessions() as db:
        claimed = DurableSandboxLifecycleRepository(db).claim_for_delete(
            fence, now=NOW, claim_ttl=timedelta(seconds=30)
        )
        assert claimed is None


def test_lost_attempt_can_be_claimed_and_tombstone_is_retryable(sessions) -> None:
    with sessions() as db:
        repo = DurableSandboxLifecycleRepository(db)
        registered = repo.register(_request())
        _set_task(db, active=True, attempt="successor-attempt")
        claimed = repo.claim_for_delete(
            registered, now=NOW, claim_ttl=timedelta(seconds=30)
        )
        assert claimed is not None
        assert claimed.state == "deleting"
        assert claimed.owner_token != registered.owner_token
        assert claimed.version == registered.version + 1
        assert claimed.delete_attempts == 1
        backed_off = repo.backoff_delete(
            claimed, now=NOW, retry_at=NOW + timedelta(minutes=1)
        )
        assert backed_off is not None
        assert backed_off.version == claimed.version + 1
        db.commit()

    with sessions() as db:
        repo = DurableSandboxLifecycleRepository(db)
        assert (
            repo.reclaim_delete(
                backed_off,
                now=NOW + timedelta(seconds=45),
                claim_ttl=timedelta(seconds=30),
            )
            is None
        )
        reclaimed = repo.reclaim_delete(
            backed_off,
            now=NOW + timedelta(minutes=2),
            claim_ttl=timedelta(seconds=30),
        )
        assert reclaimed is not None
        assert reclaimed.owner_token != backed_off.owner_token
        assert reclaimed.version == backed_off.version + 1
        assert reclaimed.delete_attempts == 2


def test_active_owner_can_tombstone_ready_terminal_generation(sessions) -> None:
    with sessions() as db:
        repo = DurableSandboxLifecycleRepository(db)
        registered = repo.register(_request())
        _set_task(db, active=True)
        begun = repo.begin_create(registered, now=NOW)
        assert begun is not None
        completed = repo.complete_create(begun, now=NOW, outcome="success")
        assert completed is not None
        ready = repo.mark_ready(
            completed,
            now=NOW,
            owner_lease_expires_at=NOW + timedelta(minutes=1),
        )
        assert ready is not None

        deleting = repo.mark_deleting_owned(
            ready, now=NOW, claim_ttl=timedelta(minutes=1)
        )

        assert deleting is not None
        assert deleting.state == "deleting"
        assert deleting.create_phase == "terminal"
        assert deleting.active_scope_digest is None
        assert deleting.owner_token != ready.owner_token
        assert deleting.version == ready.version + 1
        assert deleting.delete_attempts == 1


def test_active_owner_can_tombstone_registered_terminal_attach_failure(
    sessions,
) -> None:
    with sessions() as db:
        repo = DurableSandboxLifecycleRepository(db)
        registered = repo.register(_request())
        _set_task(db, active=True)
        begun = repo.begin_create(registered, now=NOW)
        assert begun is not None
        completed = repo.complete_create(begun, now=NOW, outcome="success")
        assert completed is not None

        deleting = repo.mark_deleting_owned(
            completed, now=NOW, claim_ttl=timedelta(minutes=1)
        )

        assert deleting is not None
        assert deleting.state == "deleting"
        assert deleting.create_phase == "terminal"


@pytest.mark.parametrize("phase", ["not_started", "observed"])
def test_active_owner_can_tombstone_other_non_ambiguous_phases(sessions, phase) -> None:
    with sessions() as db:
        repo = DurableSandboxLifecycleRepository(db)
        registered = repo.register(_request())
        if phase == "observed":
            db.execute(
                sa.update(DurableSandboxLifecycle)
                .where(DurableSandboxLifecycle.id == registered.id)
                .values(create_phase="observed")
            )

        deleting = repo.mark_deleting_owned(
            registered, now=NOW, claim_ttl=timedelta(minutes=1)
        )

        assert deleting is not None
        assert deleting.state == "deleting"
        assert deleting.create_phase == phase


def test_owned_tombstone_rejects_ambiguous_and_stale_fences(sessions) -> None:
    with sessions() as db:
        repo = DurableSandboxLifecycleRepository(db)
        registered = repo.register(_request())
        _set_task(db, active=True)
        begun = repo.begin_create(registered, now=NOW)
        assert begun is not None
        assert (
            repo.mark_deleting_owned(begun, now=NOW, claim_ttl=timedelta(minutes=1))
            is None
        )

        terminal = repo.complete_create(begun, now=NOW, outcome="success")
        assert terminal is not None
        corruptions = (
            replace(terminal, owner_token="f" * 64),
            replace(terminal, version=terminal.version + 1),
            replace(terminal, create_operation_token="e" * 64),
            replace(terminal, lifecycle_token="d" * 64),
            replace(terminal, backend_lifecycle_digest="c" * 64),
        )
        for stale in corruptions:
            assert (
                repo.mark_deleting_owned(stale, now=NOW, claim_ttl=timedelta(minutes=1))
                is None
            )

        deleting = repo.mark_deleting_owned(
            terminal, now=NOW, claim_ttl=timedelta(minutes=1)
        )
        assert deleting is not None
        assert (
            repo.mark_deleting_owned(deleting, now=NOW, claim_ttl=timedelta(minutes=1))
            is None
        )


def test_two_real_sessions_allow_only_one_delete_claim(sessions) -> None:
    with sessions() as db:
        fence = DurableSandboxLifecycleRepository(db).register(_request())
        db.commit()

    barrier = threading.Barrier(2)

    def claim():
        with sessions() as db:
            barrier.wait(timeout=5)
            result = DurableSandboxLifecycleRepository(db).claim_for_delete(
                fence, now=NOW, claim_ttl=timedelta(minutes=1)
            )
            db.commit()
            return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: claim(), range(2)))
    winners = [outcome for outcome in outcomes if outcome is not None]
    assert len(winners) == 1
    assert winners[0].delete_attempts == 1


def _race_transition(sessions, transition):
    barrier = threading.Barrier(2)

    def run():
        with sessions() as db:
            barrier.wait(timeout=5)
            result = transition(DurableSandboxLifecycleRepository(db))
            db.commit()
            return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        return list(executor.map(lambda _: run(), range(2)))


def test_two_real_sessions_allow_only_one_begin_create(sessions) -> None:
    with sessions() as db:
        fence = DurableSandboxLifecycleRepository(db).register(_request())
        _set_task(db, active=True)
        db.commit()
    outcomes = _race_transition(
        sessions, lambda repo: repo.begin_create(fence, now=NOW)
    )
    assert sum(outcome is not None for outcome in outcomes) == 1


def test_two_real_sessions_allow_only_one_complete_create(sessions) -> None:
    with sessions() as db:
        repo = DurableSandboxLifecycleRepository(db)
        fence = repo.register(_request())
        _set_task(db, active=True)
        begun = repo.begin_create(fence, now=NOW)
        assert begun is not None
        db.commit()
    outcomes = _race_transition(
        sessions,
        lambda repo: repo.complete_create(begun, now=NOW, outcome="success"),
    )
    assert sum(outcome is not None for outcome in outcomes) == 1


def test_two_real_sessions_allow_only_one_observe_create(sessions) -> None:
    service, claimed = _claimed_may_publish(sessions)
    outcomes = _race_transition(
        sessions, lambda repo: repo.observe_create(claimed, now=NOW)
    )
    assert sum(outcome is not None for outcome in outcomes) == 1


def test_two_real_sessions_allow_only_one_owner_quarantine(sessions) -> None:
    with sessions() as db:
        repo = DurableSandboxLifecycleRepository(db)
        registered = repo.register(_request())
        _set_task(db, active=True)
        begun = repo.begin_create(registered, now=NOW)
        assert begun is not None
        db.commit()
    outcomes = _race_transition(
        sessions,
        lambda repo: repo.quarantine_create(
            begun, now=NOW, claim_ttl=timedelta(minutes=1)
        ),
    )
    assert sum(outcome is not None for outcome in outcomes) == 1


@pytest.mark.parametrize("outcome", ["success", "terminal_absent"])
def test_late_completion_monotonically_completes_quarantined_create(
    sessions, outcome
) -> None:
    service, quarantined = _claimed_may_publish(sessions)
    stale_create_fence = replace(
        quarantined,
        owner_token="f" * 64,
        version=1,
    )

    completed = service.complete_quarantined_create(
        stale_create_fence, now=NOW, outcome=outcome
    )

    assert completed is not None
    assert completed.state == "deleting"
    assert completed.create_phase == "terminal"
    assert completed.owner_token == quarantined.owner_token
    assert completed.version == quarantined.version + 1
    assert (
        service.complete_quarantined_create(
            stale_create_fence, now=NOW, outcome=outcome
        )
        is None
    )


def test_quarantined_completion_requires_exact_operation_generation_and_phase(
    sessions,
) -> None:
    service = DurableSandboxLifecycleService(sessions)
    registered = service.register(_request())
    with sessions() as db:
        _set_task(db, active=True)
        db.commit()
    begun = service.begin_create(registered, now=NOW)
    assert begun is not None

    assert (
        service.complete_quarantined_create(begun, now=NOW, outcome="success") is None
    )
    quarantined = service.quarantine_create(
        begun, now=NOW, claim_ttl=timedelta(minutes=1)
    )
    assert quarantined is not None
    corruptions = (
        replace(begun, create_operation_token="f" * 64),
        replace(begun, lifecycle_token="e" * 64),
        replace(begun, backend_lifecycle_digest="d" * 64),
    )
    for stale in corruptions:
        assert (
            service.complete_quarantined_create(stale, now=NOW, outcome="success")
            is None
        )

    observed = service.observe_create(quarantined, now=NOW)
    assert observed is not None
    assert (
        service.complete_quarantined_create(begun, now=NOW, outcome="success") is None
    )
    with pytest.raises(ValueError, match="unsupported create completion outcome"):
        service.complete_quarantined_create(  # type: ignore[arg-type]
            begun, now=NOW, outcome="unknown"
        )


def test_late_completion_invalidates_sweeper_fence_and_returns_current_claim(
    sessions,
) -> None:
    service, quarantined = _claimed_may_publish(sessions)

    completed = service.complete_quarantined_create(
        quarantined, now=NOW, outcome="success"
    )

    assert completed is not None
    assert (
        service.backoff_delete(
            quarantined,
            now=NOW,
            retry_at=NOW + timedelta(minutes=1),
        )
        is None
    )
    current = service.list_by_scope(quarantined.scope_digest)[0]
    assert current == completed
    backed_off = service.backoff_delete(
        current,
        now=NOW,
        retry_at=NOW + timedelta(minutes=1),
    )
    assert backed_off is not None


def test_late_completion_racing_sweeper_leaves_only_current_terminal_fence(
    sessions,
) -> None:
    service, quarantined = _claimed_may_publish(sessions)
    barrier = threading.Barrier(2)

    def complete():
        with sessions() as db:
            barrier.wait(timeout=5)
            result = DurableSandboxLifecycleRepository(db).complete_quarantined_create(
                quarantined, now=NOW, outcome="success"
            )
            db.commit()
            return result

    def backoff():
        with sessions() as db:
            barrier.wait(timeout=5)
            result = DurableSandboxLifecycleRepository(db).backoff_delete(
                quarantined,
                now=NOW,
                retry_at=NOW + timedelta(minutes=1),
            )
            db.commit()
            return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        completion_future = executor.submit(complete)
        backoff_future = executor.submit(backoff)
        completed = completion_future.result(timeout=10)
        swept = backoff_future.result(timeout=10)

    assert completed is not None
    current = service.list_by_scope(quarantined.scope_digest)[0]
    assert current.create_phase == "terminal"
    assert current.version >= completed.version
    assert (
        service.backoff_delete(
            quarantined,
            now=NOW,
            retry_at=NOW + timedelta(minutes=2),
        )
        is None
    )
    if swept is not None:
        assert (
            service.backoff_delete(
                swept,
                now=NOW,
                retry_at=NOW + timedelta(minutes=2),
            )
            is None
        )


def test_old_late_completion_cannot_touch_successor_generation(sessions) -> None:
    service, quarantined = _claimed_may_publish(sessions)
    successor = service.register(_request(quarantined.scope_digest))

    completed = service.complete_quarantined_create(
        quarantined, now=NOW, outcome="terminal_absent"
    )

    assert completed is not None
    generations = service.list_by_scope(quarantined.scope_digest)
    assert generations[0].id == quarantined.id
    assert generations[0].create_phase == "terminal"
    assert generations[1].id == successor.id
    assert generations[1].lifecycle_token == successor.lifecycle_token
    assert generations[1].create_operation_token == successor.create_operation_token
    assert generations[1].state == "registered"
    assert generations[1].create_phase == "not_started"


def test_final_claim_dml_rechecks_a_racing_successor_attempt(sessions) -> None:
    with sessions() as db:
        repo = DurableSandboxLifecycleRepository(db)
        fence = repo.register(_request())
        assert [item.id for item in repo.list_reclaimable(now=NOW)] == [fence.id]
        db.commit()
    with sessions() as db:
        _set_task(db, active=True, attempt="attempt-1")
        db.commit()
    with sessions() as db:
        assert (
            DurableSandboxLifecycleRepository(db).claim_for_delete(
                fence, now=NOW, claim_ttl=timedelta(minutes=1)
            )
            is None
        )


def test_generation_digest_prevents_stale_deleter_aba(sessions) -> None:
    with sessions() as db:
        repo = DurableSandboxLifecycleRepository(db)
        first = repo.register(_request())
        first_claim = repo.claim_for_delete(
            first, now=NOW, claim_ttl=timedelta(minutes=1)
        )
        assert first_claim is not None
        assert first_claim.active_scope_digest is None
        second = repo.register(_request())
        db.commit()

    assert second.backend_lifecycle_digest != first.backend_lifecycle_digest
    assert second.lifecycle_token != first.lifecycle_token
    with sessions() as db:
        repo = DurableSandboxLifecycleRepository(db)
        assert repo.settle_delete(first_claim)
        assert not repo.settle_delete(first_claim)
        survivor = repo.get_by_scope(second.scope_digest)
        assert survivor is not None
        assert survivor.lifecycle_token == second.lifecycle_token
        assert survivor.backend_lifecycle_digest == second.backend_lifecycle_digest


def test_quarantined_may_publish_generation_does_not_block_successor(sessions) -> None:
    with sessions() as db:
        repo = DurableSandboxLifecycleRepository(db)
        first = repo.register(_request())
        _set_task(db, active=True)
        begun = repo.begin_create(first, now=NOW)
        assert begun is not None
        quarantined = repo.quarantine_create(
            begun, now=NOW, claim_ttl=timedelta(minutes=1)
        )
        assert quarantined is not None
        assert quarantined.create_phase == "may_publish"
        assert not repo.settle_delete(quarantined)
        successor = repo.register(_request())
        generations = repo.list_by_scope(first.scope_digest)
        db.commit()

    assert [item.id for item in generations] == [first.id, successor.id]
    assert generations[0].active_scope_digest is None
    assert generations[1].active_scope_digest == first.scope_digest
    assert generations[0].backend_lifecycle_digest != (
        generations[1].backend_lifecycle_digest
    )


def test_expired_quarantine_is_reclaimable_while_successor_attempt_is_active(
    sessions,
) -> None:
    with sessions() as db:
        repo = DurableSandboxLifecycleRepository(db)
        first = repo.register(_request())
        _set_task(db, active=True)
        begun = repo.begin_create(first, now=NOW)
        assert begun is not None
        quarantined = repo.quarantine_create(
            begun, now=NOW, claim_ttl=timedelta(seconds=1)
        )
        assert quarantined is not None
        successor = repo.register(_request())
        reclaimed = repo.reclaim_delete(
            quarantined,
            now=NOW + timedelta(seconds=2),
            claim_ttl=timedelta(minutes=1),
        )
        assert reclaimed is not None
        assert reclaimed.id == quarantined.id
        assert reclaimed.version == quarantined.version + 1
        active = repo.get_by_scope(successor.scope_digest)
        assert active is not None
        assert active.id == successor.id


def test_crashed_may_publish_owner_is_quarantined_by_stale_reclaimer(sessions) -> None:
    with sessions() as db:
        repo = DurableSandboxLifecycleRepository(db)
        registered = repo.register(_request())
        _set_task(db, active=True)
        begun = repo.begin_create(registered, now=NOW)
        assert begun is not None
        _set_task(db, active=False)
        claimed = repo.claim_for_delete(begun, now=NOW, claim_ttl=timedelta(minutes=1))
        assert claimed is not None
        assert claimed.create_phase == "may_publish"
        assert claimed.active_scope_digest is None


def test_two_real_sessions_allow_only_one_active_generation(sessions) -> None:
    barrier = threading.Barrier(2)

    def register():
        with sessions() as db:
            barrier.wait(timeout=5)
            try:
                result = DurableSandboxLifecycleRepository(db).register(_request())
                db.commit()
                return result
            except DurableLifecycleConflict:
                db.rollback()
                return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: register(), range(2)))
    assert sum(outcome is not None for outcome in outcomes) == 1


def test_stale_owner_or_version_cannot_settle(sessions) -> None:
    with sessions() as db:
        repo = DurableSandboxLifecycleRepository(db)
        registered = repo.register(_request())
        claim = repo.claim_for_delete(
            registered, now=NOW, claim_ttl=timedelta(seconds=1)
        )
        assert claim is not None
        db.commit()
    with sessions() as db:
        repo = DurableSandboxLifecycleRepository(db)
        reclaimed = repo.reclaim_delete(
            claim,
            now=NOW + timedelta(seconds=2),
            claim_ttl=timedelta(minutes=1),
        )
        assert reclaimed is not None
        assert not repo.settle_delete(claim)
        assert repo.settle_delete(reclaimed)


class _Backend:
    def __init__(self, presence=ExactGenerationProbe.ABSENT, *, fail_delete=False):
        self.presence = presence
        self.fail_delete = fail_delete
        self.probes: list[str] = []
        self.deletes: list[str] = []

    async def probe_durable_sandbox_strict(self, lifecycle_id: str):
        self.probes.append(lifecycle_id)
        if isinstance(self.presence, Exception):
            raise self.presence
        return self.presence

    async def delete_durable_sandbox_strict(self, lifecycle_id: str) -> None:
        self.deletes.append(lifecycle_id)
        if self.fail_delete:
            raise RuntimeError("delete uncertain")


def _claimed_may_publish(sessions):
    service = DurableSandboxLifecycleService(sessions)
    registered = service.register(_request())
    with sessions() as db:
        _set_task(db, active=True)
        db.commit()
    begun = service.begin_create(registered, now=NOW)
    assert begun is not None
    claimed = service.quarantine_create(begun, now=NOW, claim_ttl=timedelta(minutes=1))
    assert claimed is not None
    return service, claimed


@pytest.mark.parametrize(
    "presence",
    [
        ExactGenerationProbe.ABSENT,
        ExactGenerationProbe.UNKNOWN,
        RuntimeError("probe failed"),
    ],
)
def test_ambiguous_create_absent_or_unknown_only_backs_off(sessions, presence) -> None:
    service, claimed = _claimed_may_publish(sessions)
    backend = _Backend(presence)
    settled = asyncio.run(
        DurableSandboxDeleteCoordinator(service, backend).delete_claimed(
            claimed,
            now=NOW,
            retry_at=NOW + timedelta(minutes=1),
        )
    )
    assert not settled
    assert backend.deletes == []
    rows = service.list_by_scope(claimed.scope_digest)
    assert len(rows) == 1
    assert rows[0].create_phase == "may_publish"
    assert rows[0].retry_at.replace(tzinfo=timezone.utc) == NOW + timedelta(minutes=1)


def test_coordinator_logs_backend_failure_without_losing_backoff_error(
    sessions, caplog
) -> None:
    service, claimed = _claimed_may_publish(sessions)

    def fail_backoff(*_args, **_kwargs):
        raise RuntimeError("durable backoff failed")

    service.backoff_delete = fail_backoff  # type: ignore[method-assign]
    with caplog.at_level("WARNING"):
        with pytest.raises(RuntimeError, match="durable backoff failed"):
            asyncio.run(
                DurableSandboxDeleteCoordinator(
                    service, _Backend(RuntimeError("backend probe failed"))
                ).delete_claimed(
                    claimed,
                    now=NOW,
                    retry_at=NOW + timedelta(minutes=1),
                )
            )
    assert claimed.backend_lifecycle_digest in caplog.text
    assert "backend probe failed" in caplog.text


def test_coordinator_database_transition_does_not_block_event_loop(sessions) -> None:
    service, claimed = _claimed_may_publish(sessions)
    original = service.backoff_delete
    release = threading.Event()

    def blocking_backoff(*args, **kwargs):
        release.wait(timeout=1)
        return original(*args, **kwargs)

    service.backoff_delete = blocking_backoff  # type: ignore[method-assign]

    async def run() -> None:
        timer = threading.Timer(0.3, release.set)
        timer.start()
        started = time.monotonic()
        task = asyncio.create_task(
            DurableSandboxDeleteCoordinator(service, _Backend()).delete_claimed(
                claimed,
                now=NOW,
                retry_at=NOW + timedelta(minutes=1),
            )
        )
        await asyncio.sleep(0.02)
        elapsed = time.monotonic() - started
        release.set()
        assert not await task
        timer.cancel()
        assert elapsed < 0.15

    asyncio.run(run())


def test_invalid_retry_time_fails_before_backend_or_database_io(sessions) -> None:
    service, claimed = _claimed_may_publish(sessions)
    backend = _Backend(ExactGenerationProbe.PRESENT)
    with pytest.raises(ValueError, match="retry_at must be in the future"):
        asyncio.run(
            DurableSandboxDeleteCoordinator(service, backend).delete_claimed(
                claimed,
                now=NOW,
                retry_at=NOW,
            )
        )
    assert backend.probes == []
    current = service.list_by_scope(claimed.scope_digest)[0]
    assert current.version == claimed.version
    assert current.retry_at is None


def test_present_is_observed_before_delete_and_settlement(sessions) -> None:
    service, claimed = _claimed_may_publish(sessions)

    class InspectingBackend(_Backend):
        async def delete_durable_sandbox_strict(self, lifecycle_id: str) -> None:
            row = service.list_by_scope(claimed.scope_digest)[0]
            assert row.create_phase == "observed"
            await super().delete_durable_sandbox_strict(lifecycle_id)

    backend = InspectingBackend(ExactGenerationProbe.PRESENT)
    settled = asyncio.run(
        DurableSandboxDeleteCoordinator(service, backend).delete_claimed(
            claimed,
            now=NOW,
            retry_at=NOW + timedelta(minutes=1),
        )
    )
    assert settled
    assert backend.probes == [claimed.backend_lifecycle_digest]
    assert backend.deletes == [claimed.backend_lifecycle_digest]
    assert service.list_by_scope(claimed.scope_digest) == []


def test_crash_after_observe_retries_delete_without_reprobe(sessions) -> None:
    service, claimed = _claimed_may_publish(sessions)
    observed = service.observe_create(claimed, now=NOW)
    assert observed is not None
    backend = _Backend(ExactGenerationProbe.UNKNOWN)
    settled = asyncio.run(
        DurableSandboxDeleteCoordinator(service, backend).delete_claimed(
            observed,
            now=NOW,
            retry_at=NOW + timedelta(minutes=1),
        )
    )
    assert settled
    assert backend.probes == []


def test_crash_after_delete_before_settle_retries_idempotent_delete(sessions) -> None:
    service, claimed = _claimed_may_publish(sessions)
    observed = service.observe_create(claimed, now=NOW)
    assert observed is not None
    first_backend = _Backend()
    asyncio.run(
        first_backend.delete_durable_sandbox_strict(observed.backend_lifecycle_digest)
    )
    assert service.list_by_scope(claimed.scope_digest)[0].create_phase == "observed"

    retry_backend = _Backend()
    settled = asyncio.run(
        DurableSandboxDeleteCoordinator(service, retry_backend).delete_claimed(
            observed,
            now=NOW,
            retry_at=NOW + timedelta(minutes=1),
        )
    )
    assert settled
    assert retry_backend.probes == []
    assert retry_backend.deletes == [observed.backend_lifecycle_digest]


def test_terminal_create_deletes_without_probe(sessions) -> None:
    service = DurableSandboxLifecycleService(sessions)
    registered = service.register(_request())
    with sessions() as db:
        _set_task(db, active=True)
        db.commit()
    begun = service.begin_create(registered, now=NOW)
    assert begun is not None
    terminal = service.complete_create(begun, now=NOW, outcome="terminal_absent")
    assert terminal is not None
    with sessions() as db:
        _set_task(db, active=False)
        db.commit()
    claimed = service.claim_for_delete(
        terminal, now=NOW, claim_ttl=timedelta(minutes=1)
    )
    assert claimed is not None
    backend = _Backend(ExactGenerationProbe.UNKNOWN)
    assert asyncio.run(
        DurableSandboxDeleteCoordinator(service, backend).delete_claimed(
            claimed,
            now=NOW,
            retry_at=NOW + timedelta(minutes=1),
        )
    )
    assert backend.probes == []


def test_delete_failure_preserves_observed_tombstone(sessions) -> None:
    service, claimed = _claimed_may_publish(sessions)
    backend = _Backend(ExactGenerationProbe.PRESENT, fail_delete=True)
    settled = asyncio.run(
        DurableSandboxDeleteCoordinator(service, backend).delete_claimed(
            claimed,
            now=NOW,
            retry_at=NOW + timedelta(minutes=1),
        )
    )
    assert not settled
    row = service.list_by_scope(claimed.scope_digest)[0]
    assert row.create_phase == "observed"
    assert row.retry_at.replace(tzinfo=timezone.utc) == NOW + timedelta(minutes=1)
