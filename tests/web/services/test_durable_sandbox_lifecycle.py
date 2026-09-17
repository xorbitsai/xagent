from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from xagent.web.models.sandbox import DurableSandboxLifecycle
from xagent.web.services.durable_sandbox_lifecycle import (
    DurableLifecycleConflict,
    DurableSandboxLifecycleRepository,
    DurableSandboxLifecycleService,
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
        ready = repo.mark_ready(
            registered,
            now=NOW,
            owner_lease_expires_at=NOW + timedelta(minutes=1),
        )
        assert ready is not None
        assert ready.state == "ready"
        assert ready.version == 2
        _set_task(db, active=True, attempt="successor-attempt")
        assert (
            repo.renew(
                ready,
                now=NOW,
                owner_lease_expires_at=NOW + timedelta(minutes=2),
            )
            is None
        )


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
        assert repo.settle_delete(first_claim)
        second = repo.register(_request())
        db.commit()

    assert second.backend_lifecycle_digest != first.backend_lifecycle_digest
    assert second.lifecycle_token != first.lifecycle_token
    with sessions() as db:
        repo = DurableSandboxLifecycleRepository(db)
        assert not repo.settle_delete(first_claim)
        survivor = repo.get_by_scope(second.scope_digest)
        assert survivor is not None
        assert survivor.lifecycle_token == second.lifecycle_token
        assert survivor.backend_lifecycle_digest == second.backend_lifecycle_digest


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
