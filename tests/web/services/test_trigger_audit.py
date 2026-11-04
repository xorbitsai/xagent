"""Transaction-isolation tests for trigger audit writes.

Audit rows are the forensic record of callback handling, including failures
that roll the surrounding transaction back, so record_trigger_audit commits
each row on a short-lived session with its own connection. These tests pin
that isolation contract on SQLite and (when configured) PostgreSQL.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy.orm import Session

from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.trigger import TriggerAudit, TriggerAuditOutcome
from xagent.web.models.user import User
from xagent.web.services.trigger_providers.audit import (
    record_trigger_audit,
    record_trigger_audit_best_effort,
)


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'trigger_audit.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


@pytest.fixture()
def pg_session():
    """Session against a real Postgres. Set XAGENT_TEST_POSTGRES_URL to run
    (CI provides it in the PostgreSQL job)."""
    url = os.getenv("XAGENT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    init_db(db_url=url)
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


def _audit_rows(db: Session) -> list[TriggerAudit]:
    return db.query(TriggerAudit).order_by(TriggerAudit.id.asc()).all()


def test_audit_row_survives_caller_transaction_rollback(db_session: Session) -> None:
    """An audit row written mid-transaction outlives the caller's rollback,
    and the rollback still discards the caller's own pending state."""
    db_session.add(User(username="doomed-user", password_hash="hash", is_admin=False))

    audit = record_trigger_audit(
        db_session,
        outcome=TriggerAuditOutcome.EXECUTION_FAILURE,
        provider="webhook",
        callback_id="cb-rollback",
        detail={"stage": "fire", "error": "boom"},
    )

    db_session.rollback()

    assert db_session.query(User).filter(User.username == "doomed-user").count() == 0
    rows = _audit_rows(db_session)
    assert len(rows) == 1
    assert rows[0].id == audit.id
    assert rows[0].outcome == TriggerAuditOutcome.EXECUTION_FAILURE.value
    assert rows[0].callback_id == "cb-rollback"


def test_audit_write_does_not_commit_caller_pending_state(db_session: Session) -> None:
    """The audit commit is invisible to the caller's transaction: pending
    caller changes stay uncommitted until the caller decides."""
    db_session.add(User(username="pending-user", password_hash="hash", is_admin=False))

    record_trigger_audit(
        db_session,
        outcome=TriggerAuditOutcome.ACCEPTED,
        provider="webhook",
        callback_id="cb-pending",
    )

    with Session(bind=get_engine()) as other:
        assert other.query(User).filter(User.username == "pending-user").count() == 0
        assert other.query(TriggerAudit).count() == 1


def test_best_effort_swallows_audit_storage_failure(db_session: Session) -> None:
    """When the audit table itself is broken, best-effort returns None instead
    of raising, and the caller's session stays usable."""
    db_session.add(User(username="survivor", password_hash="hash", is_admin=False))
    TriggerAudit.__table__.drop(bind=get_engine())

    result = record_trigger_audit_best_effort(
        db_session,
        outcome=TriggerAuditOutcome.EXECUTION_FAILURE,
        provider="webhook",
        callback_id="cb-broken",
    )

    assert result is None
    db_session.commit()
    assert db_session.query(User).filter(User.username == "survivor").count() == 1


def test_audit_row_survives_rollback_of_flushed_changes_on_postgres(
    pg_session: Session,
) -> None:
    """Same isolation with caller changes already flushed to the database.

    SQLite cannot exercise this shape (a flushed writer holds the database
    lock, so a second connection cannot commit until it resolves); on
    Postgres the audit connection commits independently of the caller's
    open write transaction.
    """
    pg_session.add(User(username="doomed-user", password_hash="hash", is_admin=False))
    pg_session.flush()

    audit = record_trigger_audit(
        pg_session,
        outcome=TriggerAuditOutcome.EXECUTION_FAILURE,
        provider="webhook",
        callback_id="cb-pg-rollback",
        detail={"stage": "fire", "error": "boom"},
    )

    pg_session.rollback()

    assert pg_session.query(User).filter(User.username == "doomed-user").count() == 0
    rows = _audit_rows(pg_session)
    assert len(rows) == 1
    assert rows[0].id == audit.id
    assert rows[0].callback_id == "cb-pg-rollback"
