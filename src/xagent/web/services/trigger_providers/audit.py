"""Durable audit records for trigger callback and payload access activity."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from ...models.trigger import TriggerAudit, TriggerAuditOutcome

logger = logging.getLogger(__name__)


def record_trigger_audit(
    db: Session,
    *,
    outcome: TriggerAuditOutcome,
    provider: str | None = None,
    callback_id: str | None = None,
    trigger_id: int | None = None,
    detail: dict[str, Any] | None = None,
    remote_ip: str | None = None,
) -> TriggerAudit:
    """Persist one audit row immediately, isolated from the caller's session.

    Audit rows are the forensic record of callback handling — including the
    very failures that roll the surrounding transaction back — so each row is
    committed on a short-lived session with its own connection to the
    caller's engine. A later rollback of the caller's transaction cannot
    discard the row, committing it cannot commit the caller's pending state,
    and a caller session already in a failed state cannot block it.
    """
    audit = TriggerAudit(
        outcome=outcome.value,
        provider=provider,
        callback_id=callback_id,
        trigger_id=trigger_id,
        detail=detail,
        remote_ip=remote_ip,
    )
    # .engine deliberately resolves to the real Engine even when the caller's
    # session is bound to a Connection (e.g. a SAVEPOINT-based test fixture):
    # that is what lets audit rows survive the caller's rollback — and it also
    # means they would escape any future Connection+SAVEPOINT test isolation.
    with Session(bind=db.get_bind().engine) as audit_db:
        audit_db.add(audit)
        audit_db.commit()
        audit_db.refresh(audit)
        audit_db.expunge(audit)
    return audit


def record_trigger_audit_best_effort(
    db: Session,
    *,
    outcome: TriggerAuditOutcome,
    provider: str | None = None,
    callback_id: str | None = None,
    trigger_id: int | None = None,
    detail: dict[str, Any] | None = None,
    remote_ip: str | None = None,
) -> TriggerAudit | None:
    """Like record_trigger_audit, but never raises.

    Used on failure paths where the original error must win over any audit
    storage problem. The audit write runs on its own session, so the caller's
    session needs no cleanup here.
    """
    try:
        return record_trigger_audit(
            db,
            outcome=outcome,
            provider=provider,
            callback_id=callback_id,
            trigger_id=trigger_id,
            detail=detail,
            remote_ip=remote_ip,
        )
    except Exception:
        logger.exception("Failed to write trigger audit row (outcome=%s)", outcome)
        return None
