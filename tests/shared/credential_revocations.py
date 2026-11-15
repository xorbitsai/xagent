"""Run the credential revocation outbox from tests that delete credentials.

A deletion only enqueues its provider revocation (a database trigger); the
outbox worker makes the provider call. Tests of a disconnect path therefore
assert the provider call after draining the outbox once.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session, sessionmaker

from xagent.web.models.credential_revocation import CredentialRevocation
from xagent.web.services.credential_revocations import (
    CredentialRevocationReport,
    run_credential_revocation_batch,
)


async def drain_credential_revocations(db: Session) -> CredentialRevocationReport:
    """Attempt every due record once, on the engine behind ``db``.

    Commits ``db`` first, so the worker's own sessions see what the code
    under test committed and ``db`` holds no transaction meanwhile. The
    clock is a second ahead because SQLite's ``CURRENT_TIMESTAMP`` (a new
    record's due time) has whole seconds.
    """
    db.commit()
    return await run_credential_revocation_batch(
        sessionmaker(bind=db.get_bind(), autoflush=False, autocommit=False),
        now=lambda: datetime.now(timezone.utc) + timedelta(seconds=1),
        limit=1000,
    )


def credential_revocation_statuses(db: Session) -> list[str]:
    """Every record's status, oldest first."""
    db.expire_all()
    return [
        status
        for (status,) in db.query(CredentialRevocation.status).order_by(
            CredentialRevocation.id
        )
    ]
