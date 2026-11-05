"""``Idempotency-Key`` handling for the ``/v1`` create and append calls.

A client that times out on ``POST /v1/chat/tasks`` cannot tell whether the task
was created, and retrying creates (and runs) it a second time. With an
``Idempotency-Key`` header the server remembers the outcome per key:

* first request: the key is reserved (``in_progress``), the call runs, and the
  response is stored;
* same key, same body: the stored response is replayed and nothing is started;
* same key, different body: rejected (``reused``);
* same key while the first request is still running: rejected (``in_progress``)
  so the client retries later instead of racing the first request.

The reservation is a database row, so a retry that lands on another process, or
after a restart, sees it. The row is bound to the task inside the transaction
that creates it (:func:`bind_task_no_commit`), which is what makes a stale
reservation (the reserving process died) safe to take over: a reservation with
no bound task provably created nothing.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..models.api_idempotency_key import (
    IDEMPOTENCY_STATE_COMPLETED,
    IDEMPOTENCY_STATE_IN_PROGRESS,
    ApiIdempotencyKey,
)
from ..models.database import get_session_local
from ..utils.db_timezone import normalize_datetime_from_db

IDEMPOTENCY_KEY_TTL = timedelta(hours=24)
# How long a retry of a still-running request is answered "in progress" before
# the reserving process is presumed dead.
IDEMPOTENCY_LEASE = timedelta(seconds=120)
_PURGE_BATCH = 50

_KEY_PATTERN = re.compile(r"[\x21-\x7e]{1,255}")


class IdempotencyRejected(Exception):
    """The call must not run. ``reason`` is one of ``reused``, ``in_progress``,
    ``conflict`` (original task deleted) or ``outcome_unknown`` (the task was
    accepted but its response was never recorded)."""

    def __init__(self, reason: str, *, task_id: int | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.task_id = task_id


@dataclass(frozen=True)
class IdempotencyClaim:
    """This request owns the key and must run, then complete or release it."""

    claim_id: int


@dataclass(frozen=True)
class IdempotencyReplay:
    """The key already completed; answer with the stored response."""

    status: int
    body: dict[str, Any]


def validate_idempotency_key(raw: str | None) -> str | None:
    """The key to use, or ``None`` when the header is absent. Raises
    ``ValueError`` for a present but malformed value."""
    if raw is None:
        return None
    value = raw.strip()
    if _KEY_PATTERN.fullmatch(value) is None:
        raise ValueError("Idempotency-Key must be 1-255 printable ASCII characters")
    return value


def request_fingerprint(payload: dict[str, Any]) -> str:
    """Digest of the canonical request body; the body itself is never stored."""
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _aware(value: datetime) -> datetime:
    return normalize_datetime_from_db(value)


def _find(
    db: Session, scope_kind: str, scope_id: int, operation: str, key: str
) -> ApiIdempotencyKey | None:
    return db.execute(
        select(ApiIdempotencyKey).where(
            ApiIdempotencyKey.scope_kind == scope_kind,
            ApiIdempotencyKey.scope_id == scope_id,
            ApiIdempotencyKey.operation == operation,
            ApiIdempotencyKey.idempotency_key == key,
        )
    ).scalar_one_or_none()


def _purge_expired(db: Session, now: datetime) -> None:
    ids = (
        select(ApiIdempotencyKey.id)
        .where(ApiIdempotencyKey.expires_at <= now)
        .limit(_PURGE_BATCH)
    )
    db.execute(delete(ApiIdempotencyKey).where(ApiIdempotencyKey.id.in_(ids)))


def acquire_sync(
    *,
    scope_kind: str,
    scope_id: int,
    operation: str,
    key: str,
    fingerprint: str,
    now: datetime | None = None,
) -> IdempotencyClaim | IdempotencyReplay:
    """Reserve ``key``, or say what the earlier request with it left behind."""
    moment = now or datetime.now(timezone.utc)
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        _purge_expired(db, moment)
        db.commit()
        for _ in range(3):
            row = _find(db, scope_kind, scope_id, operation, key)
            if row is None:
                row = ApiIdempotencyKey(
                    scope_kind=scope_kind,
                    scope_id=scope_id,
                    operation=operation,
                    idempotency_key=key,
                    request_fingerprint=fingerprint,
                    state=IDEMPOTENCY_STATE_IN_PROGRESS,
                    lease_expires_at=moment + IDEMPOTENCY_LEASE,
                    expires_at=moment + IDEMPOTENCY_KEY_TTL,
                )
                db.add(row)
                try:
                    db.commit()
                except IntegrityError:
                    # A concurrent request reserved it first; judge its row.
                    db.rollback()
                    continue
                return IdempotencyClaim(claim_id=int(row.id))
            if _aware(row.expires_at) <= moment:
                db.execute(
                    delete(ApiIdempotencyKey).where(ApiIdempotencyKey.id == row.id)
                )
                db.commit()
                continue
            return _judge_existing(db, row, fingerprint, moment)
        raise IdempotencyRejected("in_progress")


def _judge_existing(
    db: Session, row: ApiIdempotencyKey, fingerprint: str, moment: datetime
) -> IdempotencyClaim | IdempotencyReplay:
    if row.request_fingerprint != fingerprint:
        raise IdempotencyRejected("reused")
    if row.state == IDEMPOTENCY_STATE_COMPLETED:
        if row.task_id is None or row.response_body is None:
            raise IdempotencyRejected("conflict")
        return IdempotencyReplay(
            status=int(row.response_status or 202), body=dict(row.response_body)
        )
    old_lease = row.lease_expires_at
    if _aware(old_lease) > moment:
        raise IdempotencyRejected("in_progress")
    if row.task_id is not None:
        # The task was accepted but the process died before the response was
        # recorded. Starting another run would duplicate it.
        raise IdempotencyRejected("outcome_unknown", task_id=int(row.task_id))
    # The reserving request died before creating anything: take it over. The
    # compare-and-set keeps two retries from both taking it.
    taken = db.execute(
        update(ApiIdempotencyKey)
        .where(
            ApiIdempotencyKey.id == row.id,
            ApiIdempotencyKey.lease_expires_at == old_lease,
            ApiIdempotencyKey.task_id.is_(None),
        )
        .values(lease_expires_at=moment + IDEMPOTENCY_LEASE)
    )
    db.commit()
    if taken.rowcount != 1:
        raise IdempotencyRejected("in_progress")
    return IdempotencyClaim(claim_id=int(row.id))


def bind_task_no_commit(db: Session, claim_id: int, task_id: int) -> None:
    """Bind the reservation to ``task_id`` in the caller's open transaction.

    Called from the transaction that creates the task (or claims the append
    turn), so the binding commits with it or not at all.
    """
    db.execute(
        update(ApiIdempotencyKey)
        .where(ApiIdempotencyKey.id == claim_id)
        .values(task_id=task_id)
    )


def complete_sync(claim_id: int, *, status: int, body: dict[str, Any]) -> None:
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        db.execute(
            update(ApiIdempotencyKey)
            .where(ApiIdempotencyKey.id == claim_id)
            .values(
                state=IDEMPOTENCY_STATE_COMPLETED,
                response_status=status,
                response_body=body,
            )
        )
        db.commit()


def release_sync(claim_id: int) -> None:
    """Drop a reservation whose call failed before creating anything, so the
    client's retry runs the call afresh. A reservation already bound to a task
    is kept: that task exists."""
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        db.execute(
            delete(ApiIdempotencyKey).where(
                ApiIdempotencyKey.id == claim_id,
                ApiIdempotencyKey.state == IDEMPOTENCY_STATE_IN_PROGRESS,
                ApiIdempotencyKey.task_id.is_(None),
            )
        )
        db.commit()
