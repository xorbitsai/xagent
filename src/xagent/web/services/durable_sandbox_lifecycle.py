"""Dormant durable lifecycle protocol for execution-scoped sandboxes.

All state transitions are synchronous database operations.  A future consumer
must commit ``begin_create`` before backend create, and commit a delete claim
before probing or deleting an exact backend generation.  The coordinator in
this module deliberately ends each transaction before backend I/O and opens a
new fenced transaction for observe, settle, or backoff.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Literal, Protocol

from sqlalchemy import and_, delete, exists, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from ...sandbox.base import ExactGenerationProbe
from ..models.sandbox import DurableSandboxLifecycle
from ..models.task import Task, TaskStatus

LifecycleState = Literal["registered", "ready", "deleting"]
CreatePhase = Literal["not_started", "may_publish", "terminal", "observed"]
CreateCompletion = Literal["success", "terminal_absent"]

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_BACKEND_LIFECYCLE_DOMAIN = b"xagent.durable-sandbox.backend-lifecycle.v1\x00"
_TURN_DOMAIN = b"xagent.durable-sandbox.turn.v1\x00"

logger = logging.getLogger(__name__)


class DurableLifecycleConflict(RuntimeError):
    """A lifecycle registration or CAS transition lost its fence."""


@dataclass(frozen=True)
class LifecycleFence:
    id: int
    scope_digest: str
    active_scope_digest: str | None
    lifecycle_token: str
    backend_lifecycle_digest: str
    owner_token: str
    create_operation_token: str
    create_phase: CreatePhase
    create_terminal_outcome: CreateCompletion | None
    version: int
    state: LifecycleState
    task_id: int
    run_id: str
    lease_attempt_id: str
    turn_digest: str | None
    eligible_at: datetime
    owner_lease_expires_at: datetime
    delete_claim_expires_at: datetime | None
    retry_at: datetime | None
    delete_attempts: int

    @classmethod
    def from_row(cls, row: Any) -> "LifecycleFence":
        return cls(
            id=row.id,
            scope_digest=row.scope_digest,
            active_scope_digest=row.active_scope_digest,
            lifecycle_token=row.lifecycle_token,
            backend_lifecycle_digest=row.backend_lifecycle_digest,
            owner_token=row.owner_token,
            create_operation_token=row.create_operation_token,
            create_phase=row.create_phase,
            create_terminal_outcome=row.create_terminal_outcome,
            version=row.version,
            state=row.state,
            task_id=row.task_id,
            run_id=row.run_id,
            lease_attempt_id=row.lease_attempt_id,
            turn_digest=row.turn_digest,
            eligible_at=row.eligible_at,
            owner_lease_expires_at=row.owner_lease_expires_at,
            delete_claim_expires_at=row.delete_claim_expires_at,
            retry_at=row.retry_at,
            delete_attempts=row.delete_attempts,
        )


@dataclass(frozen=True)
class RegisterLifecycle:
    scope_digest: str
    task_id: int
    run_id: str
    lease_attempt_id: str
    turn_digest: str | None
    eligible_at: datetime
    owner_lease_expires_at: datetime


def digest_turn_identifier(turn_id: str) -> str:
    """Domain-separated representation safe to persist for a raw turn ID."""
    if not turn_id:
        raise ValueError("turn_id must be non-empty")
    return hashlib.sha256(_TURN_DOMAIN + turn_id.encode("utf-8")).hexdigest()


def backend_lifecycle_digest(scope_digest: str, lifecycle_token: str) -> str:
    """Bind a physical backend name to one immutable lifecycle generation."""
    _validate_digest("scope_digest", scope_digest)
    _validate_digest("lifecycle_token", lifecycle_token)
    return hashlib.sha256(
        _BACKEND_LIFECYCLE_DOMAIN
        + bytes.fromhex(scope_digest)
        + bytes.fromhex(lifecycle_token)
    ).hexdigest()


def new_opaque_token() -> str:
    return secrets.token_hex(32)


def _validate_digest(name: str, value: str | None, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _validate_registration(request: RegisterLifecycle) -> None:
    _validate_digest("scope_digest", request.scope_digest)
    _validate_digest("turn_digest", request.turn_digest, optional=True)
    if request.task_id <= 0:
        raise ValueError("task_id must be positive")
    if not request.run_id or len(request.run_id) > 64:
        raise ValueError("run_id must be 1..64 characters")
    if not request.lease_attempt_id or len(request.lease_attempt_id) > 64:
        raise ValueError("lease_attempt_id must be 1..64 characters")


def _tombstone_values(*, now: datetime, claim_ttl: timedelta) -> dict[str, Any]:
    if claim_ttl <= timedelta(0):
        raise ValueError("claim_ttl must be positive")
    return {
        "state": "deleting",
        "active_scope_digest": None,
        "owner_token": new_opaque_token(),
        "version": DurableSandboxLifecycle.version + 1,
        "deleting_at": now,
        "delete_claim_expires_at": now + claim_ttl,
        "retry_at": None,
        "delete_attempts": DurableSandboxLifecycle.delete_attempts + 1,
        "updated_at": now,
    }


def _exact_attempt_is_active(row: Any, now: datetime) -> ColumnElement[bool]:
    return exists(
        select(Task.id).where(
            Task.id == row.task_id,
            Task.run_id == row.run_id,
            Task.lease_attempt_id == row.lease_attempt_id,
            Task.runner_id.is_not(None),
            Task.status == TaskStatus.RUNNING,
            Task.lease_expires_at.is_not(None),
            Task.lease_expires_at >= now,
        )
    )


class DurableSandboxLifecycleRepository:
    """Conditional-DML repository; callers own commit/rollback boundaries."""

    def __init__(self, db: Session):
        self._db = db

    def register(self, request: RegisterLifecycle) -> LifecycleFence:
        """Persist a generation before any backend create is attempted."""
        _validate_registration(request)
        lifecycle_token = new_opaque_token()
        row = DurableSandboxLifecycle(
            scope_digest=request.scope_digest,
            active_scope_digest=request.scope_digest,
            lifecycle_token=lifecycle_token,
            backend_lifecycle_digest=backend_lifecycle_digest(
                request.scope_digest, lifecycle_token
            ),
            owner_token=new_opaque_token(),
            create_operation_token=new_opaque_token(),
            create_phase="not_started",
            create_terminal_outcome=None,
            version=1,
            state="registered",
            task_id=request.task_id,
            run_id=request.run_id,
            lease_attempt_id=request.lease_attempt_id,
            turn_digest=request.turn_digest,
            eligible_at=request.eligible_at,
            owner_lease_expires_at=request.owner_lease_expires_at,
            delete_attempts=0,
        )
        self._db.add(row)
        try:
            self._db.flush()
        except IntegrityError as exc:
            raise DurableLifecycleConflict(
                "lifecycle scope already has an active generation"
            ) from exc
        return LifecycleFence.from_row(row)

    def get_by_scope(self, scope_digest: str) -> LifecycleFence | None:
        """Return the one active generation for a logical scope, if any."""
        _validate_digest("scope_digest", scope_digest)
        row = self._db.execute(
            select(DurableSandboxLifecycle).where(
                DurableSandboxLifecycle.active_scope_digest == scope_digest
            )
        ).scalar_one_or_none()
        return None if row is None else LifecycleFence.from_row(row)

    def list_by_scope(self, scope_digest: str) -> list[LifecycleFence]:
        """Return active and quarantined generations without adopting either."""
        _validate_digest("scope_digest", scope_digest)
        rows = self._db.execute(
            select(DurableSandboxLifecycle)
            .where(DurableSandboxLifecycle.scope_digest == scope_digest)
            .order_by(DurableSandboxLifecycle.id)
        ).scalars()
        return [LifecycleFence.from_row(row) for row in rows]

    def begin_create(
        self, fence: LifecycleFence, *, now: datetime
    ) -> LifecycleFence | None:
        """Commit the may-publish fence before invoking a backend create."""
        stmt = (
            update(DurableSandboxLifecycle)
            .where(
                DurableSandboxLifecycle.id == fence.id,
                DurableSandboxLifecycle.lifecycle_token == fence.lifecycle_token,
                DurableSandboxLifecycle.owner_token == fence.owner_token,
                DurableSandboxLifecycle.create_operation_token
                == fence.create_operation_token,
                DurableSandboxLifecycle.version == fence.version,
                DurableSandboxLifecycle.state == "registered",
                DurableSandboxLifecycle.create_phase == "not_started",
                _exact_attempt_is_active(DurableSandboxLifecycle, now),
            )
            .values(
                create_phase="may_publish",
                updated_at=now,
                version=DurableSandboxLifecycle.version + 1,
            )
            .returning(DurableSandboxLifecycle)
        )
        row = self._db.execute(stmt).scalar_one_or_none()
        return None if row is None else LifecycleFence.from_row(row)

    def complete_create(
        self,
        fence: LifecycleFence,
        *,
        now: datetime,
        outcome: CreateCompletion,
    ) -> LifecycleFence | None:
        """Record a proven create completion or terminal-absent outcome.

        Callers must not invoke this for timeouts, disconnects, cancellation,
        or any other outcome that leaves backend publication ambiguous.
        """
        if outcome not in ("success", "terminal_absent"):
            raise ValueError("unsupported create completion outcome")
        stmt = (
            update(DurableSandboxLifecycle)
            .where(
                DurableSandboxLifecycle.id == fence.id,
                DurableSandboxLifecycle.lifecycle_token == fence.lifecycle_token,
                DurableSandboxLifecycle.owner_token == fence.owner_token,
                DurableSandboxLifecycle.create_operation_token
                == fence.create_operation_token,
                DurableSandboxLifecycle.version == fence.version,
                DurableSandboxLifecycle.state == "registered",
                DurableSandboxLifecycle.create_phase == "may_publish",
            )
            .values(
                create_phase="terminal",
                create_terminal_outcome=outcome,
                updated_at=now,
                version=DurableSandboxLifecycle.version + 1,
            )
            .returning(DurableSandboxLifecycle)
        )
        row = self._db.execute(stmt).scalar_one_or_none()
        return None if row is None else LifecycleFence.from_row(row)

    def mark_ready(
        self, fence: LifecycleFence, *, now: datetime, owner_lease_expires_at: datetime
    ) -> LifecycleFence | None:
        stmt = (
            update(DurableSandboxLifecycle)
            .where(
                DurableSandboxLifecycle.id == fence.id,
                DurableSandboxLifecycle.lifecycle_token == fence.lifecycle_token,
                DurableSandboxLifecycle.owner_token == fence.owner_token,
                DurableSandboxLifecycle.create_operation_token
                == fence.create_operation_token,
                DurableSandboxLifecycle.version == fence.version,
                DurableSandboxLifecycle.state == "registered",
                DurableSandboxLifecycle.create_phase == "terminal",
                DurableSandboxLifecycle.create_terminal_outcome == "success",
                _exact_attempt_is_active(DurableSandboxLifecycle, now),
            )
            .values(
                state="ready",
                ready_at=now,
                owner_lease_expires_at=owner_lease_expires_at,
                updated_at=now,
                version=DurableSandboxLifecycle.version + 1,
            )
            .returning(DurableSandboxLifecycle)
        )
        row = self._db.execute(stmt).scalar_one_or_none()
        return None if row is None else LifecycleFence.from_row(row)

    def renew(
        self, fence: LifecycleFence, *, now: datetime, owner_lease_expires_at: datetime
    ) -> LifecycleFence | None:
        stmt = (
            update(DurableSandboxLifecycle)
            .where(
                DurableSandboxLifecycle.id == fence.id,
                DurableSandboxLifecycle.lifecycle_token == fence.lifecycle_token,
                DurableSandboxLifecycle.owner_token == fence.owner_token,
                DurableSandboxLifecycle.create_operation_token
                == fence.create_operation_token,
                DurableSandboxLifecycle.version == fence.version,
                DurableSandboxLifecycle.state.in_(("registered", "ready")),
                _exact_attempt_is_active(DurableSandboxLifecycle, now),
            )
            .values(
                owner_lease_expires_at=owner_lease_expires_at,
                updated_at=now,
                version=DurableSandboxLifecycle.version + 1,
            )
            .returning(DurableSandboxLifecycle)
        )
        row = self._db.execute(stmt).scalar_one_or_none()
        return None if row is None else LifecycleFence.from_row(row)

    def claim_for_delete(
        self,
        fence: LifecycleFence,
        *,
        now: datetime,
        claim_ttl: timedelta,
    ) -> LifecycleFence | None:
        """Classify and tombstone an eligible registered/ready row atomically."""
        stmt = (
            update(DurableSandboxLifecycle)
            .where(
                DurableSandboxLifecycle.id == fence.id,
                DurableSandboxLifecycle.lifecycle_token == fence.lifecycle_token,
                DurableSandboxLifecycle.owner_token == fence.owner_token,
                DurableSandboxLifecycle.create_operation_token
                == fence.create_operation_token,
                DurableSandboxLifecycle.version == fence.version,
                DurableSandboxLifecycle.state.in_(("registered", "ready")),
                DurableSandboxLifecycle.eligible_at <= now,
                DurableSandboxLifecycle.owner_lease_expires_at <= now,
                ~_exact_attempt_is_active(DurableSandboxLifecycle, now),
            )
            .values(**_tombstone_values(now=now, claim_ttl=claim_ttl))
            .returning(DurableSandboxLifecycle)
        )
        row = self._db.execute(stmt).scalar_one_or_none()
        return None if row is None else LifecycleFence.from_row(row)

    def mark_deleting_owned(
        self,
        fence: LifecycleFence,
        *,
        now: datetime,
        claim_ttl: timedelta,
    ) -> LifecycleFence | None:
        """Tombstone a non-ambiguous generation while its owner is active.

        This is the normal close/attach-failure path.  ``may_publish`` is
        deliberately excluded because an ambiguous create must instead use
        :meth:`quarantine_create` and retain its probe-before-delete policy.
        """
        stmt = (
            update(DurableSandboxLifecycle)
            .where(
                DurableSandboxLifecycle.id == fence.id,
                DurableSandboxLifecycle.lifecycle_token == fence.lifecycle_token,
                DurableSandboxLifecycle.backend_lifecycle_digest
                == fence.backend_lifecycle_digest,
                DurableSandboxLifecycle.owner_token == fence.owner_token,
                DurableSandboxLifecycle.create_operation_token
                == fence.create_operation_token,
                DurableSandboxLifecycle.version == fence.version,
                DurableSandboxLifecycle.state.in_(("registered", "ready")),
                DurableSandboxLifecycle.create_phase.in_(
                    ("not_started", "terminal", "observed")
                ),
            )
            .values(**_tombstone_values(now=now, claim_ttl=claim_ttl))
            .returning(DurableSandboxLifecycle)
        )
        row = self._db.execute(stmt).scalar_one_or_none()
        return None if row is None else LifecycleFence.from_row(row)

    def quarantine_create(
        self,
        fence: LifecycleFence,
        *,
        now: datetime,
        claim_ttl: timedelta,
    ) -> LifecycleFence | None:
        """Owner-fenced handoff of an ambiguous create to durable cleanup.

        Unlike stale-attempt reclamation, this transition intentionally does
        not require the task attempt to be inactive.  The current owner uses
        it after a bounded timeout/unknown outcome so the old generation keeps
        its durable backend digest while releasing the logical active slot for
        an immediate successor.
        """
        stmt = (
            update(DurableSandboxLifecycle)
            .where(
                DurableSandboxLifecycle.id == fence.id,
                DurableSandboxLifecycle.lifecycle_token == fence.lifecycle_token,
                DurableSandboxLifecycle.owner_token == fence.owner_token,
                DurableSandboxLifecycle.create_operation_token
                == fence.create_operation_token,
                DurableSandboxLifecycle.version == fence.version,
                DurableSandboxLifecycle.state == "registered",
                DurableSandboxLifecycle.create_phase == "may_publish",
            )
            .values(**_tombstone_values(now=now, claim_ttl=claim_ttl))
            .returning(DurableSandboxLifecycle)
        )
        row = self._db.execute(stmt).scalar_one_or_none()
        return None if row is None else LifecycleFence.from_row(row)

    def complete_quarantined_create(
        self,
        fence: LifecycleFence,
        *,
        now: datetime,
        outcome: CreateCompletion,
    ) -> LifecycleFence | None:
        """Record a late proven completion after owner handoff to cleanup.

        The create worker's owner/version became stale at quarantine time, so
        this CAS uses only immutable generation and operation identity.  Its
        version increment invalidates any delete fence read by a concurrent
        sweeper and returns the current delete owner fence to the caller.
        """
        if outcome not in ("success", "terminal_absent"):
            raise ValueError("unsupported create completion outcome")
        stmt = (
            update(DurableSandboxLifecycle)
            .where(
                DurableSandboxLifecycle.id == fence.id,
                DurableSandboxLifecycle.lifecycle_token == fence.lifecycle_token,
                DurableSandboxLifecycle.backend_lifecycle_digest
                == fence.backend_lifecycle_digest,
                DurableSandboxLifecycle.create_operation_token
                == fence.create_operation_token,
                DurableSandboxLifecycle.state == "deleting",
                DurableSandboxLifecycle.create_phase == "may_publish",
            )
            .values(
                create_phase="terminal",
                create_terminal_outcome=outcome,
                version=DurableSandboxLifecycle.version + 1,
                updated_at=now,
            )
            .returning(DurableSandboxLifecycle)
        )
        row = self._db.execute(stmt).scalar_one_or_none()
        return None if row is None else LifecycleFence.from_row(row)

    def reclaim_delete(
        self,
        fence: LifecycleFence,
        *,
        now: datetime,
        claim_ttl: timedelta,
    ) -> LifecycleFence | None:
        """Take over an expired/retryable tombstone with a new owner fence."""
        if claim_ttl <= timedelta(0):
            raise ValueError("claim_ttl must be positive")
        claim_owner = new_opaque_token()
        stmt = (
            update(DurableSandboxLifecycle)
            .where(
                DurableSandboxLifecycle.id == fence.id,
                DurableSandboxLifecycle.lifecycle_token == fence.lifecycle_token,
                DurableSandboxLifecycle.owner_token == fence.owner_token,
                DurableSandboxLifecycle.create_operation_token
                == fence.create_operation_token,
                DurableSandboxLifecycle.version == fence.version,
                DurableSandboxLifecycle.state == "deleting",
                DurableSandboxLifecycle.delete_claim_expires_at <= now,
                or_(
                    DurableSandboxLifecycle.retry_at.is_(None),
                    DurableSandboxLifecycle.retry_at <= now,
                ),
            )
            .values(
                owner_token=claim_owner,
                version=DurableSandboxLifecycle.version + 1,
                delete_claim_expires_at=now + claim_ttl,
                retry_at=None,
                delete_attempts=DurableSandboxLifecycle.delete_attempts + 1,
                updated_at=now,
            )
            .returning(DurableSandboxLifecycle)
        )
        row = self._db.execute(stmt).scalar_one_or_none()
        return None if row is None else LifecycleFence.from_row(row)

    def backoff_delete(
        self, fence: LifecycleFence, *, now: datetime, retry_at: datetime
    ) -> LifecycleFence | None:
        if retry_at <= now:
            raise ValueError("retry_at must be in the future")
        stmt = (
            update(DurableSandboxLifecycle)
            .where(*self._delete_fence(fence))
            .values(
                version=DurableSandboxLifecycle.version + 1,
                delete_claim_expires_at=now,
                retry_at=retry_at,
                updated_at=now,
            )
            .returning(DurableSandboxLifecycle)
        )
        row = self._db.execute(stmt).scalar_one_or_none()
        return None if row is None else LifecycleFence.from_row(row)

    def settle_delete(self, fence: LifecycleFence) -> bool:
        result: Any = self._db.execute(
            delete(DurableSandboxLifecycle).where(
                *self._delete_fence(fence),
                DurableSandboxLifecycle.create_phase.in_(
                    ("not_started", "terminal", "observed")
                ),
            )
        )
        return int(result.rowcount or 0) == 1

    def observe_create(
        self, fence: LifecycleFence, *, now: datetime
    ) -> LifecycleFence | None:
        """Persist PRESENT before any strict deletion of an ambiguous create."""
        stmt = (
            update(DurableSandboxLifecycle)
            .where(
                *self._delete_fence(fence),
                DurableSandboxLifecycle.create_operation_token
                == fence.create_operation_token,
                DurableSandboxLifecycle.create_phase == "may_publish",
            )
            .values(
                create_phase="observed",
                version=DurableSandboxLifecycle.version + 1,
                updated_at=now,
            )
            .returning(DurableSandboxLifecycle)
        )
        row = self._db.execute(stmt).scalar_one_or_none()
        return None if row is None else LifecycleFence.from_row(row)

    @staticmethod
    def _delete_fence(
        fence: LifecycleFence,
    ) -> tuple[ColumnElement[bool], ...]:
        return (
            DurableSandboxLifecycle.id == fence.id,
            DurableSandboxLifecycle.lifecycle_token == fence.lifecycle_token,
            DurableSandboxLifecycle.backend_lifecycle_digest
            == fence.backend_lifecycle_digest,
            DurableSandboxLifecycle.owner_token == fence.owner_token,
            DurableSandboxLifecycle.create_operation_token
            == fence.create_operation_token,
            DurableSandboxLifecycle.version == fence.version,
            DurableSandboxLifecycle.state == "deleting",
        )

    def list_reclaimable(
        self, *, now: datetime, limit: int = 100
    ) -> list[LifecycleFence]:
        """Return hints only; every destructive decision is rechecked by DML."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = self._db.execute(
            select(DurableSandboxLifecycle)
            .where(
                or_(
                    and_(
                        DurableSandboxLifecycle.state.in_(("registered", "ready")),
                        DurableSandboxLifecycle.eligible_at <= now,
                        DurableSandboxLifecycle.owner_lease_expires_at <= now,
                    ),
                    and_(
                        DurableSandboxLifecycle.state == "deleting",
                        DurableSandboxLifecycle.delete_claim_expires_at <= now,
                        or_(
                            DurableSandboxLifecycle.retry_at.is_(None),
                            DurableSandboxLifecycle.retry_at <= now,
                        ),
                    ),
                )
            )
            .order_by(
                func.coalesce(
                    DurableSandboxLifecycle.retry_at,
                    DurableSandboxLifecycle.eligible_at,
                ),
                DurableSandboxLifecycle.id,
            )
            .limit(limit)
        ).scalars()
        return [LifecycleFence.from_row(row) for row in rows]


class DurableSandboxLifecycleService:
    """Commit-bounded facade for PR2 consumers and sweep orchestration.

    Every method completes its transaction before returning.  Deliberately
    there is no async/backend callback API here: callers must never keep one
    of these transactions open across sandbox I/O.
    """

    def __init__(self, session_factory: Callable[[], Session]):
        self._session_factory = session_factory

    def register(self, request: RegisterLifecycle) -> LifecycleFence:
        with self._session_factory() as db, db.begin():
            return DurableSandboxLifecycleRepository(db).register(request)

    def get_by_scope(self, scope_digest: str) -> LifecycleFence | None:
        with self._session_factory() as db, db.begin():
            return DurableSandboxLifecycleRepository(db).get_by_scope(scope_digest)

    def list_by_scope(self, scope_digest: str) -> list[LifecycleFence]:
        with self._session_factory() as db, db.begin():
            return DurableSandboxLifecycleRepository(db).list_by_scope(scope_digest)

    def begin_create(
        self, fence: LifecycleFence, *, now: datetime
    ) -> LifecycleFence | None:
        with self._session_factory() as db, db.begin():
            return DurableSandboxLifecycleRepository(db).begin_create(fence, now=now)

    def complete_create(
        self,
        fence: LifecycleFence,
        *,
        now: datetime,
        outcome: CreateCompletion,
    ) -> LifecycleFence | None:
        with self._session_factory() as db, db.begin():
            return DurableSandboxLifecycleRepository(db).complete_create(
                fence, now=now, outcome=outcome
            )

    def mark_ready(
        self, fence: LifecycleFence, *, now: datetime, owner_lease_expires_at: datetime
    ) -> LifecycleFence | None:
        with self._session_factory() as db, db.begin():
            return DurableSandboxLifecycleRepository(db).mark_ready(
                fence, now=now, owner_lease_expires_at=owner_lease_expires_at
            )

    def renew(
        self, fence: LifecycleFence, *, now: datetime, owner_lease_expires_at: datetime
    ) -> LifecycleFence | None:
        with self._session_factory() as db, db.begin():
            return DurableSandboxLifecycleRepository(db).renew(
                fence, now=now, owner_lease_expires_at=owner_lease_expires_at
            )

    def claim_for_delete(
        self, fence: LifecycleFence, *, now: datetime, claim_ttl: timedelta
    ) -> LifecycleFence | None:
        with self._session_factory() as db, db.begin():
            return DurableSandboxLifecycleRepository(db).claim_for_delete(
                fence, now=now, claim_ttl=claim_ttl
            )

    def mark_deleting_owned(
        self, fence: LifecycleFence, *, now: datetime, claim_ttl: timedelta
    ) -> LifecycleFence | None:
        with self._session_factory() as db, db.begin():
            return DurableSandboxLifecycleRepository(db).mark_deleting_owned(
                fence, now=now, claim_ttl=claim_ttl
            )

    def quarantine_create(
        self, fence: LifecycleFence, *, now: datetime, claim_ttl: timedelta
    ) -> LifecycleFence | None:
        with self._session_factory() as db, db.begin():
            return DurableSandboxLifecycleRepository(db).quarantine_create(
                fence, now=now, claim_ttl=claim_ttl
            )

    def complete_quarantined_create(
        self,
        fence: LifecycleFence,
        *,
        now: datetime,
        outcome: CreateCompletion,
    ) -> LifecycleFence | None:
        with self._session_factory() as db, db.begin():
            return DurableSandboxLifecycleRepository(db).complete_quarantined_create(
                fence, now=now, outcome=outcome
            )

    def reclaim_delete(
        self, fence: LifecycleFence, *, now: datetime, claim_ttl: timedelta
    ) -> LifecycleFence | None:
        with self._session_factory() as db, db.begin():
            return DurableSandboxLifecycleRepository(db).reclaim_delete(
                fence, now=now, claim_ttl=claim_ttl
            )

    def backoff_delete(
        self, fence: LifecycleFence, *, now: datetime, retry_at: datetime
    ) -> LifecycleFence | None:
        with self._session_factory() as db, db.begin():
            return DurableSandboxLifecycleRepository(db).backoff_delete(
                fence, now=now, retry_at=retry_at
            )

    def settle_delete(self, fence: LifecycleFence) -> bool:
        with self._session_factory() as db, db.begin():
            return DurableSandboxLifecycleRepository(db).settle_delete(fence)

    def observe_create(
        self, fence: LifecycleFence, *, now: datetime
    ) -> LifecycleFence | None:
        with self._session_factory() as db, db.begin():
            return DurableSandboxLifecycleRepository(db).observe_create(fence, now=now)

    def list_reclaimable(
        self, *, now: datetime, limit: int = 100
    ) -> list[LifecycleFence]:
        with self._session_factory() as db, db.begin():
            return DurableSandboxLifecycleRepository(db).list_reclaimable(
                now=now, limit=limit
            )


class DurableSandboxBackend(Protocol):
    """Backend seam used by the dormant create-operation reclaimer."""

    async def probe_durable_sandbox_strict(
        self, lifecycle_id: str
    ) -> ExactGenerationProbe: ...

    async def delete_durable_sandbox_strict(self, lifecycle_id: str) -> None: ...


class DurableSandboxDeleteCoordinator:
    """No-transaction-across-I/O coordinator for one claimed tombstone."""

    def __init__(
        self,
        lifecycles: DurableSandboxLifecycleService,
        backend: DurableSandboxBackend,
    ) -> None:
        self._lifecycles = lifecycles
        self._backend = backend

    async def delete_claimed(
        self,
        fence: LifecycleFence,
        *,
        now: datetime,
        retry_at: datetime,
    ) -> bool:
        """Try one cleanup; return True only after durable settlement.

        ``may_publish`` is the ambiguity quarantine.  ABSENT and UNKNOWN are
        intentionally indistinguishable for safety and only schedule another
        probe.  PRESENT is committed as ``observed`` before deletion so a
        crash can never forget that absence has become actionable.
        """
        if fence.state != "deleting":
            raise ValueError("delete_claimed requires a deleting fence")
        if retry_at <= now:
            raise ValueError("retry_at must be in the future")

        current = fence
        if current.create_phase == "may_publish":
            try:
                presence = await self._backend.probe_durable_sandbox_strict(
                    current.backend_lifecycle_digest
                )
            except Exception:
                logger.warning(
                    "Durable sandbox generation probe failed closed for %s",
                    current.backend_lifecycle_digest,
                    exc_info=True,
                )
                presence = ExactGenerationProbe.UNKNOWN
            if presence is not ExactGenerationProbe.PRESENT:
                await asyncio.to_thread(
                    self._lifecycles.backoff_delete,
                    current,
                    now=now,
                    retry_at=retry_at,
                )
                return False
            observed = await asyncio.to_thread(
                self._lifecycles.observe_create, current, now=now
            )
            if observed is None:
                return False
            current = observed

        try:
            await self._backend.delete_durable_sandbox_strict(
                current.backend_lifecycle_digest
            )
        except Exception:
            logger.warning(
                "Durable sandbox generation deletion failed for %s",
                current.backend_lifecycle_digest,
                exc_info=True,
            )
            await asyncio.to_thread(
                self._lifecycles.backoff_delete,
                current,
                now=now,
                retry_at=retry_at,
            )
            return False
        return await asyncio.to_thread(self._lifecycles.settle_delete, current)
