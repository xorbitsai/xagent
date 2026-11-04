"""Garbage collection of orphaned task-less public uploads (#973).

A task-less public-share upload (workforce first-turn attachment) is created
BEFORE its run/task exists, then bound to the task at run start. If the guest
never completes task creation, the row + stored bytes are never bound and
never cleaned up. This reaps those orphans.

The predicate is deliberately narrow. ``task_id IS NULL`` is a system-wide
normal intermediate state (plain ``/api/files/upload`` allows an optional
task id, and turn handling binds unbound rows across every channel), so a
coarse "NULL + aged" sweep would delete logged-in users' un-sent draft
attachments. The ``upload_source`` marker (stamped only on the task-less
public-share path) scopes GC to exactly those uploads.

Cleanup commits an exact compensation claim and resource manifest before any
unlink or object deletion. The common phase runner retains that row until the
durable object, disposable managed copies, and owned previews are all gone.
Interrupted work is resumed by the existing compensation recovery loop.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, cast

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from ..models.uploaded_file import UploadedFile
from .db_runtime import is_database_pool_timeout, run_db_io_cancellation_safe
from .kb_reference_protection import guard_cleanup_claim
from .uploaded_file_cleanup import run_uploaded_file_cleanup
from .uploaded_file_cleanup_resources import build_cleanup_manifest
from .uploaded_file_store import (
    _load_uploaded_file_compensation_token_no_commit,
    _session_factory_from_reference,
    _uploaded_file_version_match_predicates,
    delete_uploaded_file_compensation_object,
    snapshot_uploaded_file_version,
)

logger = logging.getLogger(__name__)

# Provenance marker stamped on task-less public-share uploads. Orphan GC keys
# off it so the sweep only ever touches uploads created before any task
# binding on the public share path — never any other path's unbound draft.
TASKLESS_SHARE_UPLOAD_SOURCE = "taskless_share_upload"

# Bounded sweep shape: deterministic keyset pages, each in its own short
# Session, so a large backlog never materializes wholesale into worker
# memory. A tick drains up to GC_MAX_PAGES_PER_TICK full pages and then
# yields; the long poll sleep is taken ONLY once a short page proves the
# eligible backlog is drained. That decoupling is what keeps GC throughput
# independent of the poll interval: the rate-limited task-less upload
# surfaces admit far more rows per hour than any single fixed-interval
# page budget could reclaim, so a live cursor must be followed promptly
# rather than parked until the next tick. The cursor also carries ACROSS
# ticks (mirroring ``run_uploaded_file_compensation_recovery_loop``), so a
# repeatedly failing oldest page cannot starve newer orphans.
GC_BATCH_SIZE = 500
GC_MAX_PAGES_PER_TICK = 20
# Breather between page budgets while a backlog remains: bounds sustained
# database/storage-delete load to at most GC_MAX_PAGES_PER_TICK *
# GC_BATCH_SIZE rows (10,000 at the defaults) per this interval, and keeps
# the loop responsive to cancellation — without letting the long poll
# interval throttle drain throughput.
GC_BACKLOG_CONTINUE_DELAY_SECONDS = 1.0

OrphanUploadSweepCursor = tuple[datetime, int]


@dataclass(frozen=True)
class OrphanUploadSweepResult:
    """Outcome of one bounded orphan-GC page."""

    scanned: int = 0
    deleted: int = 0
    next_cursor: OrphanUploadSweepCursor | None = None


@dataclass(frozen=True)
class _OrphanUploadCandidate:
    """Detached exact identity of one reap-eligible row.

    Captured before any mutation so the storage claim and settlement fence on
    the same version the scan saw, and so the local ``storage_path`` survives
    the metadata deletion.
    """

    row_id: int
    user_id: int
    file_id: str
    storage_key: str
    storage_path: str
    created_at: datetime

    @property
    def cursor(self) -> OrphanUploadSweepCursor:
        return self.created_at, self.row_id


def _orphan_candidates(
    db: Session,
    *,
    cutoff: datetime,
    limit: int,
    after: OrphanUploadSweepCursor | None,
) -> tuple[_OrphanUploadCandidate, ...]:
    """One keyset page of reap-eligible rows, oldest first.

    Scoped to ``storage_status == 'available'`` with a durable key: that is
    the only state the registration path ever leaves a marked row in, it is
    the state the claim CAS requires, and it keeps rows another owner already
    claimed (``compensating`` — in-flight GC, request compensation, or stale-
    claim recovery) out of the scan entirely.
    """
    query = db.query(UploadedFile).filter(
        UploadedFile.upload_source == TASKLESS_SHARE_UPLOAD_SOURCE,
        UploadedFile.task_id.is_(None),
        UploadedFile.detached_reason.is_(None),
        UploadedFile.detached_at.is_(None),
        UploadedFile.created_at < cutoff,
        UploadedFile.storage_status == "available",
        UploadedFile.storage_key.isnot(None),
        UploadedFile.storage_key != "",
    )
    if after is not None:
        after_created_at, after_row_id = after
        query = query.filter(
            or_(
                UploadedFile.created_at > after_created_at,
                and_(
                    UploadedFile.created_at == after_created_at,
                    UploadedFile.id > after_row_id,
                ),
            )
        )
    records = (
        query.order_by(UploadedFile.created_at.asc(), UploadedFile.id.asc())
        .limit(limit)
        .all()
    )
    return tuple(
        _OrphanUploadCandidate(
            row_id=int(record.id),
            user_id=int(record.user_id),
            file_id=str(record.file_id),
            storage_key=str(record.storage_key),
            storage_path=str(record.storage_path),
            created_at=cast(datetime, record.created_at),
        )
        for record in records
    )


@guard_cleanup_claim
def _claim_orphan(db: Session, candidate: _OrphanUploadCandidate) -> datetime | None:
    """Commit only the exact available candidate, with its cleanup resources."""
    record = (
        db.query(UploadedFile)
        .filter(
            UploadedFile.id == candidate.row_id,
            UploadedFile.user_id == candidate.user_id,
            UploadedFile.file_id == candidate.file_id,
            UploadedFile.storage_key == candidate.storage_key,
            UploadedFile.storage_path == candidate.storage_path,
            UploadedFile.storage_status == "available",
        )
        .first()
    )
    if record is None:
        db.rollback()
        return None
    snapshot = snapshot_uploaded_file_version(record)
    db.rollback()
    manifest = build_cleanup_manifest(snapshot)
    claimed_at = datetime.now(timezone.utc)
    claimed = (
        db.query(UploadedFile)
        .filter(
            *_uploaded_file_version_match_predicates(snapshot),
            UploadedFile.created_at == candidate.created_at,
            UploadedFile.upload_source == TASKLESS_SHARE_UPLOAD_SOURCE,
            UploadedFile.task_id.is_(None),
            UploadedFile.detached_reason.is_(None),
            UploadedFile.detached_at.is_(None),
        )
        .update(
            {
                UploadedFile.storage_status: "compensating",
                UploadedFile.updated_at: claimed_at,
                UploadedFile.cleanup_manifest: manifest,
            },
            synchronize_session=False,
        )
    )
    if claimed != 1:
        db.rollback()
        return None
    # Read the token back as persisted: the database may round the datetime,
    # and the settlement fences on exact equality.
    token = _load_uploaded_file_compensation_token_no_commit(
        db, row_id=candidate.row_id
    )
    if token is None:
        db.rollback()
        return None
    return token


def _reap_orphan(db: Session, candidate: _OrphanUploadCandidate) -> bool:
    """Report completion only after all resources and exact settlement succeed."""
    token = _claim_orphan(db, candidate)
    if token is None:
        return False
    return (
        run_uploaded_file_cleanup(
            session_factory=_session_factory_from_reference(db),
            row_id=candidate.row_id,
            user_id=candidate.user_id,
            file_id=candidate.file_id,
            task_id=None,
            storage_key=candidate.storage_key,
            expected_updated_at=token,
            compensation_delete=delete_uploaded_file_compensation_object,
        )
        == "deleted"
    )


def cleanup_orphaned_taskless_uploads(
    db: Session,
    *,
    older_than_seconds: int,
    now: datetime | None = None,
    batch_size: int = GC_BATCH_SIZE,
    after: OrphanUploadSweepCursor | None = None,
) -> OrphanUploadSweepResult:
    """Reap one bounded page of task-less public uploads never bound to a task.

    Processes the oldest ``batch_size`` rows past ``after`` that (a) carry
    the task-less-share marker, (b) still have no ``task_id``, and (c) are
    older than the TTL. Per-row failures are logged and skipped so one bad
    row does not abort the page. The returned ``next_cursor`` always points
    past every processed row (spared and failing included); the GC loop
    threads it across ticks so a bad oldest page cannot starve newer orphans.
    """
    reference = now or datetime.now(timezone.utc)
    cutoff = reference - timedelta(seconds=older_than_seconds)
    candidates = _orphan_candidates(db, cutoff=cutoff, limit=batch_size, after=after)
    deleted = 0
    for candidate in candidates:
        try:
            if _reap_orphan(db, candidate):
                deleted += 1
        except Exception:
            db.rollback()
            logger.warning(
                "Failed to GC orphaned task-less upload id=%s",
                candidate.row_id,
                exc_info=True,
            )
    return OrphanUploadSweepResult(
        scanned=len(candidates),
        deleted=deleted,
        next_cursor=candidates[-1].cursor if candidates else None,
    )


def sweep_orphaned_taskless_uploads_isolated(
    *,
    older_than_seconds: int,
    batch_size: int = GC_BATCH_SIZE,
    after: OrphanUploadSweepCursor | None = None,
    session_factory: Callable[[], Session] | None = None,
) -> OrphanUploadSweepResult:
    """Run one GC page in its own short-lived Session."""
    if session_factory is None:
        from ..models.database import get_session_local

        session_factory = get_session_local()
    with session_factory() as db:
        return cleanup_orphaned_taskless_uploads(
            db,
            older_than_seconds=older_than_seconds,
            batch_size=batch_size,
            after=after,
        )


async def _run_gc_tick(
    *,
    ttl_seconds: int,
    batch_size: int,
    max_pages: int,
    after: OrphanUploadSweepCursor | None,
    session_factory: Callable[[], Session] | None = None,
) -> OrphanUploadSweepCursor | None:
    """Drain up to ``max_pages`` full pages; return where the next tick resumes.

    Each page runs in its own short Session. A short page means the eligible
    backlog is drained — the cursor resets (``None``) so the next tick starts
    a fresh bounded pass. Exhausting the page budget on full pages returns
    the live cursor so the next tick continues where this one stopped,
    keeping per-tick work bounded without capping throughput at one page per
    long sleep.
    """
    cursor = after
    for _ in range(max_pages):
        result = await run_db_io_cancellation_safe(
            lambda: sweep_orphaned_taskless_uploads_isolated(
                older_than_seconds=ttl_seconds,
                batch_size=batch_size,
                after=cursor,
                session_factory=session_factory,
            )
        )
        if result.scanned:
            logger.info(
                "Orphan task-less upload GC: scanned=%s deleted=%s",
                result.scanned,
                result.deleted,
            )
        if result.scanned < batch_size:
            return None
        cursor = result.next_cursor
    return cursor


async def run_orphan_upload_gc_loop(
    *,
    poll_interval_seconds: int,
    ttl_seconds: int,
    batch_size: int = GC_BATCH_SIZE,
    max_pages_per_tick: int = GC_MAX_PAGES_PER_TICK,
    backlog_continue_delay_seconds: float = GC_BACKLOG_CONTINUE_DELAY_SECONDS,
) -> None:
    """Reap orphans continuously while a backlog remains, rotating fairly.

    In-process counterpart of ``run_uploaded_file_compensation_recovery_loop``
    — it runs in every supported deployment (no Celery required). Each tick
    drains up to ``max_pages_per_tick`` full pages; while the returned cursor
    is still live (backlog not drained) the next budget follows after only a
    short breather, so drain throughput is set by page cost rather than by
    the poll interval — the task-less upload surfaces can admit far more rows
    per hour than one page budget reclaims. The long ``poll_interval_seconds``
    sleep is taken only once a short page proves the eligible backlog is
    drained, or after a failed tick (so a persistent error cannot hot-loop).
    The cursor carries across ticks so a repeatedly failing oldest page
    cannot starve newer orphans.
    """
    cursor: OrphanUploadSweepCursor | None = None
    while True:
        backlog_remaining = False
        try:
            cursor = await _run_gc_tick(
                ttl_seconds=ttl_seconds,
                batch_size=batch_size,
                max_pages=max_pages_per_tick,
                after=cursor,
            )
            backlog_remaining = cursor is not None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if is_database_pool_timeout(exc):
                logger.warning(
                    "Orphan task-less upload GC skipped after database pool timeout"
                )
            else:
                logger.exception("Orphan task-less upload GC tick failed")
        await asyncio.sleep(
            backlog_continue_delay_seconds
            if backlog_remaining
            else poll_interval_seconds
        )
