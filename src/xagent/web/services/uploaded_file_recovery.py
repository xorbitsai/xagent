"""Bounded recovery for stale uploaded-file compensation claims."""

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
from .uploaded_file_cleanup import run_uploaded_file_cleanup
from .uploaded_file_store import (
    StoragePresence,
    delete_uploaded_file_compensation_object,
)

logger = logging.getLogger(__name__)

CompensationDelete = Callable[..., StoragePresence]
UploadedFileCompensationRecoveryCursor = tuple[datetime, int]


@dataclass(frozen=True)
class UploadedFileCompensationCandidate:
    """Detached exact version of one stale compensation claim."""

    row_id: int
    user_id: int
    file_id: str
    task_id: int | None
    storage_key: str
    updated_at: datetime | None

    @property
    def cursor(self) -> UploadedFileCompensationRecoveryCursor:
        if self.updated_at is None:
            raise ValueError("Compensation candidate has no recovery token")
        return self.updated_at, self.row_id


@dataclass(frozen=True)
class UploadedFileCompensationRecoveryBatch:
    """Outcome of one bounded stale-compensation scan."""

    scanned: int = 0
    deleted: int = 0
    deferred_exists: int = 0
    deferred_unknown: int = 0
    failed: int = 0
    deferred_budget: int = 0
    next_cursor: UploadedFileCompensationRecoveryCursor | None = None


def get_stale_uploaded_file_compensation_candidates(
    db: Session,
    *,
    cutoff: datetime,
    limit: int,
    after: UploadedFileCompensationRecoveryCursor | None = None,
) -> tuple[UploadedFileCompensationCandidate, ...]:
    """Return a bounded detached snapshot of claims older than ``cutoff``."""

    if limit < 1:
        raise ValueError("limit must be positive")
    # ``compensating`` is introduced together with an explicit ``updated_at``
    # claim token. A null token therefore represents an unknown historical or
    # manually-written state and is deliberately not inferred from created_at.
    recovery_timestamp = UploadedFile.updated_at
    query = db.query(UploadedFile).filter(
        UploadedFile.storage_status == "compensating",
        UploadedFile.storage_key.isnot(None),
        UploadedFile.storage_key != "",
        UploadedFile.updated_at.isnot(None),
        recovery_timestamp <= cutoff,
    )
    if after is not None:
        after_timestamp, after_row_id = after
        query = query.filter(
            or_(
                recovery_timestamp > after_timestamp,
                and_(
                    recovery_timestamp == after_timestamp,
                    UploadedFile.id > after_row_id,
                ),
            )
        )
    records = (
        query.order_by(recovery_timestamp.asc(), UploadedFile.id.asc())
        .limit(limit)
        .all()
    )
    return tuple(
        UploadedFileCompensationCandidate(
            row_id=int(record.id),
            user_id=int(record.user_id),
            file_id=str(record.file_id),
            task_id=(int(record.task_id) if record.task_id is not None else None),
            storage_key=str(record.storage_key),
            updated_at=cast(datetime | None, record.updated_at),
        )
        for record in records
    )


def recover_stale_uploaded_file_compensations_batch_isolated(
    *,
    cutoff: datetime,
    batch_size: int,
    after: UploadedFileCompensationRecoveryCursor | None = None,
    session_factory: Callable[[], Session] | None = None,
    compensation_delete: CompensationDelete = delete_uploaded_file_compensation_object,
) -> UploadedFileCompensationRecoveryBatch:
    """Take over and continue one bounded page without a Session over I/O."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if session_factory is None:
        from ..models.database import get_session_local

        session_factory = get_session_local()

    with session_factory() as scan_db:
        candidates = get_stale_uploaded_file_compensation_candidates(
            scan_db,
            cutoff=cutoff,
            limit=batch_size,
            after=after,
        )

    deleted = 0
    deferred_exists = 0
    deferred_unknown = 0
    failed = 0
    deferred_budget = 0
    for candidate in candidates:
        try:
            outcome = run_uploaded_file_cleanup(
                session_factory=session_factory,
                row_id=candidate.row_id,
                user_id=candidate.user_id,
                file_id=candidate.file_id,
                task_id=candidate.task_id,
                storage_key=candidate.storage_key,
                expected_updated_at=candidate.updated_at,
                compensation_delete=compensation_delete,
                take_over=True,
            )
            if outcome == "deleted":
                deleted += 1
            elif outcome == "exists":
                deferred_exists += 1
            elif outcome == "unknown":
                deferred_unknown += 1
            elif outcome == "yielded":
                deferred_budget += 1
            elif outcome == "pending":
                failed += 1
        except Exception as exc:
            if is_database_pool_timeout(exc):
                raise
            failed += 1
            logger.exception(
                "Uploaded-file compensation recovery failed for file %s",
                candidate.file_id,
            )

    return UploadedFileCompensationRecoveryBatch(
        scanned=len(candidates),
        deleted=deleted,
        deferred_exists=deferred_exists,
        deferred_unknown=deferred_unknown,
        failed=failed,
        deferred_budget=deferred_budget,
        next_cursor=candidates[-1].cursor if candidates else None,
    )


async def run_uploaded_file_compensation_recovery_loop(
    *,
    poll_interval_seconds: int,
    stale_after_seconds: int,
    batch_size: int,
) -> None:
    """Reconcile at most one bounded page per poll while rotating fairly.

    A cursor advances across polling ticks so an oldest page whose storage
    probes remain unknown cannot starve later stale claims. Reaching a short
    page resets the cursor for the next bounded pass.
    """

    cursor: UploadedFileCompensationRecoveryCursor | None = None
    while True:
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(seconds=stale_after_seconds)
            result = await run_db_io_cancellation_safe(
                lambda: recover_stale_uploaded_file_compensations_batch_isolated(
                    cutoff=cutoff,
                    batch_size=batch_size,
                    after=cursor,
                )
            )
            cursor = result.next_cursor if result.scanned == batch_size else None
            if result.scanned:
                logger.info(
                    "Uploaded-file compensation recovery: scanned=%s "
                    "deleted=%s exists=%s unknown=%s failed=%s deferred_budget=%s",
                    result.scanned,
                    result.deleted,
                    result.deferred_exists,
                    result.deferred_unknown,
                    result.failed,
                    result.deferred_budget,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if is_database_pool_timeout(exc):
                logger.warning(
                    "Uploaded-file compensation recovery skipped after "
                    "database pool timeout"
                )
            else:
                logger.exception("Uploaded-file compensation recovery tick failed")

        await asyncio.sleep(poll_interval_seconds)
